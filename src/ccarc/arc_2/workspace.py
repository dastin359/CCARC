"""Run-workspace construction.

The workspace *is* the harness state. Claude Code owns the conversation, so
everything an orchestrator would otherwise hold in memory — puzzle data,
iteration ledger, verified invariants, distilled research state — lives on disk
here instead, where it survives context compaction and can be audited after the
fact.

Ground truth never enters the workspace. Test outputs are stripped when
``task.json`` is written and are held by the harness for post-run scoring only.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import prompt as prompt_mod
from .config import CCRunConfig

ASSETS = Path(__file__).parent / "assets"
STATE_SUBDIR = ".ccarc"


@dataclass
class Workspace:
    """A prepared puzzle workspace and the prompts that go with it."""

    root: Path
    task_id: str
    config: CCRunConfig
    system_prompt: str
    initial_prompt: str
    image_files: list[str]

    @property
    def state_path(self) -> Path:
        return self.root / STATE_SUBDIR / "state.json"

    @property
    def final_path(self) -> Path:
        return self.root / STATE_SUBDIR / "final.json"

    def read_state(self) -> dict[str, Any]:
        if not self.state_path.is_file():
            return {}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def read_final(self) -> dict[str, Any] | None:
        if not self.final_path.is_file():
            return None
        return json.loads(self.final_path.read_text(encoding="utf-8"))


def strip_test_outputs(puzzle_data: dict[str, Any]) -> dict[str, Any]:
    """Training pairs in full; test entries reduced to their inputs."""
    return {
        "train": [
            {"input": pair["input"], "output": pair["output"]}
            for pair in puzzle_data.get("train") or []
        ],
        "test": [{"input": pair["input"]} for pair in puzzle_data.get("test") or []],
    }


def ground_truth(puzzle_data: dict[str, Any]) -> list[list[list[int]] | None]:
    """Expected test outputs, held outside the workspace for scoring."""
    return [pair.get("output") for pair in puzzle_data.get("test") or []]


def _write(path: Path, text: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _render_images(root: Path, puzzle_data: dict[str, Any]) -> list[str]:
    """Render every grid to a PNG the agent can open with the Read tool."""
    try:
        from ._vendor.grid_visualizer import render_grid_to_image
    except Exception:  # noqa: BLE001 - Pillow missing, or a partial install
        return []

    images_dir = root / "task" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def emit(name: str, grid: list[list[int]]) -> None:
        data = render_grid_to_image(grid)
        if data is None:
            return
        (images_dir / name).write_bytes(data)
        written.append(f"task/images/{name}")

    for idx, pair in enumerate(puzzle_data.get("train") or []):
        emit(f"train_{idx}_input.png", pair["input"])
        emit(f"train_{idx}_output.png", pair["output"])
    for idx, pair in enumerate(puzzle_data.get("test") or []):
        emit(f"test_{idx}_input.png", pair["input"])
    return written


def _mirror_toolkit_into_explore(root: Path) -> None:
    """Make ``from arc import …`` work from a script under ``explore/``.

    ``python explore/foo.py`` puts *the script's* directory on ``sys.path``, not
    the working directory, so a toolkit sitting at the workspace root is invisible
    to exactly the invocation the contract tells the agent to use. Mirroring the
    module into ``explore/`` fixes the documented path without making the agent
    write ``sys.path`` boilerplate at the top of every experiment — friction that
    would land on the single most common action in the whole loop.
    """
    target = root / "explore" / "arc.py"
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(Path("..") / "arc.py")
    except (OSError, NotImplementedError):
        shutil.copyfile(root / "arc.py", target)  # filesystems without symlinks


def ccarc_src_root() -> str:
    """Directory to put on ``sys.path`` so a workspace can import ccarc."""
    return str(Path(__file__).resolve().parents[2])


#: Libraries worth telling the solver about. NumPy in particular: the doctrine
#: carries printing rules for it, and advice about a library that turns out not
#: to be installed is worse than no advice.
_PROBED_MODULES = ("numpy", "PIL", "scipy")

_PROBE_SOURCE = (
    "import importlib.util,sys;"
    "print(sys.version.split()[0]);"
    "print(','.join(m for m in %r if importlib.util.find_spec(m)))" % (_PROBED_MODULES,)
)


def probe_interpreter(command: str) -> dict[str, Any] | None:
    """Version and available libraries for ``command``, or None if unusable."""
    try:
        completed = subprocess.run(
            [command, "-c", _PROBE_SOURCE], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    lines = (completed.stdout or "").strip().splitlines()
    if not lines:
        return None
    modules = [m for m in (lines[1].split(",") if len(lines) > 1 else []) if m]
    return {"command": command, "version": lines[0], "modules": modules}


def choose_interpreter() -> dict[str, Any]:
    """Pick the interpreter the solver should use, and say what it provides.

    Prefers whatever ``python`` resolves to on PATH — that is what the agent
    will reach for — but falls back to the interpreter running the harness when
    that one carries libraries the PATH interpreter lacks. A workspace whose
    contract promises NumPy on a runtime without NumPy sends the solver into a
    dead end at the worst possible moment.
    """
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for command in ("python", "python3", sys.executable):
        if not command or command in seen:
            continue
        seen.add(command)
        probed = probe_interpreter(command)
        if probed is not None:
            candidates.append(probed)

    if not candidates:
        return {"command": sys.executable, "version": "", "modules": [], "probed": False}

    # More libraries wins; ties go to the earlier (more natural) command.
    best = max(candidates, key=lambda c: len(c["modules"]))
    if len(best["modules"]) == len(candidates[0]["modules"]):
        best = candidates[0]
    return {**best, "probed": True}


from ccarc.staging import stage_package  # noqa: E402


def build_workspace(
    *,
    task_id: str,
    puzzle_data: dict[str, Any],
    root: Path | str,
    config: CCRunConfig | None = None,
    overwrite: bool = False,
) -> Workspace:
    """Materialise a complete, self-contained puzzle workspace at ``root``."""
    config = config or CCRunConfig()
    root = Path(root).resolve()

    if root.exists() and any(root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Workspace {root} already exists and is not empty.")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    for subdir in ("task", "explore", "solution", STATE_SUBDIR, ".claude/hooks"):
        (root / subdir).mkdir(parents=True, exist_ok=True)

    interpreter = choose_interpreter()

    visible = strip_test_outputs(puzzle_data)
    _write(root / "task" / "task.json", json.dumps({"task_id": task_id, **visible}, indent=1))
    _write(root / "task" / "grids.md", prompt_mod.render_task_markdown(task_id, visible))

    image_files = _render_images(root, visible) if config.visual else []

    shutil.copyfile(ASSETS / "arc_toolkit.py", root / "arc.py")
    _mirror_toolkit_into_explore(root)
    shutil.copyfile(ASSETS / "dryrun.py", root / "dryrun.py")

    # **The gate imports a STAGED copy, not the repository.** Pointing the child
    # at the live `src` would hand it the whole checkout, prompt assets
    # included. The stage carries the package with every prompt removed, so
    # there is nothing there to strip and nothing to re-check.
    # Staged beside the workspace rather than inside it: the solver's own
    # directory listing stays its own material, and the jail keeps the parent.
    stage = stage_package(Path(root).resolve().parent, driver="arc_2")
    gate_source = (ASSETS / "gate_shim.py").read_text(encoding="utf-8")
    _write(root / "gate.py", gate_source.replace("__STAGE_SRC__", str(stage)), executable=True)

    hook_source = (ASSETS / "on_compact.sh").read_text(encoding="utf-8")
    hook_path = root / ".claude" / "hooks" / "on_compact.sh"
    _write(
        hook_path,
        hook_source.replace("__WORKSPACE__", str(root)).replace("__PYTHON__", sys.executable),
        executable=True,
    )
    _write(root / ".claude" / "settings.json", prompt_mod.settings_json(hook_script=hook_path))

    _write(root / STATE_SUBDIR / "harness_python", sys.executable)
    _write(
        root / "CLAUDE.md",
        prompt_mod.build_workspace_claude_md(
            task_id=task_id, puzzle_data=visible, config=config, interpreter=interpreter
        ),
    )
    _write(root / "NOTES.md", prompt_mod.build_notes_seed(task_id))
    _write(
        root / "explore" / "README.md",
        "Scratch scripts live here. One question per script, named for the question.\n"
        f"Run them with `{interpreter.get('command') or 'python'} explore/<name>.py`.\n"
        "Start a name with a letter so other scripts can import it (`from lib import ...`);\n"
        "`05_thing.py` is not a valid module name.\n"
        "Nothing here is budgeted or recorded; only `arc.verify()` results and\n"
        "`gate.py submit` are.\n",
    )

    state = {
        "task_id": task_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "harness": "claude-code",
        "max_iterations": config.max_iterations,
        "best_effort_iterations": config.best_effort_iterations,
        "min_hypothesis_chars": config.min_hypothesis_chars,
        "max_test_predictions": config.max_test_predictions,
        "solve_timeout_s": config.solve_timeout_s,
        "interpreter": interpreter,
        "iterations": [],
        "accepted": None,
        "last_hypothesis_sha": "",
        "last_code_sha": "",
        "config": config.to_dict(),
    }
    _write(root / STATE_SUBDIR / "state.json", json.dumps(state, indent=2))
    _write(root / STATE_SUBDIR / "invariants.jsonl", "")

    system_prompt = prompt_mod.build_system_prompt(config.ablate)
    initial_prompt = prompt_mod.build_initial_prompt(
        task_id=task_id,
        puzzle_data=visible,
        config=config,
        image_files=image_files,
        interpreter=interpreter,
    )
    # Written next to the workspace so any Claude Code agent — the subprocess
    # launcher, or a sub-agent inside an existing session — can be pointed at
    # the same two prompts.
    _write(root / STATE_SUBDIR / "system_prompt.md", system_prompt)
    _write(root / STATE_SUBDIR / "initial_prompt.md", initial_prompt)

    return Workspace(
        root=root,
        task_id=task_id,
        config=config,
        system_prompt=system_prompt,
        initial_prompt=initial_prompt,
        image_files=image_files,
    )


def load_workspace(root: Path | str) -> Workspace:
    """Reopen an existing workspace (for scoring or inspection)."""
    root = Path(root).resolve()
    state = json.loads((root / STATE_SUBDIR / "state.json").read_text(encoding="utf-8"))
    config = CCRunConfig.from_dict(state.get("config") or {})

    def _maybe(relative: str) -> str:
        path = root / STATE_SUBDIR / relative
        return path.read_text(encoding="utf-8") if path.is_file() else ""

    images_dir = root / "task" / "images"
    images = (
        sorted(f"task/images/{p.name}" for p in images_dir.glob("*.png"))
        if images_dir.is_dir()
        else []
    )
    return Workspace(
        root=root,
        task_id=str(state.get("task_id") or root.name),
        config=config,
        system_prompt=_maybe("system_prompt.md"),
        initial_prompt=_maybe("initial_prompt.md"),
        image_files=images,
    )



#: How long the CLI waits for a solver's background tasks before killing them.
#: The default of 600 s is shorter than the run's own wall clock and fails
#: worse: a solver that delegates to a sub-agent and ends its turn can have the
#: background task killed mid-flight, recording 0 iterations and no hypothesis.
#:
#: Raising it does not make runs longer — `wall_clock_timeout_s` still bounds
#: them, and a wall-clock timeout salvages a train-perfect submission instead
#: of discarding it. **One** authority decides when a run ends, rather than two
#: timers whose shorter one has no salvage path.
BACKGROUND_WAIT_CEILING_MS = "3600000"


def workspace_env() -> dict[str, str]:
    """Environment for a solver process: the dataset root is deliberately removed."""
    env = dict(os.environ)
    env.pop("ARC_DATA_ROOT", None)
    env.setdefault("CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS", BACKGROUND_WAIT_CEILING_MS)
    # A solver must not inherit a context-gate hook, which holds a session's
    # first turns until it has read a project document: it would block the run,
    # and the document would describe the experiment the solver is the subject
    # of.
    #
    # `off` is the value the hook checks -- `in_scope()` returns False for
    # `off|0|false`. The name reads backwards; `off` means the gate is off.
    #
    # **Set here rather than relied on from a shell.** The hook reads the
    # process environment of the process it gates, so an export elsewhere never
    # reaches a child the harness launches later.
    env["CONTEXT_GATE_DISABLE"] = "off"
    return env
