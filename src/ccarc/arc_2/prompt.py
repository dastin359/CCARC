"""System-prompt composition for the CC harness.

The ARC-specific parts of the solver prompt — who the agent is, and what it
knows about the benchmark's priors and primitives — come from a shared prompt
file, vendored under ``_vendor/``. Holding the domain knowledge constant is
what makes a measured difference attributable to the harness rather than to
the priors.

Sections 3 onward (goal, methodology, tools, loop) are harness-specific and come
from ``assets/CC_SOLVER_DOCTRINE.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .config import CCRunConfig

ASSETS = Path(__file__).parent / "assets"

#: Headings taken unchanged from the shared ARC prompt.
SHARED_SECTIONS: tuple[str, ...] = ("1. ROLE & IDENTITY", "2. ARC DOMAIN KNOWLEDGE")

PROMPT_HEADER = "Please read and follow the instructions below.\n"


def solver_prompt_path() -> Path:
    # The shared ARC sections are vendored under `_vendor/`, so composing a
    # prompt pulls in no external dependency.
    return Path(__file__).resolve().parents[0] / "_vendor" / "SOLVER_SYSTEM_PROMPT.md"


def split_sections(markdown: str) -> dict[str, str]:
    """Split a markdown document on its ``## `` headings.

    Returns ``{heading_text: body}``; the body excludes the heading line and any
    trailing horizontal rule.
    """
    sections: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        if current is None:
            return
        body = "\n".join(buffer).strip()
        if body.endswith("---"):
            body = body[: -len("---")].rstrip()
        sections[current] = body

    for line in markdown.splitlines():
        if line.startswith("## "):
            flush()
            current = line[3:].strip()
            buffer = []
        elif current is not None:
            buffer.append(line)
    flush()
    return sections


def shared_arc_sections() -> str:
    """The role + domain-knowledge half of the shared ARC prompt."""
    path = solver_prompt_path()
    sections = split_sections(path.read_text(encoding="utf-8"))
    missing = [title for title in SHARED_SECTIONS if title not in sections]
    if missing:
        raise RuntimeError(
            f"{path} no longer contains the section(s) {missing!r} that the CC harness shares "
            "with the vendored copy. Update ccarc.arc_2.prompt.SHARED_SECTIONS to match, and "
            "check whether the two still agree on ARC domain knowledge."
        )
    return "\n\n---\n\n".join(f"## {title}\n\n{sections[title]}" for title in SHARED_SECTIONS)


def doctrine() -> str:
    """The CC-harness-specific goal and methodology sections."""
    return (ASSETS / "CC_SOLVER_DOCTRINE.md").read_text(encoding="utf-8").strip()


#: The one system-prompt component an experiment may remove.
DOCTRINE = "doctrine"

#: Prefix for a workspace ``CLAUDE.md`` section, e.g. ``workspace:Rival readings``.
WORKSPACE_PREFIX = "workspace:"


def _split_ablations(ablate: Iterable[str]) -> tuple[bool, tuple[str, ...]]:
    """Partition ablation targets, rejecting names nothing would act on.

    An unrecognised target must raise rather than quietly do nothing: a silent
    no-op is indistinguishable from a performed ablation, and an experiment that
    cannot tell those apart is not an experiment.
    """
    drop_doctrine = False
    sections: list[str] = []
    for target in ablate:
        if target == DOCTRINE:
            drop_doctrine = True
        elif target.startswith(WORKSPACE_PREFIX):
            name = target[len(WORKSPACE_PREFIX):].strip()
            if not name:
                raise ValueError(f"ablation target {target!r} names no section")
            sections.append(name)
        else:
            raise ValueError(
                f"unknown ablation target {target!r}; expected {DOCTRINE!r} or "
                f"{WORKSPACE_PREFIX}<section heading>"
            )
    return drop_doctrine, tuple(sections)


def strip_markdown_sections(text: str, headings: Iterable[str]) -> str:
    """Remove ``## <heading>`` blocks, up to the next heading of the same level.

    Raises if a named heading is not present, for the same reason as above.
    """
    for heading in headings:
        lines = text.splitlines(keepends=True)
        start = next(
            (i for i, line in enumerate(lines) if line.strip() == f"## {heading}"), None
        )
        if start is None:
            raise ValueError(f"cannot ablate {heading!r}: no '## {heading}' section found")
        end = next(
            (j for j in range(start + 1, len(lines)) if lines[j].startswith("## ")), len(lines)
        )
        text = "".join(lines[:start] + lines[end:])
    return text


def build_system_prompt(ablate: Iterable[str] = ()) -> str:
    """Full appended system prompt. Task-independent, so it caches across a batch.

    ``ablate`` exists so an experiment that claims to have removed the doctrine
    has actually removed it: ablation happens here, at prompt composition,
    never by hand-editing a workspace file the composed prompt may override.
    """
    # Callers pass the whole `ablate` tuple. ``workspace:*`` entries belong to
    # build_workspace_claude_md, so this function ignores them rather than
    # rejecting them.
    drop_doctrine, _sections = _split_ablations(ablate)
    # **The shared sections are emitted ONLY when the doctrine is dropped.**
    # `doctrine()` already contains them verbatim, so appending both would ship
    # every solver sections 1-2 twice, several thousand characters of the
    # system prompt repeated. Sections 1-2 live inside the doctrine (see the
    # asset's own HTML comment) so that a passage pointing at "the ARC domain
    # knowledge above" still resolves when the doctrine is read alone.
    #
    # They cannot simply be deleted from this list: with the doctrine ablated,
    # this is the only thing carrying ARC domain knowledge, and an ablation must
    # remove the doctrine, not the domain.
    parts = [PROMPT_HEADER.strip()]
    if drop_doctrine:
        parts.append(shared_arc_sections())
    else:
        parts.append(doctrine())
    text = "\n\n---\n\n".join(parts) + "\n"
    if drop_doctrine:
        # Post-condition, not decoration: the whole point is that the caller can
        # trust the removal happened.
        marker = doctrine().splitlines()[0].strip()
        if marker and marker in text:
            raise AssertionError(f"doctrine ablation left {marker!r} in the prompt")
    return text


# ── task presentation ────────────────────────────────────────────────────────

def render_grid_text(grid: list[list[int]]) -> str:
    """One row per line, one digit per cell."""
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def _shape(grid: list[list[int]]) -> str:
    return f"{len(grid)}x{len(grid[0])}" if grid else "0x0"


def render_task_markdown(task_id: str, puzzle_data: dict[str, Any]) -> str:
    """The puzzle as text, written to ``task/grids.md`` and optionally inlined."""
    lines = [f"# Task {task_id}", ""]
    for idx, pair in enumerate(puzzle_data.get("train") or []):
        lines.append(f"## Training pair {idx}")
        lines.append("")
        lines.append(f"input ({_shape(pair['input'])}):")
        lines.append(render_grid_text(pair["input"]))
        lines.append("")
        lines.append(f"output ({_shape(pair['output'])}):")
        lines.append(render_grid_text(pair["output"]))
        lines.append("")
    for idx, pair in enumerate(puzzle_data.get("test") or []):
        lines.append(f"## Test input {idx}")
        lines.append("")
        lines.append(f"input ({_shape(pair['input'])}):")
        lines.append(render_grid_text(pair["input"]))
        lines.append("")
    return "\n".join(lines)


def build_initial_prompt(
    *,
    task_id: str,
    puzzle_data: dict[str, Any],
    config: CCRunConfig,
    image_files: list[str],
    interpreter: dict[str, Any] | None = None,
) -> str:
    """The opening user message handed to ``claude -p``."""
    n_train = len(puzzle_data.get("train") or [])
    n_test = len(puzzle_data.get("test") or [])
    python = (interpreter or {}).get("command") or "python"

    parts = [
        f"Solve ARC-AGI-2 task `{task_id}`.",
        "",
        "Read ./CLAUDE.md first — it describes this workspace, the gate commands, and the "
        "rules the gate enforces. Then read ./arc.py so you know what the toolkit gives you.",
        "",
        f"Run everything with `{python}` — that is the interpreter carrying this workspace's "
        "libraries. CLAUDE.md lists what is installed.",
        "",
        f"You have {n_train} training pair(s), {n_test} test input(s), and "
        f"{config.max_iterations} formal submissions.",
    ]

    if image_files:
        listed = ", ".join(image_files[:8])
        more = f", … ({len(image_files)} files)" if len(image_files) > 8 else ""
        parts += [
            "",
            "Every grid is rendered as a PNG under `task/images/` — "
            f"{listed}{more}. Open them with the Read tool before you start reasoning; "
            "gestalt perception catches structure that a numeric dump does not.",
        ]

    if config.inline_grids:
        parts += [
            "",
            "The grids follow as text (one row per line, one digit per cell). The same content "
            "is in `task/grids.md`, and `arc.py` loads it as `train_samples` / `test_samples`. "
            "Read them for orientation, but do not count cells or index columns off this text — "
            "solvers reliably miscount here, and a miscount can produce a false refutation of "
            "the correct rule. `arc.show(grid)` prints the same grid with row and column rulers.",
            "",
            render_task_markdown(task_id, puzzle_data),
        ]
    else:
        parts += ["", "The grids are in `task/grids.md` and loadable via `arc.py`."]

    parts += [
        "",
        "Begin with perception and verification, not with a guess. When you have a rule, write "
        f"`solution/hypothesis.md` and `solution/solve.py`, dry-run it with `{python} dryrun.py`, "
        f"and only then run `{python} gate.py submit`.",
    ]
    return "\n".join(parts)


#: Libraries the doctrine or the toolkit assumes, mapped to what they buy.
_LIBRARY_NOTES = {
    "numpy": "array work and fast whole-grid comparisons",
    "PIL": "rendering a grid to PNG via `arc.png()`",
    "scipy": "labelling and morphology helpers",
}


def describe_environment(interpreter: dict[str, Any] | None) -> str:
    """State plainly what the solver's runtime provides.

    A contract that assumes NumPy on a runtime without NumPy sends the agent
    into a dead end at the worst moment, so the workspace says what is actually
    installed rather than what is usually installed.
    """
    if not interpreter or not interpreter.get("probed"):
        return (
            "Run everything with `python`. The harness could not probe this runtime, so "
            "check for a library with `python -c \"import numpy\"` before relying on it."
        )

    command = interpreter.get("command") or "python"
    version = interpreter.get("version") or "unknown version"
    modules = list(interpreter.get("modules") or [])

    lines = [f"Run everything with `{command}` (Python {version})."]
    if modules:
        available = ", ".join(f"`{m}` ({_LIBRARY_NOTES.get(m, 'available')})" for m in modules)
        lines.append(f"Available beyond the standard library: {available}.")
    missing = [m for m in _LIBRARY_NOTES if m not in modules]
    if missing:
        lines.append(
            "NOT installed: "
            + ", ".join(f"`{m}`" for m in missing)
            + ". Use the standard library instead — do not spend an experiment discovering this."
        )
    return "\n".join(lines)


def build_resume_prompt(
    *,
    task_id: str,
    state: dict[str, Any],
    interpreter: dict[str, Any] | None = None,
) -> str:
    """Opening message for a solver picking up an interrupted run.

    This is the crash-recovery case, and it works for the same reason context
    compaction does: the workspace *is* the state. A fresh agent inherits the
    iteration ledger, the verified invariants, and NOTES.md — everything the
    previous one established — and only the untracked exploratory interpreter
    state is gone.
    """
    python = (interpreter or {}).get("command") or "python"
    iterations = state.get("iterations") or []
    used = len(iterations)
    budget = int(state.get("max_iterations") or 0)
    last = iterations[-1] if iterations else None

    parts = [
        f"You are resuming an interrupted run on ARC-AGI-2 task `{task_id}`.",
        "",
        f"A previous solver worked on this and stopped without accepting. It used "
        f"{used} of {budget} submissions, so you have {budget - used} left.",
        "",
        f"Start by running `{python} gate.py status`. That prints everything that "
        "survived: the submission history, every invariant established with "
        "arc.verify(), the last hypothesis, and the tail of NOTES.md. Read "
        "./CLAUDE.md and ./NOTES.md next.",
        "",
        "What did NOT survive: the previous agent's reasoning, its transcript, and any "
        "live Python state. Anything you need that is not in the ledger or NOTES.md has "
        "to be re-derived by running a script under explore/. Do not assume an "
        "unrecorded claim is true because it looks like something you would have checked.",
    ]
    if last:
        outcome = (
            "passed training"
            if last.get("all_train_correct")
            else f"scored {last.get('train_correct')}/{last.get('train_total')} on training"
        )
        parts += [
            "",
            f"The last submission (#{last.get('iteration')}) {outcome}. Its report is at "
            f".ccarc/iterations/{last.get('iteration')}/report.txt, with the exact "
            "hypothesis and code alongside it.",
        ]
    parts += [
        "",
        "Then continue the loop as CLAUDE.md describes, and finish with "
        f"`{python} gate.py accept`.",
    ]
    return "\n".join(parts)


def build_workspace_claude_md(
    *,
    task_id: str,
    puzzle_data: dict[str, Any],
    config: CCRunConfig,
    interpreter: dict[str, Any] | None = None,
) -> str:
    template = (ASSETS / "WORKSPACE_CLAUDE.md").read_text(encoding="utf-8")
    substitutions = {
        "__TASK_ID__": task_id,
        "__N_TRAIN__": str(len(puzzle_data.get("train") or [])),
        "__N_TEST__": str(len(puzzle_data.get("test") or [])),
        "__MAX_ITERATIONS__": str(config.max_iterations),
        "__BEST_EFFORT__": str(config.best_effort_iterations),
        "__MAX_CANDIDATES__": str(config.max_test_predictions),
        "__MIN_HYPOTHESIS_CHARS__": str(config.min_hypothesis_chars),
        "__PYTHON__": (interpreter or {}).get("command") or "python",
        "__ENVIRONMENT__": describe_environment(interpreter),
    }
    for key, value in substitutions.items():
        template = template.replace(key, value)
    _, sections = _split_ablations(getattr(config, "ablate", ()) or ())
    if sections:
        template = strip_markdown_sections(template, sections)
    return template


def build_notes_seed(task_id: str) -> str:
    return (
        f"# NOTES — task {task_id}\n\n"
        "Durable research state. Written for a fresh version of yourself: assume the\n"
        "transcript above this file is gone.\n\n"
        "## Current hypothesis\n\n_(none yet)_\n\n"
        "## Confirmed observations\n\n"
        "_(record invariants with `arc.verify()`; `python gate.py status` replays them)_\n\n"
        "## Refuted hypotheses\n\n_(what was ruled out, and by which experiment)_\n\n"
        "## Next experiment\n\n_(the specific check to run, and what each outcome would mean)_\n"
    )


def settings_json(*, hook_script: Path) -> str:
    """Workspace ``.claude/settings.json`` — the compaction-recovery hook."""
    return json.dumps(
        {
            "hooks": {
                "SessionStart": [
                    {
                        "matcher": "compact",
                        "hooks": [
                            {
                                "type": "command",
                                "command": "bash",
                                "args": [str(hook_script)],
                                "timeout": 60,
                                "statusMessage": "Restoring distilled research state…",
                            }
                        ],
                    }
                ]
            }
        },
        indent=2,
    )
