"""Workspace construction and session launch for CCARC3.

The division of labour is deliberate: **Claude Code owns the agent
loop**, and this harness supplies a workspace, a toolkit, and a gate. There is
no reviewer and no orchestration — the solver decides what to do next, and the
harness only refuses the moves that are known to destroy a run.

What a workspace contains:

===================  ======================================================
``CLAUDE.md``        the whole contract: how to drive the game, and the doctrine
``session.py``       a client and gate pre-wired to this game
``trace.jsonl``      every action and its frames, written unprompted
``rules.json``       the rule book the gate requires at level boundaries
``notes/``           the solver's own scratch space
===================  ======================================================
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ccarc.staging import stage_package

from .agent_config import (
    DEFAULT_ALLOWED_TOOLS,
    DEFAULT_DISALLOWED_TOOLS,
    resolve_permission_mode,
)
from .client import GameInfo, list_games
# **Module level, not inside `_launch`.** `_release_launch` references
# `confine` as a global, and its `except Exception` would turn a missing name
# into a printed line rather than a crash -- "credential harvest failed" on
# every run, with no refreshed token ever copied back, which is the exact
# condition that kills the *next* launch at zero actions. `confine` imports
# only the standard library, so there is no cycle and no reason for the import
# to be function-local.
from ..guardrail import confine

#: An environment variable the child must never inherit. Its name alone would
#: tell a solver that a human baseline exists and that something chooses
#: whether it may see one, so it is popped from the child environment and
#: recorded as an absence in `RECORDED_ENV`. Nothing reads it: the shim strips
#: the baseline unconditionally.
RETIRED_MODE_VAR = "CCARC3_BASELINE_MODE"

__all__ = [
    "Ccarc3Config",
    "Workspace",
    "build_workspace",
    "build_cli_args",
    "run_game",
    "collect_outcome",
    "snapshot_scorecard",
    "redact_self_reference",
    "ledger_facts",
    "ensure_scored_ledger",
    "pin_scored_card",
    "scored_arc_client",
    "side_ledgers",
]

ASSETS = Path(__file__).parent / "assets"

# The standing constraint for this project: Opus 5 at effort high, and no
# other model or effort, so runs stay comparable to one another.
DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT = "high"


@dataclass
class Ccarc3Config:
    """One solver run against one game."""

    game_id: str
    out_dir: Path = Path("runs/ccarc3")
    model: str = DEFAULT_MODEL
    effort: str = DEFAULT_EFFORT
    # Action cap as a multiple of the game's published baseline (see §2.6).
    # A flat cap cannot work: games differ in length by an order of magnitude.
    # 5.0 because that is ARC's own ceiling (5n per level); a tighter multiple
    # distorts exactly the runs it binds. A COMMENT, not an attribute
    # docstring: this file is on the solver's import path, staging strips only
    # comments, so a docstring here would be readable by the solver.
    budget_multiple: float = 5.0

    # Why no percentages: a cap is the baseline total times `budget_multiple`,
    # and that multiplier is a default in this same class, so `used/cap`
    # written out inverts to the withheld figure. The rule above is the safe
    # form to keep when editing.

    wall_clock_timeout_s: float = 28800.0
    clock_is_infrastructure: bool = True
    """Whether a wall-clock timeout with budget left is an interruption.

    True (default): the clock is a container-survival / caller cap, not the
    game's allowance. A clip with actions left is not a result.
    False: the clock *is* the caller's window. A timeout with play already on
    the board banks. Zero actions still interrupts.

    Do not infer this from the number of seconds -- the default duration
    matches neither meaning. Each caller states what its clock is.
    """
    permission_mode: str = "bypassPermissions"
    allowed_tools: tuple[str, ...] = DEFAULT_ALLOWED_TOOLS
    """Pre-approved so a headless run never stalls on a permission prompt.

    ``--allowedTools`` grants permission as well as restricting the surface.
    Without it, ``acceptEdits`` approves file writes but *not* Bash, so every
    ``python -c ...`` the solver runs is denied and the run produces nothing.
    Defined once in `agent_config` rather than re-derived, so every launch
    harnesses cannot drift.
    """

    disallowed_tools: tuple[str, ...] = DEFAULT_DISALLOWED_TOOLS
    """Denied outright: research, network-by-another-door, escaping the run.

    ARC needs no external knowledge and a network answer would contaminate
    the benchmark. Everything else Claude Code ships stays available.
    """
    api_key: str | None = None
    fresh: bool = False
    """Discard any existing trace and start the game over.

    The default is to *resume*. A container can be recycled mid-run, and the
    client already persists its session, so relaunching against the same
    out_dir continues the same game rather than paying for the first N actions
    twice. Making that the default is deliberate; making it silent would not
    be, hence the flag and the log line.

    Resuming restores the local side completely — ``card_id``, ``level``,
    ``actions_used`` and cookies all come back off ``trace.state.json``, and
    ``resume_state.json`` records exactly what was inherited. It cannot restore
    the *server* side: ARC reaps an idle game, and past the reap window the card
    is gone and the resume becomes a replay from level 0 on the old action count.
    Under the window the card is still there and the resume continues in place,
    even across a container replacement. See :func:`snapshot_scorecard` for the
    window. Resume promptly either way.
    """

    card_id: str = ""
    """Play onto a scorecard the driver already opened, instead of a new one.

    Empty mints one card per game. Set, and every run lands on one card, which
    is what a leaderboard submission takes. The id
    alone does not reach the card; the shim must also carry its stickiness
    cookies (:meth:`arc_proxy.ProxyState.adopt_session`), and the two are set
    together by the driver.
    """

    extra_cli_args: tuple[str, ...] = ()


@dataclass
class Workspace:
    root: Path
    config: Ccarc3Config
    info: GameInfo
    initial_prompt: str = ""
    env: dict[str, str] = field(default_factory=dict)
    resumed: bool = False
    #: This solver's own Claude Code session id, assigned rather than discovered.
    #:
    #: **Every event in a solver's stream carries the *parent's* id.** The
    #: child inherits `CLAUDE_CODE_SESSION_ID` and does not mint its own -- its
    #: transcript on disk is literally named after the driver's session -- so
    #: the only thing separating one solver's transcript from another's is the
    #: cwd-keyed directory holding it.
    #:
    #: So resuming a solver by an id read out of its stream would resume *the
    #: driver's own conversation*. Assigning one with `--session-id` removes the
    #: discovery problem instead of solving it, and is what makes `--resume`
    #: exact.
    session_id: str = ""

    @property
    def trace_path(self) -> Path:
        return self.root / "trace.jsonl"

    @property
    def rules_path(self) -> Path:
        return self.root / "rules.json"


# **The solver reads this template, so the reasoning behind it lives here and not
# inside it.** The rendered `session.py` sits in the workspace, the doctrine tells
# the solver to import it, and `cat session.py` is one of the first things a
# careful one does. So the template carries no operator commentary: an
# explanation of the hidden cap, of the pace figures being kept back, or of what
# some other environment scored is a map to something the harness spends real
# machinery hiding, handed over in the one file the solver is instructed to
# open. Nor may it name any environment other than this run's own:
# `redact_self_reference` rewrites only the doctrine half of CLAUDE.md, so a
# name here would reach the very run it names.
#
# The four arguments below, and why they are what they are:
#
# `max_actions=0` -- no client-side cap, deliberately. ARC's FrameResponse carries
# no budget field and the technical report designed away from a per-environment
# allowance: "we won't ... encourage AI to waste actions on levels because they're
# still 'under budget' for a given environment". A solver that knows its allowance
# paces itself against it, which is the wrong objective -- the score is completion
# first and efficiency only as a tiebreak, so points are lost by stopping, not
# by exploring. A hard stop exists far out **on the baseline-free path**,
# enforced in `arc_proxy` where the solver does not run, so a runaway loop
# cannot spend without limit there. Scoped: the ceiling is armed by the caller
# that installs the baseline strip, so a plain `run_game` arms no ceiling at
# any layer. That is fine for the local bench.
#
# `quiet_pace=True` -- withhold the human medians from every solver-facing
# surface: the pace ratio and the `raw`/ceiling half of the score block. (ARC's
# per-level 5n rule is deliberately not enforced client-side: that ceiling is a
# fixed multiple of the figure this arm hides, so a refusal would disclose it
# by arithmetic on the solver's own action count, whatever the message said. It
# stays off.)
#
# `show_score=True` -- report the running score in `status()`. With baselines
# available that is `raw`, its ceiling and the completion cap; without them the
# cap alone, which needs no baselines because it is which levels fell rather than
# how fast. Without it a solver cannot see its own score at all, and can run a
# level several times past its baseline with nothing able to tell it that its
# ceiling has already dropped.
# **No `GameInfo` in the workspace client, and no comment about one.**
# Every `self.info` use in `ArcClient` reads the withheld array and is guarded
# with `if self.info else` -- `baseline_for`, the per-level list, and the
# level-count fallback. So an INFO carrying an empty array is exactly
# equivalent to passing nothing: those three sites return `None`, `[]` and
# zero either way, while passing it would cost three disclosures for no
# behaviour: the game's action-type `tags` (its input modality, which
# `available_actions` carries from frame 1), its `title`, and the name
# `GameInfo` itself, which both withholding lists hide from `dir()` precisely
# because its field is `baseline_actions`.
#
# The rationale lives here rather than inside the template. `staging` strips
# comments from the package, but this workspace file is written with a plain
# `write_text`, and a `#` inside a string literal is not a comment to the
# tokenizer -- so anything written inside the template reaches the solver
# verbatim.
SESSION_TEMPLATE = '''\
"""Pre-wired client and gate for {game_id}. Import this; do not rebuild it.

    from session import client, gate, arc

    client.reset()
    client.act(1)
    print(client.status())

Every action is recorded to trace.jsonl automatically. The gate holds the first
action of a new level until you call ``gate.acknowledge(...)``.
"""
import os
from pathlib import Path

from ccarc import arc_3 as arc
from ccarc.arc_3 import ArcClient, LevelGate
from ccarc.arc_3.session import (
    ensure_scored_ledger,
    scored_arc_client,
)

HERE = Path(__file__).parent
ensure_scored_ledger(HERE)
ArcClient = scored_arc_client(HERE, ArcClient)

gate = LevelGate(HERE / "rules.json")
client = ArcClient(
    {game_id!r},
    trace_path=HERE / "trace.jsonl",
    gate=gate,
    max_actions=0,
{card_line}    quiet_pace=True,
    show_score=True,
)
client.open()

'''


SCORED_PIN_NAME = ".scored_card.json"


def pin_scored_card(root: Path | str, card_id: str) -> None:
    """Record which scorecard the workspace-root ledger is bound to."""
    if not card_id or str(card_id).startswith("card-"):
        return
    path = Path(root) / SCORED_PIN_NAME
    path.write_text(
        json.dumps({"card_id": card_id}, indent=2) + "\n", encoding="utf-8"
    )


def _ledger_pair(root: Path) -> tuple[Path, Path]:
    return root / "trace.jsonl", root / "trace.state.json"


def _read_card_id(state: Path) -> str:
    try:
        cid = json.loads(state.read_text(encoding="utf-8")).get("card_id") or ""
    except (OSError, ValueError, TypeError):
        return ""
    return cid if isinstance(cid, str) else ""


def _pin_card_id(root: Path) -> str:
    return _read_card_id(root / SCORED_PIN_NAME)


def _resume_card_id(root: Path) -> str:
    return _read_card_id(root / "resume_state.json")


def _bound_card_id(root: Path) -> str:
    """A card this workspace has already scored onto, if we know one."""
    trace, state = _ledger_pair(root)
    return _read_card_id(state) or _pin_card_id(root) or _resume_card_id(root)


def _archive_pairs(root: Path) -> list[tuple[float, Path, Path]]:
    """Ledger copies under notes/archive, including agent-renamed names."""
    arch = Path(root) / "notes" / "archive"
    if not arch.exists():
        return []
    out: list[tuple[float, Path, Path]] = []
    sibling_t, sibling_s = arch / "trace.jsonl", arch / "trace.state.json"
    if sibling_t.exists() and sibling_s.exists():
        out.append((sibling_t.stat().st_mtime, sibling_t, sibling_s))
    traces = list(arch.glob("trace*.jsonl")) + list(arch.glob("tr_*.jsonl"))
    states = list(arch.glob("trace.state*.json")) + list(arch.glob("ts_*.json"))
    if traces and states:
        traces = sorted(traces, key=lambda p: p.stat().st_mtime)
        states = sorted(states, key=lambda p: p.stat().st_mtime)
        out.append((traces[-1].stat().st_mtime, traces[-1], states[-1]))
    return out


def _copy_pairs(root: Path) -> list[tuple[float, Path, Path]]:
    """Complete ledger pairs anywhere under notes/, e.g. a reading copy."""
    notes = Path(root) / "notes"
    if not notes.exists():
        return []
    scored = (Path(root) / "trace.jsonl").resolve()
    out: list[tuple[float, Path, Path]] = []
    for trace in notes.rglob("trace.jsonl"):
        if trace.resolve() == scored:
            continue
        state = trace.with_suffix(".state.json")
        if state.exists():
            out.append((trace.stat().st_mtime, trace, state))
    return out


def find_displaced_ledger(root: Path | str) -> tuple[Path, Path] | None:
    """A complete ledger pair that is not the workspace-root pair, if any.

    Archive copies are always eligible. Copies under ``notes/run*`` are used
    only when they carry the card this workspace is already bound to — otherwise
    a side client would be promoted into the scored slot.
    """
    root = Path(root)
    bound = _bound_card_id(root)
    archive = _archive_pairs(root)
    copies = _copy_pairs(root)
    if bound:
        matching = [p for p in archive + copies if _read_card_id(p[2]) == bound]
        if not matching:
            return None
        matching.sort()
        return matching[-1][1], matching[-1][2]
    if not archive:
        return None
    archive.sort()
    return archive[-1][1], archive[-1][2]


def side_ledgers(root: Path | str) -> list[Path]:
    """``trace.jsonl`` files under ``notes/``, which are not the scored ledger."""
    notes = Path(root) / "notes"
    if not notes.exists():
        return []
    scored = (Path(root) / "trace.jsonl").resolve()
    found = [p for p in notes.rglob("trace.jsonl") if p.resolve() != scored]
    found += list(notes.rglob("tr_*.jsonl"))
    return sorted(found)


def ensure_scored_ledger(root: Path | str) -> str | None:
    """Put the scored ledger back at the workspace root, or stop.

    A missing root ledger would look like a first open: the next
    ``client.open()`` would mint a new scorecard and the original play would
    stop counting. Restore from a copy under notes/ when one exists; if this
    workspace is already bound to a card and the ledger cannot be restored,
    raise instead of opening another card.

    Returns a short description of what was restored, or None when the root
    ledger was already in place or this is a genuine first open.
    """
    root = Path(root)
    trace, state = _ledger_pair(root)
    have_t, have_s = trace.exists(), state.exists()
    if have_t and have_s:
        return None

    displaced = find_displaced_ledger(root)
    restored: list[str] = []
    if displaced is not None:
        src_t, src_s = displaced
        if not have_t:
            trace.write_bytes(src_t.read_bytes())
            restored.append(src_t.name)
        if not have_s:
            state.write_bytes(src_s.read_bytes())
            restored.append(src_s.name)
        have_t, have_s = trace.exists(), state.exists()

    if have_t and have_s:
        return f"restored {'+'.join(restored)}" if restored else None

    # State without jsonl is the shape of a card that has been opened but has
    # not yet recorded an action. The solver starts a new ``python -c`` for
    # every command, so the next import sees this; jsonl is written per
    # action. Refusing here bricks the run after the first ``client.open()``.
    # Otherwise pin + state at 0 actions with no jsonl would make
    # ``from session import client`` raise and the solver could not even
    # RESET. A missing jsonl after actions have been spent is still a
    # stolen ledger and still refuses below.
    if have_s and not have_t:
        try:
            actions = int(json.loads(state.read_text(encoding="utf-8")).get("actions_used") or 0)
        except (OSError, ValueError, TypeError):
            actions = -1
        if actions == 0:
            return None

    bound = _bound_card_id(root)
    displaced_left = bool(_archive_pairs(root) or _copy_pairs(root))
    # **A pin alone is not evidence of play.** The pin is written at BUILD time
    # (so a bare `ArcClient` cannot mint in the window before
    # first open -- see `build_workspace`), which means a pinned-but-empty
    # workspace is the shape of every carded run's genuine first open, not of
    # a stolen ledger. Theft leaves evidence this check still reads: a state
    # file naming the card, a resume snapshot, a trace, or a displaced copy.
    # Refusing on the pin alone would brick every carded workspace at its first
    # `from session import client`.
    pin_only = (bound
                and not (_read_card_id(state) or _resume_card_id(root))
                and not have_t and not have_s and not displaced_left)
    if (bound or have_t or have_s or displaced_left) and not pin_only:
        missing = [n for n, ok in (("trace.jsonl", have_t), ("trace.state.json", have_s)) if not ok]
        where = f" (card {bound})" if bound else ""
        raise RuntimeError(
            f"the scored ledger is missing ({', '.join(missing)}) at {root}{where}. "
            f"Restore trace.jsonl and trace.state.json to the workspace root; "
            f"opening a new scorecard would discard the play already recorded. "
            f"Refusing."
        )
    return None


def scored_arc_client(root: Path | str, cls: type) -> type:
    """Wrap ArcClient so a non-root trace cannot be used for scored play."""
    root = Path(root)
    scored = (root / "trace.jsonl").resolve()

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if "trace_path" in kwargs:
            got = Path(kwargs["trace_path"]).resolve()
        elif len(args) >= 2:
            got = Path(args[1]).resolve()
        else:
            got = Path("trace.jsonl").resolve()
        if got != scored:
            raise RuntimeError(
                f"scored play uses only {scored}; a client writing {got} "
                f"would open a side card that is not scored. Refusing."
            )
        ensure_scored_ledger(root)
        return cls(*args, **kwargs)

    return wrapped  # type: ignore[return-value]


def _claude_binary() -> str:
    return os.environ.get("CLAUDE_BINARY") or shutil.which("claude") or "claude"


def _supports_flag(flag: str) -> bool:
    try:
        out = subprocess.run(
            [_claude_binary(), "--help"], capture_output=True, text=True, timeout=30
        )
        return flag in (out.stdout + out.stderr)
    except (OSError, subprocess.SubprocessError):
        return False


def build_workspace(config: Ccarc3Config, info: GameInfo | None = None,
                    *, arc_root: str | None = None) -> Workspace:
    """Create the workspace for one game."""
    if info is None:
        matches = [g for g in list_games(config.api_key) if g.game_id == config.game_id]
        if not matches:
            raise ValueError(f"no such game: {config.game_id}")
        info = matches[0]

    root = Path(config.out_dir) / config.game_id
    root.mkdir(parents=True, exist_ok=True)
    (root / "notes").mkdir(exist_ok=True)

    # result.json describes a *finished* run. Leaving the previous one in place
    # while a new one is in flight means `report` presents a stale outcome as
    # final, and anything watching the directory sees a run that has not started
    # as already complete. It is rewritten by collect_outcome at the end.
    (root / "result.json").unlink(missing_ok=True)

    trace = root / "trace.jsonl"
    if config.fresh:
        # **Every sidecar, not just the ledger.** `fresh` means "this
        # workspace has no history": clearing the ledger, the state file and
        # the rule book while leaving `trace.decision.json` and
        # `trace.open.json` in place would let a reused directory inherit
        # both, and both are read by guards that decide whether a run is
        # continued or banked:
        #
        # - `trace.decision.json` carries `declarations`. A fresh build over a
        #   finished run would report `_finish_declared() == 2`, and at two
        #   declarations `collect_outcome` skips the give-up arm and
        #   `run_game` skips all three resume arms. The new run banks as a
        #   real loss in silence -- the exact failure the abandoned-replay arm
        #   exists to close, arriving through a stale file instead of a gap in
        #   the logic.
        # - `trace.open.json` carries this attempt's card boundary, which is
        #   only true when taken before the attempt's first play. A stale one
        #   hands `card_disagreement` somebody else's rows -- the shape that
        #   discards a winning run.
        #
        # A caller that mints a new `attempt_N/` per attempt never meets this;
        # a caller that reuses its out_dir -- `cli.py --fresh` does -- depends
        # on every sidecar being cleared. Listed explicitly rather than
        # globbed: a glob over `trace.*` would also take files a future reader
        # adds for a different purpose.
        for stale in (trace,
                      trace.with_suffix(".state.json"),
                      trace.with_suffix(".decision.json"),
                      trace.with_suffix(".open.json"),
                      root / "rules.json",
                      root / SCORED_PIN_NAME):
            stale.unlink(missing_ok=True)
    resumed = trace.exists()

    # **The pin is written at BUILD time, not at first open, and that ordering
    # is load-bearing.** Were `.scored_card.json` written by `open()`, then in
    # the window between workspace creation and the rendered client's first
    # open, a solver-constructed bare `ArcClient` (no `card_id`, `require_card`
    # defaulting False) would find nothing naming the run's card and mint its
    # own; the run would then score onto a card no submission can use, recorded
    # as `foreign_card`. With the pin on disk before any client exists,
    # `_refuse_side_ledger` can hold every construction to it.
    if config.card_id:
        pin_scored_card(root, config.card_id)

    budget = info.suggested_budget(config.budget_multiple)

    (root / "session.py").write_text(
        SESSION_TEMPLATE.format(
            game_id=info.game_id,
            # Emitted only when a card is being shared: a keyword the solver
            # can see is a question the solver can ask, and on the usual
            # one-card-per-game run there is nothing here to explain.
            # `require_card` rides with the id: a workspace built for a shared
            # card must never, on any later rebuild or resume, decide that
            # minting its own is an acceptable default.
            card_line=(
                f"    card_id={config.card_id!r},\n"
                f"    require_card=True,\n" if config.card_id else ""
            ),
        ),
        encoding="utf-8",
    )
    # **One file, not two.** Claude Code injects CLAUDE.md unasked and
    # re-injects it after a compaction, so inlining the doctrine makes delivery
    # a property of the harness rather than of the agent's compliance, and it
    # removes the class of defect where two files drift into saying different
    # things.
    (root / "CLAUDE.md").write_text(_workspace_claude_md(info, budget), encoding="utf-8")
    # **The doctrine is inlined into CLAUDE.md, never copied in as a second
    # file.** `build_workspace` writes into a stable root that a resume
    # rebuilds, and the guards below read CLAUDE.md only, so any `DOCTRINE.md`
    # left in the root would sit beside it under a name an agent opens
    # unprompted, carrying an unredacted doctrine past both guards.
    (root / "DOCTRINE.md").unlink(missing_ok=True)
    redact_self_reference(root, config.game_id)
    # Checked here, after the redactor, because this is the last moment before
    # the solver can read it. See `assert_no_environment_named`.
    assert_no_environment_named(root / "CLAUDE.md", own=config.game_id)
    # **`meta.json` lives in the solver's own working directory**, so it carries
    # neither the per-level array nor the action cap. The array is a summary of
    # how efficiently these levels have been cleared before; the cap is that
    # array's total times `budget_multiple`, so a solver that reads its cap
    # recovers the array. Neither is written here, so the withholding does not
    # depend on a later pass removing them. `_action_budget` falls back to the
    # parent's own `GameInfo`, which is the designed path.
    # **Only what harness tooling reads.** Resume tooling reads `model` and
    # `effort`; nothing anywhere reads
    # `title`, `tags` or `levels` from here. `tags` is this game's input
    # modality and `levels` is `len(baseline_actions)` -- the length of the
    # array the whole harness withholds, and silently 0 wherever it is hidden.
    # `available_actions` and `win_levels` carry both from the first frame, so
    # they are the agent's to read rather than ours to hand over. The full
    # record still goes to the scratch root below.
    meta = {
        "game_id": info.game_id,
        "model": config.model,
        "effort": config.effort,
    }
    (root / "meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                    encoding="utf-8")
    # The full record, for harness-side tooling, outside the workspace and
    # outside the guardrail. Same placement as `result.full.<game>.json`.
    scratch = os.environ.get("CCARC3_SCRATCH")
    if scratch:
        full = pathlib.Path(scratch) / f"meta.full.{info.game_id}.json"
        try:
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_text(
                json.dumps(dict(meta,
                                baseline_actions=list(info.baseline_actions),
                                action_budget=budget), indent=2) + "\n",
                encoding="utf-8",
            )
            full.chmod(0o600)
        except OSError:
            pass                      # bookkeeping must not fail a launch

    env = dict(os.environ)
    key = config.api_key or os.environ.get("ARC_API_KEY", "")
    if key:
        env["ARC_API_KEY"] = key
    # **A STAGED copy, not the repository.** Putting the checkout's own `src` on
    # the child's path would hand the solver everything a working tree holds --
    # the unstripped master doctrine included: the strip and
    # `redact_self_reference` both rewrite a copy, so the originals would sit
    # one `read_text()` away. The stage carries no prompt at all.
    #
    # Staged into the attempt directory, which the guardrail holds and re-binds, so a
    # confined run still imports normally.
    repo = Path(__file__).resolve().parents[2].parent
    src = stage_package(Path(root).resolve().parent, driver="arc_3")
    env["PYTHONPATH"] = f"{src}:{env.get('PYTHONPATH', '')}".rstrip(":")
    # **The cap is never exported to the child.** The environment is one
    # `printenv` away for the solver, and `budget_multiple` is a default in
    # this package's own source on the solver's PYTHONPATH, so a cap in the
    # environment inverts to the withheld baseline total. Nothing in the child
    # consumes it: the rendered workspace `session.py` passes `max_actions=0`
    # (see the note on SESSION_TEMPLATE) and the real ceiling is armed in
    # `arc_proxy`, in a process the solver does not run. The withholding lives
    # in the builder every run goes through, not in a later pass.

    # The scored ledger is exactly this file. A client that writes anywhere else
    # (notes/run*, an archive path) opens a side card the harness does not score.
    env["CCARC3_SCORED_TRACE"] = str((root / "trace.jsonl").resolve())

    # **The in-process belt against a baseline leak.** `baselines_for` and
    # `list_games` return nothing under this flag, so a solver that reaches the
    # baseline-bearing endpoint still learns nothing. Two network layers back it
    # up: `/api/games` is off the shim's allowlist, and `/api/scorecard/close`
    # responses are scrubbed of `level_baseline_actions` AND `level_scores` (a
    # score inverts back to a baseline, since the solver knows its own action
    # count). With baselines unreachable, `baseline_here`, `score_now` and
    # `score_ceiling` all return None even when a solver constructs the client
    # with `quiet_pace=False`.
    #
    # The belt is not redundant on top of those layers: it is the one that
    # still holds if a run is ever launched without the network namespace.
    #
    # Child env only. The DRIVER must keep reading baselines -- it scores the run.
    env["CCARC3_HIDE_BASELINES"] = "1"

    # Put the interpreter that actually has numpy first on PATH. Bare
    # `python3` can resolve to an interpreter without numpy, and a solver then
    # spends turns on `ModuleNotFoundError` and discovers the venv path by
    # trial. Making `python3` resolve to the right thing removes the whole
    # class of error rather than documenting a way around it.
    # From the REPO, not from `src`: `src` is the staged package inside the
    # attempt directory, and there is no interpreter there.
    venv_bin = repo / ".venv" / "bin"
    if not ((venv_bin / "python3").exists() or (venv_bin / "python").exists()):
        # **A repo with a `.venv` beside it is a development convenience, not a
        # guarantee.** Installed from a wheel, or checked out without one,
        # there is no such directory; leaving PATH alone would let `python3`
        # resolve to whatever comes first, and the solver would meet the
        # ModuleNotFoundError this block exists to prevent. The interpreter
        # running this has the package, and therefore numpy, by construction.
        venv_bin = Path(sys.executable).parent
    if (venv_bin / "python3").exists() or (venv_bin / "python").exists():
        env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}".rstrip(":")

    # **Take the ARC key away from the solver.**
    #
    # The in-process guard cannot close the deliberate route on its own: a
    # solver holding the credential could reach upstream by hand, and a run that
    # does so is not a clean run. The reasoning and the routes are
    # deliberately not written here: this module is on the solver's own import
    # path and `inspect.getsource` returns comments.
    #
    # With `CCARC3_PROXY_URL` set, the client is pointed at `arc_proxy`, which
    # holds the real key and forwards only what a solver legitimately needs, and
    # the key is removed from the solver's environment entirely. A hand-rolled
    # request then has nothing to authenticate with.
    #
    # Unset, nothing changes: the key stays and the client talks upstream
    # directly. That is deliberate, because this module is read fresh by every
    # `python -c` the solver runs, so a hard switch would break games already in
    # flight.
    # **Passed in, not read from the environment, when the caller knows it.**
    # With several games in flight each has its own shim on its own port, and
    # a single process-wide `CCARC3_PROXY_URL` cannot name more than one of
    # them -- two builders racing on `os.environ` would hand a solver the
    # other game's proxy, and with it the other game's action budget and ARC
    # session. The env var remains the fallback for single-game callers.
    proxy = arc_root or os.environ.get("CCARC3_PROXY_URL")
    if proxy:
        env["CCARC3_ARC_ROOT"] = proxy
        env.pop("ARC_API_KEY", None)
        env.pop("ARCPRIZE_API_KEY", None)
    # **The parent's own bookkeeping never goes to the child.** `install()`
    # keeps `CCARC3_PROXY_URL` in this process's environment, and
    # `env = dict(os.environ)` would copy it straight through -- handing a
    # solver the address of a shim that is not its own. With games running
    # concurrently that is another game's budget; with the startup probe it is
    # no budget at all. The child needs exactly one root, `CCARC3_ARC_ROOT`,
    # and it is set above.
    env.pop("CCARC3_PROXY_URL", None)

    # **The disclosure switch must not reach a solver.** It changes nothing --
    # the shim strips the baseline unconditionally -- but its NAME alone would
    # tell a solver that a human baseline exists and that something chose
    # whether it may see one, and `env` starts life as `dict(os.environ)`, so
    # a value exported on the box would otherwise be inherited.
    env.pop(RETIRED_MODE_VAR, None)

    # **A solver is never gated on reading a document.** An operator hook may
    # refuse a session's first turns until a project document has been read.
    # That is wrong for a solver: it knows nothing but its own workspace, and
    # such a gate would either block the run or hand it a document describing
    # the experiment it is the subject of.
    #
    # `CONTEXT_GATE_DISABLE=off` is the value that hook family checks; the name
    # reads backwards -- `off` means the gate is off. Harmless where no such
    # hook is installed.
    #
    # **It has to be set here, in the spawn, and cannot be set from a shell.**
    # The hook reads the process environment, so an exported shell value never
    # reaches a child the harness launches later.
    env["CONTEXT_GATE_DISABLE"] = "off"

    return Workspace(
        root=root,
        config=config,
        info=info,
        initial_prompt=_initial_prompt(info, budget, resumed=resumed),
        session_id=str(uuid.uuid4()),
        env=env,
        resumed=resumed,
    )


def _workspace_claude_md(info: GameInfo, budget: int) -> str:
    """The whole solver-facing contract, as one auto-loaded file.

    Claude Code injects ``CLAUDE.md`` from the working directory without being
    asked, and injects it again after a compaction. So anything the solver must
    always have belongs here rather than in a second file it has to remember to
    open -- delivery becomes a property of the harness instead of the agent's
    compliance, and two files can no longer drift into contradicting each other.
    """
    doctrine = (ASSETS / "CCARC3_DOCTRINE.md").read_text(encoding="utf-8")
    return _workspace_preamble(info, budget) + PREAMBLE_SEPARATOR + doctrine


#: **One document, not two stapled together.** The header and the doctrine are
#: concatenated with a blank line: no rule, no second title. The boundary the
#: redactor needs is the first `##` heading, not a marker in the text — see
#: `redact_self_reference`.
PREAMBLE_SEPARATOR = "\n"


def _workspace_preamble(info: GameInfo, budget: int) -> str:
    """The opening of `CLAUDE.md`: what the agent is doing, and nothing else.

    It names no environment. `status()` prints `<game_id>: level N/M` on every
    call, so the id reaches the agent from the API it is told to use, which
    makes `redact_self_reference` a pure backstop rather than a boundary
    something depends on.
    """
    return f"""\
You are playing an unfamiliar video game through an API. Working out its rules
**is** the task.
"""


def _initial_prompt(info: GameInfo, budget: int, *, resumed: bool = False) -> str:
    # **Everything else the agent needs is in CLAUDE.md**, which Claude Code
    # injects unasked and re-injects after a compaction. Repeating any of it
    # here is duplication; `info.levels` in particular is
    # `len(baseline_actions)` and reads 0 wherever baselines are hidden, so it
    # must not appear in a prompt.
    if resumed:
        return (
            "You are resuming an interrupted run. The game is still open and "
            "your previous actions are recorded.\n\n"
            "Start with `client.status()` and `client.history()` to see where "
            "you are. If `rules.json` exists, read it."
        )
    return "Play the game."




#: Four characters, word-bounded, starting with a letter, containing a digit.
#: Wider than the environment-id shapes the doctrine uses, on purpose: a guard
#: whose net is the current text's habits cannot see the next edit's.
_ID_SHAPE = re.compile(
    r"(?<![\w-])(?=[a-z][a-z0-9]{3}(?!\w))(?=[a-z0-9]*\d)[a-z][a-z0-9]{3}"
)

#: Tokens of that shape a solver-facing document may legitimately contain.
_ID_ALLOWED = frozenset({"utf8", "int8", "sha1", "md5s", "1e-9", "0x1f"})


def assert_no_environment_named(path: Path, *, own: str = "") -> None:
    """Refuse to ship a solver-facing document that names an environment.

    **A runtime post-condition, not a check on the source.** A check that
    reads the file in *this* working tree cannot see a run launched from a
    different checkout -- an older clone, a container rebuilt from an image --
    which ships whatever that tree contains.

    So the check happens at the moment the document is written. A stale tree
    refuses to launch and says why, instead of handing a solver a passage about
    the game it is playing.

    `own` is the run's own environment, which the document must be able to name:
    it is written into `meta.json` and the workspace guide by construction, and
    hiding a solver's own id from it would be theatre.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    short = own.split("-")[0]
    found = sorted({
        m.group(0) for m in _ID_SHAPE.finditer(text)
        if m.group(0) not in _ID_ALLOWED and m.group(0) != short
    })
    if found:
        raise RuntimeError(
            f"{path.name} names {len(found)} environment-shaped token(s) other "
            f"than this run's own: {found}. A solver-facing document must not "
            f"name an environment. This tree may predate the redaction; "
            f"rebuild from a current checkout rather than launching."
        )


def redact_self_reference(root: Path, game_id: str) -> int:
    """Remove the doctrine's worked examples that name *this* game.

    **A solver must not read about its own previous attempts.** The doctrine
    earns its keep with concrete worked examples, and those examples name the
    environments they came from. That is fine for a game you are not playing
    and contamination for one you are: without this pass, a run of a named game
    would open by reading a summary of a prior attempt at itself.

    Table rows naming the game are dropped whole -- a row is self-contained, and
    blanking the id would leave its level count and action total, which identify
    it just as well. Prose mentions have the id replaced by a neutral phrase,
    because deleting a sentence mid-paragraph mangles the argument around it.

    **This does not make the doctrine unidentifiable.** A prose example that
    reports a score and a levels-cleared
    count for a seven-level game still tells a solver on a seven-level game
    something about a seven-level game. What it removes is the direct, named,
    self-referential leak. Returns the number of lines changed.

    **It may relabel an OUTCOME. It must never be asked to relabel a MECHANIC.**
    A doctrine passage stating a mechanic of a named environment must
    keep the name: relabelled to "another environment", the run it reaches can
    record the mechanic as a finding of its own, and the self-reference is
    visible neither to an auditor nor to the solver. That hides the leak
    rather than removing it -- strictly worse than leaving the name in.

    This function cannot tell the two apart -- that is a judgement about a
    passage, not a property of a string -- so the rule is enforced upstream, on
    the doctrine itself: a game id may appear only under a heading numbered 0,
    which is where the doctrine reports scores.

    **Scope: the doctrine half of CLAUDE.md, never the preamble.** The preamble
    names this run on purpose -- its header and its `| id |` row are how the
    solver learns which game it is playing -- while the doctrine must not name
    it at all. :func:`_workspace_claude_md` therefore redacts the doctrine
    before concatenating, and this function is the file-level backstop that
    runs on the doctrine portion only.
    """
    doc = root / "CLAUDE.md"
    if not doc.exists():
        return 0
    short = game_id.split("-")[0]
    if not short:
        return 0
    text = doc.read_text(encoding="utf-8")
    # **The boundary is the first `##` heading.** Everything above it is the
    # identity header, which names this run on purpose; everything below is the
    # doctrine, which must name no environment at all. A literal marker would
    # have to appear in a document the solver reads.
    cut = text.find("\n## ")
    head, body = (text[:cut + 1], text[cut + 1:]) if cut != -1 else ("", text)
    sep = ""
    out, changed = [], 0
    for line in body.splitlines():
        if short not in line:
            out.append(line)
            continue
        changed += 1
        if line.lstrip().startswith("|"):
            continue                      # drop the whole row
        out.append(re.sub(rf"`?{re.escape(short)}[a-z0-9-]*`?",
                          "another environment", line))
    if changed:
        doc.write_text(head + sep + "\n".join(out) + "\n", encoding="utf-8")
    return changed


def build_cli_args(workspace: Workspace) -> list[str]:
    config = workspace.config
    args = [
        _claude_binary(),
        "-p",
        workspace.initial_prompt,
        "--output-format",
        "stream-json",
        "--verbose",
    ]
    # **Assigned, so the solver owns an id nothing else shares.** Without it the
    # child inherits `CLAUDE_CODE_SESSION_ID` from the driver and every solver on
    # the box reports -- and files its transcript under -- the same id. Nothing
    # downstream can then tell two solvers apart, and `--resume` on that id would
    # reach the driver's own conversation rather than the solver's.
    if workspace.session_id and _supports_flag("--session-id"):
        args += ["--session-id", workspace.session_id]
    if config.model:
        args += ["--model", config.model]
    if config.effort and _supports_flag("--effort"):
        args += ["--effort", config.effort]
    # **Capture the model's reasoning. It costs nothing extra.**
    #
    # Thinking is billed inside `output_tokens` whether or not it is returned,
    # and without this flag every block arrives as `thinking: ""` with only a
    # signature.
    #
    # Not guarded by `_supports_flag`, which greps `--help`, because this
    # option is undocumented there. It is nonetheless real:
    # `--thinking-display bogusvalue` reports *"Allowed choices are summarized,
    # omitted"*.
    #
    # Safe to pass blind: this CLI exits 0 on unrecognised options, so a build
    # without it ignores the flag rather than failing every game at launch.
    args += ["--thinking-display", "summarized"]
    if config.permission_mode:
        # bypassPermissions maps to --dangerously-skip-permissions, which the
        # CLI refuses under root -- and a containerised harness is usually
        # root. The refusal arrives as a one-line stderr and an empty run.
        args += ["--permission-mode", resolve_permission_mode(config.permission_mode)]
    if config.allowed_tools:
        args += ["--allowedTools", ",".join(config.allowed_tools)]
    if config.disallowed_tools:
        args += ["--disallowed-tools", ",".join(config.disallowed_tools)]
    args += list(config.extra_cli_args)
    return args


def _signal_group(proc: "subprocess.Popen", sig: int) -> None:
    """Signal the solver's whole process group, falling back to the child.

    `start_new_session=True` makes the child a group leader, so one `killpg`
    reaches every descendant it spawned. The fallback matters on the path where
    the child has already exited: `getpgid` then raises ProcessLookupError, and
    a timeout arm that died there would skip the wait below it.
    """
    try:
        os.killpg(os.getpgid(proc.pid), sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


#: Seconds a solver gets to shut down cleanly after SIGTERM before SIGKILL.
TERM_GRACE_S = 30


NO_DECISION_PROMPT = """You cleared every level and then stopped.

Call `client.finish()` to end the run, or keep playing.
"""
# **The one relaunch the replay design permits, and why it is not an override.**
# A bare string after a constant is a docstring in every sense that matters
# here: it survives staging and `inspect.getsource` reaches it. A comment does
# not, so the reasoning is a comment.
#
# A declared decision is never overridden; an undeclared exit may be resumed.
# `finish()` is how a solver declares, and the client's post-clear reminder
# teaches it at the moment it matters. When a solver clears the game and exits
# WITHOUT declaring, the parent cannot tell a choice from a crash or a context
# death -- the evidence is identical: a dead process and no decision sidecar. So
# it asks the question that was never answered, once, neutrally, and honours
# whatever comes back. Relaunching a solver that has already read the reminder
# and chosen to stop would overrule a decision made with the information in
# hand, so no arm here does that.
#
# The message names no ask budget and no harness vocabulary. A count the agent
# cannot act on is pressure on a message whose whole design is neutral, and the
# decision sidecar is not something the agent has heard of.
#
# No figure appears in the prompt, and none may be added.


ABANDONED_REPLAY_PROMPT = """You cleared every level, then started another play
and stopped part-way through it.

Finish that play, or call `client.finish()` to end the run.
"""
# **What to say to a solver that wins, replays, and then goes quiet mid-play.**
# In a comment, not in a string beneath the constant: `staging` strips
# comments, while any string literal in this module is shipped into the
# workspace and readable with `inspect.getsource`.
#
# **The gap this closes.** Two arms in `run_game` decide whether a solver is
# told to carry on, and between them would otherwise be a hole. The give-up arm
# is gated on `not won`, so it cannot fire once a game has been cleared. The
# no-decision arm is gated on the FINAL playthrough having cleared every level,
# so it cannot fire while a replay is unfinished. A run that wins and then
# abandons a replay satisfies neither, and the harness would say nothing at all.
#
# **Why this is directive where `NO_DECISION_PROMPT` is neutral, and why that is
# still not an override.** A solver reaches the replay only by choosing
# `restart_for_replay()` over `finish()` at the winning frame, having just read
# the reminder that teaches both. That choice IS the declaration that the banked
# play was not good enough; going quiet part-way through abandons it without
# saying so, which is exactly the undeclared exit the design permits resuming.
# The door out stays open and one-way.
#
# No game id, count or score appears in the prompt, and none may be added.

NUDGE_PROMPT = """You stopped with levels uncleared.
Keep playing until every level is cleared."""
# **What a solver is told when it quits while it can still act.**
#
# **Two phases, in that order, and the order is the whole point.** Finishing is
# a gate: a game left unfinished is capped whatever else was done well, so
# nothing else is worth asking for until every level is cleared. Once it is, the
# only remaining lever is doing the same thing in fewer moves.
#
# **Asking for the replay is free.** The scorecard keeps the better attempt
# rather than the later one, so a weaker second pass cannot cost anything the
# first pass earned.
#
# **It says only what CLAUDE.md does not.** CLAUDE.md states the objective
# ("clear every level, in as few actions as possible") and the replay rule, and
# Claude Code re-injects it unasked, including after a compaction, so restating
# either here would be restating the prompt to an agent that still has it.
#
# The replay ask belongs in `WIN_REPLAY_REMINDER`, which fires once a game is
# cleared. This message fires at a solver that has NOT cleared the game, so
# "then play it again from the beginning" would ask for something it cannot do
# yet; `enable_nudging()` turns both on together, so asking there costs no
# prompting here.
#
# **No figure appears in it, and none may be added.** "You still have actions
# left" tells a solver only what it could already infer from not having been
# refused; a count would say considerably more. This paragraph is a comment
# rather than a docstring because the module is on the solver's own PYTHONPATH.


def _max_nudges() -> int:
    """How many times a solver that quits early is told to carry on.

    Distinct from `GIVE_UP_ATTEMPTS`, which counts whole re-runs from scratch in
    fresh workspaces. A nudge continues the *same conversation*: the solver keeps
    its context, its rules.json and its place in the game, and the cost is one
    more turn rather than one more game.

    Not bounded by a count: the driver sets it effectively unlimited. What ends
    the run is the spent ceiling or the wall clock -- a solver answering this
    arm plays, so the ceiling arrives.

    **Off unless a caller asks for it, and that default is deliberate.**
    Nudging changes what `run_game` does for *every* caller -- a give-up that
    ends one launch now ends three -- and a contract change of that size is
    not one a caller should inherit silently.

    A driver turns it on for the arms that want it, so the driver an operator
    actually runs gets the behaviour while the library stays predictable. `0`
    marks the give-up and leaves the caller to discard the run and re-run the
    game from scratch.
    """
    raw = os.environ.get("CCARC3_MAX_NUDGES", "")
    # "unlimited": a fixed ask count can run out levels
    # short of the end with budget still unspent. The arm still cannot loop
    # forever on its own: the action budget, the wall clock, and a win each
    # end the attempt regardless of how many asks remain. Represented as a
    # large int rather than infinity so every consumer stays integer-typed.
    if raw.strip().lower() == "unlimited":
        return 10**9
    try:
        n = int(raw)
    except ValueError:
        if raw:
            print(f"CCARC3_MAX_NUDGES={raw!r} is not a number; nudging disabled",
                  flush=True)
        return 0
    return max(0, n)


def _rotate_stream(ws: Workspace) -> None:
    """Move `stream.jsonl` aside so the next launch cannot overwrite it.

    Every launch is a separate record — the first attempt, and each nudge after
    it. Losing one loses the only evidence of how that handoff actually went,
    which is exactly what needs reading when a nudge fails to land.
    """
    stream = ws.root / "stream.jsonl"
    if stream.exists():
        n = len(list(ws.root.glob("stream.*.jsonl"))) + 1
        stream.rename(ws.root / f"stream.{n}.jsonl")


def _nudge_args(ws: Workspace, prompt: str = NUDGE_PROMPT) -> list[str] | None:
    """CLI args that resume this solver's own session and tell it to continue.

    `None` when the CLI cannot do it, which is not a failure: the caller then
    leaves the give-up marked and the driver re-runs the game as before.

    **Resumes an id we assigned, never one read from the stream.** See
    `Workspace.session_id` — every event a solver emits carries the driver's id,
    so an id discovered from the stream would resume the wrong conversation.
    """
    if not ws.session_id:
        return None
    if not (_supports_flag("--resume") and _supports_flag("--session-id")):
        return None
    args = [
        _claude_binary(), "-p", prompt,
        "--resume", ws.session_id,
        "--output-format", "stream-json", "--verbose",
    ]
    config = ws.config
    if config.model:
        args += ["--model", config.model]
    if config.effort and _supports_flag("--effort"):
        args += ["--effort", config.effort]
    if config.permission_mode:
        args += ["--permission-mode", resolve_permission_mode(config.permission_mode)]
    if config.allowed_tools:
        args += ["--allowedTools", ",".join(config.allowed_tools)]
    if config.disallowed_tools:
        args += ["--disallowed-tools", ",".join(config.disallowed_tools)]
    return args


@dataclass
class _Asks:
    """How many times each arm of the harness may still speak, and how often one has.

    Held together because both launch paths -- one process per ask, and one
    conversation the harness speaks into -- have to spend the same budget under
    the same rules. Two copies of these counters would be two policies.
    """

    nudges: int
    """`_max_nudges()`: a solver that quit with actions in hand may be told so."""

    decision_at: int | None = None
    """`actions_used` when the decision question was last asked, or None.

    **A progress condition, not a count.** A count stops asking a solver that
    is still working, and it stops on an arithmetic fact rather than on
    anything about the run.

    What makes the give-up nudge safe to leave unlimited is that a solver
    answering it spends actions, so the ceiling ends the run. That does not hold
    here: this arm fires AFTER a full clear, so a solver that exits again
    without acting accrues nothing, `budget_exhausted` never arrives, and only
    the wall clock would stop an unconditional re-ask. So the bound is the same
    resource argument, made directly: ask again whenever the solver has PLAYED
    since the last ask, and stop when it has not -- because at that point asking
    demonstrably changes nothing. A solver that keeps engaging is never cut off.
    """

    replay_at: int | None = None
    """`actions_used` when the abandoned-replay question was last asked.

    A count is the wrong bound here for the reason it is wrong for
    `decision_at`: a promising replay abandoned mid-climb can die a hesitation
    short of the end. Same progress rule as `decision_at`.
    """

    made: int = 0
    """How many asks have actually been sent."""


def _run_is_over(ws: Workspace, outcome: dict[str, Any], deadline: float | None) -> bool:
    """Is there any point asking anything? Narrates the answer when there is not."""
    if outcome.get("budget_exhausted"):
        # **Say why the seat is being released.** A run that stops here with no
        # nudge and no error looks identical in the log to one that simply
        # finished, and the difference matters: the ceiling was spent, so the
        # next occupancy has to be a fresh attempt rather than a resume.
        #
        # **A spent ceiling ends the run whatever else is true.** No ask can be
        # answered by a solver that cannot take another action.
        print(f"    {ws.info.game_id}: action budget spent — releasing the "
              f"seat; a spent ceiling cannot take another command",
              flush=True)
        return True
    return deadline is not None and time.monotonic() >= deadline


def _next_ask(outcome: dict[str, Any], asks: _Asks, *,
              timed_out: bool) -> tuple[str, str] | None:
    """What the harness should say next, or None to let the run end.

    Spends `asks`. The arms are ordered, and the order is load-bearing: the
    third exists only because a run fell through the first two.
    """
    # **`levels_total` must be a real number, not merely equal.** A bare `==`
    # would read `None == None` as True for any outcome carrying neither
    # field, treating a result that said only `won` as a full clear and asking
    # it about a decision it never owed.
    total = outcome.get("levels_total")
    won_outright = bool(
        outcome.get("won") and total
        and outcome.get("levels_reached_final_playthrough") == total
    )
    declared = int(outcome.get("finish_declared") or 0)

    if outcome.get("gave_up") and (
            (declared == 0 and asks.nudges > 0)
            # A quit that carries one declaration still gets the statement of
            # fact -- once. The harness may state facts once; a decision
            # reaffirmed after hearing them stands (`collect_outcome` never
            # marks `gave_up` on a reaffirmed declaration), and a repeat of the
            # same nudge at a solver that has already declared is the override
            # this design exists to remove.
            or (declared == 1 and asks.made == 0)):
        asks.nudges = max(0, asks.nudges - 1)
        return NUDGE_PROMPT, "quit with actions in hand"

    used = int(outcome.get("actions_used") or 0)

    if (won_outright and not timed_out and declared == 0
            and (asks.decision_at is None or used > asks.decision_at)):
        # An exit after a full clear with no declaration is an absence of a
        # decision, not a decision: the reminder taught `finish()` at the moment
        # of the win, so a deliberate stop had every chance to say so.
        #
        # Asked again for as long as the solver keeps playing between asks. When
        # it stops playing, asking has stopped doing anything and silence stands.
        asks.decision_at = used
        return NO_DECISION_PROMPT, "cleared with no decision on record"

    if (outcome.get("won") and not timed_out and declared == 0
            and total
            and (outcome.get("levels_reached_final_playthrough") or 0) < total
            and (asks.replay_at is None or used > asks.replay_at)):
        # **The hole between the two arms above.** `gave_up` is gated on `not
        # won` and never fires here; `won_outright` is gated on the final play
        # having cleared everything and never fires here either. A solver that
        # won, chose to replay, and then went quiet part-way through would fall
        # straight through both and be banked in silence -- see
        # `ABANDONED_REPLAY_PROMPT`.
        asks.replay_at = used
        return (ABANDONED_REPLAY_PROMPT,
                "won, then left a replay unfinished with no decision")

    return None


def run_game(config: Ccarc3Config, info: GameInfo | None = None) -> dict[str, Any]:
    """Build a workspace and run one solver session against one game."""
    ws = build_workspace(config, info)
    _record_resume_state(ws)
    args = build_cli_args(ws)
    stream = ws.root / "stream.jsonl"
    if stream.exists():
        # Opening with "w" would destroy the previous attempt's stream when a
        # resume starts, removing the only record of how the handoff went --
        # and the handoff is exactly what needs reading when a resume fails.
        # Keep each attempt.
        #
        # **Not gated on ``ws.resumed``.** A run killed before it wrote a single
        # action leaves a stream but no trace, so the next launch is not a resume
        # (``resumed = trace.exists()``), and a gate on ``resumed`` would let it
        # overwrite the only record of what that attempt spent. Any existing
        # stream is a previous attempt whatever the trace says, so rotate on
        # existence.
        n = len(list(ws.root.glob("stream.*.jsonl"))) + 1
        stream.rename(ws.root / f"stream.{n}.jsonl")

    # **One deadline for the whole run, not one per launch.** A nudged run makes
    # several `claude` invocations, and giving each a fresh
    # `wall_clock_timeout_s` would let a game run for two or three times the
    # limit its caller set -- a caller picks that limit so a run fits inside a
    # container window, so multiplying it silently defeats the choice.
    deadline = (time.monotonic() + config.wall_clock_timeout_s
                if config.wall_clock_timeout_s else None)
    asks = _Asks(nudges=_max_nudges())
    if _live_enabled():
        return _run_live(ws, args, deadline, asks)

    while True:
        code, timed_out = _launch(ws, args, deadline)
        # Passed IN rather than attached after: `collect_outcome` writes
        # result.json before returning, so anything set on the dict afterwards
        # never reaches the file a post-run audit reads.
        outcome = collect_outcome(ws, exit_code=code, timed_out=timed_out,
                                  extra={"nudges": asks.made,
                                         # What the asks COST in launches, which
                                         # is what a post-run audit is really
                                         # about. Here every ask is a relaunch;
                                         # in a live session none of them is.
                                         "ask_launches": asks.made,
                                         "session_id": ws.session_id})
        if _run_is_over(ws, outcome, deadline):
            return outcome
        ask = _next_ask(outcome, asks, timed_out=timed_out)
        if ask is None:
            return outcome
        prompt, why = ask

        nudge = _nudge_args(ws, prompt)
        if nudge is None:
            # The CLI cannot resume; leave whatever was marked and let the
            # driver re-run the game from scratch.
            return outcome

        asks.made += 1
        print(f"    {ws.info.game_id}: {why} — "
              f"resume {asks.made}, resuming its session", flush=True)
        _rotate_stream(ws)
        args = nudge



# Only these. A rate limit, a 500 or a refused tool is not a credential problem,
# and rewriting the credential under a solver that is merely being throttled
# would spend a rotation for nothing and hide the real cause.
_AUTH_FAILURE = re.compile(
    r"authenticat|oauth|401|invalid[_ ]token|token (has )?expired|session expired",
    re.IGNORECASE)


def _is_auth_failure(text: str) -> bool:
    """Does this failed turn look like a dead credential?"""
    return bool(text and _AUTH_FAILURE.search(text))


def _probe_enabled() -> bool:
    """Whether a launch tests its credential against the server first.

    On by default: the cost is one haiku call on a token that is expiring
    anyway, and the failure it catches costs a whole attempt and looks exactly
    like a hard game. `CCARC3_CREDENTIAL_PROBE=0` turns it off.
    """
    return os.environ.get("CCARC3_CREDENTIAL_PROBE", "1") != "0"


def _live_enabled() -> bool:
    """Whether the harness holds the solver's session open and speaks into it.

    **Off by default, deliberately** -- not caution about the mechanism. It
    changes what a run *is*: the child no longer exits when the
    model stops talking, so the driver decides when a run ends rather than
    discovering it. That is a design decision a caller must opt into.
    `CCARC3_LIVE=1`.
    """
    return os.environ.get("CCARC3_LIVE", "0") == "1"


def _harness_outbox_path(ws: Workspace) -> Path:
    """Where the client leaves what it wants said. NOT in the workspace.

    The attempt directory, one level up: the workspace is the directory the
    solver works in, and it is a public place -- anything a file there holds
    can be `cat`ed straight into the run's own stream. Nothing here is secret,
    but the placement rule is worth keeping uniform.
    """
    return Path(ws.root).resolve().parent / "harness_outbox.jsonl"


def _forward_outbox(session: "Any", box: "Any", ws: Workspace) -> int:
    """Send everything the client has left, as user messages. Returns how many.

    Sent as soon as it appears, mid-turn included. That is not a compromise: the
    CLI holds a message written during a tool call until the call finishes and
    then delivers it attached to the completed tool result, so the words arrive
    promptly AND without interrupting anything.
    """
    from .live import SessionGone

    sent = 0
    for record in box.drain():
        try:
            session.say(record["text"])
        except SessionGone as exc:
            print(f"    {ws.info.game_id}: harness message undelivered "
                  f"({record.get('kind', 'note')}): {exc}", flush=True)
            break
        sent += 1
        print(f"    {ws.info.game_id}: said {record.get('kind', 'note')} "
              f"to the solver as a user message", flush=True)
    return sent


def _wait_quiet(session: "Any", box: "Any", ws: Workspace,
                deadline: float | None) -> bool:
    """Wait until the solver is between turns. False if the deadline fired first.

    "Between turns" means the last turn produced its `result` and nothing the
    harness is holding is still waiting to be said. Both halves matter: a
    settled turn with an undelivered message in the outbox is not a place to
    decide anything, because what the solver is about to be told may change
    what it does.
    """
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            return False
        # **0.2 s, not 1 s.** The outbox is drained once per poll, so the poll
        # interval is the window in which a message can be written, the tool
        # call end, and the turn settle -- after which the message starts a new
        # turn instead of merging into the running one. Shortening the poll
        # narrows that window by 5. It cannot close it: a record written 1 ms
        # before a turn settles always arrives late. The cost is one `stat` per
        # 0.2 s, and it also gets every message to the solver sooner.
        if session.wait_settled(timeout=0.2):
            if _forward_outbox(session, box, ws):
                continue                    # it will speak again; wait for that
            return True
        if not session.alive:
            return True
        # Mid-turn. Forward anyway -- the CLI queues it behind the tool call.
        _forward_outbox(session, box, ws)


def _run_live(ws: Workspace, args: list[str], deadline: float | None,
              asks: _Asks) -> dict[str, Any]:
    """Run one game in ONE conversation the harness can speak into.

    The difference from the launch-per-ask path is not the questions -- they are
    the same questions, chosen by the same `_next_ask` -- it is that they arrive
    as real user messages in a conversation that never stopped, instead of as a
    relaunch. A relaunch injects `"Continue from where you left off."` whenever
    the previous turn ended badly, which is text the harness did not write, says
    an interruption happened, and cannot meet the standard that a resume look
    like none did.

    The run ends when this closes the child's stdin. A model that falls silent
    is asked whether it is finished and gets to answer, rather than having its
    silence read as an answer.
    """
    from . import outbox as outbox_mod
    from .live import LiveSession, SessionGone, live_args, opening_prompt

    prompt = opening_prompt(args)
    wire = live_args(args)
    box_path = _harness_outbox_path(ws)
    # A previous attempt's messages are not this one's. The reader detects a
    # replacement on its own, but leaving the file would also leave them in the
    # evidence of a run that never heard them.
    box_path.unlink(missing_ok=True)
    box = outbox_mod.Outbox(box_path)

    def drive(wrapped: list[str], env: dict[str, str],
              repair: "Callable[[], str]") -> dict[str, Any]:
        env[outbox_mod.ENV_VAR] = str(box_path)
        session = LiveSession(wrapped, cwd=ws.root, env=env,
                              stream_path=ws.root / "stream.jsonl").start()
        print(f"    {ws.info.game_id}: live session — the harness can speak to "
              f"this solver as a user", flush=True)
        timed_out = False
        # Bounded, so a permanently dead account cannot loop. Two is enough for
        # the case this exists for -- a concurrent launch rotating the token
        # once -- and a third failure means something a rewrite will not fix.
        repairs_left = 2
        try:
            session.say(prompt)
            while True:
                if not _wait_quiet(session, box, ws, deadline):
                    timed_out = True
                    break
                if not session.alive:
                    break
                # **A live session survives what would have ended a one-shot
                # run.** A dead credential produces `is_error` results with the
                # child still up, and a driver watching only the process would
                # call that healthy and sit there until the wall clock.
                failing = session.failing()
                if failing:
                    # **A dead credential need not end a live run.** The
                    # session survives the refusal, the CLI re-reads its
                    # credential between turns, and a repaired one works
                    # immediately. Two things it does NOT do on its own: it
                    # does not retry after silence, and the message the failed
                    # turn ate does not reliably survive. So the harness
                    # repairs, then says the same thing again.
                    if repairs_left > 0 and _is_auth_failure(failing):
                        why = repair()
                        if not why:
                            repairs_left -= 1
                            again = session.last_said or prompt
                            print(f"    {ws.info.game_id}: the solver's "
                                  f"credential was refused — repaired it and "
                                  f"said the same thing again", flush=True)
                            try:
                                session.say(again)
                            except SessionGone as exc:
                                print(f"    {ws.info.game_id}: could not "
                                      f"continue — {exc}", flush=True)
                                break
                            continue
                        print(f"    {ws.info.game_id}: {why}", flush=True)
                    print(f"    {ws.info.game_id}: the solver's last turn "
                          f"failed — {failing}", flush=True)
                    break
                outcome = collect_outcome(
                    ws, exit_code=0, timed_out=False,
                    extra=_live_extra(ws, asks))
                if _run_is_over(ws, outcome, deadline):
                    break
                ask = _next_ask(outcome, asks, timed_out=False)
                if ask is None:
                    break
                text, why = ask
                asks.made += 1
                print(f"    {ws.info.game_id}: {why} — asking as a user message "
                      f"in the same conversation ({asks.made})", flush=True)
                try:
                    session.say(text)
                except SessionGone as exc:
                    print(f"    {ws.info.game_id}: could not ask — {exc}", flush=True)
                    break
        except BaseException:
            session.kill()
            raise
        finally:
            # Anything the last action left has to reach the record even if it
            # can no longer reach the solver.
            for record in box.drain():
                print(f"    {ws.info.game_id}: harness message left unsent at "
                      f"shutdown: {record.get('kind', 'note')}", flush=True)

        code = session.close(timeout=TERM_GRACE_S) if not timed_out else session.kill()
        return collect_outcome(ws, exit_code=code, timed_out=timed_out,
                               extra=_live_extra(ws, asks))

    return _launch(ws, wire, deadline, run=drive)


def _live_extra(ws: Workspace, asks: _Asks) -> dict[str, Any]:
    """What a live run has to put in `result.json` that a one-shot one does not."""
    return {
        "nudges": asks.made,
        # **Zero, and the audit depends on it.** A post-run audit rejects a run
        # whose launch count is not `1 + nudges`, because an extra launch is an
        # unexplained relaunch. In a live session an ask costs no launch at all,
        # so reporting the asks there would make every live run read as
        # over-launched and fail the audit.
        "ask_launches": 0,
        "live_session": True,
        "session_id": ws.session_id,
    }


def _free_loopback_port() -> int:
    """A port the in-namespace relay can bind. Chosen on the host and handed in.

    Picking it inside the namespace would fail at bind time, after the guardrail
    exists and with the attempt already counted as started.
    """
    import socket as _socket

    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _launch(ws: Workspace, args: list[str], deadline: float | None,
            *, run: "Callable[..., Any] | None" = None) -> Any:
    """Run one `claude` invocation to completion. Returns (exit code, timed out).

    With ``run``, everything up to the spawn happens as usual -- both namespaces, the
    per-run config directory, the credential check -- and then the wrapped argv
    and its environment are handed to the callback instead of being run to
    completion here. The live session needs exactly that setup and exactly that
    teardown; only what happens in between differs, and one copy of the setup is
    the difference between two paths that agree and two that drift.
    """
    config = ws.config
    stream = ws.root / "stream.jsonl"
    # **Both namespaces, on by default.** The solver runs inside a mount namespace
    # where nothing outside its attempt directory exists (see
    # :mod:`ccarc.arc_3.confine`) and a network namespace whose only routes are
    # its own ARC shim and the model backend: an unconfined solver would share a
    # filesystem and a network with everything else on the box, siblings'
    # workspaces included. `CCARC3_CONFINE=0` and `CCARC3_NETGUARD=0` opt out,
    # and both refuse rather than degrade: if a layer is armed and the mechanism
    # is unavailable, launching open would be a defence reported as armed while
    # not running, so it raises instead and the driver marks the attempt
    # interrupted.
    from ..guardrail import confine, network
    from ..guardrail.egress_proxy import EgressProxy

    launch_env = dict(ws.env)
    bridges: list[tuple[int, Path]] = []
    holders: list[object] = []

    if os.environ.get("CCARC3_NETGUARD", "1") != "0":
        if not network.available():
            raise RuntimeError(
                "CCARC3_NETGUARD is on but an unprivileged network namespace is "
                "not available on this box. Refusing to launch with the network "
                "open while the flag claims otherwise."
            )
        # Two destinations, and a solver that loses either cannot work: the ARC
        # shim, or it cannot play; the model backend, or it cannot think.
        attempt = Path(ws.root).resolve().parent
        arc_root = launch_env.get("CCARC3_ARC_ROOT", "")
        if arc_root:
            arc_port = int(arc_root.rsplit(":", 1)[1].rstrip("/"))
            arc_sock = network.socket_path(attempt).with_name(".arc-shim.sock")
            holders.append(network.Bridge(arc_sock, arc_port).start())
            inner_arc = _free_loopback_port()
            bridges.append((inner_arc, arc_sock))
            launch_env["CCARC3_ARC_ROOT"] = f"http://127.0.0.1:{inner_arc}"
        eg_sock = network.socket_path(attempt).with_name(".egress.sock")
        egress = EgressProxy(eg_sock).start()
        holders.append(egress)
        inner_eg = _free_loopback_port()
        bridges.append((inner_eg, eg_sock))
        # **`NO_PROXY` for loopback, and no `HTTP_PROXY` at all.** The ARC
        # shim lives on the guardrail's loopback and is plain http; routed through
        # the egress proxy it is correctly refused as a host outside the
        # allowlist, leaving the solver able to reach its model and not its
        # game.
        launch_env["HTTPS_PROXY"] = f"http://127.0.0.1:{inner_eg}"
        launch_env.pop("HTTP_PROXY", None)
        launch_env["NO_PROXY"] = "127.0.0.1,localhost"
        launch_env["no_proxy"] = "127.0.0.1,localhost"

    # **Immediately before the spawn, because a broken account file is
    # silent.** See `confine.ensure_config_readable`: a truncated
    # `.claude.json` makes the CLI exit in ~2 s, which is indistinguishable
    # from a hard game. Placed here rather than at driver start so it also
    # covers every nudge relaunch, which re-enters this function long after the
    # first check would have passed.
    # **Its own config directory, inside its own attempt tree.** The CLI
    # rewrites `.claude.json` as it starts, so every launch is a writer:
    # concurrent solvers sharing one config directory can truncate it to zero
    # bytes and take every subsequent launch down with it. The only fix is
    # that no two launches share one. The attempt directory is already a kept
    # path, so this needs no extra keep-list entry and dies with the run.
    # **Check the credential BEFORE launching.** A dead one produces "solver
    # exited 1 after 0 actions" -- the same signature as a hard game, so the
    # attempt is spent and the real cause is invisible.
    _base = confine.config_dir(launch_env)
    # **The file check, then the server's opinion.** A well-formed credential
    # the server refuses burns attempts just as fast, and reading the file
    # cannot tell the two apart. The probe costs one haiku call and only runs
    # when the token has expired or is about to, which is exactly when the
    # refresh path is about to matter.
    _why = confine.credential_is_usable(_base) or (
        confine.credential_probe(_base) if _probe_enabled() else "")
    if _why:
        raise RuntimeError(
            f"{ws.info.game_id}: refusing to launch -- {_why}. Re-authenticate "
            f"that config directory (`CLAUDE_CONFIG_DIR={_base} claude /login`) "
            f"before running again.")
    _cfg = confine.per_run_config(Path(ws.root).resolve().parent / ".cli-config")
    launch_env["CLAUDE_CONFIG_DIR"] = str(_cfg)
    confine.ensure_config_readable(launch_env)

    if os.environ.get("CCARC3_CONFINE", "1") != "0":
        if not confine.available():
            raise RuntimeError(
                "CCARC3_CONFINE is on but user-namespace confinement is not "
                "available on this box (unshare probe failed). Refusing to "
                "launch unconfined while the flag claims otherwise."
            )
        args = confine.build_strict_command(ws.root, args, env=launch_env)
        print(f"    {ws.info.game_id}: confined — only its own attempt tree, "
              f"the staged package and the interpreter", flush=True)

    if bridges:
        # OUTSIDE the mount namespace: the relay must keep the outer mount view so it
        # can still see the bridge sockets the mount namespace hides from the solver.
        # **The interpreter the harness is ACTUALLY running under**, not one
        # derived from where this file happens to sit: a checkout without a
        # `.venv` beside the package root has no such path, the relay is
        # backgrounded with its output discarded, and the only symptom would be
        # the solver's first turn reporting `Connection refused`.
        # `sys.executable` is the one interpreter guaranteed to exist and to be
        # mounted in the guardrail.
        args = network.build_command(
            args, bridges=bridges, env=launch_env, python=sys.executable)
        print(f"    {ws.info.game_id}: {network.describe(bridges[0][0], bridges[0][1])}",
              flush=True)
    if run is not None:
        def _repair() -> str:
            """Put a live credential back under a solver that is still running.

            Returns "" on success, or why it cannot help. The source is the
            directory this run was seeded from: a concurrent launch that
            refreshed and harvested leaves the live token there, which is the
            case this recovers. It cannot recover a rotation by some OTHER
            login -- that token was never written anywhere this can read.
            """
            why = confine.credential_is_usable(_base)
            if why:
                return f"the source credential cannot help either: {why}"
            confine.per_run_config(_cfg, base=_base)
            return ""

        try:
            return run(args, launch_env, _repair)
        finally:
            _release_launch(ws, holders, _cfg, _base)

    remaining = None if deadline is None else max(1.0, deadline - time.monotonic())
    with stream.open("w", encoding="utf-8") as fh:
        # **Its own session, so the timeout can reach the whole tree.** Without
        # this the child sits in the driver's process group and a signal to
        # `proc` reaches only the `claude` process. The solver drives the game
        # from `python -c ...` grandchildren -- the workspace CLAUDE.md tells it
        # to -- and each of those POSTs actions of its own. Killing the parent
        # alone leaves them running: still spending the budget, still writing
        # the ledger, against a run already declared timed out.
        proc = subprocess.Popen(
            args,
            cwd=ws.root,
            env=launch_env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            code = proc.wait(timeout=remaining)
            timed_out = False
        except subprocess.TimeoutExpired:
            # **SIGTERM first, and keep whatever exit status comes back.** The
            # solver's stdout is a FILE, not a tty, so the runtime
            # block-buffers it: without a grace period to flush, the tail of
            # `stream.jsonl` -- the only source for `run_cost`'s turn and cost
            # figures -- is lost in userspace. The scored ledger is unaffected,
            # since `trace.jsonl` is written per action by the client, so this
            # buys accounting accuracy rather than score.
            #
            # The grace period also makes a clean exit possible, and that
            # status is worth recording: a solver interrupted one
            # write short of finishing can exit 0 during the grace, and
            # hard-coding -1 would file that as a kill. `collect_outcome` acts on
            # `exit_code` only when `timed_out` is false, so carrying the real
            # value through is informative and changes no decision.
            _signal_group(proc, signal.SIGTERM)
            try:
                code = proc.wait(timeout=TERM_GRACE_S)
            except subprocess.TimeoutExpired:
                _signal_group(proc, signal.SIGKILL)
                code = proc.wait()
            timed_out = True
        finally:
            _release_launch(ws, holders, _cfg, _base)

    return code, timed_out


def _release_launch(ws: Workspace, holders: list, cfg: "Path", base: "Path") -> None:
    """Give back what the launch took: the bridges, and the refreshed token."""
    # The bridges are the only way out of the namespace, so they must not
    # outlive the run that opened them. Their listening threads are
    # daemons and would not hold the driver open, but each leaves a
    # socket file in the attempt directory, and a stale one there is
    # state a resumed attempt could inherit. `stop()` unlinks it.
    #
    # Refusals are recorded first: an attempt on a denied host is the
    # thing an audit wants, and it is gone once the proxy is stopped.
    for holder in holders:
        refused = getattr(holder, "refused", None)
        if refused:
            print(f"    {ws.info.game_id}: egress refused {sorted(set(refused))}",
                  flush=True)
        holder.stop()

    # **Harvest the refreshed credential before the run's copy is gone.**
    # A long run refreshes, and refresh tokens rotate: without this the
    # source directory keeps a token the rotation just killed, and every
    # later launch from it dies at zero actions. See
    # `confine.harvest_credentials`.
    try:
        _note = confine.harvest_credentials(cfg, base)
        if _note.startswith("harvested"):
            print(f"    {ws.info.game_id}: {_note}", flush=True)
        elif "blank" in _note:
            print(f"    {ws.info.game_id}: WARNING {_note} -- the next "
                  f"launch from {base} will fail", flush=True)
    except Exception as exc:                    # noqa: BLE001
        # Bookkeeping must never turn a completed run into a crash.
        print(f"    {ws.info.game_id}: credential harvest failed: {exc}",
              flush=True)


def _record_resume_state(ws: Workspace) -> None:
    """Snapshot what a resume inherited, before the solver can change it.

    A resume can preserve the ledger but not the game -- trace indices
    continue where they left off while the server replays the early levels on
    a fresh scorecard, re-billing every one of them. Without a record of what
    the client restored at construction time, that mechanism cannot be
    reconstructed afterwards. This records it.
    """
    state = Path(ws.trace_path).with_suffix(".state.json")
    snapshot: dict[str, Any] = {
        "resumed": ws.resumed,
        "trace_lines": sum(1 for _ in ws.trace_path.open()) if ws.trace_path.exists() else 0,
        "state_file_present": state.exists(),
    }
    if state.exists():
        try:
            saved = json.loads(state.read_text(encoding="utf-8"))
            snapshot |= {
                "card_id": saved.get("card_id", ""),
                "level": saved.get("level"),
                "actions_used": saved.get("actions_used"),
                "cookies": len(saved.get("cookies") or []),
            }
        except (OSError, json.JSONDecodeError) as exc:
            snapshot["state_file_error"] = f"{type(exc).__name__}: {exc}"
    (ws.root / "resume_state.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )


def ledger_facts(trace_path: Path | str) -> dict[str, Any]:
    """Everything about a run that can be re-derived from its ledger.

    Split out from :func:`collect_outcome` so a *finished* run can be re-read
    later. `result.json` is a derived artefact, and a harness fix can make it
    wrong after the fact — a stored figure keeps the pre-fix reading forever.
    Deriving from the trace on demand means a fix reaches history too.
    """
    # `rows`, not `load`: nothing below asks about a board, and decoding them
    # is ~99% of what reading a finished trace costs.
    from .ledger import rows

    path = Path(trace_path)
    transitions = rows(path) if path.exists() else []
    states = [t.state for t in transitions]

    # A full reset restarts the game inside the same trace, so the ledger holds
    # more than one playthrough and the two honest numbers diverge: a game's
    # total spend includes every abandoned play, while the win cost only what
    # the last playthrough spent. Report both: the total is what the budget
    # paid, the last playthrough is what the result cost.
    #
    # The restarting action is counted *in* the final playthrough, because it
    # was billed. That is why this figure runs one higher than a per-level
    # table of the same playthrough -- the reset belongs to no level.
    last_restart = max(
        (i for i, t in enumerate(transitions) if t.full_reset),
        default=0,
    )
    final = transitions[last_restart:]
    resets = sum(1 for t in transitions if t.full_reset)

    # **Count what ARC counts.** The server does not bill the opening RESET of a
    # play; the ledger records it, because it is a real event that returned a
    # frame. So the raw row count runs exactly one above the scorecard, per
    # play, every time.
    #
    # Two numbers both called "actions" would force anyone comparing a result
    # to a scorecard to rediscover the difference, so `actions_used` is the
    # billed figure and `trace_rows` keeps the raw count.
    #
    # Derived from the rule, not from arithmetic that happens to match:
    # `len(transitions) - playthroughs` agrees on most runs and is still wrong,
    # because a mid-play RESET after a death *is* billed. Only a play's *first*
    # transition is exempt, and only when it is actually a RESET.
    starts = {0, *(i for i, t in enumerate(transitions) if t.full_reset)}
    unbilled = sum(1 for i in starts
                   if i < len(transitions) and transitions[i].action == "RESET")

    # **A play the server never billed is not a play.** `resets + 1` counts a
    # play the moment its opening RESET lands in the ledger, but ARC's card only
    # grows a row once the play does something, so the two counts disagree by
    # one whenever a run ends on a bare restart -- and `card_disagreement` then
    # discards a run whose card is in fact in perfect agreement.
    #
    # Counting only plays with a billed action cannot hide the failure that check
    # exists for. A card frozen mid-run is missing rows for plays that *did* act,
    # and those are still counted here, so the comparison still fires.
    ordered = sorted(starts)
    spans = [transitions[a:b] for a, b in zip(ordered, [*ordered[1:], len(transitions)])]
    played = sum(
        1 for span in spans
        if len(span) - (1 if span and span[0].action == "RESET" else 0) > 0
    )

    return {
        "levels_reached": max((t.level for t in transitions), default=0),
        "won": "WIN" in states,
        "actions_used": len(transitions) - unbilled,
        "trace_rows": len(transitions),
        "deaths": sum(
            1
            for prev, cur in zip(["NOT_PLAYED", *states], states)
            if cur == "GAME_OVER" and prev != "GAME_OVER"
        ),
        "wasted_actions": sum(t.wasted for t in transitions),
        "full_resets": resets,
        "playthroughs": max(1, played),
        "actions_final_playthrough": len(final) - (
            1 if final and final[0].action == "RESET" else 0),
        "levels_reached_final_playthrough": max((t.level for t in final), default=0),
    }


def run_cost(stream_path: Path | str) -> dict[str, Any]:
    """Turns, cost and wall time, from the stream's final ``result`` event.

    The ledger says what a run *did*; this says what it cost to do it, and the
    two can come apart in a way worth being able to see: runs with similar
    scores can differ widely in tool calls and in cost, and nothing in the
    ledger shows it.

    **Sums every attempt, not just the last.** ``run_game`` archives a previous
    stream to ``stream.N.jsonl`` when it resumes, while the trace — and so
    ``actions_used`` — carries across attempts. Reading only ``stream.jsonl``
    would report a resumed run's actions in full against the *final* attempt's
    turns and cost, understating the bill by whatever the earlier attempts
    spent.

    Returns an empty dict when no stream has a result event — a killed run has
    none, and that is not an error worth raising over.
    """
    path = Path(stream_path)
    attempts = sorted(path.parent.glob("stream.*.jsonl")) + [path]
    turns = cost = duration = 0
    found = False
    for attempt in attempts:
        if not attempt.exists():
            continue
        # **One stream can hold several results, and the fields disagree about
        # what they count.** A launch-per-ask run writes one `result` per file;
        # a run the harness speaks into writes one per turn, into one file.
        # Across consecutive results `total_cost_usd` and `modelUsage` are
        # cumulative, while `num_turns`, `duration_ms` and `usage` each count
        # only their own result.
        #
        # So the cost is the LAST result's and the turns and the wall time are
        # the SUM of them. Taking the last of all three would understate a live
        # run's turns and time; summing all three would bill its cost several
        # times over.
        results: list[dict[str, Any]] = []
        for line in attempt.open(encoding="utf-8", errors="ignore"):
            if '"type":"result"' not in line and '"type": "result"' not in line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                # A line still being written, or the solver's stderr. Neither is
                # an error worth raising over in a function whose whole contract
                # is "never raise over a killed run".
                continue
            # `run_game` merges the solver's stderr into this file, so a line can
            # be valid JSON without being an object. `.get` on a str is an
            # AttributeError out of the same contract.
            if isinstance(record, dict) and record.get("type") == "result":
                results.append(record)
        if not results:
            continue
        found = True
        turns += sum(r.get("num_turns") or 0 for r in results)
        cost += results[-1].get("total_cost_usd") or 0.0
        duration += sum(r.get("duration_ms") or 0 for r in results)
    if not found:
        return {}
    return {
        "turns": turns,
        "cost_usd": cost,
        "duration_s": round(duration / 1000) if duration else None,
        "attempts": sum(1 for a in attempts if a.exists()),
    }


def _killing_signal(exit_code: int) -> int:
    """The signal that killed the solver, or 0 if it exited on its own.

    Two encodings, because two things report it. ``Popen.wait`` returns a
    *negative* number when the child dies on a signal; but the solver is a node
    CLI that installs its own handler and exits normally with ``128 + signum``,
    so a SIGTERM arrives as **143**, an ordinary-looking exit code. The
    second form is the one that actually arrives, and only the first is
    documented.
    """
    if exit_code < 0:
        return -exit_code
    if 128 < exit_code < 193:
        return exit_code - 128
    return 0


def _action_budget(ws: Workspace) -> int:
    """The run's action cap. Parent-side; returns 0 when nothing knows it."""
    # **Kept out of the docstring, because this module is on the solver's own
    # PYTHONPATH and a docstring is one attribute access from the namespace the
    # workspace hands it.**
    #
    # `meta.json` is read first, so the figure reflects the multiple the run
    # actually got. It cannot be the only source: the baseline strip removes
    # `action_budget` from it on purpose, and a guard that reads a field a
    # different subsystem removed is a guard that passes by not running --
    # `collect_outcome`'s "a timeout with allowance unspent is not a result"
    # arm is exactly that guard.
    #
    # So fall back to the parent's own arithmetic: `ws.info` holds the real
    # figures in memory, because the strip rewrites files in the workspace and
    # not the `GameInfo` the driver holds. Returning 0 when neither source has
    # anything leaves the decision to each caller's own `budget` guard rather
    # than guessing.
    try:
        meta = json.loads((ws.root / "meta.json").read_text(encoding="utf-8"))
        budget = meta.get("action_budget")
        if isinstance(budget, int) and budget > 0:
            return budget
    except (OSError, ValueError):
        pass
    if ws.info and ws.info.baseline_total:
        return ws.info.suggested_budget(ws.config.budget_multiple)
    return 0


def _harness_spent(outcome: dict[str, Any]) -> int:
    """What the run spent against its *ceiling*, which is not what ARC bills.

    **Two counters both called "actions", and the ceiling reads the other one.**
    `ledger_facts` publishes both on purpose: `actions_used` is the billed figure
    (`len(transitions) - unbilled`, because ARC does not bill a play's opening
    RESET) and `trace_rows` is the raw count. `ArcClient` refuses at
    `self.actions_used >= self.max_actions`, and *its* counter is every act it
    made -- the raw count. So the cap is spent when `trace_rows` reaches it, and
    `actions_used` is short by one per playthrough, always.

    Every guard that asks "is there budget left" must therefore read
    `trace_rows`: `unbilled >= 1` on every run, so the billed figure can never
    reach the cap, and a solver sitting on a spent ceiling would otherwise be
    classified as having quit with actions in hand, and nudged into an
    environment that can only answer 403.

    Not `max(billed, rows)`: `rows >= billed` by construction, so the max is a
    no-op that reads as though it were doing something. Name the counter the
    ceiling actually uses. Results written before `trace_rows` existed fall back
    to the billed figure, which is what they were judged by at the time.
    """
    rows = outcome.get("trace_rows")
    if rows is None:
        return int(outcome.get("actions_used", 0) or 0)
    return int(rows or 0)


def _remaining_quota(outcome: dict[str, Any], budget: int) -> int:
    """Actions this attempt could still spend. Zero once the ceiling is reached.

    ``budget <= 0`` means the cap is unknown -- `_action_budget` returns 0 when
    neither `meta.json` nor `ws.info` has anything -- and an unknown cap must not
    read as an exhausted one, so it reports 0 remaining only through the caller's
    own `if budget` guard. Callers check `budget` first; this function is about
    the arithmetic, not about whether to trust it.
    """
    if budget <= 0:
        return 0
    return max(0, budget - _harness_spent(outcome))


def _finish_declared(ws: Workspace) -> int:
    """How many times this run's solver called ``finish()``.

    Read from the decision sidecar the client writes
    (:attr:`ArcClient.decision_path`), never from anything the solver merely
    said: the stream is the solver's testimony, the sidecar is the client's
    record of an actual call. Zero when the file is absent, unreadable, or
    describes a different game -- an unreadable declaration must degrade to
    "nothing declared", which only ever costs one neutral question.
    """
    path = Path(ws.trace_path).with_suffix(".decision.json")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if doc.get("game_id") not in (None, ws.info.game_id):
        return 0
    try:
        return max(0, int(doc.get("declarations", 0)))
    except (TypeError, ValueError):
        return 0


def _prior_give_ups(ws: Workspace) -> int:
    """How many earlier attempts at this game already quit with budget in hand.

    **Not ``outcome["attempts"]``, which looks like this number and is not.**
    That counts ``stream.*.jsonl`` files inside ONE workspace -- solver
    relaunches into the same directory. The driver gives every retry a fresh
    ``attempt_N/`` workspace, so it reads 1 on every attempt and a bound
    written against it never bites: a proxy that resembles the quantity closely
    enough to pass reading.

    Counts the thing itself: sibling attempts whose own ``result.json`` records
    a give-up. Returns 0 for any layout without them, which leaves the caller
    unbounded rather than blocking -- and outside the driver nothing retries,
    so there is nothing to bound.
    """
    root = Path(ws.root)
    # <out_dir>/<game>/attempt_N/<game>  ->  the attempts sit two levels up.
    for base in (root.parent.parent, root.parent):
        try:
            siblings = sorted(base.glob("attempt_*/*/result.json"))
        except OSError:
            continue
        if not siblings:
            continue
        n = 0
        for r in siblings:
            if r.parent == root:
                continue                      # this run has not been written yet
            try:
                if "gave up" in (json.loads(r.read_text(encoding="utf-8")).get("error") or ""):
                    n += 1
            except (OSError, ValueError):
                continue
        return n
    return 0


# Names only, never values -- a value in `result.json` is a leak with a longer
# half-life than the run. `False` is the answer the strip path is supposed to
# give for the first two.
# `ARCPRIZE_API_KEY` is here because both spellings are guarded elsewhere and a
# record that watched only one would be a check with a hole in exactly the
# shape of its subject. `CCARC3_PROXY_URL` is deliberately absent: it
# is read on the runner and never reaches the child, so recording it would add a
# permanent `False` that reads like a verified absence.
RECORDED_ENV = ("ARC_API_KEY", "ARCPRIZE_API_KEY", "CCARC3_MAX_ACTIONS",
                "CCARC3_HIDE_BASELINES", "CCARC3_ARC_ROOT",
                RETIRED_MODE_VAR)


def _env_facts(ws: Workspace) -> dict[str, Any]:
    """Which of the env vars that matter reached the child, as presence flags.

    The child's environment cannot be recovered from a finished run, so the
    only moment it is knowable is when the workspace is built. Recording it
    here makes "`ARC_API_KEY` and the action cap never reached the solver" a
    checkable fact about each run rather than a property of the code.
    """
    return {"child_env": {k: k in ws.env for k in RECORDED_ENV}}


def _card_facts(ws: Workspace) -> dict[str, Any]:
    """Which scorecard this run scored on.

    Recording the id per run makes "every game is on one card" a fact a single
    pass over the outcomes can verify, rather than an assumption a submission
    rests on. It is the kind of statement that stays true until one game
    quietly is not -- a resume onto its old card, a driver restart that minted
    a new one -- and `foreign_card` marks a run that is not on the intended
    card.
    """
    try:
        saved = json.loads(
            Path(ws.trace_path).with_suffix(".state.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}
    facts = {"card_id": saved.get("card_id", ""),
             "card_plays_at_open": int(saved.get("card_plays_at_open", -1))}
    if saved.get("foreign_card"):
        # This game is NOT on the shared card, whatever the driver intended.
        facts["foreign_card"] = saved["foreign_card"]
    return facts


GIVE_UP_FRACTION_ENV = "CCARC3_GIVE_UP_FRACTION"


def _give_up_fraction() -> float:
    """How much of the allowance may be spent and the stop still count as quitting.

    **The default is 1.0: any allowance left over at all.** A solver that stops
    while it can still act has stopped by choice, and the harness treats that as
    something to continue rather than as a measurement of the environment.

    A fraction of `0.5` would draw the line in the middle on the reasoning that
    a solver past halfway has at least contested the game, and the cost of that
    reading is that a run stopping with a large minority of its allowance
    intact is banked as a real loss, with the nudge -- which exists precisely
    to say "you are not finished, carry on" -- never firing on it. Under a
    fraction of 1.0 the same run is continued instead.

    **A run that exhausts the allowance is not exempted because it counts as a
    result. It is exempted because there is nothing left to continue on.** Any
    run that does not score `E = 1` is a loss — grinding to the ceiling and
    losing is a loss too, not "a measurement of the game". The only question a
    nudge answers is whether the solver still has actions to spend; at
    `used >= budget` it does not, so there is nothing to say to it.

    The retry cost is bounded by `GIVE_UP_ATTEMPTS` regardless of this value, so
    loosening the fraction widens *which* runs get another go, never how many
    goes any one game gets.

    Set the environment variable to a value in (0, 1] to tighten it.
    """
    raw = os.environ.get(GIVE_UP_FRACTION_ENV, "")
    if not raw:
        return 1.0
    try:
        value = float(raw)
    except ValueError:
        print(f"{GIVE_UP_FRACTION_ENV}={raw!r} is not a number; using 1.0", flush=True)
        return 1.0
    if not 0.0 < value <= 1.0:
        print(f"{GIVE_UP_FRACTION_ENV}={raw!r} is outside (0, 1]; using 1.0", flush=True)
        return 1.0
    return value


GIVE_UP_ATTEMPTS = 3
"""How many times a game that quit early is re-run before its loss is accepted.

Not unbounded, unlike an interruption. A run cut short by a signal or a crash
says nothing about the environment, so retrying it is free information. A run
that *stopped on its own* might be reporting something real about the game,
and a run costs real money, so the difference between 3 tries and 12 is
several runs' spend to hear the same answer.
"""


#: Keys `collect_outcome` computes that must not reach the workspace copy.
#: `baseline_total` IS the withheld quantity; `levels_total` is `info.levels`,
#: i.e. `len(baseline_actions)`, so it is the same array's length.
WITHHELD_FROM_WORKSPACE = ("baseline_total", "levels_total")


def _write_outcome(ws: Workspace, outcome: dict[str, Any]) -> None:
    """Write the run's record: a full copy outside the guardrail, a public one inside."""
    # **A comment, not a docstring.** This module is staged onto the solver's
    # own PYTHONPATH, so a docstring explaining what is kept back from it and
    # where the unredacted copy lives is a map to the thing being hidden.
    #
    # **`result.json` goes in the directory the solver works in, and the solver
    # is not always gone when it is written.** `build_workspace` unlinks it
    # before a launch, which covers a run that ends once. It does not cover the
    # two paths that keep a solver alive across a `collect_outcome`: the
    # give-up nudge re-enters the same conversation in the same working
    # directory, and the live loop calls this inside its own `while` with the
    # session still up. A caller that turns nudging on holds that window open.
    #
    # The RETURN value is unredacted: every parent-side reader takes the dict
    # rather than the file.
    scratch = os.environ.get("CCARC3_SCRATCH")
    if scratch:
        try:
            full = pathlib.Path(scratch) / f"result.full.{ws.info.game_id}.json"
            full.write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
            full.chmod(0o600)
        except OSError:
            # Bookkeeping must never cost the result -- the same rule the rules
            # book parse above follows. The redaction below still happens.
            pass
    public = {k: v for k, v in outcome.items() if k not in WITHHELD_FROM_WORKSPACE}
    # tmp + os.replace: a kill mid-teardown must not leave a truncated
    # result.json -- an unreadable result reads as "no result" and re-runs the
    # game.
    _tmp = ws.root / "result.json.tmp"
    _tmp.write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    os.replace(_tmp, ws.root / "result.json")


def collect_outcome(ws: Workspace, *, exit_code: int, timed_out: bool,
                    extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read the run's result off disk — never from what the solver claims.

    ``extra`` is merged in **before** the file is written, and that ordering is
    the whole reason it exists. This function is the single writer of
    ``result.json`` (it writes, then returns). Attaching ``nudges`` and
    ``session_id`` to the returned dict afterwards would leave both in memory
    for the rest of the process and **never on disk**.

    That is not cosmetic. A post-run audit reads the FILE, and its one-shot
    rule compares launches against nudges: a nudged run legitimately has
    ``attempts = 1 + nudges``. With ``nudges`` missing it reads 0, so a nudged
    run looks like an unexplained relaunch and is discarded -- a winning run
    can be rejected on the strength of one nudge the file could not testify to.

    A value the code computes, keeps, and then drops on the floor before the
    only consumer can read it is a record that says one thing while the check
    reads a proxy for it."""
    outcome = {
        "game_id": ws.info.game_id,
        **_card_facts(ws),
        **_env_facts(ws),
        "levels_total": ws.info.levels,
        "baseline_total": ws.info.baseline_total,
        **ledger_facts(ws.trace_path),
        **run_cost(ws.root / "stream.jsonl"),
        "exit_code": exit_code,
        "timed_out": timed_out,
        # The solver's explicit decision, if it made one. `run_game` honors it
        # (a declared decision is never overridden) and the give-up arm below
        # respects a reaffirmation; recorded in result.json so the evidence can
        # testify to which exits were decisions and which were silence.
        "finish_declared": _finish_declared(ws),
    }

    # **A solver that was killed did not lose. It was interrupted.**
    #
    # A container restart can SIGTERM the solver mid-game while the parent
    # survives, collects the trace as it stood, and writes a `result.json` with
    # `won: false`, `timed_out: false` and no error. Nothing in it says the run
    # was cut short, and every consumer reads it as an environment that won.
    # Worse, the arm's resume rule is `if prior and not prior.get("error"): skip`,
    # so the false loss would be permanent.
    #
    # Marking it as an error is what makes it retryable. A *timeout* is
    # excluded deliberately: a real outcome under a rule the caller chose.
    # **A win is exempt here too.** A solver can be SIGTERM'd at any point, so
    # without the exemption a finished win interrupted during a victory lap is
    # discarded and re-played for no information.
    #
    # **The exemption is "won *and* the last play went the distance", not bare
    # "won".** A solver SIGTERM'd early in its replay has already won, but its
    # final playthrough is a stub, and banking that pairs a tiny action count
    # with a level count from a different play -- the flattering half of each.
    # Guard 2 carries that exposure already; this one would carry it far more
    # often.
    #
    # The record of the interruption is kept either way. Folding it into the
    # same branch as the error would delete `killed_by_signal` from exactly the
    # runs this is about, which is how the naive form of this rule loses the
    # evidence it exists to preserve.
    # **A spent ceiling is published before any arm reads it.**
    #
    # Hitting the action cap is not quitting: there is nothing left to spend on
    # this attempt, so "stopped with actions in hand" is false and a nudge asks
    # the solver to act in an environment that can only answer 403 -- a run at
    # its ceiling would be nudged into the wall and then discarded as having
    # quit.
    #
    # (No figures. An action cap written here inverts to a baseline total in
    # one division by the budget multiple.)
    #
    # Set here rather than inside an arm because two arms need it and one of
    # them (`timed_out`) runs first. A flag, not a recomputation in each place:
    # the give-up arm and the timeout arm disagreeing about whether the budget
    # is gone is exactly the failure this flag prevents.
    _budget = _action_budget(ws)
    if _budget and _remaining_quota(outcome, _budget) <= 0:
        outcome["budget_exhausted"] = True

    sig = _killing_signal(exit_code)
    if sig and not timed_out:
        outcome["killed_by_signal"] = sig
    finished_cleanly = (
        outcome.get("won")
        and outcome.get("levels_reached_final_playthrough") == outcome.get("levels_total")
    )
    if sig and not timed_out and not finished_cleanly:
        outcome["error"] = (
            f"solver killed by signal {sig} after {outcome.get('actions_used', 0)} "
            f"actions — interrupted, not a result; re-run this game"
        )
    # **A crash is an interruption too, not only a signal.** The guard above
    # catches SIGTERM and friends. It does not catch a plain non-zero exit, and
    # that is the same false loss by another route: a solver that dies on
    # `Connection refused` mid-game -- a container restart moving the agent
    # proxy is enough -- is banked at wherever it stood, with no error field,
    # permanently un-retryable under `if prior and not prior.get("error"): skip`.
    #
    # A win is exempt: a solver that exits non-zero partway through a replay of
    # a game it has already won produced a real result. A timeout stays exempt
    # for the reason given above.
    elif exit_code and not timed_out and not outcome.get("won"):
        outcome["error"] = (
            f"solver exited {exit_code} after {outcome.get('actions_used', 0)} "
            f"actions without winning — crashed, not a result; re-run this game"
        )
    # **A wall-clock timeout is only a result if the budget ran out, or the
    # caller declared the clock is the game's allowance.**
    #
    # The two guards above exempt timeouts, on the reasoning that a timeout is
    # "a real outcome under a rule the caller chose". That holds for the action
    # budget the score runs on. It does not hold for an *infrastructure* clock
    # -- an hour, say, sized to a container's lifetime.
    #
    # An infrastructure cap can fire mid-climb with almost all of the action
    # allowance left, `timed_out: true` and no error. Under
    # `if prior and not prior.get("error"): skip`, that is the same permanent
    # false loss as the signal and crash cases above.
    #
    # A caller's own window is a different object. With
    # `clock_is_infrastructure=False` the window was picked so a run can
    # finish, and a timeout *inside it* with play on the board is the game not
    # fitting. Marking that an interruption requeues the same clip, and a
    # driver that never banks an `error` then leaves the game unbanked -- a
    # coverage hole. Zero actions is still an interruption on any clock:
    # nothing started.
    #
    # The meaning is a flag, not a duration. A threshold on the seconds would
    # silently bank a false loss for every caller that keeps the default
    # duration (`Ccarc3Config`, `cli.py`). Each caller states what its clock
    # *is*.
    elif timed_out and not outcome.get("won"):
        budget = _action_budget(ws)
        used = outcome.get("actions_used", 0) or 0
        # **`exhausted` reads the flag, not the billed figure.** See
        # `_harness_spent`: the ceiling counts raw ledger rows, billed is short
        # by one per playthrough, and `billed >= budget` therefore cannot hold.
        # A clock firing on a run that already sits at its spent cap did not
        # interrupt anything; the run had stopped being able to act.
        exhausted = bool(outcome.get("budget_exhausted"))
        infra = bool(getattr(ws.config, "clock_is_infrastructure", True))
        clip = infra or used == 0
        if not exhausted and clip and budget and used < budget:
            # Same reason as the give-up arm below: `{budget}` and the
            # percentage each divide by `budget_multiple` back to the withheld
            # total, and this lands in the workspace `result.json`.
            outcome["error"] = (
                f"solver hit the wall clock after {used} actions — interrupted "
                f"by the clock, not a result; re-run this game"
            )
    # **A solver that stopped while it could still act did not lose either.
    # It gave up.**
    #
    # The threshold is the whole of the allowance (`_give_up_fraction`, default
    # 1.0), not the midpoint. Half draws the line on the reasoning that a
    # solver past the midpoint has at least contested the game; the cost of
    # that reading is that a run stopping with a large minority of its
    # allowance intact is banked as a real loss and never nudged.
    #
    # The shape this catches is the largest recoverable loss there is:
    # `exit_code 0`, `error: null`, `timed_out: false`, stopped levels short of
    # the end with most of the allowance unspent and the ceiling nowhere in
    # sight. Nothing stopped it; it stopped. Banked with no error it keeps that
    # score forever, and `if prior and not prior.get("error"): skip` makes it
    # permanent.
    #
    # (Deliberately no numbers: this file is on the solver's PYTHONPATH and
    # reachable by `inspect.getsource`, and an action count written beside a
    # baseline total is a human median in two subtractions.)
    #
    # Four exits look identical in `result.json` and none of them is a result:
    # interrupted, crashed, clocked out, and quit. All four are the harness
    # failing to record that the environment was never actually contested.
    #
    # **Bounded, unlike the other three.** They mark a run retryable without
    # limit because an interruption says nothing about the game. Quitting might:
    # a solver that stops short of the end several times running may be
    # reporting that the game is hard, and a dozen passes at a full run's cost
    # is a lot to spend to learn it. After `GIVE_UP_ATTEMPTS` tries the result
    # stands as real.
    #
    # **A spent ceiling is excluded.** Quitting means stopping while able to
    # act. A solver at its cap is not able to act: the client refuses and the
    # server answers 403, so the nudge this arm exists to trigger can buy
    # nothing. The attempt is over; a later occupancy is a new `attempt_N` with
    # a fresh cap, not a resumption of a run with nothing left.
    elif not exit_code and not timed_out and not outcome.get("won"):
        budget = _action_budget(ws)
        used = outcome.get("actions_used", 0) or 0
        # **A reaffirmed declaration stands, even here.** `finish()` called
        # twice means the solver quit, was told the fact (it could still act),
        # and declared again anyway. That is a decision made with the
        # information in hand, and marking it a give-up would put the harness
        # in the business of overriding decisions -- the result banks as a
        # real loss instead. One declaration is not exempt: it earns the single
        # statement of fact `run_game` sends before this arm's verdict is final.
        if (budget and not outcome.get("budget_exhausted")
                and outcome.get("finish_declared", 0) < 2
                and used < budget * _give_up_fraction()
                and _prior_give_ups(ws) < GIVE_UP_ATTEMPTS):
            # **A flag, not a substring.** The nudge loop in `run_game` has to
            # recognise this exact condition, and the obvious way -- grepping the
            # message for "gave up" -- makes the wording load-bearing, so
            # rephrasing an error string would silently switch the loop off.
            outcome["gave_up"] = True
            # **No cap and no level total in the text.** This string is written
            # into `result.json` in the solver's own working directory at the
            # moment the give-up arm guarantees a nudge back into that
            # directory. A cap or a `used/budget` ratio divides by
            # `budget_multiple` back to the withheld baseline total, and
            # `levels_total` is that array's length. The redaction in
            # `_write_outcome` cannot reach prose inside a value.
            outcome["error"] = (
                f"solver stopped at {outcome.get('levels_reached')} levels "
                f"after {used} actions with no error and no "
                f"timeout — it gave up while it could still act, which is not "
                f"a result; re-run this game"
            )
    # **Bookkeeping must never cost the result.** `rules.json` is written by
    # `RuleBook.save`, which uses `write_text` and so truncates before it
    # writes, and the solver has Write on its own workspace, so a zero-byte or
    # invalid file is reachable -- and an unguarded parse here, three lines
    # before result.json is written, would take a won run down with it.
    #
    # Every other reader in this file already guards its parse. The two fields
    # derived here are counts for the log; losing them is a line of missing
    # bookkeeping, not a lost game.
    book = {}
    if ws.rules_path.exists():
        try:
            book = json.loads(ws.rules_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            outcome["rules_error"] = f"{type(exc).__name__}: {exc}"
            book = {}
    # **Absent cost is not zero cost.** `run_cost` returns {} when the stream
    # carries no final `result` event, and a wall-clock timeout guarantees that:
    # the CLI is signalled, so it never writes one. The outcome then has no
    # `cost_usd`, `turns` or `duration_s` at all, and any total that sums across
    # runs silently treats the missing one as free -- a whole game costing $0
    # in the arm's own accounting. Recording the absence is what lets a total
    # say "over N-1 of N runs" instead of quietly meaning it.
    if "cost_usd" not in outcome:
        outcome["cost_unavailable"] = True

    outcome["mechanics_recorded"] = len(book.get("verified", []))
    outcome["refutations_recorded"] = len(book.get("refuted", []))

    outcome.update(extra or {})

    snapshot_scorecard(ws)
    _write_outcome(ws, outcome)
    return outcome


def snapshot_scorecard(ws: Workspace) -> dict[str, Any]:
    """Save the server's own scorecard next to the trace. Never raises.

    **This has to live here, in the parent.**
    ``ArcClient.close()`` snapshots too, but nothing calls it: ``close`` runs
    only from ``__exit__``, the generated ``session.py`` calls ``client.open()``
    and never closes, and a solver does ``from session import client``. So the
    client-side snapshot would never fire on a real run: a signal in a function
    nobody calls is not a signal. The harness runs after the solver exits and
    always runs.

    **Deliberately does not close the card.** ``close()`` deletes the state file,
    which is what a resume reads to continue the same game; a run that timed out
    and will be resumed would be broken by it.

    **The card is reaped server-side once the game sits idle**, on a window of
    roughly a quarter of an hour. Under the window the card is still live and
    the game continues on it. Over it the scorecard 404s and every
    ``/api/cmd`` answers ``game not found``, so the solver must open a fresh
    card and start again. It is not a documented ARC policy, so treat the
    window as a claim, not a guarantee, and resume promptly regardless.

    Losing the window is worse than losing the progress, because the ledger
    does not reset with the game. ``actions_used`` carries across attempts by
    design, so the re-played actions land on the denominator the score divides by
    while the numerator starts again -- a run that gets pushed back several
    levels is scored on everything it spent getting there the first time as
    well.

    So a resume is only cheap if it is *prompt*. Anything that relaunches an
    interrupted run should do so immediately and hold its claim while it does,
    rather than releasing the game to whatever picks it up next.

    Why bother: ``actions_by_level`` is the server's per-level action count,
    which is exactly what the score measures and which this harness otherwise
    re-derives from the trace, and it holds one row per play -- the only
    evidence that can settle which play is scored.
    """
    # The resume-window evidence is kept out of this module: it is
    # harness-side content with no reason to be legible to a solver, and this
    # module is on a solver's import path. Named here in a comment and not in
    # the docstring above it: comments are stripped from the copy a solver
    # imports (`ccarc.staging`) and docstrings are not, so this is the one
    # place the pointer is safe.
    state = ws.root / "trace.state.json"
    if not state.exists():
        return {}
    try:
        card_id = json.loads(state.read_text(encoding="utf-8")).get("card_id")
    except (json.JSONDecodeError, OSError):
        return {}
    if not card_id or card_id.startswith("card-"):   # a test stub, not a real card
        return {}
    try:
        from .client import ROOT_URL, _get, competition_mode

        if competition_mode():
            # A competition card cannot be corroborated AT ALL: the pinned
            # session gets `403 "cannot get scorecard that is in competition
            # mode"`, and this unpinned GET gets the routing 404, so the
            # except-branch marker below cannot fire. Write the marker without
            # spending the doomed request.
            (ws.root / "scorecard.json").write_text(
                json.dumps({"competition_mode_unreadable": True,
                            "error": "competition mode forbids scorecard reads; "
                                     "not attempted"}, indent=2) + "\n",
                encoding="utf-8",
            )
            return {}

        card = _get(
            f"{ROOT_URL}/api/scorecard/{card_id}/{ws.info.game_id}",
            os.environ.get("ARC_API_KEY", ""),
        )
        (ws.root / "scorecard.json").write_text(
            json.dumps(card, indent=2) + "\n", encoding="utf-8"
        )
        return card
    except Exception as exc:  # noqa: BLE001 -- bookkeeping must never replace a result
        if "competition mode" in str(exc):
            # The card is alive and REFUSES reads -- the mode's rule, not death.
            # Record that fact where the corroboration step looks, or the won
            # run is discarded as "uncorroborated (no scorecard)" and the
            # driver relaunches a finished game.
            try:
                (ws.root / "scorecard.json").write_text(
                    json.dumps({"competition_mode_unreadable": True,
                                "error": str(exc)}, indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError:
                pass
        return {}
