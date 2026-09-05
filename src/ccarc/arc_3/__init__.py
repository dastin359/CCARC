"""CCARC3 — harness primitives for a Claude-Code-as-agent ARC-AGI-3 solver.

The split this package exists to enforce: **the harness ships the converter, the
solver writes the analysis.** Every run would otherwise re-derive the same
loader differently, destroying cross-run comparability, and conversion is where
a silent bug (dropping the intermediate frames of a multi-frame action,
mis-inferring the render block size) poisons every downstream conclusion
invisibly.

Nothing here imports ``arc_agi_3``, so all of it is usable and testable without
the SDK or an API key. The seam is a plain dict of the shape
``FrameData.model_dump()`` produces.
"""

from .grids import (
    DEFAULT_BACKGROUND,
    PALETTE,
    PALETTE_SIZE,
    as_grid,
    block_size,
    cell_boundaries,
    collapse,
    counts,
    logical,
    png,
    render,
)
from .client import (
    ROOT_URL,
    ActionRefused,
    ArcClient,
    GameInfo,
    list_games,
)
from .gate import GateRefusal, LevelGate
from .ledger import (
    ACTION_NAMES,
    TraceWriter,
    Transition,
    action_name,
    infer_levels,
    load,
)
# **Nothing that could help a solver score itself is offered here.**
# `ccarc.arc_3` is the namespace a solver holds as `arc`, so every name here is
# advertised by `dir(arc)`. The withholding covers the existence of the
# quantity, not only its value: a name that announces a per-level reference
# count exists, and that the harness knows it, is itself a disclosure.
from .session import Ccarc3Config, build_workspace, ledger_facts, run_game
# **Analysis is not offered.** Encoding a board, rendering one, and reading the
# ledger are plumbing and stay. Finding connected components, diffing two
# boards, checking a rule across the history, scoring a forward model, counting
# which actions did anything, searching for a route -- each of those IS the
# reasoning the solver is here to do, and shipping it pre-built would do that
# reasoning for it. `RuleBook` is the exception and is not analysis: it is the
# on-disk store the level gate writes through.
from .rules import RuleBook

__all__ = [
    # grids
    "PALETTE_SIZE",
    "DEFAULT_BACKGROUND",
    "as_grid",
    "render",
    "block_size",
    "logical",
    "collapse",
    "cell_boundaries",
    "png",
    "PALETTE",
    "counts",
    # client
    "ArcClient",
    "ActionRefused",
    # gate
    "LevelGate",
    "GateRefusal",
    # ledger
    "ACTION_NAMES",
    "Transition",
    "load",
    "action_name",
    "infer_levels",
    # rules
    "RuleBook",
]


# **`dir(arc)` is the solver's index, and Python would otherwise fill it in.**
# Importing a submodule binds it on the parent package, so `arc.confine` and
# `arc.session` appear in the namespace a solver reads even though nothing here
# re-exports them -- one harness-side import of a name from either is enough.
# Their module docstrings then describe the guardrail and, in `confine`'s case,
# the withholding itself.
#
# A module-level `__dir__` hides them from the index WITHOUT unbinding them,
# so harness code keeps importing them by module name. Deleting the attributes
# would close the same hole and break that.
#
# Not a wall: a solver that imports a module by name still gets it, exactly as
# it can still read the staged source. What this removes is
# discovery -- the guide points the solver at `dir(arc)`, and that list should
# not be an inventory of what is being kept from it.
_HIDDEN_FROM_INDEX = frozenset({
    "confine", "session", "client", "gate", "_numpy",

    # **The harness's business:** a solver should not see what is clearly the
    # harness's, and usage is not the test -- something unused but useful for
    # solving stays.
    #
    #   build_workspace / run_game / Ccarc3Config / ledger_facts
    #       the harness IS these. A solver does not build a workspace or start
    #       a game; it is handed one already running.
    #   list_games   fetches `/api/games`, the baseline-bearing endpoint the
    #                shim allowlist exists to refuse.
    #   ROOT_URL     the real ARC address. The solver reaches the game through
    #                the shim and has no business addressing the API directly.
    #   GameInfo     the type whose field is `baseline_actions`.
    #   TraceWriter  writing the ledger is the client's job; the solver reads.
    #
    # Everything else stays -- grids, rules, the gate, `Transition`,
    # `load`, `action_name`, the refusal exceptions -- because each of them
    # helps solve the game, whether or not any given run reaches for it.
    "build_workspace", "run_game", "Ccarc3Config", "ledger_facts",
    "list_games", "ROOT_URL", "GameInfo", "TraceWriter",
})


def __dir__():
    # `_HIDDEN_FROM_INDEX` must filter itself. A private name in the index is a
    # disclosure on its own: it says a list of withheld names exists here.
    return sorted(
        n for n in globals()
        if n not in _HIDDEN_FROM_INDEX and not (n.startswith("_") and not n.startswith("__"))
    )
