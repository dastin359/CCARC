"""CCARC — driving coding agents at ARC-AGI.

Two drivers over one harness. :mod:`ccarc.arc_2` solves the static ARC-AGI-2
puzzles; :mod:`ccarc.arc_3` plays the interactive ARC-AGI-3 games. They share
the guardrail, the staging step and the workspace discipline, and differ in what
the agent is told -- see the README on why that difference is deliberate.
"""

from __future__ import annotations

__all__ = ["arc_2", "arc_3", "staging"]

# `data` and `_vendor` are deliberately absent from `__all__`. They are
# reachable by name -- this hides them from the index, not from the import
# system -- but a solver that lists this package should not be handed `data`,
# which resolves the ARC-AGI-2 dataset. `staging._EXCLUDED` is what actually
# keeps it off the solver's path; this is only the name that would advertise it.
_HIDDEN_FROM_INDEX = ("data", "_vendor")


def __dir__() -> list[str]:
    return sorted(__all__)
