"""The level-boundary gate.

The first action of a new level is refused until the rule book has been updated.
The gate does not judge the content: it checks that something was written for
this level and moves on.
"""
# **The design argument, kept out of the docstring because that is solver-
# visible.** A sibling driver for the static benchmark refuses a submission that
# does not reproduce every training pair, and that refusal is what keeps a
# solver's belief in contact with its evidence: you cannot assert your way
# past it. This benchmark has no submission to gate, but a level boundary
# serves the same purpose and costs nothing.
#
# Nothing in the gate's solver-facing text says its entries are never read:
# that is an invitation to write nothing worth reading.
#
# **Why structural and not advisory**, kept here rather than in the docstring
# above because comments do not reach the solver: an instruction to "remember
# what you learned" competes with the immediate pull of the new level, and a
# refusal does not.
#
# **This docstring is solver-visible.** `__init__.py` does
# `from .gate import GateRefusal, LevelGate`, which binds the submodule, so
# `arc.gate.__doc__` and `pydoc ccarc.arc_3.gate` both print it -- reachable
# by introspection from an object the workspace hands the solver by name. So
# it quotes no measured figure.

from __future__ import annotations

import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rules import RuleBook

__all__ = ["GateRefusal", "LevelGate"]


class GateRefusal(RuntimeError):
    """The gate declined an action. Nothing was sent and the game did not step."""


#: The gate's two refusals, as templates. Same reason as `client.REFUSALS`.
GATE_REFUSALS = {
    "unacknowledged":
        "level {level} reached and the rule book has not been "
        "updated. Before acting again, record what level "
        "{previous} established. Call "
        "gate.acknowledge(summary, mechanics=..., refuted=..., "
        "untested=...) to proceed.",
    "empty_summary":
        "summary is empty; the gate wants a claim, not a token",
}


@dataclass
class LevelGate:
    """Requires a rule-book entry at every level boundary.

    Armed by a level *advance*, not by entering a level.
    """

    rulebook_path: Path | str = "rules.json"
    pending_level: int | None = None
    acknowledged: dict[int, str] = field(default_factory=dict)
    last_level: int = 0
    refusals: int = 0
    on_change: Any = None
    """Called with no arguments when :meth:`acknowledge` clears the gate.

    ``ArcClient`` sets this to its state saver. Without it an acknowledgement is
    lost the moment the process ends: the client persists after each *action*,
    so clearing the gate and then exiting leaves ``gate_pending`` still set on
    disk, and the next process restores a gate that was already satisfied.

    **Only that transition, and the asymmetry is the point.** :meth:`observe`
    can take the gate from open to held and does not call it, and does not
    need to: arming happens *inside* an action, a few lines before
    ``ArcClient._act`` saves unconditionally, while clearing happens *between*
    actions, when the solver calls :meth:`acknowledge` directly and nothing
    else is going to write to disk. Do not widen the contract with a redundant
    save in :meth:`observe`, and do not read the missing call as a sign the
    hook is unnecessary.

    :meth:`check` does not fire it either; it increments ``refusals`` and
    raises, leaving the new count to be persisted by the next save.
    """

    def _changed(self) -> None:
        if self.on_change is not None:
            self.on_change()

    def observe(self, level: int) -> None:
        """Record the level reported by the latest frame."""
        if level > self.last_level:
            # A boundary already acknowledged has nothing new to record. This is
            # what makes a *replay* usable: after a full reset the solver
            # re-crosses every boundary it has already documented, and demanding
            # a fresh entry at each one would cost a turn apiece to restate what
            # the rule book already holds. The gate exists to catch knowledge
            # about to be lost, not to bill for knowledge already kept.
            self.pending_level = None if level in self.acknowledged else level
        elif level < self.last_level:
            # A full reset rewound the game. Whatever was pending is moot; the
            # rule book keeps what it already learned, which is the point of
            # keeping knowledge separate from progress.
            self.pending_level = None
        self.last_level = level

    def check(self) -> None:
        """Raise if a level advance is still unacknowledged."""
        # **The switch is named in a comment, not in the docstring.** The
        # workspace hands the solver this object by name, so
        # `help(gate.check)` reaches its docstring, and the switch is the
        # harness's to set, not the solver's. Staging strips comments from the
        # package and leaves docstrings, so this is the side of that line the
        # solver cannot read. The variable is `CCARC3_LEVEL_GATE`; `0`
        # disables the refusal.
        #
        # **The last channel by which a level advance reaches the solver.**
        # `_announce_level` sends a user message and has no print fallback, so
        # `CCARC3_LEVEL_ANNOUNCE=0` silences that one completely. This refusal
        # is the other: it arrives inside a tool result, names the level, and
        # tells the solver to record what the previous one established. With
        # both off the solver gets no level-advance signal at all and has to
        # read `client.level` itself -- a doctrine change, not a tuning knob,
        # because the gate exists to catch knowledge about to be lost and its
        # acknowledgement bookkeeping is what keeps a replay affordable.
        # Default stays ON.
        if os.environ.get("CCARC3_LEVEL_GATE", "1") == "0":
            return
        if self.pending_level is None:
            return
        self.refusals += 1
        raise GateRefusal(GATE_REFUSALS["unacknowledged"].format(
            level=self.pending_level, previous=self.pending_level - 1))

    def acknowledge(
        self,
        summary: str,
        *,
        mechanics: list[str] | None = None,
        refuted: list[str] | None = None,
        untested: list[str] | None = None,
        book: RuleBook | None = None,
    ) -> RuleBook:
        """Record what the finished level established, and clear the gate.

        Three separate lists:

        - ``mechanics`` — beliefs you want carried forward.
        - ``refuted`` — tested and found false.
        - ``untested`` — never exercised, so nothing is known either way.

        Callable when the gate is not holding anything: it records under the
        level you have reached and returns the book.
        """
        # **The third list exists so that a claim never exercised does not end
        # up filed under `refuted` for want of anywhere else to put it.** Kept
        # as a comment: staging strips comments, so this reasoning does not
        # reach the solver, which has to arrive at it on its own.
        # **A replay re-crosses boundaries the gate will not arm.**
        # :meth:`observe` leaves ``pending_level`` unset for a boundary already
        # in ``acknowledged``, which is what makes a replay affordable, while
        # the workspace guide tells the solver to acknowledge after every
        # level. A per-level script must therefore not raise on a replay.
        #
        # So an ungated acknowledgement records what it was given and returns,
        # under the level the solver has actually reached. There is nothing to
        # refuse: the solver is volunteering knowledge the gate did not have to
        # ask for. ``acknowledged`` is left alone -- it tracks which boundaries
        # were *gated*, and `observe` reads it to decide whether to arm.
        gated = self.pending_level is not None
        level = self.pending_level if gated else self.last_level
        if not summary.strip():
            if not gated:
                return book or RuleBook.load(self.rulebook_path)
            raise GateRefusal(GATE_REFUSALS["empty_summary"])

        book = book or RuleBook.load(self.rulebook_path)
        for m in mechanics or []:
            book.verified.append({"rule": m, "scope": "game", "level": level - 1, "note": summary})
        for r in refuted or []:
            book.refuted.append({"rule": r, "scope": "game", "level": level - 1, "note": summary})
        for u in untested or []:
            book.open_questions.append(f"level {level - 1}: UNTESTED — {u}")
        if not (mechanics or refuted or untested):
            # A level can genuinely teach nothing portable, and saying so is a
            # real answer. It is recorded as an open question rather than
            # silently dropped, so a run that keeps producing them is visible.
            book.open_questions.append(f"level {level - 1}: {summary}")
        book.save(self.rulebook_path)

        if not gated:
            # Nothing was being held, so there is no gate state to change and
            # nothing to persist beyond the book just saved. Recording this in
            # ``acknowledged`` would be a lie -- that map answers "was this
            # boundary gated", which :meth:`observe` reads to decide whether to
            # arm, and writing a never-gated level into it would silence a real
            # boundary the solver has not yet crossed.
            return book
        self.acknowledged[level] = summary
        self.pending_level = None
        self._changed()
        return book

    @property
    def held(self) -> bool:
        return self.pending_level is not None

    def status(self) -> str:
        if self.held:
            return f"GATE HELD at level {self.pending_level} — acknowledge to continue"
        return f"gate open (level {self.last_level}, {len(self.acknowledged)} acknowledged)"
