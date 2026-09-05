"""The rule book: a structured, on-disk place for what the solver believes.

One of the two mechanisms this harness is built on. `RuleBook` persists across
processes, so a belief written down at one level boundary is still there many
levels later, addressable by name rather than by scrolling a conversation.

What goes in it, and how it is organised, is the solver's own business.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


__all__ = ["RuleBook"]


@dataclass
class RuleBook:
    """A structured place to write down what is believed, what is ruled out,
    and what is still open. Persists across processes.
    """

    verified: list[dict[str, Any]] = field(default_factory=list)
    refuted: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        """Write the rule book atomically.

        A truncated ``rules.json`` leaves :meth:`LevelGate.acknowledge` unable
        to load the book, and so unable to clear the gate.
        """
        # Temp file plus `os.replace`, like every other state write in the
        # package: a bare `write_text` truncates before it writes, so a
        # process stopped in that window leaves a zero-byte book.
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, p)                       # atomic on POSIX

    @classmethod
    def load(cls, path: str | Path) -> "RuleBook":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(**json.loads(p.read_text(encoding="utf-8")))
