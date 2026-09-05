"""Claude Code as harness — a single-agent ARC-AGI-2 solver.

The agent loop belongs to Claude Code: it drives turns, executes tools and
manages its own context. This package supplies the workspace, the doctrine
and the verification gate, and encodes three mechanisms:

* **Code as verification.** Every claim about the puzzle must be established by
  executing something, not by prose enumeration. The workspace toolkit
  (`arc.verify`) makes that the literal API for asserting a fact.
* **Artifact separation.** A natural-language hypothesis and a `solve()`
  implementation are separate, auditable artifacts, and the hypothesis must be
  written before the code it explains.
* **Budgeted formal iterations.** Exploration is unlimited and free; formal
  submission through the gate is counted and recorded.

There is no reviewer agent — the quality gate is `train 100%` plus a
mandatory generalization self-audit.
"""

from .config import CCRunConfig

__all__ = ["CCRunConfig"]
