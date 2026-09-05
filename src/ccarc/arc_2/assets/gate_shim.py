#!/usr/bin/env python3
"""Verification gate for this puzzle workspace.

    python gate.py status    distilled research state (run this after a compaction)
    python gate.py submit    formal iteration: solution/hypothesis.md + solution/solve.py
    python gate.py accept    finalize, using the last submission and solution/audit.md

Thin shim over ``ccarc.arc_2.gate``; the source path below is baked in
when the workspace is created.
"""

from __future__ import annotations

import sys

STAGE_SRC = r"__STAGE_SRC__"

if STAGE_SRC not in sys.path:
    sys.path.insert(0, STAGE_SRC)

from ccarc.arc_2.gate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
