"""Subprocess entry point that evaluates model-authored ``solve.py``.

Invoked as ``python _solve_worker.py <payload.json> <result.json>``. Kept
importable-free of the rest of the harness runtime so a broken environment in
the parent cannot mask a solution-code failure.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

# The worker is executed by path, so the package root has to be put on sys.path
# explicitly before importing the shared evaluation helpers.
_SRC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from ccarc.arc_2.evaluate import evaluate_code_payload  # noqa: E402


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: _solve_worker.py <payload.json> <result.json>", file=sys.stderr)
        return 2

    payload_path, result_path = argv[1], argv[2]
    with open(payload_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    try:
        result = evaluate_code_payload(payload)
    except BaseException:  # noqa: BLE001 - report, never crash silently
        result = {
            "status": "error",
            "error": "Evaluation harness failure:\n" + traceback.format_exc(limit=8),
            "stdout": "",
            "train": [],
            "test": [],
            "train_correct": 0,
            "train_total": len(payload.get("train") or []),
            "train_pixel_accuracy": 0.0,
            "all_train_correct": False,
            "num_test_candidates": 0,
        }

    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
