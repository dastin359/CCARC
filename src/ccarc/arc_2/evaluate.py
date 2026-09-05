"""Execution and scoring of a candidate ``solve(grid)`` implementation.

The observable semantics are fixed, so that a result means the same thing
whichever harness produced it:

* ``solve(grid)`` may return one grid, or a list of up to ``max_candidates``
  grids.
* Training validation uses **only the first candidate**. Returning more than one
  candidate for a training input is a modelling smell and is reported back.
* Test inputs may carry up to ``max_candidates`` distinct candidates; exact
  duplicates are collapsed.

Two properties compensate for the absence of an independent reviewer agent:

1. ``solve.py`` is executed with **no puzzle globals injected**:
   ``solve(grid)`` sees only its argument, in a fresh interpreter, so a
   lookup-table solution has nothing to look up. With no reviewer, the
   execution contract itself carries that guarantee.
2. Submitted code is statically screened for verbatim training outputs
   (:func:`detect_hardcoding`) and the finding is surfaced in the gate report.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

Grid = list[list[int]]

#: Upper bound on captured stdout/stderr from model-authored code, in characters.
MAX_CAPTURED_OUTPUT_CHARS = 4000

_WORKER = Path(__file__).with_name("_solve_worker.py")


# ── grid primitives ──────────────────────────────────────────────────────────

def numpy_to_python(obj: Any) -> Any:
    """Recursively convert NumPy scalars/arrays into plain Python values.

    Degrades gracefully when NumPy is not importable, so the workspace
    toolkit stays dependency-free.
    """
    if hasattr(obj, "tolist") and not isinstance(obj, (list, tuple, dict, str, bytes)):
        try:
            return numpy_to_python(obj.tolist())
        except Exception:
            pass
    if hasattr(obj, "item") and not isinstance(obj, (list, tuple, dict, str, bytes)):
        try:
            return numpy_to_python(obj.item())
        except Exception:
            pass
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [numpy_to_python(v) for v in obj]
    if isinstance(obj, bool):
        return bool(obj)
    return obj


def is_valid_grid(value: Any) -> bool:
    """True when ``value`` is a non-empty rectangular 2D list of ints."""
    if not isinstance(value, list) or not value:
        return False
    if not all(isinstance(row, list) and row for row in value):
        return False
    width = len(value[0])
    if width <= 0:
        return False
    for row in value:
        if len(row) != width:
            return False
        for cell in row:
            # bool is an int subclass; a boolean grid is a bug, not a colour grid.
            if isinstance(cell, bool) or not isinstance(cell, int):
                return False
    return True


def normalize_candidates(
    raw_prediction: Any,
    *,
    max_candidates: int = 2,
) -> tuple[list[Grid] | None, str | None]:
    """Normalise a ``solve()`` return value into a list of unique grids.

    Returns ``(candidates, None)`` on success or ``(None, message)`` on failure.
    """
    prediction = numpy_to_python(raw_prediction)

    if is_valid_grid(prediction):
        return [prediction], None

    if isinstance(prediction, tuple):
        prediction = list(prediction)

    if not isinstance(prediction, list):
        return None, "solve(grid) must return either a grid or a list of one or two grids."
    if not prediction:
        return None, "solve(grid) returned an empty candidate list."
    if len(prediction) > max_candidates:
        return None, f"solve(grid) returned more than {max_candidates} candidate grid(s)."

    normalized: list[Grid] = []
    seen: set[str] = set()
    for candidate in prediction:
        candidate = numpy_to_python(candidate)
        if not is_valid_grid(candidate):
            return None, "Every item in the returned candidate list must be a valid 2D integer grid."
        signature = json.dumps(candidate, separators=(",", ":"))
        if signature in seen:
            continue
        seen.add(signature)
        normalized.append(candidate)

    if not normalized:
        return None, "solve(grid) returned only duplicate candidates and no valid unique grid remained."
    return normalized, None


def pixel_accuracy(predicted: Grid | None, expected: Grid | None) -> float:
    """Fraction of expected cells the prediction reproduces (0.0 on mismatch)."""
    if predicted is None or expected is None:
        return 0.0
    try:
        exp_rows = len(expected)
        exp_cols = len(expected[0]) if exp_rows else 0
        total = exp_rows * exp_cols
        pred_rows = len(predicted)
        pred_cols = len(predicted[0]) if pred_rows else 0
        if total == 0:
            return 1.0 if (pred_rows, pred_cols) == (exp_rows, exp_cols) else 0.0
        matching = 0
        for r in range(min(pred_rows, exp_rows)):
            for c in range(min(pred_cols, exp_cols)):
                if predicted[r][c] == expected[r][c]:
                    matching += 1
        return matching / total
    except Exception:
        return 0.0


def grid_diff(predicted: Grid | None, expected: Grid | None, *, limit: int = 40) -> dict[str, Any]:
    """Compact structural difference between two grids.

    Returns dimensions, the number of differing cells, a bounding box over the
    differences, and up to ``limit`` ``(row, col, expected, got)`` entries.
    """
    info: dict[str, Any] = {
        "expected_shape": None,
        "predicted_shape": None,
        "shape_match": False,
        "num_diff": None,
        "bbox": None,
        "cells": [],
        "truncated": False,
    }
    if expected is None:
        return info
    exp_rows = len(expected)
    exp_cols = len(expected[0]) if exp_rows else 0
    info["expected_shape"] = [exp_rows, exp_cols]
    if predicted is None:
        return info
    pred_rows = len(predicted)
    pred_cols = len(predicted[0]) if pred_rows else 0
    info["predicted_shape"] = [pred_rows, pred_cols]
    info["shape_match"] = (exp_rows, exp_cols) == (pred_rows, pred_cols)
    if not info["shape_match"]:
        return info

    cells: list[list[int]] = []
    num_diff = 0
    min_r = min_c = 1 << 30
    max_r = max_c = -1
    for r in range(exp_rows):
        for c in range(exp_cols):
            if predicted[r][c] != expected[r][c]:
                num_diff += 1
                min_r, max_r = min(min_r, r), max(max_r, r)
                min_c, max_c = min(min_c, c), max(max_c, c)
                if len(cells) < limit:
                    cells.append([r, c, expected[r][c], predicted[r][c]])
    info["num_diff"] = num_diff
    info["cells"] = cells
    info["truncated"] = num_diff > len(cells)
    if num_diff:
        info["bbox"] = [min_r, min_c, max_r, max_c]
    return info


# ── static screening ─────────────────────────────────────────────────────────

def _grid_literal_signatures(grid: Grid) -> set[str]:
    """Whitespace-insensitive spellings of a grid literal."""
    compact = json.dumps(grid, separators=(",", ":"))
    return {compact, compact.replace("[", "(").replace("]", ")")}


def detect_hardcoding(code: str, train_samples: list[dict[str, Any]]) -> list[str]:
    """Flag blatant memorisation in submitted solution code.

    Catches two patterns worth surfacing when no reviewer is present:
    verbatim training outputs embedded as literals, and reaching for the puzzle
    data by name. Advisory only — the gate reports findings but never blocks on
    them, because a legitimate solution can contain a small literal by chance.
    """
    findings: list[str] = []
    stripped = re.sub(r"\s+", "", code or "")

    for idx, sample in enumerate(train_samples or []):
        expected = sample.get("output")
        if not is_valid_grid(expected):
            continue
        # A 1x1 or otherwise tiny output is not evidence of anything.
        if len(expected) * len(expected[0]) < 6:
            continue
        if any(sig in stripped for sig in _grid_literal_signatures(expected)):
            findings.append(
                f"training example {idx}'s expected output appears verbatim as a literal in solve.py"
            )

    for name in ("train_samples", "test_samples"):
        if re.search(rf"\b{name}\b", code or ""):
            findings.append(
                f"solve.py references `{name}`, which is not available at submission time "
                "(solve(grid) receives only its argument)"
            )
    return findings


# ── execution ────────────────────────────────────────────────────────────────

def evaluate_code_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Run ``payload['code']`` against the supplied samples, in-process.

    Used by :mod:`ccarc.arc_2._solve_worker`; call
    :func:`run_solution_isolated` instead of calling this directly, so that
    model-authored code cannot take the gate process down with it.
    """
    import contextlib
    import copy
    import io
    import traceback

    code = str(payload.get("code") or "")
    train_samples = payload.get("train") or []
    test_samples = payload.get("test") or []
    max_candidates = int(payload.get("max_candidates") or 2)

    result: dict[str, Any] = {
        "status": "error",
        "error": "",
        "stdout": "",
        "train": [],
        "test": [],
        "train_correct": 0,
        "train_total": len(train_samples),
        "train_pixel_accuracy": 0.0,
        "all_train_correct": False,
        "num_test_candidates": 0,
        "multi_candidate_on_train": False,
    }

    out, err = io.StringIO(), io.StringIO()
    namespace: dict[str, Any] = {"__name__": "__solution__", "__builtins__": __builtins__}

    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            exec(compile(code, "solve.py", "exec"), namespace)
        except BaseException:  # noqa: BLE001 - SystemExit from model code counts as a failure
            result["error"] = "Failed while defining solve():\n" + traceback.format_exc(limit=6)
            result["stdout"] = _merge_streams(out, err)
            return result

    solve_fn = namespace.get("solve")
    if not callable(solve_fn):
        result["error"] = "solve.py does not define a callable solve(grid)."
        result["stdout"] = _merge_streams(out, err)
        return result

    total_pixel = 0.0
    correct = 0
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        for idx, sample in enumerate(train_samples):
            expected = sample.get("output")
            row: dict[str, Any] = {"index": idx, "correct": False, "pixel_accuracy": 0.0}
            try:
                candidates, message = normalize_candidates(
                    solve_fn(copy.deepcopy(sample.get("input"))),
                    max_candidates=max_candidates,
                )
                if message:
                    raise ValueError(message)
                if len(candidates) > 1:
                    row["multi_candidate_warning"] = True
                    result["multi_candidate_on_train"] = True
                predicted = candidates[0]  # training scores the first candidate only
                row["predicted"] = predicted
                row["correct"] = predicted == expected
                row["pixel_accuracy"] = pixel_accuracy(predicted, expected)
            except BaseException as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
            total_pixel += float(row["pixel_accuracy"])
            correct += 1 if row["correct"] else 0
            result["train"].append(row)

        max_seen = 0
        for idx, sample in enumerate(test_samples):
            row = {"index": idx, "candidates": []}
            try:
                candidates, message = normalize_candidates(
                    solve_fn(copy.deepcopy(sample.get("input"))),
                    max_candidates=max_candidates,
                )
                if message:
                    raise ValueError(message)
                row["candidates"] = candidates
                max_seen = max(max_seen, len(candidates))
            except BaseException as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}: {exc}"
            result["test"].append(row)
        result["num_test_candidates"] = max_seen

    if train_samples:
        result["train_pixel_accuracy"] = total_pixel / len(train_samples)
    result["train_correct"] = correct
    result["all_train_correct"] = bool(train_samples) and correct == len(train_samples)
    result["status"] = "ok"
    result["stdout"] = _merge_streams(out, err)
    return result


def _merge_streams(out, err) -> str:
    text = out.getvalue()
    errors = err.getvalue()
    if errors:
        text = f"{text}\n[stderr]\n{errors}" if text else f"[stderr]\n{errors}"
    if len(text) > MAX_CAPTURED_OUTPUT_CHARS:
        omitted = len(text) - MAX_CAPTURED_OUTPUT_CHARS
        text = text[:MAX_CAPTURED_OUTPUT_CHARS] + f"\n… [{omitted} more characters omitted]"
    return text


def run_solution_isolated(
    code: str,
    train_samples: list[dict[str, Any]],
    test_samples: list[dict[str, Any]],
    *,
    max_candidates: int = 2,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Evaluate ``code`` in a fresh interpreter, bounded by ``timeout_seconds``.

    Isolation matters here: the gate owns the run ledger, and model-authored
    code that hangs, exits, or corrupts interpreter state must not be able to
    take that ledger with it.
    """
    payload = {
        "code": code,
        "train": train_samples,
        "test": [{"input": s.get("input")} for s in test_samples],  # never pass test outputs
        "max_candidates": max_candidates,
    }
    with tempfile.TemporaryDirectory(prefix="ccarc-solve-") as tmp:
        payload_path = os.path.join(tmp, "payload.json")
        result_path = os.path.join(tmp, "result.json")
        Path(payload_path).write_text(json.dumps(payload), encoding="utf-8")

        env = dict(os.environ)
        env.pop("ARC_DATA_ROOT", None)
        try:
            completed = subprocess.run(
                [sys.executable, str(_WORKER), payload_path, result_path],
                capture_output=True,
                text=True,
                timeout=max(1.0, float(timeout_seconds)),
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "error",
                "error": f"solve() evaluation timed out after {timeout_seconds:.0f} seconds.",
                "stdout": "",
                "train": [],
                "test": [],
                "train_correct": 0,
                "train_total": len(train_samples),
                "train_pixel_accuracy": 0.0,
                "all_train_correct": False,
                "num_test_candidates": 0,
                "timed_out": True,
            }

        if os.path.exists(result_path):
            try:
                return json.loads(Path(result_path).read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                worker_error = f"Could not read evaluation result: {exc}"
        else:
            worker_error = (completed.stderr or completed.stdout or "").strip() or (
                f"Evaluation subprocess exited with code {completed.returncode} and produced no result."
            )

        return {
            "status": "error",
            "error": worker_error[:MAX_CAPTURED_OUTPUT_CHARS],
            "stdout": "",
            "train": [],
            "test": [],
            "train_correct": 0,
            "train_total": len(train_samples),
            "train_pixel_accuracy": 0.0,
            "all_train_correct": False,
            "num_test_candidates": 0,
        }
