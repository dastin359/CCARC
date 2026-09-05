"""Scoring and integrity checks for CC-harness runs.

Scoring follows ARC-AGI-2: each test example is scored independently and counts
as solved when **any** submitted candidate matches its ground-truth output
exactly; the task score is the fraction of test examples solved.

Ground truth is held here, never in the workspace — which makes the integrity
check meaningful: if the solver reached the benchmark data anyway, the run is
contaminated and the score is worthless.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

Grid = list[list[int]]


def score_final(
    final: dict[str, Any] | None,
    expected_outputs: list[Grid | None],
) -> dict[str, Any]:
    """Score an accepted run against the held-out test outputs."""
    result: dict[str, Any] = {
        "scored": False,
        "solved": False,
        "score": 0.0,
        "num_test": len(expected_outputs),
        "num_solved": 0,
        "per_test": [],
    }
    if not final:
        return result

    rows = {int(row.get("index", i)): row for i, row in enumerate(final.get("test") or [])}
    solved_count = 0
    for index, expected in enumerate(expected_outputs):
        row = rows.get(index) or {}
        candidates: list[Grid] = row.get("candidates") or []
        matched = next(
            (i for i, candidate in enumerate(candidates, start=1) if candidate == expected),
            None,
        )
        solved = matched is not None and expected is not None
        solved_count += 1 if solved else 0
        result["per_test"].append(
            {
                "index": index,
                "solved": solved,
                "matched_candidate": matched,
                "num_candidates": len(candidates),
                "error": row.get("error"),
            }
        )

    result["scored"] = True
    result["num_solved"] = solved_count
    result["score"] = solved_count / len(expected_outputs) if expected_outputs else 0.0
    result["solved"] = bool(expected_outputs) and solved_count == len(expected_outputs)
    return result


# ── integrity ────────────────────────────────────────────────────────────────

_FORBIDDEN_TOOLS = ("WebFetch", "WebSearch")


#: Absolute paths a solver mentions in a command or tool argument. Relative
#: paths are workspace-relative by construction and are not interesting here.
_OUT_OF_WORKSPACE = re.compile(r"/(?:root|home|etc|usr|var|opt|srv|mnt|media)/[\w./\\-]{2,}")

#: Paths every run legitimately touches. The solver's interpreter lives outside
#: the workspace by design, and the harness tells it so in CLAUDE.md.
#:
#: **Derived from the running interpreter and this package, not from a literal.**
#: A hard-coded checkout path stops matching the moment the repo lives anywhere
#: else, and then the interpreter the harness *tells the solver to run* gets
#: flagged as reaching outside its workspace. That direction is over-reporting,
#: and it is the one that buries a real finding under noise on every single run.
#:
#: `sys.prefix` is the venv the harness is running in and this module's
#: ``__file__`` is where the package was imported from -- each is the thing
#: itself, so both stay correct under an editable install, a site-packages
#: install, or a repo checked out anywhere at all.
def _benign_prefixes() -> tuple[str, ...]:
    import sys as _sys

    here = Path(__file__).resolve()
    # <...>/ccarc/arc_2/scoring.py -> the directory holding `ccarc`
    package_root = here.parents[2]
    prefixes = ["/usr/", _sys.prefix.rstrip("/") + "/", str(package_root) + "/"]
    # De-duplicate while keeping order; a venv inside the repo makes two of
    # these overlap, and a duplicated prefix is harmless but noisy to read.
    seen, out = set(), []
    for pre in prefixes:
        if pre not in seen:
            seen.add(pre)
            out.append(pre)
    return tuple(out)


_BENIGN_PATH_PREFIXES = _benign_prefixes()


def contamination_scan(
    *,
    workspace_root: Path | str,
    stream_path: Path | str | None = None,
    dataset_root: Path | str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Look for signs the solver reached outside its workspace.

    Advisory, not authoritative: it catches the obvious routes (opening the
    benchmark file, using a network tool, test outputs appearing in the
    workspace), which is what a research harness needs to be able to say it
    checked.
    """
    workspace_root = Path(workspace_root).resolve()
    evidence: list[str] = []
    own_spill_marker = str(workspace_root).replace("/", "-")
    stage_marker = str(workspace_root.parent / "_pkg")

    task_file = workspace_root / "task" / "task.json"
    if task_file.is_file():
        try:
            task = json.loads(task_file.read_text(encoding="utf-8"))
            if any("output" in (entry or {}) for entry in task.get("test") or []):
                evidence.append("task/task.json contains test outputs — the workspace was tampered with")
        except json.JSONDecodeError:
            evidence.append("task/task.json is no longer valid JSON")

    if stream_path and Path(stream_path).is_file():
        stream_text = Path(stream_path).read_text(encoding="utf-8", errors="replace")

        if dataset_root:
            needle = str(Path(dataset_root).resolve())
            if needle in stream_text:
                evidence.append(f"transcript references the dataset root {needle}")

        if task_id:
            pattern = re.compile(r"[\w./\\-]*" + re.escape(task_id) + r"\.json")
            for match in set(pattern.findall(stream_text)):
                normalized = match.replace("\\", "/")
                if normalized.endswith("task/task.json") or normalized == f"{task_id}.json":
                    continue
                evidence.append(f"transcript references an out-of-workspace task file: {match}")

        for tool in _FORBIDDEN_TOOLS:
            if f'"name":"{tool}"' in stream_text or f'"name": "{tool}"' in stream_text:
                evidence.append(f"transcript contains a {tool} tool call")

        # A solver reading a file outside its workspace is the route the other
        # checks do not cover. The scan itself must say which references are
        # benign — leaving that adjudication to a human is the wrong division
        # of labour.
        for match in set(_OUT_OF_WORKSPACE.findall(stream_text)):
            resolved = match.replace("\\", "/")
            if resolved.startswith(str(workspace_root)):
                continue
            if any(resolved.startswith(prefix) for prefix in _BENIGN_PATH_PREFIXES):
                continue
            # Claude Code spills large tool results to a project directory whose
            # name is the workspace path with separators replaced. A reference
            # to *its own* spill is the CLI's storage, not an external read —
            # and it is common enough that flagging it would bury a real one.
            # A spill path encoding a *different* workspace is still flagged.
            if own_spill_marker and own_spill_marker in resolved:
                continue
            # **The staged package is the run's own machinery, not an escape.**
            # It sits beside the workspace rather than inside it, so the rule
            # above reads it as an outside path -- and the very first jailed run
            # was flagged for importing the package it is required to import.
            # The stage exists precisely so the solver does NOT reach the
            # repository, so treating it as a violation inverts the finding.
            if stage_marker and resolved.startswith(stage_marker):
                continue
            evidence.append(f"transcript references a path outside the workspace: {match}")

    return {"suspected": bool(evidence), "evidence": sorted(set(evidence))}


# ── batch aggregation ────────────────────────────────────────────────────────

def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Roll per-task results up into the numbers the README cares about."""
    scored = [r for r in results if (r.get("score") or {}).get("scored")]
    solved = [r for r in scored if r["score"]["solved"]]
    costs = [float(r.get("cost_usd") or 0.0) for r in results if r.get("cost_usd") is not None]
    iterations = [int(r.get("iterations_used") or 0) for r in results]
    partial = sum(float(r["score"]["score"]) for r in scored)

    return {
        "tasks": len(results),
        "scored": len(scored),
        "solved": len(solved),
        "accuracy": len(solved) / len(scored) if scored else 0.0,
        "partial_credit": partial / len(scored) if scored else 0.0,
        "total_cost_usd": round(sum(costs), 4),
        "mean_cost_usd": round(sum(costs) / len(costs), 4) if costs else 0.0,
        "mean_iterations": round(sum(iterations) / len(iterations), 2) if iterations else 0.0,
        "accepted": sum(1 for r in results if r.get("accepted")),
        "errors": [r.get("task_id") for r in results if r.get("error")],
        "contaminated": [
            r.get("task_id") for r in results if (r.get("integrity") or {}).get("suspected")
        ],
    }


def format_aggregate(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    lines = [
        f"tasks      : {summary['tasks']}  (accepted {summary['accepted']}, scored {summary['scored']})",
        f"solved     : {summary['solved']}/{summary['scored']}  = {summary['accuracy'] * 100:.1f}%",
        f"partial    : {summary['partial_credit'] * 100:.1f}%  (per-test-example credit)",
        f"cost       : ${summary['total_cost_usd']:.2f} total, ${summary['mean_cost_usd']:.3f} per task",
        f"iterations : {summary['mean_iterations']} mean",
    ]
    if summary["errors"]:
        lines.append(f"errors     : {', '.join(str(t) for t in summary['errors'])}")
    if summary["contaminated"]:
        lines.append(f"CONTAMINATED: {', '.join(str(t) for t in summary['contaminated'])}")
    lines.append("")
    for record in results:
        score = record.get("score") or {}
        mark = "solved" if score.get("solved") else ("partial" if score.get("num_solved") else "  --  ")
        lines.append(
            f"  {record.get('task_id'):<12} {mark:<7} "
            f"{score.get('num_solved', 0)}/{score.get('num_test', 0)} test  "
            f"{record.get('iterations_used', 0)} iter  "
            f"${float(record.get('cost_usd') or 0.0):.3f}"
            + (f"  [{record['error']}]" if record.get("error") else "")
        )
    return "\n".join(lines)
