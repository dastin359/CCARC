"""The verification gate — the harness's only enforcement point.

Claude Code owns the agent loop, so everything a bespoke orchestrator would
enforce by construction has to be enforced here instead:

============================  ==================================================
Orchestrator mechanism        CC harness gate
============================  ==================================================
tool ordering forces the      ``submit`` refuses code that changed without a
hypothesis first              correspondingly updated hypothesis
one solution execution        ``submit`` consumes one budgeted iteration and
counts as one iteration       appends an immutable record under ``.ccarc/``
reflection prompt injected    the failure report ends with the reflection
into the conversation         directive (see :mod:`.reporting`)
test-generalization           ``accept`` requires ``solution/audit.md`` with an
self-audit turn               explicit CONFIDENCE / DECISION
best-effort prompt near       the gate lifts the train-100% requirement over the
budget exhaustion             trailing iterations
============================  ==================================================

Exploration is untouched by all of this: running ``python explore/whatever.py``
costs nothing and is the intended way to do almost all of the work.

Usage (from inside a run workspace)::

    python gate.py status
    python gate.py submit
    python gate.py accept
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .evaluate import detect_hardcoding, run_solution_isolated
from .reporting import format_status, format_submission_report
from .signals import output_space_signals

STATE_DIR = ".ccarc"
STATE_FILE = "state.json"
INVARIANTS_FILE = "invariants.jsonl"
RIVALS_FILE = "rivals.jsonl"
EVENTS_FILE = "events.jsonl"
FINAL_FILE = "final.json"

HYPOTHESIS_PATH = "solution/hypothesis.md"
CODE_PATH = "solution/solve.py"
AUDIT_PATH = "solution/audit.md"
NOTES_PATH = "NOTES.md"
TASK_PATH = "task/task.json"

NOTES_TAIL_CHARS = 3000


class GateError(Exception):
    """A refusal: the precondition failed, so no iteration was consumed."""


# ── workspace plumbing ───────────────────────────────────────────────────────

def find_workspace(start: Path | None = None) -> Path:
    """Walk up from ``start`` to the directory holding ``.ccarc/state.json``."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / STATE_DIR / STATE_FILE).is_file():
            return candidate
    raise GateError(
        "Not inside a ccarc run workspace (no .ccarc/state.json found). "
        "Run this from the workspace root."
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def load_state(workspace: Path) -> dict[str, Any]:
    return json.loads((workspace / STATE_DIR / STATE_FILE).read_text(encoding="utf-8"))


def save_state(workspace: Path, state: dict[str, Any]) -> None:
    path = workspace / STATE_DIR / STATE_FILE
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_task(workspace: Path) -> dict[str, Any]:
    return json.loads((workspace / TASK_PATH).read_text(encoding="utf-8"))


def load_invariants(workspace: Path) -> list[dict[str, Any]]:
    path = workspace / STATE_DIR / INVARIANTS_FILE
    if not path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # Later verifications of the same claim supersede earlier ones, and a
    # retraction removes it: the ledger is only worth reading if everything in
    # it has actually been executed.
    deduped: dict[str, dict[str, Any]] = {}
    for entry in entries:
        deduped[str(entry.get("key") or entry.get("claim"))] = entry
    return [entry for entry in deduped.values() if not entry.get("retracted")]


def load_rivals(workspace: Path) -> list[dict[str, Any]]:
    """Alternative readings the solver registered with arc.rival()."""
    path = workspace / STATE_DIR / RIVALS_FILE
    if not path.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[str(entry.get("name"))] = entry
    return list(latest.values())


def unhedged_rivals(
    workspace: Path, evaluation: dict[str, Any], allowed_candidates: int
) -> list[dict[str, Any]]:
    """Registered rivals that fit training and are missing from the candidates.

    The precise form of a forfeited second attempt: an alternative reading the
    solver implemented, which reproduces every training pair, which predicts
    something different on a test input, and which is not among the grids being
    submitted for it.
    """
    rows = {int(r.get("index", i)): r for i, r in enumerate(evaluation.get("test", []))}
    findings: list[dict[str, Any]] = []
    for entry in load_rivals(workspace):
        if not entry.get("fits_training"):
            continue
        differing: list[int] = []
        for index, prediction in enumerate(entry.get("predictions") or []):
            row = rows.get(index)
            if row is None or row.get("error") or prediction is None:
                continue
            candidates = row.get("candidates") or []
            if len(candidates) < allowed_candidates and prediction not in candidates:
                differing.append(index)
        if differing:
            findings.append({"name": entry.get("name"), "test_indices": differing})
    return findings


def append_event(workspace: Path, event: dict[str, Any]) -> None:
    path = workspace / STATE_DIR / EVENTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"at": _now(), **event}) + "\n")


def _read_text(workspace: Path, relative: str) -> str:
    path = workspace / relative
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def best_effort_active(state: dict[str, Any], *, used: int | None = None) -> bool:
    """True once the trailing best-effort window has been entered."""
    used = len(state.get("iterations") or []) if used is None else used
    remaining = int(state.get("max_iterations") or 0) - used
    return remaining <= int(state.get("best_effort_iterations") or 0)


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_status(workspace: Path, *, brief: bool = False) -> tuple[str, int]:
    """Distilled research state.

    ``brief`` drops the stored hypothesis and the NOTES tail, leaving the
    iteration history and the executed ledger — a mid-run status should not be
    long enough to send a solver reaching for ``head``/``tail``, which is the
    opposite of the "print signal, not dumps" rule this harness holds it to.
    The full form stays the default because the compaction hook injects it, and
    after a compaction the hypothesis is exactly what you have lost.
    """
    state = load_state(workspace)
    iterations = state.get("iterations") or []
    hypothesis = ""
    if iterations:
        last = iterations[-1]
        stored = workspace / STATE_DIR / "iterations" / str(last.get("iteration")) / "hypothesis.md"
        if stored.is_file():
            hypothesis = stored.read_text(encoding="utf-8")
    if not hypothesis:
        hypothesis = _read_text(workspace, HYPOTHESIS_PATH)

    notes = _read_text(workspace, NOTES_PATH)
    if len(notes) > NOTES_TAIL_CHARS:
        notes = "… (earlier entries omitted) …\n" + notes[-NOTES_TAIL_CHARS:]

    if brief:
        hypothesis = ""
        notes = ""

    report = format_status(
        state=state,
        invariants=load_invariants(workspace),
        rivals=load_rivals(workspace),
        hypothesis=hypothesis,
        notes_excerpt=notes,
    )
    append_event(workspace, {"command": "status"})
    return report, 0


def cmd_submit(workspace: Path) -> tuple[str, int]:
    state = load_state(workspace)

    if state.get("accepted"):
        raise GateError("This run has already been accepted. No further submissions are recorded.")

    used = len(state.get("iterations") or [])
    max_iterations = int(state.get("max_iterations") or 0)
    if used >= max_iterations:
        raise GateError(
            f"Iteration budget exhausted ({used}/{max_iterations}). "
            "Write solution/audit.md and run `python gate.py accept` on your best submission."
        )

    hypothesis = _read_text(workspace, HYPOTHESIS_PATH).strip()
    code = _read_text(workspace, CODE_PATH)

    if not hypothesis:
        raise GateError(
            f"{HYPOTHESIS_PATH} is missing or empty. The hypothesis is a first-class artifact: "
            "write the rule in prose before you submit the code that implements it."
        )
    minimum = int(state.get("min_hypothesis_chars") or 0)
    if len(hypothesis) < minimum:
        raise GateError(
            f"{HYPOTHESIS_PATH} is {len(hypothesis)} characters; at least {minimum} are required. "
            "It has to be complete enough that a programmer who has never seen this puzzle could "
            "reimplement solve() from it alone: high-level summary, step-by-step algorithm, every "
            "edge case and conditional, why the rule generalizes, and — if a genuine ambiguity "
            "survives — how candidate 1 differs from candidate 2."
        )
    if not code.strip():
        raise GateError(f"{CODE_PATH} is missing or empty.")
    if "def solve" not in code:
        raise GateError(f"{CODE_PATH} does not define solve(grid).")

    hypothesis_sha = _sha(hypothesis)
    code_sha = _sha(code)
    last_hypothesis_sha = state.get("last_hypothesis_sha") or ""
    last_code_sha = state.get("last_code_sha") or ""

    if used and code_sha == last_code_sha and hypothesis_sha == last_hypothesis_sha:
        raise GateError(
            f"Neither {HYPOTHESIS_PATH} nor {CODE_PATH} changed since iteration {used}. "
            "Resubmitting the same artifacts cannot produce a different result."
        )
    if used and code_sha != last_code_sha and hypothesis_sha == last_hypothesis_sha:
        raise GateError(
            f"{CODE_PATH} changed but {HYPOTHESIS_PATH} did not. Every code change is a change of "
            "rule, of implementation, or of edge-case handling — update the hypothesis to say which, "
            "so the artifact still describes the code it ships with. (No iteration was consumed.)"
        )

    task = load_task(workspace)
    train_samples = task.get("train") or []
    test_samples = task.get("test") or []

    started = time.time()
    evaluation = run_solution_isolated(
        code,
        train_samples,
        test_samples,
        max_candidates=int(state.get("max_test_predictions") or 2),
        timeout_seconds=float(state.get("solve_timeout_s") or 60.0),
    )
    elapsed = time.time() - started

    findings = detect_hardcoding(code, train_samples)
    # Mechanical stand-in for a reviewer: regularities that every training
    # output obeys and a test prediction breaks.
    signals = output_space_signals(
        train_samples,
        [sample.get("input") for sample in test_samples],
        [row.get("candidates") or [] for row in evaluation.get("test", [])],
    )
    allowed_candidates = int(state.get("max_test_predictions") or 2)
    rivals = unhedged_rivals(workspace, evaluation, allowed_candidates)
    iteration = used + 1

    record = {
        "iteration": iteration,
        "at": _now(),
        "hypothesis_sha": hypothesis_sha,
        "hypothesis_chars": len(hypothesis),
        "code_sha": code_sha,
        "code_chars": len(code),
        "status": evaluation.get("status"),
        "error": evaluation.get("error") or "",
        "timed_out": bool(evaluation.get("timed_out")),
        "train_correct": int(evaluation.get("train_correct") or 0),
        "train_total": int(evaluation.get("train_total") or len(train_samples)),
        "train_pixel_accuracy": float(evaluation.get("train_pixel_accuracy") or 0.0),
        "all_train_correct": bool(evaluation.get("all_train_correct")),
        "multi_candidate_on_train": bool(evaluation.get("multi_candidate_on_train")),
        "num_test_candidates": int(evaluation.get("num_test_candidates") or 0),
        "hardcoding_findings": findings,
        "generalization_signals": signals,
        "unhedged_rivals": rivals,
        "elapsed_s": round(elapsed, 3),
    }

    iteration_dir = workspace / STATE_DIR / "iterations" / str(iteration)
    iteration_dir.mkdir(parents=True, exist_ok=True)
    (iteration_dir / "hypothesis.md").write_text(hypothesis, encoding="utf-8")
    (iteration_dir / "solve.py").write_text(code, encoding="utf-8")
    (iteration_dir / "predictions.json").write_text(
        json.dumps(
            {"test": [{"index": r.get("index"), "candidates": r.get("candidates") or [],
                       "error": r.get("error")} for r in evaluation.get("test", [])]},
            indent=2,
        ),
        encoding="utf-8",
    )

    state.setdefault("iterations", []).append(record)
    state["last_hypothesis_sha"] = hypothesis_sha
    state["last_code_sha"] = code_sha
    save_state(workspace, state)

    # Dead ends the solver recorded, paired with test examples that still carry a
    # single candidate: the exact shape of a forfeited second attempt.
    ruled_out_claims = [
        str(entry.get("claim"))
        for entry in load_invariants(workspace)
        if entry.get("mode") == "ruled_out" and entry.get("holds")
    ]
    unspent_candidates = [
        int(row.get("index"))
        for row in evaluation.get("test", [])
        if not row.get("error") and 0 < len(row.get("candidates") or []) < allowed_candidates
    ]
    fitting_rivals = sum(1 for entry in load_rivals(workspace) if entry.get("fits_training"))
    # A solver that hedged on some test examples and not others almost certainly
    # meant to hedge on all of them — the alternative path may have returned
    # early on a bug. "You chose not to hedge" and "your hedge silently failed"
    # are different situations and must stay distinguishable.
    partial_hedge = bool(unspent_candidates) and any(
        len(row.get("candidates") or []) >= allowed_candidates
        for row in evaluation.get("test", [])
        if not row.get("error")
    )

    report = format_submission_report(
        iteration=iteration,
        max_iterations=max_iterations,
        evaluation=evaluation,
        train_samples=train_samples,
        hardcoding_findings=findings,
        best_effort_active=best_effort_active(state, used=iteration),
        generalization_signals=signals,
        unspent_candidates=unspent_candidates,
        ruled_out_claims=ruled_out_claims,
        unhedged_rivals=rivals,
        fitting_rivals=fitting_rivals,
        partial_hedge=partial_hedge,
    )
    (iteration_dir / "report.txt").write_text(report, encoding="utf-8")

    append_event(
        workspace,
        {
            "command": "submit",
            "iteration": iteration,
            "all_train_correct": record["all_train_correct"],
            "train_correct": record["train_correct"],
        },
    )
    return report, 0


def _revalidate_current_solution(
    workspace: Path, state: dict[str, Any], last: dict[str, Any]
) -> dict[str, Any] | None:
    """Score ``solution/solve.py`` as it stands, for acceptance.

    Returns None when the file is unchanged since the last submission (nothing
    to do) or when re-running it would weaken the result — a regression, an
    error, or a rule that no longer reproduces training. In those cases
    acceptance falls back to the recorded submission, which is the artifact the
    solver actually stood behind.
    """
    code = _read_text(workspace, CODE_PATH)
    if not code.strip() or _sha(code) == str(last.get("code_sha") or ""):
        return None

    task = load_task(workspace)
    train_samples = task.get("train") or []
    evaluation = run_solution_isolated(
        code,
        train_samples,
        task.get("test") or [],
        max_candidates=int(state.get("max_test_predictions") or 2),
        timeout_seconds=float(state.get("solve_timeout_s") or 60.0),
    )
    if evaluation.get("status") != "ok":
        return None
    # Never let acceptance silently take a worse rule than the one submitted.
    if int(evaluation.get("train_correct") or 0) < int(last.get("train_correct") or 0):
        return None

    return {
        "test": [
            {"index": row.get("index"), "candidates": row.get("candidates") or [],
             "error": row.get("error")}
            for row in evaluation.get("test", [])
        ],
        "record": {
            "code_sha": _sha(code),
            "code_chars": len(code),
            "train_correct": int(evaluation.get("train_correct") or 0),
            "train_total": int(evaluation.get("train_total") or len(train_samples)),
            "train_pixel_accuracy": float(evaluation.get("train_pixel_accuracy") or 0.0),
            "all_train_correct": bool(evaluation.get("all_train_correct")),
            "revalidated_at_accept": True,
        },
    }


_DECISION_RE = re.compile(r"^\s*DECISION\s*:\s*(ACCEPT|RETRY)\b", re.IGNORECASE | re.MULTILINE)
_CONFIDENCE_RE = re.compile(r"^\s*CONFIDENCE\s*:\s*([1-5])\b", re.IGNORECASE | re.MULTILINE)


def salvage_unaccepted(workspace: Path) -> dict[str, Any] | None:
    """Score a train-perfect submission that was never accepted.

    A run killed by the wall clock writes no ``final.json``, so it scores 0.00
    even with a train-perfect submission recorded in its ledger.

    This is deliberately narrower than :func:`cmd_accept`. It requires a last
    iteration that ran and reproduced every training pair, so it can never
    manufacture a score the gate would have refused; and it does **not** require
    the audit, because a solver that was interrupted cannot be asked for one.
    The record says plainly who accepted: ``accepted_by: harness-salvage``.

    Returns the salvaged ``final`` payload, or ``None`` if there was nothing to
    salvage.
    """
    state = load_state(workspace)
    if state.get("accepted"):
        return None
    iterations = state.get("iterations") or []
    if not iterations:
        return None
    last = iterations[-1]
    if last.get("status") != "ok" or not last.get("all_train_correct"):
        return None

    iteration_dir = workspace / STATE_DIR / "iterations" / str(last.get("iteration"))
    predictions_path = iteration_dir / "predictions.json"
    if not predictions_path.is_file():
        return None
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))

    final = {
        "task_id": state.get("task_id"),
        "accepted_at": _now(),
        "accepted_by": "harness-salvage",
        "salvaged": True,
        "iteration": last.get("iteration"),
        "iterations_used": len(iterations),
        "max_iterations": state.get("max_iterations"),
        "all_train_correct": True,
        "train_correct": last.get("train_correct"),
        "train_total": last.get("train_total"),
        "train_pixel_accuracy": last.get("train_pixel_accuracy"),
        "best_effort": False,
        "confidence": None,
        "hypothesis": (iteration_dir / "hypothesis.md").read_text(encoding="utf-8")
        if (iteration_dir / "hypothesis.md").is_file() else "",
        "code": (iteration_dir / "solve.py").read_text(encoding="utf-8")
        if (iteration_dir / "solve.py").is_file() else "",
        "revalidated_at_accept": False,
        "audit": "",
        "test": predictions.get("test") or [],
        "hardcoding_findings": last.get("hardcoding_findings") or [],
        "verified_invariants": load_invariants(workspace),
    }
    (workspace / STATE_DIR / FINAL_FILE).write_text(json.dumps(final, indent=2), encoding="utf-8")
    state["accepted"] = {
        "iteration": last.get("iteration"),
        "at": final["accepted_at"],
        "best_effort": False,
        "confidence": None,
        "salvaged": True,
    }
    save_state(workspace, state)
    return final


def cmd_accept(workspace: Path) -> tuple[str, int]:
    state = load_state(workspace)
    if state.get("accepted"):
        return "This run was already accepted. Nothing more to do — stop here.", 0

    iterations = state.get("iterations") or []
    if not iterations:
        raise GateError("Nothing to accept: no submission has been made yet.")

    last = iterations[-1]
    if last.get("status") != "ok":
        raise GateError(
            f"Iteration {last.get('iteration')} failed to run, so there is nothing to accept. "
            "Fix solve.py and submit again."
        )

    lenient = best_effort_active(state)
    if not last.get("all_train_correct") and not lenient:
        raise GateError(
            f"Iteration {last.get('iteration')} covers "
            f"{last.get('train_correct')}/{last.get('train_total')} training examples. "
            "A rule that cannot reproduce the examples you can check is not a rule you can trust "
            "on the ones you cannot. Keep iterating — the train-100% requirement is lifted only "
            "over the last "
            f"{state.get('best_effort_iterations')} iteration(s) of the budget."
        )

    audit = _read_text(workspace, AUDIT_PATH).strip()
    if not audit:
        raise GateError(
            f"{AUDIT_PATH} is missing. Passing training is not evidence of generalization — "
            "write the audit (CONFIDENCE / DECISION / REASONS) before accepting."
        )
    decision_match = _DECISION_RE.search(audit)
    if not decision_match:
        raise GateError(
            f"{AUDIT_PATH} has no `DECISION: ACCEPT` or `DECISION: RETRY` line. "
            "The audit needs an explicit verdict."
        )
    if decision_match.group(1).upper() == "RETRY":
        raise GateError(
            "Your own audit says DECISION: RETRY. Act on it — the concerns you just wrote down "
            "are the most informative signal you have. Revise and submit again."
        )

    confidence_match = _CONFIDENCE_RE.search(audit)
    iteration_dir = workspace / STATE_DIR / "iterations" / str(last.get("iteration"))
    predictions = json.loads((iteration_dir / "predictions.json").read_text(encoding="utf-8"))

    # Re-run the solution as it stands now, so adding a second candidate after a
    # train-perfect submission does not cost an iteration — a solver must never
    # be charged budget for recording a hedge the harness itself asked for.
    revalidated = _revalidate_current_solution(workspace, state, last)
    if revalidated is not None:
        # The new code is what will be accepted, so the coupling `submit`
        # enforces has to hold here too — otherwise acceptance quietly pairs new
        # code with a stale hypothesis, the exact mismatch the submit rule
        # exists to prevent, slipped in through the door that makes hedging free.
        # Checked at this point only: when the re-run regressed or failed,
        # the submitted artifact stands and there is nothing to describe.
        if _sha(_read_text(workspace, HYPOTHESIS_PATH).strip()) == str(
            last.get("hypothesis_sha") or ""
        ):
            raise GateError(
                f"{CODE_PATH} has changed since iteration {last.get('iteration')} but "
                f"{HYPOTHESIS_PATH} has not. Acceptance re-runs the current code, so the "
                "hypothesis has to describe it — say what changed and accept again. This "
                "costs no iteration."
            )
        predictions = {"test": revalidated["test"]}
        last = {**last, **revalidated["record"]}

    final = {
        "task_id": state.get("task_id"),
        "accepted_at": _now(),
        "iteration": last.get("iteration"),
        "iterations_used": len(iterations),
        "max_iterations": state.get("max_iterations"),
        "all_train_correct": last.get("all_train_correct"),
        "train_correct": last.get("train_correct"),
        "train_total": last.get("train_total"),
        "train_pixel_accuracy": last.get("train_pixel_accuracy"),
        "best_effort": bool(lenient and not last.get("all_train_correct")),
        "confidence": int(confidence_match.group(1)) if confidence_match else None,
        "hypothesis": _read_text(workspace, HYPOTHESIS_PATH).strip()
        or (iteration_dir / "hypothesis.md").read_text(encoding="utf-8"),
        "code": (
            _read_text(workspace, CODE_PATH)
            if last.get("revalidated_at_accept")
            else (iteration_dir / "solve.py").read_text(encoding="utf-8")
        ),
        "revalidated_at_accept": bool(last.get("revalidated_at_accept")),
        "audit": audit,
        "test": predictions.get("test") or [],
        "hardcoding_findings": last.get("hardcoding_findings") or [],
        "verified_invariants": load_invariants(workspace),
    }
    (workspace / STATE_DIR / FINAL_FILE).write_text(json.dumps(final, indent=2), encoding="utf-8")

    state["accepted"] = {
        "iteration": last.get("iteration"),
        "at": final["accepted_at"],
        "best_effort": final["best_effort"],
        "confidence": final["confidence"],
    }
    save_state(workspace, state)
    append_event(workspace, {"command": "accept", "iteration": last.get("iteration")})

    lines = [
        "RUN COMPLETE — solution accepted.",
        f"  iteration      : {final['iteration']} of {final['max_iterations']}",
        f"  training       : {final['train_correct']}/{final['train_total']}"
        + ("  (best-effort acceptance)" if final["best_effort"] else ""),
        f"  confidence     : {final['confidence'] if final['confidence'] is not None else 'unstated'}",
        f"  test candidates: "
        + ", ".join(
            f"test {row.get('index')}: {len(row.get('candidates') or [])}" for row in final["test"]
        ),
        "",
        "Your predictions are recorded. Stop now — do not submit again, and do not "
        "keep working on this puzzle.",
    ]
    return "\n".join(lines), 0


# ── entry point ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gate.py",
        description="ccarc CC-harness verification gate.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    status = sub.add_parser(
        "status", help="Print the distilled research state (use this after a compaction)."
    )
    status.add_argument(
        "--brief",
        action="store_true",
        help="Iteration history and executed ledger only — no hypothesis dump, no NOTES tail.",
    )
    sub.add_parser("submit", help="Record a formal iteration: hypothesis + solve() vs. the training pairs.")
    sub.add_parser("accept", help="Finalize the run using the last submission and solution/audit.md.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace: Path | None = None
    try:
        workspace = find_workspace()
        if args.command == "status":
            report, code = cmd_status(workspace, brief=bool(getattr(args, "brief", False)))
        else:
            report, code = {"submit": cmd_submit, "accept": cmd_accept}[args.command](workspace)
    except GateError as exc:
        # Record refusals: the event log is the only place that says which
        # precondition was tripped, and by which command.
        if workspace is not None:
            try:
                append_event(
                    workspace,
                    {"command": "refused", "attempted": args.command, "reason": str(exc)},
                )
            except OSError:
                pass
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface harness bugs to the agent, don't hide them
        print(f"GATE ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
