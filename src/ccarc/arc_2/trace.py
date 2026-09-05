"""Reconstruct what a solver agent actually did, from workspace artifacts alone.

A score says whether a run worked; a trace says how it got there, and
specifically whether the doctrine was followed or merely satisfied.

The headline number is **verification density** — exploration scripts and
recorded invariants per formal iteration. An agent that submits repeatedly
having verified nothing is not doing code-as-verification no matter what its
hypothesis says, and that shows up here as a number.

Density measures **compliance with the doctrine, not quality of reasoning**.
It does not predict correctness, and a high number is not evidence a run went
well. It is the right instrument for one job — catching a run that guessed
first and verified afterwards — and the wrong thing to optimise.

Everything is read from the workspace, so a trace can be taken from a finished
run, an abandoned one, or one still in progress.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .gate import EVENTS_FILE, INVARIANTS_FILE, STATE_DIR, load_invariants


#: A recovered condition expression is evidence, but a 400-character
#: comprehension is evidence nobody reads. The long
#: `all((... for s in train_samples ...))` unparses are close to unreadable,
#: and truncating them loses nothing.
EXPRESSION_CHARS = 160


def _short_expression(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= EXPRESSION_CHARS else text[: EXPRESSION_CHARS - 1] + "…"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


#: Shipped with the workspace, not written by the agent. Counting it would
#: inflate verification density in every trace — including a run that explored
#: nothing at all.
_HARNESS_OWNED_EXPLORE_FILES = {"arc.py"}


def _explore_scripts(root: Path) -> list[dict[str, Any]]:
    """Exploration scripts the agent actually wrote."""
    explore_dir = root / "explore"
    if not explore_dir.is_dir():
        return []
    return sorted(
        (
            {
                "name": path.name,
                "lines": len(path.read_text(encoding="utf-8", errors="replace").splitlines()),
            }
            for path in explore_dir.glob("*.py")
            if path.name not in _HARNESS_OWNED_EXPLORE_FILES
        ),
        key=lambda script: script["name"],
    )


def _parse_time(value: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def collect_trace(workspace_root: Path | str) -> dict[str, Any]:
    """Everything knowable about a run from its workspace."""
    root = Path(workspace_root).resolve()
    if (root / "workspace").is_dir() and not (root / STATE_DIR).is_dir():
        root = root / "workspace"  # a run directory was passed instead

    state = _read_json(root / STATE_DIR / "state.json") or {}
    final = _read_json(root / STATE_DIR / "final.json")
    events = _read_jsonl(root / STATE_DIR / EVENTS_FILE)
    invariants = load_invariants(root) if (root / STATE_DIR / INVARIANTS_FILE).is_file() else []

    scripts = _explore_scripts(root)

    iterations = state.get("iterations") or []
    refusals = [e for e in events if e.get("command") == "refused"]

    started = _parse_time(state.get("created_at"))
    hypothesis_history = []
    for record in iterations:
        stored = root / STATE_DIR / "iterations" / str(record.get("iteration")) / "hypothesis.md"
        hypothesis_history.append(
            {
                "iteration": record.get("iteration"),
                "chars": record.get("hypothesis_chars"),
                "sha": record.get("hypothesis_sha"),
                "text": stored.read_text(encoding="utf-8") if stored.is_file() else "",
            }
        )

    per_iteration = max(1, len(iterations))
    return {
        "workspace": str(root),
        "task_id": state.get("task_id"),
        "started_at": state.get("created_at"),
        "max_iterations": state.get("max_iterations"),
        "iterations": iterations,
        "accepted": state.get("accepted"),
        "final": final,
        "invariants": invariants,
        # A refute() entry that holds is a *ruled-out hypothesis*, not an
        # established fact. Ruled-out entries are counted separately, in both
        # directions, so a run with five dead ends and no verified facts does
        # not read as one with five established facts.
        "invariants_held": sum(
            1 for entry in invariants
            if entry.get("holds") and entry.get("mode") not in {"ruled_out", "sweep"}
        ),
        "invariants_refuted": sum(
            1 for entry in invariants
            if not entry.get("holds") and entry.get("mode") not in {"ruled_out", "sweep"}
        ),
        "sweeps": sum(1 for entry in invariants if entry.get("mode") == "sweep"),
        "sweeps_undecided": sum(
            1 for entry in invariants
            if entry.get("mode") == "sweep" and not entry.get("decisive")
        ),
        "invariants_ruled_out": sum(
            1 for entry in invariants
            if entry.get("holds") and entry.get("mode") == "ruled_out"
        ),
        # Recorded from a constant condition: an assertion that measured nothing.
        # A non-zero count means the ledger is partly self-poisoned.
        "invariants_unmeasured": sum(1 for entry in invariants if entry.get("literal")),
        # Recorded where the call site could not be read (a -c one-liner or a
        # heredoc), so the entry carries no evidence of what was executed.
        "invariants_unsourced": sum(1 for entry in invariants if entry.get("unsourced")),
        "explore_scripts": scripts,
        "events": events,
        "refusals": refusals,
        "hypothesis_history": hypothesis_history,
        "notes_chars": len((root / "NOTES.md").read_text(encoding="utf-8")) if (root / "NOTES.md").is_file() else 0,
        "audit": (root / "solution" / "audit.md").read_text(encoding="utf-8")
        if (root / "solution" / "audit.md").is_file()
        else "",
        "density": {
            "scripts_per_iteration": round(len(scripts) / per_iteration, 2),
            "invariants_per_iteration": round(len(invariants) / per_iteration, 2),
            "verified_before_first_submission": _verified_before_first_submission(invariants, iterations),
        },
        "elapsed_s": _elapsed(started, iterations),
    }


def _verified_before_first_submission(
    invariants: list[dict[str, Any]], iterations: list[dict[str, Any]]
) -> int | None:
    """How many facts were established before the agent first committed.

    A zero here is the clearest possible signal that the agent guessed first and
    verified afterwards — the exact failure mode the doctrine targets.
    """
    if not iterations:
        return len(invariants)
    first = _parse_time(iterations[0].get("at"))
    if first is None:
        return None
    counted = 0
    for entry in invariants:
        at = _parse_time(entry.get("at"))
        if at is not None and at <= first:
            counted += 1
    return counted


def _elapsed(started: datetime | None, iterations: list[dict[str, Any]]) -> float | None:
    if started is None or not iterations:
        return None
    last = _parse_time(iterations[-1].get("at"))
    if last is None:
        return None
    return round((last - started).total_seconds(), 1)


def format_trace(trace: dict[str, Any], *, verbose: bool = False) -> str:
    """Human-readable trace: totals first, then submissions and evidence."""
    lines: list[str] = []
    accepted = trace.get("accepted")
    lines.append(f"TASK {trace.get('task_id')}  ({trace.get('workspace')})")
    lines.append(
        f"  iterations {len(trace['iterations'])}/{trace.get('max_iterations')}"
        + (f"  accepted at #{accepted.get('iteration')}" if accepted else "  not accepted")
        + (f"  elapsed {trace['elapsed_s']}s" if trace.get("elapsed_s") else "")
    )

    density = trace["density"]
    lines.append("")
    lines.append("VERIFICATION DENSITY")
    lines.append(f"  exploration scripts     : {len(trace['explore_scripts'])}"
                 f"  ({density['scripts_per_iteration']} per iteration)")
    breakdown = f"{trace['invariants_held']} held, {trace['invariants_refuted']} refuted"
    if trace.get("invariants_ruled_out"):
        breakdown += f", {trace['invariants_ruled_out']} ruled out"
    if trace.get("sweeps"):
        breakdown += f", {trace['sweeps']} sweep(s)"
    lines.append(f"  recorded invariants     : {len(trace['invariants'])}"
                 f"  ({density['invariants_per_iteration']} per iteration)"
                 f"  [{breakdown}]")
    if trace.get("invariants_unmeasured"):
        lines.append(
            f"  UNMEASURED invariants   : {trace['invariants_unmeasured']}"
            "  <- constant conditions; assertions, not verifications"
        )
    if trace.get("invariants_unsourced"):
        lines.append(
            f"  UNSOURCED invariants    : {trace['invariants_unsourced']}"
            "  <- call site unreadable; no record of what ran"
        )
    lines.append(f"  verified before 1st sub.: {density['verified_before_first_submission']}")
    lines.append(f"  NOTES.md                : {trace['notes_chars']} chars")

    if trace["iterations"]:
        lines.append("")
        lines.append("SUBMISSIONS")
        for record in trace["iterations"]:
            mark = "PASS" if record.get("all_train_correct") else "fail"
            extra = []
            if record.get("error"):
                extra.append(record["error"].splitlines()[0][:60])
            if record.get("timed_out"):
                extra.append("TIMED OUT")
            if record.get("multi_candidate_on_train"):
                extra.append("multi-candidate on train")
            for finding in record.get("hardcoding_findings") or []:
                extra.append(f"overfit: {finding[:60]}")
            lines.append(
                f"  #{record.get('iteration')}  {mark}  "
                f"train {record.get('train_correct')}/{record.get('train_total')}  "
                f"pixel {float(record.get('train_pixel_accuracy') or 0.0):.3f}  "
                f"hyp {record.get('hypothesis_chars')}c  code {record.get('code_chars')}c  "
                f"{float(record.get('elapsed_s') or 0.0):.1f}s"
                + ("  | " + "; ".join(extra) if extra else "")
            )

    if trace["refusals"]:
        lines.append("")
        lines.append("GATE REFUSALS  (preconditions the agent tripped)")
        for event in trace["refusals"]:
            lines.append(f"  [{event.get('at')}] {event.get('attempted')}: {event.get('reason', '')[:120]}")
    else:
        lines.append("")
        lines.append("GATE REFUSALS: none recorded")

    if trace["explore_scripts"]:
        lines.append("")
        lines.append("EXPLORATION")
        for script in trace["explore_scripts"]:
            lines.append(f"  explore/{script['name']}  ({script['lines']} lines)")

    if trace["invariants"]:
        lines.append("")
        lines.append("INVARIANTS")
        for entry in trace["invariants"]:
            if entry.get("mode") == "sweep":
                mark = "SWEEP "
            elif entry.get("mode") == "ruled_out":
                mark = "KILLED" if entry.get("holds") else "OPEN  "
            else:
                mark = "OK    " if entry.get("holds") else "FAILED"
            lines.append(f"  [{mark}] {entry.get('claim')}   ({entry.get('source')})")
            if entry.get("mode") == "sweep":
                survivors = entry.get("survivors") or []
                killed = entry.get("killed") or []
                lines.append(
                    f"         {len(survivors) + len(killed)} tested, {len(survivors)} survive: "
                    f"{', '.join(survivors) or 'none'}"
                )
            if entry.get("expression"):
                lines.append(f"         {_short_expression(entry['expression'])}")
            if entry.get("measured"):
                lines.append(f"         measured: {entry['measured']}")
            if entry.get("literal"):
                lines.append("         ^ NOT MEASURED — constant condition")
            if entry.get("unsourced"):
                lines.append("         ^ NO EVIDENCE — call site could not be read")
            superseded = entry.get("supersedes")
            if isinstance(superseded, dict) and superseded.get("claim"):
                if superseded["claim"] != entry.get("claim"):
                    lines.append(
                        f"         replaced [{superseded.get('verdict', '?')}]: "
                        f"{superseded['claim']}"
                    )
                else:
                    lines.append(
                        f"         ^ verdict changed from [{superseded.get('verdict', '?')}]"
                    )

    if verbose and trace.get("hypothesis_history"):
        lines.append("")
        lines.append("HYPOTHESIS EVOLUTION")
        for entry in trace["hypothesis_history"]:
            lines.append(f"  --- iteration {entry['iteration']} ({entry['chars']} chars) ---")
            lines.append(entry["text"].strip())

    if verbose and trace.get("audit"):
        lines.append("")
        lines.append("AUDIT")
        lines.append(trace["audit"].strip())

    return "\n".join(lines)
