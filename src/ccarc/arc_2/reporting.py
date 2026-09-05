"""Token-efficient rendering of gate output.

The gate is the only channel the harness has for speaking to the solver agent:
Claude Code owns the agent loop, so a reflection prompt has to arrive as *tool
output* instead. Everything an orchestrator would push into the conversation —
the failure diff, the reflection schema, the generalization audit, the
best-effort switch — is emitted here, appended to the result of
``python gate.py …``.

Formatting follows the same rule the solver is held to: spend tokens on
information the model cannot cheaply recompute, and nothing else.
"""

from __future__ import annotations

from typing import Any

from .evaluate import Grid, grid_diff

#: Grids at or below this cell count are printed in full; larger ones are
#: summarised by their differences. A 20x20 grid is 400 cells.
FULL_GRID_CELL_LIMIT = 900

#: Tighter cap for a prediction whose shape is already wrong: the shape *is* the
#: finding, and dumping a 30x30 grid of noise underneath it buys nothing.
WRONG_SHAPE_CELL_LIMIT = 300

#: Failed examples that get their grids printed in full. Beyond this, the
#: difference listing carries the information at a fraction of the cost.
FULL_GRID_EXAMPLE_LIMIT = 3

#: Cap on per-example difference listings.
DIFF_CELL_LIMIT = 40

#: Test predictions the solver generated itself. It can reload them at will
#: (`arc.load_solution()`, or the archived predictions.json), so the gate prints
#: them in full only while they are cheap. Dumping two 30x30 grids on each of
#: eight iterations is ~16k tokens of material the doctrine tells the solver
#: never to print.
TEST_CANDIDATE_CELL_LIMIT = 400


#: A recovered condition expression is evidence, but a 400-character
#: comprehension is evidence nobody reads. The long
#: `all((... for s in train_samples ...))` unparses are close to unreadable,
#: and truncating them loses nothing.
EXPRESSION_CHARS = 160


def _short_expression(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= EXPRESSION_CHARS else text[: EXPRESSION_CHARS - 1] + "…"


# ── grid rendering ───────────────────────────────────────────────────────────

def render_grid(grid: Grid | None) -> str:
    """One row per line, one digit per cell — compact and spatially readable."""
    if not grid:
        return "<empty>"
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def grid_shape(grid: Grid | None) -> str:
    if not grid:
        return "0x0"
    return f"{len(grid)}x{len(grid[0])}"


def render_grid_block(title: str, grid: Grid | None, *, limit: int = FULL_GRID_CELL_LIMIT) -> str:
    """Labelled grid, elided when it is too large to be worth the tokens."""
    if not grid:
        return f"{title}: <none>"
    cells = len(grid) * len(grid[0])
    header = f"{title} ({grid_shape(grid)}):"
    if cells > limit:
        return f"{header} <{cells} cells — not inlined; inspect it with arc.show()>"
    return f"{header}\n{render_grid(grid)}"


# ── directives (reflection prompts, retargeted at the filesystem) ────────────

TRAIN_FAILURE_DIRECTIVE = """\
NEXT — reflect before your next submission. Append a dated entry to NOTES.md covering:

1. Diff analysis — for each failed example, what differs between prediction and
   ground truth, and where the logic went wrong.
2. Failure root cause — wrong assumption, coding bug, or unhandled edge case?
   "I assumed X, but actually Y" beats "my code was wrong".
3. Verified rules — invariants that hold regardless of your current hypothesis.
   Establish each one by running arc.verify() in an exploration script; do not
   assert them from memory.
4. Dead ends — hypotheses this attempt rules out, so you do not revisit them.
5. Next experiment — the specific check you will run, and what each outcome
   would tell you.

Then go back to exploration. Run the experiment before writing the next
hypothesis: a submission that is not preceded by a new verified fact is a guess.
"""

GENERALIZATION_AUDIT_DIRECTIVE = """\
NEXT — training is solved, generalization is not. Nothing above tells you the
rule is right; it tells you the rule is consistent with the examples you were
allowed to see.

Write solution/audit.md with exactly these sections:

  CONFIDENCE: <1-5>
  DECISION: ACCEPT | RETRY
  REASONS:
  <your analysis>

Anchor the confidence number, because an unanchored one is worthless: an
unanchored audit gravitates to 4 or 5 and never says RETRY, including when it
is wrong. So:

  5  every situation the test input requires is witnessed in a training pair,
     and you have executed a check saying so.
  4  as above, except one such situation is unwitnessed and you have hedged it
     with your second candidate.
  3  a situation the test needs is unwitnessed and you have NOT hedged it, or
     you hedged a different axis than the one you are unsure about.
  2  the rule fits training and you cannot say why it should generalise.
  1  you are submitting because the budget is running out.

A 3 or below is a reason to spend another iteration, not a reason to write a
longer justification.

Cover, in REASONS:
- Assumption audit. Does the code lean on any training-only coincidence — exact
  grid sizes, absolute coordinates, specific counts, colours with no semantic
  role? For every constant in solve.py, say why it is role-based rather than
  incidental.
- Prediction plausibility. Do the test predictions above follow from the same
  rule that explains all training pairs? Name any artefact that smells like
  overfitting: stray pixels, missing structure, wrong recolour, bad alignment.
- Verification. Back the audit with executed checks where you can — for
  instance, confirm the predicted test outputs satisfy the invariants you
  verified on the training outputs.
- Candidate budget. ARC-AGI-2 scores two predictions per test example, so a
  single candidate forfeits one for free. If you considered an alternative
  reading and ruled it out, examine *how*. Killing it with a training pair it
  fails is a proof. Killing it by extending a regularity that merely holds
  across the training outputs to the test input is an inductive leap wearing a
  proof's clothes — the invariant is real, but nothing established that it
  survives out of sample. When an alternative reproduces every training pair and
  died only to a leap like that, emit it as the second candidate. That is
  exactly the case the second attempt exists for, and it costs you nothing.

Then run: python gate.py accept
DECISION: RETRY refuses acceptance and returns you to the loop with an iteration
spent — which is the correct outcome when the audit finds a real hole.
"""

BEST_EFFORT_DIRECTIVE = """\
STRATEGY SHIFT — the training-accuracy requirement is lifted.

Most of your iteration budget is gone without a train-perfect rule. That happens
when the rule is unusually subtle, or when the training pairs are not quite
self-consistent under any simple rule.

`gate.py accept` will now accept your best submission regardless of training
score. Stop optimising for training coverage and optimise for the test outputs
being right: pick the hypothesis with the strongest evidence behind it, and use
your second candidate on the genuine alternative reading rather than on a
throwaway variant.
"""


# ── submission report ────────────────────────────────────────────────────────

def _unhedged_rival_prompt(rivals: list[dict[str, Any]]) -> str:
    """The strongest form of the second-attempt prompt: a measured fact.

    Where :func:`_unspent_candidate_prompt` asks the solver to re-examine its own
    reasoning, this states an executed result — the rival was run, it reproduces
    every training pair, and it disagrees with the submission on a specific test
    input. Nothing here rests on the solver's judgement about its own inference.
    """
    lines = [
        "UNHEDGED RIVAL — you registered an alternative reading that reproduces every "
        "training pair and predicts something different:",
    ]
    for entry in rivals[:5]:
        where = ", ".join(f"test {index}" for index in entry.get("test_indices") or [])
        lines.append(f"  - {entry.get('name')}  (differs on {where})")
    if len(rivals) > 5:
        lines.append(f"  … and {len(rivals) - 5} more")
    lines.append(
        "Training cannot separate these from your reading — that is measured, not inferred. "
        "ARC-AGI-2 scores two attempts per test example, so unless you can point at evidence "
        "that rules a rival out, make it your second candidate. Leaving the slot empty "
        "discards a free attempt on a reading the data does not contradict."
    )
    return "\n".join(lines)


def _agreeing_rival_note(fitting_rivals: int, unspent: int = 1) -> str:
    """Say when the solver's live rivals simply do not reach the open slot.

    When a registered rival fits training but agrees with the submission on the
    unhedged examples, say so explicitly: the open candidate slot is not for
    that rival, and silence about it is indistinguishable from having ignored
    the work.
    """
    if fitting_rivals <= 0:
        return ""
    if fitting_rivals == 1:
        subject = "You have 1 registered rival reading that reproduces every training pair, "
        subject += "and it does not disagree"
    else:
        subject = (
            f"You have {fitting_rivals} registered rival readings that reproduce every "
            "training pair, and none of them disagrees"
        )
    where = "the examples above" if unspent > 1 else "the example above"
    return (
        f"{subject} with you on {where} — so they are not what the open slot is for. "
        "Whatever belongs there is a reading you have not named yet."
    )


def _unspent_candidate_prompt(
    unspent: list[int], ruled_out: list[str], fitting_rivals: int = 0
) -> str:
    """Confront the solver with the alternatives it killed and did not hedge on.

    Killing a rival reading by extending a training-only regularity to the test
    input can bury the reading that was actually right, and the unspent second
    candidate is exactly the hedge against that.

    This fires only when both halves of that situation are present: recorded
    dead ends, and an unspent second attempt.
    """
    which = ", ".join(f"test {index}" for index in unspent)
    lines = [
        f"UNSPENT SECOND ATTEMPT — {which} carries one candidate. You recorded "
        f"{len(ruled_out)} ruled-out hypothesis/hypotheses along the way:",
    ]
    agreeing = _agreeing_rival_note(fitting_rivals, len(unspent))
    for claim in ruled_out[:6]:
        lines.append(f"  - {claim}")
    if len(ruled_out) > 6:
        lines.append(f"  … and {len(ruled_out) - 6} more")
    lines.append(
        # Not everything a solver refutes is a reading of the puzzle. Some
        # refuted claims are claims about what the evidence can separate, and
        # listing those under "rivals you ruled out, reconsider each" would
        # invert their meaning.
        "Some of these may be claims about the evidence rather than alternative readings of "
        "the transformation; skip those. For the ones that do name an alternative reading: "
        "check how each one died. If a training pair it fails killed it, that is a proof and "
        "the slot should stay empty. If it reproduced every training pair and died to a "
        "regularity you observed on the training outputs and then applied to the test input, "
        "that is an inductive leap — the invariant is real, but nothing established that it "
        "holds out of sample."
    )
    lines.append(
        "Do not assume the answer is in the list above. Those are the rivals you thought to "
        "write down, and they may all be proofs. The leap is often somewhere you never framed "
        "as a rival at all: enumerate the situation types your rule has to handle on the test "
        "input, and check that each one is actually witnessed in a training pair. A case the "
        "test needs and training never shows is an unhedged assumption regardless of how "
        "confident the rule feels. ARC-AGI-2 scores two attempts per test example; an unspent "
        "one is a free attempt discarded."
    )
    if agreeing:
        lines.append(agreeing)
    return "\n".join(lines)


def _colour_histogram(grid: Grid | None) -> str:
    """`{colour: count}`, most frequent first — cheap plausibility evidence."""
    if not grid:
        return "{}"
    counts: dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[int(cell)] = counts.get(int(cell), 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return "{" + ", ".join(f"{colour}:{count}" for colour, count in ordered) + "}"


def format_submission_report(
    *,
    iteration: int,
    max_iterations: int,
    evaluation: dict[str, Any],
    train_samples: list[dict[str, Any]],
    hardcoding_findings: list[str],
    best_effort_active: bool,
    generalization_signals: list[str] | None = None,
    unspent_candidates: list[int] | None = None,
    ruled_out_claims: list[str] | None = None,
    unhedged_rivals: list[dict[str, Any]] | None = None,
    fitting_rivals: int = 0,
    partial_hedge: bool = False,
) -> str:
    """Render one formal iteration's outcome plus the directive that follows."""
    lines: list[str] = []
    remaining = max_iterations - iteration

    if evaluation.get("status") != "ok":
        lines.append(f"ITERATION {iteration}/{max_iterations} — SUBMISSION FAILED TO RUN")
        lines.append("")
        lines.append(str(evaluation.get("error") or "unknown error").rstrip())
        if evaluation.get("stdout"):
            lines.append("")
            lines.append("--- stdout/stderr from your code ---")
            lines.append(str(evaluation["stdout"]).rstrip())
        lines.append("")
        lines.append(f"Iterations remaining: {remaining}")
        lines.append("")
        lines.append(
            "NEXT — this iteration is spent. Reproduce the failure in an exploration "
            "script (`python explore/<name>.py`) and fix it there before resubmitting; "
            "the gate is not a debugger."
        )
        return "\n".join(lines)

    correct = int(evaluation.get("train_correct") or 0)
    total = int(evaluation.get("train_total") or 0)
    pixel = float(evaluation.get("train_pixel_accuracy") or 0.0)
    passed = bool(evaluation.get("all_train_correct"))

    lines.append(
        f"ITERATION {iteration}/{max_iterations} — TRAIN {correct}/{total}"
        f" (mean pixel accuracy {pixel:.3f})"
    )
    lines.append("")

    if evaluation.get("multi_candidate_on_train"):
        lines.append(
            "WARNING: solve() returned multiple candidates for a training input. Training "
            "scores the first candidate only. Ambiguity on an example whose output you can "
            "see means the rule is not yet understood — resolve it rather than hedging."
        )
        lines.append("")

    for finding in hardcoding_findings:
        lines.append(f"WARNING: {finding}")
    if hardcoding_findings:
        lines.append("")

    failures = [row for row in evaluation.get("train", []) if not row.get("correct")]
    if failures:
        lines.append(f"--- failed training examples ({len(failures)}/{total}) ---")
        for position, row in enumerate(failures):
            idx = int(row.get("index", -1))
            lines.append("")
            lines.append(f"[train {idx}]")
            if row.get("error"):
                lines.append(f"  raised: {row['error']}")
                continue
            expected = (train_samples[idx] or {}).get("output") if 0 <= idx < len(train_samples) else None
            predicted = row.get("predicted")
            diff = grid_diff(predicted, expected, limit=DIFF_CELL_LIMIT)
            lines.append(
                f"  shape expected {grid_shape(expected)}, got {grid_shape(predicted)}"
                f" | pixel accuracy {float(row.get('pixel_accuracy') or 0.0):.3f}"
            )
            if not diff["shape_match"]:
                lines.append("  SHAPE MISMATCH — the output-size rule is wrong.")
            else:
                bbox = diff.get("bbox")
                lines.append(
                    f"  {diff['num_diff']} differing cells"
                    + (f", bounded by rows {bbox[0]}-{bbox[2]}, cols {bbox[1]}-{bbox[3]}" if bbox else "")
                )
                if diff["cells"]:
                    listing = " ".join(
                        f"({r},{c}) want {want} got {got}" for r, c, want, got in diff["cells"]
                    )
                    lines.append(f"  diffs: {listing}")
                    if diff["truncated"]:
                        lines.append(f"  … {diff['num_diff'] - len(diff['cells'])} more differing cells")

            if position >= FULL_GRID_EXAMPLE_LIMIT:
                lines.append(
                    "  grids omitted (differences above carry the information; "
                    "use arc.check(solve) to see them all)"
                )
                continue
            predicted_limit = FULL_GRID_CELL_LIMIT if diff["shape_match"] else WRONG_SHAPE_CELL_LIMIT
            lines.append("  " + render_grid_block("expected", expected).replace("\n", "\n  "))
            lines.append(
                "  " + render_grid_block("predicted", predicted, limit=predicted_limit).replace("\n", "\n  ")
            )
        lines.append("")

    lines.append("--- test predictions ---")
    elided = False
    for row in evaluation.get("test", []):
        idx = row.get("index")
        if row.get("error"):
            lines.append(f"[test {idx}] raised: {row['error']}")
            continue
        candidates = row.get("candidates") or []
        shapes = ", ".join(grid_shape(c) for c in candidates) or "none"
        lines.append(f"[test {idx}] {len(candidates)} candidate(s): {shapes}")
        if len(candidates) > 1:
            # Two candidates can share a colour histogram and a shape while
            # differing in a couple of cells, which leaves the printed summary
            # looking identical. Say how far apart they actually are.
            spread = grid_diff(candidates[1], candidates[0])
            if spread["shape_match"]:
                lines.append(f"  candidates differ in {spread['num_diff']} cell(s)")
            else:
                lines.append("  candidates differ in shape")
        for cand_idx, candidate in enumerate(candidates, start=1):
            histogram = _colour_histogram(candidate)
            lines.append(f"  candidate {cand_idx}: {grid_shape(candidate)}  colours {histogram}")
            cells = len(candidate) * len(candidate[0]) if candidate else 0
            if cells <= TEST_CANDIDATE_CELL_LIMIT:
                lines.append("  " + render_grid(candidate).replace("\n", "\n  "))
            else:
                elided = True
    if elided:
        lines.append(
            f"  (grids over {TEST_CANDIDATE_CELL_LIMIT} cells not printed — you generated them, so "
            f"reload with arc.load_solution() or read "
            f".ccarc/iterations/{iteration}/predictions.json)"
        )

    for signal in generalization_signals or []:
        lines.append(f"  GENERALIZATION: {signal}")
    lines.append("")

    if evaluation.get("stdout"):
        lines.append("--- stdout/stderr from your code ---")
        lines.append(str(evaluation["stdout"]).rstrip())
        lines.append("")

    lines.append(f"Iterations remaining: {remaining}")
    lines.append("")

    if passed:
        lines.append("TRAINING PASSED (100%).")
        lines.append("")
        if unhedged_rivals:
            lines.append(_unhedged_rival_prompt(unhedged_rivals))
            lines.append("")
        elif unspent_candidates and ruled_out_claims:
            lines.append(_unspent_candidate_prompt(
                unspent_candidates, ruled_out_claims, fitting_rivals
            ))
            lines.append("")
        if partial_hedge:
            which = ", ".join(f"test {i}" for i in unspent_candidates or [])
            lines.append(
                # The asymmetry is worth flagging; the explanation is not the
                # harness's to guess — an alternative that genuinely agrees on
                # the unhedged example is indistinguishable from outside.
                f"PARTIAL HEDGE — solve() returned two candidates for at least one test input "
                f"and only one for {which}. Two readings that diverge on one example and agree "
                "on another is perfectly possible, so this may be exactly right. It is also "
                "what an alternative path returning early looks like from outside. Confirm "
                "which before you spend the audit."
            )
            lines.append("")
        lines.append(GENERALIZATION_AUDIT_DIRECTIVE.rstrip())
    else:
        if best_effort_active:
            lines.append(BEST_EFFORT_DIRECTIVE.rstrip())
            lines.append("")
        lines.append(TRAIN_FAILURE_DIRECTIVE.rstrip())

    return "\n".join(lines)


# ── status / resume report ───────────────────────────────────────────────────

def format_status(
    *,
    state: dict[str, Any],
    invariants: list[dict[str, Any]],
    hypothesis: str,
    notes_excerpt: str,
    rivals: list[dict[str, Any]] | None = None,
) -> str:
    """The distilled research state, printed on demand.

    Claude Code compacts on its own schedule, so the harness keeps the research
    state on disk instead of in the conversation and reprints it here: one
    command restores continuity after a compaction.
    """
    lines: list[str] = []
    iterations = state.get("iterations") or []
    max_iterations = int(state.get("max_iterations") or 0)
    used = len(iterations)

    lines.append(f"TASK {state.get('task_id', '?')} — iteration {used}/{max_iterations}")
    accepted = state.get("accepted")
    if accepted:
        lines.append(f"STATUS: accepted at iteration {accepted.get('iteration')} — the run is finished.")
    elif used >= max_iterations:
        lines.append("STATUS: iteration budget exhausted. `gate.py accept` on your best submission.")
    else:
        lines.append("STATUS: in progress.")
    lines.append("")

    if iterations:
        lines.append("--- submission history ---")
        for record in iterations:
            marker = "PASS" if record.get("all_train_correct") else "fail"
            detail = record.get("error") or ""
            lines.append(
                f"  #{record.get('iteration')}  {marker}  "
                f"train {record.get('train_correct')}/{record.get('train_total')}  "
                f"pixel {float(record.get('train_pixel_accuracy') or 0.0):.3f}"
                + (f"  [{detail.splitlines()[0][:80]}]" if detail else "")
            )
        lines.append("")

    if invariants:
        suspect = [entry for entry in invariants if entry.get("literal")]
        dead = sum(1 for e in invariants if e.get("mode") == "ruled_out" and e.get("holds"))
        swept = sum(1 for e in invariants if e.get("mode") == "sweep")
        held = len(invariants) - dead - swept
        parts = [f"{held} verified"]
        if dead:
            parts.append(f"{dead} ruled out")
        if swept:
            parts.append(f"{swept} sweep{'s' if swept > 1 else ''}")
        lines.append(f"--- executed ledger ({', '.join(parts)}) ---")
        for entry in invariants:
            # A refute() entry that *holds* means the hypothesis is dead, not
            # that it is true. Rendering it "[OK  ]" beside a claim worded as
            # the hypothesis inverts its meaning to anyone skimming — which,
            # after a compaction, is the only way this gets read.
            if entry.get("mode") == "sweep":
                mark = "SWEEP "
            elif entry.get("mode") == "ruled_out":
                mark = "KILLED" if entry.get("holds") else "OPEN  "
            else:
                mark = "OK    " if entry.get("holds") else "FAILED"
            source = entry.get("source") or "?"
            lines.append(f"  [{mark}] {entry.get('claim')}   ({source})")
            if entry.get("mode") == "sweep":
                survivors = entry.get("survivors") or []
                killed = entry.get("killed") or []
                lines.append(
                    f"         {len(survivors) + len(killed)} readings tested, "
                    f"{len(survivors)} survive: {', '.join(survivors) or 'none'}"
                )
                if len(survivors) > 1:
                    lines.append(
                        "         ^ more than one reading survives — this is a hedging "
                        "obligation, not a preference"
                    )
            # The expression is the evidence. A claim without one was recorded
            # from a context the parser could not read; a literal one measured
            # nothing at all.
            if entry.get("expression"):
                lines.append(f"         {_short_expression(entry['expression'])}")
            if entry.get("measured"):
                lines.append(f"         measured: {entry['measured']}")
            if entry.get("literal"):
                lines.append("         ^ NOT MEASURED — constant condition; re-verify or retract")
            if entry.get("unsourced"):
                lines.append(
                    "         ^ NO EVIDENCE — recorded from a -c/heredoc; nothing here says "
                    "what ran"
                )
            # A claim that replaced an earlier one carries what it displaced.
            # Post-compaction this is the only surviving record of the reading
            # you already ruled out — without it the dead branch looks unexplored.
            superseded = entry.get("supersedes")
            if isinstance(superseded, dict) and superseded.get("claim"):
                if superseded["claim"] != entry.get("claim"):
                    lines.append(
                        f"         replaced [{superseded.get('verdict', '?')}]: {superseded['claim']}"
                    )
                else:
                    lines.append(
                        f"         ^ verdict changed from [{superseded.get('verdict', '?')}] "
                        "— re-check whatever assumed the old one"
                    )
        if suspect:
            lines.append("")
            lines.append(
                f"  {len(suspect)} invariant(s) rest on a constant condition. They are assertions, "
                "not verifications — re-run them against the data or retract them before you rely "
                "on them."
            )
        lines.append("")
        if rivals:
            # The rival ledger is replayed alongside the invariant ledger:
            # after a compaction, status is the only surviving record of
            # either.
            lines.append(f"--- registered rivals ({len(rivals)}) ---")
            for entry in rivals:
                fit = (
                    "fits training"
                    if entry.get("fits_training")
                    else f"{entry.get('train_correct')}/{entry.get('train_total')} — ruled out"
                )
                lines.append(f"  [{fit}] {entry.get('name')}")
            lines.append("")
    else:
        lines.append(
            "--- verified invariants: none ---\n"
            "  Nothing about this puzzle has been established by execution yet. "
            "Use arc.verify() in an exploration script before forming a hypothesis.\n"
        )

    if hypothesis:
        lines.append("--- last submitted hypothesis ---")
        lines.append(hypothesis.strip())
        lines.append("")

    if notes_excerpt:
        lines.append("--- NOTES.md (tail) ---")
        lines.append(notes_excerpt.strip())
        lines.append("")

    return "\n".join(lines)
