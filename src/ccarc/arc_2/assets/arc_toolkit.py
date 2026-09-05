"""arc — observation and verification helpers for this puzzle workspace.

Import it the same way from anywhere in the workspace::

    from arc import train_samples, test_samples, verify, check, show, diff

That works from a script under ``explore/`` (``python explore/foo.py``), from a
one-liner at the workspace root (``python -c "from arc import ..."``), and from
a module invocation (``python -m explore.foo``). No ``sys.path`` boilerplate is
needed in any of them.

Design rule, and it is deliberate: **this module contains no transformation
primitives.** No rotate, no flood-fill, no connected components. Building those
is the reasoning work, and handing them over would change what is being
measured. What you get here is the ability to *look* and to *check*.

The important function is :func:`verify`. Anything you believe about this puzzle
should pass through it, because a claim that has been executed is worth more
than a claim that has been asserted — and it costs a fraction of the tokens.
Every call is appended to ``.ccarc/invariants.jsonl``, which survives context
compaction and is replayed by ``python gate.py status``.
"""

from __future__ import annotations

import inspect
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = [
    "WORKSPACE",
    "train_samples",
    "test_samples",
    "verify",
    "unreached",
    "refute",
    "sweep",
    "check",
    "load_solution",
    "solution_module",
    "explore_module",
    "show",
    "diff",
    "shape",
    "colors",
    "histogram",
    "png",
    "invariants",
]

Grid = list[list[int]]


def _find_workspace() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in [here, *here.parents]:
        if (candidate / ".ccarc" / "state.json").is_file():
            return candidate
    return here


#: Root of this run's workspace.
WORKSPACE: Path = _find_workspace()

_TASK = json.loads((WORKSPACE / "task" / "task.json").read_text(encoding="utf-8"))

#: Training pairs — ``{'input': grid, 'output': grid}``.
train_samples: list[dict[str, Grid]] = _TASK.get("train") or []

#: Test inputs — ``{'input': grid}``. Test outputs are not in this workspace.
test_samples: list[dict[str, Grid]] = _TASK.get("test") or []

#: ARC colour names, for readable reports.
COLOR_NAMES = {
    0: "black", 1: "blue", 2: "red", 3: "green", 4: "yellow",
    5: "gray", 6: "magenta", 7: "orange", 8: "lightblue", 9: "maroon",
}

_PALETTE = {
    0: (0, 0, 0), 1: (0, 116, 217), 2: (255, 65, 54), 3: (46, 204, 64), 4: (255, 220, 0),
    5: (170, 170, 170), 6: (240, 18, 190), 7: (255, 133, 27), 8: (127, 219, 255), 9: (135, 12, 37),
}


# ── verification ─────────────────────────────────────────────────────────────

def _caller_frame() -> Any:
    """The frame of the script that called into this module."""
    for frame in inspect.stack()[1:]:
        filename = frame.filename or ""
        if filename == __file__ or filename.startswith("<"):
            continue
        return frame
    return None


def _caller_source() -> str:
    """Best-effort workspace-relative path of the script that called verify()."""
    frame = _caller_frame()
    if frame is None:
        return "<unknown>"
    try:
        return str(Path(frame.filename).resolve().relative_to(WORKSPACE))
    except ValueError:
        return os.path.basename(frame.filename)


_AST_CACHE: dict[str, Any] = {}


def _parse_caller(filename: str) -> Any:
    key = f"{filename}:{os.path.getmtime(filename)}" if os.path.exists(filename) else filename
    if key not in _AST_CACHE:
        try:
            import ast

            _AST_CACHE[key] = ast.parse(Path(filename).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, ValueError):
            _AST_CACHE[key] = None
    return _AST_CACHE[key]


def _condition_evidence() -> tuple[str, bool, bool, bool]:
    """Recover the *expression* the caller passed as ``condition``.

    Returns ``(source_text, is_literal, is_opaque, is_unsourced)``. The ledger
    records what was actually executed, not just what was claimed — and a
    condition that is a compile-time constant is an assertion wearing a
    verification's clothes, which is exactly what the whole discipline exists to
    prevent.

    ``is_unsourced`` marks the case where the call site cannot be read at all:
    a ``python -c`` one-liner, a heredoc piped to stdin, a REPL. The entry then
    carries no expression, and the constant-condition warning cannot fire
    because it is derived from source the parser never reached. It is a
    separate flag so that "no evidence" and "good evidence" do not look alike.
    """
    import ast

    frame = _caller_frame()
    if frame is None:
        return "", False, False, True
    tree = _parse_caller(frame.filename)
    if tree is None:
        return "", False, False, True

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name not in {"verify", "refute"} or node.lineno != frame.lineno:
            continue

        condition = None
        if len(node.args) >= 2:
            condition = node.args[1]
        else:
            for keyword in node.keywords:
                if keyword.arg == "condition":
                    condition = keyword.value
                    break
        if condition is None:
            return "", False, False, False

        body = condition.body if isinstance(condition, ast.Lambda) else condition
        literal = isinstance(body, ast.Constant)
        # A bare name records "allok" as the evidence, which tells a future
        # context nothing about what was executed — the ledger's whole value is
        # that it says what ran.
        opaque = isinstance(body, ast.Name)
        # `True is (...)` and `... == False` slip past the constant check while
        # being the same anti-pattern: a boolean literal standing in for a
        # measurement.
        if isinstance(body, ast.Compare):
            operands = [body.left, *body.comparators]
            if any(isinstance(o, ast.Constant) and isinstance(o.value, bool) for o in operands):
                literal = True
        try:
            return ast.unparse(condition), literal, opaque, False
        except Exception:  # noqa: BLE001 - unparse is a convenience, not a contract
            return "", literal, opaque, False
    return "", False, False, False


def _render_evidence(value: Any, limit: int = 400) -> str:
    """Compact repr of what a check measured, for the ledger."""
    try:
        text = value if isinstance(value, str) else repr(value)
    except Exception:  # noqa: BLE001 - a bad __repr__ must not break a run
        return "<unrepresentable>"
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def sweep(
    question: str,
    outcomes: dict[str, bool | Callable[[], Any]],
    note: str | None = None,
    *,
    key: str | None = None,
) -> list[str]:
    """Record a whole tie-break sweep as one ledger entry.

    ``outcomes`` maps each candidate reading's name to whether it **survives**
    the evidence — a bool, or a callable returning one::

        survivors = sweep("which object is the odd one out?", {
            "the only one with a hole":       fits(has_hole),
            "the only asymmetric one":        fits(not_symmetric),
            "the only one in a unique colour": fits(unique_colour),
        })

    Returns the survivors, and prints the count. The point is what happens when
    there is more than one: a sweep that leaves several readings alive *is* the
    hedging obligation, discovered before any budget is spent, and it names the
    exact readings to register with :func:`rival`.

    Recording a sweep through :func:`refute` one candidate at a time produces a
    row of near-identical ledger lines, so most of the readings end up in prose
    instead of in the ledger — and prose does not survive a compaction. A sweep
    is a single finding and belongs in a single entry.
    """
    resolved: dict[str, bool] = {}
    details: dict[str, str] = {}
    for name, outcome in outcomes.items():
        # A bare bool loses the score that produced it, and without the score
        # the entry does not explain itself after a compaction. An
        # ``(outcome, detail)`` tuple keeps both.
        detail: Any = None
        if isinstance(outcome, tuple) and len(outcome) == 2:
            outcome, detail = outcome
        if callable(outcome):
            try:
                resolved[str(name)] = bool(outcome())
            except Exception:  # noqa: BLE001 - a candidate that blows up did not survive
                resolved[str(name)] = False
        else:
            resolved[str(name)] = bool(outcome)
        if detail is not None:
            details[str(name)] = _render_evidence(detail, limit=80)

    survivors = [name for name, alive in resolved.items() if alive]
    killed = [name for name, alive in resolved.items() if not alive]

    entry = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claim": str(question),
        # A sweep always holds: it ran and produced a finding. Decisiveness is
        # a separate field, because a sweep that leaves every reading alive is
        # a hedging signal, not a failed check.
        "holds": True,
        "decisive": len(survivors) == 1,
        "mode": "sweep",
        "survivors": survivors,
        "killed": killed,
        "details": details,
        "source": _caller_source(),
        "key": str(key) if key else str(question),
    }
    if note:
        entry["note"] = str(note)

    ledger = WORKSPACE / ".ccarc" / "invariants.jsonl"
    try:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:
        pass

    print(f"[SWEEP   ] {question}  — {len(resolved)} readings tested, {len(survivors)} survive")
    def _label(name: str) -> str:
        return f"{name}  [{details[name]}]" if name in details else name

    for name in survivors:
        print(f"           alive: {_label(name)}")
    if killed:
        shown = killed[:6]
        for name in shown:
            print(f"           dead : {_label(name)}")
        if len(killed) > len(shown):
            print(f"           dead : ... and {len(killed) - len(shown)} more")
    if len(survivors) > 1:
        print(
            f"           ^ {len(survivors)} readings survive the evidence you have. That is a\n"
            "             hedging obligation, not a tie to break by preference: run each\n"
            "             through arc.rival(name, fn) and see which of them diverge on the\n"
            "             test input. Only a training pair one of them fails is a proof."
        )
    elif not survivors:
        print(
            "           ^ nothing survived. Either the question is wrong or the candidate\n"
            "             set is missing the right reading — do not pick the least-bad one."
        )
    return survivors


def refute(claim: str, condition: bool | Callable[[], Any] = False, note: str | None = None,
           *, evidence: Any = None, key: str | None = None, retract: bool = False) -> bool:
    """Record a hypothesis as *ruled out* by an executed check.

    ``verify()`` has one channel for two different findings, and a false one
    leaves a `[REFUTED]` line that reads like a defect in your own work rather
    than like the discovery it is. Ruling something out is a result::

        refute("8-connectivity explains the selection",
               selected_under_8 != expected)

    Returns True when the hypothesis was successfully ruled out. Dead ends are
    worth as much as live ones — re-deriving a hypothesis you already killed is
    the most common way to burn an iteration budget.

    Withdraw a mistaken dead end with ``refute(claim, retract=True)``.
    """
    return verify(claim, condition, note, evidence=evidence, key=key, retract=retract,
                  _mode="ruled_out")


def _previous_entry(entry_key: str) -> dict[str, Any] | None:
    """The most recent ledger entry filed under ``entry_key``, if any."""
    ledger = WORKSPACE / ".ccarc" / "invariants.jsonl"
    if not ledger.is_file():
        return None
    found: dict[str, Any] | None = None
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if str(entry.get("key") or entry.get("claim")) == entry_key:
            found = entry
    return found


def _verdict_word(entry: dict[str, Any]) -> str:
    if entry.get("retracted"):
        return "RETRACTED"
    if entry.get("mode") == "ruled_out":
        return "RULED OUT" if entry.get("holds") else "STILL OPEN"
    return "VERIFIED" if entry.get("holds") else "REFUTED"


def verify(
    claim: str,
    condition: bool | Callable[[], Any] = False,
    note: str | None = None,
    *,
    over: Iterable[Any] | None = None,
    evidence: Any = None,
    retract: bool = False,
    key: str | None = None,
    _mode: str = "invariant",
) -> bool:
    """Establish a claim about this puzzle by executing it.

    ``condition`` is either a boolean or a zero-argument callable (use a
    callable when the check itself might raise — the exception is recorded as a
    refutation rather than crashing your script)::

        verify("every output has the same shape as its input",
               all(shape(s['input']) == shape(s['output'])
                   for s in train_samples))

        verify("no output introduces a colour its input did not contain",
               lambda: all(set(colors(s['output'])) <= set(colors(s['input']))
                           for s in train_samples))

    Prints ``[VERIFIED]`` or ``[REFUTED]``, records the result, and returns the
    boolean so you can branch on it.

    When the check is a multi-line function rather than an expression, the AST
    capture has nothing useful to read — the honest options are a bare name
    (which the ledger flags as opaque) or a single unreadable comprehension.
    Pass ``evidence=`` with what you actually measured::

        holes = {label: hole_count(label) for label in shapes}
        verify("every shape that gets recoloured has at least one hole",
               all(holes[label] > 0 for label in recoloured),
               evidence=holes)

    That is recorded as the entry's measurement and answers the opaque-name and
    unreadable-call-site warnings, because it supplies exactly what they ask
    for. ``note=`` is prose about the finding; ``evidence=`` is the value.

    ``over=`` applies a predicate to each item of a collection, so the *same*
    predicate can be checked against training and later against your own
    predictions::

        def is_square(grid): return len(grid) == len(grid[0])

        verify("every training output is square", is_square,
               over=[s["output"] for s in train_samples])
        ...
        verify("my test prediction is square", is_square, over=[prediction])

    The doctrine asks you to run your verified invariants against your own test
    predictions, and the ledger stores claim text rather than a callable.
    Without ``over=`` the invariant has to be hand-copied into an audit script,
    and that copy can drift from what the ledger says was checked. Write the
    predicate once and point it at different grids. The entry records how many
    items it ran over.

    The ledger is append-only and **the most recent entry for a claim wins**, so
    re-verifying a claim supersedes the earlier record. If a check was wrong —
    a condition that was accidentally a tautology, say — withdraw it::

        verify("every object keeps its own colour", retract=True)

    A retracted claim disappears from ``invariants()`` and from
    ``gate.py status``. Withdrawing a bad invariant matters: the whole point of
    the ledger is that everything in it has been executed, and one claim that
    only looks verified poisons the rest.

    Supersession is keyed on the claim string, which silently fails the moment
    you reword a claim while correcting it. Pass ``key=`` to make it explicit::

        verify("output height equals input height", ..., key="height-relation")
        verify("output height equals input height times 2", ..., key="height-relation")

    When a record replaces an earlier one, the replacement carries the claim it
    displaced and prints it. So **state only what is currently true** — do not
    write the correction into the claim text ("X is 1; it is instead 2"), which
    leaves a live invariant whose own first clause is false. The ledger keeps
    the history; the claim should carry the finding.

    A claim whose verdict changes between runs prints ``CHANGED VERDICT``, which
    is worth stopping for: code written while it held is now built on sand.

    Use :func:`refute` rather than a false ``verify`` when the finding is that a
    hypothesis is dead.
    """
    error = ""
    checked = None
    if over is not None and not retract:
        # The ledger stores claim text, not callables. Applying one predicate
        # to a different set of grids runs the same check against training and
        # against the solver's own predictions, with no hand-copied duplicate
        # to drift and no callable to persist.
        subjects = list(over)
        checked = len(subjects)
        if not callable(condition):
            holds, error = False, "over= requires condition to be callable: f(item) -> bool"
        else:
            failures: list[int] = []
            raised = ""
            for index, subject in enumerate(subjects):
                try:
                    if not condition(subject):
                        failures.append(index)
                except Exception as exc:  # noqa: BLE001
                    failures.append(index)
                    if not raised:
                        raised = f"{type(exc).__name__}: {exc}"
            holds = not failures
            if failures:
                # "fails on item N" is fine for four grids and thin for forty:
                # one failure and half of them failing are different findings.
                shown = ", ".join(str(i) for i in failures[:5])
                more = f" (+{len(failures) - 5} more)" if len(failures) > 5 else ""
                error = f"fails on {len(failures)}/{checked}: item {shown}{more}"
                if raised:
                    error += f" [{raised}]"
    elif retract:
        holds = False
    elif callable(condition):
        try:
            holds = bool(condition())
        except Exception as exc:  # noqa: BLE001 - a check that blows up has not held
            holds, error = False, f"{type(exc).__name__}: {exc}"
    else:
        holds = bool(condition)

    expression, literal, opaque, unsourced = (
        ("", False, False, False) if retract else _condition_evidence()
    )

    entry = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "claim": str(claim),
        "holds": holds,
        "source": _caller_source(),
        "key": str(key) if key else str(claim),
    }

    prior = None if retract else _previous_entry(entry["key"])
    if prior is not None and not prior.get("retracted"):
        reworded = str(prior.get("claim", "")) != str(claim)
        flipped = bool(prior.get("holds")) != holds
        if reworded or flipped:
            entry["supersedes"] = {
                "claim": str(prior.get("claim", "")),
                "verdict": _verdict_word(prior),
                "at": prior.get("at", ""),
            }
    if _mode == "ruled_out":
        entry["mode"] = "ruled_out"
    if expression:
        entry["expression"] = expression
    if literal:
        entry["literal"] = True
    if opaque:
        entry["opaque"] = True
    if unsourced:
        entry["unsourced"] = True
    if evidence is not None:
        entry["measured"] = _render_evidence(evidence)
    if checked is not None:
        entry["checked_over"] = checked
    if retract:
        entry["retracted"] = True
    if note:
        entry["note"] = str(note)
    if error:
        entry["error"] = error

    ledger = WORKSPACE / ".ccarc" / "invariants.jsonl"
    try:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:
        pass  # a read-only workspace must not break exploration

    if retract:
        label = "RETRACTED"
    elif _mode == "ruled_out":
        label = "RULED OUT" if holds else "STILL OPEN"
    else:
        label = "VERIFIED " if holds else "REFUTED  "
    suffix = f"  ({note})" if note else ""
    if error:
        suffix += f"  [{error}]"
    print(f"[{label}] {claim}{suffix}")
    superseded = entry.get("supersedes")
    if superseded:
        was = superseded["verdict"]
        old_claim = superseded["claim"]
        if old_claim != str(claim):
            print(
                f"           ^ supersedes under key `{entry['key']}`: "
                f'"{old_claim}" [{was}]'
            )
            if was == "REFUTED":
                print(
                    "           The ledger now carries both the dead reading and its "
                    "replacement, so you do not need to encode the correction in the "
                    "claim text. State only what is true."
                )
        else:
            print(
                f"           ^ CHANGED VERDICT: this same claim was [{was}] at "
                f"{superseded['at']}. Anything you built on the old verdict is now "
                "suspect — re-check the code that assumed it."
            )
    # A note carrying the measured values is the evidence this warning asks
    # for, so note= and evidence= both suppress it.
    if opaque and not literal and not note and evidence is None and checked is None:
        print(
            f"           ^ the recorded evidence is just the name `{expression}`. After a "
            "compaction that says nothing about what ran — inline the check, or pass "
            "note= with the measured value."
        )
    if literal:
        print(
            "           ^ WARNING: that condition is a compile-time constant "
            f"({expression or 'literal'}). Nothing was measured, so this records an "
            "assertion, not a verification. Re-run it with a real check, or retract it."
        )
    if unsourced and evidence is None and checked is None:
        print(
            "           ^ NO EVIDENCE CAPTURED: this ran from a -c one-liner, a heredoc "
            "or a REPL, so the condition's source could not be read. The ledger entry "
            "carries the claim but nothing about what was executed, and the "
            "constant-condition check could not run at all. Put the check in a file "
            "under explore/ — or at minimum pass note= with the measured value."
        )
    return holds


def invariants() -> list[dict[str, Any]]:
    """Live invariants: most recent entry per claim, retractions removed."""
    ledger = WORKSPACE / ".ccarc" / "invariants.jsonl"
    if not ledger.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[str(entry.get("key") or entry.get("claim"))] = entry
    return [entry for entry in latest.values() if not entry.get("retracted")]


# ── free dry-run ─────────────────────────────────────────────────────────────

def check(
    solve_fn: Callable[[Grid], Any], *, verbose: bool = True, show_diff: bool = False
) -> dict[str, Any]:
    """Score a candidate ``solve`` against the training pairs. Costs nothing.

    Use this as many times as you like while iterating. It does **not** consume
    an iteration and does **not** record anything — ``python gate.py submit`` is
    the formal checkpoint, and this exists so you never spend one on a bug you
    could have caught here::

        from arc import check
        def solve(grid): ...
        check(solve)
    """
    import copy

    rows: list[dict[str, Any]] = []
    correct = 0
    total_pixel = 0.0

    for idx, sample in enumerate(train_samples):
        expected = sample["output"]
        row: dict[str, Any] = {"index": idx, "correct": False, "pixel_accuracy": 0.0}
        try:
            predicted = _first_candidate(_plain(solve_fn(copy.deepcopy(sample["input"]))))
            row["predicted"] = predicted
            row["correct"] = predicted == expected
            row["pixel_accuracy"] = _pixel_accuracy(predicted, expected)
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        correct += 1 if row["correct"] else 0
        total_pixel += row["pixel_accuracy"]
        rows.append(row)

    test_rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(test_samples):
        row = {"index": idx}
        try:
            raw = _plain(solve_fn(copy.deepcopy(sample["input"])))
            candidates = raw if _looks_like_candidate_list(raw) else [raw]
            row["candidates"] = candidates
            row["shapes"] = [shape(c) for c in candidates]
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        test_rows.append(row)

    summary = {
        "train": rows,
        "test": test_rows,
        "train_correct": correct,
        "train_total": len(train_samples),
        "train_pixel_accuracy": total_pixel / len(train_samples) if train_samples else 0.0,
        "all_train_correct": bool(train_samples) and correct == len(train_samples),
    }

    if verbose:
        for row in rows:
            if row.get("error"):
                print(f"train {row['index']}  ERROR  {row['error']}")
            elif row["correct"]:
                print(f"train {row['index']}  PASS   {shape(row.get('predicted'))}")
            else:
                expected = train_samples[row["index"]]["output"]
                wrong = _count_wrong(row.get("predicted"), expected)
                print(
                    f"train {row['index']}  FAIL   got {shape(row.get('predicted'))} "
                    f"want {shape(expected)}  "
                    f"wrong={wrong}  pixel={row['pixel_accuracy']:.3f}"
                )
                if show_diff:
                    predicted = row.get("predicted")
                    if isinstance(predicted, list) and shape(predicted) == shape(expected):
                        cells = [
                            (r, c, expected[r][c], predicted[r][c])
                            for r in range(len(expected))
                            for c in range(len(expected[0]))
                            if expected[r][c] != predicted[r][c]
                        ]
                        listing = " ".join(f"({r},{c}) want {w} got {g}" for r, c, w, g in cells[:30])
                        print(f"           {listing}")
                        if len(cells) > 30:
                            print(f"           … {len(cells) - 30} more")
        for row in test_rows:
            if row.get("error"):
                print(f"test  {row['index']}  ERROR  {row['error']}")
            else:
                candidates = row.get("candidates") or []
                print(f"test  {row['index']}  -> {len(candidates)} candidate(s) "
                      f"{row.get('shapes')}")
                # Without this the only way to see how far apart two
                # candidates are is the gate's report, which costs an
                # iteration to read.
                if len(candidates) > 1:
                    first, second = candidates[0], candidates[1]
                    if shape(first) != shape(second):
                        print("           candidates differ in shape")
                    else:
                        n = sum(
                            1
                            for r in range(len(first))
                            for c in range(len(first[0]))
                            if first[r][c] != second[r][c]
                        )
                        print(f"           candidates differ in {n} cell(s)")
        print(
            f"== {summary['train_correct']}/{summary['train_total']} training examples reproduced "
            f"(mean pixel {summary['train_pixel_accuracy']:.3f})"
        )
        if summary["all_train_correct"]:
            _hedging_advice(test_rows)
            hypothesis = WORKSPACE / "solution" / "hypothesis.md"
            if hypothesis.is_file() and hypothesis.read_text(encoding="utf-8").strip():
                print("== ready for `python gate.py submit`")
            else:
                print("== write solution/hypothesis.md, then `python gate.py submit`")
    return summary


def _hedging_advice(test_rows: list[dict[str, Any]]) -> None:
    """Raise the second-attempt question here, where acting on it is free.

    The gate asks it too, but a candidate comes out of ``solve()``, so acting on
    the gate's version costs a whole iteration to resubmit. Asking at dry-run
    time means the hedge lands in the submission that was going to happen anyway.
    """
    unspent = [
        row["index"]
        for row in test_rows
        if not row.get("error") and 0 < len(row.get("candidates") or []) < 2
    ]
    if not unspent:
        return

    # A rival only belongs in the advice for a test example where it actually
    # predicts something different. Naming a rival that diverges on test 1 while
    # test 0 is the one with the free slot trains the solver to skim the nudge.
    live_rivals = []
    for entry in rivals():
        if not entry.get("fits_training"):
            continue
        predictions = entry.get("predictions") or []
        relevant = [
            index for index in unspent
            if index < len(predictions) and predictions[index] is not None
        ]
        divergent = _rival_divergence(predictions)
        if divergent is not None:
            relevant = [index for index in relevant if index in divergent]
        if relevant:
            live_rivals.append({**entry, "relevant": relevant})
    dead_ends = [e for e in invariants() if e.get("mode") == "ruled_out" and e.get("holds")]
    # A sweep that left several readings standing is the strongest hedging
    # signal the workspace holds: the solver has already established, by
    # execution, that training cannot separate them. Naming them here costs
    # nothing and beats asking for rivals the solver has in fact already found.
    open_sweeps = [
        entry for entry in invariants()
        if entry.get("mode") == "sweep" and len(entry.get("survivors") or []) > 1
    ]

    where = ", ".join(f"test {index}" for index in unspent)
    print(f"\n== {where} carries one candidate, and ARC-AGI-2 scores two.")

    if open_sweeps and not live_rivals:
        for entry in open_sweeps[:2]:
            survivors = entry.get("survivors") or []
            print(f"   your sweep '{entry.get('claim')}' left {len(survivors)} readings alive:")
            print(f"     {', '.join(survivors)}")
        print("   You established by execution that training cannot separate these. Put them")
        print("   through arc.rival(name, fn) and see which diverges on the example above —")
        print("   that is what the second slot is for, and you already did the hard part.")
        # A sweep records no per-test-index information, so unlike a rival it
        # cannot be filtered to the example with the free slot. Say so rather
        # than implying relevance the harness has not established.
        print("   (A sweep records no per-example detail, so this may bear on a different")
        print("   test example than the one above. Running the rival is how you find out.)")
        _print_situation_types_prompt()
        return

    if not live_rivals and not dead_ends:
        # An empty rival ledger still earns the prompt. The first dry run is
        # the moment before the first submission when acting on it is free,
        # and it is also the moment when nothing has been registered yet.
        print("   You have registered no rival readings, which is not the same as there being")
        print("   none. Name the interpretation you rejected on the way here and run it through")
        print("   arc.rival(name, fn): if it reproduces every training pair and predicts")
        print("   something different, it is the best possible use of the second slot.")
        _print_situation_types_prompt()
        return

    if live_rivals:
        for entry in live_rivals[:3]:
            where = ", ".join(f"test {i}" for i in entry["relevant"])
            print(f"   rival fitting every training pair, differing on {where}: {entry['name']}")
        print("   Training cannot separate it from your reading. Unless you can point at")
        print("   evidence that rules it out, return it as the second candidate.")
    else:
        print(f"   You ruled out {len(dead_ends)} rival reading(s). Check how each died:")
        print("   a training pair it fails is a proof; extending a training-output regularity")
        print("   to the test input is not. Hedge here — after submitting it costs an iteration.")

    _print_situation_types_prompt()


#: Mirrored deliberately. The gate prints this after a submission and holds
#: its own copy (reporting.GENERALIZATION_AUDIT_DIRECTIVE). ``arc.py`` is
#: copied standalone into each workspace and cannot import from the harness
#: package, so the duplication is structural.
_SITUATION_TYPES_PROMPT = (
    "   Do not assume the answer is among the readings you named. Those are the rivals you\n"
    "   thought to write down, and they may all be proofs. The leap is usually somewhere you\n"
    "   never framed as a rival at all: enumerate the situation types your rule has to handle\n"
    "   on the test input, and check that each one is actually witnessed in a training pair.\n"
    "   A case the test needs and training never shows is an unhedged assumption, however\n"
    "   confident the rule feels."
)


def _print_situation_types_prompt() -> None:
    """Ask the out-of-sample question for free, not only after a submission.

    The gate prints this paragraph on its post-submission path. Anything that
    can change what a solver ships must also be reachable without spending an
    iteration, so the dry run prints it too.
    """
    print(_SITUATION_TYPES_PROMPT)


def rival(name: str, solve_fn: Callable[[Grid], Any]) -> dict[str, Any]:
    """Register an alternative reading of the puzzle, scored against training.

    Use this the moment you implement a rival interpretation in order to
    *compare* it — which is usually the moment you are about to discard it::

        def strict(grid): ...        # the reading you suspect is wrong
        arc.rival("an object touching the edge does not count", strict)

    If the rival reproduces every training pair and predicts something different
    from your own solution on a test input, the gate will say so at submission
    time. ARC-AGI-2 scores two attempts per test example, and a rival that fits
    all the evidence you have is the single best use of the second one.

    A rival discarded because its answer on the unlabelled input looks wrong,
    rather than because a training pair refutes it, forfeits that second
    attempt for nothing.
    """
    summary = check(solve_fn, verbose=False)
    predictions: list[Any] = []
    for sample in test_samples:
        try:
            raw = _plain(solve_fn([row[:] for row in sample["input"]]))
            predictions.append(_first_candidate(raw))
        except Exception:  # noqa: BLE001 - a rival that crashes on test is still a record
            predictions.append(None)

    entry = {
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "name": str(name),
        "fits_training": bool(summary["all_train_correct"]),
        "train_correct": summary["train_correct"],
        "train_total": summary["train_total"],
        "predictions": predictions,
        "source": _caller_source(),
    }
    # Re-registering a name replaces what that name means, so supersession is
    # announced here the way verify() announces it. Otherwise re-registering a
    # fixed implementation drops the earlier entry, and any divergence it
    # claimed, without trace.
    prior = None
    for existing in rivals():
        if str(existing.get("name")) == entry["name"]:
            prior = existing
            break
    if prior is not None:
        changed_fit = bool(prior.get("fits_training")) != entry["fits_training"]
        changed_predictions = (prior.get("predictions") or []) != predictions
        if changed_fit or changed_predictions:
            entry["supersedes"] = {
                "fits_training": bool(prior.get("fits_training")),
                "train_correct": prior.get("train_correct"),
                "train_total": prior.get("train_total"),
                "at": prior.get("at", ""),
            }

    ledger = WORKSPACE / ".ccarc" / "rivals.jsonl"
    try:
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:
        pass

    superseded = entry.get("supersedes")
    if superseded:
        was = (
            f"{superseded['train_correct']}/{superseded['train_total']}"
            if superseded.get("train_total") is not None else "?"
        )
        print(
            f"[REPLACED] a rival named '{name}' was already registered ({was} on training, "
            f"recorded {superseded['at']}). This implementation predicts differently, so the\n"
            "           earlier reading is gone from the ledger. If the first one was buggy "
            "that is what you want; if they are two different readings, give them two names."
        )

    if entry["fits_training"]:
        print(
            f"[RIVAL   ] {name} — reproduces all "
            f"{entry['train_total']} training pairs. Training cannot separate it from yours."
        )
        # The whole point is to stop a slot being forfeited, so the payload is
        # whether spending it would change anything. This function holds the
        # rival's predictions; comparing them costs nothing.
        shipped = _shipped_candidates()
        standing = _rival_standing(predictions, shipped)
        if standing is None:
            print(
                "           No solution/solve.py to compare against yet — the gate will check "
                "at submission time."
            )
        else:
            differs = [i for i, s in standing.items() if s == "differs"]
            hedged = [i for i, s in standing.items() if s == "hedged"]
            contested = [i for i, s in standing.items() if s == "contested"]
            if contested:
                where = ", ".join(f"test {i}" for i in contested)
                print(
                    f"           On {where} both slots are already spent, on readings that are "
                    "not this one. Taking this rival means dropping one of them — so the "
                    "question is which two of the three survive the most evidence, not "
                    "whether to add a third."
                )
            if differs:
                where = ", ".join(f"test {i}" for i in differs)
                print(
                    f"           It predicts differently on {where}. That is your second candidate."
                )
                # There is one slot, and standing is computed against solve.py
                # alone, so two rivals diverging on the same example would each
                # be told "that is your second candidate" independently. Name
                # the other claimants instead of implying the slot is free.
                claimants = [
                    other.get("name")
                    for other in rivals()
                    if other.get("fits_training")
                    and str(other.get("name")) != entry["name"]
                    and any(
                        index in differs
                        for index, standing_ in (
                            _rival_standing(other.get("predictions") or [], shipped) or {}
                        ).items()
                        if standing_ == "differs"
                    )
                ]
                if claimants:
                    listed = ", ".join(str(name) for name in claimants[:3])
                    print(
                        f"           But {len(claimants)} other registered rival(s) also diverge "
                        f"on {where}: {listed}.\n"
                        "           There is one slot. Which of them survives the most evidence?"
                    )
            if hedged:
                where = ", ".join(f"test {i}" for i in hedged)
                print(
                    f"           On {where} it is already your second candidate — the slot is "
                    "spent on it and the hedge is doing its job. Leave it in place."
                )
            if not differs and not hedged and not contested:
                print(
                    "           It predicts exactly what your first candidate does on every test "
                    "input, so spending a slot on it would change nothing."
                )
    else:
        print(
            f"[RIVAL   ] {name} — fails training "
            f"({entry['train_correct']}/{entry['train_total']}). Ruled out by evidence, not by "
            "assumption."
        )
    return entry


def _shipped_candidates() -> dict[int, list[Any]] | None:
    """The current solution's candidates per test index, computed once.

    Returns None when there is no loadable solution. An index missing from the
    mapping is one whose solve() raised.
    """
    try:
        solve = load_solution()
    except Exception:  # noqa: BLE001 - no solution yet, or it does not load
        return None
    out: dict[int, list[Any]] = {}
    for index, sample in enumerate(test_samples):
        try:
            raw = _plain(solve([row[:] for row in sample["input"]]))
        except Exception:  # noqa: BLE001
            continue
        out[index] = raw if _looks_like_candidate_list(raw) else [raw]
    return out


def _rival_standing(
    predictions: list[Any], shipped: dict[int, list[Any]] | None = None
) -> dict[int, str] | None:
    """How a rival stands against the current solution, per test index.

    ``"differs"``  — predicts something neither of your candidates does.
    ``"hedged"``   — it *is* one of your later candidates already.
    ``"same"``     — identical to your first candidate; spending a slot is a no-op.

    The distinction is the point: "already your second candidate" and
    "redundant with your first" are opposite situations. Collapsed into
    "diverges or not", a rival already sitting in the candidate list reads as
    needing no slot, and acting on that deletes a correct hedge.

    Returns None when there is no loadable solution to compare against.

    ``shipped`` lets a caller supply the shipped candidates it has already
    computed. Without it, checking N rivals for contention costs N module loads
    and N x T solve() calls, for an advisory print, on a solve() that may do
    real search.
    """
    if shipped is None:
        shipped = _shipped_candidates()
    if shipped is None:
        return None

    standing: dict[int, str] = {}
    for index, prediction in enumerate(predictions):
        if prediction is None or index >= len(test_samples):
            continue
        mine = shipped.get(index)
        if mine is None:
            continue
        if not mine:
            standing[index] = "differs"
        elif prediction == mine[0]:
            standing[index] = "same"
        elif prediction in mine[1:]:
            standing[index] = "hedged"
        elif len(mine) >= 2:
            # Divergent, but there is no free slot to put it in. "That is your
            # second candidate" is wrong here — the second candidate exists and
            # is a different reading, so this is a swap, not an addition.
            standing[index] = "contested"
        else:
            standing[index] = "differs"
    return standing


def _rival_divergence(predictions: list[Any]) -> list[int] | None:
    """Test indices where a rival's prediction is not among the shipped candidates."""
    standing = _rival_standing(predictions)
    if standing is None:
        return None
    return [index for index, status in standing.items() if status == "differs"]


def rivals() -> list[dict[str, Any]]:
    """Alternative readings registered so far, most recent per name."""
    ledger = WORKSPACE / ".ccarc" / "rivals.jsonl"
    if not ledger.is_file():
        return []
    latest: dict[str, dict[str, Any]] = {}
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        latest[str(entry.get("name"))] = entry
    return list(latest.values())


def solution_module(path: str | os.PathLike[str] | None = None) -> Any:
    """Import ``solution/solve.py`` and return the whole module.

    ``load_solution()`` hands back only ``solve``, which is not enough when you
    want to build a rival that shares the shipped parse. Reaching the module's
    other helpers otherwise means hand-rolling ``importlib``, which is the
    boilerplate this toolkit exists to remove::

        from arc import solution_module, rival
        shipped = solution_module()
        def alt(grid):
            parsed = shipped._parse(grid)     # reuse, do not re-derive
            ...
        rival("ties go to the larger object, not the earlier one", alt)

    Reusing the shipped helpers is the point: a rival that re-implements the
    parse is testing two changes at once.
    """
    import importlib.util

    target = Path(path) if path else (WORKSPACE / "solution" / "solve.py")
    if not target.is_absolute():
        target = WORKSPACE / target
    if not target.is_file():
        raise FileNotFoundError(f"No solution at {target}")

    spec = importlib.util.spec_from_file_location("ccarc_candidate_solution", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def explore_module(name: str) -> Any:
    """Import an exploration script by name, whatever it is called.

    ``explore/`` is on ``sys.path`` for scripts run from it, so
    ``from lib import ...`` works — until the script's name starts with a digit,
    which is not a legal Python identifier. That collides with the documented
    habit of naming scripts for the question they answer plus a numeric prefix.
    Working around it means hand-rolling ``SourceFileLoader``, which is exactly
    the ``importlib`` boilerplate this toolkit exists to remove::

        from arc import explore_module
        census = explore_module("02_object_census")
        census.objects(grid)

    The ``.py`` suffix is optional.
    """
    import importlib.util

    stem = str(name)
    if stem.endswith(".py"):
        stem = stem[:-3]
    target = WORKSPACE / "explore" / f"{stem}.py"
    if not target.is_file():
        raise FileNotFoundError(f"No exploration script at {target}")

    spec = importlib.util.spec_from_file_location(f"ccarc_explore_{abs(hash(stem))}", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_solution(path: str | os.PathLike[str] | None = None) -> Callable[[Grid], Any]:
    """Return the ``solve`` function from ``solution/solve.py``.

    The doctrine asks you to run your verified invariants against your own test
    predictions before accepting — which means loading the solution you just
    wrote into an exploration script. Do it with this, from anywhere::

        from arc import load_solution, test_samples
        solve = load_solution()
        prediction = solve(test_samples[0]['input'])

    Resolves relative to the workspace, so the calling script works from any
    directory.
    """
    module = solution_module(path)
    solve = getattr(module, "solve", None)
    if not callable(solve):
        raise AttributeError("solution/solve.py does not define a callable solve(grid)")
    return solve


def _plain(value: Any) -> Any:
    """Coerce numpy arrays to nested lists, recursively."""
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict, str, bytes)):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _looks_like_candidate_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and value
        and all(isinstance(v, list) and v and isinstance(v[0], list) for v in value)
    )


def _first_candidate(value: Any) -> Any:
    return value[0] if _looks_like_candidate_list(value) else value


def _count_wrong(predicted: Any, expected: Grid) -> int | str:
    if not isinstance(predicted, list) or shape(predicted) != shape(expected):
        return "n/a (shape mismatch)"
    return sum(
        1
        for r in range(len(expected))
        for c in range(len(expected[0]))
        if predicted[r][c] != expected[r][c]
    )


def _pixel_accuracy(predicted: Any, expected: Grid) -> float:
    try:
        exp_rows, exp_cols = len(expected), len(expected[0])
        total = exp_rows * exp_cols
        if not total:
            return 0.0
        pred_rows = len(predicted)
        pred_cols = len(predicted[0]) if pred_rows else 0
        return sum(
            1
            for r in range(min(pred_rows, exp_rows))
            for c in range(min(pred_cols, exp_cols))
            if predicted[r][c] == expected[r][c]
        ) / total
    except Exception:  # noqa: BLE001
        return 0.0


# ── observation ──────────────────────────────────────────────────────────────

def shape(grid: Any) -> str:
    """``"HxW"`` for a grid, ``"?"`` for anything that is not one."""
    try:
        return f"{len(grid)}x{len(grid[0])}"
    except Exception:  # noqa: BLE001
        return "?"


def colors(grid: Grid) -> list[int]:
    """Sorted distinct colours present in a grid."""
    return sorted({int(cell) for row in grid for cell in row})


def histogram(grid: Grid) -> dict[int, int]:
    """``{colour: count}``, most useful for spotting the background."""
    counts: dict[int, int] = {}
    for row in grid:
        for cell in row:
            counts[int(cell)] = counts.get(int(cell), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


#: Fired at most once per process, so a long exploration session gets the
#: reminder without every call nagging.
_DURABILITY_REMINDED = False

#: Exploration scripts written before an empty ledger becomes worth mentioning.
_UNRECORDED_SCRIPT_THRESHOLD = 4


def _exploration_durability_check() -> None:
    """Mention an empty ledger during exploration, not only at dry-run time.

    The dry-run nudges only reach a solver that already has a candidate
    solution. Several exploration scripts, no ``solve.py`` and nothing recorded
    is the window where a compaction costs the most.

    It runs from ``show()`` and again at interpreter exit. The exit hook is
    both broader and more precise than any single call site: a script that
    records something during its run clears the condition and stays silent, and
    one that records nothing says so where the solver is already reading.
    """
    global _DURABILITY_REMINDED
    if _DURABILITY_REMINDED:
        return
    try:
        explore = WORKSPACE / "explore"
        scripts = [
            p for p in explore.glob("*.py") if p.name != "arc.py"
        ] if explore.is_dir() else []
        if len(scripts) < _UNRECORDED_SCRIPT_THRESHOLD or invariants():
            return
    except Exception:  # noqa: BLE001 - a reminder must never break exploration
        return
    _DURABILITY_REMINDED = True
    print(
        f"\n[note] {len(scripts)} exploration scripts, nothing recorded with arc.verify().\n"
        "       A compaction now would discard everything you have worked out; the ledger\n"
        "       and NOTES.md are what `gate.py status` replays. Record the facts you are\n"
        "       already relying on.\n"
    )


def _register_durability_atexit() -> None:
    import atexit

    def _at_exit() -> None:
        try:
            _exploration_durability_check()
        except Exception:  # noqa: BLE001 - never let a reminder break a run
            pass

    atexit.register(_at_exit)


_register_durability_atexit()


def show(grid: Grid, title: str | None = None, ruler: bool = True) -> None:
    """Print a grid with row and column indices."""
    _exploration_durability_check()
    if title:
        print(title)
    if not grid:
        print("<empty>")
        return
    height, width = len(grid), len(grid[0])
    pad = len(str(height - 1))
    if ruler:
        tens = "".join(str((c // 10) % 10) if c >= 10 else " " for c in range(width))
        ones = "".join(str(c % 10) for c in range(width))
        if width > 10:
            print(" " * (pad + 1) + tens)
        print(" " * (pad + 1) + ones)
    for r, row in enumerate(grid):
        print(f"{r:>{pad}} " + "".join(str(cell) for cell in row))
    print(f"({height}x{width}, colours {colors(grid)})")


def diff(a: Grid, b: Grid, *, limit: int = 40, labels: tuple[str, str] = ("a", "b")) -> None:
    """Print where two grids disagree."""
    if shape(a) != shape(b):
        print(f"shape mismatch: {labels[0]}={shape(a)} {labels[1]}={shape(b)}")
        return
    entries = [
        (r, c, a[r][c], b[r][c])
        for r in range(len(a))
        for c in range(len(a[0]))
        if a[r][c] != b[r][c]
    ]
    if not entries:
        print(f"identical ({shape(a)})")
        return
    rows = sorted({e[0] for e in entries})
    cols = sorted({e[1] for e in entries})
    print(
        f"{len(entries)} differing cells in {shape(a)}; "
        f"rows {rows[0]}-{rows[-1]}, cols {cols[0]}-{cols[-1]}"
    )
    for r, c, va, vb in entries[:limit]:
        print(f"  ({r},{c}) {labels[0]}={va} ({COLOR_NAMES.get(va, '?')}) "
              f"{labels[1]}={vb} ({COLOR_NAMES.get(vb, '?')})")
    if len(entries) > limit:
        print(f"  … {len(entries) - limit} more")


def png(grid: Grid, path: str | os.PathLike[str], cell: int = 24) -> str | None:
    """Render a grid to a PNG you can then open with the Read tool.

    Returns the path, or None when Pillow is unavailable. Useful for looking at
    a predicted output the way you looked at the puzzle: as a picture.
    """
    if not grid:
        return None
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # The interpreter running this script has no Pillow, but the one that
        # built the workspace does — it rendered task/images/. Borrow it rather
        # than making the doctrine's "look at your prediction" advice a dead end.
        return _png_via_harness_python(grid, path, cell)

    height, width = len(grid), len(grid[0])
    image = Image.new("RGB", (width * cell, height * cell), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    for r in range(height):
        for c in range(width):
            draw.rectangle(
                [c * cell, r * cell, c * cell + cell, r * cell + cell],
                fill=_PALETTE.get(int(grid[r][c]), (128, 128, 128)),
                outline=(50, 50, 50),
            )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="PNG", compress_level=1)
    return str(target)


def _png_via_harness_python(grid: Grid, path: str | os.PathLike[str], cell: int) -> str | None:
    """Render through the interpreter that built this workspace."""
    import subprocess

    recorded = WORKSPACE / ".ccarc" / "harness_python"
    if not recorded.is_file():
        print("Pillow is not installed here and no fallback interpreter was recorded.")
        return None
    executable = recorded.read_text(encoding="utf-8").strip()
    if not executable:
        print("Pillow is not installed here and no fallback interpreter was recorded.")
        return None

    payload = json.dumps({"grid": grid, "path": str(path), "cell": int(cell), "palette": _PALETTE})
    source = (
        "import json,sys\n"
        "from PIL import Image, ImageDraw\n"
        "a=json.loads(sys.stdin.read())\n"
        "g=a['grid']; c=a['cell']; pal={int(k):tuple(v) for k,v in a['palette'].items()}\n"
        "h,w=len(g),len(g[0])\n"
        "im=Image.new('RGB',(w*c,h*c),(255,255,255)); d=ImageDraw.Draw(im)\n"
        "for r in range(h):\n"
        "    for x in range(w):\n"
        "        d.rectangle([x*c,r*c,x*c+c,r*c+c],fill=pal.get(int(g[r][x]),(128,128,128)),outline=(50,50,50))\n"
        "import os\n"
        "os.makedirs(os.path.dirname(os.path.abspath(a['path'])) or '.',exist_ok=True)\n"
        "im.save(a['path'],format='PNG',compress_level=1)\n"
    )
    try:
        done = subprocess.run(
            [executable, "-c", source], input=payload, capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"Could not render PNG via {executable}: {exc}")
        return None
    if done.returncode != 0:
        print(f"Could not render PNG via {executable}: {(done.stderr or '').strip()[:200]}")
        return None
    return str(path)


def _pairs() -> Iterable[tuple[Grid, Grid]]:
    for sample in train_samples:
        yield sample["input"], sample["output"]


def _executable_lines(source: str, filename: str) -> set[int]:
    """Line numbers inside function bodies of ``source``.

    Module-level lines are excluded on purpose: imports, constants and ``def``
    headers all run at import time, so counting them would report every function
    header as unreached and bury the signal.
    """
    import types

    top = compile(source, filename, "exec")
    lines: set[int] = set()
    stack = [top]
    seen: set[int] = set()
    while stack:
        code = stack.pop()
        if id(code) in seen:
            continue
        seen.add(id(code))
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                stack.append(const)
        if code is top:
            continue
        for _start, _end, lineno in code.co_lines():
            if lineno is not None and lineno != code.co_firstlineno:
                lines.add(lineno)
    return lines


def unreached(solve_fn=None, *, verbose: bool = True) -> dict:
    """Report the lines of your solution that no training pair ever executes.

    Training is the only thing standing behind your solution, and it can only
    vouch for code it runs. Every line this prints is a line ``check()`` and
    ``gate.py submit`` were blind to — most often a branch written for a case the
    examples do not contain, which is exactly the case a test input might be::

        from arc import unreached
        unreached()

    Worth running before you accept, and worth running twice if you shipped a
    second candidate. A hedge exists *because* the training pairs cannot
    discriminate it, so the alternative reading is code training cannot reach
    almost by definition — and it ships with whatever bugs it has.

    **Scope: the whole file the function lives in**, not just that function's own
    body — which is what you want for ``solution/solve.py``, where the helpers are
    part of the solution, and worth knowing if you pass a function from an
    ``explore/`` script, where the rest of the file is scaffolding. The report
    names the file it measured. Module-level lines are excluded either way: they
    run at import, so counting them would flag every ``def`` as unreached.

    Returns ``{"unreached": [...], "executed": n, "executable": n, "path": str,
    "errors": [...]}``.
    """
    import copy
    import sys
    from pathlib import Path

    if solve_fn is None:
        solve_fn = load_solution()

    origin = getattr(solve_fn, "__code__", None)
    if origin is None:
        raise TypeError(f"unreached() needs a Python function, not {type(solve_fn).__name__}")
    filename = origin.co_filename
    if filename.startswith("<") and filename.endswith(">"):
        # `python -c "..."` gives <string>, a REPL gives <stdin>. There is no
        # file to read line numbers out of, and the fix is to move the function
        # somewhere durable rather than to guess.
        raise RuntimeError(
            f"unreached() cannot measure a function defined in {filename} — there is no "
            "source file. Put solve() in solution/solve.py and call unreached() with no "
            "arguments, or define it in an explore/ script and pass it from there."
        )
    path = Path(filename).resolve()
    try:
        source = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read the solution source at {path}: {exc}") from None

    executable = _executable_lines(source, str(path))
    target = str(path)
    executed: set[int] = set()

    def _local(frame, event, arg):
        if event == "line":
            executed.add(frame.f_lineno)
        return _local

    def _global(frame, event, arg):
        if frame.f_code.co_filename == target:
            executed.add(frame.f_lineno)
            return _local
        return None

    errors: list[str] = []
    previous = sys.gettrace()
    sys.settrace(_global)
    try:
        for sample in train_samples:
            try:
                solve_fn(copy.deepcopy(sample["input"]))
            except Exception as exc:  # noqa: BLE001 - a crash still leaves coverage worth reading
                errors.append(f"{type(exc).__name__}: {exc}")
    finally:
        sys.settrace(previous)

    missing = sorted(executable - executed)
    text = source.splitlines()
    result = {
        "unreached": missing,
        "executed": len(executable & executed),
        "executable": len(executable),
        "path": str(path),
        "errors": errors,
    }

    if verbose:
        for err in errors:
            print(f"! solve raised on a training input: {err}")
        if not executable:
            print("no function bodies found in the solution — nothing to measure")
        elif not missing:
            print(
                f"every one of the {len(executable)} lines in {path.name} runs on the "
                "training pairs. Training reaches all of your code."
            )
        else:
            pct = 100.0 * len(executable & executed) / len(executable)
            print(
                f"{len(missing)} of {len(executable)} lines in {path.name} never run on any "
                f"training pair ({pct:.0f}% reached).\n"
                "Training cannot vouch for these:"
            )
            for lineno in missing:
                body = text[lineno - 1].rstrip() if lineno - 1 < len(text) else ""
                print(f"  {lineno:>4}  {body}")
            print(
                "\nThese lines are not wrong. They are untested — a test input that reaches "
                "one is running code no example has ever exercised."
            )

    return result


if __name__ == "__main__":  # `python arc.py` prints a quick orientation dump
    print(f"workspace: {WORKSPACE}")
    print(f"{len(train_samples)} training pairs, {len(test_samples)} test input(s)\n")
    for i, (inp, out) in enumerate(_pairs()):
        print(f"train {i}: {shape(inp)} -> {shape(out)}   "
              f"colours in {colors(inp)} -> out {colors(out)}")
    for i, sample in enumerate(test_samples):
        print(f"test  {i}: {shape(sample['input'])}          colours {colors(sample['input'])}")
