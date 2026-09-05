# Puzzle workspace — task `__TASK_ID__`

Solve one ARC-AGI-2 task in this directory. Everything you need is here; nothing
outside this directory is relevant, and the test outputs are not in it.

## Environment

__ENVIRONMENT__

`from arc import ...` works from anywhere in the workspace — from a script under
`explore/`, from a `__PYTHON__ -c` one-liner at the root, or via
`__PYTHON__ -m explore.foo`. You never need `sys.path` boilerplate.

## Layout

```
task/task.json        __N_TRAIN__ training pairs + __N_TEST__ test input(s). No test outputs.
task/grids.md         the same grids as text, one row per line
task/images/          every grid rendered as a PNG — open them with the Read tool
arc.py                observation + verification helpers (read it; it is short)
dryrun.py             free scoring of solution/solve.py against the training pairs
gate.py               the verification gate
explore/              your scratch scripts. Unlimited, free, not recorded.
solution/hypothesis.md   the rule, in prose        (you create this)
solution/solve.py        the rule, as code         (you create this)
solution/audit.md        generalization audit      (you create this, at the end)
NOTES.md              your durable research state — keep it current
.ccarc/             run ledger: iterations, verified invariants, reports
```

## The loop

1. **Look.** `Read task/images/train_0.png` … and the test input(s). Then dump
   structure with a script under `explore/`. ARC is a vision task; the pictures
   and the numbers tell you different things.
2. **Verify.** Establish facts by executing them:
   `arc.verify("every output is 9x9", ...)`. Every recorded invariant narrows
   the space of rules that can be correct.
3. **Hypothesize.** Write the rule into `solution/hypothesis.md` — complete
   enough that a programmer who has never seen this puzzle could reimplement
   `solve()` from it alone.
4. **Implement and dry-run.** Write `solution/solve.py`, then run
   `__PYTHON__ dryrun.py`. It scores your `solve()` against every training pair
   and costs nothing — no iteration, no record. Iterate here until it passes.
   Every bug you catch with `dryrun.py` is a budgeted submission you keep.
5. **Submit.** `__PYTHON__ gate.py submit`. Budgeted and permanent.
6. **Reflect.** The gate's output ends with what to do next. Follow it, and
   append the reflection to `NOTES.md` — `Read` that file before your first
   `Write`, since it already exists with template content.
7. **Accept.** Once training passes, write `solution/audit.md`, then
   `__PYTHON__ gate.py accept`. That ends the run — stop working after it
   succeeds.

## Gate commands

| Command | Cost | What it does |
|---|---|---|
| `__PYTHON__ dryrun.py` | free | Scores `solution/solve.py` against the training pairs. Use it before every submission. |
| `__PYTHON__ gate.py status` | free | Distilled research state: iterations, executed ledger, last hypothesis, notes tail. **Run this first after any context compaction.** |
| `__PYTHON__ gate.py status --brief` | free | Same, minus the hypothesis dump and notes tail — iteration history and ledger only. Use this for a mid-run check. |
| `__PYTHON__ gate.py submit` | 1 iteration | Runs `solution/solve.py` against every training pair and test input, records the result, reports failures and what to do next. |
| `__PYTHON__ gate.py accept` | free | Finalizes the run, using `solution/audit.md`. It re-runs `solution/solve.py` as it stands, so adding a second candidate after a train-perfect submission costs you nothing — if the rule regresses, it falls back to what you submitted. |

## The invariant ledger

`arc.verify(claim, condition)` appends to `.ccarc/invariants.jsonl`, and the
gate replays it. What is worth knowing about it:

- **The expression is recorded, not just the claim.** `verify()` reads the
  condition's source from your script, so the ledger says what was executed. A
  condition that is a compile-time constant is flagged: it measured nothing.
- **Ruling something out is a result — use `arc.refute()` for it.**
  `refute("8-connectivity explains the selection", ...)` records a dead end as a
  finding rather than leaving a `[REFUTED]` line that reads like a defect in
  your own work. Re-deriving a hypothesis you already killed is the most common
  way to burn an iteration budget.
- **Most recent entry per key wins**, and the key defaults to the claim string —
  so rewording a claim while correcting it silently creates a duplicate instead
  of superseding. Pass `key="height-relation"` when you expect to revise.
- **State only what is currently true.** A replacement prints the claim it
  displaced, so you never need to write a correction into the claim text. "The
  gap between stripes is 2 cells; it is instead always one less than the stripe
  itself" leaves a live *verified* invariant whose own first clause is false —
  and after a compaction that sentence is what gets read back. Say
  `verify("the gap is always one less than the stripe", ..., key="gap-width")`
  and let the ledger carry the history.
- **A changed verdict is loud, and worth stopping for.** If a claim held
  earlier and fails now, `CHANGED VERDICT` prints: whatever you built while it
  held is now suspect.
- **The history is on disk, and nothing shows it to you.** `invariants()` and
  `gate.py status` both return the *live* set — most recent entry per key,
  retractions removed. Every superseded line is still in
  `.ccarc/invariants.jsonl`, and you can read it directly:

  ```bash
  python -c "import json;[print(json.loads(l)['at'][11:19], json.loads(l).get('key','')[:60]) for l in open('.ccarc/invariants.jsonl')]"
  ```

  Most of that history is worth nothing — a script re-run with unchanged wording
  supersedes itself and says only that you ran it twice. There is one case worth
  the read: **a count inside a claim that has gone DOWN.** A count that rises
  means you found more occasions, which is ordinary. A count that falls means
  you split a compound claim and discovered its support was thinner than the
  compound made it look — the clause you thought every training pair stood
  behind turns out to rest on a single occasion. That is the shape that loses
  tasks, and once the entry is superseded the live readout shows the new number
  with no sign that it dropped.
- **`gate.py status` marks dead ends `[KILLED]`, not `[OK]`.** A `refute()`
  entry that "holds" means the hypothesis is dead, so the claim beside
  `[KILLED]` is something you ruled *out*. Read the marker, not just the
  sentence.
- **One clause per claim, and write the count into the claim itself.** A
  compound invariant — "an object is recoloured by its size, unless it touches
  the border, unless it is the largest, unless it is a single cell" — is
  verified as a unit, and a unit passes on the
  strength of its easiest clause. Split it. Then say how much held each clause
  up **inside the claim text**, because the claim text is what `gate.py status`
  prints and what you read back after a compaction:

  ```python
  verify("every output grid is 10x10 — 4 of 4 training grids", ...,
         over=train_outputs, key="output-size")
  verify("the tie-break by column order decides the outcome — 1 occasion "
         "in training", ..., over=tie_events, key="tie-break")
  ```

  A number in a field is unitless and easy to leave behind. A number in a
  sentence must carry its unit or the sentence reads as nonsense — and then
  "4 grids" and "1 occasion where the tie-break decided anything" are visibly
  different quantities rather than two entries that both say `[OK]`.

  **Count in the units the clause is about.** A claim about grids counts grids.
  A claim about one branch of your rule counts the times that branch decided
  anything. "Checked over 4 grids" says nothing about a branch that fired once
  inside them.

  Two habits this requires, both cheap:

  - **Pass `key=`.** Supersession is keyed on the claim string, and the count is
    the part of the sentence most likely to change. Re-verify over 4 occasions
    where you once had 3, and the reworded claim lands under a *new* key: the
    old entry stays live beside it, and the ledger then shows two counts for one
    fact, both marked `[OK]`, with nothing to say which is current. An explicit
    key pins the slot so the wording can move underneath it. Re-running with
    unchanged wording is already safe — that path supersedes correctly — so this
    matters exactly when the number is inside the text, which is the whole point
    of writing it there.
  - **Still pass `over=`.** The sentence is asserted; `checked_over` is
    executed. Keep both, so the number is measured as well as stated — and a
    disagreement between what you wrote and what ran is itself a finding.

  A claim with no count is not a claim with a good count. A count in the wrong
  unit is worse than none, because it lends authority to the wrong number.

  The shape to watch for is a fused claim. Several conditions folded into one
  invariant will hold on every training pair, carried by the clauses that fire
  in nearly every grid, while the clause that actually decides the answer may
  have decided anything only rarely. A compound claim is verified as a unit and
  passes on the strength of its easiest clause, so it says nothing about its
  thinnest one — and in a ledger it reads exactly like the claims that carry
  counts.

- **A multi-line check? Pass `evidence=`.** The expression capture reads one
  expression at the call site, so a check that is really a function leaves you
  choosing between a bare name and an unreadable comprehension. Neither records
  anything useful. Instead:

  ```python
  runs = {r: longest_run(row) for r, row in enumerate(grid)}
  arc.verify("every row's longest run is odd",
             all(n % 2 == 1 for n in runs.values()),
             evidence=runs)
  ```

  `note=` is prose about the finding; `evidence=` is the value you measured.
- **Re-run an invariant against your own predictions with `over=`.** Write the
  predicate once and point it at different grids, rather than copying the check
  into an audit script where it can drift from what the ledger says:

  ```python
  def width_is_12(grid): return len(grid[0]) == 12

  arc.verify("every training output is 12 wide", width_is_12,
             over=[s["output"] for s in train_samples])
  ...
  arc.verify("my test prediction is 12 wide", width_is_12, over=[prediction])
  ```
- **Record from a script, not from `python -c`.** `verify()` recovers evidence
  by reading your script's source; from a `-c` one-liner or a heredoc there is
  no source to read, so the entry carries the claim and nothing about what ran,
  and the constant-condition check cannot fire at all. Such entries are marked
  `NO EVIDENCE`.
- **Sweeping many readings at once? Record it as one entry with `arc.sweep()`.**
  When you score a dozen candidate rules against training in a single pass, that
  is one finding, not a dozen — and it is usually the highest-yield thing in the
  run.

  ```python
  survivors = arc.sweep("which object supplies the output colour?", {
      "the largest one":       fits(by_area),
      "the one with a hole":   fits(by_holes),
      "the asymmetric one":    fits(by_asymmetry),
  })
  ```

  Pass `(survives, detail)` instead of a bare bool to keep whatever makes the
  entry self-explaining after a compaction — the score that produced the verdict
  (`{"the largest one": (True, "holds on every training pair")}`), or, when
  several readings survive, whether they still agree once you run them out of
  sample (`(True, "chooses a different object here than 'the largest one'
  does")`). The second is usually the more useful, because a sweep with more than one survivor
  is by definition a hedging obligation and the next question is always whether
  they diverge.

  If more than one reading survives, that is a **hedging obligation** you have
  found before spending any budget: run the survivors through `arc.rival()` and
  see which diverge on the test input.
- **Retraction.** If a check was wrong — a tautology, say — withdraw it with
  `arc.verify("the claim", retract=True)`, or `arc.refute("the claim",
  retract=True)` for a dead end. It then disappears from `gate.py status`. Do this rather than leaving it: the ledger is only worth reading
  because everything in it has been executed, and one claim that merely looks
  verified devalues all of them.

## Rival readings

When you implement an alternative interpretation in order to compare it —
usually right before discarding it — register it:

```python
from arc import rival
def strict(grid): ...          # the reading you suspect is wrong
rival("an object touching the frame is deleted, not recoloured", strict)
```

`rival()` scores it against every training pair, and if `solution/solve.py`
already exists it compares predictions and tells you immediately whether the
rival diverges — that is, whether spending the second slot on it would change
anything. Register rivals **once `solve.py` exists but before you submit**:
that comparison is the useful half of the output, and without a solution to
compare against all `rival()` can say is that there is nothing to compare yet.
Everything before submission is free, so there is no cost to waiting until you
have something to hold the rival up against.

The shape to watch for: you rule out a rival not with an example it fails, but
because its answer on the unlabelled input looks wrong to you — and "looks
wrong" rests on a regularity you noticed rather than one anything checked. That
is a preference, not a refutation. It leaves the second slot empty for a reading
the evidence never contradicted, with the rival already implemented and already
paid for. Killing an alternative with a training pair it fails is a proof; killing
it with an out-of-sample extrapolation is not.

**A refuted rival is not a closed question, and it is never a reason for an
empty slot.** Killing a rival with a training pair it fails is a proof about
*that rival, as stated*. It says nothing about the same idea carrying a
condition. "Always X" and "never X" are the two endpoints of a spectrum; a
counterexample that kills one endpoint leaves the entire interior — every
conditional form of X — untouched, and the interior is where an
under-determined rule usually lives. The counterexample you killed it with is
the object that tells you where the condition goes. Read it before you move on.

Two rules follow, and the second one costs nothing:

- **When two occasions look identical, compare what the branch would DO.** The
  hard case is not a rival you never thought of. It is a rival you thought of,
  went looking for a condition on, and could not find one for. Before you accept
  that verdict, check *where you looked*. If two occasions are indistinguishable
  at the point of decision — same neighbourhood, same immediate state — then no
  amount of staring at that point can separate them, and "I could not find
  another reading" means only "there is no LOCAL reading". Simulate instead: run
  the alternative branch forward from each occasion and compare the *finished
  grids*. The everyday version: two records a comparator calls equal are
  indistinguishable at the moment of comparison, and the tie-break you chose is
  invisible there — run the sort all the way out and they land in different
  places, and only the finished order tells you which rule you actually
  implemented. What separates two occasions can live far from the decision, in
  space or in the order of operations, and be invisible at it.

  The shape to watch for is a search that could not have succeeded. You count
  occasions, isolate the single clause the answer turns on, go looking for a
  conditional form of the rival you refuted, and write *"I could not find any
  other reading of that occasion."* If the two occasions are identical in
  everything local to them, no amount of looking there could have found one —
  the separator was never at the decision point. Run each reading all the way
  out and compare the endings instead. Endings can differ where starting points
  do not.

- **A refinement competes for the second slot, never for the first.** You may
  not rescue a dead rival by bolting a guard onto it and promoting it — that
  way nothing is ever refuted and the ledger stops meaning anything.
  `solve.py` keeps the reading that survived on its own. A refinement's prize
  is the hedge slot, so you need at most as many refinements as you have empty
  slots: one probe, not a search.
- **The second slot is a right you hold, it is free, and it is yours to spend.**
  Two answers are allowed. The first belongs to the reading that survived on its
  own. The second belongs to the strongest reading you could not rule out — and
  a rival that a training pair refuted is still a candidate for it, because
  "refuted" means "wrong about that pair", not "worth nothing". An empty slot
  scores nothing every time. A rival you implemented and ran scores whenever it
  happens to be right.

  This is not permission to pad. A perturbation you never believed is worth as
  little as an empty slot, and you should refuse it — the test is whether the
  thing in the slot is an interpretation you actually constructed and executed.
  Choose among those: it must predict something different from your first
  answer, because a rival that predicts the same grid is a wasted slot and not a
  hedge. Among the ones that differ, take the one that came closest to
  surviving — best training score first, then the *smallest* divergence. A rival
  that misses your answer by a few cells is a likelier correction than one that
  rewrites half the grid.

  If you decide to leave the slot empty, write the reason in `audit.md` and name
  the rival you passed over. That sentence is the difference between a judgement
  you made and an option you forgot you had.

## What training witnesses, and how often

`arc.unreached()` reports the lines of your solution that no training pair ever
executes, and its docstring carries the reason: training is the only thing
standing behind your solution, and it can only vouch for code it runs. Run it
before you accept. Run it **twice** if you shipped a second candidate — a hedge
is unreached code almost by definition, since it exists precisely because
training cannot discriminate it, and it ships with whatever bugs it has.

But `unreached()` is a set, not a histogram. It records *whether* a line ran and
discards *how often*. A line that ran once sits in it exactly like a line that
ran eight hundred times, so it is silent about the branch training touched once
— which is the branch a test input is most likely to turn on.

So write the table it cannot. One row per situation your rule can meet, and for
each: does training witness it, **how many times**, and does the test exercise it?

Here is the shape, on a rule that has nothing to do with any puzzle — parsing a
date string:

    | situation the rule can meet  | witnessed in the examples?      |
    |------------------------------|---------------------------------|
    | four-digit year              | yes, many                       |
    | two-digit year               | yes, a few                      |
    | no year at all               | yes, once — and the test needs it|
    | an impossible date           | NO — does the test need it?     |

Write your own rows: one per situation your rule can actually meet, named in the
terms of the rule you are testing, with the real count beside each.

A row reading **once, and the test uses it** is worth more than the rest of the
table put together. A row reading **NO** is where a hedge belongs. A row whose
count is **one** is where a rival belongs — and where you should apply the rule
above about comparing what the branch would do.


## Rules the gate enforces

- `solution/hypothesis.md` must exist and be at least __MIN_HYPOTHESIS_CHARS__
  characters before any submission.
- If `solution/solve.py` changed, `solution/hypothesis.md` must have changed
  too. A code change is a change of rule, implementation, or edge case — say
  which. (A refusal here does not cost an iteration.)
- `solve(grid)` receives **only** its argument. It cannot reach the puzzle data,
  and embedding a training output as a literal will be reported.
- `solve(grid)` should return one grid for training inputs; up to
  __MAX_CANDIDATES__ candidate grids are allowed for test inputs, and only for a
  real ambiguity. Returning extras on a training input is **not refused** —
  training validation simply scores the first and discards the rest — but an
  ambiguity on an example whose correct output you can *see* means the rule is
  unfinished. Resolve it rather than hedging. You do not need to restructure
  `solve()` to guarantee a single training candidate; you need the first one to
  be right.
- Budget: **__MAX_ITERATIONS__ submissions.** Over the last __BEST_EFFORT__ of
  them the train-100% requirement is lifted so you can commit to your best
  reading of the puzzle.
- `accept` requires `solution/audit.md` with a `CONFIDENCE: <1-5>` line and a
  `DECISION: ACCEPT` line.

## Working habits

- One question per exploration script, named for the question it answers.
- Shared helpers go in `explore/lib.py`; another script imports them with
  `from lib import ...` (scripts under `explore/` can import each other by bare
  module name — except when the name starts with a digit, which is not a legal
  identifier. `arc.explore_module("04_object_sizes")` reaches those, so you never
  need `importlib` boilerplate). Guard any report in an imported script behind
  `if __name__ == "__main__":` — otherwise its output re-prints into every
  downstream run and pollutes your context.
- **`solution/solve.py` must stand alone**, so helpers you developed in
  `explore/lib.py` have to be inlined into it. That duplication can drift: an
  experiment can pass against `lib.py` while the code that actually ships says
  something subtly different. After inlining, re-check the shipped version —
  `__PYTHON__ dryrun.py` runs `solution/solve.py` itself, and
  `arc.load_solution()` hands you that same function for use in an experiment.
- To audit your own predictions, load the real solution rather than a copy:

  ```python
  from arc import load_solution, test_samples, verify
  solve = load_solution()
  prediction = solve(test_samples[0]["input"])
  ```

  `arc.solution_module()` returns the whole module when you need its helpers —
  building a rival that reuses the shipped parse rather than re-deriving it.
  **`solution/` is not a package**: `from solution.solve import _helper` raises
  `ModuleNotFoundError`, which is the natural next thought once you have inlined
  helpers into `solve.py`. Reach into it with `arc.solution_module()` instead.

- Print summaries, not grids you have already seen. A boolean, a count, or a
  set of shapes usually carries the finding.
- `arc.png()` writes relative to the current directory, so `cd` into the
  workspace first if you invoke it from elsewhere.

## Out of bounds

- **There is no network.** The run is closed at the operating-system level and
  the routing table is empty. A connection that fails is not a bug and not
  something to diagnose. Do not spend a turn on it. ARC needs no outside
  knowledge, and this task is fully specified by the files you already have.
- **You can see only this run's own directory.** Everything else is removed
  from view, not forbidden by a rule. A path outside it does not exist for you,
  so there is nothing to try and nothing to work around.
- **You have the full Claude Code toolset**, including `Task` (sub-agents),
  `Workflow`, `Skill` and background tasks. Use whatever genuinely helps. A
  sub-agent that reviews your solution against `hypothesis.md` without seeing
  your reasoning is a legitimate and well-tested use; so is farming out an
  expensive sweep. Nothing here asks you to work single-threaded.
- What you may **not** do is reach outside this run: no web search or fetch, no
  publishing, no notifications, no scheduling work that outlives the run. Those
  tools are denied rather than discouraged. Any sub-agent you spawn inherits the
  same restriction and the same workspace.

## When context gets compacted

Exploratory interpreter state and your transcript do not survive it.
`.ccarc/invariants.jsonl` and `NOTES.md` do. Keep `NOTES.md` written for a
fresh version of yourself: current hypothesis, confirmed facts, refuted
hypotheses *and why*, and the next experiment you meant to run.
