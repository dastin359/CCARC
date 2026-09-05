# CCARC

A harness for driving general coding agents at ARC-AGI, built on two
mechanisms:

**Code execution as first-class reasoning.** The agent does not stare at a grid
and intuit an answer. It writes a short program, runs it, and reads what came
back. A snippet that executes and returns a concrete result replaces pages of
chain-of-thought enumeration, and that compression compounds across iterations.
On the interactive ARC-AGI-3 games the same move becomes play: the agent builds
forward models, search routines and route executors as code, and the harness
prices every action so that executed knowledge — not narrated belief — is what
scores.

**An explicit, file-based memory system.** The harness makes the agent write
down what it has worked out, in named files on a real filesystem. A fact stays
addressable, survives a context compaction, and can be read back after the run.
Neither driver lets a run continue until it is written.

On ARC-AGI-3 the agent has to record what a level taught it — what it proved,
what it ruled out, what it has not tested — in `rules.json` before it can play
the next one. On ARC-AGI-2 it writes the rule out in `solution/hypothesis.md`,
and the gate will not take its code without it.

## Results

| Benchmark | Set | Result | Model | Cost | Where to check it |
|---|---|---|---|---|---|
| ARC-AGI-3 | 25 / 25 public games, 183 / 183 levels | **RHAE 100** · 7,809 actions | Claude Opus 5, effort `high` | $614.65 | [scorecard `7fb310c1`](https://arcprize.org/scorecards/7fb310c1-3801-4b9b-a215-3f84a8b035ff), competition mode |
| ARC-AGI-2 | 120 / 120 public eval tasks | **116.1667 points = 96.81%** | Claude Opus 5 | $570.49 | submitted to the ARC-AGI Community Leaderboard |

Both costs are the CLI's `total_cost_usd` over the winning runs, which is an
API-list-price equivalent. Each benchmark run's token usage lies comfortably
within Claude Code Max 20x's weekly quota, so the effective cost of running
CCARC on either benchmark is ≤ $200.

## What is here

| | |
|---|---|
| `src/ccarc/arc_3` | ARC-AGI-3 driver: interactive games, scored on action efficiency |
| `src/ccarc/arc_3/rules.py` | the rule book — proven / refuted / untested game mechanics |
| `src/ccarc/arc_3/ledger.py` | append-only action ledger; every action the client records is appended, and the record can be replayed for audit |
| `src/ccarc/arc_2` | ARC-AGI-2 driver: static puzzles, scored on partial credit |
| `src/ccarc/arc_2/assets` | everything an ARC-AGI-2 solver reads — the doctrine, the workspace guide, the toolkit it imports as `arc` |
| `src/ccarc/guardrail/` | the four layers behind the section below: filesystem, network, egress allowlist, response filtering. The first three contain both drivers; `arc_proxy` is the ARC-AGI-3 shim that holds the API key and withholds the baselines |
| `src/ccarc/staging.py` | builds the copy of this package a solver may import — one driver's, never both |

## Reward hacking guardrails

Whichever benchmark it plays, the agent is contained the same way. None of
what follows is a policy it is asked to respect.

**No network of its own.** The run executes inside a network namespace with an
empty routing table, except for `api.anthropic.com` and `platform.claude.com`.

**No filesystem outside its workspace.** A mount namespace re-binds only the
run's own tree, the staged package and the interpreter.

### ARC-AGI-3 specific guardrail

The prompt teaches one general method — write code to test a hypothesis, record
what it establishes in named fields on disk — and says nothing about these
games: no rules or mechanics, no pre-built analysis helpers, no tactics carried
over from earlier runs, no human baseline, no action budget, and no way to look
any of it up. Every environment is worked out by playing it.

### ARC-AGI-2 specific guardrail

The task file the solver is given carries the training pairs in full and the
test *inputs* only. The ground truth is never written into the workspace and is
never in the prompt.

## Compared to the official ARC-AGI-3 baseline

Everything except the harness is held constant: same model, same reasoning
effort, same 25 environments, same metric. ARC Prize publish Claude Opus 5 at
high reasoning effort on this set; averaged per environment, their result is
40.7%.

| Environment | ARC Prize Official Baseline, Opus 5 `high` | CCARC, Opus 5 `high` |
|---|---|---|
| `bp35` | 0.0 | **100** |
| `g50t` | 0.0 | **100** |
| `sc25` | 0.0 | **100** |
| `cd82` | 0.6 | **100** |
| `lf52` | 1.8 | **100** |
| `tu93` | 3.6 | **100** |
| `ls20` | 5.4 | **100** |
| `tn36` | 6.0 | **100** |
| `sk48` | 8.3 | **100** |
| `ka59` | 10.4 | **100** |
| `su15` | 10.4 | **100** |
| `wa30` | 10.6 | **100** |
| `m0r0` | 28.6 | **100** |
| `dc22` | 44.8 | **100** |
| `cn04` | 47.6 | **100** |
| `tr87` | 47.6 | **100** |
| `sp80` | 56.3 | **100** |
| `re86` | 58.3 | **100** |
| `sb26` | 77.8 | **100** |
| `vc33` | 98.8 | **100** |
| `ar25` | 100.0 | **100** |
| `ft09` | 100.0 | **100** |
| `lp85` | 100.0 | **100** |
| `r11l` | 100.0 | **100** |
| `s5i5` | 100.0 | **100** |
| **mean** | **40.7** | **100** |

Five environments where the same model at the same reasoning effort scored
essentially nothing — `bp35`, `g50t` and `sc25` at zero, `cd82` at 0.6, `lf52`
at 1.8 — are cleared here at the ceiling.

## Getting started

    pip install -e .

## Running an ARC-AGI-3 game

    export ARC_API_KEY=...      # from arcprize.org; the proxy holds it and
                                # injects it upstream
    ccarc arc-3 games           # what is available
    ccarc arc-3 run --game <GAME_ID>
    ccarc arc-3 report --out-dir runs/ccarc3

## Running an ARC-AGI-2 task

    export ARC_DATA_ROOT=/path/to/ARC-AGI-2    # the dataset, cloned separately

    ccarc arc-2 run <TASK_ID> --out-dir cc_runs
    ccarc arc-2 batch --all --out-dir cc_runs
    ccarc arc-2 score cc_runs

**Requirements.** Linux with unprivileged user namespaces, and the Claude Code
CLI on `PATH`. Without the namespaces the ARC-AGI-3 driver refuses to launch —
set `CCARC3_NETGUARD=0` and `CCARC3_CONFINE=0` to run it anyway, with the agent
not contained by the reward hacking guardrail. The ARC-AGI-2 driver runs
uncontained and says so in its result.

