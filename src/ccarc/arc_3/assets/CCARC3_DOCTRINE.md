## How you are scored

**Clear every level, in as few actions as possible. Every action counts,
including the ones you spend working out what to do.**

You get one attempt at this game.

## Driving it

```python
from session import client, gate, arc

client.reset()                  # start
client.act(1)                   # ACTION1..5,7 take no arguments
client.act(6, x=10, y=20)       # ACTION6 is a click; x,y in [0,63]
client.status()                 # level, state, actions
client.last                     # the transition your last action produced
client.history()                # every transition so far
```

Each frame carries `available_actions`: exactly which actions this game accepts.
`win_levels` is how many levels the game has.

**The game persists across commands.** Each `python -c ...` is a new process;
importing `session` resumes the same game from disk.

**The game is deterministic: the same (state, action) pair always produces the
same next state.**

## The history

`client.history()` returns every action you have taken, in order. Each entry
carries the board before, the action, the board after, and:

* `t.state` — the state that action produced.
* `t.full_reset` — this action returned the game to level 0.
* `t.crosses_level` — this action cleared a level; `t.level` is the level it cleared.
* `t.board_replaced` — `before` and `after` are different boards rather than two
  states of one board. True on a level advance and on a full reset.
* `t.intermediate` — every grid this action rendered.
* `t.changed` — whether the board changed.

**Reading `client.last` takes constant time. `client.history()` takes time
linear in `n`, the number of actions you have taken so far.** Use `last`
inside a loop; use `history()` for questions about the whole game.

## Analysing

Every action and its frames land in `trace.jsonl` without you asking. Write
code to ask questions of the ledger: it is a list of transitions, in memory, in
your own process.

```python
t  = client.last                # the transition your last action produced
ts = client.history()           # every transition so far
t.before, t.after               # 64x64 numpy arrays, read-only
```

`t.after.copy()` gives you one you can write into.

`print(arc.render(board))` is one character per cell;
`arc.png(board, "notes/now.png", scale=8)` writes an image you open with the
Read tool.

### Any `Bash` call that plays actions must pass an explicit `timeout`.

**`Bash` defaults to 120 seconds and does not warn you.** A shell `timeout 300`
inside the command does not help — the tool's own limit fires first, at 120 s,
and one of two things happens:

- the call is **moved to the background**. Your loop keeps running and keeps
  spending real actions on the real game, but you no longer see its output. You
  are now flying blind on a board that is still changing.
- the call is **killed** — `Exit code 143` — mid-loop, leaving the game in
  whatever state the last completed action produced.

The tool accepts `timeout` in **milliseconds, up to 600000** (10 minutes). Pass it
whenever a call drives the game or runs a search:

```python
# Bash(command=..., timeout=600000)
from session import client, gate, arc
for action in [1, 3, 3, 5]:
    client.act(action)
print(client.status())
```

If the work genuinely needs more than ten minutes, **split it** — drive a dozen
actions, print `client.status()`, return, and continue in the next call.

## Level boundaries

When you complete a level you will be **refused** the next action until you
call:

```python
gate.acknowledge(
    "what level N established",
    mechanics=["what you believe and are carrying forward"],
    refuted=["tested and found false"],
    untested=["never exercised — nothing known either way"],
)
```

That writes `rules.json`. `notes/` is a directory you can write anything into.
Both survive across processes.

## Constraints

- **Play the game through `client`, and only through `client`.**
- **Do not use the network.** The run is closed at the operating-system level;
  nothing outside the game and your own model inference is reachable.
- **You can see only this run's own directory.** A path outside it does not
  exist for you.
- **Leave the ledger where it is.** Do not move, alter or delete `trace.jsonl`
  or `trace.state.json` at the workspace root; they are the only record the
  harness scores. Copying into `notes/` is fine; moving the originals ends the
  scored run.
- **Do not start a second client.** Import `client` from `session`; do not
  construct another `ArcClient` or write a `trace.jsonl` under `notes/`. Those
  plays land on a different scorecard and do not count.
