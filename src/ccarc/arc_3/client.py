"""Live ARC-AGI-3 client — the surface a solver actually drives.

Written directly against the HTTP API rather than through ``arc_agi_3``. That is
not a preference: the installed SDK declares ``FrameData.score`` while the server
sends ``levels_completed``, so pydantic fills the default and **every score reads
0 forever, silently**. Speaking HTTP is the correct call here, and it keeps
this package free of any dependency beyond numpy.

What the client owns, so the solver does not have to:

- session plumbing (``card_id``, ``guid``, retries)
- appending every frame to the transition ledger, unprompted
- level tracking from ``levels_completed``
- the two moves that silently destroy progress, refused before they are sent

What the solver owns: which action to take, and why.
"""

from __future__ import annotations

import hashlib
import http.cookiejar
import http.client
import json
import os
import time
import warnings
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .gate import GateRefusal, LevelGate
from .ledger import TraceWriter, action_name

__all__ = [
    "ROOT_URL",
    "ArcClient",
    "GameInfo",
    "ActionRefused",
    "GateRefusal",
    "list_games",
    "baselines_for",
]

# **Redirectable, so the key can live somewhere the solver cannot read it.**
# Unset, this is the live API -- the case for a harness-side process that holds
# the key itself. Set, it points at `arc_proxy`, which holds the real key
# and forwards only the four endpoints a solver needs -- refusing `/api/games`,
# the one that carries `baseline_actions`.
#
# **This module is re-imported on every action.** The workspace tells the solver
# "each `python -c ...` is a new process", so a change here is live for games
# already running, not just future ones. Both branches below therefore have to be
# safe for a solver mid-game.
ROOT_URL = os.environ.get("CCARC3_ARC_ROOT") or "https://three.arcprize.org"

_TERMINAL = ("WIN", "GAME_OVER")


class ActionRefused(RuntimeError):
    """The harness declined an action the solver asked for.

    Distinct from a transport error: the request was never sent and the game
    did not step. Carries the reason so the solver can act on it.
    """


def new_session() -> urllib.request.OpenerDirector:
    """An opener with its own cookie jar.

    **The API binds a scorecard to the HTTP session, not to the API key.** Open a
    scorecard on one connection and send RESET on another and the server reports
    ``game <id> not found`` -- a message that points at the game and means the
    session. Closing then fails with ``scorecard <id> not found`` for the same
    reason. Both are cured by carrying cookies, which is why the stock SDK
    threads a ``RequestsCookieJar`` from ``Swarm`` into every ``Agent``.

    Bare ``urllib.request.urlopen`` uses no cookie jar, so every call would be a
    new session. Each client keeps one of these for its whole lifetime.
    """
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


# **Stamped on every request this client makes, so the shim can tell a
# client-driven action from a hand-rolled one.** Every guard that prevents a
# run-destroying move lives in this class, and the ledger is written here too —
# so an action sent to the shim by raw HTTP would arrive with neither. The shim
# therefore logs a loud `BARE ACTION` line for any `/api/cmd/*` arriving without
# this header: a hand-rolled action is loud in the shim log, not invisible. The
# header carries no information — its presence is the whole signal — and ARC
# ignores unknown headers, so it is harmless on a direct connection.
_CLIENT_MARK = "X-CCARC3-Client"


def _post(
    url: str,
    payload: dict[str, Any],
    key: str,
    *,
    retries: int = 3,
    opener: urllib.request.OpenerDirector | None = None,
) -> dict[str, Any]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={"X-API-Key": key, "Accept": "application/json",
                 "Content-Type": "application/json", _CLIENT_MARK: "1"},
        method="POST",
    )
    return _send(req, retries, opener)


def _get(
    url: str,
    key: str,
    *,
    retries: int = 3,
    opener: urllib.request.OpenerDirector | None = None,
) -> Any:
    req = urllib.request.Request(
        url,
        headers={"X-API-Key": key, "Accept": "application/json", _CLIENT_MARK: "1"},
        method="GET",
    )
    return _send(req, retries, opener)


#: How many times a `429` may be waited out before it is treated as fatal.
#:
#: **A rate limit is "later", not "wrong", and must never reach the fatal
#: 4xx branch.** A `429` means the request was REJECTED, not
#: executed: nothing changed server-side, so repeating it is safe, and it is the
#: one 4xx where repeating it is also correct. Sent to the fatal branch it
#: ends whatever is in flight -- under concurrency several clients can draw a
#: `429` apiece and each would lose its entire game to one rejected request.
#:
#: Given its own budget rather than sharing `retries`: the transport budget
#: exists for a flaky connection, and spending it on throttling would leave
#: nothing for the failure it was meant to absorb.
_RATE_LIMIT_RETRIES = 8
_RATE_LIMIT_MAX_WAIT = 30.0


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    """The server's own instruction in seconds, or None if it did not give one.

    `None` rather than `0.0` for absent, because a server that says
    ``Retry-After: 0`` is saying *now* -- and `0.0 or <backoff>` silently
    reads that as "no instruction" and waits anyway.
    """
    raw = exc.headers.get("Retry-After") if exc.headers else None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None


#: Extra `_send` attempts across the process: sends beyond the first for one
#: request. Read it with `transport_retries()`; a caller reconciling a request
#: count takes the difference across the window it cares about.
_RETRIED = [0]


def transport_retries() -> int:
    """How many times `_send` has re-sent a request in this process."""
    return _RETRIED[0]


def _send(
    req: urllib.request.Request,
    retries: int,
    opener: urllib.request.OpenerDirector | None = None,
) -> Any:
    do_open = (opener or urllib.request).open if opener else urllib.request.urlopen
    last: Exception | None = None
    throttled = 0
    attempt = 0
    sends = 0
    while attempt < retries:
        # **Every send after the first is a REQUEST THE SERVER MAY HAVE APPLIED.**
        # `/api/cmd/*` is not idempotent: a gateway can fail after the upstream
        # has already performed the action, so the re-send applies it a second
        # time and the billed count comes back above the steps intended. Every
        # frame can still match, because the repeat may happen to be a no-op in
        # that state, so nothing downstream notices.
        #
        # Counting them is what lets a caller tell an explained extra request
        # from an unexplained one. `replay`'s post-condition refuses on the
        # difference, and without this number it could not say which it is
        # looking at -- it would refuse a replay that had rebuilt the position
        # perfectly.
        if sends:
            _RETRIED[0] += 1
        sends += 1
        try:
            with do_open(req, timeout=30) as fh:
                return json.loads(fh.read().decode())
        except urllib.error.HTTPError as exc:
            # A bare "HTTP Error 400: Bad Request" is undebuggable, and this API
            # puts the actual reason in the response body. Read it before the
            # handle closes -- it is the only place the reason exists.
            try:
                detail = exc.read().decode()[:500]
            except Exception:  # noqa: BLE001 -- body may already be consumed
                detail = "<no body>"
            if exc.code == 429 and throttled < _RATE_LIMIT_RETRIES:
                throttled += 1
                told = _retry_after(exc)
                time.sleep(told if told is not None
                           else min(_RATE_LIMIT_MAX_WAIT, 2.0**throttled))
                continue                     # not an attempt; see the note above
            if 400 <= exc.code < 500:
                raise RuntimeError(
                    f"{req.get_method()} {req.full_url} -> {exc.code}: {detail}"
                ) from exc
            last = RuntimeError(f"{exc.code}: {detail}")
            time.sleep(2**attempt)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                # **A dropped connection is a transport fault like the others,
                # and the easiest one to leave uncovered.**
                # `http.client.RemoteDisconnected` is `(ConnectionResetError,
                # BadStatusLine)`. urllib converts only `h.request(...)` errors
                # into `URLError`; `h.getresponse()` -- where `_read_status`
                # runs -- sits outside that conversion and is re-raised
                # unwrapped. Unhandled, the fault escapes `_send` entirely,
                # kills the solver's whole process on the first attempt, and
                # leaves `transport_retries()` reading 0 as though nothing had
                # happened -- one such drop can end a replay of hundreds of
                # requests.
                #
                # Retrying it re-sends a non-idempotent `/api/cmd/*`, exactly as
                # the 5xx branch above already does, and for the same reason:
                # a re-send that may double-apply is better than a run that
                # ends. `_RETRIED` counts it, so `replay`'s post-condition can
                # still tell an explained extra request from an unexplained one.
                ConnectionError, http.client.HTTPException) as exc:
            last = exc
            time.sleep(2**attempt)
        attempt += 1
    raise RuntimeError(f"request failed after {retries} attempts: {last}")


def _api_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("ARC_API_KEY", "")
    if key:
        return key
    # Behind the proxy the solver has no key by design -- a key in its
    # environment would let it reach upstream endpoints by hand. The proxy
    # injects the real one. Send a placeholder so the request still carries
    # the header the API expects.
    if os.environ.get("CCARC3_ARC_ROOT"):
        return "proxied"
    raise RuntimeError("no ARC_API_KEY; the live API returns 401 without one")


# Policy: the cap is `max(5 x baseline_total, BUDGET_FLOOR)`.
# The multiple is the caller's (`Ccarc3Config.budget_multiple`, 5.0); this is the
# floor beneath it. Harness-side only -- `strip_baselines` keeps every derived
# cap out of the workspace and the child environment.
# The floor is env-overridable so one run can be granted a bigger allowance
# without touching the multiple -- a multiple override written into a launch
# script would encode the withheld per-game figure; a flat floor encodes
# nothing about any game.
BUDGET_FLOOR = int(os.environ.get("CCARC3_BUDGET_FLOOR") or 5000)


@dataclass(frozen=True)
class GameInfo:
    """A row from ``GET /api/games``."""

    game_id: str
    title: str = ""
    tags: tuple[str, ...] = ()
    baseline_actions: tuple[int, ...] = ()

    @property
    def levels(self) -> int:
        return len(self.baseline_actions)

    @property
    def baseline_total(self) -> int:
        return sum(self.baseline_actions)

    def baseline_for(self, level: int) -> int | None:
        """The published per-level figure for ``level``, or ``None`` past its end."""
        if 0 <= level < len(self.baseline_actions):
            return self.baseline_actions[level]
        return None

    def suggested_budget(self, multiple: float = 4.0) -> int:
        # A flat ceiling cannot work: real games differ by an order of magnitude
        #
        # **Why a floor exists beneath the multiple.** A game that offers
        # only a click can demand an exhaustive enumeration of a large control
        # space -- mechanical, correct, and affordable -- and a short game's
        # baseline times the multiple can come out smaller than one such
        # sweep, so the ceiling would bind before the game did. The floor
        # keeps a short game's baseline from setting the ceiling on its own.
        #
        # It also narrows the inversion the strip exists to stop. `cap /
        # multiple` recovers the baseline total exactly, which is why
        # `strip_baselines` removes `action_budget` and `CCARC3_MAX_ACTIONS`
        # from everything the solver reads. At the floor the cap is a
        # constant and carries no game-specific bit at all.
        """A ceiling derived from the game rather than guessed. Harness-side only."""
        return max(BUDGET_FLOOR, int(self.baseline_total * multiple))


HIDE_BASELINES_ENV = "CCARC3_HIDE_BASELINES"

#: Sentinel: the one-game endpoint has not been asked yet. Distinct from `None`,
#: which is the answer "asked, and nothing was disclosed".
_UNFETCHED = object()

# Policy: the solver must never be made aware of the existence of the
# methods and properties it should not touch.
#
# Withholding the VALUE is sound as far as it goes -- the baseline-bearing
# endpoint is off the shim's allowlist, `/api/scorecard/close` bodies are
# scrubbed, and all three
# return None for a solver even when the client is built without the gate. But a
# name in `dir()` or `help()` is itself a disclosure: it says a reference count
# and a score exist, are known here, and are being kept back. The rest of the
# withholding is built to not say that.
#
# They stay on the class -- `status()` computes from them and the harness reads
# them -- and leave the offered surface. A module constant rather than a class
# attribute because a class attribute appears in `help()`, which is the exact
# leak this closes. Comments do not survive `staging.stage_package`, so this
# reasoning is not in the copy a solver imports.
# `__dir__` on a class filters `dir(instance)` and NOT `dir(TheClass)` -- Python
# looks the latter up on the metaclass. The guide hands the solver an instance,
# so the instance path is the one that matters, but the class is named in the
# guide too, and without the metaclass `dir(ArcClient)` would still list every
# hidden name. The guard runs against the STAGED package, which is the copy
# that matters.
class _Offered(type):
    def __dir__(cls):
        return sorted(n for n in super().__dir__() if n not in _NOT_OFFERED)


_NOT_OFFERED = frozenset({
    # Hidden by DEFAULT and revealed by `ArcClient.__dir__` only on an instance
    # whose shim disclosed a baseline. See that method. `_fetch_baselines` is
    # hidden unconditionally: it is machinery, and its NAME alone says a
    # reference count exists and is fetched from somewhere.
    "baseline_actions", "_fetch_baselines", "_baselines",
    "_baseline_here", "_baseline_here_enforced",
    "quiet_pace", "show_score",
    # `completion_cap` is computable without baselines and therefore harmless
    # to compute, but it is a term of the score.
    "completion_cap",

    # **Policy: equip the solver with what it needs, and nothing else --
    # what the solver does not need to know is not made visible to it, even
    # when it leaks nothing.**
    #
    # These are harness bookkeeping, credentials, or paths to them; none
    # answers a question about the game.
    #
    #   api_key       the ARC credential, as an attribute
    #   session_path  the path to the cookie jar -- kept OUT of the workspace,
    #                 because session cookies are live credentials. Hiding the
    #                 file and then publishing a signpost to it would be half a
    #                 fix.
    #   state_path / boundary_path / decision_path
    #                 harness sidecars the parent reads after the run
    #   info          carries `baseline_actions` -- the field NAME is the last
    #                 route by which the withheld quantity is advertised
    #   max_actions   the ceiling. Doctrine keeps it out of reach on purpose:
    #                 "a limit the solver can read is a limit the solver can
    #                 invert." It reads None here anyway.
    #   require_card / foreign_card / close_error / scorecard_error /
    #   card_plays_at_open
    #                 card plumbing owned by the harness
    "api_key", "session_path", "state_path", "boundary_path", "decision_path",
    "info", "max_actions", "require_card", "foreign_card", "close_error",
    "scorecard_error", "card_plays_at_open",

    # **Card and session lifecycle is the harness's, not the solver's.**
    # The GENERATED workspace `session.py` constructs the client and calls
    # `open()` itself (see `SESSION_TEMPLATE`); the solver does
    # `from session import client` and is handed one already open. `close` runs
    # only from `__exit__`, which a solver never reaches, and the shim refuses a
    # close on a lent card anyway. So neither is a call a solver has any reason
    # to make; a redundant `open()` is declined.
    #
    # `scorecard` reads the card's server record, which is the solver's OWN
    # actions, levels and plays -- every one already on this object. Card
    # liveness is verified harness-side by `_assert_server_agrees`.
    #
    # Hiding is not removing: these still work if called, so no solver code
    # breaks. They leave `dir()` and `help()`, which is where the guide points.
    "open", "close", "scorecard", "finish_declarations", "last_transition",

    # Harness plumbing and constants -- the proxy URL, the game handle, the card
    # bookkeeping, the reap stopwatch. None answers a question about the game.
    # `finish` STAYS: it is the solver's one explicit act of submission and the
    # harness asks for it by name.
    "root", "guid", "card_id", "tags", "last_touched",
    "REAP_DEADLINE_S", "SNAPSHOT_EVERY",

    # **Effectiveness counters, not offered.** These four count how many of a
    # level's actions changed nothing, repeated something already tried, or
    # returned the board to a state already seen. That is a reading of the
    # ledger, and reading the ledger for what worked is the solver's own work.
    # They stay for harness-side reporting; `status()` does not print them.
    "level_tried", "level_dead", "level_repeats", "level_revisits",
})
"""Set to ``1`` in a solver's environment to make :func:`list_games` refuse.

Every path in the package that builds a `GameInfo` from the API comes through
here, which is why the flag is applied at this one point. What it is for, and
what it cannot reach on its own, are deliberately not stated here: this
docstring renders on a page the solver can call up.

Scoped to the environment rather than a call argument because the solver's
process is where it must apply: the *runner* needs the real numbers to size a
game's budget and score the result.

Set on the child's environment by the caller that builds a baseline-free
workspace. Nothing in `session.py` sets it -- `build_cli_args` assembles
argv and never touches an environment at all -- so the flag guards the child
*process*, not the workspace file the child imports, and nothing sets it on
the runner.

**It is the whole rule, with no bypass.** Harness-side callers need no
exemption -- they live in a process where the flag is unset -- and an exemption
reachable from the solver's namespace would make the flag advisory. Under the
flag, no path in this package returns a baseline.

**Still not airtight, and should not be described as such.** A solver holding
the key can issue its own HTTP request to `/api/games`. What the flag closes is
*incidental* exposure — a name surfaced by ordinary orienting introspection to
a solver that went looking for nothing at all.
"""


# Harness-side only; raises under `HIDE_BASELINES_ENV`, which only a solver's
# environment carries. Not re-exported from `ccarc.arc_3`, so it is absent from
# `dir(arc)` in the namespace a solver holds.
def baselines_for(
    game_id: str, api_key: str | None = None, root: str = ROOT_URL
) -> tuple[int, ...]:
    """One game's per-level reference counts. Harness-side callers only."""
    if os.environ.get(HIDE_BASELINES_ENV) == "1":
        # **A refusal must not name the category of the thing it refuses.**
        # Flat is enough: the caller learns the call is not available here and
        # nothing else.
        raise PermissionError("baselines_for() is not available in this process.")
    for game in list_games(api_key=api_key, root=root):
        if game.game_id == game_id:
            return tuple(game.baseline_actions)
    raise KeyError(f"{game_id} is not in the public set")


# Per-level reference counts come back empty under `HIDE_BASELINES_ENV`. There
# is deliberately no bypass argument: a keyword any caller can pass is not a
# boundary, so the environment variable is the whole rule, honoured in exactly
# one place. Harness callers are unaffected -- the runner never sets the flag
# on itself, only on the child it spawns.
# One GET per (root, key) per process: several harness-side callers walk this
# list, and the memo keeps it to a single fetch. Hide is applied after the
# cache, so toggling HIDE_BASELINES_ENV does not need a second fetch.
_GAMES_RAW: dict[tuple[str, str], list] = {}


def list_games(api_key: str | None = None, root: str = ROOT_URL) -> list[GameInfo]:
    """Every public game, with its title and action-type tags."""
    # **Refused before any HTTP under the hide flag, and with no URL in the
    # message.** The child holds no key and its proxy 403s the endpoint, so the
    # call cannot succeed there in any case -- but a refusal that echoed the
    # endpoint and its status into a tool result would read as network contact
    # where none occurred. The one sentence below is all a solver ever sees.
    if os.environ.get(HIDE_BASELINES_ENV) == "1":
        raise RuntimeError(
            "list_games() is not available in this run. Play the game you were "
            "given, through the client you were given."
        )
    key = _api_key(api_key)
    cache_key = (root, key)
    raw = _GAMES_RAW.get(cache_key)
    if raw is None:
        raw = _get(f"{root}/api/games", key)
        _GAMES_RAW[cache_key] = raw
    hide = os.environ.get(HIDE_BASELINES_ENV) == "1"
    return [
        GameInfo(
            game_id=g["game_id"],
            title=g.get("title", ""),
            tags=tuple(g.get("tags") or ()),
            baseline_actions=() if hide else tuple(g.get("baseline_actions") or ()),
        )
        for g in raw
    ]


# The reminder states the scoring rule, which no amount of play reveals, and the
# two calls available at this frame. It states no figure, and it does not frame
# actions as an allowance being drawn down.
#: Every refusal the solver can receive, as templates rather than inline
#: literals: a refusal is solver-facing text, so it belongs where it can be
#: read and audited rather than reconstructed by triggering it.
#:
#: The rule they follow: state the API constraint, never the environment's rule.
#: A refusal costs nothing -- no action billed, no ledger row, unlimited retries
#: -- so an explanatory one is a free oracle a solver can farm.
REFUSALS = {
    # The guard is the only thing standing between the agent and silently
    # discarding the run, and this refusal is the only place the rule is stated
    # to the solver. What the message must carry is the API semantics of the
    # call it refuses and the switch that overrides it; the counter-zero
    # mechanism and the scoring consequence are the environment's business.
    "reset_would_start_a_play":
        "reset() is not accepted here. restart_for_replay() starts a new play.",
    "no_second_play":
        "restart_for_replay() is not available: this run is one attempt.",
    "replay_counter_not_zero":
        "restart_for_replay() is legal only while the server's action "
        "counter is zero, which is the state immediately after a level "
        "advance. It is not zero now.",
    "terminal_state":
        "state is {state}; only RESET is accepted.",
    "action6_needs_xy":
        "ACTION6 requires x and y",
    "action6_out_of_range":
        "ACTION6 coordinates must lie in [0, 63]; got ({x}, {y})",
    "not_available":
        "{name} is not in this game's available_actions {available}.",
    # **Unreachable on a scored run, and minimal anyway.** A scored workspace
    # constructs the client with `max_actions=0`, so the guard that raises this
    # is falsy and never fires; the ceiling that binds is the shim's, and its
    # refusal is what an agent actually meets. This remains for a client built
    # with a cap of its own, i.e. the local bench.
    "allowance_spent":
        "action allowance spent.",
}


# Plain text, with no harness banner: the other harness messages reach the
# solver through the same channel without one.
WIN_REPLAY_REMINDER = """
You cleared every level.

Each play is scored separately and the highest score among them stands.

restart_for_replay() plays again; finish() ends the run.
"""

FINISH_ACK = """
Your decision is recorded.
"""

FINISH_REAFFIRM_ACK = """
Your decision is reaffirmed and recorded.
"""

COMPETITION_ENV = "CCARC3_COMPETITION"


def competition_mode() -> bool:
    """Whether this run plays under ARC's competition rules. **On by default.**

    Competition mode is a property of the SCORECARD, not of the client: the
    official toolkit sets one field, `competition_mode: True`, on
    `/api/scorecard/open`, and ARC's server enforces the rest. Its published
    rules, which are what the rest of this module conditions on:

      * "Only *Level Resets* are permitted, *Game Resets* are not allowed and
        become *Level Resets*"
      * "Can only interact (call `make`) a single time for each environment"
      * "Can only open a single Scorecard"; `get_scorecard` does not work in-run
      * it is "REQUIRED to show up on the Unverified leaderboard"

    The consequence that reaches the agent: there is ONE play, so exploration
    is not free. Under best-of scoring a first play costs nothing, so working
    the game out without counting the cost is sound; here every action spent
    working it out is an action on the scored play.

    `CCARC3_COMPETITION=0` turns competition mode off for the local bench,
    where best-of scoring allows more than one play. It must not be set for
    anything that will be submitted.
    """
    return os.environ.get(COMPETITION_ENV, "1") != "0"


WIN_REMINDER_ENV = "CCARC3_WIN_REMINDER"
# **Off by default.** Importing this module must not start prompting solvers;
# the driver opts in explicitly by setting this in the child's environment.


LEVEL_ANNOUNCE_ENV = "CCARC3_LEVEL_ANNOUNCE"


def _level_announce_enabled() -> bool:
    """Whether a level advance is announced to the solver as a user message."""
    # On by default, and safe rather than bold: it reaches the solver only
    # through the outbox, and there is no outbox unless a driver is holding the
    # session open. A run without one is unchanged. Comment, not docstring:
    # the solver can neither set this default nor observe it.
    return os.environ.get(LEVEL_ANNOUNCE_ENV, "1") != "0"


def _win_reminder_enabled() -> bool:
    # Under competition rules there is no second play to offer, so the reminder
    # has nothing to say and firing it would advertise a call that refuses.
    if competition_mode():
        return False

    """Whether the post-clear reminder speaks at all.

    A switch, not a counter. The reminder fires on every transition into WIN.
    The principle: the agent alone decides whether to keep
    replaying; the harness states the facts when all levels are won, and nothing
    the harness does where the solver can observe it may depend on the score.
    """
    return os.environ.get(WIN_REMINDER_ENV, "").strip() == "1"


@dataclass
class ArcClient(metaclass=_Offered):
    """One game, one scorecard, one ledger.

    Usage::

        from session import client
        client.reset()
        while not client.done:
            client.act(1)
    """

    game_id: str
    trace_path: str | Path = "trace.jsonl"
    card_plays_at_open: int = -1
    """Plays the shared card already held for this game when this attempt began.

    -1 means unknown (no lent card, or the read failed). Under a shared card the
    server keeps appending to one ``cards[game_id]`` entry across *attempts*, so
    without this boundary a corroboration check cannot tell this attempt's rows
    from a discarded predecessor's.
    """
    last_touched: float = 0.0
    """Epoch seconds when the server last answered an action for this game.

    Zero on a client that has not acted, and absent from older state files --
    both of which are read as "unknown", never as "stale".
    """
    api_key: str | None = None
    root: str = ROOT_URL
    tags: tuple[str, ...] = ("ccarc3",)
    info: GameInfo | None = None
    gate: LevelGate | None = None
    max_actions: int = 0
    """Hard action cap across the whole game. 0 means uncapped.

    Enforced here rather than left to the solver's discipline. A cap the solver
    is merely told about is not a cap, and the failure mode is a run that spends
    its entire budget executing a plan it should have abandoned.
    """

    # Silences the pace ratio and the raw/ceiling half of the score block while
    # the harness keeps using the underlying values.
    quiet_pace: bool = False
    """Stop reporting pace against the per-level reference count."""

    show_score: bool = False
    """Report the running RHAE score and its ceiling in :meth:`status`.

    The workspace template sets this. The ``False`` default keeps a bare
    ``ArcClient`` silent, which is what the tests and the local bench want.
    """

    card_id: str = ""
    """The scorecard this game is scored on.

    **Set it to share one card across many games; leave it empty for the usual
    one-card-per-game.** A leaderboard submission takes exactly one
    `scorecard_url`, so a sweep that mints one card per game has nothing to
    submit. When this is set at construction the client skips
    `/api/scorecard/open` and plays onto the card it was handed.

    Passing the id is necessary and **not sufficient**: the card is reachable
    only from a session carrying its `AWSALBAPP-*` stickiness cookies, and in a
    proxied run those live in the shim, not here. See
    :mod:`ccarc.arc_3.shared_card`.
    """

    require_card: bool = False
    """This run is part of a shared-card sweep, so minting is a bug, not a default.

    **`card_id` empty and "standalone run" must not be the same state.**
    `open()` mints whenever `card_id` is falsy, which is correct for a
    one-card-per-game run and wrong for a sweep: a launcher generated with
    `card_id=''` quietly opens its own card and keeps scoring onto it, so the
    game scores onto a card the driver does not track and cannot submit, and
    nothing fails.

    Set this and the client refuses instead of minting. A refused run is
    recoverable; a run scored onto a card nobody tracks is not.
    """

    guid: str = ""
    actions_used: int = 0
    level: int = 0
    win_levels: int = 0
    state: str = "NOT_PLAYED"
    available_actions: tuple[str, ...] = ()
    full_resets: int = 0
    wasted_actions: int = 0
    close_error: str = ""
    scorecard_error: str = ""
    """Why the scorecard could not be snapshotted before the card was closed.

    Recorded rather than raised: see :meth:`_snapshot_scorecard`.
    """
    level_actions: int = 0
    """Actions spent on the current level, reset when it advances.

    Maintained per action rather than derived, so :meth:`status` need not parse
    the trace to report it. It is the solver's own count for the level it is on.
    """
    level_costs: tuple[int, ...] = ()
    """Actions spent on each level **this play** has cleared, in order.

    Maintained incrementally for the same reason as :attr:`level_repeats`:
    deriving it means parsing the whole trace, which ``status()`` cannot afford.

    Cleared on a full reset, because each play is accounted separately -- the
    server records ``actions_by_level`` per play.
    """

    level_tried: int = 0
    level_dead: int = 0
    level_repeats: int = 0
    """Actions on this level that were tried, changed nothing, and were repeats.

    Maintained per action rather than derived from the trace.
    """

    level_revisits: int = 0
    """Actions on this level that landed on a board seen earlier on this level.

    Distinct from :attr:`level_repeats`, which counts repeating an action that
    changed *nothing*.
    """

    _dead_keys: list[str] = field(default_factory=list, repr=False)
    _seen_keys: set[str] = field(default_factory=set, repr=False)
    _last_frame_key: str = field(default="", repr=False)
    _writer: TraceWriter | None = field(default=None, repr=False)
    _reader: Any = field(default=None, repr=False)
    """Decodes the ledger once and then only where it grew. See `history`."""
    _last: Any = field(default=None, repr=False)
    """The newest transition, held so that asking for it costs nothing.

    **This is the single most expensive question the client answers.**
    `history()` decodes the whole ledger, and a solver that wants the frame
    its last action produced has no other way to ask -- so it writes
    `history()[-1]` after every action, and a linear read inside a
    per-action loop makes the run quadratic in its own length: on a long trace
    the re-read comes to dwarf the action itself by orders of magnitude.

    `_send` already holds the frame. Keeping the transition it builds turns the
    question into an attribute read. `_chain` is what makes the NEXT one cheap
    too.
    """
    _chain: Any = field(default=None, repr=False)
    """Chain state for the row after `_last`, or None when it is not yet known.

    Seeded from the end of the ledger on first use, never from all of it, and
    only when something actually asks -- a solver that never reads the newest
    transition should not pay to have one ready.
    """
    _opener: Any = field(default=None, repr=False)
    _key: str = field(default="", repr=False)
    _last_advanced: bool = field(default=False, repr=False)
    """The last action completed a level, so the next RESET is a *full* reset.

    Persisted, and that is the whole point: a CC solver takes every action in
    a new process, so a flag that defaults to ``False`` per process is lost
    exactly when it matters. A run that ends one action after clearing a level
    and resumes with the flag lost would open with RESET exactly as the
    workspace guide says to, and throw away a whole play of progress — with
    the refusal built to prevent that standing unarmed.
    """
    _last_was_reset: bool = field(default=False, repr=False)
    """The last action taken was a RESET, so the server's counter is zero NOW.

    **The flag above guards one door of a two-door room.** A RESET
    issued while the server's action counter is zero is a full game reset, and
    the counter is zero in TWO states: right after a level advance
    (``_last_advanced``) and right after any RESET. A scripted explore loop
    that issues RESET twice in a row -- the first an ordinary level retry --
    makes the second a silent full wipe. The hazard is "counter at zero", not
    "after an advance"; this flag covers the other door. Persisted for the
    same reason ``_last_advanced`` is: every action arrives in a new process.
    """

    _resumed: bool = field(default=False, repr=False)
    _owns_card: bool = field(default=False, repr=False)
    """True only when this client opened the card itself.

    A shared card outlives the game that happens to finish first, so closing one
    we were merely lent would end the sweep for every game still playing.
    """

    foreign_card: str = field(default="", repr=False)
    """Set when a resumed game is on a different card than the one injected.

    A trace is bound to the card it was played on; it cannot be moved to another
    one after the fact. Resuming wins -- the alternative is a resume that replays
    from level 0 -- and this records that the game is NOT on the shared card, so
    the driver can say so instead of counting it in a submission it is missing
    from.
    """

    @property
    def state_path(self) -> Path:
        return Path(self.trace_path).with_suffix(".state.json")

    @property
    def session_path(self) -> Path:
        """Where the cookie jar lives — deliberately NOT in the agent's workspace.

        Session cookies are live credentials, and the agent has no use for
        them. Anything in the directory the agent works in can be read and
        quoted verbatim into ordinary output, so credentials do not belong
        there: the jar lives one directory up, as a dotfile, where neither
        `ls` nor `cat *` nor a glob in the workspace finds it. Mode 600, like
        the API key.

        This is not confinement -- an agent with a shell can read anything the
        process can. It removes the accident, and it is worth doing on that
        basis alone.
        """
        state = self.state_path
        name = f".{self.game_id or 'run'}.session.json"
        parent = state.parent.parent
        # The parent only when there is one that is not the workspace itself; a
        # standalone run with a trace at the filesystem root must not silently
        # write its credentials somewhere unrelated.
        if parent != state.parent and parent.is_dir():
            return parent / name
        return state.parent / name

    def _save_session(self, cookies: list[dict]) -> None:
        """Persist the jar beside the workspace, never inside it."""
        path = self.session_path
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps({"game_id": self.game_id, "cookies": cookies}),
                           encoding="utf-8")
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)
        except OSError:
            # A jar that cannot be written costs a resume, which costs a replay.
            # Losing the run to an exception here would cost the whole game, so
            # this stays best-effort -- the caller's state file is what matters.
            tmp.unlink(missing_ok=True)

    def _load_session(self) -> list[dict]:
        try:
            doc = json.loads(self.session_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if doc.get("game_id") not in (None, self.game_id):
            return []
        jar = doc.get("cookies")
        return jar if isinstance(jar, list) else []

    def _write_state_atomically(self, payload: str, encoding: str = "utf-8") -> None:
        """Write the state file so a reader never sees a half-written one."""
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding=encoding)
        os.replace(tmp, self.state_path)      # atomic on POSIX

    def _save_state(self) -> None:
        # The jar goes to its own file, outside the workspace, before the state
        # file is written -- so a reader that sees the state file always has a
        # session to go with it, never the other way round.
        self._save_session([
            {"name": c.name, "value": c.value, "domain": c.domain, "path": c.path}
            for c in self._cookiejar()
        ])
        # Written via a temporary file and os.replace, which is atomic on POSIX.
        # `write_text` issues a single write() syscall at these sizes, so a torn
        # file is not something this code produces on its own -- but a reader
        # can still observe a partial file, and the cost of an unreadable one is
        # a refused start. Two lines to remove the possibility entirely.
        self._write_state_atomically(
            json.dumps(
                {
                    "game_id": self.game_id,
                    "card_id": self.card_id,
                    "owns_card": self._owns_card,
                    "foreign_card": self.foreign_card,
                    "guid": self.guid,
                    "actions_used": self.actions_used,
                    "level": self.level,
                    "win_levels": self.win_levels,
                    "state": self.state,
                    "available_actions": list(self.available_actions),
                    "full_resets": self.full_resets,
                    "wasted_actions": self.wasted_actions,
                    "level_actions": self.level_actions,
                    "last_touched": self.last_touched,
                    "card_plays_at_open": self.card_plays_at_open,
                    "level_costs": list(self.level_costs),
                    "last_advanced": self._last_advanced,
                    "last_was_reset": self._last_was_reset,
                    # **The four effectiveness counters are NOT persisted.**
                    # `trace.state.json` sits at the workspace root and the
                    # guide names it, so the solver will read it -- and
                    # counting which actions did anything IS the reasoning the
                    # solver is here to do. Nothing consumes them, and they are
                    # in `_NOT_OFFERED`.
                    "dead_keys": self._dead_keys,
                    "seen_keys": sorted(self._seen_keys),
                    "last_frame_key": self._last_frame_key,
                    # **The values live in `session_path`, not here.** This file
                    # is in the agent's working directory, and credentials never
                    # go there; see `session_path`. The names stay, because a
                    # reader has to be able to tell a run that had a session from
                    # one that never did.
                    "cookies": [
                        {"name": c.name, "value": "", "domain": c.domain,
                         "path": c.path, "stored": "session_path"}
                        for c in self._cookiejar()
                    ],
                    "gate_last_level": self.gate.last_level if self.gate else 0,
                    "gate_pending": self.gate.pending_level if self.gate else None,
                    "gate_acknowledged": (
                        {str(k): v for k, v in self.gate.acknowledged.items()}
                        if self.gate else {}
                    ),
                }
            ),
            encoding="utf-8",
        )

    def _cookiejar(self):
        for h in self._opener.handlers:
            if isinstance(h, urllib.request.HTTPCookieProcessor):
                return h.cookiejar
        return []

    def _restore_state(self) -> bool:
        """Resume a game left by an earlier process. Returns True if resumed.

        **This is what makes the client usable by a Claude Code solver at all.**
        An agent drives it with one-shot ``python -c`` commands, so a new process
        starts for every single action. Without resumption each command builds
        a fresh client, which deletes the trace and opens a brand new
        scorecard -- the game restarts every time and no run gets past action
        one.
        """
        if not self.state_path.exists():
            return False
        try:
            saved = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            # **Refuse, do not fall through.** Returning False here means "no
            # previous run", and the caller then deletes trace.jsonl and opens a
            # fresh scorecard -- discarding a game because a small sidecar file
            # became unreadable. The file being present says a run exists; only
            # its details are lost, and those are cheaper to lose than the game.
            raise RuntimeError(
                f"{self.state_path} exists but cannot be read ({exc}). A run is "
                f"in progress and its trace has NOT been touched. Move the state "
                f"file aside to start over, or repair it to resume."
            ) from exc
        if saved.get("game_id") != self.game_id:
            return False

        # A card injected at construction must not silently displace the one
        # this trace was actually played on -- and vice versa. Resume wins,
        # because the alternative is replaying from level 0, but the divergence
        # is recorded rather than swallowed.
        injected, saved_card = self.card_id, saved.get("card_id", "")
        if injected and saved_card and injected != saved_card:
            self.foreign_card = injected
        self.card_id = saved_card or injected
        # A state file with no `owns_card` key predates card injection, so the
        # client that wrote it opened its own card; default accordingly rather
        # than to False, which would stop such a run closing the card it owns.
        self._owns_card = bool(saved.get("owns_card", not injected))
        self.foreign_card = self.foreign_card or saved.get("foreign_card", "")
        self.guid = saved.get("guid", "")
        self.actions_used = int(saved.get("actions_used", 0))
        self.level = int(saved.get("level", 0))
        self.win_levels = int(saved.get("win_levels", 0))
        self.state = saved.get("state", "NOT_PLAYED")
        self.available_actions = tuple(saved.get("available_actions") or ())
        self.full_resets = int(saved.get("full_resets", 0))
        self.wasted_actions = int(saved.get("wasted_actions", 0))
        # A state file carrying none of these counter keys is still resumable.
        # Defaulting `level_actions` to 0 makes `status()` report 0.0x pace and
        # suppress OVER BASELINE on precisely the resumed runs the warning is
        # for -- a silently wrong number is worse than an absent one, so derive
        # it from the trace instead.
        self.last_touched = float(saved.get("last_touched") or 0.0)
        self.card_plays_at_open = int(saved.get("card_plays_at_open", -1))
        if "level_actions" in saved:
            self.level_actions = int(saved["level_actions"])
        else:
            self.level_actions = self._level_actions_from_trace(int(saved.get("level", 0)))
        # A state file with no `level_costs` key is rebuilt from the trace, the
        # same choice made for `level_actions` just above and for the same
        # reason: a resumed run would otherwise report a score computed from
        # an empty history, which reads as "you have cleared nothing" on a run
        # that has cleared several levels.
        if "level_costs" in saved:
            self.level_costs = tuple(int(c) for c in saved["level_costs"])
        else:
            self.level_costs = self._level_costs_from_trace()
        self._last_advanced = bool(saved.get("last_advanced", False))
        # Absent on a state file that did not record it; False is the safe
        # default -- it merely means one unguarded bare RESET.
        self._last_was_reset = bool(saved.get("last_was_reset", False))
        self.level_tried = int(saved.get("level_tried", 0))
        self.level_dead = int(saved.get("level_dead", 0))
        self.level_repeats = int(saved.get("level_repeats", 0))
        self.level_revisits = int(saved.get("level_revisits", 0))
        self._dead_keys = list(saved.get("dead_keys") or [])
        self._seen_keys = set(saved.get("seen_keys") or ())
        self._last_frame_key = saved.get("last_frame_key", "")

        jar = self._cookiejar()
        # **Sidecar first, state file second.** A state file may carry cookie
        # NAMES with empty values, and those must not overwrite a good sidecar
        # with blanks. Filtering on a non-empty value covers every case without
        # a version field: an entry with no value restores nothing.
        restored = self._load_session() or saved.get("cookies", [])
        for c in restored:
            if not c.get("value"):
                continue
            jar.set_cookie(
                http.cookiejar.Cookie(
                    0, c["name"], c["value"], None, False,
                    c["domain"], True, c["domain"].startswith("."),
                    c["path"], True, False, None, True, None, None, {},
                )
            )
        if self.gate is not None:
            self.gate.last_level = int(saved.get("gate_last_level", 0))
            self.gate.pending_level = saved.get("gate_pending")
            self.gate.acknowledged = {
                int(k): v for k, v in (saved.get("gate_acknowledged") or {}).items()
            }
        return True

    def __post_init__(self) -> None:
        self._key = _api_key(self.api_key)
        self._opener = new_session()
        if self.gate is not None:
            # An acknowledgement must survive the process that made it.
            self.gate.on_change = self._save_state
        self._refuse_side_ledger()
        self._writer = TraceWriter(self.trace_path)
        if self._restore_state():
            self._resumed = True
            self._writer._index = self.actions_used
            return

        path = Path(self.trace_path)
        if path.exists():
            # An append-only ledger silently welds runs together and makes
            # load() chain one run's `before` from another's `after`. Never
            # inherit a previous run's trace by accident.
            path.unlink()
        self._writer = TraceWriter(path)

    def _refuse_side_ledger(self) -> None:
        """On a scored run, only the workspace-root ledger may be opened.

        A missing root ledger would look like a first open, and the next
        process would mint a new scorecard. Restore it or stop; a client
        pointed at ``notes/run*`` is a side card and is refused.
        """
        scored = os.environ.get("CCARC3_SCORED_TRACE", "")
        if not scored:
            return
        wanted = Path(scored).resolve()
        got = Path(self.trace_path).resolve()
        if got != wanted:
            raise RuntimeError(
                f"{self.game_id}: scored play uses only the workspace-root "
                f"ledger ({wanted}); a client at {got} would open a side "
                f"card that is not scored. Refusing."
            )
        # **A pinned workspace holds every construction to its card.** The path
        # check above stops a client writing elsewhere; it says nothing about
        # the CARD, so a bare `ArcClient(game_id, trace_path=<root trace>)` --
        # no `card_id`, `require_card` defaulting False -- would pass it and
        # mint a fresh card into the scored slot: the sweep card becomes
        # `foreign_card` in its own workspace and every action scores onto a
        # card no submission can use. The pin is written at
        # workspace build (see `build_workspace`), so it exists before any
        # client does; a construction naming no card, or a different card, is
        # the same wrong-by-construction state `require_card` refuses, arriving
        # through direct construction instead of the template.
        from .session import _pin_card_id, _read_card_id, ensure_scored_ledger

        # Restore first: a displaced ledger's state file is evidence the check
        # below needs, and `ensure_scored_ledger` is what puts it back.
        ensure_scored_ledger(wanted.parent)
        pinned = _pin_card_id(wanted.parent)
        if pinned and self.card_id != pinned:
            saved = _read_card_id(wanted.parent / "trace.state.json")
            # A bare construction is still legitimate when the state file
            # carries the pinned card -- that is every standalone resume, where
            # the template names no card and `_restore_state` adopts it. What
            # is refused is a client that would play elsewhere: one naming a
            # different card outright, or one naming none in a workspace with
            # no resumable state on the pinned card -- the mint described
            # above.
            if self.card_id or saved != pinned:
                raise RuntimeError(
                    f"{self.game_id}: this workspace is pinned to scorecard "
                    f"{pinned}, and this client "
                    + (f"names a different card ({self.card_id})."
                       if self.card_id else "names no card, so it would mint "
                       "or adopt one.")
                    + " Plays on any other card are not scored and cannot be "
                      "submitted; opening a new scorecard would orphan this "
                      "game from the sweep. Use `from session import client` "
                      "instead of constructing an ArcClient. Refusing."
                )

    def _pin_scored_card(self) -> None:
        scored = os.environ.get("CCARC3_SCORED_TRACE", "")
        if not scored or not self.card_id:
            return
        from .session import pin_scored_card

        pin_scored_card(Path(scored).parent, self.card_id)

    # -- lifecycle ------------------------------------------------------- #

    def open(self) -> "ArcClient":
        """Open a scorecard, or keep the one a previous process opened.

        Re-opening on resume would abandon the in-flight game and start scoring
        from zero, which is precisely the bug this class exists to avoid.
        """
        if self._resumed and self.card_id:
            self._assert_server_agrees()
            self._pin_scored_card()
            return self
        if self.card_id:
            # A card the driver opened and lent us. Opening another one here is
            # exactly the bug that leaves a sweep with one card per game and
            # nothing to submit.
            #
            # **Record where this attempt's rows start.** The server appends to
            # one `cards[game_id]` entry for every attempt of this game, so a
            # later corroboration check reading the last N rows can be handed a
            # DISCARDED attempt's rows instead -- and whether that passes depends
            # only on how far the thrown-away run got. The boundary is knowable
            # exactly once, here, and the read is a GET the proxy allows and the
            # server does not bill. A failure leaves it unknown rather than
            # wrong: the driver probes the card's liveness at start and
            # `_assert_server_agrees` re-checks it on resume, so an unreadable
            # card is caught elsewhere and need not break the attempt here.
            #
            # **Taken once per attempt, not once per open.** See
            # `boundary_path`: a mid-attempt re-open must inherit the boundary,
            # never re-take it, or the rows this attempt already produced are
            # attributed to whatever ran before it.
            recorded = self._recorded_boundary()
            if recorded is not None:
                self.card_plays_at_open = recorded
            else:
                try:
                    entry = (self.scorecard().get("cards") or {}).get(self.game_id) or {}
                    self.card_plays_at_open = len(entry.get("levels_completed") or [])
                except Exception:                  # noqa: BLE001 -- see above
                    self.card_plays_at_open = -1
                if self.card_plays_at_open >= 0:
                    self._record_boundary(self.card_plays_at_open)
            self._save_state()
            self._pin_scored_card()
            return self
        if self.require_card:
            raise RuntimeError(
                f"{self.game_id}: this run is part of a shared-card sweep but no "
                f"card_id reached it, so opening one here would score this game "
                f"onto a card the driver does not know about and cannot submit. "
                f"Refusing. Recover the sweep's card id from the driver's card "
                f"history and pass it in, or start a deliberately fresh sweep."
            )
        payload: dict[str, Any] = {"tags": list(self.tags)}
        if competition_mode():
            # The whole of compliance is this field. Verified against the
            # official toolkit's source: every `OperationMode.COMPETITION`
            # branch in it either means "behave like ONLINE" or sets exactly
            # this. The rules are enforced by ARC's server on the card.
            payload["competition_mode"] = True
        card = _post(f"{self.root}/api/scorecard/open", payload,
                     self._key, opener=self._opener)
        self.card_id = card["card_id"]
        self._owns_card = True
        self._save_state()
        self._pin_scored_card()
        return self

    def _assert_server_agrees(self) -> None:
        """Refuse a resume where the server is not where the ledger thinks it is.

        A resume can preserve the ledger but not the game: trace indices
        continue from where the ledger stopped while the server replays from
        level 0 on a fresh scorecard, and the run re-spends every one of those
        actions before anything notices. `_record_resume_state` only snapshots
        what was inherited -- it makes such a failure reconstructable, not
        preventable, which is the detect-instead-of-enforce gap this check
        closes.

        The mismatch is observable before a single action is spent, and for
        free: reading a scorecard is a `GET` the proxy allows and the server does
        not bill. If the card says this play has cleared fewer levels than the
        ledger claims, the two are describing different games and continuing
        would rewrite one with the other.

        Raising rather than warning is deliberate. The cost of stopping is one
        relaunch; the cost of continuing is the whole attempt, spent invisibly.
        """
        # Nothing spent, nothing to protect. A resume at the very start of a game
        # has no ledger for the server to disagree with, and demanding a card
        # read there would make an offline or not-yet-opened game unresumable
        # for no gain. Losing banked progress requires banked progress to
        # already exist.
        if self.level <= 0 and self.actions_used <= 0:
            return

        # **Ask more than once before declaring the card unreadable.**
        #
        # A scorecard is bound to one backend by a sticky session, and a request
        # that lands elsewhere gets 404 for a card that is perfectly alive --
        # a live card can take many attempts to answer while the misses 404.
        # A single `_get` therefore refuses healthy runs, and a solver that
        # reads the failure as "the environment is over" stops playing with
        # most of its budget left.
        #
        # This does not soften the guard. It still refuses if every attempt
        # fails; it just stops mistaking one unlucky route for a dead card.
        # **The idle check runs FIRST, because it is the one check competition
        # mode cannot take away.** It needs only `last_touched` and a clock; the
        # scorecard read below 403s on every competition card, so a check
        # placed after it would never run on the runs that need it most.
        if self.last_touched:
            idle = time.time() - self.last_touched
            if idle > self.REAP_DEADLINE_S:
                raise RuntimeError(
                    f"resume: {idle / 60:.1f} minutes since the server last "
                    f"answered this game, past the {self.REAP_DEADLINE_S / 60:.1f}"
                    f"-minute reap deadline. The game is gone even if the card "
                    f"still reads; continuing would replay from level 0 with "
                    f"{self.actions_used} actions already banked. Start fresh."
                )

        card = None
        last: Exception | None = None
        for pause in (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0, 8.0, 8.0, 10.0, 10.0):
            if pause:
                time.sleep(pause)
            try:
                card = self.scorecard()
                last = None
                break
            except Exception as exc:              # noqa: BLE001
                last = exc
                # **Competition mode forbids reading a scorecard, so this check
                # cannot be performed at all -- and retrying costs a minute of
                # sleeps to reach the same 403.** ARC answers
                # `403 "cannot get scorecard that is in competition mode"`.
                #
                # Stopping here is a REAL LOSS of protection and is stated
                # rather than hidden: a resume whose server has silently gone
                # back to level 0 will no longer be caught before it re-spends
                # the run. Nothing can catch it, because the only free
                # observation of the server's position is the card read the
                # mode denies.
                if "competition mode" in str(exc):
                    warnings.warn(
                        "resume: competition mode forbids reading the scorecard, "
                        "so the server's position cannot be confirmed; continuing "
                        "unverified", RuntimeWarning, stacklevel=2)
                    return
        if last is not None:
            exc = last
            # A resume that cannot be verified is one that should not proceed:
            # this call is the only thing standing between a mismatched card and
            # a silently re-spent run.
            raise RuntimeError(
                f"resume: could not read the scorecard to confirm the server is "
                f"at level {self.level} ({exc.__class__.__name__}: {exc}). "
                f"Refusing to continue on an unverified card."
            ) from exc

        # **The game's own entry, with no fallback to the card.** A fallback
        # of `... or card` would mean that when the card carries no entry for
        # this game -- the exact "this trace was never played on this card"
        # state the guard exists to catch -- the read falls through to the
        # CARD-LEVEL aggregate. On a card shared by many games that aggregate
        # is produced by the others, and being an int rather than the
        # per-play list it skips the `isinstance` branch and is compared
        # directly, so a large number from other games silently satisfies the
        # check.
        entry = (card.get("cards") or {}).get(self.game_id)
        if entry is None:
            raise RuntimeError(
                f"resume: card {self.card_id} carries no entry for "
                f"{self.game_id}, so this trace was not written against it; "
                f"continuing would re-spend {self.actions_used} actions. "
                f"Start fresh instead of resuming."
            )
        done = entry.get("levels_completed")
        if isinstance(done, list):
            # **Find OUR row by guid, not by position.** `done[-1]` assumes
            # the last row is the play in flight, and that is false: a card
            # can gain a trailing zero-action row (a play opened by something
            # else against the same card) after ours, and then the guard would
            # compare our ledger level against a stranger's empty play and
            # refuse a resume that is perfectly sound. The guid is the
            # identity the server itself uses,
            # so match on it and only fall back to the old positional guess.
            guids = entry.get("guids") or []
            if self.guid and self.guid in guids:
                # Match this *lineage*, not ``done[-1]`` (a stranger's trailing
                # empty play carries a different guid and is already filtered)
                # and not ``list.index`` (ARC repeats the guid on every
                # playthrough, so the first hit is an old row). A full reset
                # appends a new row under the same guid; ``self.level`` tracks
                # the current frame, so the server's position is the *last*
                # matching row. ``max`` would freeze the guard at the best
                # playthrough forever -- ledger stale at 9, live play at 4,
                # resume allowed. Rightmost match refuses.
                matched = [
                    i for i, g in enumerate(guids)
                    if g == self.guid and i < len(done)
                ]
                done = done[matched[-1]] if matched else None
            else:
                done = done[-1] if done else None  # the play now in flight
        # **A readable card is not evidence the GAME is alive.** The two are
        # on different clocks (see `shared_card`): an idle game is reaped, a
        # card is not. Under a card shared by many games the others keep it
        # warm, so the read succeeds -- and the reaped play's row
        # still records the level the ledger claims, so the comparison below is
        # `5 < 5` and cannot fire. On a per-game card a bad resume is caught by
        # a 404; the shared card removes that detector by
        # construction, which is why the gap is checked directly.
        #
        # (The idle check above compares with `>` and not `>=`: at exactly the
        # edge the game is not yet known to be gone.)
        if done is None:
            return                                 # nothing to compare against
        if int(done) < self.level:
            raise RuntimeError(
                f"resume: the ledger says level {self.level} but the server's "
                f"card says {done} for this play. The game was replayed or the "
                f"card is not the one this trace was written against; "
                f"continuing would re-spend {self.actions_used} actions. "
                f"Start fresh instead of resuming."
            )

    def close(self) -> dict[str, Any]:
        """Close the scorecard. Never raises.

        A failed close is bookkeeping, not a result: the actions are already
        recorded server-side and the trace is already on disk. Raising here
        would replace whatever the solver was actually doing with a cleanup
        error. The stock SDK takes the same posture and logs a warning.

        The reason, if any, is left on ``close_error`` rather than discarded.
        """
        if not self.card_id:
            return {}
        self._snapshot_scorecard()
        card_id, self.card_id = self.card_id, ""
        self.state_path.unlink(missing_ok=True)
        # The jar goes with it. `_load_session` already refuses a sidecar from a
        # different game, so this is not needed for correctness -- it is here so
        # a finished run leaves no live session credentials on disk.
        self.session_path.unlink(missing_ok=True)
        if not self._owns_card:
            # Someone else's card, still carrying games that have not finished.
            # Snapshotting it is the useful half; closing it would end the
            # sweep on whichever game happened to return first.
            return {}
        try:
            return _post(f"{self.root}/api/scorecard/close", {"card_id": card_id},
                         self._key, opener=self._opener)
        except Exception as exc:  # noqa: BLE001 -- deliberate: see docstring
            self.close_error = f"{type(exc).__name__}: {exc}"
            return {}

    REAP_DEADLINE_S = 18.2 * 60
    """Seconds of game idleness after which ARC is taken to have reaped the game.

    The UPPER edge of the working bracket for the server's idle-reap window
    (see :mod:`ccarc.arc_3.shared_card`), not the lower: refusing at the lower
    edge would reject gaps a resume can survive, while past the upper edge a
    resume finds a replayed game.
    """

    SNAPSHOT_EVERY = 50
    """Actions between scorecard snapshots, on top of the event-driven ones."""

    def _snapshot_scorecard_if_due(self, *, advanced: bool) -> None:
        """Keep a live copy of the scorecard while the card still exists.

        **Snapshotting after the run is too late.** A card read after the run,
        from a fresh session, cannot be relied on to answer (see
        :meth:`_snapshot_scorecard`). The only reliable moment is *during*
        the game, which is here.

        Fires on a level advance, on reaching a terminal state, and every
        :attr:`SNAPSHOT_EVERY` actions as a backstop for a run that dies
        mid-level. A game of nine levels costs about a dozen GETs, against
        hundreds of action POSTs, so the overhead is noise.

        The file is overwritten each time: the newest snapshot strictly
        dominates, since the scorecard only accumulates.
        """
        due = (
            advanced
            or self.state in ("WIN", "GAME_OVER")
            or (self.SNAPSHOT_EVERY and self.actions_used % self.SNAPSHOT_EVERY == 0)
        )
        if due:
            self._snapshot_scorecard()

    def _snapshot_scorecard(self) -> None:
        """Persist the server's own scorecard beside the trace, before closing.

        A card is state on **one backend instance**, reachable only from a
        session carrying its stickiness cookies. A read from a fresh process
        with a fresh jar lands on a backend at random, so it can answer 404
        even though the card still exists: "you asked the wrong instance", not
        "it is gone", and one try cannot tell those apart. Same mechanism the
        shim pins sessions for. An unreliable read is not a way to score a
        run, and the local copy costs nothing -- so the scorecard is captured
        here, from the session that played it, while the game is live. The
        reason is reliability, not permanence.

        What would otherwise be lost is not incidental. ``actions_by_level``
        is the *server's* per-level action count, which this harness otherwise
        re-derives from the trace. And the scorecard carries one row per play,
        the only
        direct record of how the server accounts multiple plays.

        Never raises, for the same reason :meth:`close` does not: a failed
        snapshot is bookkeeping, and burying a real solver exception under a
        bookkeeping error would be worse than a missing file.
        """
        # A `card-` id is a placeholder, not a scorecard the server knows.
        # Without this the action path fires a real GET on every level advance
        # for one -- swallowed by the handler below, so the only symptom is a
        # process that quietly needs the network.
        if not self.card_id or self.card_id.startswith("card-"):
            return
        try:
            card = self.scorecard()
        except Exception as exc:  # noqa: BLE001 -- deliberate: see docstring
            self.scorecard_error = f"{type(exc).__name__}: {exc}"
            return
        try:
            # `trace_path` is declared `str | Path` and the class's own
            # solver-facing example passes a str, so `.parent` on it is an
            # AttributeError that `except OSError` would not catch. The
            # docstring promises this never raises, so the handler is widened
            # to match: a snapshot is bookkeeping, and burying a solver's real
            # exception under a cleanup failure is the mistake this function
            # exists to avoid.
            path = Path(self.trace_path).parent / "scorecard.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(card, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 -- deliberate: see docstring
            self.scorecard_error = f"{type(exc).__name__}: {exc}"

    def __enter__(self) -> "ArcClient":
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- state ------------------------------------------------------------ #

    @property
    def done(self) -> bool:
        return self.state == "WIN"

    @property
    def dead(self) -> bool:
        return self.state == "GAME_OVER"

    # Every solver-facing surface reads this one property, so returning None
    # here silences all of them at once.
    @property
    def _baseline_here(self) -> int | None:
        """Reference action count for this level, or ``None`` when unavailable."""
        return None if self.quiet_pace else self._baseline_here_enforced

    @property
    def _baseline_here_enforced(self) -> int | None:
        """The enforced value for this level, whatever the solver may see."""
        return self.info.baseline_for(self.level) if self.info else None

    # -- score ------------------------------------------------------------ #
    #
    # The solver is shown its running score and its ceiling: the question that
    # decides what to do next -- *is what I am doing still worth anything?* --
    # cannot be answered from a pace ratio alone, and a solver that has
    # overspent early levels has already fixed its ceiling.
    #
    # (No action counts here: a count printed beside a ratio would give up the
    # figure the ratio is computed against.)

    @property
    def completion_cap(self) -> float:
        """`C` if this play stopped here: the weighted fraction of levels cleared."""
        # Pure structure -- which levels fell, not how fast -- so unlike
        # everything else in this block it needs no per-level figures. Kept in a
        # comment: this module is staged onto the solver's own PYTHONPATH, and a
        # docstring saying what the rest of the block cannot compute, and why,
        # describes the withholding to the thing it is withheld from.
        n = self.win_levels
        if not n:
            return 0.0
        k = min(len(self.level_costs), n)
        return sum(range(1, k + 1)) / sum(range(1, n + 1))

    def scorecard(self) -> dict[str, Any]:
        return _get(f"{self.root}/api/scorecard/{self.card_id}/{self.game_id}",
                    self._key, opener=self._opener)

    # -- actions ---------------------------------------------------------- #

    @property
    def decision_path(self) -> Path:
        """Where :meth:`finish` records the solver's declaration.

        Its own sidecar rather than a key in ``trace.state.json``, because
        :meth:`close` deletes the state file and the parent reads the
        declaration *after* the run -- a decision that vanishes with the state
        file is a decision the parent can never honor.
        """
        return Path(self.trace_path).with_suffix(".decision.json")

    @property
    def boundary_path(self) -> Path:
        """Where this attempt's card boundary is recorded, once.

        Its own sidecar for the same reason as :attr:`decision_path`: it must
        outlive :meth:`close`, which deletes the state file. The boundary says
        which of a shared card's rows belong to *this attempt*, and it is only
        true when taken before the attempt's first play. A client that opens a
        second time mid-attempt -- after a server reap, which is the sanctioned
        recovery -- would otherwise re-take it against a card that has since
        grown, moving the boundary past rows this attempt really did produce --
        and the corroboration that reads it would then discard a winning run.
        """
        return Path(self.trace_path).with_suffix(".open.json")

    def _recorded_boundary(self) -> int | None:
        """This attempt's boundary if one was already taken, else ``None``."""
        try:
            doc = json.loads(self.boundary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if doc.get("game_id") not in (None, self.game_id):
            return None
        try:
            return int(doc["card_plays_at_open"])
        except (KeyError, TypeError, ValueError):
            return None

    def _record_boundary(self, plays: int) -> None:
        """Write the boundary sidecar atomically, once per attempt."""
        tmp = self.boundary_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"game_id": self.game_id, "card_plays_at_open": plays}) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, self.boundary_path)      # atomic on POSIX

    def finish_declarations(self) -> int:
        """How many times :meth:`finish` has been called for this game."""
        try:
            doc = json.loads(self.decision_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        if doc.get("game_id") not in (None, self.game_id):
            return 0
        try:
            return max(0, int(doc.get("declarations", 0)))
        except (TypeError, ValueError):
            return 0

    def finish(self) -> None:
        """Declare this run complete. The one explicit act of submission.

        **A declaration to the harness, not an action on ARC.** There is no
        submit endpoint: every play is already on the scorecard as it was made,
        and the card is scored as it stands. What this records is intent -- so
        that the parent can tell a decision from an accident. An exit *without*
        this call is ambiguous (satisfied? crashed? out of context?), and the
        parent answers ambiguity with one neutral question. An exit *with* it is
        final: a declared decision is never overridden.

        Repeatable on purpose. If the harness states a fact after your first
        declaration (for instance that you quit with actions still in hand),
        calling this again reaffirms it, and a reaffirmed decision stands.
        """
        count = self.finish_declarations() + 1
        tmp = self.decision_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({
                "game_id": self.game_id,
                "decision": "submit",
                "declarations": count,
                "actions_used": self.actions_used,
                "level": self.level,
                "state": self.state,
            }),
            encoding="utf-8",
        )
        os.replace(tmp, self.decision_path)      # atomic on POSIX
        print(FINISH_REAFFIRM_ACK if count > 1 else FINISH_ACK, flush=True)

    def _announce_level(self, previous_level: int, spent: int) -> None:
        """Tell the solver, as a user message, that a level just closed."""
        # Sent through the outbox so it arrives as something the harness said
        # rather than an exception the solver tripped over, and it costs no
        # extra turn: a level closes inside a tool call and the CLI holds a
        # message written then until the call finishes. Internal plumbing of our
        # delivery channel -- the solver never calls this and can act on none
        # of it -- so it is a comment.
        # **No fallback to printing, and that is the difference from the win
        # reminder.** The announcement reaches the solver only through the
        # outbox, so a run with no live driver simply does not get one. The win
        # reminder does fall back, because the run depends on it arriving.
        #
        # **Every advance, not only the first.**
        # The gate deliberately does not re-arm on a boundary already
        # documented -- that is what makes a replay affordable -- but the
        # announcement is not the gate, and a solver replaying a level still
        # wants to know it cleared it and what it cost this time.
        #
        # **Carries the solver's own numbers and nothing else.** `status()`
        # already prints `actions=` unconditionally, `level_actions` is a public
        # field, and the solver can count its own ledger, so none of this is new
        # to it. The withheld figure is the BASELINE, and the ratio against it;
        # neither appears here. See `_baseline_here_enforced` for what is fenced
        # and why.
        if not _level_announce_enabled():
            return
        from . import outbox

        held = self.gate is not None and self.gate.pending_level is not None
        # What is announced is the event and, only when the gate is held, the
        # call that clears it. Action counts are not restated: `status()`
        # prints `actions=` on every call.
        lines = [
            f"Level {previous_level} cleared — you are on level {self.level}"
            f"{'/' + str(self.win_levels) if self.win_levels else ''}."
        ]
        if held:
            lines.append(f"Record what level {previous_level} established with "
                         f"gate.acknowledge(...).")
        outbox.say("\n".join(lines), kind="level_cleared")

    def _offer_win_replay_reminder(self) -> None:
        """State the facts after a full clear. Every clear; never a verdict."""
        # **Why it exists.** The environment score is the maximum across plays,
        # so a worse replay cannot lower what is banked -- and the solver cannot
        # verify that alone, because checking needs the score and the score is
        # built from the withheld figure. So the harness states the rule and the
        # cost, and the decision is the solver's.
        #
        # **Uncapped, deliberately.** Firing is a constant function of an event
        # the solver caused, so it carries no bit of the score. A cap on asking
        # would make the harness's silence informative, and nothing
        # solver-observable may be a function of the score.
        #
        # Carries no figures: this module is on the solver's PYTHONPATH and
        # every line is reachable by `inspect.getsource`; a count of plays
        # beside a budget is a baseline in two steps.
        if not _win_reminder_enabled():
            return
        # **A user message when there is a driver to send one, a tool result
        # otherwise.** Printed from the winning action, the words arrive as a
        # *tool result* -- the harness talking in the voice of the game. The
        # outbox hands it to the driver instead, which owns the live session's
        # stdin and sends it as a real user message.
        #
        # The fallback is not decoration. Without a live session there is
        # nothing to deliver the file, and a reminder that reaches nobody is
        # worse than one in the wrong channel: it is what makes an exit after a
        # clear a decision rather than an oversight.
        from . import outbox

        if not outbox.say(WIN_REPLAY_REMINDER, kind="win_replay_reminder"):
            print(WIN_REPLAY_REMINDER, flush=True)

    def reset(self, *, force_full: bool = False) -> dict[str, Any]:
        """RESET: restart the current level."""
        # **The quirk is not exposed, rather than explained or discovered.**
        # The server's RESET is overloaded: with its action counter at zero --
        # the state immediately after a level advance -- it discards the whole
        # game instead of restarting the level, irreversibly.
        #
        # That is a rule of the environment, so it does not belong in the
        # prompt. But it is not a rule an agent can afford to
        # learn by meeting it: discovery costs the entire run, which makes it a
        # footgun rather than a lesson.
        #
        # The resolution is to make our own API's semantics clean, so there is
        # no quirk in the agent's world at all: `reset()` restarts the level,
        # `restart_for_replay()` starts a new play, and the overloaded state is
        # simply not reachable through either. Nothing to warn about, nothing
        # to discover, nothing to lose a run to.
        #
        # `force_full` stays for the two callers that genuinely need it --
        # `restart_for_replay` and the harness replayer -- and is not
        # advertised.
        # **The competition guard sits on the primitive, not only on
        # `restart_for_replay`.** ARC does not reject a game reset on a
        # competition card, it silently DEMOTES it to a level reset, so the call
        # returns success while the agent believes it opened a fresh play and
        # has actually discarded its position on the current level. A guard on
        # the convenience method and not on the thing that acts is not a guard.
        #
        # The replayer needs the full reset and runs harness-side, where
        # `CCARC3_COMPETITION=0` applies; a solver has no way to set it.
        if force_full and competition_mode():
            raise ActionRefused(REFUSALS["no_second_play"])
        if (self._last_advanced or self._last_was_reset) and not force_full:
            raise ActionRefused(REFUSALS["reset_would_start_a_play"])
        return self._send(0)

    def restart_for_replay(self) -> dict[str, Any]:
        """Start a **new play** of the same game, keeping everything you learned.

        This is the one legitimate use of the RESET that :meth:`reset` refuses.
        Behaviour of the API: a RESET issued while the server's action
        counter is zero — the state immediately after a level advance — begins a
        new play with its own ``guid``, its own ``actions`` row and its own
        ``actions_by_level`` row. Per-level action counts are recorded per play
        and never summed, and the benchmark scores the *best* play. The actions
        themselves are cumulative: those spent on an abandoned play are still
        spent.

        This starts a new play on the server. It does not re-send anything
        from your trace file.

        Refuses unless the counter is at zero, because a RESET anywhere else is
        a level reset and would silently leave you in the same play.
        """
        # No play-count ceiling here, or anywhere, on purpose. A play count
        # is a proxy for cost denominated in the wrong
        # unit; the real bounds -- the action allowance `_send` refuses past and
        # the caller's wall clock -- already wall every action a replay spends,
        # and best-of scoring means a losing replay loop cannot touch what is
        # banked. A play-count ceiling would also refuse at exactly the shape
        # of the worst-case game, with zero margin.
        # **Refused outright under competition rules, and this one is dangerous
        # to leave working.** ARC demotes a game reset to a LEVEL reset on a
        # competition card rather than rejecting it, so the call would appear to
        # succeed while doing something else entirely: the agent would believe it
        # had started a fresh play and would actually have thrown away its
        # position on the current level. A silent wrong action is worse than a
        # refusal.
        if competition_mode():
            raise ActionRefused(REFUSALS["no_second_play"])
        if not self._last_advanced:
            raise ActionRefused(REFUSALS["replay_counter_not_zero"])
        return self.reset(force_full=True)

    #: Not persisted, and a CLASS default: `__dir__` reads it, so every
    #: instance must have it however it was constructed.
    #: Whether the shim discloses a baseline is a property of the run's config,
    #: not of the position, so it must not be cached into `trace.state.json`,
    #: which sits in the agent's own working directory.
    _baselines: Any = _UNFETCHED

    def __dir__(self):
        """The names this object offers."""
        # `baseline_actions` is in `_NOT_OFFERED`, so it is hidden by default
        # and REVEALED here only when the shim actually disclosed one. Hidden by
        # default is the safe direction: the metaclass `__dir__` that filters
        # `dir(ArcClient)` has no instance to ask, so a name offered by default
        # would leak from the class even in strict mode -- which is the exact
        # hole the metaclass closes.
        hidden = _NOT_OFFERED
        if self.baseline_actions is not None:
            hidden = hidden - {"baseline_actions"}
        return sorted(n for n in super().__dir__() if n not in hidden)

    @property
    def baseline_actions(self) -> tuple[int, ...] | None:
        """Reference actions per level, or ``None`` if this run does not have them."""
        # **The mode is not read here, and that is the design.** This asks the
        # one-game endpoint and reports what came back; `arc_proxy` decides
        # whether `baseline_actions` survives the response. One point of truth,
        # in the process the solver does not run, rather than a flag in this one
        # that a solver could read or a caller could pass.
        #
        # `HIDE_BASELINES_ENV` is deliberately NOT consulted: it guards
        # `list_games`/`baselines_for`, which answer for all 25 environments and
        # stay refused in both modes.
        if self._baselines is _UNFETCHED:
            self._baselines = self._fetch_baselines()
        return self._baselines

    def _fetch_baselines(self) -> tuple[int, ...] | None:
        # Never fatal: a shim that strips the field, refuses the path, or is not
        # there at all must leave the game playable. `__dir__` calls this too.
        try:
            raw = _get(f"{self.root}/api/games/{self.game_id}", self._key, retries=1)
            # ^ self._key, NOT self.api_key: the workspace client is constructed
            # with no api_key, and a None header value fails client-side before
            # any HTTP -- which the handler below swallows.
        except Exception:                       # noqa: BLE001 -- see above
            return None
        values = (raw or {}).get("baseline_actions") if isinstance(raw, dict) else None
        return tuple(values) if values else None

    def act(self, action: int, x: int | None = None, y: int | None = None) -> dict[str, Any]:
        """Take a game action. ``ACTION6`` needs ``x``/``y`` in ``[0, 63]``."""
        if action == 0:
            return self.reset()
        if self.state in _TERMINAL:
            raise ActionRefused(REFUSALS["terminal_state"].format(state=self.state))
        if action == 6 and (x is None or y is None):
            raise ActionRefused(REFUSALS["action6_needs_xy"])
        if action == 6 and not (0 <= x <= 63 and 0 <= y <= 63):
            raise ActionRefused(REFUSALS["action6_out_of_range"].format(x=x, y=y))
        if self.available_actions and action_name(action) not in self.available_actions:
            raise ActionRefused(REFUSALS["not_available"].format(
                name=action_name(action), available=list(self.available_actions)))
        return self._send(action, x=x, y=y)

    def _send(self, action: int, x: int | None = None, y: int | None = None) -> dict[str, Any]:
        if self.max_actions and self.actions_used >= self.max_actions:
            # **No figure, deliberately.** At exhaustion the count equals the
            # cap, and the cap is the baseline total times the budget multiple
            # -- so a refusal that printed it would state the withheld number's
            # product out loud, right beside the word "ceiling". The running
            # count is already visible in status(); this message adds only the
            # fact that it has run out.
            raise ActionRefused(REFUSALS["allowance_spent"])
        # **The play-restarting RESET is exempt, like the play-opening one.**
        #
        # `gate.check()` refuses every action after a level advance, and
        # `restart_for_replay()` is an action -- so clearing the final level
        # arms the gate and, without this exemption, the replay call comes
        # back as a `GateRefusal` traceback. Every winning run reaches that
        # frame, the one the
        # whole replay design turns on. `WIN_REPLAY_REMINDER` says "call
        # restart_for_replay() to play again" and the next thing the solver
        # would see is a stack trace.
        #
        # The gate exists to catch knowledge about to be lost *at a level
        # transition* -- that is why a refusal beats an instruction there, and
        # why it never fires on the opening RESET either. A full reset is not a
        # transition into a new level; it ends the play. Nothing is about to be
        # lost, the rule book already holds every earlier boundary, and
        # `observe` will decline to re-arm for those levels on the way back
        # through. So the refusal has no work left to do, and its only
        # remaining effect would be to throw at the decision point.
        #
        # **The final level's entry is voluntary.** No level follows it in this
        # game, so its knowledge is the least transferable of any, and the
        # reminder asks for it in words rather than compelling it with a
        # refusal.
        #
        # Scoped to a RESET that will actually start a new play: `_last_advanced`
        # is exactly the state `restart_for_replay` requires and `reset` refuses
        # without `force_full`. An ordinary mid-play level reset is still gated.
        replay_restart = action == 0 and self._last_advanced
        if self.gate is not None and not replay_restart:
            self.gate.check()
        name = action_name(action)
        payload: dict[str, Any] = {"game_id": self.game_id}
        if action == 0:
            payload["card_id"] = self.card_id
        if self.guid:
            payload["guid"] = self.guid
        if action == 6:
            payload["x"], payload["y"] = x, y

        frame = _post(f"{self.root}/api/cmd/{name}", payload, self._key, opener=self._opener)
        if "error" in frame:
            raise RuntimeError(f"{name} refused by server: {frame['error']}")
        # **Stamped here, not in `_save_state`.** What the reap clock measures is
        # time since the SERVER last heard from this game, and `_save_state` also
        # runs from `open()`'s lent-card branch and from `gate.on_change`, which
        # `gate.acknowledge()` fires before any server contact at all. Stamping
        # there would refresh the clock without touching ARC, which is precisely
        # the proxy-for-the-thing mistake.
        #
        # Epoch seconds, because this is a machine comparison.
        self.last_touched = time.time()

        previous_level = self.level
        previous_state = self.state
        self.guid = frame.get("guid") or self.guid
        self.state = str(frame.get("state", self.state))
        self.win_levels = int(frame.get("win_levels", self.win_levels) or 0)
        # **A missing level field must not read as level 0.** `self.level <
        # previous_level` is taken as a full reset below, so a frame without
        # the field would silently wipe `level_costs` and report a replay that
        # never happened -- hence the explicit refusal rather than a `0`
        # fallback.
        #
        # The fallback itself is already dead behind the shim: `score` is in
        # `arc_proxy.HIDDEN_FIELDS` and `_strip` runs on every forwarded body, so
        # a proxied run never sees it however the server answers. That is fine
        # while `levels_completed` is always present, and silent the moment it
        # is not.
        if "levels_completed" not in frame and "score" not in frame:
            raise RuntimeError(
                f"{name} returned a frame with neither `levels_completed` nor "
                f"`score`, so the level cannot be read. Frame keys: "
                f"{sorted(frame)}"
            )
        self.level = int(frame.get("levels_completed", frame.get("score", 0)) or 0)
        self.available_actions = tuple(
            action_name(a) for a in (frame.get("available_actions") or ())
        )
        self.actions_used += 1
        cleared_cost = 0
        if self.level > previous_level:
            # This action is the last one taken *from* the level just cleared,
            # which is the attribution :func:`ledger.actions_per_level` uses, so
            # its cost is the running tally plus this one.
            #
            # A jump of more than one level is not a case the API is known
            # to produce; if the
            # server ever credits two at once, the extra levels really did cost
            # zero actions of their own and are recorded as such.
            #
            # **Zero is not read as "cleared for free".** A zero-action
            # completed level is refused rather than resolved generously: 0
            # would earn the cap, so accepting it awards a
            # parsing fault the best score a level can get -- and it means a
            # genuine double advance is a decision someone has to make, not a
            # case already handled.
            gained = self.level - previous_level
            # The same figure that goes into `level_costs`, held for the
            # announcement so the two cannot disagree about what a level cost.
            cleared_cost = self.level_actions + 1
            self.level_costs = self.level_costs + (cleared_cost,) + (0,) * (gained - 1)
        self.level_actions = 0 if self.level > previous_level else self.level_actions + 1
        # **The play-opening RESET is not an action, and this tally must not
        # count it.**
        # `ledger.actions_per_level` drops it explicitly and the server agrees:
        # seven ACTION6 calls preceded by the opening RESET come back as
        # `actions: [7]`, not 8. Counted, it is +1 against the scorer on
        # element 0 of any run
        # with no full reset, and exact everywhere else -- a run whose last play
        # began with a full reset already zeroes here, further down.
        #
        # It matters because the workspace template sets `show_score=True`, so
        # the client's running score must agree with the offline scorer rather
        # than differ by one action on the first level.
        #
        # `actions_used` carries the same +1 and is left as it is: it feeds the
        # budget ceiling, where counting the opening RESET is the conservative
        # direction.
        if action == 0 and self.actions_used == 1:
            self.level_actions = 0
        # The server's own flag is not reliable: a full reset can take the
        # level 6 -> 0 with ``full_reset`` coming
        # back **False**, so a counter keyed on the flag alone reads zero
        # while the entire game replays. A level that goes *down*
        # is the fact; the flag is a hint.
        # **The reminder fires here because it is the one channel that reaches
        # every solver.** Printed into the tool result of the winning action
        # itself, it reaches the solver that never exits (WIN -> replay ->
        # WIN in one long-lived process) AND the
        # solver that reads a clear as "done" -- which therefore exits *having
        # heard it*, so its exit is a decision, not an oversight. That is what
        # lets the parent treat an undeclared exit as a question to ask rather
        # than a mistake to override.
        if frame.get("full_reset") or self.level < previous_level:
            self.full_resets += 1
            self.level_actions = 0
            self.level_costs = ()
        if not frame.get("frame"):
            self.wasted_actions += 1
        # `or frame.get("full_reset")` is defensive, not decorative: the
        # server can be wrong in the other direction too (``full_reset`` false
        # on a 6 -> 0 transition), so trust neither signal alone. Either one means
        # the board this level's tally describes is gone.
        self._account_effect(
            frame, name, payload,
            board_replaced=self.level != previous_level or bool(frame.get("full_reset")),
        )
        # Set *after* reading, so the flags describe the state the next call
        # will act in -- which is exactly when the RESET trap fires.
        self._last_advanced = self.level > previous_level
        self._last_was_reset = action == 0

        if self.gate is not None:
            self.gate.observe(self.level)

        # **After the gate observes, because the wording depends on it.** On a
        # boundary the solver has already documented the gate does not re-arm
        # (`LevelGate.observe`), and telling it to acknowledge a gate that is
        # not held would be an instruction it cannot carry out.
        if self.level > previous_level:
            self._announce_level(previous_level, cleared_cost)
        # **After the level announcement, because a winning action does both and
        # the order they are read in is the order they happened.** Fired
        # further up, before the level tally has been updated, the reminder
        # would put "you have won" in front of "level 9 cleared", which is
        # backwards; nothing about it depends on that position -- it reads a
        # switch and writes a message.
        if self.state == "WIN" and previous_state != "WIN":
            self._offer_win_replay_reminder()

        assert self._writer is not None
        record = self._writer.append(frame, level=self.level)
        self._note_row(record)
        # Persist after every action: the next action usually arrives in a
        # different process, and anything not on disk is gone.
        self._save_state()
        self._snapshot_scorecard_if_due(advanced=self.level > previous_level)
        return frame

    def history(self):
        """Every transition recorded so far, oldest first.

        The ledger is decoded once and after that only where it grew, so asking
        again after an action re-reads that action and nothing else.

        Use :meth:`last_transition` for the newest one on its own: this builds a
        list of the whole game, and the whole game is not what a question about
        the last action needs.
        """
        # The list is copied so that a caller's `ts.pop()` cannot shorten what
        # every later caller in this process sees. The transitions are frozen
        # and their grids read-only, so the copy is of references, not boards.
        return list(self._trace_reader().refresh())

    def _trace_reader(self):
        from .ledger import TraceReader

        if self._reader is None:
            self._reader = TraceReader(self.trace_path)
        return self._reader

    def last_transition(self):
        """The newest transition, or ``None`` before the first action.

        Costs the same at action 3,000 as at action 3. This is what a per-action
        loop should call; :meth:`history` decodes a whole game to answer the
        same question and gets slower as the game gets longer.
        """
        if self._chain is None:
            from .ledger import tail_state

            self._last, self._chain = tail_state(self.trace_path)
        return self._last

    @property
    def last(self):
        """The newest transition, or ``None`` before the first action.

        The same as :meth:`last_transition`, as an attribute::

            client.act(6, x=10, y=20)
            t = client.last
        """
        return self.last_transition()

    def _note_row(self, record: dict[str, Any]) -> None:
        """Advance the cached newest transition past a row just written.

        Only when the chain is already known. Seeding it here instead would
        make the first action of every process read the end of the ledger for a
        transition nobody has asked for -- and `last_transition` seeds from disk
        anyway, correctly, because the row is on disk before this is called.
        """
        if self._chain is None:
            return
        from .ledger import chain

        transition = chain(record, self._chain)
        if transition is not None:
            self._last = transition

    def status(self) -> str:
        """Level, state, action count, and this run's score terms, on one line."""
        # **Facts on the first line, warnings on their own lines after it**, so
        # a counter is never read as the tail of a sentence telling the solver
        # what to do.
        #
        # This paragraph is a comment rather than part of the docstring:
        # `status()` is the most-called method on the offered surface, so its
        # docstring is the first thing a solver reads about the call it is told
        # to make. It must not name the withheld quantity or restate the
        # warning text below.
        facts = [
            f"{self.game_id}: level {self.level}/{self.win_levels or '?'}",
            f"state={self.state}",
            f"actions={self.actions_used}",
        ]
        warnings: list[str] = []

        base = self._baseline_here
        if base:
            ratio = self.level_actions / base
            facts.append(
                f"[{self.level_actions}/{base} on this level = {ratio:.1f}x]"
            )
            # Warn at 1.0x: a cleared level typically finishes well under the
            # reference count, so crossing it is already the unusual case, and
            # warning at a higher multiple costs the actions in between.
            if ratio >= 1.0:
                warnings.append("OVER BASELINE: re-explore rather than grind")
        if self.show_score:
            # **A CEILING, not a score.** `E = min(cap, raw)`, and `raw`
            # needs the per-level figures the harness withholds, so only
            # `cap` is computable here. Printing it unlabelled would read
            # as a perfect game to a solver that cleared every level.
            facts.append(f"[cap {self.completion_cap:.3f} = {len(self.level_costs)}"
                         f"/{self.win_levels or '?'} levels — a CEILING, not "
                         f"your score: efficiency is unmeasurable here and can "
                         f"only lower it]")
            # At the winning frame the replay instrument is legal and one
            # action from being gone forever, so saying so at exactly the
            # moment it applies is worth more than a doctrine paragraph
            # read an hour earlier. Under competition rules the call is
            # refused outright, and advertising it there sends a winner
            # looking for a call that cannot be made. Say it only where it
            # is true.
            if self.state == "WIN" and not competition_mode():
                warnings.append(
                    f"WON — and `restart_for_replay()` is legal RIGHT NOW "
                    f"and illegal after any further action."
                )
        if self.wasted_actions:
            facts.append(f"wasted={self.wasted_actions}")
        if self.full_resets:
            facts.append(f"FULL RESETS={self.full_resets}")
        # The effectiveness counters are deliberately not reported here:
        # reading its own ledger for what worked is the solver's job. They stay
        # for harness-side reporting and are not offered.

        return " ".join(facts) + "".join(f"\n  <- {w}" for w in warnings)

    def _level_costs_from_trace(self) -> tuple[int, ...]:
        """Per-level costs for the **current play**, rebuilt from the ledger.

        Only used to repair a state file that predates the counter.
        :func:`ccarc.arc_3.ledger.actions_per_level` already cuts to the
        last play and attributes by the level an action was taken *from*, which
        is exactly this list; the trailing ``None`` entries are levels not yet
        cleared and are dropped.
        """
        from .ledger import rows, actions_per_level

        try:
            transitions = rows(self.trace_path)
        except Exception:  # noqa: BLE001 -- a resume must not die on a bad trace
            return ()
        n = self.win_levels or (len(self.info.baseline_actions) if self.info else 0)
        if not n:
            return ()
        counts = actions_per_level(transitions, n)
        return tuple(c for c in counts if c is not None)

    def _level_actions_from_trace(self, level: int) -> int:
        """How many actions the ledger says were spent on ``level``.

        Only used to repair a state file that predates the counter. Attribution
        matches :func:`ccarc.arc_3.ledger.actions_per_level`: an action
        belongs to the level it was taken *from*.
        """
        from .ledger import rows

        try:
            transitions = rows(self.trace_path)
        except Exception:  # noqa: BLE001 -- a resume must not die on a bad trace
            return 0
        count = 0
        previous: int | None = None
        for t in transitions:
            if (previous if previous is not None else 0) == level:
                count += 1
            previous = t.level
        return count

    def _account_effect(
        self, frame: dict[str, Any], name: str, payload: dict[str, Any], *,
        board_replaced: bool,
    ) -> None:
        """Tally, in O(1), whether this action changed anything on this level."""
        # Counted here rather than derived in `status()`: deriving means
        # parsing the whole trace, and `status()` is called many times a run.
        # Memoising would not help -- every action arrives in a new process, so
        # a per-process cache never gets a second hit.
        grids = frame.get("frame") or []
        if not grids:
            # A wasted action never reached the game, so it is not evidence
            # about the action -- but if the *level* changed anyway, the tally
            # describes a board that no longer exists and must not carry over.
            if board_replaced:
                self.level_tried = self.level_dead = self.level_repeats = 0
                self.level_revisits = 0
                self._dead_keys = []
                self._seen_keys = set()
                self._last_frame_key = ""
            return
        key = hashlib.blake2b(
            json.dumps(grids[-1], separators=(",", ":")).encode(), digest_size=16
        ).hexdigest()
        previous, self._last_frame_key = self._last_frame_key, key
        if board_replaced or not previous:
            self.level_tried = self.level_dead = self.level_repeats = 0
            self.level_revisits = 0
            self._dead_keys = []
            self._seen_keys = {key}
            return
        # Membership before insertion, and only for boards that actually moved:
        # a no-op leaves the board on a key already in the set, so counting it
        # here would double-report what ``level_dead`` already covers.
        if key != previous:
            if key in self._seen_keys:
                self.level_revisits += 1
            else:
                self._seen_keys.add(key)
        self.level_tried += 1
        if key != previous:
            return
        self.level_dead += 1
        what = f"{name}:{payload.get('x')},{payload.get('y')}" if name == "ACTION6" else name
        if what in self._dead_keys:
            self.level_repeats += 1
        else:
            self._dead_keys.append(what)
