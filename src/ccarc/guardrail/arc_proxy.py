"""A local shim that forwards the four ARC endpoints this harness uses.

Runs in the harness process, not the solver's. Holds the credential and the
per-game ceiling, and refuses every path outside its allowlist.
"""
# **The rationale for this module is NOT in this file, and that is the point.**
# `ccarc.guardrail.arc_proxy` sits on a solver's `PYTHONPATH`, so `pydoc` prints
# the docstring and `inspect.getsource` returns the comments too. A complete
# description of the defence -- every route, every seam -- written here would
# be printed to the person it guards against, so this note says where the
# reasoning lives and not what it says.
#
# It is kept outside the package, where the solver has no path to it. Same separation as
# `ccarc.staging`: the repository is written for one audience and the copy the
# solver imports for another.
#
# What stays here is what someone editing THIS file has to know, stated as rules
# rather than as a threat model:
#
# **Allowlist, never denylist.** Only the endpoints a solver legitimately needs
# are forwarded; anything else is refused. A new upstream endpoint is closed by
# default rather than open until someone notices.
#
# **The ceiling is counted here, not handed to the child.** A limit the solver
# can read is a limit the solver can work backwards from, so this process keeps
# the count and the child's environment carries no such number. It also makes
# the ceiling unforgeable: the count is kept where the solver cannot write it.
#
# **The credential stays in this process.** The solver is given a loopback URL
# and no key, so a hand-rolled request has nothing to authenticate with.
#
# **What this does not claim.** This closes the easy paths, by construction. It
# is not a sandbox, and it is not the only layer -- see `confine.py` and
# `network.py`, which are.
from __future__ import annotations

import json
import os
import re
import http.cookiejar
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = "https://three.arcprize.org"

# Every path a solver legitimately needs, and nothing else. Derived from
# client.py's network call sites **minus one**: the games LIST is excluded
# deliberately and must stay excluded. The reason is not stated here,
# because a comment is not a hiding place.
#
# Reading this tuple as a transcription of a client's surface is how the
# exclusion gets "restored" by someone tidying up. It is not a transcription.
#
# `/api/games/<id>` IS forwarded, and that is not a relaxation of the rule above.
# The official toolkit's `make()` fetches it and returns None on a refusal, so a
# 403 here does not withhold anything -- it stops the agent playing at all. What
# that response is allowed to CARRY is decided by `_hidden_fields`, one layer
# down. The list endpoint stays refused in every mode: it answers for all 25
# games, and no run needs 24 environments it is not playing.
ALLOW = (
    re.compile(r"^/api/cmd/[A-Z0-9_]+$"),                     # RESET, ACTION1..7
    re.compile(r"^/api/games/[a-z0-9-]+$"),                   # ONE game's metadata
    re.compile(r"^/api/scorecard/open$"),
    re.compile(r"^/api/scorecard/close$"),
    re.compile(r"^/api/scorecard/[0-9a-f-]+/[a-z0-9-]+$"),    # read one card
)


def _allowed(path: str) -> bool:
    return any(p.match(path) for p in ALLOW)


def _sc():
    """`shared_card`, imported on use. This module keeps its imports to the
    standard library on purpose -- `shared_card` pulls in `client`, and a
    module-scope package import here would make the shim's import graph the
    package's. One stamp per accepted RESET does not justify that."""
    from . import shared_card  # noqa: PLC0415 -- see docstring

    return shared_card


# Every action the client sends -- RESET is `/api/cmd/RESET` and counts, exactly
# as `ArcClient.actions_used` counts it, because both increment on the same call.
_CMD = re.compile(r"^/api/cmd/[A-Z0-9_]+$")
# The one path that names a card. Kept beside `_CMD` because the alias below
# rewrites exactly these two shapes and nothing else.
_CARD_PATH = re.compile(r"^/api/scorecard/([0-9a-f-]{8,})/([a-z0-9-]+)$")

#: Header `ArcClient` stamps on every request it makes, so the shim can tell a
#: client-driven action from a hand-rolled one. Carries no information — its
#: presence IS the signal. See the BARE ACTION tripwire in `Handler._forward`.
CLIENT_MARK = "X-CCARC3-Client"

class ProxyState:
    """One game's cap and one game's HTTP session to ARC.

    **Per game, because both are per game.** A driver may hold two environments
    in one process, and module-level state is silently wrong there: the action
    counter would bill one game for the other's moves, and -- worse -- a single
    cookie jar would pin both games' scorecards to one ARC session. Card reads
    are session-bound, so the second game's card answers 404 while its actions
    keep succeeding -- a failure with no other symptom.

    A shared counter is a bug you would eventually see in the numbers. A shared
    session is one you would not.

    **One exception: a deliberately shared card.** A
    leaderboard submission takes one `scorecard_url`, so the whole sweep has to
    land on one card -- and a card is reachable only from a session carrying its
    `AWSALBAPP-*` stickiness cookies. :meth:`adopt_session` seeds this shim's jar
    with the driver's, so every game routes to the instance holding the shared
    card. The counter stays per game either way; it is the pinning that is
    shared, and only when the card is. See :mod:`ccarc.arc_3.shared_card`.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.max_actions = 0        # 0 disables the cap
        self.actions_used = 0
        # Which game this shim serves. Empty means "do not check", which is what
        # the module-level default and the connectivity probe want.
        self.game_id = ""
        self._opener: urllib.request.OpenerDirector | None = None
        # Cookies to seed every session this shim builds, so it lands on the
        # backend holding a shared card. Empty means "your own session", which
        # is the per-game default.
        self._adopted: tuple[dict[str, str], ...] = ()
        # **A card is a 24-hour object; a run can be longer.** A scorecard
        # expires about 24 hours after it is opened, into a 404 that never
        # recovers, so a long run can outlive its card and the replacement has
        # a different id.
        #
        # A resumed solver's context already contains the FIRST card's id, in
        # tool results it cannot be made to un-see. Showing it a second id
        # invites it to reason about harness plumbing instead of the game, so
        # the solver keeps addressing `alias` -- the id it first saw -- while
        # this rewrites to `real` on the way out and back on the way in. The
        # swap is driver-side machinery, set through `lend_card_as` by the
        # driver, which holds both ids; only the solver-facing view is kept
        # stable. Neither half is optional: an un-rewritten response body
        # carries the real id in `card_id` and in the text of a 404.
        self.card_alias = ""
        self.card_real = ""

    def set_budget(self, max_actions: int, *, used: int = 0) -> None:
        """Arm the cap for one game.

        `used` seeds the counter from a run already in progress. The proxy's
        count lives in memory and the solver's does not, so a resumed game whose
        proxy restarted would otherwise start from zero -- the ceiling would bind
        at `used + max_actions` and grow with every interruption.
        Callers read the figure from the workspace's `trace.state.json`.
        """
        with self._lock:
            self.max_actions, self.actions_used = int(max_actions), int(used)
        # A new game means a new card; carrying the previous game's pinning over
        # is how a stale session survives into a run that did not open it.
        self.reset_session()

    def reserve(self) -> str | None:
        """Take one action slot, or return the refusal. Atomic.

        Deciding and taking under ONE lock acquisition is the whole point. A
        check that answers under the lock and a charge that bills under the
        lock, run as two acquisitions with the upstream round-trip in between,
        would let every request already in flight pass the check before any of
        them paid: N concurrent actions at the ceiling would all read "not
        exhausted", and the ceiling would overshoot by however many connections
        the caller chose to open.

        **The caller is the party the ceiling exists to constrain.** It reaches
        this shim over loopback with no key, and nothing stops it opening fifty
        connections at once.

        The slot comes back via :meth:`refund` when upstream did not accept it,
        which is what keeps the rule that a 502 costs the solver nothing.
        """
        with self._lock:
            over = self._over()
            if over:
                return over
            self.actions_used += 1
        return None

    def refund(self) -> None:
        """Hand back a reserved slot upstream did not accept.

        Reserving up front bills the action before knowing whether ARC performed
        it, so this keeps the rule intact: a 502 costs the solver nothing
        at either end, and the count still tracks `ArcClient.actions_used`, which
        increments only while processing a frame the server actually returned.

        Refunding is deliberately the only way the counter goes down, and it
        never goes below zero: over-counting caps a run early, which is a bad
        run; under-counting uncaps it, which is a bad number in a submission.
        """
        with self._lock:
            if self.actions_used > 0:
                self.actions_used -= 1

    def _over(self) -> str | None:
        """The refusal text if the ceiling is reached. **Caller holds the lock.**

        One copy of the ceiling rule, in one place: `reserve` and `exhausted`
        would otherwise state the same condition and the same message twice,
        and two copies of a rule drift.

        No figure in the message: this docstring is on the
        solver's import path, and stating the arithmetic that makes the figure
        worth hiding is the same disclosure as printing it.
        """
        if self.max_actions and self.actions_used >= self.max_actions:
            # One fact, once: the run is finished, so anything past the
            # refusal is noise.
            return "action budget exhausted."
        return None

    def exhausted(self) -> str | None:
        """The refusal if the ceiling is reached, without taking a slot.

        Read-only, for callers that want to ask. `_forward` must use
        :meth:`reserve` instead -- asking and then acting is the race this
        method cannot be used to avoid.
        """
        with self._lock:
            return self._over()

    def upstream(self) -> urllib.request.OpenerDirector:
        """This game's persistent session to ARC, cookie jar and all.

        **Forwarding the client's cookies is not enough.** ARC binds a scorecard
        to the HTTP session and its load balancer pins that session with four
        `AWSALBAPP-*` cookies. Building a fresh request per call -- which
        `urllib.request.urlopen` does -- lets the balancer re-pin on every hop,
        so the card lands on a backend that never heard of it and reads through
        a shim without a persistent jar mostly fail. The jar lives here rather
        than in the client because the client is the thing being kept at arm's
        length.
        """
        with self._lock:
            if self._opener is None:
                jar = http.cookiejar.CookieJar()
                for c in self._adopted:
                    domain = c["domain"]
                    jar.set_cookie(
                        http.cookiejar.Cookie(
                            0, c["name"], c["value"], None, False,
                            domain, True, domain.startswith("."),
                            c.get("path", "/"), True, False, None, True, None, None, {},
                        )
                    )
                self._opener = urllib.request.build_opener(
                    urllib.request.HTTPCookieProcessor(jar)
                )
            return self._opener

    @property
    def card_is_lent(self) -> bool:
        """True once :meth:`adopt_session` pinned this shim to a driver's card."""
        return getattr(self, "_card_is_lent", False)

    def adopt_session(self, cookies: "tuple[dict[str, str], ...]") -> None:
        """Route this shim to the backend holding a shared scorecard.

        Called before the game starts. `set_budget` drops the session on purpose
        -- a new game must not inherit the previous one's pinning -- so the
        adopted cookies are kept separately and re-seeded into every session
        this shim builds afterwards. Storing them only in the live jar would
        make the order of these two calls silently load-bearing.
        """
        with self._lock:
            self._adopted = tuple(cookies)
            self._opener = None
            # A card reached through adopted cookies belongs to the driver, not
            # to this game. See the close refusal in `_forward`.
            self._card_is_lent = True

    def reset_session(self) -> None:
        with self._lock:
            self._opener = None

    def lend_card_as(self, alias: str, real: str) -> None:
        """Serve `real` while the solver keeps addressing `alias`.

        Called by the driver after a card is replaced mid-run. `alias == real`
        is the ordinary case and disables the rewrite entirely.
        """
        with self._lock:
            self.card_alias, self.card_real = alias or "", real or ""

    def to_upstream(self, path: str, payload: bytes | None):
        """(path, payload, refusal) with the alias resolved to the real card.

        Returns a refusal string if the request names the REAL card id. The
        solver has never been shown that string, so a request carrying it is
        either a probe or a harness bug, and forwarding it would work -- which
        is what makes it worth failing closed on rather than logging.
        """
        alias, real = self.card_alias, self.card_real
        if not alias or not real or alias == real:
            return path, payload, None
        if real in path or (payload and real.encode() in payload):
            return path, payload, (
                "this request names a card id this shim never issued")
        m = _CARD_PATH.match(path)
        if m and m.group(1) == alias:
            path = f"/api/scorecard/{real}/{m.group(2)}"
        if payload and alias.encode() in payload:
            try:
                body = json.loads(payload)
            except ValueError:
                return path, payload, None
            if isinstance(body, dict) and body.get("card_id") == alias:
                body["card_id"] = real
                payload = json.dumps(body).encode()
        return path, payload, None

    def to_solver(self, body: bytes) -> bytes:
        """Put the alias back into whatever ARC said, error bodies included."""
        alias, real = self.card_alias, self.card_real
        if not alias or not real or alias == real:
            return body
        return body.replace(real.encode(), alias.encode())


# A server built straight from `Handler` has no game attached, so it falls
# back to this. Real runs always carry their own.
_DEFAULT = ProxyState()


def set_budget(max_actions: int, *, used: int = 0) -> None:
    """Arm the default state. Per-game callers use `Proxy.set_budget`."""
    _DEFAULT.set_budget(max_actions, used=used)


# **An allowlisted endpoint still leaks the baselines, so responses are filtered
# too.** `POST /api/scorecard/close` is a call every solver legitimately makes,
# and its response body carries, per run,
#
#     "level_actions":          [a0, a1, ...]
#     "level_baseline_actions": [h0, h1, ...]
#     "level_scores":           [s0, s1, ...]
#
# -- the human medians for every level of the game, handed over verbatim. That is
# exactly what `CCARC3_HIDE_BASELINES` withholds.
#
# `level_scores` has to go with them: ARC's score is 100*min(1.15, (h/a)^2) and the
# solver knows its own `a`, so a score inverts straight back to `h`. The aggregate
# `score` fields go too -- a solver has no legitimate use for its own RHAE while
# the run is in progress, and that is the number the arm is built to withhold.
#
# **Written symbolically because this file is on the solver's PYTHONPATH.** A
# worked example carrying a real card's arrays would put one environment's
# complete median list inside the module whose whole job is to withhold it --
# reachable by `inspect.getsource(arc_proxy)` even from a stripped workspace.
# Twice over: `level_actions[0]` beside `level_scores[0]` invert to `h0` on
# their own. A sample that demonstrates a leak must not be one.
#
# `client.py` reads exactly one of these -- `score`, as the fallback when a frame
# carries no `levels_completed` -- so that fallback is dead behind this shim,
# because `_strip` runs on every forwarded body. Harmless while the server always
# sends `levels_completed`, and silent if it ever stops, which is why the client
# refuses a frame carrying neither rather than inferring level 0.
# Nothing else here is read; `close()` is called for its side effect
# and `snapshot_scorecard` reads `actions_by_level`, which is untouched.
#
# **The baseline never crosses this shim, and there is no switch that lets it.**
#
# ARC's deployed API serves `baseline_actions` from `/api/games` and
# `/api/games/<id>` to any key holder, and `/api/scorecard/close` returns
# `level_baseline_actions`; the official toolkit hangs the array on the object
# `make()` returns. So the field does arrive, and removing it here is the job.
#
# Scores go with it. A per-level score inverts to the baseline against an
# action count the agent already holds, so leaving scores in would surrender
# the same quantity by a second route.
#
# There is no mode that lets it through, and no switch that can be defaulted
# off. A switch able to disclose the withheld number would turn the
# withholding into a claim a reader must take on trust: they would have to
# believe it was never set, and no artefact can show them that. With no such
# switch the property is structural, which is the only kind worth publishing.
_BASELINE_FIELDS = frozenset({
    "level_baseline_actions",
    "baseline_actions",
})
_SCORE_FIELDS = frozenset({
    "level_scores",
    "score",
    "scores",
    "tags_scores",
})
HIDDEN_FIELDS = _BASELINE_FIELDS | _SCORE_FIELDS

def _hidden_fields() -> frozenset[str]:
    """The response fields this shim removes. Not configurable."""
    return HIDDEN_FIELDS


def _unwrap(body: bytes, content_encoding: str = "") -> bytes:
    """Undo gzip/deflate so the filter sees JSON, not a compressed body.

    ``_filtered`` fail-opens on anything ``json.loads`` cannot parse, so a
    compressed JSON response must be decompressed before the field filter
    runs: otherwise the filter reads a form of the body that is not the form
    being sent.
    """
    enc = (content_encoding or "").lower()
    gzipped = "gzip" in enc or (len(body) >= 2 and body[:2] == b"\x1f\x8b")
    if gzipped:
        import gzip as _gzip  # noqa: PLC0415
        import zlib as _zlib  # noqa: PLC0415
        try:
            return _gzip.decompress(body)
        # **A truncated stream raises `EOFError`, not `OSError`.** `gzip` raises
        # `BadGzipFile` (an `OSError`) for a bad header, `EOFError` when the
        # stream stops early, and `zlib.error` for some corrupt payloads, so
        # all three are caught. An escaping one would unwind out of
        # `Handler._forward` and give the solver a connection reset with no
        # status line, instead of the pass-through this module promises: an
        # empty or truncated 5xx body must reach the solver unchanged, and a
        # body that cannot be parsed is not JSON and carries no baselines.
        except (OSError, EOFError, _zlib.error):
            pass
    if "deflate" in enc:
        import zlib  # noqa: PLC0415
        try:
            return zlib.decompress(body)
        except zlib.error:
            try:
                return zlib.decompress(body, -zlib.MAX_WBITS)
            except zlib.error:
                pass
    return body


def _strip(node, hidden=None):
    """Recursively drop every hidden field from a decoded JSON body."""
    if hidden is None:
        hidden = _hidden_fields()
    if isinstance(node, dict):
        return {k: _strip(v, hidden) for k, v in node.items() if k not in hidden}
    if isinstance(node, list):
        return [_strip(v, hidden) for v in node]
    if isinstance(node, str):
        # A string that is itself JSON is still a body. ``json.loads`` on the
        # outer object leaves it opaque, so it is parsed and filtered too.
        s = node.lstrip()
        if s[:1] in "{[":
            try:
                inner = json.loads(node)
            except ValueError:
                return node
            if isinstance(inner, (dict, list)) and _hides_a_secret(inner, hidden):
                return json.dumps(_strip(inner, hidden), separators=(",", ":"))
        return node
    return node


def _hides_a_secret(node, hidden=None) -> bool:
    if hidden is None:
        hidden = _hidden_fields()
    if isinstance(node, dict):
        return any(k in hidden or _hides_a_secret(v, hidden) for k, v in node.items())
    if isinstance(node, list):
        return any(_hides_a_secret(v, hidden) for v in node)
    return False


def _filtered(body: bytes, content_encoding: str = "") -> bytes:
    """Strip baselines from a response, passing anything unparseable through.

    Deliberately fail-open on a body that is not JSON: the client has careful
    error handling and an empty or truncated 5xx body must reach it unchanged.
    A non-JSON body cannot contain the fields anyway — once it has been
    decompressed. Compressed JSON could, which is why :func:`_unwrap` runs first.
    """
    body = _unwrap(body, content_encoding)
    try:
        decoded = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body
    return json.dumps(_strip(decoded)).encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):        # noqa: A003 - stdlib signature
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def _refuse(self, path: str) -> None:
        # Loud on purpose: a refusal is the signal that a solver went looking,
        # and it belongs in the log.
        sys.stderr.write(f"REFUSED {path} — not on the solver allowlist\n")
        sys.stderr.flush()
        body = json.dumps({"error": "endpoint not available to solvers"}).encode()
        self.send_response(403)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _deny(self, code: int, message: str) -> None:
        body = json.dumps({"error": message}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def state(self) -> ProxyState:
        return getattr(self.server, "state", _DEFAULT)

    def _forward(self, method: str) -> None:
        path = bare = self.path.split("?", 1)[0]
        # **Drain the body before answering anything, including a refusal.** On
        # a keep-alive connection an unread request body is parsed as the next
        # request line, so the victim of a reply that returns early is the NEXT
        # call on that connection, which is a legitimate one. A defence that
        # corrupts the traffic it permits is worse than the request it
        # declined.
        n = int(self.headers.get("Content-Length") or 0)
        payload = self.rfile.read(n) if n else None

        # **Resolve the alias FIRST, so the string that is checked is the string
        # that is sent.** One variable is matched and forwarded; deriving the
        # upstream URL from anything else would let an unchecked byte reach
        # ARC.
        path, payload, wrong_card = self.state.to_upstream(path, payload)
        if wrong_card:
            sys.stderr.write(f"WRONG CARD {path}: {wrong_card}\n")
            sys.stderr.flush()
            self._deny(403, wrong_card)
            return

        if not _allowed(path):
            self._refuse(path)
            return
        charge = bool(_CMD.match(path))

        # **A shim is dedicated to one game, and it checks.** Without the
        # check, a shim armed for game B would forward game A's `/api/cmd/*`
        # and charge B -- and with games in flight concurrently, every
        # sibling's port is a loopback scan away. The cap lives outside the
        # child's environment precisely so the solver cannot raise it; billing
        # it to a neighbour raises it just as effectively.
        # **And it fails closed.** A guard of the shape `if wanted and payload:`
        # runs only when a request volunteers a body naming a game, so a
        # `/api/cmd/*` request with no body would skip the check and be billed
        # to whichever shim received it -- a solver choosing what to put in its
        # own request body must not decide whether the check runs.
        #
        # So, for commands only: a shim dedicated to a game requires every
        # action to name that game. Scorecard open/close carry no `game_id` and
        # are left alone.
        wanted = self.state.game_id
        if wanted and charge:
            try:
                asked = (json.loads(payload) or {}).get("game_id") if payload else None
            except ValueError:
                asked = None
            if asked != wanted:
                sys.stderr.write(f"WRONG GAME {path}: shim serves {wanted}, "
                                 f"request names {asked!r}\n")
                sys.stderr.flush()
                self._deny(403, f"this shim serves {wanted}, not {asked!r}; "
                                f"every action must name its own game")
                return

        # **Closing a lent card is never a solver's call.** `/api/scorecard/close`
        # is on the allowlist because a per-game run legitimately closes its own
        # card through this shim at the end. Under the sweep's shared card it is
        # the one forwarded endpoint that can finalize the whole submission
        # artifact -- 25 games' scores on one card -- and `ArcClient.close`
        # already declines to call it for a card it does not own. That guard
        # lives in the client, which a solver can bypass with a hand-rolled
        # request; this one lives where the solver cannot reach it.
        # **Minting a second card is never a solver's call either.** A shim
        # whose session was adopted is playing on a card the DRIVER opened, and
        # `ArcClient.open` already refuses to mint another on an adopted
        # session -- a second card would leave the sweep with one card per game
        # and nothing to submit. That guard is in the client too, so it is
        # repeated here where the solver cannot reach it.
        #
        # Under the sweep both endpoints are surface the solver has no use for,
        # and `open` is the more dangerous of the two: a card the harness never
        # learns about scores silently alongside the real one, and the loss is
        # invisible until submission.
        #
        # A standalone per-game run does NOT adopt a session, so it mints its
        # own card through here.
        if self.state.card_is_lent and path == "/api/scorecard/open":
            sys.stderr.write("REFUSED open: this shim plays on a card the "
                             "driver opened; minting another would score "
                             "somewhere nobody collects\n")
            sys.stderr.flush()
            self._deny(403, "this shim already plays on the driver's card; "
                            "opening another would score onto a card the "
                            "harness will not submit")
            return

        if self.state.card_is_lent and path == "/api/scorecard/close":
            sys.stderr.write("REFUSED close: this shim's card is lent by the "
                             "driver and closing it would end the sweep\n")
            sys.stderr.flush()
            self._deny(403, "this card is lent by the driver; closing it would "
                            "finalize a sweep that is still running")
            return

        # **An action that did not come through the client is logged, and by
        # default refused.** Every guard that prevents a run-destroying move --
        # the double-RESET trap, acting-while-dead, the gate -- lives in
        # `ArcClient`, so a solver that drives this shim with raw HTTP walks
        # past all of them, and the ledger records nothing because the ledger
        # is written by the client too. A bare RESET during orientation closes
        # a play at zero actions -- harmless there because a RESET at counter
        # zero is unbilled, and a catastrophe one action after a level advance,
        # where the same request is a silent full game reset that no ledger row
        # would show.
        #
        # The log line comes first either way: the client marks its requests
        # and a bare action gets one loud line in the log.
        if charge and not self.headers.get(CLIENT_MARK):
            sys.stderr.write(
                f"BARE ACTION {path} — this request did not come through "
                f"ArcClient: no guard checked it and no ledger row records it\n")
            sys.stderr.flush()
            # **The wall is policy: the solver reaches the game through the
            # Python client API and nothing else.**
            #
            # The header is forgeable -- the client runs inside the solver's
            # process and its source is staged where the solver can read it --
            # so this is not authentication and must never be described as
            # such. What the refusal buys is the *accident*: a bare RESET one
            # action after a level advance is a silent full game reset that
            # writes NO ledger row, so the trace stops matching the game while
            # still looking complete -- and the trace is what replay, scoring
            # and every published claim rest on. Corrupted evidence is worse
            # than a lost run.
            #
            # It also draws a line worth having: a bypass requires reading
            # this package's source and deliberately forging a header, which is
            # visible in the transcript and cannot be mistaken for exploration.
            #
            # Every legitimate `/api/cmd/*` sender goes through `client._post`,
            # which sets the mark -- including `shared_card.hold_reset`, which
            # imports that same `_post`.
            if os.environ.get("CCARC3_REQUIRE_CLIENT", "1") != "0":
                # 403 for the reason the ceiling refusal gives below:
                # `client._send` raises on 4xx without retrying, so the caller
                # gets one clear terminal error instead of three rounds of
                # backoff against a wall that will not move.
                self._deny(403, "this shim serves ArcClient only: send actions "
                                "through the client so that its guards run and "
                                "the ledger records the frame")
                return
        if charge:
            # 403 and not 429: `client._send` raises on 4xx without retrying, so
            # the solver gets one clear terminal error rather than three rounds
            # of backoff against a wall that will not move.
            over = self.state.reserve()
            if over:
                sys.stderr.write(f"BUDGET {path} — {over}\n")
                sys.stderr.flush()
                self._deny(403, over)
                return
        headers = {
            "X-API-Key": os.environ["ARC_API_KEY"],
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        # **Cookies must survive the hop, or nothing works.** ARC binds a
        # scorecard to the HTTP *session*, not to the API key -- open a card on
        # one connection and RESET on another and the server answers
        # ``game <id> not found`` (see :func:`ccarc.arc_3.client.new_session`).
        # The saved sessions carry a ``GAMESESSION`` cookie and an
        # ``AWSALBAPP-0`` load-balancer stickiness cookie, so a proxy that builds
        # a clean request per call makes every solver call a new session and
        # every game unreachable the moment it is opened.
        #
        # **The client's cookies are deliberately NOT forwarded.**
        # `http.cookiejar` refuses to set a `Cookie` header on a request that
        # already has one, so relaying the client's header would silence this
        # process's own jar entirely. The client's jar meanwhile holds the
        # ALB's `AWSALBAPP-N=_remove_` tombstones as if they were values and
        # sends them back, which unpins the session. The card then lands on a
        # backend that has never heard of it and every read answers
        # `404 card_id not found`.
        #
        # The proxy owns the upstream session. That is the whole point of it
        # holding the key, and the session belongs with the credential.
        # **Send the string that was checked, not the one it was derived from.**
        # Closed structurally rather than by a rule about query strings: one
        # variable is both matched and sent, so a query appended to an
        # allowlisted path is dropped rather than forwarded and an unchecked
        # byte has no path to ARC. A separate "reject queries" guard would be a
        # second rule to keep in step with this one.
        #
        # Nothing legitimate is lost: every URL `client.py` builds is bare.
        # **Compare against the query-stripped ORIGINAL, not against `path`**,
        # which the card alias may have rewritten: comparing against the
        # rewritten path would log a DROPPED QUERY for every aliased request
        # that made none, and a false alarm on a security warning is what
        # teaches a reader to ignore the true ones.
        if self.path != bare:
            # Still loud. A solver appending a query is the same signal as a
            # solver probing a path -- it belongs in the log either way.
            sys.stderr.write(f"DROPPED QUERY {self.path!r} -> {bare!r}\n")
            sys.stderr.flush()
        req = urllib.request.Request(
            UPSTREAM + path, data=payload, method=method, headers=headers,
        )
        set_cookies: list[str] = []
        encoding = ""
        try:
            with self.state.upstream().open(req, timeout=120) as r:
                body, code = r.read(), r.getcode()
                set_cookies = r.headers.get_all("Set-Cookie") or []
                encoding = r.headers.get("Content-Encoding") or ""
        except urllib.error.HTTPError as exc:
            # Pass the upstream status and body through unchanged. The client has
            # careful 4xx/5xx handling and retry logic; flattening errors here
            # would break it in ways that look like game bugs. Cookies come back
            # on error responses too -- an ALB re-pins on a 4xx like any other.
            body, code = exc.read(), exc.code
            set_cookies = exc.headers.get_all("Set-Cookie") or []
            encoding = exc.headers.get("Content-Encoding") or ""
        except (urllib.error.URLError, TimeoutError) as exc:
            body, code = json.dumps({"error": f"upstream: {exc}"}).encode(), 502
        if charge and not (200 <= code < 300):
            # The slot was taken before the request went out; upstream declined
            # it, so give it back.
            self.state.refund()
        # **The card's reap clock runs on card_id-carrying commands, and this
        # is where they pass.** In normal play only RESET carries `card_id`
        # (client.py `_send`: `if action == 0: payload["card_id"] = ...`), so a
        # RESET the upstream accepted is the one event that provably held the
        # card. Stamping it here -- driver-side, where every solver's traffic
        # converges -- is what lets a keepalive measure reset-silence rather
        # than fire on a blind interval. Success only: a refused RESET held
        # nothing, and stamping it would hide a starving card behind a
        # healthy-looking clock.
        # Wrapped whole: this sits in the serving path before the response is
        # written, and a bookkeeping stamp that can turn a served frame into a
        # dropped connection is worse than a missed stamp.
        if (path == "/api/cmd/RESET" and 200 <= code < 300 and payload):
            try:
                if (json.loads(payload) or {}).get("card_id"):
                    _sc().stamp_reset_touch()
            except Exception:                       # noqa: BLE001 -- see above
                pass
        # Filter on the way back, not just on the way in. Content-Length is
        # recomputed below from the filtered body, so this must happen first.
        body = _filtered(body, encoding)
        # Last, after the baseline filter and after Content-Length is recomputed
        # below from whatever this returns.
        body = self.state.to_solver(body)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for value in set_cookies:
            self.send_header("Set-Cookie", value)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):       # noqa: N802
        self._forward("GET")

    def do_POST(self):      # noqa: N802
        self._forward("POST")


class _Server(ThreadingHTTPServer):
    """A server that knows which game it is serving."""

    daemon_threads = True
    state: ProxyState


class Proxy:
    """One listening shim, dedicated to one game.

    In-process rather than a subprocess, because the caller is the one thing
    that already holds the key legitimately -- the runner builds the queue and
    scores the result from the same baselines it keeps away from the solver. A
    subprocess would need the key handed to it anyway and would outlive a driver
    that died.

    Port 0 always: with several games in flight there is no single well-known
    port to claim, and a fixed one turns a not-yet-reaped predecessor into
    `Address already in use` at exactly the wrong moment.
    """

    def __init__(self, port: int = 0, game_id: str = "") -> None:
        if not os.environ.get("ARC_API_KEY"):
            raise RuntimeError("arc_proxy: no ARC_API_KEY in env; refusing to start")
        self.state = ProxyState()
        self.state.game_id = game_id
        self._server = _Server(("127.0.0.1", port), Handler)
        self._server.state = self.state
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, bound = self._server.server_address[:2]
        self.url = f"http://{host}:{bound}"

    def set_budget(self, max_actions: int, *, used: int = 0) -> None:
        self.state.set_budget(max_actions, used=used)

    def adopt_session(self, cookies: "tuple[dict[str, str], ...]") -> None:
        self.state.adopt_session(cookies)

    def lend_card_as(self, alias: str, real: str) -> None:
        self.state.lend_card_as(alias, real)

    @property
    def actions_used(self) -> int:
        return self.state.actions_used

    @property
    def max_actions(self) -> int:
        return self.state.max_actions

    def shutdown(self) -> None:
        """Stop listening and free the port. Safe to call twice."""
        try:
            self._server.shutdown()
            self._server.server_close()
        except OSError:
            pass




def main() -> int:
    if not os.environ.get("ARC_API_KEY"):
        sys.stderr.write("arc_proxy: no ARC_API_KEY in env; refusing to start\n")
        return 2
    port = int(os.environ.get("CCARC3_PROXY_PORT", "8787"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    sys.stderr.write(f"arc_proxy: listening on 127.0.0.1:{port} -> {UPSTREAM}\n")
    sys.stderr.flush()
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
