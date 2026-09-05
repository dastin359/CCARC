"""One scorecard spanning many games -- what a leaderboard submission needs.

**One card, because a submission takes one.** :meth:`ArcClient.open` opens a
card per client, so a 25-game sweep mints 25 cards. The ARC community
leaderboard takes exactly one ``scorecard_url`` per submission and reads the
score off ARC's own card rather than anything self-reported, so 25 cards is not
a submission at all -- it is 25 of them, none complete.

**What actually binds a card to a caller.** Not the API key, and not the
``GAMESESSION`` cookie. A scorecard is state on **one backend instance**, and
the load balancer's ``AWSALBAPP-*`` stickiness cookies are the only thing that
routes a request back to it. Carry those cookies and any process, thread or jar
can play any game against the card; lose them and the card is unreachable while
the API key keeps working perfectly -- which is why the failure reads as
``game <id> not found`` and points at the wrong thing entirely.

Two consequences worth stating plainly:

* **Injecting ``card_id`` alone cannot work.** It fails on the first RESET,
  in every process.
* **The card is only as durable as one backend.** A sweep that takes days is
  betting that instance is not recycled in the meantime. Nothing here can
  prevent that, so the local ledger stays the source of truth for scoring and
  the shared card is the artifact for submission -- not the other way round.

**Idle survival.** A *game* left idle is reaped after roughly a quarter hour
(see :func:`ccarc.arc_3.session.snapshot_scorecard`). A *card* is not on that
clock. "Readable" is not "playable" -- they are different operations against
different server state -- and the sweep does not read an idle card, it adds
game N+1 to it. So the property that matters is: **a shared card accepts a new
game after a sweep-sized gap.**

**A multi-hour gap is a different claim.** The property above covers the pause
between games -- solver startup and bookkeeping. It does not cover a pause of many
hours. Treat a long pause as a card at risk: snapshot before it and verify with
a read after it.

The stickiness cookies carry a ~7-day expiry and are re-issued on every
response, so a sweep that keeps playing keeps its card reachable. The
``GAMESESSION`` cookie expires in ~1 day and does not matter for routing.
"""
from __future__ import annotations

import http.cookiejar
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .client import ROOT_URL, _api_key, _get, _post, new_session, competition_mode

__all__ = [
    "SharedCard",
    "adopt",
    "close_card",
    "export_cookies",
    "hold_reset",
    "load",
    "open_card",
    "reset_touch_path",
    "save",
    "stamp_reset_touch",
]

# The cookies that actually matter. `GAMESESSION` is carried too -- it costs
# nothing and the server may start relying on it -- but stickiness is what does
# the routing, so this is the set whose ABSENCE is an error rather than a
# curiosity.
PINNING_PREFIX = "AWSALB"


@dataclass(frozen=True)
class SharedCard:
    """A card plus the session that can reach it. Neither half is any use alone."""

    card_id: str
    cookies: tuple[dict[str, str], ...] = ()

    @property
    def pinned(self) -> bool:
        return any(c["name"].startswith(PINNING_PREFIX) for c in self.cookies)

    def to_json(self) -> str:
        return json.dumps({"card_id": self.card_id, "cookies": list(self.cookies)}, indent=2)


def export_cookies(opener: urllib.request.OpenerDirector) -> tuple[dict[str, str], ...]:
    """Lift a jar out of an opener as plain data, so it can cross a process."""
    for h in opener.handlers:
        if isinstance(h, urllib.request.HTTPCookieProcessor):
            return tuple(
                {"name": c.name, "value": c.value, "domain": c.domain, "path": c.path}
                for c in h.cookiejar
            )
    return ()


def adopt(jar: http.cookiejar.CookieJar, cookies: Iterable[dict[str, str]]) -> None:
    """Put exported cookies into a jar, so this caller lands on the card's backend."""
    for c in cookies:
        domain = c["domain"]
        jar.set_cookie(
            http.cookiejar.Cookie(
                0, c["name"], c["value"], None, False,
                domain, True, domain.startswith("."),
                c.get("path", "/"), True, False, None, True, None, None, {},
            )
        )


def open_card(
    *,
    tags: Sequence[str] = ("ccarc3",),
    api_key: str | None = None,
    root: str = ROOT_URL,
) -> SharedCard:
    """Open one card and keep the session that can reach it.

    Called by the driver, which holds the ARC key legitimately -- it builds the
    queue and scores the result from the same baselines it keeps from the solver.
    """
    opener = new_session()
    key = _api_key(api_key)
    # **The competition flag belongs on EVERY card this project opens.** A sweep
    # that submits one card and a client that opens its own must agree, or the
    # rules a run played under depend on which path happened to mint it.
    payload: dict[str, object] = {"tags": list(tags)}
    if competition_mode():
        payload["competition_mode"] = True
    card = _post(f"{root}/api/scorecard/open", payload, key, opener=opener)
    cookies = export_cookies(opener)
    shared = SharedCard(card["card_id"], cookies)
    if not shared.pinned:
        # Better to stop here than to hand every game a card none of them can
        # reach. Without a stickiness cookie each shim lands on a random backend
        # and the first RESET fails with a message that names the game.
        raise RuntimeError(
            f"scorecard {shared.card_id} opened but the response carried no "
            f"{PINNING_PREFIX}* stickiness cookie, so nothing can be routed back "
            f"to it. Cookies seen: {[c['name'] for c in cookies] or 'none'}."
        )
    return shared


def close_card(card: SharedCard, *, api_key: str | None = None, root: str = ROOT_URL) -> dict[str, Any]:
    """Close the shared card from a session that can actually see it."""
    opener = new_session()
    adopt(_jar_of(opener), card.cookies)
    return _post(f"{root}/api/scorecard/close", {"card_id": card.card_id},
                 _api_key(api_key), opener=opener)




def hold_reset(
    card: SharedCard,
    game_id: str,
    *,
    api_key: str | None = None,
    root: str = ROOT_URL,
) -> Any:
    """Issue a real ``RESET`` on the card -- the only call that holds it.

    **What actually keeps a card alive.** The reap clock does not run on
    reads, and it does not run on ordinary actions either -- those are
    guid-scoped. It runs on time since the last **card_id-carrying command**,
    and in normal play only RESET carries ``card_id``. A solver that stays
    inside one level for a long stretch never resets, so the card under it can
    starve MID-PLAY -- actions still flowing, scorecard GETs answering 200 --
    while every monitor reports health.

    So the keepalive's answer is a real RESET on a *sacrificial* game -- one no
    solver is currently playing, preferably one already banked. The cost is a
    zero-action play row on that game's card entry, which is harmless twice
    over: scoring is best-of-plays, so an empty row cannot lower a banked
    result, and a later real attempt snapshots the card at open and judges only
    its own rows (the boundary sidecar).

    Raises on a refusal or transport error, so a caller can treat any failure
    as "the card may be dying" and say so. It must not be swallowed here.
    """
    opener = new_session()
    adopt(_jar_of(opener), card.cookies)
    frame = _post(f"{root}/api/cmd/RESET",
                  {"game_id": game_id, "card_id": card.card_id},
                  _api_key(api_key), opener=opener)
    if isinstance(frame, dict) and frame.get("error"):
        raise RuntimeError(f"hold RESET on {game_id} refused: {frame['error']}")
    return frame


def reset_touch_path(scratch: str | Path | None = None) -> Path:
    """Where the last card-holding RESET's timestamp lives.

    One definition, because three writers share it: the per-game shims stamp it
    when they forward a solver's card_id-carrying RESET, :func:`hold_reset`
    callers stamp it after a keepalive RESET, and the keepalive reads its
    mtime to decide whether the card is starving. Two spellings of this path
    would split the clock in half and each half would report the other's
    silence.
    """
    root = Path(scratch or os.environ.get("CCARC3_SCRATCH")
                or (Path.home() / "ccarc3-scratch"))
    return root / "card_reset.touch"


def stamp_reset_touch(scratch: str | Path | None = None) -> None:
    """Record "the card was just held" -- never raises.

    Called from the shim's request path and from daemon threads; a stamp that
    can take either of those down is worse than a missed stamp, which costs at
    most one redundant keepalive RESET.
    """
    try:
        path = reset_touch_path(scratch)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    except OSError:
        pass


def _jar_of(opener: urllib.request.OpenerDirector) -> http.cookiejar.CookieJar:
    for h in opener.handlers:
        if isinstance(h, urllib.request.HTTPCookieProcessor):
            return h.cookiejar
    raise RuntimeError("opener has no cookie jar; it cannot hold a card")


def save(card: SharedCard, path: str | Path) -> Path:
    """Persist the card for the driver's own restarts.

    Deliberately NOT written into a workspace: the cookies are the credential
    half of the card, and the workspace is the one directory the solver is told
    to read.

    **Written atomically, like the run state in client.py.** A plain
    ``write_text`` leaves a window in which the file exists and is truncated,
    and a host that can kill the process at any moment will eventually land in
    it. The cost of landing in that window is not one lost write: a reader
    finds the file present, :func:`load` raises on the partial JSON, and the
    card carrying every banked game is stranded and unreachable.

    The mode is set on the temporary file BEFORE the rename, so the card is
    never briefly world-readable at its final path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(card.to_json(), encoding="utf-8")
    tmp.chmod(0o600)
    os.replace(tmp, path)                     # atomic on POSIX
    return path


def load(path: str | Path) -> SharedCard:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return SharedCard(raw["card_id"], tuple(raw.get("cookies") or ()))
