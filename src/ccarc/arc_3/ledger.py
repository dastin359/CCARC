"""The transition ledger: an append-only record of ``(before, action) -> after``.

The SDK already records enough to reconstruct transitions -- ``FrameData``
carries ``action_input``, so the action that produced a frame travels with it --
but it records *states*, and hypothesis testing is about *transitions*. This
module writes the transition view and reads it back, so that no solver has to
re-derive a loader and every run's trace has the same shape.

The design rule this serves: **frames leave
context, they never leave disk.** A cleared level's frames stop being worth
attending to, but they remain evidence, and :mod:`ccarc.arc_3.rules` replays
against all of them.

Nothing here imports ``arc_agi_3``. Records are plain dicts of the shape
``FrameData.model_dump()`` produces, so this is testable without the SDK.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

# Deferred: see `._numpy`. A call that only reads metadata never builds an
# array, so numpy is imported the first time a grid is decoded rather than at
# import time.
from ._numpy import lazy_numpy

np = lazy_numpy(globals())

from .grids import as_grid

__all__ = [
    "ACTION_NAMES",
    "Chain",
    "LedgerRow",
    "Transition",
    "TraceReader",
    "TraceWriter",
    "chain",
    "load",
    "action_name",
    "infer_levels",
    "rows",
    "tail_state",
]

ACTION_NAMES = {
    0: "RESET",
    1: "ACTION1",
    2: "ACTION2",
    3: "ACTION3",
    4: "ACTION4",
    5: "ACTION5",
    6: "ACTION6",
    7: "ACTION7",
}


def action_name(raw: Any) -> str:
    """Normalise an action id to its name.

    ``GameAction`` sets ``_value_`` to the numeric id, so a JSON dump yields an
    int; callers holding the enum itself pass a name. Accept either.
    """
    if isinstance(raw, str):
        return raw.upper()
    if isinstance(raw, bool):  # bool is an int subclass; reject it explicitly
        raise ValueError(f"not an action id: {raw!r}")
    if isinstance(raw, int):
        try:
            return ACTION_NAMES[raw]
        except KeyError:
            raise ValueError(f"no GameAction with id {raw}") from None
    name = getattr(raw, "name", None)
    if isinstance(name, str):
        return name.upper()
    raise ValueError(f"cannot read an action name from {raw!r}")


class _Boards:
    """The grids of one record: decoded, or the bytes to decode them from.

    Decoding is deferred because it is nearly all of the cost of reading a
    ledger, and most reads do not want it. On a long trace the boards are the
    overwhelming share of a full read -- parsing every record with the
    `frames` array cut out costs a small fraction of decoding them.

    A caller that filters on `level`, `action` or `board_replaced` never looks
    at a board, and must not pay for every board in the file to find the
    handful it wants: the only way to ask a cheap question must not be an
    expensive call.

    Holds a reference to the previous record's boards rather than its array, so
    `before` costs nothing until something asks for it, and a wasted action
    (which renders no frame and carries its predecessor forward) resolves
    through the same reference instead of forcing the chain to decode.
    """

    __slots__ = ("_source", "_grids", "_previous")

    def __init__(self, source: Any, previous: "_Boards | None") -> None:
        # `source` is the raw bytes of the `frames` array, or an already
        # decoded list when the record had to be parsed whole.
        self._source = source
        self._grids: list[np.ndarray] | None = None
        self._previous = previous

    def _decode(self) -> list[np.ndarray]:
        if self._grids is None:
            raw = self._source
            self._grids = _grids(json.loads(raw) if isinstance(raw, (bytes, str))
                                 else raw)
            self._source = None            # let the bytes go
        return self._grids

    @property
    def after(self) -> "np.ndarray | None":
        grids = self._decode()
        if grids:
            return grids[-1]
        return None if self._previous is None else self._previous.after

    @property
    def before(self) -> "np.ndarray | None":
        return None if self._previous is None else self._previous.after

    @property
    def intermediate(self) -> tuple:
        grids = self._decode()
        if grids:
            return tuple(grids)
        if self._previous is None:
            return ()
        return (self._previous.after,)


@dataclass(frozen=True)
class Transition:
    """One action and the state change it produced."""

    index: int
    """Position in the game, counting every action including RESET."""

    level: int
    action: str
    params: dict[str, Any]
    boards: "_Boards" = field(repr=False, compare=False)
    """The grids, decoded when something asks for them.

    Reached through ``before``, ``after`` and ``intermediate``.
    """

    score_before: int
    score_after: int
    state: str
    full_reset: bool
    available_actions: tuple[str, ...]
    hypothesis: Any = None
    """Whatever was stamped into ``ActionInput.reasoning`` -- the server stores
    and echoes it verbatim, which makes the trace self-describing for free."""

    crosses_level: bool = False
    """This action ended a level, so ``before`` and ``after`` are different boards.

    Such a transition is not a transition *within* a game state: the whole
    board is replaced, so anything spatial concluded across one is about two
    different boards. :attr:`board_replaced` is the check.

    Exclude any transition whose :attr:`board_replaced` is true from anything
    that compares ``before`` to ``after``.
    """

    wasted: bool = False
    """The action returned no frame and changed nothing, but was still counted."""
    # A non-RESET action sent while the game is in GAME_OVER is rejected by
    # the server with `400 GAME_NOT_STARTED_ERROR` ("game <id> is available
    # but has not been started, send RESET to begin playing"): no frame comes
    # back, no ledger row is written, and nothing is billed. An explicit RESET
    # is required after a death, which is why the client refuses such an
    # action rather than forwarding it.

    @classmethod
    def of(cls, *, before=None, after=None, intermediate=None,
           **rest: Any) -> "Transition":
        """Build one from grids already in hand.

        :func:`chain` is how a transition comes off a ledger; this is for the
        callers that hold the boards themselves -- a test fixture, a forward
        model, a one-off comparison. ``before``, ``after`` and ``intermediate``
        mean exactly what the properties of the same name return.
        """
        grids = (list(intermediate) if intermediate
                 else ([] if after is None else [after]))
        previous = None if before is None else _Boards([before], None)
        return cls(boards=_Boards(grids, previous), **rest)

    @property
    def before(self) -> "np.ndarray | None":
        """``None`` for the first transition, which has no predecessor."""
        return self.boards.before

    @property
    def after(self) -> "np.ndarray":
        return self.boards.after

    @property
    def intermediate(self) -> tuple:
        """Every grid this action rendered, ``after`` included.

        One action can render several frames; all of them are kept.
        """
        return self.boards.intermediate

    @property
    def board_replaced(self) -> bool:
        """``before`` and ``after`` are different boards, for any reason.

        Two things cause it: completing a level (:attr:`crosses_level`) and a
        full reset, which rewinds the whole game to level 0. Both make a spatial
        comparison meaningless, and ``crosses_level`` alone only catches the
        first -- a full reset moves the level *down*, so it slips past a test
        that looks for an increase.

        **This is the check a spatial rule wants**, not ``crosses_level``.
        """
        return self.crosses_level or self.full_reset

    @property
    def score_delta(self) -> int:
        return self.score_after - self.score_before

    @property
    def changed(self) -> bool:
        """Whether the board changed."""
        if self.before is None:
            return True
        if self.before.shape != self.after.shape:
            return True
        return not np.array_equal(self.before, self.after)

    @property
    def is_game_over(self) -> bool:
        return self.state == "GAME_OVER"

    @property
    def is_win(self) -> bool:
        return self.state == "WIN"


FROZEN_BOARDS_ENV = "CCARC3_FROZEN_BOARDS"


def _boards_are_frozen() -> bool:
    """Whether a board read from the ledger refuses to be written into.

    On by default: the ledger is evidence, and a cached array handed to
    several callers would let one caller's in-place write change what every
    later caller reads. Set ``CCARC3_FROZEN_BOARDS=0`` only to resume a
    conversation whose solver code was written against writeable boards, where
    freezing them mid-run could raise inside code the harness has not
    executed. Reading a ledger incrementally is exactly as fast either way.
    """
    return os.environ.get(FROZEN_BOARDS_ENV, "1") != "0"


def _grids(frame_field: Sequence[Any]) -> list[np.ndarray]:
    frozen = _boards_are_frozen()
    out: list[np.ndarray] = []
    for g in frame_field:
        arr = as_grid(g)
        # **Frozen because the ledger is evidence, and it is cached.** A
        # reader that keeps what it decoded hands the same array to every
        # later caller, so an in-place write would rewrite the recorded past
        # for the rest of the process -- silently, and only where it mattered.
        # Read-only turns that into an immediate error on the line that did
        # it. `as_grid` builds a new array from the record's lists here, so
        # nothing a caller owns is frozen; the flag is not set inside
        # `as_grid` for exactly that reason, since `np.asarray` returns the
        # caller's own array unchanged when it already has the right dtype.
        arr.flags.writeable = not frozen
        out.append(arr)
    return out


def _score_of(frame: dict[str, Any]) -> int:
    """The frame's level count, under either of the two names it goes by."""
    for key in ("score", "levels_completed"):
        value = frame.get(key)
        if value is not None:
            return int(value)
    return 0


class TraceWriter:
    """Appends one record per action to a JSONL ledger.

    Level tracking is the caller's business. :func:`infer_levels` offers a
    score-based heuristic, but whether levels really are delimited by a score
    increment is an open question, so nothing
    here assumes it silently.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._index = 0

    def append(self, frame: dict[str, Any], *, level: int = 0) -> dict[str, Any]:
        """Record one frame dict as a transition. Returns the record written."""
        record = row(frame, level=level, i=self._index)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._index += 1
        return record


def row(frame: dict[str, Any], *, level: int = 0, i: int = 0) -> dict[str, Any]:
    """One trace record, built from one frame.

    :meth:`TraceWriter.append` writes exactly this, so a replay can rebuild
    the same record from a live frame and compare it against a recorded one. A
    replay that built its own comparison shape would drift from the writer the
    moment either changed, and the symptom would be a divergence that is not
    one.
    """
    action_input = frame.get("action_input") or {}
    record = {
        "i": i,
        "level": level,
        "action": action_name(action_input.get("id", 0)),
        "params": {
            k: v for k, v in (action_input.get("data") or {}).items()
            if k in ("x", "y")
        },
        "hypothesis": action_input.get("reasoning"),
        "frames": [
            np.asarray(g, dtype=np.int16).tolist() for g in frame.get("frame", [])
        ],
        # arcengine 0.9.3 renamed `score` to `levels_completed`; arc_agi_3
        # 0.0.1 still sends `score`. Reading only one of them silently
        # records zero for every frame the other produced.
        #
        # Keyed on the *value*, not on the key. A `frame.get("score", ...)`
        # fallback is only reached when the key is absent, so a frame
        # carrying `score: None` beside a real `levels_completed` would take
        # the None. A zeroed score is not a small error: `infer_levels` reads
        # score increments as level boundaries, so every level in the trace
        # collapses into level 0.
        "score": int(_score_of(frame)),
        "state": str(frame.get("state", "NOT_PLAYED")),
        "full_reset": bool(frame.get("full_reset", False)),
        "available_actions": [
            action_name(a) for a in frame.get("available_actions", [])
        ],
    }
    return record

@dataclass
class Chain:
    """What one record needs from the record before it.

    ``before`` chains from the previous record's ``after``, and the score and
    level a transition is measured against are the previous record's. Holding
    the three together is what lets a reader add one record to a ledger it has
    already read without reading that ledger again.
    """

    boards: "_Boards | None" = None
    """The previous record's boards, undecoded.

    Holding the holder rather than the grid is what keeps a chained read lazy:
    ``before`` is the predecessor's ``after``, and neither is decoded until a
    caller asks. Internal chaining tests ``boards is None`` rather than
    ``previous is None`` for exactly that reason -- the second would decode a
    board to answer a question about whether one exists.
    """

    score: int = 0
    level: int = 0

    @property
    def previous(self) -> "np.ndarray | None":
        """The previous record's board, decoded.

        Decoding it is the point of asking, so this is not the accessor the
        chaining path uses.
        """
        return None if self.boards is None else self.boards.after


def _record(raw: bytes) -> tuple[dict[str, Any], Any, bool]:
    """One record split into ``(metadata, frames-source, empty)``.

    The frames source is the raw bytes of the ``frames`` array, left undecoded,
    or an already-decoded list when the record was not the shape :func:`row`
    writes and had to be parsed whole.
    """
    head = raw.find(_FRAMES_HEAD)
    tail = raw.rfind(_FRAMES_TAIL)
    if head >= 0 and tail > head:
        start = head + len(_FRAMES_HEAD) - 1
        chunk = raw[start:tail + 1]
        try:
            meta = json.loads(raw[:head] + raw[tail + 2:])
        except json.JSONDecodeError:
            meta = None
        if isinstance(meta, dict) and set(meta) == _META_KEYS:
            return meta, chunk, chunk == b"[]"
    full = json.loads(raw)
    frames = full.get("frames") or []
    return full, frames, not frames


def _chain(meta: dict[str, Any], frames: Any, empty: bool,
           state: Chain) -> Transition | None:
    """The one place a :class:`Transition` is built."""
    if empty and state.boards is None:
        # A record with no frame: the action rendered nothing and carries
        # its predecessor's board forward. With no predecessor there is
        # nothing to represent, so it is skipped only in that case.
        return None
    boards = _Boards(frames, state.boards)
    level = int(meta.get("level", 0))
    score = int(meta.get("score", 0))
    transition = Transition(
        index=int(meta["i"]),
        level=level,
        action=meta["action"],
        params=dict(meta.get("params") or {}),
        boards=boards,
        wasted=empty,
        # The first transition has no predecessor, so it cannot have crossed
        # anything -- comparing it against a level-0 default would flag
        # any trace that starts mid-game.
        crosses_level=(state.boards is not None and level > state.level),
        score_before=state.score,
        score_after=score,
        state=str(meta.get("state", "NOT_PLAYED")),
        # A level that goes *down* is a full reset whatever the server
        # said: the API can report ``full_reset: False`` on a transition
        # that returns the game to level 0, and ``crosses_level`` only
        # catches an increase -- so without this ``board_replaced``, the
        # check a spatial rule wants, reports False on a whole-board
        # replacement.
        full_reset=bool(meta.get("full_reset", False)) or (
            state.boards is not None and level < state.level
        ),
        available_actions=tuple(meta.get("available_actions") or ()),
        hypothesis=meta.get("hypothesis"),
    )
    state.boards = boards
    state.score = score
    state.level = level
    return transition


def chain(rec: dict[str, Any], state: Chain) -> Transition | None:
    """Build the transition ``rec`` describes and advance ``state`` past it.

    Returns ``None`` when the record is a wasted action with no predecessor:
    there is no board to represent, so nothing is built and ``state`` does not
    move.
    """
    frames = rec.get("frames") or []
    return _chain(rec, frames, not frames, state)


def chain_raw(raw: bytes, state: Chain) -> Transition | None:
    """:func:`chain`, from an undecoded line, leaving the boards undecoded."""
    meta, frames, empty = _record(raw)
    return _chain(meta, frames, empty, state)


def load(path: str | Path) -> list[Transition]:
    """Read a ledger back as transitions, chaining ``before`` from ``after``.

    Reads every record, and decodes a board only when something asks for one.
    A caller that filters on ``level`` or ``action`` never pays for the boards
    it discards. :class:`TraceReader` answers the same question without
    re-reading what it has already read, and is what to use when the question
    is asked more than once.
    """
    out: list[Transition] = []
    state = Chain()
    p = Path(path)
    if not p.exists():
        return out
    with p.open("rb") as fh:
        blob = fh.read()
    lines = blob.split(b"\n")
    # The split leaves an empty piece after a final newline. It is not a line,
    # and counting it would make the corruption message name a line number the
    # file does not have.
    total = len(lines) - 1 if blob.endswith(b"\n") else len(lines)
    for number, raw in enumerate(lines, start=1):
        # No `strip()`: it copies every line to remove a newline the split
        # has already taken, and the JSON decoder ignores whitespace anyway.
        if not raw or raw.isspace():
            continue
        try:
            transition = chain_raw(raw, state)
        except json.JSONDecodeError:
            if number == len(lines) or not any(l.strip() for l in lines[number:]):
                break        # a writer is mid-append; everything before is good
            raise ValueError(
                f"{p}: line {number} of {total} is not valid JSON. A bad "
                f"line in the middle of a trace is corruption, not a partial "
                f"write."
            ) from None
        if transition is not None:
            out.append(transition)
    return out


class TraceReader:
    """A ledger read once, and after that only where it grew.

    :func:`load` decodes the whole file on every call, so asking it for the
    newest transition after every action costs more each time and makes a run
    quadratic in its own length. This keeps what it has decoded and reads only
    the bytes appended since, so the same question costs the same at any
    length.

    Safe against the file being replaced or rewritten underneath it: the reader
    checks the file's identity and the index of the first new record, and
    starts again from the beginning if either says that what it holds no longer
    describes what is on disk. A record still being written has no terminating
    newline, so it is left for the next call rather than half-read.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._forget()

    def _forget(self) -> None:
        self._offset = 0
        self._identity: tuple[int, int] | None = None
        self._seen: bytes = b""
        """A digest of the last record consumed, and how long it was.

        This is what detects a ledger rewritten in place: same file, same inode,
        no shrink, and long enough that the offset still lands on a record
        boundary. Comparing the bytes just before the offset catches it for the
        cost of one short read. It does NOT catch a rewrite that leaves the last
        consumed record byte-identical and changes only earlier ones -- nothing
        short of reading the file again can, and the case that matters here (a
        fresh run reusing the path) unlinks first and so changes the inode.
        """
        self._seen_len = 0
        self._state = Chain()
        self._transitions: list[Transition] = []

    @property
    def history(self) -> list[Transition]:
        """Everything decoded so far. Call :meth:`refresh` first."""
        return self._transitions

    @property
    def last(self) -> Transition | None:
        """The newest decoded transition, or ``None``. Call :meth:`refresh` first."""
        return self._transitions[-1] if self._transitions else None

    def refresh(self) -> list[Transition]:
        """Decode whatever has been appended since the last call."""
        try:
            st = self.path.stat()
        except FileNotFoundError:
            self._forget()
            return self._transitions
        identity = (st.st_dev, st.st_ino)
        if identity != self._identity or st.st_size < self._offset:
            # A different file, or this one truncated: what was decoded
            # describes a ledger that is no longer there.
            self._forget()
            self._identity = identity
        if self._offset and self._rewritten():
            # Same file, no shrink, and yet what was already read is not what is
            # there now. Nothing decoded can be trusted.
            self._forget()
            self._identity = identity
            return self.refresh()
        if st.st_size == self._offset:
            return self._transitions

        with self.path.open("rb") as fh:
            fh.seek(self._offset)
            blob = fh.read()
        # Split once and drop the unterminated remainder, rather than slicing
        # the complete part out first: on a long trace that slice is a second
        # copy of the whole file.
        lines = blob.split(b"\n")
        if blob.endswith(b"\n"):
            lines.pop()                     # the empty piece after the last newline
            consumed = len(blob)
        else:
            consumed = len(blob) - len(lines.pop())
        if not lines:
            # A writer is mid-append and no complete record has landed yet.
            return self._transitions

        for raw in lines:
            # No `strip()`: it copies every line to remove a newline the split
            # has already taken, and the JSON decoder ignores whitespace around
            # a value anyway.
            if not raw or raw.isspace():
                continue
            try:
                transition = chain_raw(raw, self._state)
            except json.JSONDecodeError:
                raise ValueError(
                    f"{self.path}: a complete line is not valid JSON. A bad "
                    f"line in the middle of a trace is corruption, not a "
                    f"partial write."
                ) from None
            if transition is not None:
                self._transitions.append(transition)
        self._offset += consumed
        self._remember(lines[-1] + b"\n")
        return self._transitions

    def _rewritten(self) -> bool:
        """Have the bytes this reader already consumed changed underneath it?"""
        if not self._seen_len:
            return False
        with self.path.open("rb") as fh:
            fh.seek(self._offset - self._seen_len)
            return hashlib.sha256(fh.read(self._seen_len)).digest() != self._seen

    def _remember(self, tail: bytes) -> None:
        """Note the last record consumed, so a rewrite of it can be detected."""
        self._seen = hashlib.sha256(tail).digest()
        self._seen_len = len(tail)


def tail_state(path: str | Path, *, window: int = 1 << 20) -> tuple[Transition | None, Chain]:
    """The newest transition, and the chain state for whatever is appended next.

    Reads the end of the ledger instead of all of it, so the newest transition
    costs the same on a long game as on a short one. The result is the same
    object ``load(path)[-1]`` would give, including ``before``, which is read
    from the records ahead of it.
    """
    p = Path(path)
    try:
        size = p.stat().st_size
    except FileNotFoundError:
        return None, Chain()
    if size == 0:
        return None, Chain()

    while True:
        start = max(0, size - window)
        with p.open("rb") as fh:
            fh.seek(start)
            blob = fh.read()
        lines = [raw for raw in blob.split(b"\n") if raw.strip()]
        if start > 0 and lines:
            # The window opened mid-record; that fragment is not a record.
            lines = lines[1:]
        if not lines:
            if start == 0:
                return None, Chain()
            window *= 4
            continue

        records = [_record(raw) for raw in lines]
        state = Chain()
        last: Transition | None = None
        for meta, frames, empty in records[:-1]:
            transition = _chain(meta, frames, empty, state)
            if transition is not None:
                last = transition
        # Whether a predecessor was found inside the window. Without one the
        # final transition would carry ``before=None`` and a wrong
        # ``crosses_level``, which is a quiet lie rather than a slow answer.
        seeded = state.boards is not None
        transition = _chain(*records[-1], state)
        if transition is not None:
            last = transition
        if seeded or start == 0:
            return last, state
        window *= 4



# The boards are nearly all of a trace's bytes and nearly all of the time it
# costs to read one, and most questions asked of a ledger never look at them:
# which level an action was taken from, where a play started, what the score
# was. `row` writes every record with the same key order, so the boards can be
# cut out of a line with two byte searches and the rest handed to the JSON
# decoder intact -- orders of magnitude cheaper than decoding every line of a
# long trace.
#
# The markers are exact, not heuristic. `"score"` cannot occur inside the frames
# -- they are nested integer arrays -- and `hypothesis`, the one field carrying
# solver-supplied text, is written BEFORE the frames, so the LAST `],"score":`
# in a line is always the end of the frames. Anything that does not match the
# shape falls back to decoding the whole line, so a record written by some other
# writer is read correctly and merely slowly.
_FRAMES_HEAD = b'"frames":['
_FRAMES_TAIL = b'],"score":'
_META_KEYS = frozenset(
    ("i", "level", "action", "params", "hypothesis", "score", "state",
     "full_reset", "available_actions")
)


def _meta(raw: bytes) -> tuple[dict[str, Any], bool]:
    """One record without its boards, and whether it had none.

    Returns ``(metadata, empty)``. Falls back to decoding the whole line when
    the record is not the shape :func:`row` writes.
    """
    head = raw.find(_FRAMES_HEAD)
    tail = raw.rfind(_FRAMES_TAIL)
    if head >= 0 and tail > head:
        start = head + len(_FRAMES_HEAD) - 1
        empty = raw[start:tail + 1] == b"[]"
        try:
            meta = json.loads(raw[:head] + raw[tail + 2:])
        except json.JSONDecodeError:
            meta = None
        if isinstance(meta, dict) and set(meta) == _META_KEYS:
            return meta, empty
    full = json.loads(raw)
    return full, not full.get("frames")


@dataclass(frozen=True)
class LedgerRow:
    """One recorded action, without the boards it produced.

    Everything :class:`Transition` carries except ``before``, ``after`` and
    ``intermediate``. Reading a trace this way answers any question about what
    was done -- levels, plays, scores, resets -- without decoding a single
    board.
    """

    index: int
    level: int
    action: str
    params: dict[str, Any]
    score_before: int
    score_after: int
    state: str
    full_reset: bool
    available_actions: tuple[str, ...]
    hypothesis: Any = None
    crosses_level: bool = False
    wasted: bool = False

    @property
    def board_replaced(self) -> bool:
        """``before`` and ``after`` would be different boards, for any reason."""
        return self.crosses_level or self.full_reset

    @property
    def score_delta(self) -> int:
        return self.score_after - self.score_before

    @property
    def is_game_over(self) -> bool:
        return self.state == "GAME_OVER"

    @property
    def is_win(self) -> bool:
        return self.state == "WIN"


def rows(path: str | Path) -> list[LedgerRow]:
    """Read a ledger as :class:`LedgerRow` -- every field except the boards.

    The same records :func:`load` returns and in the same order, so anything
    that asks a ledger only about levels, plays, actions or scores can read it
    this way instead. Use :func:`load` when the boards are the question.
    """
    out: list[LedgerRow] = []
    seen = False
    previous_score = 0
    previous_level = 0
    p = Path(path)
    if not p.exists():
        return out
    with p.open("rb") as fh:
        blob = fh.read()
    lines = blob.split(b"\n")
    for number, raw in enumerate(lines, start=1):
        # No `strip()`: it copies every line to remove a newline the split
        # has already taken, and the JSON decoder ignores surrounding
        # whitespace anyway.
        if not raw or raw.isspace():
            continue
        try:
            meta, empty = _meta(raw)
        except json.JSONDecodeError:
            if number == len(lines) or not any(l.strip() for l in lines[number:]):
                break        # a writer is mid-append; everything before is good
            raise ValueError(
                f"{p}: line {number} of {len(lines)} is not valid JSON. A bad "
                f"line in the middle of a trace is corruption, not a partial "
                f"write."
            ) from None
        if empty and not seen:
            # Mirrors `load`: a wasted action with no predecessor represents
            # nothing and is not a row of the ledger's transition view.
            continue
        level = int(meta.get("level", 0))
        score = int(meta.get("score", 0))
        out.append(
            LedgerRow(
                index=int(meta["i"]),
                level=level,
                action=meta["action"],
                params=dict(meta.get("params") or {}),
                score_before=previous_score,
                score_after=score,
                state=str(meta.get("state", "NOT_PLAYED")),
                full_reset=bool(meta.get("full_reset", False)) or (
                    seen and level < previous_level
                ),
                available_actions=tuple(meta.get("available_actions") or ()),
                hypothesis=meta.get("hypothesis"),
                crosses_level=(seen and level > previous_level),
                wasted=empty,
            )
        )
        seen = True
        previous_score = score
        previous_level = level
    return out


def infer_levels(scores: Sequence[int]) -> list[int]:
    """Derive per-row levels from a sequence of scores. Parent-side."""
    # **Comment, not docstring: this module is staged onto the solver's own
    # PYTHONPATH.** How the environment computes its score is the
    # environment's behaviour and the agent's to discover, so the inference
    # rule is documented on this side of the line.
    #
    # The caveats, which are the parent's business: a level reset does not
    # decrement, so replays of a failed level attribute correctly while a full
    # reset does not, and a caller with an explicit level should prefer it.
    # The client records `Transition.level` on every row, so a caller almost
    # always has one -- this is a fallback for a trace that does not carry
    # it.
    level = 0
    out: list[int] = []
    previous = scores[0] if scores else 0
    for s in scores:
        if s > previous:
            level += 1
        out.append(level)
        previous = s
    return out


# ── play and level analysis ──────────────────────────────────────────────────
#
# Reading a finished trace back: which transitions belong to which play, and
# how many actions each level cost. Structure only -- nothing here divides by
# anything the harness withholds.

def plays(transitions: Sequence["object"]) -> list[list["object"]]:
    """Split a ledger into playthroughs, one list per play.

    A full reset starts a new play — a new guid, a new ``actions`` row and a new
    ``actions_by_level`` row server-side — so the boundaries are exactly the
    transitions flagged ``full_reset``, excluding the one that opens the trace.

    Note the flag is only partly the server's: :mod:`ccarc.arc_3.ledger` ORs
    ``full_reset`` with "the recorded level went down", because the API can
    report ``full_reset: False`` on a transition that returns the game to
    level 0. A new play always returns to level 0, so the level drop is the
    reliable signal.
    """
    kept = list(transitions)
    starts = [0] + [i for i, t in enumerate(kept) if t.full_reset and i > 0]
    return [kept[a:b] for a, b in zip(starts, starts[1:] + [len(kept)])]


def actions_per_level(
    transitions: Sequence["object"],
    n_levels: int,
    *,
    cumulative: bool = False,
) -> list[int | None]:
    """Actions used to complete each level, from a ledger. ``None`` if not completed.

    **Attribution is by the level the action was taken *from*, not the level it
    landed in**, and that distinction is worth a paragraph because getting it
    backwards silently changes the score.

    The trace records ``level`` as ``levels_completed`` *after* the action. So
    the action that finishes level 3 is recorded at level 4. Grouping by the
    recorded level therefore credits every level's final, decisive action to the
    next one, shortening each level by one and inventing a phantom level at the
    end.

    (The play-starting RESET is excluded from the count -- see the block
    further down -- so a hand tally that includes it sits one action high on
    the first level.)

    **Replayed levels are NOT summed.** The API records, per game:

    ===================  ==================================================
    ``total_plays``      how many plays this scorecard holds
    ``actions``          actions **per play**, e.g. ``[3, 1]``
    ``actions_by_level`` **cumulative** actions at each level's completion,
                         per play -- see the warning below
    ``total_actions``    the sum across plays — never a scoring denominator
    ===================  ==================================================

    A ``RESET`` while the action counter is non-zero is a *level* reset:
    ``total_plays`` stays put and ``resets`` increments. A ``RESET`` when the
    counter is zero — which is the state immediately after a level advance —
    **starts a new play**: a new guid, a new ``actions`` row and a new
    ``actions_by_level`` row.

    **The field is a list of PLAYS, each a list of ``[level, cumulative]``
    pairs.** Omitting the outer list, or dropping the level index, would raise
    or mis-parse in :func:`server_actions_per_level`, which is the function
    that reads it. A two-play card of a three-level game (synthetic figures)::

        [[[1, 10], [2, 30], [3, 45]],     # play 1
         [[1, 6], [2, 20], [3, 31]]]      # play 2

    **And it is cumulative, despite its name.** In play 1 above the entries sum
    to 85 for a play that spent 45 actions, which is the tell -- they are
    ``[level, actions spent by the time that level fell]``, so the per-level
    ``a_l`` the rubric divides by is the **difference** between consecutive
    entries: 10, 20, 15. Reading them as per-level counts inflates every level
    after the first and scores such a run at a fraction of what it earns.

    A one-level game cannot show this -- cumulative and per-level coincide --
    so a probe of one is no evidence either way.

    So for a one-level game won in 10 actions and then replayed and won in
    7: ``actions_by_level`` is ``[[[1, 10]], [[1, 7]]]`` -- two plays, each
    one ``[level, cumulative]`` pair -- and ``total_actions`` is 17. Since
    ``Card.high_score = max(scores)`` scores the *best* play, ``a_l`` is
    **7**. The 17 is the running total, not the denominator.

    Summing is tempting because ``total_actions`` accumulates; that argument
    conflates a game-level total with a per-level denominator, and the server
    keeps both, separately.

    ``cumulative=True`` still gives the summed reading for comparison. On a
    replayed run the two diverge; on a single-play run they agree.

    Actions spent on abandoned attempts and on failed level retries are included
    either way. They were spent, and "actions used to complete this level" is
    what the rubric divides by.
    """
    from .ledger import Transition  # noqa: F401 -- documents the expected element type

    kept = list(transitions)
    if not cumulative:
        cut = max((i for i, t in enumerate(kept) if t.full_reset), default=0)
        kept = kept[cut:]

    counts: dict[int, int] = {}
    previous: int | None = None
    for i, t in enumerate(kept):
        source = previous if previous is not None else 0
        # **A RESET that starts a play is not an action, and the server
        # agrees.** Its own counts exclude the opening RESET: a play of seven
        # ACTION6 calls after it is reported as `actions: [7]` and
        # `actions_by_level: [[[1, 7]]]`, not 8. A *level* reset -- a RESET
        # while the counter is non-zero, taken after a death -- is counted.
        #
        # So the exclusion is precisely the play-starting RESET: the first
        # transition of the trace, and any RESET that performs a full reset.
        if t.action == "RESET" and (i == 0 or t.full_reset):
            previous = t.level
            continue
        counts[source] = counts.get(source, 0) + 1
        previous = t.level

    completed = max((t.level for t in kept), default=0)
    return [counts.get(i, 0) if i < completed else None for i in range(n_levels)]
