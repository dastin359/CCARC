"""Grid representation primitives for ARC-AGI-3.

ARC-AGI-3 hands back 64x64 integer grids over a sixteen-colour palette. Rendered
the way the SDK does it -- ``[5, 5, 5, ...]`` rows, one per line -- a single
grid is large, and one action can produce several. These helpers exist so no
solver has to re-derive its own encoding, and so every run's traces stay
comparable with every other run's.

Nothing here imports ``arc_agi_3``. Grids are plain nested lists or arrays, so
this module is usable (and testable) without the SDK or an API key.

**One grid per action is the wrong assumption.** ``FrameData.frame`` is a *list*
of grids -- the engine renders every frame until the action completes -- so code
that takes ``frame[0]`` or ``frame[-1]`` silently drops the intermediate
states. ``as_grid`` handles one; map it over the sequence rather than picking
from it.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable, Sequence

# Deferred: see `._numpy`. A call that only reads metadata never builds an
# array, so numpy is imported the first time a grid is decoded.
from ._numpy import lazy_numpy

np = lazy_numpy(globals())

PALETTE_SIZE = 16
"""ARC-AGI-3 uses sixteen colours, not ARC-AGI-2's ten."""

DEFAULT_BACKGROUND = 5
"""The SDK renders 5 as black and treats it as the default backdrop."""

_CHARS = "0123456789abcdef"

PALETTE = (
    "#FFFFFF", "#CCCCCC", "#999999", "#666666", "#333333", "#000000",
    "#E53AA3", "#FF7BCC", "#F93C31", "#1E93FF", "#88D8F1", "#FFDC00",
    "#FF851B", "#921231", "#4FCC30", "#A356D6",
)
"""The official sixteen colours, read out of the SDK's own renderer.

0-5 is a white-to-black greyscale ramp, which is why 5 (black) reads as the
default backdrop; 6-15 are the chromatic colours.
"""

__all__ = [
    "PALETTE_SIZE",
    "DEFAULT_BACKGROUND",
    "as_grid",
    "render",
    "block_size",
    "logical",
    "collapse",
    "cell_boundaries",
    "png",
    "PALETTE",
    "counts",
]


def as_grid(grid: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Coerce a nested sequence to a 2-D ``int16`` array, validating the palette."""
    arr = np.asarray(grid, dtype=np.int16)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D grid, got shape {arr.shape}")
    if arr.size and (arr.min() < 0 or arr.max() >= PALETTE_SIZE):
        raise ValueError(
            f"grid values must lie in [0, {PALETTE_SIZE}); "
            f"got [{arr.min()}, {arr.max()}]"
        )
    return arr


def render(grid: Sequence[Sequence[int]] | np.ndarray) -> str:
    """Render a grid as one character per cell, ``0-9a-f``.

    A 64x64 grid becomes 64 lines of 64 characters: 4,159 characters against
    12,416 for the bracketed-integer rows, a 3x reduction. The margin against
    the SDK's ``pretty_print_3d`` is larger, since that adds a header and a
    two-space indent per grid -- and one action can return several grids.
    """
    arr = as_grid(grid)
    return "\n".join("".join(_CHARS[v] for v in row) for row in arr)


def block_size(grid: Sequence[Sequence[int]] | np.ndarray) -> int:
    """Largest ``k`` for which the grid is exactly a ``k x k``-block image.

    Returns 1 when no larger factor makes every block uniform, so the result is
    always safe to divide by. Exact: it never claims a ``k`` that does not hold.
    """
    arr = as_grid(grid)
    h, w = arr.shape
    if h == 0 or w == 0:
        return 1
    for k in range(min(h, w), 1, -1):
        if h % k or w % k:
            continue
        blocks = arr.reshape(h // k, k, w // k, k)
        # Every block is uniform iff its min and max agree everywhere.
        if np.array_equal(blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))):
            return k
    return 1


def logical(grid: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Collapse a block-rendered grid to its logical board, losslessly.

    Divides only by a factor :func:`block_size` has proved, and is idempotent
    on grids that are already logical.

    Always returns a fresh array, on BOTH branches, so writing to the result
    never reaches back into the frame it came from.
    """
    arr = as_grid(grid)
    k = block_size(arr)
    return arr.copy() if k == 1 else arr[::k, ::k].copy()


def collapse(grid: Sequence[Sequence[int]] | np.ndarray) -> np.ndarray:
    """Collapse runs of identical adjacent rows and columns.

    **This preserves structure and destroys position.** It is not a downscale
    and the result is not the logical board: its coordinates do not index the
    frame. For the logical board with its metric intact, recover the cell grid
    across a whole trace with :func:`cell_boundaries` and index into the raw
    frame.
    """
    arr = as_grid(grid)
    if arr.size == 0:
        return arr.copy()
    keep_rows = np.ones(arr.shape[0], dtype=bool)
    keep_rows[1:] = (arr[1:] != arr[:-1]).any(axis=1)
    arr = arr[keep_rows]
    keep_cols = np.ones(arr.shape[1], dtype=bool)
    keep_cols[1:] = (arr[:, 1:] != arr[:, :-1]).any(axis=0)
    return arr[:, keep_cols].copy()


def png(
    grid: Sequence[Sequence[int]] | np.ndarray,
    path: str | "Path",
    *,
    scale: int = 8,
    grid_lines: bool = False,
) -> str:
    """Write a grid as a PNG in the game's palette and return the path.

    You can open the file with the Read tool and view it directly.

    ``scale`` is pixels per cell. ``grid_lines`` overlays faint cell borders.
    """
    from pathlib import Path as _Path

    from PIL import Image, ImageDraw

    arr = as_grid(grid)
    h, w = arr.shape
    img = Image.new("RGB", (w * scale, h * scale), PALETTE[5])
    draw = ImageDraw.Draw(img)
    for y in range(h):
        for x in range(w):
            draw.rectangle(
                [x * scale, y * scale, (x + 1) * scale - 1, (y + 1) * scale - 1],
                fill=PALETTE[int(arr[y, x])],
            )
    if grid_lines and scale >= 4:
        for x in range(w + 1):
            draw.line([(x * scale, 0), (x * scale, h * scale)], fill="#444444")
        for y in range(h + 1):
            draw.line([(0, y * scale), (w * scale, y * scale)], fill="#444444")

    out = _Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return str(out)


def cell_boundaries(
    grids: Iterable[Sequence[Sequence[int]] | np.ndarray],
) -> tuple[list[int], list[int]]:
    """Recover the render's cell grid by pooling boundaries across many frames.

    Returns ``(row_starts, col_starts)`` into the raw frame.

    Boundaries only appear where the raw pixels change, so any one frame
    reveals only its own. Frames whose contents differ reveal different
    boundaries, and pooling many of them converges on the grid: the extra
    frames are the mechanism, which is why this takes an iterable and will
    usually under-report on one frame.

    The result is always a subset of the true cell grid -- it under-reports
    rather than inventing splits. Cross-check ``len(row_starts)`` against a
    board size you have independent reason to believe before trusting it as
    complete.

    **Every grid must have the same shape, and a mismatch is refused.** Pooling
    is defined over one board; frames of two different shapes pool two cell
    grids into one answer and report it as indices into whichever came last.
    Filter first::

        arc.cell_boundaries([t.after for t in ts if t.level == client.level])
    """
    if isinstance(grids, np.ndarray) and grids.ndim == 2:
        raise ValueError(
            "cell_boundaries takes an iterable of grids, not one grid -- passing "
            "a single 2-D array iterates its rows and fails on the first. For one "
            "frame, wrap it: cell_boundaries([grid]). But pooling across a trace "
            "is the whole mechanism; one frame will under-report badly."
        )
    rows: set[int] = {0}
    cols: set[int] = {0}
    height = width = 0
    for g in grids:
        arr = as_grid(g)
        if arr.size == 0:
            continue
        if height and arr.shape != (height, width):
            raise ValueError(
                f"cell_boundaries pools one board: got {(height, width)} then "
                f"{arr.shape}. Boundaries pooled across two shapes index "
                "neither. Pass frames of one shape: "
                "[t.after for t in ts if t.level == lvl]."
            )
        height, width = arr.shape
        changed_rows = (arr[1:] != arr[:-1]).any(axis=1)
        rows.update(int(i) + 1 for i in np.nonzero(changed_rows)[0])
        changed_cols = (arr[:, 1:] != arr[:, :-1]).any(axis=0)
        cols.update(int(i) + 1 for i in np.nonzero(changed_cols)[0])
    if not height:
        return [], []
    return sorted(r for r in rows if r < height), sorted(c for c in cols if c < width)


def counts(grid: Sequence[Sequence[int]] | np.ndarray) -> dict[int, int]:
    """Cell count per colour, for the colours actually present."""
    arr = as_grid(grid)
    return dict(Counter(arr.ravel().tolist()))


