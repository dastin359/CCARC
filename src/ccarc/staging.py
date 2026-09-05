"""Stage the package into a run's own tree, so a solver imports a copy.

Line numbers are preserved, so a traceback raised inside the guardrail still points
at the line the repository has.
"""

# This module is part of what it stages, so its own docstring reaches the
# staged copy. The rationale below is therefore a comment, which staging
# strips.
#
# **Why a copy, when everything else is already copied.** The prompts, the
# toolkit and the gate are written into the workspace and stripped there.
# Handing the package over live -- the repository `src` on the child's
# PYTHONPATH -- would let the child read the repository itself: the strip and
# the redactor rewrite a copy, so the unstripped originals would sit one
# `read_text()` away. Staging closes that by construction rather than by
# vigilance.
#
# **What is removed.** Every `.md` under the package: the doctrines, the
# prompts, the workspace template. No child-side code reads any of them. Every
# `#` comment: a docstring is API surface, but a comment is written for whoever
# edits the file and is reachable only by `getsource` or `cat`. That gap is the
# exposure -- on `ArcClient`, `getsource` returns several times what a rendered
# page does. And every path in `_EXCLUDED`, which is where a module the child
# never imports is kept off its disk entirely rather than being sanitised in
# place.
#
# **What is rewritten.** Environment names, in whatever survives, docstrings
# included.
#
# **What staging does NOT reach.** The figures beside the names. Those are
# withheld at the source, and a guard over a real stage is what keeps them
# out. Staging is the floor, not the ceiling.

from __future__ import annotations

import re
import shutil
from pathlib import Path

#: The package root in this checkout — the thing being copied.
_PACKAGE = Path(__file__).resolve().parent

#: Directory name inside a run's tree. Leading underscore so it sorts away from
#: the solver's own files and reads as machinery rather than material.
STAGE_DIR = "_pkg"

#: The ARC environment-id shape, in every form it is written in.
#:
#: **Match bare mentions, not only backticked ones.** A pattern that requires
#: backticks misses every name written bare in a comment or a prose docstring,
#: where backticks are not the house style. And the corpus is not all
#: two-letters-two-digits: it also contains letter-digit-digit-letter and
#: letter-digit-letter-digit.
#:
#: So: the three real shapes, optionally carrying the eight-hex suffix that
#: makes a full id, anchored with a lookbehind that forbids a preceding word
#: character or hyphen. The lookbehind is load-bearing -- without it this
#: rewrites the id-shaped tail of a hyphenated identifier (``-k2p5``) and the
#: staged copy stops resolving that name.
#:
#: Still the SHAPE and not a list of real ids: writing the ids down here would
#: put the thing being hidden into the file that hides it. The guard that
#: checks this scans a real stage.
_GAME_ID = re.compile(
    r"(?<![\w-])"
    r"(?:[a-z]{2}\d{2}|[a-z]\d{2}[a-z]|[a-z]\d[a-z]\d)"
    r"(?:-[0-9a-f]{6,12})?"
    r"(?!\w)"
)


#: An encoding declaration is a comment that the *parser* reads, so it is the
#: one comment that cannot be removed. Only lines 1 and 2 count, per PEP 263.
_CODING = re.compile(r"coding[:=]\s*([-\w.]+)")


#: The post-condition's net, wider than the redactor's on purpose -- see the
#: assertion at the end of `stage_package`.
_ID_NET = re.compile(
    r"(?<![\w-])(?=[a-z][a-z0-9]{3}(?!\w))(?=[a-z0-9]*\d)[a-z][a-z0-9]{3}"
)

#: Tokens of that shape this package legitimately contains.
_ID_ALLOWED = frozenset({"utf8", "int8", "sha1", "md5s", "0x1f", "k2p5"})


def _strip_comments(text: str) -> tuple[str, int]:
    """Remove every ``#`` comment, byte-for-byte elsewhere. Returns (text, n).

    Docstrings stay: they are the API surface a caller reads. A comment is
    written for whoever edits the file, never renders on a ``pydoc`` page, and
    is reachable only through ``getsource`` or ``cat`` -- which is where a name
    or a rationale slips past a docstring-only audit.

    Tokenised, not matched: a ``#`` inside a string literal is not a comment,
    and a regex cannot tell the difference. Each comment is cut at its own
    start column and the rest of the line is left untouched, so **line numbers
    are preserved exactly** -- a traceback inside the guardrail still points at the
    line the repository has.

    A file that will not tokenise is left alone rather than mangled; it is
    counted as zero and the caller's own post-conditions still apply.
    """
    try:
        import io
        import tokenize
        cuts: dict[int, int] = {}
        reader = io.StringIO(text).readline
        for tok in tokenize.generate_tokens(reader):
            if tok.type != tokenize.COMMENT:
                continue
            row, col = tok.start
            if row <= 2 and _CODING.search(tok.string):
                continue
            cuts[row] = min(col, cuts.get(row, col))
    except Exception:
        return text, 0
    if not cuts:
        return text, 0
    lines = text.splitlines(keepends=True)
    for row, col in cuts.items():
        line = lines[row - 1]
        # The index where the line's own newline starts, NOT the count of
        # newline characters: reversed, this deletes the `#` and keeps the
        # comment text, which then fails to compile several lines further on.
        end = len(line.rstrip("\r\n"))
        lines[row - 1] = line[:col].rstrip() + line[end:]
    return "".join(lines), len(cuts)


#: Package-relative directories and files the solver's import graph never
#: touches, and which say things it should not read.
#:
#: * `arc_3/cli.py` -- carries the withheld human baseline for the public set
#:   and a definition of that quantity beside it.
# **Parent-only. Derived from the import closure, not from judgement.**
# The solver reaches this package by exactly two routes: `gate_shim.py` in its
# workspace imports `ccarc.arc_2.gate`, and `evaluate` spawns `_solve_worker`
# as a subprocess. Resolving every import from those two seeds -- relative ones
# included, which is the step that is easy to get wrong -- gives the reachable
# set: `arc_2/{gate,evaluate,reporting,signals,config,_solve_worker}.py` and the
# package `__init__`. Everything below is outside it.
#
# Two of these are the reason the list exists at all:
#
#   `data.py`         resolves and reads task JSON off the dataset. The mount
#                     namespace already hides the dataset and `workspace_env`
#                     already drops `ARC_DATA_ROOT`, so this is the third lock
#                     on that door -- but it is the only one that does not
#                     depend on the kernel granting a namespace.
#   `arc_2/scoring.py` is the contamination check. Handing the solver the code
#                     that decides whether it cheated is the same mistake as
#                     staging the scoring rubric beside the score.
#: Modules the shared guardrail keeps off every solver's disk, whichever driver
#: is running. `confine` is NOT here and must not be: `arc_3/session.py:50`
#: imports it at module level, so a child that imports `session` at all executes
#: that line. `network` and `egress_proxy` are imported inside `_launch`, which
#: runs harness-side, so they are absent from the stage entirely.
_SHARED_EXCLUDED = frozenset({
    "guardrail/network.py",
    "guardrail/egress_proxy.py",
})

#: Per driver, the modules that driver's own solver never imports. Each set was
#: DERIVED from an import closure, not chosen: seed with what the solver's
#: workspace actually reaches, resolve every import including the relative ones,
#: and everything outside the result is parent-only.
_EXCLUDED = {
    # An arc_2 solver reaches the package by two routes: `gate_shim.py` in its
    # workspace imports `arc_2.gate`, and `evaluate` spawns `_solve_worker`.
    #
    # `data.py` resolves and reads task JSON off the dataset, and `scoring.py`
    # is the contamination check -- handing the solver the code that decides
    # whether it cheated.
    "arc_2": frozenset({
        "arc_2/cli.py",
        "arc_2/data.py",
        "arc_2/prompt.py",
        "arc_2/runner.py",
        "arc_2/scoring.py",
        "arc_2/trace.py",
        "arc_2/workspace.py",
        "arc_2/__main__.py",
        # `workspace.py` copies the toolkit, the dry run and the gate shim into
        # the workspace from the SOURCE package, not from the stage, so the
        # solver still gets all three. Staging them again would only add an
        # unstripped second copy.
        "arc_2/assets",
        "arc_2/_vendor",
        # Outside this driver's closure. The guardrail goes whole: `confine` is
        # staged for arc_3 only, because `arc_3/session.py:50` imports it at
        # module level, and nothing an arc_2 solver imports touches it.
        # `staging` and `__main__` describe the harness rather than serve it.
        "guardrail",
        "staging.py",
        "__main__.py",
    }),
    # Parent-only for arc_3. `session.py` imports these inside `_launch` /
    # `_run_live`, which run in the harness process; the child imports `session`
    # only for `ensure_scored_ledger` and `scored_arc_client` and never reaches
    # them. `outbox` is NOT here and must not be: `client._announce_level`
    # imports it lazily and that code runs in the solver's own process.
    "arc_3": frozenset({
        "arc_3/cli.py",
        "guardrail/arc_proxy.py",
        "arc_3/shared_card.py",
        "arc_3/live.py",
    }),
}

#: Every driver this package ships. A stage carries one of them.
DRIVERS = tuple(sorted(_EXCLUDED))


def _excluded_for(driver: str) -> frozenset[str]:
    """What a stage for ``driver`` must not carry.

    The other driver's whole subtree goes: an ARC-AGI-2 solver has no use for
    the ARC-AGI-3 client, and shipping it only widens the surface. This is the
    part a single shared list could not express.
    """
    if driver not in _EXCLUDED:
        raise ValueError(
            f"unknown driver {driver!r}; expected one of {', '.join(DRIVERS)}"
        )
    others = {d for d in _EXCLUDED if d != driver}
    return frozenset(_SHARED_EXCLUDED | _EXCLUDED[driver] | others)


def _ignore_for(driver: str):
    """Build the `copytree` filter for one driver."""
    excluded = _excluded_for(driver)

    def _ignore(directory: str, names: list[str]) -> set[str]:
        drop = {n for n in names if n == "__pycache__" or n.endswith((".pyc", ".pyo"))}
        here = Path(directory).resolve()
        for name in names:
            try:
                rel = (here / name).relative_to(_PACKAGE).as_posix()
            except ValueError:
                continue
            if rel in excluded:
                drop.add(name)
        return drop

    return _ignore


def stage_package(into: Path | str, *, driver: str, name: str = STAGE_DIR) -> Path:
    """Copy one driver's package under ``into``; return the path for ``PYTHONPATH``.

    ``driver`` is one of :data:`DRIVERS`. The stage carries that driver's
    solver-reachable modules and the shared guardrail, and nothing from the
    other driver -- so an ARC-AGI-2 solver never has the ARC-AGI-3 client on its
    import path, or the reverse.

    Idempotent: an existing stage is replaced, so a resumed or re-run attempt
    cannot inherit a stale copy from a previous launch.
    """
    excluded = _excluded_for(driver)

    # **A denylist of paths fails open, so it is checked at both ends.**
    # It names paths as strings. A rename, a move or a typo leaves an entry
    # matching nothing, the filter drops nothing, and the module is staged --
    # silently, because nothing downstream asks. The pre-condition below turns
    # that into a launch failure instead of a leak.
    missing = sorted(e for e in excluded if not (_PACKAGE / e).exists())
    if missing:
        raise AssertionError(
            "staging exclusions name paths that do not exist: "
            + ", ".join(missing)
            + ". A stale entry excludes nothing, so fix the list rather than "
            "deleting the entry."
        )

    root = Path(into).resolve() / name
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    dest = root / _PACKAGE.name
    shutil.copytree(_PACKAGE, dest, ignore=_ignore_for(driver))
    for asset in dest.rglob("*.md"):
        asset.unlink()

    # The post-condition. `_ignore` runs per directory during the walk, so a
    # path under a directory that was itself dropped is never offered to it --
    # asserting on the result is what proves the outcome rather than the intent.
    leaked = sorted(e for e in excluded if (dest / e).exists())
    if leaked:
        raise AssertionError(
            "staging copied paths it was told to exclude: " + ", ".join(leaked)
        )

    # **Docstrings and comments carry environment names, and the stage is where
    # they come out.** They are there for a good reason -- they carry the
    # provenance and rationale an auditor reads -- so the repository keeps them
    # and the copy the solver imports does not.
    #
    # The redaction pattern matches bare mentions as well as backticked ones;
    # see the note on `_GAME_ID`. The wider net below re-checks the result,
    # because a check written with the redaction's own pattern agrees with it
    # by construction.
    redacted = 0
    decommented = 0
    for mod in dest.rglob("*.py"):
        text = mod.read_text(encoding="utf-8", errors="surrogateescape")
        stripped, comments = _strip_comments(text)
        swapped, n = _GAME_ID.subn("an environment", stripped)
        if n or comments:
            mod.write_text(swapped, encoding="utf-8", errors="surrogateescape")
            redacted += n
            decommented += comments
    # A post-condition, not decoration: the whole point is that the caller can
    # trust the removal happened. An empty stage would otherwise look like a
    # success and fail at import time, inside the guardrail, with the run already
    # started.
    leftover = sorted(p.relative_to(dest) for p in dest.rglob("*.md"))
    if leftover:
        raise AssertionError(f"staged package still carries prompts: {leftover}")
    if not (dest / "__init__.py").exists():
        raise AssertionError(f"staged package is not importable: {dest}")

    # **The removal is asserted here, on the path every launch takes.** A
    # check that reads the package in *this* working tree says nothing about
    # the tree a run actually launches from -- an older clone, or a rebuilt
    # container -- which stages whatever it holds. Checking at stage time means
    # a stale tree refuses to launch instead of handing a solver names it was
    # supposed to lose.
    #
    # The net is deliberately WIDER than `_GAME_ID`: four characters, word
    # bounded, letter first, containing a digit. A check written with the
    # redaction's own pattern agrees with it by construction, which is why
    # this one is wider.
    named = sorted({
        m.group(0)
        for mod in dest.rglob("*.py")
        for m in _ID_NET.finditer(mod.read_text(encoding="utf-8", errors="replace"))
        if m.group(0) not in _ID_ALLOWED
    })
    if named:
        raise AssertionError(
            f"staged package still names {len(named)} environment-shaped "
            f"token(s): {named}. This tree's redaction pattern did not cover "
            f"them; rebuild from a current checkout rather than launching."
        )
    return root


