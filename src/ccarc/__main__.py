"""`ccarc` on the command line: one entry point, two benchmark drivers."""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ccarc",
        description="Drive a coding agent at ARC-AGI, inside a guardrail that makes "
                    "the result checkable.")
    # **Hyphens, not the bare forms.** `staging` refuses to hand a solver any
    # four-character letter-first token containing a digit -- the shape of a
    # game identifier -- and this module is part of the staged package. The
    # bare forms are that shape; the hyphenated ones are not.
    sub = ap.add_subparsers(dest="benchmark", required=True)
    sub.add_parser("arc-2", help="ARC-AGI-2: static puzzles, scored on partial credit")
    sub.add_parser("arc-3", help="ARC-AGI-3: interactive games, scored on "
                                "action efficiency")
    return ap


def main(argv: list[str] | None = None) -> int:
    # **Parse only our own first token and hand the rest on untouched.** Each
    # driver owns a large, independent option set; re-declaring them here would
    # be a second copy to keep in step with the first.
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        build_parser().print_help()
        return 0
    which, rest = argv[0], argv[1:]
    if which == "arc-2":
        from ccarc.arc_2 import cli
        return cli.main(rest)
    if which == "arc-3":
        from ccarc.arc_3 import cli
        return cli.main(rest)
    build_parser().error(f"unknown benchmark {which!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
