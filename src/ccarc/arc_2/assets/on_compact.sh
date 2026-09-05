#!/usr/bin/env bash
# SessionStart(compact) hook — ccarc CC harness.
#
# Claude Code compacts context on its own schedule. The workspace keeps its
# research state on disk continuously — the invariant ledger and NOTES.md —
# so recovery is a matter of replaying that state into the fresh window.
set -uo pipefail

cd "__WORKSPACE__" 2>/dev/null || exit 0

echo "--- ccarc CC harness: context was compacted. Distilled research state follows. ---"
echo
"__PYTHON__" gate.py status 2>&1 || true
echo
echo "Re-read CLAUDE.md and the tail of NOTES.md before doing anything else."
echo "Exploratory Python state did not survive the compaction; anything you need"
echo "must be re-derived by running a script under explore/."
