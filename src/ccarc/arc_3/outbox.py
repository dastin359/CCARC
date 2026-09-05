"""Where the harness leaves what it wants said, for the driver to say.

The client does not run in the driver. It runs in a ``python -c`` grandchild of
the solver, inside the mount namespace, and it exits after every action -- so it
cannot hold the solver's stdin and cannot speak to the model itself. The only
thing that outlives it and that both sides can see is a file.

So: the client appends a line here; the driver, which owns the live session,
reads the new lines and sends each as a real user message. Append-only, one JSON
object per line, read back by byte offset, exactly like the trace -- and for the
same reason, that it is written while it is being read.

The file is also the record of what the harness said and when, so a message
that was never delivered is distinguishable from one that was.
"""

# **Not in the solver's working directory.** The workspace is a public place --
# anything in it can be read by the solver and quoted verbatim into the stream.
# Nothing here is secret (it is text the solver is about to be told anyway) but
# the placement rule -- harness state lives outside the workspace -- is worth
# keeping uniform, so the driver names the path and puts it in the attempt
# directory rather than the workspace.
#
# **The driver names the path through the environment**, and its absence is the
# signal that there is no live session to deliver anything: with no outbox
# configured the client prints instead, so the words still reach the solver as
# a tool result rather than vanishing.

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

__all__ = ["ENV_VAR", "Outbox", "configured_path", "say"]

ENV_VAR = "CCARC3_HARNESS_OUTBOX"


def configured_path(env: dict[str, str] | None = None) -> Path | None:
    """Where to leave messages, or ``None`` when nothing will deliver them."""
    value = (env if env is not None else os.environ).get(ENV_VAR, "")
    return Path(value) if value else None


def say(text: str, *, kind: str = "note", env: dict[str, str] | None = None) -> bool:
    """Ask the driver to send ``text`` to the solver as a user message.

    Returns True if it was written, False when no outbox is configured -- in
    which case the caller is responsible for saying it some other way.
    """
    path = configured_path(env)
    if path is None or not text:
        return False
    record = {"at": time.time(), "kind": kind, "text": text}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")) + "\n")
    return True


class Outbox:
    """The driver's side: everything left here since the last look."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._offset = 0
        self._digest = hashlib.sha256(b"").digest()

    def drain(self) -> list[dict[str, Any]]:
        """Return the records appended since the previous call.

        A record still being written has no terminating newline and is left for
        the next call rather than half-read. A file that has been replaced or
        truncated is read again from the beginning: the alternative is silently
        skipping whatever a new run put there.
        """
        try:
            st = self.path.stat()
        except FileNotFoundError:
            self._forget()
            return []
        # **The whole consumed prefix is hashed, not the tail and not the
        # inode.** A file deleted and rewritten at the same length can reuse
        # the inode, so neither identity nor size shows that the content
        # changed. The outbox holds a handful of short lines, so re-reading
        # what has already been read costs nothing and is exact.
        if st.st_size < self._offset:
            self._forget()
        elif self._offset:
            with self.path.open("rb") as fh:
                if hashlib.sha256(fh.read(self._offset)).digest() != self._digest:
                    self._forget()
        if st.st_size == self._offset:
            return []
        with self.path.open("rb") as fh:
            fh.seek(self._offset)
            blob = fh.read()
        lines = blob.split(b"\n")
        if blob.endswith(b"\n"):
            lines.pop()
            consumed = len(blob)
        else:
            consumed = len(blob) - len(lines.pop())
        out: list[dict[str, Any]] = []
        for raw in lines:
            if not raw or raw.isspace():
                continue
            try:
                record = json.loads(raw)
            except ValueError:
                # A malformed line must not stop the ones after it. Nothing here
                # is scored, so dropping it and carrying on costs a message;
                # raising would cost the run.
                continue
            if isinstance(record, dict) and record.get("text"):
                out.append(record)
        self._offset += consumed
        with self.path.open("rb") as fh:
            self._digest = hashlib.sha256(fh.read(self._offset)).digest()
        return out

    def _forget(self) -> None:
        self._offset = 0
        self._digest = hashlib.sha256(b"").digest()
