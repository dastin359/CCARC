"""A solver session that stays alive, so the harness can speak to it.

**Why this exists.** Without it the harness has exactly one way to say
anything to a solver: print into the stdout of a tool call, so the words arrive
as a *tool result*. Everything else -- the nudges, the finish question -- is a
relaunch, `claude -p <prompt> --resume <id>`, which starts a new process on an
old conversation. Two costs follow from that shape:

* A relaunch injects ``"Continue from where you left off."`` whenever the
  previous turn ended badly -- mid-turn, or on an API error. That text is not
  ours and it is solver-visible: nothing should reach the conversation that
  the harness did not choose to say, and a relaunch cannot guarantee that.
* A run ends when the *model* stops talking. If it stops for a reason the
  harness could fix -- an expired card, a stale position, a question it did
  not know it is allowed to ask -- there is no way to say so. The process is
  already gone.

With the session held open, the harness writes a real user message onto the
child's stdin and the model receives it as one, in the same conversation, with
nothing injected. The run ends when the harness closes stdin, not when the model
falls silent.

**The delivery semantics this relies on.** A message written onto the child's
stdin arrives as a real user message, in the same conversation, with no
injected text anywhere in the stream or the transcript. A message written
while a tool call is running is **not** an interruption -- the CLI holds it
until the call finishes and then delivers it attached to the completed tool
result. That is the behaviour the harness wants and it needs no arrangement
of ours.

**What this module does not do.** It does not decide what to say or when. It is
the transport: start the child, write messages, read events, notice where turns
end, and close down. Policy belongs to the driver.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

__all__ = ["LiveSession", "SessionGone", "user_message"]


class SessionGone(RuntimeError):
    """The child is not running, so nothing can be said to it."""


def user_message(text: str) -> dict[str, Any]:
    """One line of ``--input-format stream-json``: a real user message.

    The shape the CLI accepts. Kept as a function rather than inlined so the
    one place that knows the wire format is the one place a change has to
    reach.
    """
    return {
        "type": "user",
        "message": {"role": "user", "content": [{"type": "text", "text": text}]},
    }


class LiveSession:
    """A `claude` child held open on its stdin.

    ``args`` must already carry ``--input-format stream-json`` and
    ``--output-format stream-json``; :func:`live_args` builds them. Every line
    the child writes is appended verbatim to ``stream_path`` as well as being
    decoded, because ``stream.jsonl`` is the verbatim record of the run's cost
    and conversation, and its format is part of the contract.

    Not reusable: one instance drives one child. Start it, talk to it, close it.
    """

    def __init__(self, args: list[str], *, cwd: Path | str, env: dict[str, str],
                 stream_path: Path | str,
                 on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.args = list(args)
        self.cwd = Path(cwd)
        self.env = dict(env)
        self.stream_path = Path(stream_path)
        self._on_event = on_event

        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []
        self._results: list[dict[str, Any]] = []
        self._said: list[tuple[float, str]] = []
        self._settled = threading.Event()
        self._fh = None
        self._start_error: str = ""

    # -- lifetime ---------------------------------------------------------

    def start(self) -> "LiveSession":
        """Spawn the child and begin reading it."""
        if self._proc is not None:
            raise RuntimeError("this session has already been started")
        self.stream_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.stream_path.open("w", encoding="utf-8")
        # `start_new_session` for the same reason `_launch` uses it: the solver
        # drives the game from `python -c` grandchildren, and a signal that
        # reaches only `claude` leaves those running -- still spending the
        # budget, still writing the ledger, against a run already declared over.
        self._proc = subprocess.Popen(
            self.args, cwd=str(self.cwd), env=self.env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=True,
        )
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        return self

    def _pump(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            if self._fh is not None:
                self._fh.write(line if line.endswith("\n") else line + "\n")
                self._fh.flush()
            text = line.strip()
            if not text:
                continue
            try:
                event = json.loads(text)
            except ValueError:
                # The CLI writes its stderr here too, and a traceback is not
                # JSON. Keeping it as an event rather than dropping it is how
                # the driver gets to report why a launch died.
                event = {"type": "stderr", "text": text}
            with self._lock:
                self._events.append(event)
                if event.get("type") == "result":
                    self._results.append(event)
                    self._settled.set()
                elif event.get("type") == "system" and event.get("subtype") == "init":
                    # A new turn is starting; it is no longer settled.
                    self._settled.clear()
            if self._on_event is not None:
                try:
                    self._on_event(event)
                except Exception:                       # noqa: BLE001
                    pass                                # a watcher must not kill the pump

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    # -- speaking ---------------------------------------------------------

    def say(self, text: str) -> None:
        """Send one real user message. Never interrupts work in flight.

        The CLI queues it until the current tool call completes and then
        delivers it as a user message in the same conversation.
        """
        if self._proc is None:
            raise SessionGone("the session was never started")
        if self._proc.poll() is not None:
            raise SessionGone(f"the solver exited {self._proc.returncode} before this could be sent")
        assert self._proc.stdin is not None
        # **Cleared here, not when the child says a turn began.** `say` returns
        # as soon as the bytes are written, and the child's `system/init` lands
        # some milliseconds later. In that gap the previous turn's result would
        # still be standing, `wait_settled` would return True immediately, and
        # the driver would believe a turn it had just started was already over.
        self._settled.clear()
        try:
            self._proc.stdin.write(json.dumps(user_message(text)) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, ValueError) as exc:
            # A race the driver cannot avoid: the child can die between the
            # poll above and the write. Report it as the same condition rather
            # than as a different kind of failure.
            raise SessionGone(f"the solver's input closed while sending: {exc}") from None
        with self._lock:
            self._said.append((time.time(), text))

    @property
    def said(self) -> list[tuple[float, str]]:
        """Every message the harness sent, with when it was sent."""
        with self._lock:
            return list(self._said)

    @property
    def last_said(self) -> str:
        """The most recent message the harness sent, or ``""``.

        Exists so a caller that wants to repeat itself does not have to know
        that `said` holds pairs. A driver recovering from a failed turn needs
        exactly this and nothing else.
        """
        with self._lock:
            return self._said[-1][1] if self._said else ""

    # -- turn boundaries --------------------------------------------------

    def wait_settled(self, timeout: float | None = None) -> bool:
        """Block until the current turn has produced its ``result`` event.

        Returns False on timeout, or if the child died without settling. A
        settled turn is the property that matters for a later ``--resume``: the
        CLI injects its own text only when the previous turn ended *badly*, so
        a driver that always closes on a settled turn leaves a transcript that
        can be resumed with nothing added.
        """
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            if self._settled.wait(0.2):
                return True
            if not self.alive:
                # Drain: the pump may still be finishing the last few lines.
                if self._reader is not None:
                    self._reader.join(timeout=5)
                return self._settled.is_set()
            if end is not None and time.monotonic() >= end:
                return False

    @property
    def settled(self) -> bool:
        return self._settled.is_set()

    @property
    def turns(self) -> int:
        """How many turns have produced a result."""
        with self._lock:
            return len(self._results)

    @property
    def results(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._results)

    @property
    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def last_result(self) -> dict[str, Any] | None:
        with self._lock:
            return self._results[-1] if self._results else None

    def failing(self) -> str:
        """Why the last turn failed, or ``""``.

        A live session survives an error that would have ended a `-p` run --
        the child stays up and keeps accepting messages. That is useful and it
        is also a trap: a driver that only watches the process would see a
        healthy session doing nothing. A dead credential, for example,
        produces consecutive ``is_error`` results reading *"Failed to
        authenticate: OAuth session expired and could not be refreshed"*,
        with the child alive throughout.
        """
        last = self.last_result()
        if not last or not last.get("is_error"):
            return ""
        return str(last.get("result") or last.get("api_error_status") or "error")

    # -- shutdown ---------------------------------------------------------

    def close(self, timeout: float = 120.0) -> int:
        """Close stdin and wait for the child to finish. Returns its exit code.

        Closing stdin is how a run ends. The model falling silent ends nothing,
        which is the point: the harness gets to ask whether the solver is
        finished and hear the answer before deciding.
        """
        if self._proc is None:
            return 0
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:                               # noqa: BLE001
            pass
        try:
            code = self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            code = self.kill()
        self._finish_reader()
        return code

    def kill(self, grace: float = 10.0) -> int:
        """Stop the child and everything it started. Returns its exit code."""
        if self._proc is None:
            return 0
        import signal

        for sig in (signal.SIGTERM, signal.SIGKILL):
            if self._proc.poll() is not None:
                break
            try:
                # The whole process group: the solver's `python -c`
                # grandchildren POST actions of their own, and killing only
                # `claude` leaves them spending a budget the driver has
                # already written off.
                os.killpg(os.getpgid(self._proc.pid), sig)
            except (ProcessLookupError, PermissionError):
                try:
                    self._proc.send_signal(sig)
                except Exception:                       # noqa: BLE001
                    pass
            try:
                self._proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                continue
        self._finish_reader()
        return self._proc.returncode if self._proc.returncode is not None else -9

    def _finish_reader(self) -> None:
        if self._reader is not None:
            self._reader.join(timeout=10)
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def __enter__(self) -> "LiveSession":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        if self.alive:
            self.kill()
        else:
            self._finish_reader()


def live_args(base: Iterable[str]) -> list[str]:
    """Turn a one-shot `claude -p <prompt>` argv into a live-session argv.

    The prompt stops being an argument and becomes the first user message, so
    the caller has to send it; that is deliberate, because it makes the opening
    prompt the same kind of thing as everything the harness says afterwards
    rather than a special case.
    """
    args = list(base)
    out: list[str] = []
    skip = False
    for i, item in enumerate(args):
        if skip:
            skip = False
            continue
        if item == "-p" or item == "--print":
            out.append("--print")
            # `-p` takes the prompt as its value; `--print` does not.
            if item == "-p" and i + 1 < len(args) and not args[i + 1].startswith("-"):
                skip = True
            continue
        out.append(item)
    if "--print" not in out:
        out.insert(1, "--print")
    for flag, value in (("--input-format", "stream-json"),
                        ("--output-format", "stream-json")):
        if flag in out:
            out[out.index(flag) + 1] = value
        else:
            out += [flag, value]
    if "--verbose" not in out:
        out.append("--verbose")
    # Echoes each user message back on stdout, so `stream.jsonl` records what
    # the harness said as well as what the solver did. Without it the harness's
    # own words are the one part of the conversation the evidence omits.
    if "--replay-user-messages" not in out:
        out.append("--replay-user-messages")
    return out


def opening_prompt(base: Iterable[str]) -> str:
    """The prompt `-p` carried, which the caller must now send as a message."""
    args = list(base)
    for i, item in enumerate(args):
        if item == "-p" and i + 1 < len(args):
            return args[i + 1]
    return ""
