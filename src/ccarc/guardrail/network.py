"""Run a child with exactly one reachable network destination.

A fresh network namespace has one interface and an empty routing table. Nothing
is filtered; there is nowhere for traffic to go. The child keeps reaching the
local shim on the loopback address it already uses, because a UNIX socket
crosses the namespace boundary and a relay inside forwards to it.

Unprivileged: no root, no veth, no firewall. `build_command` arms nothing on its
own -- a caller has to opt in.

Design notes are deliberately not written here: a module the child imports
is a module it can render.
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
import threading
from pathlib import Path

# Design notes. Kept as comments rather than docstrings: `help()` and `pydoc`
# render docstrings and cannot reach a `#` comment. (`inspect.getsource` on the
# module still can, so nothing here names a figure, a store, or a route to one.)
#
# * Routing, not filtering. An empty routing table is the whole guarantee and a
#   reviewer checks it with one command. A ruleset would have to be audited, and
#   a rule that is wrong is wrong silently.
# * The bridge is a UNIX socket because moving a veth into the host namespace
#   needs CAP_NET_ADMIN there, and this harness must not require a privileged
#   setup step.
# * The relay binds the same loopback port the child already targets, so the
#   proxy URL keeps its shape and no recorded run shows a different one.
# * Nested user namespaces restore the real uid before exec, for the reason
#   `confine.py` documents: the CLI refuses its permission flag under root.

#: `sun_path` is 108 bytes including its NUL, so 107 characters are usable.
#: Checked before the namespace is created: `bind` would otherwise fail with
#: `OSError: AF_UNIX path too long` at launch, after the namespace already
#: exists. Refusing early turns a mid-launch surprise into a startup error
#: that names the path.
SUN_PATH_MAX = 107

SOCKET_NAME = ".arc-shim.sock"


def available() -> str:
    """``"unshare"`` if an unprivileged network namespace really works here.

    Probes the mechanism rather than the binary, for the reason `confine.available`
    gives: the binary existing proves nothing about the kernel's userns policy,
    and a launch-time surprise costs a game. The probe asserts the property this
    module sells -- an empty routing table -- not merely that `unshare` exited 0.
    """
    if not shutil.which("unshare"):
        return ""
    try:
        probe = subprocess.run(
            ["unshare", "--user", "--map-root-user", "--net", "/bin/sh", "-c",
             "ip route show | wc -l"],
            capture_output=True, timeout=15, text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if probe.returncode != 0:
        return ""
    return "unshare" if probe.stdout.strip() == "0" else ""


def socket_path(attempt: Path | str) -> Path:
    """The bridge socket, inside the attempt directory so the guardrail can see it.

    The attempt directory is what `confine.build_command` holds read-write, so a
    socket there survives the mount namespace. Anywhere else is hidden by design.
    """
    path = Path(attempt).resolve() / SOCKET_NAME
    if len(str(path)) > SUN_PATH_MAX:
        raise ValueError(
            f"bridge socket path is {len(str(path))} bytes and the kernel "
            f"allows {SUN_PATH_MAX}; shorten the sweep or attempt directory "
            f"name: {path}"
        )
    return path


# The relay runs INSIDE the guardrail. It is passed as `python3 -c` source rather
# than written to a file, so nothing new appears in a workspace a solver reads
# and no file has to be cleaned up after a kill.
#
# `SO_REUSEADDR` because a nudge relaunches the solver into the same namespace
# lifetime and the previous relay's socket may still be in TIME_WAIT.
_RELAY = """\
import os, socket, sys, threading, time
pairs = [a.rsplit(":", 1) for a in sys.argv[1:]]
# The relay must die with its run, or every launch leaks one for good.
# The shell backgrounds this process and then execs the solver, so our parent IS
# the solver and keeps its pid across the exec; when the solver exits we are
# reparented and would otherwise serve a socket that no longer exists, forever.
# Test for a CHANGED ppid, never for a reparented-to pid: on WSL an orphan goes
# to a subreaper that is not the init process, so testing against init's pid
# silently never fires. Its literal value is deliberately not written here -- a
# bare integer in importable prose can collide with a real action cap, and the
# leak guard is built to refuse exactly that.
#
# The expected parent is passed IN by the shell rather than read here. Python
# takes long enough to start that a launch which fails immediately reparents
# this process before its first instruction runs, and a ppid sampled then is
# already the reparented one -- so the comparison below could never differ and
# the guard could never fire. That is the case where cleanup matters most.
# Check before sleeping, so an already-orphaned relay exits at once.
_ppid0 = int(os.environ.get("CCARC3_NETGUARD_PPID") or os.getppid())
def _exit_with_parent():
    while True:
        if os.getppid() != _ppid0:
            os._exit(0)
        time.sleep(5)
threading.Thread(target=_exit_with_parent, daemon=True).start()
def pump(a, b):
    try:
        while True:
            d = a.recv(65536)
            if not d:
                break
            b.sendall(d)
    except OSError:
        pass
    finally:
        try:
            b.shutdown(socket.SHUT_WR)
        except OSError:
            pass
def serve(port, sock):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", int(port)))
    srv.listen(64)
    while True:
        c, _ = srv.accept()
        try:
            u = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            u.connect(sock)
        except OSError:
            c.close()
            continue
        threading.Thread(target=pump, args=(c, u), daemon=True).start()
        threading.Thread(target=pump, args=(u, c), daemon=True).start()
ts = [threading.Thread(target=serve, args=(p, s), daemon=True) for p, s in pairs]
for t in ts:
    t.start()
for t in ts:
    t.join()
"""

# `ip link set lo up` is required: without it the guardrail's own 127.0.0.1 refuses
# every connection and the relay cannot even bind, so the game dies rather than
# the internet. `|| true` because a box without iproute2 should fail on the
# relay's own bind with a clear error, not here.
#
# The relay is backgrounded and then the solver is exec'd under a nested user
# namespace that restores the real uid -- the same nesting `confine.py` needs,
# and for the same reason: the CLI refuses `--dangerously-skip-permissions`
# under root.
#
# **The relay's file descriptors must be detached.** A backgrounded child
# inherits stdout, so the launcher's pipe stays open for as long as the relay
# lives -- which is forever. Any launcher that captures the launch's output
# would hang there, with no error.
#: Run inside the namespace between starting the relay and exec'ing the solver.
#: Polls every bridged port until it accepts, then exits 0; exits 1 on the
#: deadline so the failure is loud here rather than arriving as the model's
#: first turn saying "Connection refused".
_WAIT = """\
import socket, sys, time
pairs = [a.rsplit(":", 1) for a in sys.argv[1:]]
deadline = time.monotonic() + float(__import__("os").environ.get(
    "CCARC3_NETGUARD_WAIT_S") or 20)
for port, sock in pairs:
    while True:
        try:
            s = socket.create_connection(("127.0.0.1", int(port)), 1.0)
            s.close()
            break
        except OSError:
            if time.monotonic() > deadline:
                sys.stderr.write(
                    "network: relay never bound 127.0.0.1:%s\\n" % port)
                raise SystemExit(1)
            time.sleep(0.05)
"""

_SCRIPT = """\
set -e
ip link set lo up 2>/dev/null || true
CCARC3_NETGUARD_PPID=$$ "$CCARC3_NETGUARD_PY" -c "$CCARC3_NETGUARD_RELAY" \
    $CCARC3_NETGUARD_PAIRS >/dev/null 2>&1 </dev/null &
"$CCARC3_NETGUARD_PY" -c "$CCARC3_NETGUARD_WAIT" $CCARC3_NETGUARD_PAIRS
exec unshare --user --map-user="$CCARC3_NETGUARD_UID" \
     --map-group="$CCARC3_NETGUARD_GID" -- "$@"
"""


def build_command(argv: list[str], *, port: int | None = None,
                  sock: Path | str | None = None,
                  bridges: "list[tuple[int, Path | str]] | None" = None,
                  env: dict[str, str] | None = None,
                  python: str | None = None) -> list[str]:
    """Wrap ``argv`` so it runs with one reachable destination. Mutates ``env``.

    ``port`` is the loopback port the solver already believes the shim is on, so
    ``CCARC3_PROXY_URL`` does not change. ``sock`` is the UNIX socket the host
    side serves -- see :class:`Bridge`.
    """
    # One destination or several. ARC-AGI-2 needs only the model backend;
    # ARC-AGI-3 needs the game shim as well, and a solver that cannot reach the
    # shim cannot play at all. Each pair gets its own relay thread inside.
    pairs = list(bridges or [])
    if port is not None and sock is not None:
        pairs.append((port, sock))
    if not pairs:
        raise ValueError("a network namespace with no bridge would strand the child")
    for prt, sk in pairs:
        sk = Path(sk)
        if len(str(sk)) > SUN_PATH_MAX:
            raise ValueError(f"bridge socket path is {len(str(sk))} bytes and "
                             f"the kernel allows {SUN_PATH_MAX}: {sk}")
        if any(c.isspace() for c in str(sk)):
            raise ValueError(f"bridge socket path cannot contain whitespace: {sk}")
    if env is not None:
        env["CCARC3_NETGUARD_PY"] = python or "python3"
        env["CCARC3_NETGUARD_RELAY"] = _RELAY
        env["CCARC3_NETGUARD_WAIT"] = _WAIT
        env["CCARC3_NETGUARD_PAIRS"] = " ".join(f"{prt}:{sk}" for prt, sk in pairs)
        env["CCARC3_NETGUARD_PORT"] = str(pairs[0][0])
        env["CCARC3_NETGUARD_SOCK"] = str(pairs[0][1])
        env["CCARC3_NETGUARD_UID"] = str(os.getuid())
        env["CCARC3_NETGUARD_GID"] = str(os.getgid())
    return ["unshare", "--user", "--map-root-user", "--net",
            "/bin/sh", "-c", _SCRIPT, "netguard", *argv]


#: Seconds to wait for the shim to ACCEPT a connection. Named rather than
#: inline so a caller can shrink it instead of waiting out the real deadline.
CONNECT_TIMEOUT_S = 10


class Bridge:
    """Host side of the crossing: a UNIX socket that forwards to the shim.

    Runs in the launching process, OUTSIDE the guardrail, so it keeps ordinary
    network access. It is the only route out of the namespace, which is what
    makes the shim's own endpoint allowlist load-bearing rather than advisory.
    """

    def __init__(self, sock: Path | str, upstream_port: int,
                 upstream_host: str = "127.0.0.1") -> None:
        self.sock = Path(sock)
        self.upstream = (upstream_host, upstream_port)
        self._srv: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> "Bridge":
        if len(str(self.sock)) > SUN_PATH_MAX:
            raise ValueError(f"bridge socket path is {len(str(self.sock))} "
                             f"bytes and the kernel allows {SUN_PATH_MAX}")
        # A stale socket from a killed run would make `bind` fail with EADDRINUSE
        # on every restart; the file is not state worth keeping.
        if self.sock.exists():
            self.sock.unlink()
        self.sock.parent.mkdir(parents=True, exist_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.sock))
        srv.listen(64)
        srv.settimeout(0.5)
        os.chmod(self.sock, 0o600)
        self._srv = srv
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        return self

    def _serve(self) -> None:
        assert self._srv is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            up = socket.create_connection(self.upstream, timeout=CONNECT_TIMEOUT_S)
        except OSError:
            conn.close()
            return
        # **Clear it. `create_connection`'s timeout is a CONNECT deadline
        # and Python leaves it on the socket for every later `recv`.** Left in
        # place, a shim response slower than the deadline becomes a silent
        # half-close toward the solver: `_pump`'s `recv` raises
        # `socket.timeout`, `except OSError` swallows it, and the `finally`
        # sends FIN with zero bytes, which the solver reads as
        # `RemoteDisconnected` while the shim is still waiting upstream.
        #
        # This bridge cannot see HTTP and does not log, so a deadline here
        # would be the tightest one in the stack sitting in the only layer
        # that can neither report nor retry. Deadlines belong in the client
        # and the shim. `EgressProxy._handle` clears its inbound socket for
        # the same reason.
        up.settimeout(None)
        conn.settimeout(None)
        for a, b in ((conn, up), (up, conn)):
            threading.Thread(target=self._pump, args=(a, b), daemon=True).start()

    @staticmethod
    def _pump(a: socket.socket, b: socket.socket) -> None:
        try:
            while True:
                data = a.recv(65536)
                if not data:
                    break
                b.sendall(data)
        except TimeoutError:
            # **A timeout is not end-of-stream.** Both sockets are blocking
            # by the time they reach here, so this cannot fire; it exists so
            # that anyone who reintroduces a deadline gets a loud failure
            # instead of a truncated response delivered as a clean EOF.
            raise
        except OSError:
            pass
        finally:
            try:
                b.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def stop(self) -> None:
        self._stop.set()
        if self._srv is not None:
            self._srv.close()
        if self.sock.exists():
            try:
                self.sock.unlink()
            except OSError:
                pass

    def __enter__(self) -> "Bridge":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


def describe(port: int, sock: Path | str) -> str:
    """One log line, so a launch is auditable from its own output."""
    return (f"network: empty routing table; only 127.0.0.1:{port} answers, "
            f"bridged to {sock}")
