"""A CONNECT proxy that serves one allowlist and refuses everything else.

Pairs with `network`. The guardrail leaves a child with an empty routing table, which
also removes the route the child's own model backend needs. This restores
exactly that one destination and nothing more.

CONNECT only. A plain-HTTP forward path is not implemented because no client
here needs one, and every method not implemented is surface that cannot be
misused.

Listens on a UNIX socket so it can be reached across a network namespace.
"""
from __future__ import annotations

import os
import socket
import threading
from pathlib import Path

# Design notes as comments: `pydoc` renders docstrings and cannot reach these.
#
# * The allowlist matches a host exactly, or as a suffix on a DOT boundary, so
#   `api.anthropic.com` never matches `api.anthropic.com.evil.test` and
#   `anthropic.com` never matches `notanthropic.com`. Substring matching is how
#   an allowlist becomes a wildcard.
# * The port is checked too. An allowed host on an unexpected port is refused,
#   because a permitted name is not a permitted service.
# * Refusal is 403 and the socket closes. No upstream connection is opened, so a
#   refused destination never learns it was asked for.
# * The proxy runs OUTSIDE the guardrail, in the launcher, and is therefore the only
#   way out. That is the point: one process, one list, auditable in one place.

# `platform.claude.com` is the OAuth token-refresh endpoint, and it is here
# because leaving it out silently kills any run that outlives its access token:
# the run in flight dies logging `egress refused [... 'platform.claude.com:443']`,
# and every relaunch on the expired token exits within seconds.
#
# The host copy being fresh does not help: a solver holds a per-run copy of the
# credentials made at launch, and with no route to the refresh endpoint it cannot
# renew it from inside the guardrail. Refreshing on the host fixes the NEXT launch,
# never the run in flight.
#
# This grants no capability the solver did not already have. It already holds an
# authenticated channel to `api.anthropic.com`; this is the same account, the
# same TLS, one more Anthropic host. Match is exact-or-dot-boundary, so this
# entry admits `platform.claude.com` and its subdomains and nothing else --
# writing `claude.com` here would admit the entire domain and would be wrong.
DEFAULT_ALLOW: tuple[str, ...] = ("api.anthropic.com", "platform.claude.com")
DEFAULT_PORTS: frozenset[int] = frozenset({443})

ALLOW_ENV = "CCARC3_EGRESS_ALLOW"


def allowed_hosts() -> tuple[str, ...]:
    """The allowlist, overridable for a different backend or a test."""
    raw = os.environ.get(ALLOW_ENV, "").strip()
    if not raw:
        return DEFAULT_ALLOW
    return tuple(h.strip() for h in raw.split(",") if h.strip())


def permits(host: str, port: int, allow: tuple[str, ...],
            ports: frozenset[int] = DEFAULT_PORTS) -> bool:
    """Exact host, or a dot-boundary suffix, on an expected port."""
    if port not in ports:
        return False
    host = host.strip().lower().rstrip(".")
    for entry in allow:
        entry = entry.strip().lower().rstrip(".")
        if host == entry or host.endswith("." + entry):
            return True
    return False


class EgressProxy:
    """CONNECT proxy on a UNIX socket, serving one allowlist."""

    def __init__(self, sock: Path | str, allow: tuple[str, ...] | None = None,
                 ports: frozenset[int] = DEFAULT_PORTS) -> None:
        self.sock = Path(sock)
        self.allow = tuple(allow) if allow is not None else allowed_hosts()
        self.ports = ports
        self.refused: list[str] = []
        self.permitted: list[str] = []
        self._srv: socket.socket | None = None
        self._stop = threading.Event()

    def start(self) -> "EgressProxy":
        if self.sock.exists():
            self.sock.unlink()
        self.sock.parent.mkdir(parents=True, exist_ok=True)
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(self.sock))
        srv.listen(64)
        srv.settimeout(0.5)
        os.chmod(self.sock, 0o600)
        self._srv = srv
        threading.Thread(target=self._serve, daemon=True).start()
        return self

    def _serve(self) -> None:
        assert self._srv is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,),
                             daemon=True).start()

    def _handle(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(20)
            head = b""
            while b"\r\n\r\n" not in head:
                chunk = conn.recv(4096)
                if not chunk:
                    conn.close()
                    return
                head += chunk
                if len(head) > 32768:            # a header this large is not a CONNECT
                    conn.close()
                    return
            line = head.split(b"\r\n", 1)[0].decode("latin-1")
            parts = line.split()
            if len(parts) < 2 or parts[0].upper() != "CONNECT":
                self._deny(conn, line)
                return
            target = parts[1]
            host, _, port_s = target.rpartition(":")
            try:
                port = int(port_s)
            except ValueError:
                self._deny(conn, target)
                return
            if not permits(host, port, self.allow, self.ports):
                self._deny(conn, target)
                return
            try:
                up = socket.create_connection((host, port), timeout=20)
                # `create_connection`'s timeout is a CONNECT deadline that
                # Python leaves on the socket for every later `recv`. Clear it
                # once the tunnel is up -- as the inbound socket's is cleared
                # after the handshake -- or silence from an allowlisted host
                # would cut the solver's own TLS tunnel by a swallowed
                # timeout that reads as a clean close.
                up.settimeout(None)
            except OSError:
                conn.sendall(b"HTTP/1.1 502 Bad Gateway\r\n\r\n")
                conn.close()
                return
            self.permitted.append(target)
            conn.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            conn.settimeout(None)
            for a, b in ((conn, up), (up, conn)):
                threading.Thread(target=self._pump, args=(a, b),
                                 daemon=True).start()
        except OSError:
            try:
                conn.close()
            except OSError:
                pass

    def _deny(self, conn: socket.socket, target: str) -> None:
        self.refused.append(target)
        try:
            conn.sendall(b"HTTP/1.1 403 Forbidden\r\n"
                         b"Content-Length: 0\r\n\r\n")
        except OSError:
            pass
        conn.close()

    @staticmethod
    def _pump(a: socket.socket, b: socket.socket) -> None:
        try:
            while True:
                data = a.recv(65536)
                if not data:
                    break
                b.sendall(data)
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

    def __enter__(self) -> "EgressProxy":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
