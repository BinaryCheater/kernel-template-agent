"""The agent proxy: detection, the injected environment, and the fail-closed preflight (WP22).

Codex and Claude Code do not read the macOS system proxy, so the runner injects the proxy
into their environment explicitly (D581). The proxy comes from the system configuration
(:mod:`kta.core.sysconfig`): an override, else what :func:`detect` finds (``scutil
--proxy``, then the control-plane environment, then Clash's default HTTP port). A proxy
URL handled here never carries credentials; the user:password part lives only in
``secrets.env`` (``KTA_PROXY_USERINFO``) and is added on the host that starts the CLI.

The preflight opens the proxy port and tunnels an HTTPS CONNECT plus a TLS handshake to
each API host of the provider. It sends no request to the API, so it spends no tokens.
"""

from __future__ import annotations

import base64
import ipaddress
import os
import socket
import ssl
import struct
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit

from kta.core.envprobe import run_probe
from kta.core.errors import ConfigError, ExecutionError

DEFAULT_URL = "http://127.0.0.1:7897"  # Clash's mixed/HTTP port
SCHEMES = ("http", "socks5", "socks5h")
# Every variable the injection writes; an `off` provider gets none of them.
PROXY_VARS: tuple[str, ...] = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
)
NO_PROXY_VARS: tuple[str, ...] = ("NO_PROXY", "no_proxy")
ALL_VARS: frozenset[str] = frozenset({*PROXY_VARS, *NO_PROXY_VARS})
LOOPBACK: tuple[str, ...] = ("127.0.0.1", "localhost", "::1")
USERINFO_SECRET = "KTA_PROXY_USERINFO"  # name in secrets.env, never its value
ENV_SOURCES = ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy")
PREFLIGHT_TIMEOUT_S = 8.0
# Ops pointer prefix of each provider's last preflight outcome (the breaker, D585).
STATUS_POINTER = "proxy:status:"

Connect = Callable[[tuple[str, int], float], socket.socket]


class ProxyUnavailable(ExecutionError):
    """A `required` provider's proxy failed its preflight: the CLI is not started (D584)."""

    def __init__(self, message: str, result: dict[str, Any] | None = None) -> None:
        super().__init__("infra", message)
        self.result = result or {}


@dataclass(frozen=True)
class Endpoint:
    """A proxy without credentials: what records, logs and the UI may show."""

    scheme: str
    host: str
    port: int

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"{self.scheme}://{host}:{self.port}"

    def with_userinfo(self, userinfo: str | None) -> str:
        if not userinfo:
            return self.url
        return self.url.replace("://", f"://{userinfo}@", 1)

    @property
    def loopback(self) -> bool:
        return is_loopback(self.host)

    def to_dict(self) -> dict[str, Any]:
        return {"scheme": self.scheme, "host": self.host, "port": self.port, "url": self.url}


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def parse_url(url: str) -> tuple[Endpoint, bool]:
    """(endpoint without credentials, whether the URL carried credentials)."""
    raw = url.strip()
    if "://" not in raw:
        raw = "http://" + raw
    parts = urlsplit(raw)
    scheme = parts.scheme.lower()
    if scheme == "https":
        scheme = "http"  # an HTTPS URL for a proxy means CONNECT over plain HTTP in practice
    if scheme not in SCHEMES:
        raise ConfigError(f"proxy scheme {parts.scheme!r} is not one of {SCHEMES}")
    try:
        port = parts.port
    except ValueError as exc:
        raise ConfigError(f"proxy URL {url!r}: {exc}") from exc
    if not parts.hostname or port is None:
        raise ConfigError(f"proxy URL {url!r} needs a host and a port (http://127.0.0.1:7897)")
    if parts.path not in ("", "/") or parts.query:
        raise ConfigError(f"proxy URL {url!r} must not have a path or query")
    creds = parts.username is not None or parts.password is not None
    return Endpoint(scheme, parts.hostname, port), creds


@dataclass(frozen=True)
class Detected:
    endpoint: Endpoint
    source: str  # scutil | env | default | override
    detail: str
    credentials_dropped: bool = False
    scutil: dict[str, str] = field(default_factory=lambda: dict[str, str]())

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["endpoint"] = self.endpoint.to_dict()
        return out


def parse_scutil(text: str) -> dict[str, str]:
    """``scutil --proxy`` as flat keys; ExceptionsList joined with commas."""
    out: dict[str, str] = {}
    items: list[str] = []
    in_list = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("ExceptionsList"):
            in_list = True
            continue
        if in_list:
            if s.startswith("}"):
                in_list = False
                out["ExceptionsList"] = ",".join(items)
                continue
            _, sep, value = s.partition(" : ")
            if sep:
                items.append(value.strip())
            continue
        key, sep, value = s.partition(" : ")
        if sep:
            out[key.strip()] = value.strip()
    return out


def from_scutil(values: Mapping[str, str]) -> tuple[Endpoint, str] | None:
    """The first enabled proxy of scutil: HTTPS, then HTTP, then SOCKS."""
    for prefix, scheme in (("HTTPS", "http"), ("HTTP", "http"), ("SOCKS", "socks5")):
        host, port = values.get(f"{prefix}Proxy"), values.get(f"{prefix}Port")
        if values.get(f"{prefix}Enable") == "1" and host and port and port.isdigit():
            return Endpoint(scheme, host, int(port)), f"{prefix}Enable=1 {host}:{port}"
    return None


def detect(scutil_text: str | None, environ: Mapping[str, str]) -> Detected:
    """scutil, then the environment, then Clash's default port; credentials are dropped."""
    values = parse_scutil(scutil_text) if scutil_text else {}
    found = from_scutil(values)
    if found is not None:
        return Detected(found[0], "scutil", found[1], scutil=values)
    for name in ENV_SOURCES:
        value = environ.get(name, "").strip()
        if not value:
            continue
        try:
            ep, creds = parse_url(value)
        except ConfigError:
            continue
        return Detected(ep, "env", f"${name}", credentials_dropped=creds, scutil=values)
    ep, _ = parse_url(DEFAULT_URL)
    return Detected(ep, "default", "no system or environment proxy found", scutil=values)


def detect_system(environ: Mapping[str, str] | None = None) -> Detected:
    """detect() on this host: ``scutil --proxy`` only exists on macOS (read-only probe, D78)."""
    text = run_probe(["/usr/sbin/scutil", "--proxy"]) if sys.platform == "darwin" else None
    return detect(text, os.environ if environ is None else environ)


def resolve(override: str | None, environ: Mapping[str, str] | None = None) -> Detected:
    """The proxy in force: the system configuration's override, else what detection finds."""
    if override:
        ep, creds = parse_url(override)
        if creds:
            raise ConfigError(
                f"the proxy override must not carry credentials; put them in secrets.env as "
                f"{USERINFO_SECRET}=user:password"
            )
        return Detected(ep, "override", "system configuration override")
    return detect_system(environ)


def no_proxy(api_url: str | None, extra: Sequence[str] = ()) -> str:
    """Loopback, the control-plane API host and any extra hosts, comma separated."""
    hosts: list[str] = list(LOOPBACK)
    if api_url:
        host = urlsplit(api_url if "://" in api_url else f"http://{api_url}").hostname
        if host:
            hosts.append(host)
    hosts += [h.strip() for h in extra if h.strip()]
    return ",".join(dict.fromkeys(hosts))


def proxy_env(ep: Endpoint, no_proxy_value: str, userinfo: str | None = None) -> dict[str, str]:
    """The variables injected into a `required` provider's process."""
    url = ep.with_userinfo(userinfo)
    out = {name: url for name in PROXY_VARS}
    out.update({name: no_proxy_value for name in NO_PROXY_VARS})
    return out


# ---- preflight -----------------------------------------------------------------------


def _connect(addr: tuple[str, int], timeout_s: float) -> socket.socket:
    return socket.create_connection(addr, timeout=timeout_s)


def _recv_until(sock: socket.socket, marker: bytes, limit: int = 65536) -> bytes:
    buf = b""
    while marker not in buf and len(buf) < limit:
        chunk = sock.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf


def _http_connect(sock: socket.socket, host: str, port: int, userinfo: str | None) -> None:
    lines = [f"CONNECT {host}:{port} HTTP/1.1", f"Host: {host}:{port}"]
    if userinfo:
        token = base64.b64encode(userinfo.encode("utf-8")).decode("ascii")
        lines.append(f"Proxy-Authorization: Basic {token}")
    sock.sendall(("\r\n".join(lines) + "\r\n\r\n").encode("ascii"))
    head = _recv_until(sock, b"\r\n\r\n").split(b"\r\n", 1)[0].decode("latin-1")
    parts = head.split()
    if len(parts) < 2 or parts[1] != "200":
        raise OSError(f"CONNECT {host}:{port} answered {head or 'nothing'!r}")


def _socks5_connect(sock: socket.socket, host: str, port: int, userinfo: str | None) -> None:
    methods = b"\x00\x02" if userinfo else b"\x00"
    sock.sendall(b"\x05" + bytes([len(methods)]) + methods)
    reply = sock.recv(2)
    if len(reply) < 2 or reply[0] != 5 or reply[1] == 0xFF:
        raise OSError(f"SOCKS5 greeting refused: {reply!r}")
    if reply[1] == 2:
        user, _, password = (userinfo or "").partition(":")
        u, p = user.encode(), password.encode()
        sock.sendall(b"\x01" + bytes([len(u)]) + u + bytes([len(p)]) + p)
        auth = sock.recv(2)
        if len(auth) < 2 or auth[1] != 0:
            raise OSError("SOCKS5 authentication refused")
    name = host.encode("idna")
    sock.sendall(b"\x05\x01\x00\x03" + bytes([len(name)]) + name + struct.pack(">H", port))
    head = sock.recv(4)
    if len(head) < 4 or head[1] != 0:
        raise OSError(f"SOCKS5 CONNECT {host}:{port} failed (reply {head!r})")
    rest = {1: 4, 4: 16}.get(head[3])
    length = rest if rest is not None else sock.recv(1)[0]
    sock.recv(length + 2)


def _split_host(target: str) -> tuple[str, int]:
    host, _, port = target.rpartition(":")
    if not host or not port.isdigit():
        return target, 443
    return host, int(port)


def check_host(
    ep: Endpoint,
    target: str,
    userinfo: str | None = None,
    timeout_s: float = PREFLIGHT_TIMEOUT_S,
    connect: Connect = _connect,
) -> dict[str, Any]:
    """CONNECT through the proxy to ``host:port`` and complete a TLS handshake."""
    host, port = _split_host(target)
    start = time.monotonic()
    row: dict[str, Any] = {"host": host, "port": port, "ok": False}
    try:
        with connect((ep.host, ep.port), timeout_s) as sock:
            sock.settimeout(timeout_s)
            if ep.scheme == "http":
                _http_connect(sock, host, port, userinfo)
            else:
                _socks5_connect(sock, host, port, userinfo)
            row["tunnel"] = "ok"
            tls = ssl.create_default_context().wrap_socket(sock, server_hostname=host)
            row["tls"] = tls.version()
            tls.close()
        row["ok"] = True
    except (OSError, ssl.SSLError, ValueError, IndexError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["ms"] = round((time.monotonic() - start) * 1000, 1)
    return row


def _with_retry(
    ep: Endpoint, target: str, userinfo: str | None, timeout_s: float, connect: Connect
) -> dict[str, Any]:
    """One retry per host: a proxy may reset a single tunnel (seen once with Clash, D584)."""
    row = check_host(ep, target, userinfo, timeout_s, connect)
    if row["ok"]:
        return row
    again = check_host(ep, target, userinfo, timeout_s, connect)
    again["attempts"] = 2
    if not again["ok"]:
        again["first_error"] = row.get("error")
    return again


def preflight(
    ep: Endpoint,
    hosts: Sequence[str],
    userinfo: str | None = None,
    timeout_s: float = PREFLIGHT_TIMEOUT_S,
    connect: Connect = _connect,
) -> dict[str, Any]:
    """Port reachable, then a tunnel and TLS handshake to every host; ``ok`` only if all pass.

    No API request is sent (no model call, no token spend). An empty host list is refused:
    a `required` provider with nothing to check could not prove the proxy works.
    """
    out: dict[str, Any] = {"proxy": ep.to_dict(), "hosts": [], "ok": False}
    if not hosts:
        out["error"] = "no API host to check for this provider (system configuration)"
        return out
    start = time.monotonic()
    try:
        with connect((ep.host, ep.port), timeout_s):
            pass
        out["port"] = {"ok": True, "ms": round((time.monotonic() - start) * 1000, 1)}
    except OSError as exc:
        out["port"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        out["error"] = f"proxy {ep.url} is not reachable: {exc}"
        return out
    out["hosts"] = [_with_retry(ep, h, userinfo, timeout_s, connect) for h in hosts]
    bad = [h for h in out["hosts"] if not h["ok"]]
    out["ok"] = not bad
    if bad:
        out["error"] = "; ".join(f"{h['host']}:{h['port']}: {h.get('error')}" for h in bad)
    return out
