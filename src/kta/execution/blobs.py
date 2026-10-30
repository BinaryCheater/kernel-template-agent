"""Blob transfer between the control plane and workers, verified by sha256.

Blobs move as a framed byte stream: for each blob a header line ``<sha256> <size>\\n``
followed by exactly ``size`` bytes, and a final ``end\\n``. The receiver recomputes every
hash and refuses a mismatch, so a truncated or corrupted transfer never enters a store or
a worker cache.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import BinaryIO

from kta.core.errors import ExecutionError
from kta.store.blobs import BlobFiles

END = b"end\n"
MAX_HEADER = 200


def frame(items: Iterable[tuple[str, bytes]]) -> bytes:
    parts: list[bytes] = []
    for sha, data in items:
        parts += [f"{sha} {len(data)}\n".encode("ascii"), data]
    parts.append(END)
    return b"".join(parts)


def write_frames(out: BinaryIO, cache: BlobFiles, shas: Iterable[str]) -> None:
    for sha in shas:
        path = cache.path(sha)
        if not path.exists():
            raise ExecutionError("infra", f"blob {sha[:12]} is not in the cache {cache.root}")
        data = path.read_bytes()
        out.write(f"{sha} {len(data)}\n".encode("ascii"))
        out.write(data)
    out.write(END)
    out.flush()


def read_frames(stream: BinaryIO) -> Iterator[tuple[str, bytes]]:
    """Yield verified (sha256, data) pairs; raises infra errors on any inconsistency."""
    while True:
        header = stream.readline(MAX_HEADER)
        if header == END:
            return
        if not header.endswith(b"\n"):
            raise ExecutionError("infra", "blob stream ended before its end marker")
        sha, _, size_text = header.decode("ascii").strip().partition(" ")
        size = int(size_text)
        data = stream.read(size)
        if len(data) != size:
            raise ExecutionError("infra", f"blob {sha[:12]}: got {len(data)} of {size} bytes")
        if hashlib.sha256(data).hexdigest() != sha:
            raise ExecutionError("infra", f"blob {sha[:12]}: content does not match its sha256")
        yield sha, data


def parse_frames(data: bytes) -> dict[str, bytes]:
    return dict(read_frames(io.BytesIO(data)))


class WorkerCache:
    """A worker's local content-addressed blob cache (``<data>/worker/cache``)."""

    def __init__(self, root: Path) -> None:
        self.files = BlobFiles(root)

    def has(self, sha: str) -> bool:
        return self.files.path(sha).exists()

    def missing(self, shas: Iterable[str]) -> list[str]:
        return sorted({s for s in shas if not self.has(s)})

    def add(self, data: bytes | Path) -> str:
        return self.files.write(data)[0]

    def add_verified(self, sha: str, data: bytes) -> None:
        got = self.add(data)
        if got != sha:
            raise ExecutionError("infra", f"blob {sha[:12]} stored as {got[:12]}")

    def path(self, sha: str) -> Path:
        path = self.files.path(sha)
        if not path.exists():
            raise ExecutionError("infra", f"blob {sha[:12]} is not in the worker cache")
        return path
