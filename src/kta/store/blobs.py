"""Content-addressed blob store: ``<data>/objects/ab/cdef…`` (docs/design/20 §3)."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

from kta.core.errors import SchemaError

CHUNK_BYTES = 1 << 20


class BlobFiles:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, sha256: str) -> Path:
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise SchemaError(f"{sha256!r} is not a sha256")
        return self.root / sha256[:2] / sha256[2:]

    def write(self, data: bytes | Path) -> tuple[str, int]:
        """Store the content once; returns (sha256, size). Writes are atomic renames."""
        self.root.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=self.root, prefix=".tmp-")
        tmp = Path(tmp_name)
        digest = hashlib.sha256()
        size = 0
        try:
            with os.fdopen(fd, "wb") as out:
                for chunk in _chunks(data):
                    digest.update(chunk)
                    out.write(chunk)
                    size += len(chunk)
            sha = digest.hexdigest()
            dest = self.path(sha)
            if dest.exists():
                tmp.unlink()
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                os.replace(tmp, dest)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return sha, size

    def verify(self, sha256: str) -> bool:
        path = self.path(sha256)
        if not path.exists():
            return False
        digest = hashlib.sha256()
        for chunk in _chunks(path):
            digest.update(chunk)
        return digest.hexdigest() == sha256


def _chunks(data: bytes | Path) -> Iterator[bytes]:
    if isinstance(data, bytes):
        yield data
        return
    with data.open("rb") as src:
        while chunk := src.read(CHUNK_BYTES):
            yield chunk
