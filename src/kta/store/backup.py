"""Store backups and the hand-over between control planes (WP23, D601, D609).

The store follows the control plane: one control plane writes a store at a time. Moving
the control plane between modes (dev on the Mac, long-run on spark-hk) is

1. ``kta store backup <dir> --handover <to>`` on the old side: a consistent copy of the
   database (SQLite online backup), the blobs and the system configuration, plus a
   ``manifest.json`` with hashes; ``--handover`` also leaves ``HANDOVER.json`` in the old
   data directory, after which the API and the processes there refuse to start;
2. copy the backup directory to the new side (rsync over ssh);
3. ``kta store restore <dir>`` there, into an empty data directory (it refuses to
   overwrite a store), which checks the hashes.

Taking the store back is the same three steps in the other direction; the old data
directory is never overwritten, so the history on both sides stays as it was. A plain
backup (without ``--handover``) is the periodic backup and changes nothing.

Blobs are copied incrementally: a blob already in the destination with the same name is
kept (blob names are their sha256, so equal names are equal content).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from kta.core.errors import ConfigError
from kta.store import DB_NAME

MANIFEST, HANDOVER = "manifest.json", "HANDOVER.json"
SYSTEM = Path("system") / "config.json"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_objects(src: Path, dst: Path) -> tuple[int, int]:
    copied = total = 0
    if not src.is_dir():
        return 0, 0
    for p in sorted(src.rglob("*")):
        if not p.is_file():
            continue
        total += 1
        target = dst / p.relative_to(src)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        copied += 1
    return copied, total


def backup(
    data_dir: Path, dest: Path, now: datetime, handover_to: str | None = None
) -> dict[str, Any]:
    db = data_dir / DB_NAME
    if not db.is_file():
        raise ConfigError(f"no store at {data_dir}")
    if (data_dir / HANDOVER).exists():
        raise ConfigError(
            f"{data_dir} was already handed over: {(data_dir / HANDOVER).read_text()}"
        )
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / DB_NAME).exists():
        raise ConfigError(f"{dest} already holds a backup; use a new directory")
    src = sqlite3.connect(db)
    out = sqlite3.connect(dest / DB_NAME)
    try:
        src.backup(out)  # consistent even while the API writes
    finally:
        out.close()
        src.close()
    copied, total = _copy_objects(data_dir / "objects", dest / "objects")
    if (data_dir / SYSTEM).is_file():
        (dest / SYSTEM).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_dir / SYSTEM, dest / SYSTEM)
    manifest = {
        "source_host": socket.gethostname(),
        "source_data_dir": str(data_dir),
        "created_at": now.isoformat(),
        "db_sha256": _sha(dest / DB_NAME),
        "blobs": total,
        "blobs_copied": copied,
        "system_config": (dest / SYSTEM).is_file(),
        "handover_to": handover_to,
    }
    (dest / MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    if handover_to:
        note = {"to": handover_to, "at": now.isoformat(), "backup": str(dest), **manifest}
        (data_dir / HANDOVER).write_text(json.dumps(note, sort_keys=True), encoding="utf-8")
    return manifest


def restore(src: Path, data_dir: Path) -> dict[str, Any]:
    manifest = json.loads((src / MANIFEST).read_text(encoding="utf-8"))
    if _sha(src / DB_NAME) != manifest["db_sha256"]:
        raise ConfigError(f"{src / DB_NAME} does not match its manifest hash")
    if (data_dir / DB_NAME).exists():
        raise ConfigError(f"{data_dir} already holds a store; restore only into an empty one")
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / DB_NAME, data_dir / DB_NAME)
    copied, total = _copy_objects(src / "objects", data_dir / "objects")
    if total != manifest["blobs"]:
        raise ConfigError(f"backup holds {total} blobs, manifest says {manifest['blobs']}")
    if (src / SYSTEM).is_file():
        (data_dir / SYSTEM).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / SYSTEM, data_dir / SYSTEM)
    return {**manifest, "restored_to": str(data_dir), "blobs_restored": copied}


def check_owner(data_dir: Path) -> None:
    """Refuse to run a writer on a store that was handed to another control plane."""
    note = data_dir / HANDOVER
    if note.exists():
        info = json.loads(note.read_text(encoding="utf-8"))
        raise ConfigError(
            f"this store was handed over to {info.get('to')} at {info.get('at')}; run the "
            f"control plane there, or take the store back (kta store backup/restore) and "
            f"remove {note}"
        )
