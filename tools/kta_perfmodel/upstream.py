"""Upstream validation data, read at a pinned commit of the research repository (read-only).

The data are extracted once with ``git archive`` into ``<kta>/data/perfmodel/upstream-<sha12>/``
(``data/`` is git-ignored), so the table always reads one identified version of the inputs.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

UPSTREAM = Path("/Users/dingyi/Projects/ai-infra-codesign-ventus")
# upstream origin/main on 2026-11-04: the copy's commit 0eef497 plus I2FP on the ALU, the
# conversion probe (cvt.sass, cvt.txt) and grid_x; the GEMM and TMA ablation inputs are identical
UPSTREAM_REV = "a628c9d963cf9c029fa596b872d6e8920ca883e8"
CAL_PATH = "calibration/spark-6951"
ROOT = Path(__file__).resolve().parents[2]


def resolve(repo: Path, rev: str) -> str:
    out = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", f"{rev}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def calibration_dir(repo: Path = UPSTREAM, rev: str = UPSTREAM_REV) -> tuple[Path, str]:
    """The extracted calibration directory for `rev` and the full commit id."""
    sha = resolve(repo, rev)
    dest = ROOT / "data" / "perfmodel" / f"upstream-{sha[:12]}"
    cal = dest / CAL_PATH
    if not (cal / "gb10.json").exists():
        raw = subprocess.run(
            ["git", "-C", str(repo), "archive", "--format=tar", sha, CAL_PATH],
            check=True,
            capture_output=True,
        ).stdout
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
            tar.extractall(dest, filter="data")
    return cal, sha


def kta_commit() -> tuple[str, bool]:
    """HEAD of this checkout and whether tracked files differ from it."""
    head = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return head, bool(dirty)
