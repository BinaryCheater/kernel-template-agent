"""Read-only probes of local tool versions and git state.

This is the one place outside ``kta.execution`` that starts subprocesses: the doctor and
EnvManifest must work before any Executor exists, and core (layer 0) cannot import
execution (layer 3). Probes only run fixed, read-only version commands (decision D78).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

PROBE_TIMEOUT_S = 20.0


def run_probe(argv: list[str], cwd: Path | None = None, stderr_fallback: bool = True) -> str | None:
    """Output of a version command, or None when the tool is not installed or fails.

    Some tools print their version on stderr; ``stderr_fallback`` uses it when stdout is empty.
    """
    exe = argv[0] if "/" in argv[0] else shutil.which(argv[0])
    if exe is None or not Path(exe).exists():
        return None
    proc = subprocess.run(
        [exe, *argv[1:]],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=PROBE_TIMEOUT_S,
        check=False,
    )
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    if not out and stderr_fallback:
        out = proc.stderr.strip()
    return out


def git_state(root: Path) -> tuple[str, bool]:
    """(commit, dirty) of the source tree.

    Worker deployments are git archives without .git; their deploy step writes
    ``.kta-commit`` instead. Unknown provenance counts as dirty.
    """
    if (root / ".git").exists():
        commit = run_probe(["git", "-C", str(root), "rev-parse", "HEAD"], stderr_fallback=False)
        status = run_probe(["git", "-C", str(root), "status", "--porcelain"], stderr_fallback=False)
        if commit is None or status is None:
            return ("unknown", True)
        return (commit, status != "")
    marker = root / ".kta-commit"
    if marker.exists():
        return (marker.read_text(encoding="utf-8").strip(), False)
    return ("unknown", True)
