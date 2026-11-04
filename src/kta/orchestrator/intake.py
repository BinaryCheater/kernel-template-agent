"""Freeze a Source's files for decomposition (LEARNDEMO-1, D743).

The expand loop's ``frozen`` state fetches the source files of a git Source at its
revision through the Executor (a cpu job on a worker; the control plane never clones).
The result is a ``source_tree`` Report on the Source: blob ``source`` is a gzip tar of the
tree's source files, blob ``files`` lists every path of the tree. The T2 workspace
receives it as an input, so the agent reads the frozen code under
``inputs/Report-<id>/source/``. One Report per Source: later loops reuse it.

Only source files enter the archive (``INCLUDE``): whole repositories exceed the
workspace's input limit (vLLM at 2eae56c4 is 48 MB gzip, its source files 18 MB).
"""

from __future__ import annotations

import re
import shlex
from typing import Any

from kta.core.errors import ConfigError, NotSupported
from kta.core.ids import record_id
from kta.core.jobs import Job
from kta.core.schemas import Actor, Record
from kta.orchestrator.pipeline import Services

TITLE = "source_tree"
ACTOR = Actor(type="system", name="kta.orchestrator.intake")
# git pathspecs of the files a decomposition reads (kernels, headers, Python/Triton, build).
INCLUDE = (
    "*.cu",
    "*.cuh",
    "*.h",
    "*.hpp",
    "*.inl",
    "*.cpp",
    "*.cc",
    "*.c",
    "*.py",
    "*.cmake",
    "CMakeLists.txt",
    "LICENSE*",
    "COPYING*",
)


def include_regex() -> str:
    """``INCLUDE`` as one POSIX ERE on a repository path (basename globs: ``*`` stays in a
    path segment). A pathspec that matches no file makes ``git archive`` fail, so the
    archive is given the matching file list instead."""
    names = [re.escape(p).replace(r"\*", "[^/]*") for p in INCLUDE]
    return f"(^|/)({'|'.join(names)})$"


def _script(url: str, rev: str) -> str:
    r = shlex.quote(rev)
    return "\n".join(
        [
            "set -eu",
            "cd {job}",
            "mkdir -p out",
            f"git clone -q --filter=blob:none --no-checkout {shlex.quote(url)} src",
            "cd src",
            f"git rev-parse {shlex.quote(rev + '^{commit}')} > ../out/commit.txt",
            f"git rev-parse {shlex.quote(rev + '^{tree}')} > ../out/tree.txt",
            f"git ls-tree -r --name-only {r} > ../out/files.txt",
            f"grep -E {shlex.quote(include_regex())} ../out/files.txt > ../include.txt || true",
            "set --",
            'while IFS= read -r f; do set -- "$@" "$f"; done < ../include.txt',
            f'git archive --format=tar {r} -- "$@" > ../source.tar',
            "gzip -n -c ../source.tar > ../out/source.tar.gz",
            "tar -tf ../source.tar | wc -l > ../out/included.txt",
        ]
    )


def existing(svc: Services, source_id: str) -> str | None:
    found = svc.store.query("Report", {"title": TITLE, "subject": source_id}, limit=1)
    return record_id(found[0]) if found else None


def source_tree(svc: Services, source_id: str) -> str:
    """The ``source_tree`` Report of a Source, fetched through the Executor if absent."""
    if (rid := existing(svc, source_id)) is not None:
        return rid
    src = svc.store.get(source_id).body
    if src.get("type") != "git":
        raise NotSupported(f"freezing a {src.get('type')} Source: only git sources are fetched")
    url, rev = str(src["locator"]), str(src["revision"])
    job = Job("shell", ["sh", "-c", _script(url, rev)], timeout_s=1800)
    res = svc.executor.run(job)
    if not res.ok:
        err = svc.executor.read_blob(res.stderr_blob).decode("utf-8", "replace")[-800:]
        raise ConfigError(f"fetching {url} at {rev[:12]} failed ({res.error_kind}): {err}")
    out = res.outputs

    def text(name: str) -> str:
        return svc.executor.read_blob(out[name]).decode("utf-8").strip()

    commit = text("commit.txt")
    if not commit.startswith(rev) and commit != rev:
        raise ConfigError(f"{url}: revision {rev} resolved to {commit}")
    files = text("files.txt").splitlines()
    data: dict[str, Any] = {
        "locator": url,
        "revision": commit,
        "tree": text("tree.txt"),
        "files_total": len(files),
        "files_included": int(text("included.txt") or 0),
        "include": list(INCLUDE),
        "job": res.job_id,
        "worker": res.worker_name,
    }
    body = {
        "title": TITLE,
        "subject": source_id,
        "summary": (
            f"{data['files_included']} of {len(files)} files of {url} at {commit[:12]} "
            f"(tree {data['tree'][:12]})"
        ),
        "data": data,
    }
    blobs = {"source": out["source.tar.gz"], "files": out["files.txt"]}
    rec = Record.new(
        "Report", body, ACTOR, svc.ctx.clock.now(), refs={"source": source_id}, blobs=blobs
    )
    return svc.store.put(rec)
