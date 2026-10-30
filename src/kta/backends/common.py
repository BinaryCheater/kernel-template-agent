"""Shared build steps: the input-hash cache, ptxas feedback, Build records.

Cache (decision D189): the key is the sha256 of a canonical JSON of the backend, every
source file's hash, the entry, the launch spec, the compile flags, the Target body and
the container image. It is stored as the Build's ``build_key`` blob, so a later build
with the same key finds the Build by querying ``blobs.build_key``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from kta.backends.base import BuiltModule, GeneratedKernel
from kta.core.context import Clock
from kta.core.ids import canonical_json, record_id, sha256_hex
from kta.core.jobs import JobResult, JobRunner
from kta.core.schemas import Actor, Record
from kta.core.schemas.evaluation import Build
from kta.core.schemas.hardware import Target
from kta.store.base import Store

ACTOR = Actor(type="tool", name="kta build")
LOG_SUMMARY_CHARS = 1500
_REGS = re.compile(r"Used (\d+) registers")
_SPILL = re.compile(r"(\d+) bytes spill stores, (\d+) bytes spill loads")
_SMEM = re.compile(r"(\d+) bytes smem")
_FUNC = re.compile(r"Compiling entry function '([^']+)'")


def parse_ptxas(log: str, entry: str) -> dict[str, Any]:
    """regs, spills (stores + loads bytes), smem of ``entry`` from ``ptxas -v`` output."""
    lines = log.splitlines()
    current: str | None = None
    found: dict[str, Any] = {}
    summary: list[str] = []
    for line in lines:
        if m := _FUNC.search(line):
            current = m.group(1)
        if current != entry or "ptxas info" not in line:
            continue
        summary.append(line.strip())
        if m := _REGS.search(line):
            found["regs"] = int(m.group(1))
        if m := _SMEM.search(line):
            found["smem"] = int(m.group(1))
        if m := _SPILL.search(line):
            found["spills"] = int(m.group(1)) + int(m.group(2))
    found["ptxas"] = summary
    return found


@dataclass(frozen=True)
class BuildKey:
    sha256: str
    data: bytes


def build_key(
    gk: GeneratedKernel, target: Target, flags: list[str], image: str | None, put: JobRunner
) -> BuildKey:
    files = {name: sha256_hex(text.encode("utf-8")) for name, text in sorted(gk.files.items())}
    data = canonical_json(
        {
            "backend": gk.backend,
            "files": files,
            "entry": gk.entry,
            "launch": gk.launch.to_dict(),
            "flags": flags,
            "target": target.model_dump(mode="json"),
            "image": image,
        }
    )
    sha = put.put_blob(data, "application/json")
    return BuildKey(sha, data)


def cached(store: Store, key: BuildKey) -> tuple[Build, BuiltModule] | None:
    found = store.query("Build", {"blobs.build_key": key.sha256}, order="-created_at", limit=1)
    if not found:
        return None
    return from_record(store, found[0])


def from_record(store: Store, rec: Record) -> tuple[Build, BuiltModule]:
    build = Build.model_validate(rec.body)
    entry = store.blob_path(rec.blobs["entry_name"]).read_text(encoding="utf-8")
    feedback = {"regs": build.regs, "spills": build.spills, "smem": build.smem}
    module = BuiltModule(
        sha256=rec.blobs.get("module", ""),
        kind="cubin",
        feedback=feedback,
        sass_blob=rec.blobs.get("sass"),
        entry=entry,
        build_id=record_id(rec),
    )
    return build, module


def latest_target_id(store: Store, target: Target) -> str | None:
    for rec in store.query("Target", {"name": target.name}, order="-created_at", limit=20):
        if rec.body == target.model_dump(mode="json"):
            return record_id(rec)
    return None


def write_build(
    store: Store,
    clock: Clock,
    gk: GeneratedKernel,
    target: Target,
    key: BuildKey,
    result: JobResult,
    entry: str,
    extra_blobs: dict[str, str],
    failure: str | None = None,
) -> tuple[Build, BuiltModule]:
    """The Build record for a finished compile job (``failure`` overrides success)."""
    log = store.blob_path(result.stderr_blob).read_text(encoding="utf-8", errors="replace")
    feedback = parse_ptxas(log, entry)
    cubin = result.outputs.get("module.cubin")
    ok = result.ok and cubin is not None and failure is None
    summary = failure or (log[-LOG_SUMMARY_CHARS:] if log else f"exit {result.exit_code}")
    body = Build(
        ok=ok,
        regs=feedback.get("regs"),
        spills=feedback.get("spills"),
        smem=feedback.get("smem"),
        log_summary=summary if not ok else "\n".join(feedback["ptxas"])[-LOG_SUMMARY_CHARS:],
    )
    blobs = {"build_key": key.sha256, "log": result.stderr_blob, **extra_blobs}
    blobs |= {
        f"src/{n}": store.put_blob(t.encode("utf-8"), "text/plain") for n, t in gk.files.items()
    }
    if cubin is not None:
        blobs["module"] = cubin
    if "module.sass" in result.outputs:
        blobs["sass"] = result.outputs["module.sass"]
    blobs["entry_name"] = store.put_blob(entry.encode("utf-8"), "text/plain")
    refs: dict[str, str | list[str]] = {"env": result.env_manifest, "worker": result.worker}
    if (tid := latest_target_id(store, target)) is not None:
        refs["target"] = tid
    rec = Record.new("Build", body, ACTOR, clock.now(), refs=refs, blobs=blobs)
    build_id = store.put(rec)
    module = BuiltModule(
        sha256=cubin or "",
        kind="cubin",
        feedback={k: feedback.get(k) for k in ("regs", "spills", "smem", "ptxas")},
        sass_blob=blobs.get("sass"),
        entry=entry,
        build_id=build_id,
    )
    return body, module
