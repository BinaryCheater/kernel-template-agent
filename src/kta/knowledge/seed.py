"""Seed import: the functions behind ``kta init`` (spec/workplan.md WP4).

Layout of the seed directory (decision D204); every YAML file holds one record body:

    sources/*.yaml          Source          (cited by every seed Template)
    patterns/*.yaml         Pattern
    structures/*.yaml       Structure
    capabilities/*.yaml     Capability      refs.sources = all seed Sources (D442)
    targets/<t>/*.yaml      CapabilityDecl  one per (target, capability) (D442)
    templates/*.yaml        Template        refs.sources = all seed Sources
    realisations/*.yaml     Realisation     refs.template by name@version (D443)
    references/<ref>/       one decomposed reference (WP10a layout, refs implied, D100);
                            its realisations/ as above

A batch is validated as a whole against the store plus the batch itself, and written in
one ``put_many`` only when it has no error, so a bad batch writes nothing. Records whose
identity key (name, name@version, ...) already exists with the same content are reused,
so importing twice writes nothing new; the same key with different content is an error.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import yaml

from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.errors import ConfigError, SchemaError
from kta.core.ids import record_id
from kta.core.schemas import Actor, InvalidBody, Record, RefValue
from kta.core.schemas.hardware import HwFeature
from kta.core.schemas.knowledge import Chain, Realisation, Template
from kta.knowledge.chains import validate_chain
from kta.knowledge.pending import PendingView
from kta.knowledge.pool import names as pool_names
from kta.knowledge.realisations import validate_realisation
from kta.knowledge.templates import validate_template
from kta.knowledge.types import Issue, errors_of
from kta.store import CodeState, code_state
from kta.store.base import Store

# Fields that identify a record of each kind for reuse and conflict detection.
IDENTITY: dict[str, tuple[str, ...]] = {
    "Source": ("name",),
    "Problem": ("name",),
    "Reference": ("name",),
    "Pattern": ("name",),
    "Structure": ("name",),
    "Template": ("name", "version"),
    "Capability": ("name",),
    "CapabilityDecl": ("target", "capability"),
    "Realisation": ("template", "backend"),
    "Gap": ("description",),
    "Chain": ("workload",),
    "Reconstruction": ("notes",),
}
_ORDER_LINE = re.compile(r"^\s*\d+\.\s+`([^`]+)`")


class SeedError(SchemaError):
    def __init__(self, batch: str, issues: list[Issue]) -> None:
        self.issues = issues
        lines = "\n  ".join(str(i) for i in issues[:50])
        super().__init__(f"seed batch {batch!r} not imported ({len(issues)} issue(s)):\n  {lines}")


@dataclass
class ImportResult:
    batch: str
    written: list[str] = field(default_factory=lambda: list[str]())
    reused: list[str] = field(default_factory=lambda: list[str]())
    warnings: list[Issue] = field(default_factory=lambda: list[Issue]())
    counts: dict[str, int] = field(default_factory=lambda: dict[str, int]())


def seed_dir(ctx: Context) -> Path:
    """``[knowledge] seed_dir`` from the config, relative to the source tree root."""
    value = ctx.config.section("knowledge").get("seed_dir")
    if not isinstance(value, str):
        raise ConfigError("config [knowledge] seed_dir is not set")
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path


def _body(path: Path) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SchemaError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: expected one mapping (a record body)")
    return cast(dict[str, Any], data)


def _yamls(directory: Path) -> list[Path]:
    return (
        sorted([*directory.glob("*.yaml"), *directory.glob("*.yml")]) if directory.is_dir() else []
    )


class _Batch:
    """Builds records in write order, reusing identical ones already in the store."""

    def __init__(self, name: str, store: Store, actor: Actor, now: datetime) -> None:
        self.name, self.store, self.actor, self.now = name, store, actor, now
        self.records: list[Record] = []
        self.paths: dict[str, Path] = {}  # pending record ID -> file
        self.result = ImportResult(name)
        self.issues: list[Issue] = []

    def _match(self, rec: Record) -> list[Record]:
        where = {k: rec.body[k] for k in IDENTITY[rec.kind]}
        pending = [
            r
            for r in self.records
            if r.kind == rec.kind and all(r.body[k] == v for k, v in where.items())
        ]
        return self.store.query(rec.kind, where=where, limit=10) + pending

    def add(self, kind: str, path: Path, refs: Mapping[str, RefValue] | None = None) -> str | None:
        try:
            rec = Record.new(kind, _body(path), self.actor, self.now, refs=dict(refs or {}))
        except (InvalidBody, SchemaError) as exc:
            self.issues.append(Issue("invalid_record", str(exc).splitlines()[0][:400], str(path)))
            return None
        for other in self._match(rec):
            if other.body == rec.body and other.refs == rec.refs:
                rid = record_id(other)
                if rid not in self.paths:
                    self.result.reused.append(rid)
                return rid
            key = ", ".join(f"{k}={rec.body[k]!r}" for k in IDENTITY[kind])
            self.issues.append(
                Issue("conflict", f"{kind} {key} exists with other content", str(path))
            )
            return None
        rid = record_id(rec)
        self.records.append(rec)
        self.paths[rid] = path
        self.result.counts[kind] = self.result.counts.get(kind, 0) + 1
        return rid

    def validate(self) -> None:
        view = PendingView(self.store, self.records)
        for rid, rec in view.pending.items():
            where = str(self.paths[rid])
            found: list[Issue] = []
            if rec.kind == "Template":
                found = validate_template(Template.model_validate(rec.body), view, None)
            elif rec.kind == "Chain":
                found = validate_chain(Chain.model_validate(rec.body), view)
            elif rec.kind == "Realisation":
                found = validate_realisation(Realisation.model_validate(rec.body), view)
            for i in found:
                issue = Issue(i.kind, i.message, f"{where}: {i.where}", i.severity)
                (self.issues if i.severity == "error" else self.result.warnings).append(issue)

    def commit(self) -> ImportResult:
        self.validate()
        if errors_of(self.issues):
            raise SeedError(self.name, self.issues)
        self.result.written = self.store.put_many(self.records) if self.records else []
        return self.result


def _actor(code: CodeState) -> Actor:
    return Actor(type="import", name="kta init", commit=code.commit, dirty=code.dirty)


def import_seed(
    ctx: Context, store: Store, root: Path, code: CodeState | None = None
) -> ImportResult:
    """Import Sources, Patterns, Structures and Templates of the seed (one batch)."""
    batch = _Batch("seed", store, _actor(code or code_state()), ctx.clock.now())
    sources = [rid for p in _yamls(root / "sources") if (rid := batch.add("Source", p))]
    for p in _yamls(root / "patterns"):
        batch.add("Pattern", p)
    for p in _yamls(root / "structures"):
        batch.add("Structure", p)
    src: dict[str, RefValue] = {"sources": sources} if sources else {}
    for p in _yamls(root / "capabilities"):
        batch.add("Capability", p, src)
    targets = root / "targets"
    for d in sorted(targets.iterdir()) if targets.is_dir() else []:
        for p in _yamls(d):
            batch.add("CapabilityDecl", p, src)
    for p in _yamls(root / "templates"):
        batch.add("Template", p, {"sources": sources})
    _add_realisations(batch, root / "realisations", sources)
    return batch.commit()


def _add_realisations(batch: _Batch, directory: Path, sources: list[str]) -> None:
    """Realisations reference their Template (``template: name@version``) by record ID."""
    for p in _yamls(directory):
        refs: dict[str, RefValue] = {"sources": sources} if sources else {}
        tid = _template_id(batch, str(_body(p).get("template", "")))
        if tid is None:
            batch.issues.append(Issue("unknown_template", "Template not in the store", str(p)))
            continue
        batch.add("Realisation", p, {**refs, "template": tid})


def _template_id(batch: _Batch, ref: str) -> str | None:
    name, _, version = ref.partition("@")
    view = PendingView(batch.store, batch.records)
    found = view.query(
        "Template", {"name": name, "version": int(version) if version.isdigit() else -1}
    )
    return record_id(found[-1]) if found else None


def reference_order(references: Path) -> list[Path]:
    """Reference directories in pool order: the numbered list in references/README.md,
    then any other directory by name (the order is the discovery-curve order, WP10a)."""
    dirs = (
        {p.name: p for p in sorted(references.iterdir()) if p.is_dir()}
        if references.is_dir()
        else {}
    )
    readme = references / "README.md"
    listed: list[str] = []
    if readme.is_file():
        for line in readme.read_text(encoding="utf-8").splitlines():
            m = _ORDER_LINE.match(line)
            if m and m.group(1) in dirs and m.group(1) not in listed:
                listed.append(m.group(1))
    return [dirs[n] for n in listed] + [p for n, p in dirs.items() if n not in listed]


def _chain_refs(batch: _Batch, path: Path) -> dict[str, RefValue]:
    """refs.templates of a chain: the record ID of every distinct Template it names."""
    view = PendingView(batch.store, batch.records)
    ids: list[str] = []
    for item in _body(path).get("items", []):
        name, _, version = str(item.get("template", "")).partition("@")
        found = view.query(
            "Template", {"name": name, "version": int(version) if version.isdigit() else -1}
        )
        if found:
            ids.append(record_id(found[-1]))
    return {"templates": sorted(set(ids))} if ids else {}


def import_hw_catalog(
    ctx: Context, store: Store, features: list[HwFeature], code: CodeState | None = None
) -> ImportResult:
    """Write declared HwFeature records for names the store does not know yet.

    HwFeatures are ISA instruction families: Realisations name them in ``requires`` and
    Capabilities in ``realised_by`` (D442, D443). Any HwFeature with the same name
    (declared earlier or probed) already serves, so those names are skipped.
    """
    actor = _actor(code or code_state())
    result = ImportResult("hw_catalog")
    known = pool_names(store, "HwFeature")
    records = [
        Record.new("HwFeature", f.model_dump(), actor, ctx.clock.now())
        for f in features
        if f.name not in known
    ]
    if records:
        result.counts["HwFeature"] = len(records)
    result.written = store.put_many(records) if records else []
    return result


def import_references(
    ctx: Context, store: Store, references: Path, code: CodeState | None = None
) -> ImportResult:
    """Import every reference decomposition (WP10a layout) as one batch, in pool order."""
    batch = _Batch("references", store, _actor(code or code_state()), ctx.clock.now())
    refs = reference_order(references)
    per: dict[Path, dict[str, Any]] = {}
    for ref in refs:  # phase 1: everything that refers to nothing in other references
        source = batch.add("Source", ref / "source.yaml")
        problem = batch.add("Problem", ref / "problem.yaml")
        reference = batch.add(
            "Reference",
            ref / "reference.yaml",
            {k: v for k, v in (("source", source), ("problem", problem)) if v},
        )
        for p in _yamls(ref / "patterns"):
            batch.add("Pattern", p)
        gaps = [rid for p in _yamls(ref / "gaps") if (rid := batch.add("Gap", p))]
        per[ref] = {"source": source, "reference": reference, "gaps": gaps}
    for ref in refs:  # phase 2: Templates (Patterns of all references exist now)
        for p in _yamls(ref / "templates"):
            batch.add(
                "Template", p, {"sources": [per[ref]["source"]]} if per[ref]["source"] else {}
            )
    for ref in refs:  # phase 2b: Realisations of those Templates (D443)
        _add_realisations(
            batch, ref / "realisations", [per[ref]["source"]] if per[ref]["source"] else []
        )
    for ref in refs:  # phase 3: chains and reconstructions, in pool order
        chain = batch.add("Chain", ref / "chain.yaml", _chain_refs(batch, ref / "chain.yaml"))
        links: dict[str, RefValue] = {"gaps": per[ref]["gaps"]} if per[ref]["gaps"] else {}
        links |= {k: v for k, v in (("reference", per[ref]["reference"]), ("chain", chain)) if v}
        batch.add("Reconstruction", ref / "reconstruction.yaml", links)
    return batch.commit()
