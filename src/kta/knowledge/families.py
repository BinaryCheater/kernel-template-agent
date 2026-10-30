"""Family directories: loading and consistency checks (spec/schemas.md §4.4, 22 §3.2)."""

from __future__ import annotations

import importlib.util
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml

from kta.core.errors import SchemaError
from kta.core.schemas.knowledge import Chain, Family, RangeSpec, Template, VarSpec
from kta.knowledge.chains import PARAM_PREFIX, resolve_templates, validate_chain
from kta.knowledge.templates import check_expr
from kta.knowledge.types import FamilyPackage, Issue
from kta.store.base import Store

FAMILY_FILES = ("family.yaml", "chain.yaml", "edge_cases.yaml", "README.md")


def _yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SchemaError(f"cannot read {path}: {exc}") from exc


def load_family(path: Path) -> FamilyPackage:
    """Parse a family directory; raises SchemaError when a file is missing or invalid."""
    missing = [f for f in FAMILY_FILES if not (path / f).is_file()]
    if missing:
        raise SchemaError(f"family directory {path} lacks {missing}")
    family = Family.model_validate(_yaml(path / "family.yaml"))
    chain = Chain.model_validate(_yaml(path / "chain.yaml"))
    edge = _yaml(path / "edge_cases.yaml") or []
    if not isinstance(edge, list) or not all(isinstance(e, dict) for e in cast(list[Any], edge)):
        raise SchemaError(f"{path}/edge_cases.yaml must be a list of mappings")
    module, sep, func = family.entry.partition(":")
    if not sep or not func:
        raise SchemaError(f"family entry {family.entry!r} is not <file>:<function>")
    gen = path / module
    if not gen.is_file():
        raise SchemaError(f"family generator {gen} does not exist")
    return FamilyPackage(
        path=path,
        family=family,
        chain=chain,
        edge_cases=cast(list[dict[str, Any]], edge),
        readme=(path / "README.md").read_text(encoding="utf-8"),
        gen_path=gen,
        gen_function=func,
    )


def default_value(v: VarSpec) -> int | str | bool:
    """The first value of a domain: list head, or the smallest admissible range value."""
    d = v.domain
    if not isinstance(d, RangeSpec):
        return d[0]
    x = d.lo
    while d.pow2 and x <= d.hi and (x <= 0 or x & (x - 1)):
        x += d.step
    return x


def _binding_names(
    fp: FamilyPackage, templates: Mapping[str, Template]
) -> tuple[set[str], list[Issue]]:
    """Family variable names the chain binds to, and bindings that name nothing."""
    fam_vars = {v.name for v in fp.family.variables}
    used: set[str] = set()
    issues: list[Issue] = []
    for item in fp.chain.items:
        t = templates.get(item.template)
        tvars = {v.name: v for v in t.variables} if t else {}
        for key, value in item.bindings.items():
            if not isinstance(value, str):
                continue
            if value in fam_vars:
                used.add(value)
                continue
            var = tvars.get(key)
            in_list = (
                var is not None and not isinstance(var.domain, RangeSpec) and value in var.domain
            )
            if not in_list and not key.startswith(PARAM_PREFIX):
                issues.append(
                    Issue(
                        "unbound_name",
                        f"{key} -> {value!r} is neither a family variable nor a domain value",
                        f"chain.items.{item.key}",
                    )
                )
    return used, issues


def _glue(fp: FamilyPackage, hw: Mapping[str, Any] | None) -> tuple[set[str], list[Issue]]:
    declared = {v.name for v in fp.family.variables}
    used: set[str] = set()
    issues: list[Issue] = []
    for g in fp.family.glue_constraints:
        if not g.meaning.strip():
            issues.append(Issue("missing_field", "glue constraint without meaning", f"glue.{g.id}"))
        expr, found = check_expr(g.expr, declared, f"glue.{g.id}", hw)
        issues += found
        if expr is not None:
            used |= expr.free_vars
    return used, issues


def _load_generator(fp: FamilyPackage) -> Any:
    spec = importlib.util.spec_from_file_location(f"kta_family_{fp.family.name}", fp.gen_path)
    if spec is None or spec.loader is None:
        raise SchemaError(f"cannot load {fp.gen_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    func = getattr(module, fp.gen_function, None)
    if not callable(func):
        raise SchemaError(f"{fp.gen_path} has no function {fp.gen_function}")
    return func


def _source_text(generated: Any) -> str:
    files = (
        generated.get("files")
        if isinstance(generated, Mapping)
        else getattr(generated, "files", None)
    )
    if not isinstance(files, Mapping):
        raise SchemaError("generate() must return a GeneratedKernel (with files: name -> source)")
    return "\n".join(str(v) for v in cast(Mapping[str, Any], files).values())


def _marker_issues(fp: FamilyPackage, text: str) -> list[Issue]:
    issues: list[Issue] = []
    for item in fp.chain.items:
        for marker in item.code_markers:
            for edge in ("begin", "end"):
                pattern = rf"kta:step\s+{re.escape(marker)}\s+{edge}\b"
                if not re.search(pattern, text):
                    issues.append(
                        Issue(
                            "missing_marker",
                            f"'kta:step {marker} {edge}' not in the generated source",
                            f"chain.items.{item.key}",
                        )
                    )
        if not item.code_markers:
            issues.append(
                Issue(
                    "missing_marker",
                    "family chain items need code_markers",
                    f"chain.items.{item.key}",
                )
            )
    return issues


def _generation(fp: FamilyPackage) -> list[Issue]:
    """Generate for the default assignment and for every edge case.

    The generator runs in this process (decision D213): it only produces source text;
    compiling and running go through the Executor.
    """
    try:
        generate = _load_generator(fp)
    except (SchemaError, OSError, SyntaxError, ImportError) as exc:
        return [Issue("generator_unloadable", str(exc), str(fp.gen_path))]
    assignment = {v.name: default_value(v) for v in fp.family.variables}
    issues: list[Issue] = []
    for i, case in enumerate([{}, *fp.edge_cases]):
        where = "default" if i == 0 else f"edge_cases[{i - 1}]"
        try:
            text = _source_text(generate(dict(assignment), dict(case)))
        except Exception as exc:  # noqa: BLE001  # kta:allowed the failure is reported as an Issue
            issues.append(Issue("generate_failed", f"{type(exc).__name__}: {exc}", where))
            continue
        if i == 0:
            issues += _marker_issues(fp, text)
    return issues


def validate_family(
    fp: FamilyPackage, store: Store, hw: Mapping[str, Any] | None = None
) -> list[Issue]:
    issues = [
        Issue(i.kind, i.message, f"chain.{i.where}", i.severity)
        for i in validate_chain(fp.chain, store)
    ]
    found, _ = resolve_templates(fp.chain, store)
    bound, binding_issues = _binding_names(fp, {ref: t for ref, (_, t) in found.items()})
    glue_vars, glue_issues = _glue(fp, hw)
    issues += binding_issues + glue_issues
    runtime = set(fp.family.dispatch.runtime_vars) if fp.family.dispatch else set()
    issues += [
        Issue(
            "extra_family_variable",
            f"{v.name} is not bound by any item nor used by glue",
            "family.variables",
        )
        for v in fp.family.variables
        if v.name not in bound | glue_vars | runtime
    ]
    if not fp.edge_cases:
        issues.append(Issue("missing_field", "a family lists its edge shapes", "edge_cases.yaml"))
    issues += _generation(fp)
    return issues
