"""Builders for knowledge tests: the real seed, and edited copies of seed Templates."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.schemas import Actor, Record
from kta.core.schemas.knowledge import Template
from kta.knowledge import ImportResult, import_seed
from kta.store import CodeState
from kta.store.base import Store
from storehelp import T0, TEST_COMMIT

SEED = repo_root() / "knowledge" / "seed"
CODE = CodeState(TEST_COMMIT, False)
SEED_TEMPLATES = (
    "reduction_partition",
    "physical_storage",
    "async_transfer",
    "multi_buffer_rotation",
    "shared_resource_binding",
)


def seed(ctx: Context, store: Store) -> ImportResult:
    return import_seed(ctx, store, SEED, CODE)


def template_body(name: str) -> dict[str, Any]:
    data = yaml.safe_load((SEED / "templates" / f"{name}.yaml").read_text(encoding="utf-8"))
    return copy.deepcopy(data)


def template(seed_name: str, **changes: Any) -> Template:
    body = template_body(seed_name)
    body.update(changes)
    return Template.model_validate(body)


def source_id(store: Store) -> str:
    return store.put(
        Record.new(
            "Source",
            {"name": "test-source", "type": "doc", "locator": "tests", "revision": "1"},
            Actor(type="tool", name="test"),
            T0,
        )
    )


def put_template(store: Store, t: Template | dict[str, Any], sources: list[str]) -> str:
    body = t.model_dump(mode="json") if isinstance(t, Template) else t
    return store.put(
        Record.new("Template", body, Actor(type="tool", name="test"), T0, refs={"sources": sources})
    )


def put(store: Store, kind: str, body: dict[str, Any], refs: dict[str, Any] | None = None) -> str:
    return store.put(Record.new(kind, body, Actor(type="tool", name="test"), T0, refs=refs or {}))


def write_yaml(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


GEN_PY = """
def generate(assignment, case):
    if case.get("M") == -1:
        raise ValueError("negative extent")
    body = []
    for key in KEYS:
        body += [f"// kta:step {key} begin", f"// {key} code", f"// kta:step {key} end"]
    return {"files": {"kernel.cu": "\\n".join(body)}, "markers": KEYS}
"""


def gemm_chain(markers: bool = True) -> dict[str, Any]:
    items: list[dict[str, Any]] = [
        {
            "key": "k_split",
            "template": "reduction_partition@1",
            "applies_to": "sum over K",
            "bindings": {
                "num_groups": "KT",
                "group_size": "BK",
                "assignment": "contiguous_sequential",
            },
        },
        {
            "key": "ab_smem",
            "template": "physical_storage@1",
            "applies_to": "A and B tiles",
            "bindings": {"level": "smem", "buffer": "dedicated", "share_order": "none"},
            "after": ["k_split"],
        },
        {
            "key": "ab_stages",
            "template": "multi_buffer_rotation@1",
            "applies_to": "A and B k-tiles",
            "bindings": {"S": "S", "P": "P", "beta": 1},
            "after": ["ab_smem"],
        },
    ]
    for item in items:
        item["code_markers"] = [item["key"]] if markers else []
    return {"workload": "fp16 GEMM C = A @ B", "items": items}


def family_dir(root: Path, **changes: Any) -> Path:
    """A minimal family directory over the seed Templates (test data, not a real kernel)."""
    family: dict[str, Any] = {
        "name": "gemm_test",
        "version": 1,
        "backend": "cuda",
        "variables": [
            {
                "name": "KT",
                "kind": "numeric",
                "binding": "runtime",
                "role": "software",
                "domain": {"lo": 1, "hi": 4096},
                "meaning": "k-tiles",
            },
            {
                "name": "BK",
                "kind": "numeric",
                "binding": "compile",
                "role": "software",
                "domain": [16, 32, 64],
                "meaning": "k-tile size",
            },
            {
                "name": "S",
                "kind": "numeric",
                "binding": "compile",
                "role": "software",
                "domain": {"lo": 2, "hi": 4},
                "meaning": "slots",
            },
            {
                "name": "P",
                "kind": "numeric",
                "binding": "compile",
                "role": "software",
                "domain": {"lo": 0, "hi": 3},
                "meaning": "prefetch",
            },
        ],
        "glue_constraints": [
            {
                "id": "g_smem",
                "expr": "S * BK * 256 * 2 <= hw.smem_per_cta",
                "meaning": "couples slots (step 3) with the k-tile size (step 1)",
                "cls": "resource",
            }
        ],
        "edge_cases": [{"M": 1, "N": 1, "K": 1}],
        "entry": "gen.py:generate",
    }
    chain = gemm_chain()
    family.update(changes.pop("family", {}))
    chain.update(changes.pop("chain", {}))
    write_yaml(root / "family.yaml", family)
    write_yaml(root / "chain.yaml", chain)
    write_yaml(root / "edge_cases.yaml", changes.pop("edge_cases", [{"M": 1, "N": 1, "K": 1}]))
    keys = [i["key"] for i in chain["items"]]
    gen = changes.pop("gen", f"KEYS = {keys!r}\n" + GEN_PY)
    (root / "gen.py").write_text(gen, encoding="utf-8")
    (root / "README.md").write_text("Test family.\n", encoding="utf-8")
    return root
