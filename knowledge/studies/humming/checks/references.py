"Validate candidate identities, chains, domains, Gap bodies and frozen non-Humming inputs."

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from kta.core.expr import Expr
from kta.core.schemas.knowledge import Chain, Gap, RangeSpec, Realisation, Template

STUDY = Path(__file__).resolve().parents[1]


def main() -> None:
    snapshot = json.loads((STUDY / "inputs/pool-snapshot.json").read_text())
    archive = Path(snapshot["archive_root"])
    for row in [*snapshot["files"], *snapshot["navigation"]]:
        assert hashlib.sha256((archive / row["path"]).read_bytes()).hexdigest() == row["sha256"]
    templates = {}
    for row in snapshot["templates"]:
        template = Template.model_validate(yaml.safe_load((archive / row["path"]).read_text()))
        templates[f"{template.name}@{template.version}"] = template
    new = []
    for path in sorted((STUDY / "delta").glob("*/template.yaml")):
        template = Template.model_validate(yaml.safe_load(path.read_text()))
        key = f"{template.name}@{template.version}"
        assert key not in templates, f"Identity collision: {key}"
        templates[key] = template
        new.append(key)
    chains = []
    for path in sorted((STUDY / "delta").glob("*chain.yaml")):
        chain = Chain.model_validate(yaml.safe_load(path.read_text()))
        seen = set()
        for item in chain.items:
            assert item.key not in seen
            assert set(item.after) <= seen, f"Forward or missing dependency: {item.key}"
            variables = {v.name: v for v in templates[item.template].variables}
            assert set(item.bindings) == set(variables), item.key
            for name, binding in item.bindings.items():
                domain = variables[name].domain
                if isinstance(domain, RangeSpec):
                    assert isinstance(binding, int) and domain.lo <= binding <= domain.hi
                    assert (binding - domain.lo) % domain.step == 0
                    assert not domain.pow2 or (binding > 0 and binding & (binding - 1) == 0)
                else:
                    assert binding in domain, (item.key, name, binding)
            seen.add(item.key)
        chains.append({"file": path.name, "steps": len(seen)})
    for path in (STUDY / "delta/realisations").glob("*.yaml"):
        realisation = Realisation.model_validate(yaml.safe_load(path.read_text()))
        assert realisation.template in templates
    gaps = yaml.safe_load((STUDY / "delta/gaps.yaml").read_text())
    for row in gaps:
        Gap.model_validate(row["body"])
        assert row["disposition"] == "open" and row["next_evidence"]
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    for row in [*manifest["vllm"]["files"], *manifest["documentation"]]:
        assert hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest() == row["sha256"]
    result = {
        "outcome": "PASS",
        "new_identities": new,
        "chains": chains,
        "gap_bodies": len(gaps),
        "frozen_inputs": "PASS",
        "boundary": (
            "References and literal variable domains; partial chain obligations remain open"
        ),
    }
    (STUDY / "checks/references-result.json").write_text(json.dumps(result, indent=2) + "\n")


def check_constraints() -> None:
    common = {
        "length": 128,
        "min_code": 0,
        "max_code": 15,
        "partial_count": 4,
        "reassociation_allowed": True,
        "atomic_supported": True,
        "lock_initial": 0,
        "no_overlapping_generation": True,
        "normalized_hadamard": True,
        "group_size": 128,
        "code_bits": 4,
        "zero_point": "integer",
        "finite_input": True,
        "affine_quantizer_requested": True,
        "scale_nonzero_finite": True,
    }
    witnesses = {
        "bitstream_packing@1": ({"code_bits": 4, "word_bits": 32}, {"length": 127}),
        "first_writer_atomic_combine@1": (
            {"partials": 4, "update": "packed_atomic_add"},
            {"partial_count": 5},
        ),
        "butterfly_linear_transform@1": (
            {"H": 128, "V": 4, "threads": 32},
            {"normalized_hadamard": False},
        ),
        "groupwise_affine_quantization@1": (
            {"group_size": 128, "code_bits": 4, "zero_point": "integer"},
            {"zero_point": "floating"},
        ),
    }
    outcomes = []
    for key, (binding, invalid) in witnesses.items():
        template = Template.model_validate(
            yaml.safe_load((STUDY / f"delta/{key}/template.yaml").read_text())
        )
        expressions = [
            Expr.parse(e.expr, set(binding))
            for e in [*template.constraints, *template.preconditions]
        ]
        assert all(e.eval(binding, p=common) for e in expressions)
        assert not all(e.eval(binding, p=common | invalid) for e in expressions)
        outcomes.append(
            {
                "template": key,
                "valid_binding": "accepted",
                "invalid_premise": invalid,
                "invalid_binding": "rejected",
            }
        )
    (STUDY / "checks/constraints-result.json").write_text(
        json.dumps({"outcome": "PASS", "rows": outcomes}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
    check_constraints()
