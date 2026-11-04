"Validate concrete reused identities and bindings against the pinned pool; no admission."

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

from kta.core.schemas.knowledge import Chain, Template

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[2]


def main() -> None:
    chain = Chain.model_validate(yaml.safe_load((STUDY / "delta/chain.yaml").read_text()))
    snapshot = json.loads((STUDY / "inputs/pool-snapshot.json").read_text())
    identities: dict[str, list[Template]] = {}
    for entry in snapshot["files"]:
        path = REPO / entry["path"]
        if path.suffix != ".yaml":
            continue
        body = yaml.safe_load(path.read_text())
        if isinstance(body, dict) and "level_from" in body:
            identity = f"{body.get('name')}@{body.get('version')}"
            if identity in {item.template for item in chain.items}:
                assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
                identities.setdefault(identity, []).append(Template.model_validate(body))
    previous = set()
    for item in chain.items:
        assert set(item.after) <= previous and item.key not in previous
        assert len(identities[item.template]) == 1
        template = identities[item.template][0]
        assert set(item.bindings) == {variable.name for variable in template.variables}
        for variable in template.variables:
            value, domain = item.bindings[variable.name], variable.domain
            if isinstance(domain, list):
                assert value in domain
            else:
                assert isinstance(value, int) and domain.lo <= value <= domain.hi
                assert (value - domain.lo) % domain.step == 0
        previous.add(item.key)
    result = {
        "outcome": "PASS",
        "chain_nodes": len(chain.items),
        "boundary": (
            "Pinned identities, schema and binding domains only; no premise proof or admission."
        ),
    }
    (STUDY / "checks/chain-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write("PASS: four concrete reused-rule bindings and partial-chain ordering.\n")


if __name__ == "__main__":
    main()
