"Validate proposed record bodies and expressions, without admission to a knowledge store."

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from kta.core.expr import Expr
from kta.core.schemas.knowledge import Chain, Realisation, Template
from kta.core.schemas.proof import ProofOutline

STUDY = Path(__file__).resolve().parents[1]


def main() -> None:
    checked = []
    for path in sorted((STUDY / "delta").rglob("template.yaml")):
        template = Template.model_validate(yaml.safe_load(path.read_text()))
        proof_path = path.with_name("proof.yaml")
        proof = ProofOutline.model_validate(yaml.safe_load(proof_path.read_text()))
        assert proof.template == f"{template.name}@{template.version}"
        assert path.parent.name == proof.template
        names = {var.name for var in template.variables}
        for expression in [*template.constraints, *template.preconditions]:
            Expr.parse(expression.expr, names)
        Expr.parse(template.scope.predicate, names)
        for assumption in proof.assumes:
            if assumption.expr:
                Expr.parse(assumption.expr, set())
        assert {m.obligation for m in proof.models} <= {o.name for o in template.obligations}
        checked.append(
            {
                "template": proof.template,
                "proof_version": proof.version,
                "schema": "PASS",
                "expression_parse": "PASS",
            }
        )
    for path in sorted((STUDY / "delta/realisations").glob("*.yaml")):
        Realisation.model_validate(yaml.safe_load(path.read_text()))
        checked.append({"realisation": path.name, "schema": "PASS"})
    chain = STUDY / "delta/chain.yaml"
    if chain.exists():
        Chain.model_validate(yaml.safe_load(chain.read_text()))
        checked.append({"chain": "schema PASS; semantic bindings checked separately"})
    result = {
        "outcome": "PASS",
        "records": checked,
        "boundary": "Record-body shape and expression syntax; no proof or admission.",
    }
    (STUDY / "checks/schema-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(f"PASS: {len(checked)} proposal units validated; no admission performed.")


if __name__ == "__main__":
    main()
    sys.stdout.write("\n")
