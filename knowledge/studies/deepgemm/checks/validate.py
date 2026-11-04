"Validate proposed record bodies and expressions, without admission to a knowledge store."

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import yaml

from kta.core.expr import Expr
from kta.core.schemas.knowledge import Chain, Realisation, Template
from kta.core.schemas.proof import ProofOutline

STUDY = Path(__file__).resolve().parents[1]


def check_premises() -> None:
    cases = json.loads((STUDY / "checks/premise-inputs.json").read_text())
    results = []
    for case in cases:
        path = STUDY / "delta" / case["template"] / "template.yaml"
        template = Template.model_validate(yaml.safe_load(path.read_text()))
        names = {var.name for var in template.variables}
        expressions = [*template.constraints, *template.preconditions]
        assert all(
            Expr.parse(expr.expr, names).eval(case["variables"], p=case["p"])
            for expr in expressions
        )
        rejected = []
        for mutation in case["invalid"]:
            variables = case["variables"] | mutation.get("variables", {})
            params = case["p"] | mutation.get("p", {})
            failed = [
                expr.id
                for expr in expressions
                if not Expr.parse(expr.expr, names).eval(variables, p=params)
            ]
            assert failed, mutation
            rejected.append({"mutation": mutation, "failed_conditions": failed})
        results.append(
            {
                "template": case["template"],
                "valid_binding": "PASS",
                "invalid_bindings_rejected": rejected,
            }
        )
    (STUDY / "checks/premise-result.json").write_text(json.dumps(results, indent=2) + "\n")


def resolve_template(identity: str) -> Template:
    local = STUDY / "delta" / identity / "template.yaml"
    if local.exists():
        return Template.model_validate(yaml.safe_load(local.read_text()))
    repository = STUDY.parents[2]
    files = json.loads((STUDY / "inputs/pool-snapshot.json").read_text())["files"]
    matches = []
    for item in files:
        path = repository / item["path"]
        if path.suffix != ".yaml":
            continue
        body = yaml.safe_load(path.read_text())
        if isinstance(body, dict) and "level_from" in body:
            if f"{body.get('name')}@{body.get('version')}" == identity:
                assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
                matches.append(Template.model_validate(body))
    assert len(matches) == 1, identity
    return matches[0]


def check_chain_bindings(chain: Chain) -> None:
    previous = set()
    for item in chain.items:
        assert item.key not in previous and set(item.after) <= previous
        template = resolve_template(item.template)
        assert set(item.bindings) == {variable.name for variable in template.variables}
        for variable in template.variables:
            value, domain = item.bindings[variable.name], variable.domain
            if isinstance(domain, list):
                assert value in domain
            else:
                assert isinstance(value, int) and domain.lo <= value <= domain.hi
                assert (value - domain.lo) % domain.step == 0
                if domain.pow2:
                    assert value > 0 and value & (value - 1) == 0
        previous.add(item.key)


def main() -> None:
    check_premises()
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
        realisation = Realisation.model_validate(yaml.safe_load(path.read_text()))
        target = resolve_template(realisation.template)
        names = {var.name for var in target.variables}
        for expression in realisation.isa_constraints:
            Expr.parse(expression.expr, names)
        checked.append({"realisation": path.name, "schema": "PASS"})
    for chain in sorted((STUDY / "delta").glob("*chain.yaml")):
        check_chain_bindings(Chain.model_validate(yaml.safe_load(chain.read_text())))
        checked.append(
            {
                "chain": chain.name,
                "schema": "PASS; identities, variable domains and dependency order checked",
            }
        )
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
