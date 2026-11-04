"""Validate Worker D source-study artifacts without promotion or GPU work."""

import gzip
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from kta.core.schemas.knowledge import Template
from kta.core.schemas.proof import ProofOutline

root = Path(__file__).resolve().parents[4]
ids = defaultdict(list)
for p in (root / "knowledge/studies").glob("*/delta/*/template.yaml"):
    t = yaml.safe_load(p.read_text())
    ids[f"{t['name']}@{t['version']}"].append(p)
witnesses = {
    "fuser": ["csrc/scheduler/normalization_utils.cpp", "runtime/block_welford_outer.cu"],
    "cccl": [
        "cub/cub/agent/agent_scan.cuh",
        "cudax/include/cuda/experimental/coop/__algorithm/reduce.cuh",
    ],
    "triton": [
        "python/tutorials/06-fused-attention.py",
        (
            "python/triton_kernels/triton_kernels/tensor_details/bitmatrix_det"
            "ails/sum_bitmatrix_rows.py"
        ),
    ],
}
for lib in witnesses:
    base = root / "knowledge/studies" / lib
    proposals = list((base / "delta").glob("*/template.yaml"))
    for p in proposals:
        t = Template.model_validate(yaml.safe_load(p.read_text()))
        o = ProofOutline.model_validate(yaml.safe_load(p.with_name("proof.yaml").read_text()))
        key = f"{t.name}@{t.version}"
        assert o.template == key and p.parent.name == key, (p, o.template)
        assert len(ids[key]) == 1, (key, ids[key])
    for p in base.rglob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", p.read_text()):
            if target.startswith(("http:", "https:", "#", "mailto:")):
                continue
            target = target.split("#")[0].strip("<>")
            assert (p.parent / target).exists(), (p, target)
    manifest = json.loads((base / "inputs/sources.json").read_text())
    tree = Path(manifest["tree"])
    archive = Path(manifest["archive"])
    assert tree.is_dir() and archive.is_file()
    with gzip.open(archive, "rb") as f:
        assert len(f.read(512)) == 512
    for rel in witnesses[lib]:
        p = tree / rel
        assert p.exists() and len(p.read_bytes()) > 500, p
    pins = json.loads((base / "inputs/pool.json").read_text())["identities"]
    for key, entry in pins.items():
        p = root / entry["retained_template"]
        t = Template.model_validate(yaml.safe_load(p.read_text()))
        assert key == f"{t.name}@{t.version}"
    references = set()
    for p in list(base.glob("*.md")) + list((base / "delta").glob("*/template.yaml")):
        references.update(re.findall(r"\b([a-z][a-z0-9_]*@[0-9]+)\b", p.read_text()))
    assert references <= set(pins), references - set(pins)
    for script in ["replay.py", "design_witnesses.py"] + (
        ["production_witnesses.py"] if lib == "triton" else []
    ):
        r = subprocess.run(
            [str(root / ".venv/bin/python"), str(base / "checks" / script)],
            capture_output=True,
            text=True,
            cwd=root,
        )
        assert r.returncode == 0, (script, r.stderr)
        sys.stdout.write(" ".join(str(value) for value in [lib, r.stdout.strip()]) + "\n")
    result = (
        f"PASS: {lib}: {len(proposals)} candidate/proof pairs; "
        f"unique proposal identities; {len(pins)} retained version pins; "
        "Markdown links; readable source/archive and representative "
        "source bodies; semantic witness scripts."
    )
    (base / "checks/validation.md").write_text(
        "# Final artifact validation\n\n"
        + result
        + (
            "\n\nCommand: `.venv/bin/python knowledge/studies/triton/checks/vali"
            "date_artifacts.py`.\n\nCandidate/outline and retained Template bodi"
            "es passed repository schema models. Source coverage is reconciled"
            " in inventory.md and source traces. Witness outputs are in result"
            "s.md. Native realization obligations remain in outstanding.md. Th"
            "is source study performed no WP gate, commit, GPU experiment or S"
            "tore admission.\n"
        )
    )
    sys.stdout.write(" ".join(str(value) for value in [result]) + "\n")
