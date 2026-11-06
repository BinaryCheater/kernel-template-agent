"""Freeze one session's workspace. Only the information arm differs; gold stays out."""

import json
import shutil
from pathlib import Path

from tools.template_chain.tasks import TASKS

from template_access.store import Library
from template_recall.gold import ROOT, load_task
from template_recall.package import access_arm, flat_arm

ASSETS = Path(__file__).with_name("assets")
MAX_CHECKS = 30
WALL_S = 2700

COMMON = f"""Read TASK.md, FORM.md and ENVIRONMENT.md. Write family.py and form.py as FORM.md
describes, for the kernel in TASK.md. The aim is the fastest correct kernel the search can
find, so put the real design choices in FACTORS and model what they cost in form.py. Use
./chain check '<json assignment>' to compile and validate; you have {MAX_CHECKS} checks and
no timing. Finish when ANCHOR and at least two other assignments pass, then write DESIGN.md:
the structure you chose, why, what you expect to dominate time, and which choices came from
where. Work only in this workspace. Do not read the installed libraries' attention
implementations, other runs or files outside this workspace, and start no other agent.
"""

ARMS = {
    "none": "",
    "prescribed": "\nlibrary/ holds optimisation rules chosen for this kernel. Read them, use "
    "those that fit, and in DESIGN.md name each rule you used and each choice from your own "
    "knowledge. Write in NEEDS.md any need no rule met.\n",
    "selection": "\nA library of optimisation rules is available through "
    ".agents/skills/template-access/SKILL.md. Read that skill and use ./template-access for "
    "all library reading. Record what you apply with ./template-access select and unmet needs "
    "with ./template-access need. In DESIGN.md name each rule you used and each choice from "
    "your own knowledge.\n",
}

ENVIRONMENT = """# Environment

Checks run in a container on an NVIDIA GB10 (sm_121, 48 SMs, 2.5 GHz, 24 MB L2, 102400 bytes
of shared memory per block limit in use here) with Torch 2.11, TileLang 0.1.15 and CUDA 13.
TileLang is not installed here: read docs/tilelang-api.md, docs/torch-runtime-api.md and
example/. `prepare` may only use torch and tilelang, and may import other .py files you keep
in this workspace. `run()` must use only the GPU and work under CUDA graph capture: no Python
reads of tensor values, no host control flow on device data.

The kernels are JIT compiled by TileLang: expect each check to take a minute or so.
"""

CHAIN = """#!/bin/sh
here="$(cd "$(dirname "$0")" && pwd)"
cd "{root}" && exec "{python}" -m tools.template_chain chain-check "$here" "$@"
"""


def build(workspace: Path, task_name: str, arm: str) -> Path:
    task = TASKS[task_name]
    workspace.mkdir(parents=True)
    (workspace / "TASK.md").write_text(task["statement"])
    (workspace / "ENVIRONMENT.md").write_text(ENVIRONMENT)
    for name in ("FORM.md", "form_api.py"):
        shutil.copyfile(ASSETS / name, workspace / name)
    for name in ("docs", "example"):
        shutil.copytree(ASSETS / name, workspace / name)
    (workspace / "state.json").write_text(json.dumps({"task": task_name, "checks": 0}))
    chain = workspace / "chain"
    chain.write_text(CHAIN.format(root=str(ROOT), python=str(ROOT / ".venv/bin/python")))
    chain.chmod(0o755)
    gold = load_task(task["gold"])
    library = Library(gold.source_study)
    if arm == "selection":
        access_arm(workspace, library)
    elif arm == "prescribed":
        keep = sorted(
            {r for m in gold.moves for r in m.accept | m.equivalent} & set(library.by_identity)
        )
        narrow = Library(gold.source_study)
        narrow.entries = [e for e in library.entries if e["identity"] in keep]
        narrow.by_identity = {i: library.by_identity[i] for i in keep}
        flat_arm(workspace, narrow)
    path = workspace.parent / "prompt.md"
    path.write_text(COMMON + ARMS[arm])
    return path
