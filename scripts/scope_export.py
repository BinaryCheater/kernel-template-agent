"""Runs this checkout's timeline tool against the engine on PYTHONPATH (scripts/scope-export).
At module level so the sensitivity workers, which import this file again, find the tool too."""

import importlib.util
import sys
from pathlib import Path

import kta_perfmodel

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "kta_perfmodel"
for name in ("atlas", "timeline"):
    spec = importlib.util.spec_from_file_location(f"kta_perfmodel.{name}", TOOLS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    setattr(kta_perfmodel, name, mod)
    spec.loader.exec_module(mod)

if __name__ == "__main__":
    sys.modules["kta_perfmodel.timeline"].main()
