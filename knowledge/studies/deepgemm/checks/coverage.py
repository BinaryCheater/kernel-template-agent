"Reconcile explicit source-study trace assignments to the current pinned inventory."

from __future__ import annotations

import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def main() -> None:
    inventory = json.loads((STUDY / "checks/inventory-result.json").read_text())
    coverage = json.loads((STUDY / "inputs/coverage.json").read_text())
    actual_apis = {(x["path"], x["name"], x["line"]) for x in inventory["api_registrations"]}
    assigned_apis = {(x["path"], x["name"], x["line"]) for x in coverage["apis"]}
    assert len(assigned_apis) == len(coverage["apis"]) and assigned_apis == actual_apis
    assigned_devices = {x["path"] for x in coverage["device_sources"]}
    assert len(assigned_devices) == len(coverage["device_sources"])
    assert assigned_devices == set(inventory["device_source_files"])
    assert not coverage["unassigned_sources"]
    traces = set()
    for entry in [*coverage["apis"], *coverage["device_sources"]]:
        assert entry["disposition"] == "source_traced_with_declared_implementation_obligations"
        assert entry["traces"]
        for trace in entry["traces"]:
            assert (STUDY / trace).is_file()
            traces.add(trace)
    assert (STUDY / coverage["shared_helper_trace"]).is_file()
    result = {
        "outcome": "PASS",
        "assigned_apis": len(actual_apis),
        "assigned_device_sources": len(assigned_devices),
        "trace_documents": sorted(traces),
        "boundary": (
            "Assignment completeness and links only; semantic arguments remain"
            " in traces for independent review."
        ),
    }
    (STUDY / "checks/coverage-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        f"PASS: {len(actual_apis)} API and "
        f"{len(assigned_devices)} device-source trace assignments.\n"
    )


if __name__ == "__main__":
    main()
