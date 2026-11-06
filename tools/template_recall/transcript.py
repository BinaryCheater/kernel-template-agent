"""Recover what a session saw and opened from its log and native transcript."""

from __future__ import annotations

import json
import re
from pathlib import Path

IDENTITY = re.compile(r"\b[a-z][a-z0-9_]*@\d+\b")
RULE_PATH = re.compile(r"templates/([a-z][a-z0-9_]*)/(\d+)/")
GUIDE_HEAD = re.compile(r"(?m)^# ([a-z][a-z0-9_]*@\d+):")
OUTSIDE = re.compile(r"/Users/|knowledge/|kta-data|\.\./\.\./")
BYPASS = re.compile(r"bundle\.json|index\.json|gzip\.open")


def tool_events(path: Path) -> list[tuple[str, str]]:
    """Return (kind, text) for tool calls and outputs in a native transcript."""
    rows = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        payload = record.get("payload")
        if record.get("type") != "response_item" or not isinstance(payload, dict):
            continue
        kind = payload.get("type", "")
        if kind in ("custom_tool_call", "function_call"):
            rows.append(("call", str(payload.get("input") or payload.get("arguments") or "")))
        elif kind in ("custom_tool_call_output", "function_call_output"):
            rows.append(("output", json.dumps(payload.get("output"))))
    return rows


def flat_observations(events: list[tuple[str, str]], view: set[str]) -> dict:
    surfaced, opened = set(), set()
    for kind, text in events:
        text = text.replace("\\n", "\n")
        if kind == "output":
            surfaced |= set(IDENTITY.findall(text)) & view
            opened |= set(GUIDE_HEAD.findall(text)) & view
        else:
            opened |= {f"{n}@{v}" for n, v in RULE_PATH.findall(text)} & view
    return {"surfaced": sorted(surfaced), "opened": sorted(opened)}


def access_observations(log: Path, view: set[str]) -> dict:
    surfaced, opened, selected, layers, needs = set(), set(), [], {}, []
    for line in log.read_text().splitlines():
        row = json.loads(line)
        event = row["event"]
        layers[event] = layers.get(event, 0) + 1
        if event == "search":
            surfaced |= set(row["results"])
        elif event == "card":
            surfaced.add(row["identity"])
        elif event in ("guide", "template", "proof", "realisation"):
            opened.add(row["identity"])
        elif event == "select":
            selected.append(row["identity"])
        elif event == "need":
            needs.append(row["need"])
    return {
        "surfaced": sorted(surfaced & view),
        "opened": sorted(opened & view),
        "selected": selected,
        "layer_counts": layers,
        "needs": needs,
    }


def audit(events: list[tuple[str, str]], arm: str, workspace: Path) -> dict:
    """Isolation is by instruction, so report anything that stepped around it."""
    calls = [text.replace(str(workspace), ".") for kind, text in events if kind == "call"]
    outside = [c[:200] for c in calls if OUTSIDE.search(c)]
    bypass = [c[:200] for c in calls if arm == "access" and BYPASS.search(c)]
    return {"calls": len(calls), "outside_workspace": outside, "bundle_bypass": bypass}
