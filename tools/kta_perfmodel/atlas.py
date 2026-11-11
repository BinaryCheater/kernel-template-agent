"""The mechanisms behind Scope: what the programme found, read from its own records, nothing written
by hand. `perfmodel/scope/traces/atlas.json` holds, per inventory row (29-mechanism-inventory.md),
its current state, rule text, probes and decisions; the probes' measured and replayed rows
(error-table.json); the hypotheses rejected on the way (STATE.md); the coverage matrix; and each
operation class's service rule as the engine runs it (sm.py).

A row's resources and rule code are matched from its words (RES, RULE): a reading aid, so the page
shows the match as such. Status comes from the row's own text and is as current as the inventory.
"""

from __future__ import annotations

import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

# inventory rows to resources on the map, by their words
RES = {
    "scheduler": (
        r"scheduler|warp i |dependency counter|stall count|yield|read-counter|result latency"
    ),
    "instruction_buffer": r"fetch|instruction cache|branch|divergence",
    "tensor": r"tensor|\bmma\b|hmma",
    "fma": r"\bFMA\b|FFMA|HADD2|half.rate",
    "alu": r"\bALU\b|LOP3",
    "xu": r"\bSFU\b|\bXU\b|MUFU|conversion|\bex2\b",
    "mio": r"\bMIO\b|memory-instruction slot",
    "smem_array": r"shared.memory|ldmatrix|\bLDSM\b|\bSTSM\b|\bLDS\b|\bSTS\b|SHFL|bank conflict",
    "lsu": r"\bLSU\b|address stage",
    "l1": r"\bL1\b",
    "l2_chip": r"\bL2\b|sector",
    "l2_return": r"return port|return",
    "dram": r"\bDRAM\b|\bTLB\b|burst",
    "store_port": r"\bstore|\bSTG\b|atomic|\bRED\b",
    "tma_engine": r"\bTMA\b|mbarrier",
    "chip": r"launch|clock|residency|dispatch|block replacement|occupancy|turnover",
    "barrier": r"barrier",
}
# operation classes' service rules to inventory rows, by their words
RULE = {
    "ldsm": r"ldmatrix|\bLDSM\b",
    "stsm": r"\bSTSM\b",
    "smem_op": r"\bLDS\b|\bSTS\b|SHFL|shared loads",
    "cp_async": r"cp\.async|LDGSTS|\bcop(y|ies)\b",
    "ldg": r"\bLDG\b|global load|plain load|\bL1\b|\bL2\b.*read",
    "stg": r"\bSTG\b|\bstores?\b",
    "red": r"atomic|\bRED\b|reduction",
    "local": r"\bLDL\b|\bSTL\b|\bstack\b",
    "tma_store": r"TMA store|UTMASTG",
    "barrier": r"barrier",
    "pipe": r"tensor|\bFMA\b|\bALU\b|MUFU|\bpipe\b",
}
CODE = re.compile(r"(?:^|[\s(:;,])([MAPXF])(?=[\s:;,.)]|$)")


def _cells(line: str) -> list[str]:
    return [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def inventory(text: str) -> list[dict[str, Any]]:
    """Rows of sections 1 to 7."""
    out: list[dict[str, Any]] = []
    section = ""
    head: list[str] = []
    for line in text.splitlines():
        m = re.match(r"## (\d+[a-z]?)\.? (.+)", line)
        if m:
            section = "" if m.group(1) in ("7a", "8") or int(m.group(1)[0]) > 7 else m.group(2)
            head = []
            continue
        if not section or not line.startswith("|"):
            continue
        cells = _cells(line)
        if cells[0] == "Mechanism":
            head = cells
            continue
        if not head or set(cells[0]) <= set("-: "):
            continue
        row = dict(zip(head, cells, strict=False))
        status = row.get("Status", "")
        code = CODE.search(status)
        words = f"{row['Mechanism']} {status}"
        name_words = row["Mechanism"]
        res = [k for k, p in RES.items() if re.search(p, name_words, re.I)]
        if not res:
            res = [k for k, p in RES.items() if re.search(p, words)][:3]
        out.append(
            {
                "id": len(out),
                "section": section,
                "name": re.sub(r"[`*]", "", row["Mechanism"]),
                "sources": row.get("Sources", ""),
                "status": status,
                "code": code.group(1) if code else "?",
                "depth": "empirical"
                if "**empirical**" in status
                else "structural"
                if "**structural**" in status
                else None,
                "kernels": row.get("Kernels", ""),
                "var": row.get("Var", ""),
                "probes": sorted(
                    {
                        *re.findall(r"probes? `?([a-z][a-z0-9_]+)`?", status),
                        *re.findall(r"`([a-z][a-z0-9_]+)`", row.get("Sources", "")),
                    }
                ),
                "decisions": sorted(set(re.findall(r"\bD\d{3}\b", status))),
                "rounds": sorted({int(r) for r in re.findall(r"[Rr]ound (\d+)", status)}),
                "resources": res,
                "rules": [k for k, p in RULE.items() if re.search(p, name_words)],
            }
        )
    return out


def decisions(text: str, wanted: set[str]) -> dict[str, dict[str, str]]:
    out = {}
    for line in text.splitlines():
        m = re.match(r"\| (D\d+) \|", line)
        if m and m.group(1) in wanted:
            c = _cells(line)
            out[c[0]] = {"title": c[1], "text": c[2] if len(c) > 2 else ""}
    return out


def section_table(text: str, title: str) -> list[list[str]]:
    """The rows of the first table under a heading."""
    body = text.split(f"## {title}", 1)
    if len(body) < 2:
        return []
    rows = []
    for line in body[1].splitlines()[1:]:
        if line.startswith("## "):
            break
        if line.startswith("|"):
            c = _cells(line)
            if not set(c[0]) <= set("-: "):
                rows.append(c)
    return rows[1:]


def rejected(state: str) -> list[dict[str, Any]]:
    out = []
    for c in section_table(state, "Rejected hypotheses, an index"):
        if len(c) >= 3:
            out.append(
                {
                    "topic": c[0],
                    "text": c[1],
                    "where": c[2],
                    "decisions": re.findall(r"\bD\d{3}\b", c[2]),
                }
            )
    return out


def open_questions(state: str) -> list[dict[str, str]]:
    return [
        {"id": c[0], "text": " · ".join(c[1:])[:400]}
        for c in section_table(state, "Open questions (experiment counts per mechanism, LOOP §3)")
    ]


def matrix(text: str) -> list[dict[str, Any]]:
    """The coverage matrix: one group per operation × target."""
    part = text.split("## 7a.", 1)[-1].split("\n## 8", 1)[0]
    out: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None
    head: list[str] = []
    for line in part.splitlines():
        m = re.match(r"\*\*(.+?) × (.+?)\*\*:?\s*(.*)", line)
        if m:
            cur = {
                "op": m.group(1).split(" (")[0],
                "target": m.group(2),
                "ops": m.group(1),
                "why": m.group(3),
                "rows": [],
            }
            out.append(cur)
            head = []
            continue
        if cur is None or not line.startswith("|"):
            continue
        c = _cells(line)
        if c[0] == "Pattern":
            head = c
        elif head and not set(c[0]) <= set("-: "):
            cur["rows"].append(dict(zip(head, c, strict=False)))
    return out


def rule_code(sm_py: str, handlers: set[str]) -> dict[str, dict[str, Any]]:
    """Each class's `_op_<handler>` as the engine runs it, and every `sim` parameter it reads,
    itself or through the helpers it calls: read from the code, so the page cannot mis-bind one."""
    tree = ast.parse(sm_py)
    lines = sm_py.splitlines()
    fns: dict[str, tuple[int, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "SM":
            for f in node.body:
                if isinstance(f, ast.FunctionDef):
                    fns[f.name] = (f.lineno, "\n".join(lines[f.lineno - 1 : f.end_lineno]))

    def reads(name: str, seen: set[str]) -> set[str]:
        if name in seen or name not in fns:
            return set()
        seen.add(name)
        body = fns[name][1]
        out = set(re.findall(r'P\["(\w+)"\]', body))
        for helper in re.findall(r"self\.(_\w+)\(", body):
            out |= reads(helper, seen)
        return out

    return {
        h: {
            "line": fns[f"_op_{h}"][0],
            "code": fns[f"_op_{h}"][1],
            "reads": sorted(f"sim.{k}" for k in reads(f"_op_{h}", set())),
        }
        for h in handlers
        if f"_op_{h}" in fns
    }


def build(root: Path, machine: dict[str, Any]) -> dict[str, Any]:
    inv_text = (root / "docs/design/29-mechanism-inventory.md").read_text()
    state = (root / "docs/design/perfmodel/STATE.md").read_text()
    decision_text = (root / "docs/decisions.md").read_text()
    rows = inventory(inv_text)
    rej = rejected(state)
    wanted = {d for r in rows for d in r["decisions"]} | {d for r in rej for d in r["decisions"]}
    table = json.loads((root / "docs/design/perfmodel/error-table.json").read_text())
    by_probe: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in table["rows"]:
        name = r["set"].split(" ", 1)
        if name[0] == "probe" and len(name) == 2 and r.get("measured") == r.get("measured"):
            by_probe[name[1].removesuffix(".cu")].append(
                {
                    "id": r["id"],
                    "measured": r.get("measured"),
                    "predicted": r.get("predicted"),
                    "unit": r.get("unit"),
                    "excluded": r.get("excluded"),
                }
            )
    names = {p for r in rows for p in r["probes"]}
    rejected_by_decision = {d: i for i, r in enumerate(rej) for d in r["decisions"]}
    for r in rows:
        r["rejected"] = sorted(
            {rejected_by_decision[d] for d in r["decisions"] if d in rejected_by_decision}
        )
    classes = machine["classes"]
    handlers = {c["name"]: c.get("handler") for c in classes}
    wanted_rules = {h for h in handlers.values() if h} | {k for r in rows for k in r["rules"]}
    head = re.search(r"Updated (\S+) \(round (\d+)\)", state)
    return {
        "as_of": {"date": head.group(1), "round": int(head.group(2))} if head else None,
        "table_date": table.get("date"),
        "mechanisms": rows,
        "decisions": decisions(decision_text, wanted),
        "rejected": rej,
        "questions": open_questions(state),
        "probes": {k: v for k, v in by_probe.items() if k in names},
        "matrix": matrix(inv_text),
        "classes": classes,
        "params": machine["params"],
        "yaml_resources": machine["resources"],
        "rules": rule_code((root / "src/kta/models/gb10sim/sm.py").read_text(), wanted_rules),
    }
