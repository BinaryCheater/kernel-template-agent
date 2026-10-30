"""What goes into the full-text index, and how plain text becomes an FTS5 query.

Indexed fields are a per-kind table (decision D151): the title field and the prose
fields people and agents search. Kinds not listed are not indexed (hashes, numbers and
IDs only add noise). Adding a kind here affects records written afterwards.
"""

from __future__ import annotations

import re
from typing import Any

# kind -> (title path, text paths); paths are dotted into the body, "*" maps over lists.
TEXT_FIELDS: dict[str, tuple[str, tuple[str, ...]]] = {
    "Pattern": ("name", ("description", "examples", "category", "params.*.meaning")),
    "Template": (
        "name",
        (
            "summary",
            "refines",
            "produces",
            "object.description",
            "scope.text",
            "backward_map",
            "program.description",
            "variables.*.meaning",
            "constraints.*.meaning",
            "obligations.*.statement",
            "signature.algorithm",
            "numeric_risk",
        ),
    ),
    "Chain": ("workload", ("items.*.applies_to", "items.*.template")),
    "Family": ("name", ("entry", "numeric_risk", "glue_constraints.*.meaning")),
    "Reference": ("name", ("problem", "build")),
    "Reconstruction": ("notes", ("unmatched_transactions.*.description",)),
    "Structure": ("name", ("description", "evidence_needed")),
    "Regime": ("structure", ("context",)),
    "Negative": ("subject", ("condition", "effect")),
    "Finding": ("claim", ("kind", "subject", "scope.workload", "scope.condition")),
    "Coupling": ("effect", ("members.*.name", "scope.workload", "scope.condition")),
    "Realisation": ("template", ("backend", "summary", "requires", "code_markers")),
    "Capability": ("name", ("description", "realised_by", "unit")),
    "Gap": ("description", ("missing", "attribution", "pattern_guess")),
    "HwFeature": ("name", ("description", "instructions", "unit")),
    "HwFact": ("quantity", ("feature", "unit")),
    "Problem": ("name", ("reference_impl",)),
    "Source": ("name", ("locator", "note")),
    "Protocol": ("name", ("description",)),
    "Report": ("title", ("summary", "subject")),
    "Submission": ("note", ("files",)),
    "HumanAction": ("action", ("reason",)),
    "TaskCard": ("goal", ("done_checks",)),
}

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _values(node: Any, parts: list[str]) -> list[str]:
    if not parts:
        if isinstance(node, str):
            return [node]
        if isinstance(node, list):
            return [str(v) for v in node if isinstance(v, (str, int, float))]  # pyright: ignore[reportUnknownVariableType]
        return []
    head, rest = parts[0], parts[1:]
    if head == "*":
        if not isinstance(node, list):
            return []
        found: list[str] = []
        for item in node:  # pyright: ignore[reportUnknownVariableType]
            found += _values(item, rest)
        return found
    if isinstance(node, dict) and head in node:
        return _values(node[head], rest)
    return []


def extract(kind: str, body: dict[str, Any]) -> tuple[str, str] | None:
    """(title, text) for the index, or None when the kind is not indexed."""
    fields = TEXT_FIELDS.get(kind)
    if fields is None:
        return None
    title_path, text_paths = fields
    title = " ".join(_values(body, title_path.split(".")))
    text = "\n".join(v for p in text_paths for v in _values(body, p.split(".")))
    return title, text


def match_query(text: str) -> str | None:
    """Plain words to an FTS5 query: every token quoted (no operators), all must match."""
    tokens = _TOKEN.findall(text)
    if not tokens:
        return None
    return " ".join(f'"{t}"' for t in tokens)
