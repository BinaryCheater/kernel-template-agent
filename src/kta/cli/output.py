"""The only place in src/ allowed to write to stdout/stderr directly (spec/conventions.md §4)."""

from __future__ import annotations

import json
import sys
from typing import Any


def emit_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def emit_text(text: str) -> None:
    print(text)


def emit_error(text: str) -> None:
    print(text, file=sys.stderr)
