"""Patterns for credentials that must never reach git, records, logs or prompts (conventions §8).

Shared by the log redaction filter and by the gate's secret scan.
"""

from __future__ import annotations

import re

SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{20,}")),
    ("openai_key", re.compile(r"sk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{32,}")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b")),
    ("github_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("hf_token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
    ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----")),
    (
        "assigned_secret",
        re.compile(
            r"(?i)\b(?:api[_-]?key|secret|token|password)\b\s*[:=]\s*['\"][A-Za-z0-9_\-/+]{24,}['\"]"
        ),
    ),
)

REDACTED = "[REDACTED]"


def find_secrets(text: str) -> list[tuple[str, str]]:
    """Return (pattern name, matched text) pairs."""
    hits: list[tuple[str, str]] = []
    for name, pattern in SECRET_PATTERNS:
        hits.extend((name, m.group(0)) for m in pattern.finditer(text))
    return hits


def redact(text: str) -> str:
    for _, pattern in SECRET_PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text
