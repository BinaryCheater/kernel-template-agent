"""Check results shared by all gate checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Status = Literal["PASS", "FAIL", "WARN"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: Status
    detail: str = ""
    findings: list[str] = field(default_factory=lambda: list[str]())

    def render(self, max_findings: int = 20) -> str:
        head = f"{self.status:4} {self.name}"
        if self.detail:
            head += f" — {self.detail}"
        lines = [head]
        lines += [f"       {f}" for f in self.findings[:max_findings]]
        if len(self.findings) > max_findings:
            lines.append(f"       ... {len(self.findings) - max_findings} more")
        return "\n".join(lines)


def from_findings(name: str, findings: list[str], ok_detail: str = "") -> CheckResult:
    if findings:
        return CheckResult(name, "FAIL", f"{len(findings)} finding(s)", findings)
    return CheckResult(name, "PASS", ok_detail)
