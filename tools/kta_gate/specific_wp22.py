"""WP22 specific checks (spec/gate.md "WP22", D580–D592): system settings and agent proxy.

Static part (both stages): the named WP22 tests passed; the agent environment whitelist
holds no proxy variable (they enter only through the proxy injection); the system
configuration's provider defaults are the user's (codex and claude_code required, pi and
deepseek_harness off); the spec sections exist; Settings has its System tab and
/mechanisms is routed; the web app builds.

Real-run part: an agent launch whose AgentRun records `proxy.used` with a passed preflight.
WARN in the code stage (D320); the control-host preflight against the local proxy is in
the evidence list.
"""

from __future__ import annotations

from pathlib import Path

from kta_gate.result import CheckResult, from_findings

WP22_TESTS = (
    "test_whitelist_injection_per_policy",
    "test_no_proxy_always_holds_loopback_and_the_api_host",
    "test_fail_closed_preflight_refuses_the_launch",
    "test_no_credentials_in_records",
    "test_off_provider_launch_gets_no_proxy_variable",
    "test_detect_prefers_scutil_then_env_then_clash_default",
    "test_system_config_defaults_revisions_and_stale_writes",
    "test_preflight_reports_a_down_port_and_a_refused_connect",
    "test_system_view_shows_proxy_providers_workers_and_stage",
    "test_system_write_needs_the_token_and_is_audited",
    "test_proxy_test_reports_failure_and_a_pass_clears_the_breaker",
    "test_worker_test_runs_a_trivial_job_through_the_executor",
    "test_mechanisms_view_lists_each_mechanism_with_origin",
)
USER_POLICY = {
    "codex": "required",
    "claude_code": "required",
    "pi": "off",
    "deepseek_harness": "off",
}
SPEC_MARKS = (
    ("spec/workplan.md", "## WP22 "),
    ("spec/modules/agents.md", "## WP22: agent proxy"),
    ("spec/modules/api-web.md", "## WP22: Settings"),
    ("spec/schemas.md", "proxy: dict?"),
)


def check_tests(passed: set[str]) -> CheckResult:
    names = {t.split("::")[-1].split("[")[0] for t in passed}
    absent = [f"{t} did not pass" for t in WP22_TESTS if t not in names]
    return from_findings("WP22 tests", absent, f"{len(WP22_TESTS)} named tests passed")


def check_policy() -> CheckResult:
    from kta.agents.guard import ENV_WHITELIST
    from kta.core.proxy import ALL_VARS
    from kta.core.sysconfig import SystemConfig

    findings = [f"{v} is on the agent whitelist" for v in sorted(ENV_WHITELIST & ALL_VARS)]
    got = {k: v.proxy for k, v in SystemConfig().providers.items()}
    if got != USER_POLICY:
        findings.append(f"provider proxy defaults {got} != the user's {USER_POLICY}")
    return from_findings("WP22 proxy policy", findings, "no inherited proxy; user defaults")


def check_docs(root: Path) -> CheckResult:
    findings = [
        f"{path}: missing {mark!r}"
        for path, mark in SPEC_MARKS
        if mark not in (root / path).read_text(encoding="utf-8")
    ]
    app = (root / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    if 'path="/mechanisms"' not in app:
        findings.append("/mechanisms is not routed")
    settings = (root / "web" / "src" / "pages" / "Settings.tsx").read_text(encoding="utf-8")
    if "SystemSettings" not in settings:
        findings.append("Settings has no System tab")
    return from_findings("WP22 spec and pages", findings, "spec sections, System tab, /mechanisms")


def wp22(root: Path, passed: set[str]) -> list[CheckResult]:
    from kta_gate.specific import code_stage, pending_real, web_build

    results = [check_tests(passed), check_policy(), check_docs(root), web_build(root)]
    name = "WP22 agent launch through the proxy"
    what = "an AgentRun whose proxy.used is true with a passed preflight on the agent host"
    if code_stage():
        return [*results, pending_real(name, what)]
    return [*results, CheckResult(name, "FAIL", f"needs a real run: {what}")]
