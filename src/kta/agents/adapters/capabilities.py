"""Admission table: what each agent CLI satisfies (spec/modules/agents.md "准入要求").

Values are ``yes``, ``no``, ``unknown`` (not yet checked) or ``partial`` with a note. A
``yes`` that has not been seen in a real KTA run says so in its note; the real-run
checks are listed in deliveries/WP8.md (decision D303).
"""

from __future__ import annotations

from dataclasses import dataclass, field

REQUIREMENTS: tuple[str, ...] = (
    "non_interactive",
    "persistent_sessions",
    "parsable_events",
    "token_usage",
    "resume",
    "config_isolation",
    "auth",
    "reads_workspace_agents_md",
    "disable_web_and_mcp",
    "model_and_api_base",
    "sandbox",  # not an admission requirement; recorded because isolation depends on it
)
# Without these an adapter cannot be admitted at all.
MANDATORY: frozenset[str] = frozenset(
    {"non_interactive", "persistent_sessions", "disable_web_and_mcp", "model_and_api_base"}
)
# Options that turn off session persistence; adapters refuse them and the WP8 gate scans
# adapter code for them (conventions §7.1).
NON_PERSISTENT_OPTIONS: tuple[str, ...] = (
    "--" + "ephemeral",  # codex
    "--no-session",  # pi
    "--no-session-persistence",
)


@dataclass(frozen=True)
class Capability:
    value: str  # yes | no | partial | unknown
    note: str = ""


@dataclass(frozen=True)
class AdapterCapabilities:
    adapter: str
    checked_version: str | None
    items: dict[str, Capability] = field(default_factory=lambda: dict[str, Capability]())

    def admitted(self) -> bool:
        return all(
            self.items.get(r, Capability("unknown")).value in ("yes", "partial") for r in MANDATORY
        )


CODEX = AdapterCapabilities(
    "codex",
    "0.117.0 (spark), 0.163.0 (control host); help text and existing session files only",
    {
        "non_interactive": Capability("yes", "`codex exec`, prompt from stdin (`-`)"),
        "persistent_sessions": Capability(
            "yes", "rollout JSONL under $CODEX_HOME/sessions/; persistence is the default"
        ),
        "parsable_events": Capability(
            "yes",
            "session entries with timestamps (response_item, event_msg exec_command_end, "
            "patch_apply_end, token_count); `--json` stdout events give the thread id",
        ),
        "token_usage": Capability("yes", "event_msg token_count in the session file"),
        "resume": Capability(
            "yes", "`codex exec resume <id> -`; sandbox set with -c, cwd by process"
        ),
        "config_isolation": Capability(
            "partial",
            "CODEX_HOME replaces ~/.codex (config, AGENTS.md, skills, sessions); system files "
            "/etc/codex/* would still be read and are listed in the context manifest; "
            "not yet confirmed by the marker run",
        ),
        "auth": Capability("yes", "subscription: auth.json linked into CODEX_HOME (D76)"),
        "sandbox": Capability(
            "partial",
            "spark: bubblewrap refused (AppArmor userns restriction); Landlock "
            "(features.use_legacy_landlock) denies writes outside the workspace and network by "
            "default, but not reads (D317); probed with `codex sandbox linux`, no model call",
        ),
        "reads_workspace_agents_md": Capability(
            "yes", "AGENTS.md from the git root down to cwd; the workspace is its own git root"
        ),
        "disable_web_and_mcp": Capability(
            "partial",
            'web_search="disabled" and no [mcp_servers] in the isolated config; apps, '
            "plugins, multi_agent, tool_suggest features off; to be confirmed in a real run",
        ),
        "model_and_api_base": Capability(
            "yes", "-m / model; model_providers.<name>.base_url in config.toml for an API base"
        ),
    },
)

# Interfaces kept, not run in the first phase (spec/workplan.md WP8 allowed simplification).
CLAUDE_CODE = AdapterCapabilities(
    "claude_code",
    None,
    {r: Capability("unknown", "not checked; adapter raises NotSupported") for r in REQUIREMENTS},
)
# Checked on the control host from `pi --help` of pi 1.3.0 (/opt/homebrew/bin/pi); no run.
PI = AdapterCapabilities(
    "pi",
    "1.3.0 (control host; help text only)",
    {
        "non_interactive": Capability("yes", "`pi -p` (print mode), `--mode json` for JSON output"),
        "persistent_sessions": Capability(
            "yes", "sessions saved by default; `--session-dir` / PI_CODING_AGENT_SESSION_DIR"
        ),
        "parsable_events": Capability("unknown", "`--mode json` output format not inspected"),
        "token_usage": Capability("unknown", "not inspected"),
        "resume": Capability("yes", "`--session <path|id>` / `--session-id <id>`"),
        "config_isolation": Capability(
            "yes",
            "PI_CODING_AGENT_DIR replaces ~/.pi/agent; --no-extensions, --no-skills, "
            "--no-prompt-templates, --no-themes, --no-approve exist",
        ),
        "auth": Capability("yes", "API keys from provider environment variables (api auth)"),
        "reads_workspace_agents_md": Capability(
            "yes", "AGENTS.md discovery (can be turned off with --no-context-files)"
        ),
        "disable_web_and_mcp": Capability(
            "yes", "--no-mcp; tools allowlist with --tools; no built-in web tool listed"
        ),
        "model_and_api_base": Capability(
            "partial",
            "--provider/--model; a custom API base needs a provider definition in the config "
            "directory (format not inspected)",
        ),
        "sandbox": Capability("unknown", "no sandbox option listed; would rely on kta-agent"),
    },
)
DEEPSEEK_HARNESS = AdapterCapabilities(
    "deepseek_harness",
    None,
    {
        r: Capability("unknown", "not installed on the control host; not checked")
        for r in REQUIREMENTS
    },
)

ALL: dict[str, AdapterCapabilities] = {
    c.adapter: c for c in (CODEX, CLAUDE_CODE, PI, DEEPSEEK_HARNESS)
}
