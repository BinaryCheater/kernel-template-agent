"""Fixture records for the WP9b pages: patterns, a structure, gaps, agent profiles, an
AgentRun with its prompt, an experiment protocol with runs and results, an improvement
report and one batch environment lock.

# kta:fixture synthetic data for frontend development only (D168, D349).
Every record has origin "fixture"; the batch lock says ``"fixture": true``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kta.core.schemas import Actor, Record
from kta.store import SqliteOps
from kta_devseed.knowledge import Seeder


def _blob(s: Seeder, text: str) -> str:
    return s.store.put_blob(text.encode("utf-8"), "text/markdown")


def _patterns(s: Seeder) -> None:
    rows = [
        ("reduce", 1, "reduce", "Combine many values into one with an associative operator."),
        ("transfer", 3, "transfer", "Move a tile between memory levels."),
        (
            "warp_vote_mask",
            3,
            "other",
            "Ballot-based masking in a decode loop; no category fits yet.",
        ),
    ]
    for name, level, cat, desc in rows:
        s.put(
            f"pattern:{name}",
            "Pattern",
            {
                "name": name,
                "level": level,
                "category": cat,
                "params": [],
                "description": desc,
                "examples": [f"{name} example a", f"{name} example b"],
            },
        )
    s.put(
        "structure:monotone_k",
        "Structure",
        {
            "name": "monotone_in_k_split",
            "effect": "enumeration",
            "description": "Latency is monotone in k_split beyond the wave boundary.",
            "evidence_needed": "measurements at three k_split values per shape bucket",
        },
    )
    for i, (attr, missing) in enumerate(
        [
            ("model", "model_term"),
            ("intent_sass", "check"),
            ("space", "pattern"),
            ("process", "other"),
        ]
    ):
        s.put(
            f"gap:extra{i}",
            "Gap",
            {
                "missing": missing,
                "description": f"fixture gap {i}: {attr} shortfall",
                "level": 3,
                "pattern_guess": None,
                "attribution": attr,
            },
        )


def _profile(s: Seeder, version: int, model: str) -> str:
    skill = {n: f"{n}@{_blob(s, f'fixture skill {n}')}" for n in ("use-kta", "find-knowledge")}
    types = ("T1", "T2", "T3", "T4", "T5", "T6")
    body: dict[str, Any] = {
        "name": "codex-v1",
        "version": version,
        "provider": "codex",
        "cli_version": "0.117.0",
        "auth": "subscription",
        "isolation": "config+user",
        "model": model,
        "reasoning_effort": "medium",
        "cli_flags": ["--json", "-c", 'sandbox_mode="workspace-write"'],
        "home_files": {"config.toml": _blob(s, "# fixture config")},
        "prefix_template": _blob(s, "# fixture prefix"),
        "step_templates": {t: _blob(s, f"# fixture step {t}") for t in types},
        "message_templates": {"rejection": _blob(s, "fixture rejection")},
        "skills": {t: list(skill.values()) for t in types},
        "allowed_commands": {t: ["show", "query", "submit"] for t in types},
        "context_policy": {"cards_k": 5},
        "loop_policy": {"max_retries": 2, "idle_s": 600},
    }
    return s.put(f"profile:v{version}", "AgentProfile", body)


def _agent_run(s: Seeder, ops: SqliteOps, profile: str) -> None:
    task = next(t for t in _tasks(ops) if t[1] == "T1" and t[2] == "running")
    prompt = _blob(
        s,
        "# TASK\n\nType: T1 (fixture)\n\nGoal: extend gemm_tiled to bf16 inputs.\n\n"
        "## Done checks\n- validate_family\n- verify_space\n- delivery note complete\n",
    )
    runner = Actor(type="system", name="kta.agents.runner", task=task[0])
    for i, (wall, tool, exit_, tokens, violations) in enumerate(
        [
            (840.0, 310.0, "submitted", (182_000, 9_400), []),
            (120.0, 40.0, "rejected", (None, None), [{"kind": "command_not_allowed"}]),
        ]
    ):
        body: dict[str, Any] = {
            "session": "fixture-session-0001",
            "profile": profile,
            "cli_version": "0.117.0",
            "model": "gpt-5.4",
            "tokens_in": tokens[0],
            "tokens_out": tokens[1],
            "session_archive": _blob(s, f"fixture session archive {i}"),
            "wall_s": wall,
            "tool_s": tool,
            "queue_s": 12.0,
            "exit": exit_,
            "context_manifest": _blob(s, f"fixture manifest {i}"),
            "violations": violations,
        }
        refs: dict[str, Any] = {"env": s.ids["env:spark"], "card": task[3], "profile": profile}
        rec = Record.new(
            "AgentRun", body, runner, s.clock.now(), refs, {"prompt": prompt}, origin="fixture"
        )
        s.store.put(rec)


def _tasks(ops: SqliteOps) -> list[tuple[str, str, str, str]]:
    with ops.db.read() as con:
        rows = con.execute("SELECT id, card_type, state, card_id FROM tasks ORDER BY created_at")
        return [(r[0], r[1], r[2], r[3]) for r in rows.fetchall()]


def _experiment(s: Seeder) -> None:
    proto = s.put(
        "protocol:pool-vs-none",
        "Protocol",
        {
            "name": "pool-vs-no-pool",
            "version": 1,
            "type": "experiment",
            "description": "T1 tasks with the Template pool vs with source code only (fixture).",
            "params": {"arms": ["pool", "no-pool"], "runs_per_arm": 4, "cases": 2},
        },
    )
    for i, arm in enumerate(["pool", "pool", "pool", "no-pool", "no-pool"]):
        s.put(
            f"exprun:{i}",
            "ExperimentRun",
            {"name": f"run {i}", "condition": arm, "seed": i, "params": {}, "note": ""},
            refs={"protocol": proto},
        )
    s.put(
        "report:exp",
        "Report",
        {
            "title": "experiment",
            "subject": proto,
            "summary": "Interim: pool arm accepted 3/3, no-pool 1/2 (fixture numbers).",
            "data": {
                "rows": [
                    {"arm": "pool", "runs": 3, "accepted": 3, "median_wall_min": 14.0},
                    {"arm": "no-pool", "runs": 2, "accepted": 1, "median_wall_min": 31.5},
                ]
            },
        },
        refs={"protocol": proto},
    )
    s.put(
        "protocol:draft",
        "Protocol",
        {
            "name": "profile-v2-vs-v1",
            "version": 1,
            "type": "experiment",
            "description": "Compare agent profile v2 with v1 on the dev cases (fixture).",
            "params": {"arms": ["v1", "v2"], "runs_per_arm": 6},
        },
    )


def _improvement(s: Seeder) -> None:
    s.put(
        "report:improve",
        "Report",
        {
            "title": "improvement",
            "subject": "batch-0",
            "summary": "Most shortfalls come from the model and SASS intent (fixture).",
            "data": {
                "batch": "batch-0",
                "attribution": {"model": 2, "intent_sass": 1, "space": 2},
                "items": [
                    {
                        "key": "i1",
                        "title": "Add an L2 reuse term",
                        "kind": "model",
                        "description": "T4 diagnosis",
                    },
                    {
                        "key": "i2",
                        "title": "Check ldmatrix in SASS",
                        "kind": "system_code",
                        "description": "checker change",
                    },
                ],
            },
        },
    )


def _batch(data_dir: Path, profile: str) -> None:
    root = data_dir / "batches" / "fixture-batch-0"
    (root / "work" / "gemm_bf16_256").mkdir(parents=True, exist_ok=True)
    lock = {
        "fixture": True,
        "batch": "fixture-batch-0",
        "profile": profile,
        "commit": "0" * 40,
        "cli_version": "0.117.0",
        "files": {"agent-home-template/config.toml": "0" * 64, "prompts/prefix.md": "1" * 64},
    }
    (root / "env.lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")


def seed_supervision(s: Seeder, ops: SqliteOps, data_dir: Path) -> None:
    _patterns(s)
    v1 = _profile(s, 1, "gpt-5.4")
    _profile(s, 2, "gpt-5.5")
    if not ops.set_pointer("current_profile", None, v1):
        raise RuntimeError("current_profile already set in a fresh store")
    _agent_run(s, ops, v1)
    _experiment(s)
    _improvement(s)
    _batch(data_dir, v1)
