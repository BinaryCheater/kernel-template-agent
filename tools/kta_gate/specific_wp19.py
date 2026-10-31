"""WP19 specific checks (spec/gate.md; spec/workplan.md WP19 and WP17, folded in by D328).
Kept apart from specific.py (500-line limit); registered there.

- The named WP19 tests passed (projection compiled and solved on the GEMM seed with the
  objective equal to L1's projection, the enumerated minimum reached, outer enumeration
  and its limit; default plan; agent-plan fixture diffed and scored; enumerator finds the
  seed chain within its bound; composition statistics as a view; header on one screen;
  skill table identical to the data; pull log written and shown on the Release page;
  ``solve`` selects the projection arm).
- The worker skill ``use-kta`` carries the access-discipline table rendered from
  ``core/access_discipline.yaml`` for the version the registry knob names.
- The FormulationPlan and Pull kinds are registered; spec sections exist.
- The MIP <-> L2 consistency check passes on the GEMM seed family and fails, naming the
  effect, on the wrong-reservation-table fixture (named tests); its thresholds are the
  registry knob ``model.mip_l2_check`` set by X11 (D551).

No real-run check: WP19 is a code-stage WP (D320).
"""

from __future__ import annotations

from pathlib import Path

from kta_gate.result import CheckResult, from_findings

WP19_TESTS = (
    "test_gemm_seed_formulation_compiles_and_solves",
    "test_projection_is_the_minimum_of_enumerated_projection",
    "test_outer_enumeration_and_its_limit",
    "test_default_plan_for_the_gemm_seed_case",
    "test_agent_plan_fixture_is_diffed_and_scored",
    "test_enumerator_finds_the_seed_chain_within_its_bound",
    "test_composition_stats_is_a_view_over_chains",
    "test_header_fits_one_screen_and_states_the_situation",
    "test_skill_carries_the_rendered_table",
    "test_task_md_starts_with_the_header",
    "test_pull_log_written_for_a_task_and_shown_on_release",
    "test_solve_selects_the_projection_arm",
    "test_mip_l2_check_passes_on_the_gemm_seed_family",
    "test_mip_l2_check_fails_naming_a_wrong_reservation_table",
)
SKILL = Path("src") / "kta" / "agents" / "skills" / "use-kta" / "SKILL.md"


def _skill(root: Path) -> list[str]:
    from kta.core.access import SKILL_BEGIN, SKILL_END, render, table
    from kta.core.controls import baseline

    text = (root / SKILL).read_text(encoding="utf-8")
    if SKILL_BEGIN not in text or SKILL_END not in text:
        return [f"{SKILL} has no access-discipline block"]
    block = text[text.index(SKILL_BEGIN) : text.index(SKILL_END) + len(SKILL_END)]
    version = baseline().access_discipline_version
    if block != render(table(version)):
        return [f"{SKILL}: the table differs from access_discipline.yaml {version}"]
    return []


def _kinds_and_spec(root: Path) -> list[str]:
    from kta.core.schemas.registry import KINDS

    findings = [f"record kind {k} is not registered" for k in ("FormulationPlan", "Pull")
                if k not in KINDS]  # fmt: skip
    schemas = (root / "spec" / "schemas.md").read_text(encoding="utf-8")
    models = (root / "spec" / "modules" / "formulation-solve-models.md").read_text("utf-8")
    need = [
        ("spec/schemas.md", schemas, "#### 4.8.1 FormulationPlan"),
        ("spec/modules/formulation-solve-models.md", models, "### Projection compiler"),
        ("spec/modules/formulation-solve-models.md", models, "### FormulationPlan and chains"),
    ]
    return findings + [f"{w} lacks {t!r}" for w, doc, t in need if t not in doc]


def wp19(root: Path, passed: set[str]) -> list[CheckResult]:
    names = {t.split("::")[-1] for t in passed}
    absent = [f"{t} did not pass" for t in WP19_TESTS if t not in names]
    return [
        from_findings("WP19 tests", absent, f"{len(WP19_TESTS)} named tests passed"),
        from_findings("access discipline in the skill", _skill(root), "use-kta carries v1"),
        from_findings("WP19 kinds and spec", _kinds_and_spec(root), "kinds and sections exist"),
        from_findings("MIP <-> L2 thresholds set by X11", _x11(), "model.mip_l2_check in X11 sets"),
    ]


def _x11() -> list[str]:
    from kta.core.controls import definitions

    defs = definitions()
    knob = next((k for k in defs.knobs if k.id == "model.mip_l2_check"), None)
    x11 = next((e for e in defs.experiments if e.id == "X11"), None)
    if knob is None or x11 is None:
        return ["knob model.mip_l2_check or experiment X11 is missing"]
    bad = [] if "X11" in knob.set_by else ["model.mip_l2_check is not set_by X11"]
    return bad + ([] if knob.id in x11.sets else ["X11 does not list model.mip_l2_check in sets"])
