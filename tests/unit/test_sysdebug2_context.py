"""SYSDEBUG-2: context across rounds and the projection MIP in the loop (D715, D716)."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kta.agents import handoff, hostfs, sessions, workspace
from kta.agents.batches import BatchRoot
from kta.agents.profile import CODEX_V4, CODEX_V5, profile_body
from kta.agents.runner import _prompt  # pyright: ignore[reportPrivateUsage]
from kta.core.errors import ConfigError, SchemaError
from kta.core.schemas import Actor, Record
from kta.core.schemas.plan import TaskHeader
from kta.core.schemas.process import TaskCard
from kta.orchestrator import solve_work
from kta.orchestrator.outputs import family_diff
from kta.store.sqlite import SqliteStore
from storehelp import T0

ACTOR = Actor(type="tool", name="test")
FAMILY = {"name": "gemm", "backend": "tilelang", "entry": "gen.py:generate"}


def _blob(store: SqliteStore, text: str) -> str:
    return store.put_blob(text.encode(), "text/plain")


def _family(
    store: SqliteStore, version: int, files: dict[str, str], refs: dict[str, Any], minutes: int = 0
) -> str:
    blobs = {f"files/{k}": _blob(store, v) for k, v in files.items()}
    rec = Record.new(
        "Family", FAMILY | {"version": version}, ACTOR, T0 + timedelta(minutes=minutes),
        refs=refs, blobs=blobs,
    )  # fmt: skip
    return store.put(rec)


def _case(store: SqliteStore, name: str = "case") -> str:
    # any stored record stands in for the Case: handoff only compares IDs
    return store.put(
        Record.new("Report", {"title": name, "subject": "x", "summary": "", "data": {}}, ACTOR, T0)
    )


def _install(job: Path, ws: Path, task: str, files: dict[str, str]) -> None:
    for name, text in {"TASK.md": f"Task `{task}`", **files}.items():
        p = job / "bundle/workspace" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    hostfs.install_workspace(job, ws, ["TASK.md", *files])


def test_out_is_prefilled_once_and_a_retry_keeps_the_draft(tmp_path: Path) -> None:
    job, ws = tmp_path / "job", tmp_path / "ws"
    _install(job, ws, "t1", {"out/gen.py": "v1", "refs/tilelang/a.py": "doc"})
    assert (ws / "out/gen.py").read_text() == "v1" and (ws / "refs/tilelang/a.py").is_file()
    (ws / "out/gen.py").write_text("agent draft")
    _install(job, ws, "t1", {"out/gen.py": "v1"})  # retry of the same task
    assert (ws / "out/gen.py").read_text() == "agent draft"
    assert not (ws / "refs").exists()  # refs are replaced wholesale every step
    _install(job, ws, "t2", {"out/gen.py": "v2"})  # the next round's task
    assert (ws / "out/gen.py").read_text() == "v2"


def test_family_out_leaves_the_delivery_note_to_the_agent(store: SqliteStore) -> None:
    fid = _family(store, 1, {"gen.py": "code", "DELIVERY.md": "old note"}, {})
    assert workspace.family_out(store, [fid]) == {"out/gen.py": b"code"}
    assert workspace.family_out(store, []) == {}


def test_reference_files_take_text_only(tmp_path: Path) -> None:
    (tmp_path / "lang").mkdir()
    (tmp_path / "lang/ops.py").write_text("def gemm(): ...")
    (tmp_path / "lib.so").write_bytes(b"\0")
    assert workspace.reference_files(str(tmp_path)) == {"lang/ops.py": b"def gemm(): ..."}
    assert workspace.reference_files("") == {}
    with pytest.raises(ConfigError):
        workspace.reference_files(str(tmp_path / "missing"))


def test_family_diff_names_the_changed_files_only(store: SqliteStore) -> None:
    old = _family(store, 1, {"gen.py": "a\nb\n", "family.yaml": "x\n"}, {})
    new = {
        "files/gen.py": _blob(store, "a\nc\n"),
        "files/family.yaml": store.get(old).blobs["files/family.yaml"],
    }
    text = family_diff(SimpleNamespace(store=store), old, new)  # pyright: ignore[reportArgumentType]
    assert "b/gen.py" in text and "-b" in text and "+c" in text and "family.yaml" not in text


def test_handoff_lists_this_cases_versions_with_their_notes(store: SqliteStore) -> None:
    case, other = _case(store), _case(store, "other")
    _family(store, 1, {"DELIVERY.md": "## Done\nbaseline tiles"}, {"case": case}, 1)
    _family(
        store,
        2,
        {"DELIVERY.md": "intro\n## Done\npipeline S stages\n## Other\nx"},
        {"case": case},
        2,
    )
    _family(store, 3, {"DELIVERY.md": "## Done\nnot this case"}, {"case": other}, 3)
    head = TaskHeader(text="h", situation="eps", discipline="v1", case=case)
    card = TaskCard(type="T1", goal="g", inputs=[case], profile="p", work_unit="u", header=head)
    lines = handoff.handoff_text(store, card).splitlines()
    assert [ln.split(" ")[1] for ln in lines] == ["gemm@1", "gemm@2"]
    assert lines[1].endswith("pipeline S stages")


def test_v5_profile_has_the_context_messages_and_no_one_mechanism_rule(store: SqliteStore) -> None:
    v4, v5 = profile_body(store, CODEX_V4), profile_body(store, CODEX_V5)
    assert "new_round" not in v4.message_templates
    assert {"new_round", "handoff"} <= set(v5.message_templates)
    text = store.blob_path(v5.step_templates["T1"]).read_text()
    assert "one mechanism" not in text.lower() and "no family yet" in text


def _batch(tmp_path: Path) -> BatchRoot:
    root = tmp_path / "batch"
    (root / "prompts").mkdir(parents=True)
    for name, text in {
        "new_session": "NEW SESSION $work_unit$extra",
        "rejection": "REJECTED",
        "human_note": "NOTE $note",
        "new_round": "NEW ROUND $task_id\n$header",
        "handoff": "HANDOFF\n$handoff",
    }.items():
        (root / "prompts" / f"msg-{name}.md").write_text(text)
    return BatchRoot(root, {})


def test_prompt_gives_the_delta_on_resume_and_the_handoff_on_a_fresh_start(
    store: SqliteStore, tmp_path: Path
) -> None:
    case = _case(store)
    _family(store, 1, {"DELIVERY.md": "## Done\nbaseline"}, {"case": case})
    old_card, new_card = _case(store, "old card"), _case(store, "new card")
    run = store.put(
        Record.new(
            "Report",
            {"title": "run", "subject": "x", "summary": "", "data": {}},
            ACTOR,
            T0,
            refs={"card": old_card},
        )
    )
    state = sessions.SessionState(session="s", runs=[run])
    header = TaskHeader(text="s0 measured 2013 us", situation="eps", discipline="v1", case=case)
    card = TaskCard(type="T1", goal="g", inputs=[case], profile="p", work_unit="u", header=header)
    task = SimpleNamespace(id="T-new", card_id=new_card)
    b = _batch(tmp_path)
    resumed = _prompt(b, "resume", card, task, None, None, store, state)  # pyright: ignore[reportArgumentType]
    assert resumed.startswith("NEW ROUND T-new") and "2013 us" in resumed
    same = SimpleNamespace(id="T-old", card_id=old_card)
    assert _prompt(b, "resume", card, same, None, None, store, state).startswith("Continue")  # pyright: ignore[reportArgumentType]
    fresh = _prompt(b, "start", card, task, None, None, store, state)  # pyright: ignore[reportArgumentType]
    assert "NEW SESSION u" in fresh and "HANDOFF" in fresh and fresh.endswith("UTC): baseline")
    first = _prompt(b, "start", card, task, None, None, store, None)  # pyright: ignore[reportArgumentType]
    assert "HANDOFF" not in first


def test_projection_inputs_fall_back_to_enumeration_with_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(*_a: Any, **_k: Any) -> Any:
        raise SchemaError("no menus")

    monkeypatch.setattr(solve_work, "default_plan", broken)
    proj, note = solve_work.projection_inputs(None, None, None, {"name": "s0", "shape": {}})
    assert proj is None and note == "no plan, enumeration: no menus"


def test_a_malformed_family_yaml_names_its_fields(tmp_path: Path) -> None:
    """D718: `kta validate-family` got an HTTP 500 (pydantic error not mapped), so the agent
    could not see what was wrong and guessed the format."""
    from kta.knowledge.families import load_family

    for name in ("chain.yaml", "edge_cases.yaml", "README.md", "gen.py"):
        (tmp_path / name).write_text("[]\n" if name.endswith(".yaml") else "")
    (tmp_path / "family.yaml").write_text(
        "name: f\nversion: 1\nbackend: tilelang\nvariables: {BM: {}}\nentry: gen.py:generate\n"
        "constraints: []\n"
    )
    with pytest.raises(SchemaError) as err:
        load_family(tmp_path)
    assert "family.yaml: variables: Input should be a valid list" in str(err.value)
    assert "constraints: Extra inputs are not permitted" in str(err.value)


def test_a_long_rejection_does_not_stop_the_next_step() -> None:
    """D719: a long failure (two checks with the same pydantic text) pushed TASK.md over
    step_header_max_chars, and the step stopped before the agent ran."""
    from kta.agents.cards import FAILURE_CHARS, RetryNote, failure_lines

    long = "ValidationError: " + "x " * 3000
    lines = failure_lines(RetryNote([("validate_family", long), ("verify_space", long)]))
    assert len(lines[0]) < FAILURE_CHARS + 120 and lines[0].endswith("to see all of it)")
    assert lines[1] == "`verify_space`: same failure as `validate_family`"


def test_task_md_states_an_open_task_after_a_submission(store: SqliteStore, tmp_path: Path) -> None:
    """D721: after a force retry the fresh session read 'submitted' in NOTES.md and stopped;
    TASK.md now states that the task is open while no submission was accepted."""
    from kta.agents import cards

    card = TaskCard(type="T1", goal="g", profile="p", work_unit="u")
    policy = {"cards_k": 5, "card_max_lines": 5, "step_header_max_chars": 6000}
    kw: dict[str, Any] = dict(
        task_id="T", card_id="c", card=card, cards=[], skills=[], commands=[], used={},
        retry=None, policy=policy,
    )  # fmt: skip
    assert "This task is open" not in cards.task_md("$previous", **kw)
    text = cards.task_md("$previous", **kw, last_submission="sub-1")
    assert "This task is open" in text and "`sub-1`" in text
    # D726: after a force retry the agent is told the reason was not its work
    forced = cards.task_md("$previous", **kw, last_submission="sub-1", forced=True)
    assert "outside your work" in forced and "submit again" in forced


def test_a_launch_without_arguments_is_refused_before_the_gpu(tmp_path: Path) -> None:
    """D723: a generator that listed no kernel arguments failed every point on the GPU with
    CUDA_ERROR_INVALID_VALUE, and the rejection only said 'check numeric.run'."""
    from kta.checks.full import failure_reason
    from kta.orchestrator.pipeline import generate

    gen = tmp_path / "gen.py"
    gen.write_text(
        "def generate(a, case):\n"
        "    launch = {'grid': [1], 'block': [128]}\n"
        "    return {'files': {'k.py': ''}, 'entry': 'main', 'launch': launch}\n"
    )
    fp = SimpleNamespace(gen_path=gen, gen_function="generate", path=tmp_path,
                         family=SimpleNamespace(name="f", backend="tilelang"))  # fmt: skip
    with pytest.raises(SchemaError, match=r"launch\.params is empty"):
        generate(fp, {}, {})  # pyright: ignore[reportArgumentType]
    detail = "Traceback ...\nkta.core.errors.ExecutionError: [nonzero] cuLaunchKernel failed: X\n"
    reason = failure_reason([{"check": "numeric.run", "detail": detail}])
    assert (
        reason
        == "check numeric.run, kta.core.errors.ExecutionError: [nonzero] cuLaunchKernel failed: X"
    )


def test_the_write_family_example_is_valid(store: SqliteStore) -> None:
    """D724: the v6 skill's example family.yaml and chain.yaml parse as the checks parse them,
    and generate()'s example names launch params (the formats agents got wrong in SYSDEBUG11)."""
    import re

    import yaml

    from kta.agents.profile import CODEX_V6, SKILLS, versioned
    from kta.core.schemas.knowledge import Chain, Family

    path = versioned(SKILLS, "write-family", CODEX_V6.version) / "SKILL.md"
    text = path.read_text()
    assert "/v6/" in str(path)
    blocks = re.findall(r"```yaml\n(.*?)```", text, re.S)
    Family.model_validate(yaml.safe_load(blocks[0]))
    chain = Chain.model_validate(yaml.safe_load(blocks[1]))
    assert any(k.startswith("p.") for k in chain.items[0].bindings)
    assert '"params"' in text and '"smem_bytes"' in text
    assert profile_body(store, CODEX_V6).skills  # the profile builds


def test_the_harness_turns_tma_off_for_every_tilelang_kernel() -> None:
    """D725: an agent's kernel used TMA (CUtensorMap) and failed with an illegal instruction:
    the harness cannot build tensor maps. The codegen (it runs in the worker image and
    imports tilelang, so it is read here, not imported) puts HARNESS_CONFIGS last."""
    import ast
    from importlib import resources

    src = resources.files("kta.backends.tilelang").joinpath("gen.py").read_text()
    tree = ast.parse(src)
    consts = {
        t.id: ast.literal_eval(n.value)
        for n in tree.body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name) and t.id == "HARNESS_CONFIGS"
    }
    assert consts["HARNESS_CONFIGS"] == {
        "tl.disable_tma_lower": True,
        "tl.disable_warp_specialized": True,
    }
    merge = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "configs"
    )  # fmt: skip
    assert (
        isinstance(merge.value, ast.Dict)
        and ast.unparse(merge.value.values[-1]) == "HARNESS_CONFIGS"
    )


def test_a_changed_build_script_invalidates_cached_builds() -> None:
    """D727: after the TileLang codegen changed (D725), builds were still served from the
    cache (the key named the kernel files, not the backend's build script)."""
    from kta.backends import common

    gk = SimpleNamespace(
        backend="tilelang", files={"kernel.py": "x"}, entry="main",
        launch=SimpleNamespace(to_dict=lambda: {"grid": [1]}),
    )  # fmt: skip
    target = SimpleNamespace(model_dump=lambda mode="json": {"name": "gb10"})
    put = SimpleNamespace(put_blob=lambda data, media: data.decode())

    def key(tc: str | None) -> str:
        return common.build_key(gk, target, [], "img", put, tc).sha256  # pyright: ignore[reportArgumentType]

    assert key("a") != key("b") != key(None)


def test_a_dead_holders_process_lock_is_taken_over(ops: Any) -> None:
    import socket
    import subprocess

    from kta.orchestrator import processes

    gone = subprocess.Popen(["true"])
    gone.wait()
    old = f"solve:X@{socket.gethostname()}:{gone.pid}"
    assert ops.acquire("process:solve", old, 3600)
    assert processes.take_lock(ops, "process:solve", processes.holder_id("solve", "Y"), 60)
    # a live holder keeps its lock
    me = processes.holder_id("solve", "Z")
    assert not processes.take_lock(ops, "process:solve", me, 60)


def test_the_host_lock_span_lane_is_valid() -> None:
    from kta.core import trace

    assert trace.LANE_RE.match("host:spark-hk")  # the measurement window's host lock span


def test_analysis_gates_on_the_model_then_names_the_bottleneck() -> None:
    from kta.orchestrator import analysis

    units = [
        {"unit": "tensor", "bound_us": 1495.0, "predicted_us": 0.02},
        {"unit": "l2", "bound_us": None, "predicted_us": 0.03},
    ]
    tie = {"name": "s0", "pairs": [(8.5, 2033.0), (8.5, 6081.0), (8.5, 3411.0)],
           "best_us": 2033.0, "bound_us": 1495.0, "units": units}  # fmt: skip
    a = analysis.analyze([tie], None)["buckets"][0]
    assert a["status"] == "model" and set(a["misaligned"]) >= {"count", "missing_term"}
    good = {"name": "s0", "pairs": [(2000.0, 2100.0), (2600.0, 2700.0), (3000.0, 3200.0)],
            "best_us": 2100.0, "bound_us": 1495.0,
            "units": [{"unit": "tensor", "bound_us": 1495.0, "predicted_us": 1500.0},
                      {"unit": "l2", "bound_us": 400.0, "predicted_us": 2000.0}]}  # fmt: skip
    b = analysis.analyze([good], None)["buckets"][0]
    assert b["aligned"] and b["bottleneck"] == "l2" and b["status"] == "inflation"
    same = analysis.analyze([good], {"buckets": [b]})["buckets"][0]
    assert same["verification"] == {**same["verification"], "outcome": "no_effect", "stuck": 1}
    assert "main contradiction" in analysis.report([b])


def test_the_recurrence_runs_once_per_wave() -> None:
    from types import SimpleNamespace as NS

    from kta.models.l1 import waves

    mm: Any = NS(hw={"sm_count": 48, "max_ctas_per_sm": 32})
    assert waves(2048, 1, mm) == 43  # one resident CTA per SM
    assert waves(None, 1, mm) == 1
    assert waves(2048, None, mm) == 2  # no residency wall: the optimistic hardware limit


def test_the_routed_task_follows_the_analyst_with_its_report() -> None:
    from kta.orchestrator import solve_loop
    from kta.orchestrator.transition import Loop

    stash = {"type": "T1", "goal": "Revise the family", "inputs": ["fam", "case"], "header": None}
    loop = Loop("L", "solve", "case", "waiting", {"after_analysis": stash})
    task = solve_loop.follow_up(loop, "## Main contradiction\nl2 does 3x the work")
    assert task.type == "T1" and task.inputs == ["fam", "case"]
    assert task.goal.startswith("Revise the family") and "l2 does 3x" in task.goal


def test_the_analysis_report_needs_its_sections_and_a_direction(tmp_path: Path) -> None:
    from kta.orchestrator import done_checks

    inp: Any = SimpleNamespace(workdir=tmp_path)
    assert not done_checks.analysis_report(inp).passed  # no file
    sections = ["Main contradiction", "Direction", "Expected effect", "What would refute it",
                "Per shape"]  # fmt: skip
    (tmp_path / "analysis.md").write_text(
        "\n".join(f"## {s}\ntext" for s in sections).replace(
            "## Direction\ntext", "## Direction\nnew technique: swizzle"
        )
    )
    assert done_checks.analysis_report(inp).passed


def test_residency_in_the_projection_reads_only_the_storage() -> None:
    """SYSDEBUG12: the recurrence's wave factor evaluated every transaction, so a variable
    outside the atom's support (threads) raised ExprError at every combination and the MIP
    excluded the whole space (Infeasible while enumeration found 30 points)."""
    from types import SimpleNamespace as NS

    from kta.core.expr import Expr
    from kta.formulation.blocks import BoundStorage
    from kta.solve import projection

    store = BoundStorage("ops", "buf", "smem", "cta", Expr.parse("BM * 4", {"BM"}), None)
    needs_threads = NS(count=Expr.parse("threads * 2", {"threads"}))
    f: Any = NS(storage=[store], transactions=[needs_threads])
    c = projection._Ctx(f, NS(walls=[], hw={}), {}, {}, {})  # type: ignore[arg-type]
    assert projection._residency(c, {"BM": 64}) is None  # no ExprError for 'threads'
