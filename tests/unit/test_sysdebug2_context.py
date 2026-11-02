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
