"""WP1 acceptance (spec/modules/core-store.md "验收", unit-test part)."""

from __future__ import annotations

import dataclasses
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from kta.core.config import config_from_mapping
from kta.core.context import Context
from kta.core.errors import PermissionDenied, SchemaError
from kta.core.ids import canonical_json, record_id
from kta.core.registry import Registry
from kta.core.schemas import Actor, Record
from kta.store import CodeState, open_ops, open_store, register_implementations, stale
from kta.store.sqlite import SqliteStore
from storehelp import T0, TEST_COMMIT, actor, gap, pattern, status_change

pytestmark = pytest.mark.acceptance

# A record with Chinese text and floats; its ID must be the same in every process and on
# every machine, so the expected value is fixed here.
STABLE = {
    "kind": "HwFact",
    "schema": 1,
    "created_at": "2026-10-30T05:00:00.123456Z",
    "actor": {"type": "tool", "name": "测量工具", "task": None, "commit": "a" * 40, "dirty": False},
    "refs": {},
    "body": {
        "feature": None,
        "unit": "smem",
        "quantity": "带宽 bandwidth",
        "value": 0.1 + 0.2,
        "ci_low": 1e-300,
        "ci_high": 123456789.125,
        "method": "measured",
    },
    "blobs": {},
    "origin": "real",
}
STABLE_ID = "b1082f47da4ca8c516e8a14a6e7fcc53b83f62f550bb601f75261ce190432a63"


def _main_store(ctx: Context, dirty: bool) -> SqliteStore:
    return open_store(dataclasses.replace(ctx, env="main"), CodeState(TEST_COMMIT, dirty))


def test_main_env_rejects_fixture_and_simulated_records(ctx: Context) -> None:
    store = _main_store(ctx, dirty=False)
    for origin in ("fixture", "simulated"):
        with pytest.raises(PermissionDenied, match="origin"):
            store.put(pattern("p", origin=origin))
    assert store.count() == 0
    store.put(pattern("p", origin="real"))
    assert store.count() == 1


def test_main_env_rejects_dirty_writes(ctx: Context) -> None:
    with pytest.raises(PermissionDenied, match="dirty"):
        _main_store(ctx, dirty=True).put(pattern("p"))
    dirty_actor = Actor(type="tool", name="t", commit=TEST_COMMIT, dirty=True)
    with pytest.raises(PermissionDenied, match="dirty"):
        _main_store(ctx, dirty=False).put(pattern("q", who=dirty_actor))


def test_actor_carries_commit_and_dirty_in_stored_json(store: SqliteStore) -> None:
    rid = store.put(pattern("p"))
    row = store.db.con.execute("SELECT json FROM records WHERE id = ?", (rid,)).fetchone()
    assert f'"commit":"{TEST_COMMIT}"' in row[0] and '"dirty":false' in row[0]


def fixture_store_factory(ctx: Context) -> str:
    return "a fixture store"


def test_registry_rejects_fixture_store_in_dev(ctx: Context) -> None:
    registry = Registry()
    register_implementations(registry)
    registry.register("store", "memory", fixture_store_factory, fixture=True)
    cfg = config_from_mapping({"impl": {"store": "memory"}})
    dev = dataclasses.replace(ctx, env="dev", config=cfg, registry=registry)
    with pytest.raises(PermissionDenied, match="fixture"):
        registry.resolve("store", dev)
    test_env = dataclasses.replace(dev, env="test")
    assert registry.resolve("store", test_env) == "a fixture store"


def test_record_id_is_stable_across_processes() -> None:
    assert record_id(STABLE) == STABLE_ID
    assert record_id(Record.model_validate(STABLE)) == STABLE_ID
    code = (
        "import json,sys; from kta.core.ids import record_id; "
        "print(record_id(json.loads(sys.stdin.read())))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        input=canonical_json(STABLE).decode("utf-8"),
        capture_output=True,
        text=True,
        check=True,
        env={"PYTHONHASHSEED": "123", "PATH": ""},
    )
    assert out.stdout.strip() == STABLE_ID


def test_update_and_delete_are_refused_by_triggers(store: SqliteStore) -> None:
    rid = store.put(pattern("p"))
    child = store.put(gap("has an edge", {"trigger": rid}))
    for sql, key in (
        ("UPDATE records SET origin = 'fixture' WHERE id = ?", rid),
        ("DELETE FROM records WHERE id = ?", rid),
        ("UPDATE edges SET rel = 'x' WHERE src = ?", child),
        ("DELETE FROM edges WHERE src = ?", child),
    ):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.db.con.execute(sql, (key,))
    assert store.get(rid).origin == "real"
    assert store.neighbors(child) == [rid]


def test_put_fails_when_refs_are_missing(store: SqliteStore) -> None:
    missing = "f" * 64
    with pytest.raises(SchemaError, match="refs not in the store"):
        store.put(gap("needs a trigger", refs={"trigger": missing}))
    assert store.count() == 0
    trigger = store.put(pattern("p"))
    store.put(gap("needs a trigger", refs={"trigger": trigger}))


RULES = [
    # (setup changes, change, actor, allowed)
    ([], ("reproduced", None), "checker", True),
    ([], ("calibrated", None), "checker", True),
    ([], ("holdout_validated", None), "checker", True),
    ([], ("reproduced", None), "agent", False),
    ([], ("reproduced", None), "human", False),
    ([("reproduced", "checker")], ("calibrated", None), "checker", True),
    ([("calibrated", "checker")], ("reproduced", None), "checker", False),
    ([], ("rejected", None), "human", True),
    ([], ("deprecated", None), "human", True),
    ([], ("suspect", None), "human", True),
    ([], ("rejected", None), "checker", False),
    ([], ("suspect", None), "agent", False),
    ([], ("stale", None), "system", True),
    ([], ("stale", None), "human", False),
    ([("stale", "system")], ("reproduced", None), "checker", True),
    ([("rejected", "human")], ("reproduced", None), "checker", False),
    ([], ("proposed", None), "agent", True),
    ([("rejected", "human")], ("proposed", None), "agent", False),
    ([("rejected", "human")], ("proposed", None), "human", True),
    ([], (None, "tested"), "checker", True),
    ([], (None, "proved"), "checker", True),
    ([], (None, "tested"), "agent", False),
    ([], (None, "tested"), "human", False),
]


@pytest.mark.parametrize(("setup", "change", "who", "allowed"), RULES)
def test_status_promotion_rules(
    store: SqliteStore,
    setup: list[tuple[str, str]],
    change: tuple[str | None, str | None],
    who: str,
    allowed: bool,
) -> None:
    subject = store.put(pattern("subject"))
    for to_status, by in setup:
        store.put(status_change(subject, by, to_status=to_status))  # type: ignore[arg-type]
    before = store.status(subject)
    record = status_change(subject, who, to_status=change[0], to_assurance=change[1])  # type: ignore[arg-type]
    if allowed:
        store.put(record)
        status, assurance = store.status(subject)
        assert status == (change[0] or before[0])
        assert assurance == (change[1] or before[1])
    else:
        with pytest.raises(PermissionDenied):
            store.put(record)
        assert store.status(subject) == before


def test_assurance_never_moves_down(store: SqliteStore) -> None:
    subject = store.put(pattern("subject"))
    store.put(status_change(subject, "checker", to_assurance="bounded"))
    with pytest.raises(PermissionDenied):
        store.put(status_change(subject, "checker", to_assurance="tested"))
    assert store.status(subject) == ("proposed", "bounded")


def test_stale_propagation_on_five_level_graph(store: SqliteStore) -> None:
    # L0 <- L1a, L1b <- L2 (refs both) <- L3 <- L4; plus an unrelated record.
    l0 = store.put(pattern("root"))
    l1a = store.put(gap("level 1a", {"trigger": l0}, 1))
    l1b = store.put(gap("level 1b", {"trigger": l0}, 2))
    l2 = store.put(gap("level 2", {"a": l1a, "b": l1b}, 3))
    l3 = store.put(gap("level 3", {"up": l2}, 4))
    l4 = store.put(gap("level 4", {"up": l3}, 5))
    unrelated = store.put(pattern("unrelated"))
    human_rejected = store.put(gap("level 2 rejected by a human", {"up": l1a}, 6))
    store.put(status_change(human_rejected, "human", to_status="rejected"))

    marked = stale.reject(store, l0, "wrong semantics", actor("human"), T0)

    assert set(marked) == {l1a, l1b, l2, l3, l4}
    assert len(marked) == len(set(marked))
    assert store.status(l0)[0] == "rejected"
    for rid in (l1a, l1b, l2, l3, l4):
        assert store.status(rid)[0] == "stale"
    assert store.status(unrelated)[0] == "proposed"
    assert store.status(human_rejected)[0] == "rejected"
    # Propagating again finds nothing new to mark.
    assert stale.propagate(store, l0, "again", T0) == []


def test_concurrent_cas_pointer_update_has_one_winner(ctx: Context) -> None:
    contenders = [open_ops(ctx) for _ in range(8)]
    contenders[0].set_pointer("current_pool", None, "base")
    barrier = threading.Barrier(len(contenders))
    results: list[bool] = []
    lock = threading.Lock()

    def attempt(i: int) -> None:
        barrier.wait()
        ok = contenders[i].set_pointer("current_pool", "base", f"pool-{i}")
        with lock:
            results.append(ok)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(len(contenders))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [False] * (len(contenders) - 1) + [True]
    winner = contenders[0].get_pointer("current_pool")
    assert winner is not None and winner.startswith("pool-")


def test_store_files_live_in_the_context_data_dir(ctx: Context, store: SqliteStore) -> None:
    store.put(pattern("p"))
    assert (Path(ctx.data_dir) / "kta.db").exists()
