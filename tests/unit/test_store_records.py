from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from kta.core.context import Context
from kta.core.errors import ConfigError, PermissionDenied, SchemaError
from kta.core.schemas import Actor, Record
from kta.store import CodeState, open_ops, open_store, snapshot
from kta.store.sqlite import RecordNotFound, SqliteStore
from storehelp import T0, TEST_COMMIT, actor, gap, pattern


def test_put_is_idempotent_and_get_round_trips(store: SqliteStore) -> None:
    rec = pattern("reduce")
    rid = store.put(rec)
    assert store.put(rec) == rid
    assert store.count() == 1
    got = store.get(rid)
    assert got.body == rec.body and got.actor.commit == TEST_COMMIT and got.actor.dirty is False
    with pytest.raises(RecordNotFound):
        store.get("0" * 64)


def test_worker_provenance_is_kept(store: SqliteStore) -> None:
    worker = Actor(type="tool", name="worker", commit="b" * 40, dirty=False)
    rid = store.put(pattern("p", who=worker))
    assert store.get(rid).actor.commit == "b" * 40


def test_dev_env_accepts_real_and_fixture_only(ctx: Context) -> None:
    dev = open_store(dataclasses.replace(ctx, env="dev"), CodeState(TEST_COMMIT, False))
    dev.put(pattern("a", origin="fixture"))
    dev.put(pattern("b", origin="real"))
    for origin in ("simulated", "historical"):
        with pytest.raises(PermissionDenied):
            dev.put(pattern("c", origin=origin))


def test_ref_kinds_are_checked(store: SqliteStore) -> None:
    p = store.put(pattern("p"))
    body = {"members": [], "note": "x"}
    with pytest.raises(SchemaError, match="must point to PoolSnapshot"):
        store.put(Record.new("PoolSnapshot", body, actor(), T0, refs={"previous": p}))


def test_blobs_must_exist_and_are_content_addressed(store: SqliteStore, tmp_path: Path) -> None:
    sha = store.put_blob(b"hello", "text/plain")
    assert store.put_blob(b"hello", "text/plain") == sha
    src = tmp_path / "big.bin"
    src.write_bytes(b"x" * 3_000_000)
    big = store.put_blob(src, "application/octet-stream")
    assert store.blob_path(big).read_bytes() == src.read_bytes()
    assert store.blob_path(sha).parent.name == sha[:2]
    assert store.blobs.verify(sha)
    body = {"title": "t", "subject": "s", "summary": "x"}
    store.put(Record.new("Report", body, actor(), T0, blobs={"raw": sha}))
    with pytest.raises(SchemaError, match="not stored"):
        store.put(Record.new("Report", body, actor(), T0, blobs={"raw": "e" * 64}))


def test_query_filters_and_order(store: SqliteStore) -> None:
    ids = [store.put(gap(f"gap number {i}", n=i)) for i in range(6)]
    store.put(pattern("p"))
    assert [store.put(gap("gap number 0", n=0))] == ids[:1]
    assert len(store.query("Gap")) == 6
    assert [r.body["level"] for r in store.query("Gap", {"level": {">=": 4}})] == [4, 5]
    assert len(store.query("Gap", {"level": {"in": [1, 3]}})) == 2
    assert len(store.query("Gap", {"level": {"!=": 1}})) == 5
    assert len(store.query("Gap", {"pattern_guess": None})) == 6
    assert len(store.query("Gap", {"actor.name": "test", "origin": "real"})) == 6
    assert store.query("Gap", order="-body.level", limit=2)[0].body["level"] == 5
    assert store.query("Pattern", {"name": "p"})[0].body["name"] == "p"
    for bad in ({"name; DROP": 1}, {"level": {"~": 1}}, {"level": {"in": []}}):
        with pytest.raises(ConfigError):
            store.query("Gap", bad)
    with pytest.raises(SchemaError, match="unknown record kind"):
        store.query("Nope")


def test_query_contains_on_list_refs(store: SqliteStore) -> None:
    a, b = store.put(pattern("a")), store.put(pattern("b"))
    snap = store.put(
        Record.new(
            "PoolSnapshot", {"members": [a, b], "note": "n"}, actor(), T0, refs={"members": [a, b]}
        )
    )
    assert [store.query("PoolSnapshot", {"refs.members": {"contains": a}})[0]] == [store.get(snap)]


def test_search_ranks_titles_and_filters_kinds(store: SqliteStore) -> None:
    store.put(pattern("rescale", "online softmax keeps a running max"))
    store.put(pattern("other", "split the reduction and rescale partial sums"))
    store.put(gap("no template for rescale epilogues"))
    for i in range(10):  # documents without the term, so its idf is positive
        store.put(pattern(f"filler{i}", "unrelated words about tiles"))
    hits = store.search("rescale")
    titles = [h.title for h in hits]
    assert len(hits) == 3
    # Same kind, similar length: the title match outranks a match in the description.
    assert titles.index("rescale") < titles.index("other")
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert {h.kind for h in store.search("rescale", kinds=["Gap"])} == {"Gap"}
    assert "[rescale]" in store.search("rescale epilogues")[0].snippet
    assert store.search('"; DROP TABLE records; --') == []
    assert store.search("") == []


def test_neighbors_and_lineage(store: SqliteStore) -> None:
    root = store.put(pattern("root"))
    mid = store.put(gap("mid", {"trigger": root}))
    leaf = store.put(gap("leaf", {"up": mid, "also": [root]}))
    assert store.neighbors(leaf) == sorted([mid, root])
    assert store.neighbors(leaf, rel="up") == [mid]
    assert store.neighbors(root, direction="in") == sorted([mid, leaf])
    graph = store.lineage(leaf)
    assert set(graph.nodes) == {leaf, mid, root} and graph.nodes[root] == "Pattern"
    assert (leaf, "up", mid) in graph.edges
    assert set(store.lineage(leaf, depth=1).nodes) == {leaf, mid, root}
    assert set(store.lineage(mid, depth=0).nodes) == {mid}


def test_snapshot_writes_record_and_moves_pointer(ctx: Context, store: SqliteStore) -> None:
    ops = open_ops(ctx)
    a = store.put(pattern("a"))
    first = snapshot.snapshot(store, ops, [a], None, "first", actor("system"), T0)
    assert ops.get_pointer(snapshot.CURRENT_POOL) == first
    second = snapshot.snapshot(store, ops, [a], first, "second", actor("system"), T0)
    assert store.neighbors(second, rel="previous") == [first]
    with pytest.raises(snapshot.SnapshotConflict):
        snapshot.snapshot(store, ops, [], first, "stale parent", actor("system"), T0)
    assert ops.get_pointer(snapshot.CURRENT_POOL) == second
