from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from kta.core.context import Context
from kta.core.errors import SchemaError
from kta.core.schemas import Record
from kta.store import CodeState, bench, open_store, transfer
from kta.store.sqlite import SqliteStore
from storehelp import T0, actor, gap, pattern


def _other_store(ctx: Context, tmp_path: Path) -> SqliteStore:
    other = dataclasses.replace(ctx, data_dir=tmp_path / "control")
    return open_store(other, CodeState("d" * 40, True))


def test_bundle_moves_records_with_their_refs_and_blobs(
    ctx: Context, store: SqliteStore, tmp_path: Path
) -> None:
    root = store.put(pattern("root"))
    sha = store.put_blob(b"raw samples", "application/json")
    body = {"title": "t", "subject": "s", "summary": "x"}
    leaf = store.put(Record.new("Report", body, actor(), T0, refs={"base": root}, blobs={"r": sha}))
    store.put(gap("unrelated"))
    info = transfer.export_bundle(store, [leaf], tmp_path / "b.jsonl")
    assert info["records"] == [root, leaf] and info["blobs"] == [sha]

    control = _other_store(ctx, tmp_path)
    result = transfer.import_bundle(control, tmp_path / "b.jsonl")
    assert result["records"] == [root, leaf]
    # Provenance of the producing side is kept, not replaced by the importer's code state.
    assert control.get(leaf).actor.commit == store.code.commit
    assert control.blob_path(sha).read_bytes() == b"raw samples"
    assert control.count() == 2


def test_tampered_bundle_is_refused(ctx: Context, store: SqliteStore, tmp_path: Path) -> None:
    rid = store.put(pattern("p", "original text"))
    path = tmp_path / "b.jsonl"
    transfer.export_bundle(store, [rid], path)
    text = path.read_text(encoding="utf-8").replace("original text", "edited text")
    path.write_text(text, encoding="utf-8")
    control = _other_store(ctx, tmp_path)
    with pytest.raises(SchemaError, match="does not match its ID"):
        transfer.import_bundle(control, path)
    assert control.count() == 0


def test_bench_small_run_writes_report(store: SqliteStore) -> None:
    ticks = iter(float(i) / 1000 for i in range(10**6))
    result = bench.run(
        store, n_records=600, n_queries=5, seed=3, start=T0, timer=lambda: next(ticks)
    )
    data = result.data
    assert data["records_in_store"] == 600 and data["records_inserted"] == 600
    assert set(data["queries"]) == set(result.samples)
    assert all(len(v) == 5 for v in result.samples.values())
    # Synthetic records are fixtures; re-running adds nothing (same content, same IDs).
    assert {r.origin for r in store.query("Gap", limit=10)} == {"fixture"}
    again = bench.run(store, 600, 1, 3, T0, timer=lambda: next(ticks))
    assert again.data["records_inserted"] == 0
    env = store.put(
        Record.new(
            "EnvManifest",
            {
                "role": "worker",
                "host": "h",
                "os": "o",
                "arch": "a",
                "python": "3.12",
                "uv": "0.11",
                "uv_lock_sha256": "x",
                "commit": "c",
                "dirty": False,
            },
            actor(),
            T0,
        )
    )
    report = store.put(bench.report_record(store, result, env, T0))
    rec = store.get(report)
    assert rec.origin == "real" and rec.refs == {"env": env}
    samples = json.loads(store.blob_path(rec.blobs["latencies_ms"]).read_text())
    assert samples == result.samples
