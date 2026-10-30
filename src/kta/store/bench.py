"""Store benchmark: query and full-text latency on a large synthetic store
(spec/modules/core-store.md "真实运行").

The synthetic records are stored with origin="fixture"; the timings are real and are
written as a Report record (origin="real") that references the EnvManifest of the
machine that ran it, with every raw latency kept as a blob. (decision D156)
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from kta.core.ids import record_id
from kta.core.schemas import Actor, Record, RefValue
from kta.store.sqlite import SqliteStore

THRESHOLD_MS = 50.0
BATCH = 5000
ACTOR = Actor(type="tool", name="kta store bench")
DOMAIN_WORDS = (
    "gemm attention decode prefill split reduce smem tma wgmma warp cta tile swizzle pipeline "
    "async transfer barrier epilogue softmax rescale bf16 fp8 layout bank conflict occupancy "
    "register spill latency bandwidth l2 dram tensor fma issue launch persistent stream-k "
    "split-k cluster multicast predicate vectorize unroll prefetch double-buffer"
).split()
# kind -> share of the synthetic records
MIX = {
    "Pattern": 0.05,
    "Gap": 0.30,
    "Negative": 0.25,
    "HwFact": 0.20,
    "Structure": 0.05,
    "Candidate": 0.15,
}


@dataclass(frozen=True)
class BenchResult:
    data: dict[str, Any]
    samples: dict[str, list[float]]


class _Words:
    """Zipf-like word picker over domain words plus a synthetic vocabulary."""

    def __init__(self, rng: random.Random, vocab: int = 3000) -> None:
        self.rng = rng
        self.words = [*DOMAIN_WORDS, *(f"w{i:04d}" for i in range(vocab))]
        self.weights = [1.0 / (i + 1) for i in range(len(self.words))]

    def text(self, n: int) -> str:
        return " ".join(self.rng.choices(self.words, self.weights, k=n))


# kta:fixture synthetic bodies for the store benchmark (core-store.md), stored as origin=fixture
def _body(kind: str, i: int, words: _Words, rng: random.Random) -> dict[str, Any]:
    if kind == "Pattern":
        return {
            "name": f"pattern_{i}",
            "level": i % 6,
            "category": "other",
            "description": words.text(30),
            "examples": [words.text(6), words.text(6)],
        }
    if kind == "Gap":
        return {
            "missing": rng.choice(["template", "hw_fact", "check", "other"]),
            "description": words.text(25),
            "level": i % 6,
            "attribution": rng.choice(["space", "model", "process", "other"]),
        }
    if kind == "Negative":
        return {
            "subject": f"pattern_{rng.randrange(1000)}",
            "condition": words.text(12),
            "effect": words.text(12),
        }
    if kind == "HwFact":
        return {
            "quantity": words.text(3),
            "value": rng.random() * 1000,
            "unit": "smem",
            "method": "measured",
        }
    if kind == "Structure":
        return {
            "name": f"structure_{i}",
            "effect": "variables",
            "description": words.text(20),
            "evidence_needed": words.text(8),
        }
    return {"assignment": {"bm": 2 ** rng.randrange(4, 9), "stages": rng.randrange(1, 6)}}


def synthetic_records(n: int, seed: int, start: datetime, actor: Actor) -> Iterator[Record]:
    """Refs point at earlier records, so ``actor`` must already carry commit and dirty."""
    rng = random.Random(seed)
    words = _Words(rng)
    kinds = list(MIX)
    weights = list(MIX.values())
    patterns: list[str] = []
    for i in range(n):
        kind = "Pattern" if i < 50 else rng.choices(kinds, weights)[0]
        refs: dict[str, RefValue] = {}
        if kind in ("Gap", "Negative") and patterns:
            refs["trigger"] = rng.choice(patterns)
        rec = Record.new(
            kind,
            _body(kind, i, words, rng),
            actor,
            start + timedelta(milliseconds=i),
            refs=refs,
            origin="fixture",
        )
        if kind == "Pattern":
            patterns.append(record_id(rec))
        yield rec


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def _queries(store: SqliteStore, rng: random.Random) -> dict[str, Callable[[], object]]:
    patterns = [r.body["name"] for r in store.query("Pattern", limit=2000)]
    pattern_ids = [record_id(r) for r in store.query("Pattern", limit=2000)]
    gaps = [record_id(r) for r in store.query("Gap", order="-created_at", limit=2000)]
    words = _Words(rng, vocab=300)
    return {
        "get": lambda: store.get(rng.choice(gaps)),
        "query_name_eq": lambda: store.query("Pattern", {"name": rng.choice(patterns)}),
        "query_filter_recent": lambda: store.query(
            "Gap", {"attribution": rng.choice(["model", "space"])}, order="-created_at"
        ),
        "query_level_range": lambda: store.query(
            "Gap", {"level": {">=": rng.randrange(6)}}, limit=100
        ),
        "neighbors_in": lambda: store.neighbors(rng.choice(pattern_ids), direction="in"),
        "lineage": lambda: store.lineage(rng.choice(gaps), depth=3),
        "search_1word": lambda: store.search(words.text(1)),
        "search_2words": lambda: store.search(words.text(2)),
        "search_kinds": lambda: store.search(words.text(1), kinds=["Gap", "Negative"]),
    }


def run(
    store: SqliteStore,
    n_records: int,
    n_queries: int,
    seed: int,
    start: datetime,
    timer: Callable[[], float] = time.perf_counter,
) -> BenchResult:
    """Fill ``store`` with ``n_records`` synthetic records and time each query type.

    ``timer`` measures elapsed durations (monotonic); it is not a wall clock.
    """
    t0 = timer()
    batch: list[Record] = []
    before = store.count()
    actor = ACTOR.model_copy(update={"commit": store.code.commit, "dirty": store.code.dirty})
    for rec in synthetic_records(n_records, seed, start, actor):
        batch.append(rec)
        if len(batch) >= BATCH:
            store.put_many(batch)
            batch = []
    store.put_many(batch)
    build_s = timer() - t0
    rng = random.Random(seed + 1)
    samples: dict[str, list[float]] = {}
    for name, fn in _queries(store, rng).items():
        fn()  # warm the page cache once per query type
        times: list[float] = []
        for _ in range(n_queries):
            q0 = timer()
            fn()
            times.append((timer() - q0) * 1000.0)
        samples[name] = times
    stats = {
        name: {
            "n": len(t),
            "p50_ms": round(statistics.median(t), 3),
            "p95_ms": round(_percentile(t, 0.95), 3),
            "p99_ms": round(_percentile(t, 0.99), 3),
            "max_ms": round(max(t), 3),
        }
        for name, t in samples.items()
    }
    data = {
        "n_records_requested": n_records,
        "records_in_store": store.count(),
        "records_inserted": store.count() - before,
        "by_kind": {k: store.count(k) for k in MIX},
        "build_s": round(build_s, 3),
        "seed": seed,
        "queries_per_type": n_queries,
        "threshold_p95_ms": THRESHOLD_MS,
        "queries": stats,
        "max_p95_ms": max(s["p95_ms"] for s in stats.values()),
        "passed": all(s["p95_ms"] < THRESHOLD_MS for s in stats.values()),
        "db_bytes": Path(store.db.path).stat().st_size,
    }
    return BenchResult(data=data, samples=samples)


def report_record(
    store: SqliteStore, result: BenchResult, env_manifest_id: str, now: datetime
) -> Record:
    blob = store.put_blob(
        json.dumps(result.samples, sort_keys=True).encode("utf-8"), "application/json"
    )
    d = result.data
    summary = (
        f"{d['records_in_store']} records; p95 over {len(d['queries'])} query types "
        f"max {d['max_p95_ms']} ms (threshold {THRESHOLD_MS} ms): "
        f"{'passed' if d['passed'] else 'FAILED'}"
    )
    body = {"title": "store benchmark", "subject": "store-benchmark", "summary": summary, "data": d}
    return Record.new(
        "Report", body, ACTOR, now, refs={"env": env_manifest_id}, blobs={"latencies_ms": blob}
    )
