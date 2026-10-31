"""Round baselines must preserve shape/protocol identity, including after later runs."""

import json
from dataclasses import replace

import pytest

from kta.core.context import Context
from kta.core.errors import SchemaError
from kta.core.schemas import Actor, Record
from kta.measure.protocol import ensure_v1
from kta.orchestrator.pipeline import Services
from kta.orchestrator.solve_baseline import baseline_us
from kta.orchestrator.transition import Loop
from kta.store.base import Store
from kta.store.sqlite_ops import SqliteOps
from orchesthelp import world

SHAPES = [{"M": 4096, "N": 4096, "K": 4096}, {"M": 8192, "N": 1024, "K": 4096}]


def measurement(
    ctx: Context, store: Store, refs: dict[str, str], shape: dict[str, int], us: float
) -> str:
    params = store.put_blob(json.dumps({"shape": shape}).encode(), "application/json")
    return store.put(
        Record.new(
            "Measurement",
            {
                "median_us": us,
                "ci_low_us": us,
                "ci_high_us": us,
                "n": 20,
                "protocol": refs["protocol"],
            },
            Actor(type="tool", name="test"),
            ctx.clock.now(),
            refs=refs,
            blobs={"params": params},
        )
    )


def test_baseline_uses_this_rounds_shape_even_after_newer_measurements(
    ctx: Context,
    store: Store,
    ops: SqliteOps,
) -> None:
    w = world(Services(ctx, store, ops), hw=False)

    def put(kind: str, body: dict[str, object]) -> str:
        return store.put(Record.new(kind, body, Actor(type="tool", name="test"), ctx.clock.now()))

    refs = {
        "case": w["case"],
        "baseline": put("Report", {"title": "baseline", "subject": "baseline", "summary": "test"}),
        "protocol": ensure_v1(store, ctx.clock),
        "env": put(
            "EnvManifest",
            {
                "role": "worker",
                "host": "test",
                "os": "test",
                "arch": "test",
                "python": "3",
                "uv": "test",
                "uv_lock_sha256": "x",
                "commit": "c",
                "dirty": False,
            },
        ),
        "worker": put(
            "Worker",
            {
                "name": "test",
                "host": "test",
                "transport": "local",
                "labels": {},
                "devices": [],
                "commit": "c",
            },
        ),
    }
    ids = {
        f"s{i}": measurement(ctx, store, refs, shape, us)
        for i, (shape, us) in enumerate(zip(SHAPES, [100.0, 60.0], strict=True))
    }
    loop = Loop(
        "round",
        "solve",
        refs["case"],
        "judging",
        {"protocol": refs["protocol"], "baseline_measurements": ids},
    )
    measurement(ctx, store, refs, SHAPES[0], 999.0)
    assert baseline_us(store, loop, "s0", SHAPES[0], refs["baseline"]) == 100.0
    assert baseline_us(store, loop, "s1", SHAPES[1], refs["baseline"]) == 60.0
    assert baseline_us(store, loop, "s0", SHAPES[0], None) is None
    with pytest.raises(SchemaError, match="shape mismatch"):
        baseline_us(store, loop, "s0", SHAPES[1], refs["baseline"])
    with pytest.raises(SchemaError, match="identity mismatch"):
        baseline_us(
            store,
            replace(loop, data={**loop.data, "protocol": "d" * 64}),
            "s0",
            SHAPES[0],
            refs["baseline"],
        )
    with pytest.raises(SchemaError, match="no baseline measurement"):
        baseline_us(store, loop, "s2", SHAPES[0], refs["baseline"])
