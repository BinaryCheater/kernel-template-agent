"""Resolve a round's library baseline without crossing shapes or protocols."""

from __future__ import annotations

import json
from typing import Any

from kta.core.errors import SchemaError
from kta.orchestrator.transition import Loop
from kta.store.base import Store


def baseline_us(
    store: Store, loop: Loop, name: str, shape: dict[str, Any], baseline: str | None
) -> float | None:
    if baseline is None:
        return None
    mid = loop.data.get("baseline_measurements", {}).get(name)
    if mid is None:
        raise SchemaError(f"bucket {name}: no baseline measurement in this round")
    rec = store.get(mid)
    if (
        rec.kind != "Measurement"
        or rec.refs.get("case") != loop.subject
        or rec.refs.get("baseline") != baseline
        or rec.refs.get("protocol") != loop.data["protocol"]
    ):
        raise SchemaError(f"bucket {name}: baseline measurement identity mismatch")
    meta = json.loads(store.blob_path(rec.blobs["params"]).read_bytes())
    if meta["shape"] != shape:
        raise SchemaError(f"bucket {name}: baseline measurement shape mismatch")
    return float(rec.body["median_us"])
