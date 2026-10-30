from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from fakes.clock import ManualClock
from kta.core.errors import SchemaError
from kta.core.ids import UlidFactory, canonical_json, record_id, ulid_time
from kta.core.schemas import KINDS, Actor, InvalidBody, Record, validate_body
from storehelp import T0, pattern, pattern_body


def test_canonical_json_rules() -> None:
    data = {"b": 1.5, "a": "中文", "c": [0.1, 2], "d": datetime(2026, 1, 1, tzinfo=UTC)}
    assert canonical_json(data) == (
        '{"a":"中文","b":1.5,"c":[0.1,2],"d":"2026-01-01T00:00:00+00:00"}'.encode()
    )
    with pytest.raises(SchemaError):
        canonical_json({"x": float("nan")})
    with pytest.raises(SchemaError):
        canonical_json({"x": {1, 2}})
    with pytest.raises(SchemaError):
        canonical_json({"x": datetime(2026, 1, 1)})


def test_body_defaults_do_not_change_the_id() -> None:
    sparse = Record.new("Pattern", pattern_body("p"), Actor(type="tool", name="t"), T0)
    full_body = {**pattern_body("p"), "params": []}
    full = Record.new("Pattern", full_body, Actor(type="tool", name="t"), T0)
    assert record_id(sparse) == record_id(full)


def test_created_at_is_normalized_to_utc() -> None:
    plus8 = timezone(timedelta(hours=8))
    local = Record.new(
        "Pattern", pattern_body("p"), Actor(type="tool", name="t"), T0.astimezone(plus8)
    )
    assert record_id(local) == record_id(pattern("p", who=Actor(type="tool", name="t")))
    with pytest.raises(ValueError, match="timezone"):
        Record.new("Pattern", pattern_body("p"), Actor(type="tool", name="t"), datetime(2026, 1, 1))


def test_record_round_trips_through_json() -> None:
    rec = pattern("p")
    again = Record.model_validate(rec.to_json_dict())
    assert record_id(again) == record_id(rec)
    assert rec.to_json_dict()["schema"] == KINDS["Pattern"].version


def test_validate_body_raises_value_error_subclass() -> None:
    validate_body("Pattern", pattern_body("ok"))
    with pytest.raises(ValueError) as info:
        validate_body("Pattern", {**pattern_body("x"), "examples": ["only one"]})
    assert isinstance(info.value, InvalidBody) and isinstance(info.value, SchemaError)
    with pytest.raises(ValueError, match="unknown record kind"):
        validate_body("NoSuchKind", {})
    with pytest.raises(ValueError, match="extra"):
        validate_body("Pattern", {**pattern_body("x"), "typo_field": 1})


def test_envelope_rules() -> None:
    actor = Actor(type="tool", name="t")
    with pytest.raises(ValueError, match="not a record ID"):
        Record.new(
            "Gap",
            {"missing": "other", "description": "d", "attribution": "other"},
            actor,
            T0,
            refs={"x": "nope"},
        )
    with pytest.raises(ValueError, match="needs refs"):
        Record.new(
            "Measurement",
            {"median_us": 1.0, "ci_low_us": 1.0, "ci_high_us": 1.0, "n": 3, "protocol": "p"},
            actor,
            T0,
        )
    with pytest.raises(ValueError, match="set together"):
        Actor(type="tool", name="t", commit="a" * 40)
    with pytest.raises(ValueError, match="schema"):
        Record.model_validate({**pattern("p").to_json_dict(), "schema": 99})


TEMPLATE: dict[str, Any] = {
    "name": "async_transfer",
    "version": 1,
    "level_from": 2,
    "level_to": 3,
    "change": "vertical",
    "refines": "transfer",
    "produces": ["transfer", "sync"],
    "object": {"type": "smem_slot", "description": "a shared-memory slot"},
    "variables": [
        {
            "name": "stages",
            "kind": "numeric",
            "binding": "compile",
            "role": "software",
            "domain": {"lo": 1, "hi": 8},
            "meaning": "pipeline depth",
        },
        {
            "name": "swizzle",
            "kind": "structure",
            "binding": "compile",
            "role": "software",
            "domain": [True, False],
            "meaning": "swizzled layout",
        },
    ],
    "constraints": [
        {
            "id": "c1",
            "expr": "stages * p.tile_bytes <= hw.smem_per_cta",
            "meaning": "fits",
            "cls": "resource",
        },
    ],
    "program": {"description": "cp.async ring buffer"},
    "backward_map": "slot i holds tile k mod stages",
    "scope": {"predicate": "hw.has_cp_async", "text": "GPUs with cp.async"},
    "signature": {"algorithm": "ring buffer"},
    "summary": "Overlap global loads with compute through a shared-memory ring.",
}


def test_template_requires_sources_unless_human() -> None:
    validate_body("Template", TEMPLATE)
    with pytest.raises(ValueError, match="sources"):
        Record.new("Template", TEMPLATE, Actor(type="agent", name="w"), T0)
    Record.new("Template", TEMPLATE, Actor(type="human", name="me"), T0)


def test_every_kind_has_a_model_and_version() -> None:
    for kind, spec in KINDS.items():
        assert spec.version >= 1, kind
        assert spec.model.model_config.get("extra") == "forbid", kind


def test_ulid_factory() -> None:
    clock = ManualClock()
    counter = iter(range(1000))
    ids = UlidFactory(clock, lambda n: next(counter).to_bytes(n, "big"))
    a, b = ids.new(), ids.new()
    assert len(a) == 26 and a != b and a < b
    assert ulid_time(a) == clock.now()
