"""Template effects: what a refinement step does to cost, in the transaction vocabulary
only (docs/design/02 §3, spec/schemas.md §4.2.1, decisions D460, D461).

A Template declares the transactions its step introduces, the storage it occupies and
which operations may overlap; the hardware model (spec/schemas.md §4.6.1) turns these
into time. No hardware number appears here.

``upcast_cost`` turns a schema-1 ``cost`` list into effects: each cost term becomes one
transaction of the legacy operation class ``unit:<unit>`` (one unit of work on that
resource per count), and a ``serial`` term gets an overlap entry that never holds.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import Field, model_validator

from kta.core.schemas.common import Body

# Operation-class prefix of transactions produced from a schema-1 cost term (D461).
LEGACY_KIND_PREFIX = "unit:"
# Selector prefixes in OverlapEffect.between (anything else names a transaction id).
SELECT_PATTERN = "pattern:"
SELECT_KIND = "kind:"


class TxEffect(Body):
    """Transactions of one operation class introduced by the step.

    ``count`` is per instance of the refined transaction (a chain item's ``p.instances``
    multiplies it); ``amounts`` are per transaction (bytes, flops, wavefronts, ...), read
    by the operation class's reservation table as ``tx.<name>``.
    """

    id: str
    pattern: str  # transaction pattern (open table: transfer, contract, sync, ...)
    # Operation class name in the hardware model (open; unknown -> unpriced). When it names
    # a Template variable or a p.* parameter, the class is that binding's value looked up
    # in ``kind_map`` (the value itself when absent), e.g. kind: instruction (D460).
    kind: str
    kind_map: dict[str, str] = Field(default_factory=lambda: dict[str, str]())
    count: str  # expression
    amounts: dict[str, str] = Field(default_factory=lambda: dict[str, str]())
    src: str | None = None  # storage level a transfer reads
    dst: str | None = None  # storage level a transfer writes
    when: str | None = None  # the transactions exist only when this holds
    note: str | None = None


class StorageEffect(Body):
    """Bytes the step keeps at one level, per executor instance of ``scope``.

    ``level`` is a level name, or the name of a Template variable / ``p.*`` parameter
    whose binding gives the level name.
    """

    id: str
    level: str
    scope: str = "cta"  # thread, warp, cta, sm, chip (open)
    bytes: str  # expression
    when: str | None = None
    note: str | None = None


class OverlapEffect(Body):
    """Which transactions may run concurrently.

    ``between`` holds one or two selectors: a transaction id of this Template,
    ``pattern:<name>`` or ``kind:<name>`` (chain-wide). One selector means "with every
    other transaction". The operations overlap while ``when`` holds (always when absent)
    and serialise otherwise. ``inflight`` is the number of generations in flight and
    ``trips`` the sequential iterations per executor; with both, L1 adds the recurrence
    bound ``trips * (lat_a + lat_b) / inflight``.
    """

    between: list[str] = Field(min_length=1, max_length=2)
    when: str | None = None
    inflight: str | None = None
    trips: str | None = None
    note: str | None = None


class Effects(Body):
    transactions: list[TxEffect] = Field(default_factory=lambda: list[TxEffect]())
    storage: list[StorageEffect] = Field(default_factory=lambda: list[StorageEffect]())
    overlap: list[OverlapEffect] = Field(default_factory=lambda: list[OverlapEffect]())
    # "cost" when these effects were produced from a schema-1 cost list (D461).
    upcast_from: str | None = None

    @model_validator(mode="after")
    def _ids(self) -> Effects:
        ids = [t.id for t in self.transactions] + [s.id for s in self.storage]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"effects ids declared twice: {dup}")
        local = {t.id for t in self.transactions}
        for o in self.overlap:
            for sel in o.between:
                if sel.startswith((SELECT_PATTERN, SELECT_KIND)):
                    continue
                if sel not in local:
                    raise ValueError(f"overlap selector {sel!r} names no transaction")
        return self


def legacy_id(index: int) -> str:
    return f"cost{index}"


def upcast_cost(cost: list[dict[str, Any]]) -> dict[str, Any]:
    """Effects body equivalent to a schema-1 ``cost`` list (D461)."""
    txs: list[dict[str, Any]] = []
    overlap: list[dict[str, Any]] = []
    for i, term in enumerate(cost):
        unit = str(term["unit"])
        txs.append(
            {
                "id": legacy_id(i),
                "pattern": "other",
                "kind": LEGACY_KIND_PREFIX + unit,
                "count": str(term["work"]),
            }
        )
        if term.get("overlap", "overlappable") == "serial":
            overlap.append({"between": [legacy_id(i)], "when": "False", "note": "v1 serial"})
    return {"transactions": txs, "overlap": overlap, "upcast_from": "cost"}


def upcast_template_body(body: dict[str, Any]) -> dict[str, Any]:
    """Template schema 1 -> 2: ``cost`` becomes ``effects`` (D461)."""
    if "cost" not in body:
        return body
    out = dict(body)
    cost = cast(list[dict[str, Any]], out.pop("cost") or [])
    if out.get("effects"):
        raise ValueError("a Template body has both schema-1 cost and schema-2 effects")
    out["effects"] = upcast_cost(cost)
    return out
