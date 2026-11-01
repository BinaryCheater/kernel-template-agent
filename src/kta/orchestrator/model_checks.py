"""Candidate-aligned model pairs and checker-owned, frozen holdout candidates."""

from __future__ import annotations

from collections.abc import Sequence

from kta.core.errors import NotSupported, SchemaError
from kta.core.ids import record_id
from kta.core.schemas import Record
from kta.core.schemas.evaluation import Measurement, Prediction
from kta.store.base import Store


def paired_records(store: Store, ids: Sequence[str]) -> list[tuple[Record, Record]]:
    records = [store.get(i) for i in dict.fromkeys(ids)]
    predictions = [r for r in records if r.kind == "Prediction"]
    measurements = [r for r in records if r.kind == "Measurement"]
    if not predictions or len(predictions) != len(measurements):
        raise SchemaError("T4 needs equal nonempty Prediction/Measurement sets")
    measured: dict[str, Record] = {}
    for r in measurements:
        candidate = r.refs.get("candidate")
        if not isinstance(candidate, str) or candidate in measured:
            raise SchemaError("T4 measurements need distinct candidate identities")
        measured[candidate] = r
    pairs: list[tuple[Record, Record]] = []
    for p in predictions:
        candidate = p.refs.get("candidate")
        if not isinstance(candidate, str) or candidate not in measured:
            raise SchemaError("T4 prediction has no measurement of the same candidate")
        pairs.append((p, measured.pop(candidate)))
    protocols = {str(m.body["protocol"]) for _, m in pairs}
    if len(protocols) != 1:
        raise SchemaError("T4 cannot mix measurement protocols")
    return sorted(pairs, key=lambda pair: str(pair[0].refs["candidate"]))


def partition(store: Store, ids: Sequence[str]) -> tuple[list[str], list[str]]:
    """Stable 20% partition by candidate hash: repeated tasks never reveal holdout labels."""
    if len(ids) < 5:
        return list(ids), []
    pairs = paired_records(store, ids[1:])
    if len(pairs) < 2:
        return list(ids), []
    held_indices = {
        i for i, (p, _) in enumerate(pairs) if int(str(p.refs["candidate"]), 16) % 5 == 0
    }
    held = [record_id(r) for i in held_indices for r in pairs[i]]
    visible = [i for i in ids if i not in set(held)]
    return visible, held


def holdout(store: Store, card_id: str) -> list[tuple[Prediction, Measurement]]:
    card = store.get(card_id)
    ids = card.refs.get("holdout")
    if not isinstance(ids, list) or not ids:
        raise NotSupported(
            "T4 has no independent frozen holdout; create a task with >=2 candidates"
        )
    if set(ids) & set(card.body["inputs"]):
        raise SchemaError("T4 holdout overlaps agent-visible inputs")
    if not any(store.get(i).kind == "Prediction" for i in card.body["inputs"]):
        raise NotSupported("T4 has no training candidates outside its frozen holdout")
    return [
        (Prediction.model_validate(p.body), Measurement.model_validate(m.body))
        for p, m in paired_records(store, ids)
    ]
