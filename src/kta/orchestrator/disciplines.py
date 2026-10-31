"""Loop disciplines of the case loop (docs/design/02 §6.1, 12 §3; WP18 D525-D527).

Pure functions over the loop data that ``judge`` keeps between rounds:

- ``basis``: what a round was built from (chain templates, family, model version);
- ``mechanisms``: what changed since the previous round, coupled templates counted as
  one group (12 §3 "one mechanism per round, coupled groups excepted");
- ``credit``: a gain counts only beyond the improvement band and attributed to a named
  mechanism within the per-round limit; a round without credit is a null round; a round
  that regresses beyond the band on its fresh measurement (the hold-out of the change that
  produced it) is rolled back: the incumbent stays and the mechanism is recorded as rejected.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def basis(templates: Iterable[str], family: str, model: str) -> dict[str, Any]:
    return {"templates": sorted(set(templates)), "family": family, "model": model}


def _stem(template_id: str) -> str:
    return template_id.split("@", 1)[0]


def mechanisms(
    prev: Mapping[str, Any] | None,
    now: Mapping[str, Any],
    couplings: Sequence[Sequence[str]] = (),
) -> tuple[list[str], int]:
    """(named changes, mechanism count). Round one names ``initial`` and counts 0."""
    if not prev:
        return ["initial"], 0
    before, after = set(prev["templates"]), set(now["templates"])
    changed = sorted(after - before) + sorted(before - after)
    names = [f"template:+{t}" for t in sorted(after - before)]
    names += [f"template:-{t}" for t in sorted(before - after)]
    model = now["model"] != prev["model"]
    generator = now["family"] != prev["family"] and not changed
    if model:
        names.append(f"model:{now['model']}")
    if generator:
        names.append(f"family:{now['family']}")  # same chain, new generator (a T6 fix)
    count = int(model) + int(generator)
    left = {_stem(t) for t in changed}
    for g in ({str(m) for m in group} for group in couplings):
        if len(left & g) > 1:  # a coupled group changed together counts once
            left -= g
            count += 1
    return names, count + len(left)


def credit(
    data: Mapping[str, Any],
    t_us: float,
    band: float,
    names: Sequence[str],
    count: int,
    family: str,
    s: Any,
) -> dict[str, Any]:
    """The round's discipline verdict; ``data`` is the loop data (incumbent, null_rounds).
    ``s`` is the orchestrator Settings (mechanism_attribution, one_mechanism)."""
    inc: Mapping[str, Any] | None = data.get("incumbent")
    out: dict[str, Any] = {"mechanisms": list(names), "mechanism_count": count}
    if inc is None:
        out.update(gain_us=None, credited=False, null_round=False, rolled_back=False)
        out["null_rounds"] = 0
        out["incumbent"] = {"t_us": t_us, "round": int(data.get("round", 1)), "family": family}
        return out
    gain = float(inc["t_us"]) - t_us
    named = bool(names) or not s.mechanism_attribution
    within = count <= s.one_mechanism
    credited = gain > band and named and within
    rolled = -gain > band
    out["gain_us"] = gain
    out["credited"] = credited
    out["rolled_back"] = rolled
    out["null_round"] = not credited
    out["null_rounds"] = 0 if credited else int(data.get("null_rounds", 0)) + 1
    if credited:
        out["incumbent"] = {"t_us": t_us, "round": int(data.get("round", 1)), "family": family}
    else:
        out["incumbent"] = dict(inc)
    if not within:
        out["note"] = f"{count} mechanisms changed; limit {s.one_mechanism}: gain not credited"
    elif gain > band and not named:
        out["note"] = "gain names no mechanism: not credited"
    return out
