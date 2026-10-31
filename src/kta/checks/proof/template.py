"""Once per Template (paper §2.2, §4.2): K1, K2 through the outline's models, K3.

The level a check can support:

- every K1 and K3 result proved, and every obligation of the Template's Σ checked with
  outcome proved or bounded -> ``proved`` when all are proved, else ``bounded``;
- anything else -> no level; each reason is a gap line, and obligations the checker did
  not check are listed as unchecked (they stay at the code level, ``tested``).

An obligation's outcome is capped by its planned method: ``exhaustive``, ``test`` and
``unspecified`` reach at most ``bounded``; only ``smt`` and ``proof`` reach ``proved``.
The K2 local conditions the outline maps (initialisation, step, progress, completion) are
reported; the level does not require all four (decision D636).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from kta.core.schemas.knowledge import Template
from kta.core.schemas.proof import ProofOutline

from .contract import assumption_issues, k1, k3, vacuity
from .encode import Sym, Unsupported
from .models import GROUPS, MODELS, ModelInput
from .smt import DEFAULT_TIMEOUT_MS
from .space import Space
from .types import Draft, cap

METHOD_CEILING = {
    "smt": "proved",
    "proof": "proved",
    "exhaustive": "bounded",
    "test": "bounded",
    "unspecified": "bounded",
}
LOCAL = ("initialisation", "step", "progress", "completion")


@dataclass
class TemplateCheck:
    drafts: list[Draft]
    unchecked: list[dict[str, Any]]
    level: str | None
    gaps: list[str]
    k2_conditions: dict[str, list[str]] = field(default_factory=lambda: dict[str, list[str]]())


def _k2(
    sp: Space, outline: ProofOutline | None, timeout_ms: int, only: set[str] | None
) -> tuple[list[Draft], list[dict[str, Any]], dict[str, list[str]]]:
    drafts: list[Draft] = []
    unchecked: list[dict[str, Any]] = []
    cover: dict[str, list[str]] = {c: [] for c in LOCAL}
    bindings = {b.obligation: b for b in (outline.models if outline else [])}
    for ob in sp.template.obligations:
        if only is not None and ob.name not in only:
            continue
        b = bindings.get(ob.name)
        if b is None:
            why = "no outline" if outline is None else "the outline names no checker model"
            unchecked.append(
                {
                    "obligation": ob.name,
                    "method": ob.method,
                    "reason": f"{why}; stays at the code level (tested)",
                }
            )
            continue
        fn = MODELS.get(b.model)
        try:
            if fn is None:
                raise Unsupported(f"unknown checker model {b.model!r}")
            d = fn(ModelInput(sp, b, ob, timeout_ms))
        except Unsupported as exc:
            d = Draft(
                "K2", ob.name, "unresolved", f"not checked: {exc}", reason=f"unsupported: {exc}"
            )
        d.groups = GROUPS.get(b.model, ("gamma", "assumes", "X", "pre", "program"))
        ceiling = METHOD_CEILING[ob.method]
        if ceiling != "proved" and d.outcome == "proved":
            d.data["capped_by_method"] = ob.method
        d.outcome = cap(d.outcome, ceiling)
        d.data.update(model=b.model, method=ob.method, local_condition=b.condition, params=b.params)
        if b.condition in cover and d.outcome in ("proved", "bounded"):
            cover[b.condition].append(ob.name)
        drafts.append(d)
    return drafts, unchecked, cover


def fold(
    drafts: list[Draft], unchecked: list[dict[str, Any]], gaps: list[str]
) -> tuple[str | None, list[str]]:
    """The level the results support, and why it is not higher."""
    gaps = list(gaps)
    for d in drafts:
        if d.outcome not in ("proved", "bounded"):
            gaps.append(
                f"{d.condition} {d.obligation}: {d.outcome}"
                + (f" ({d.reason})" if d.reason else "")
            )
    for u in unchecked:
        gaps.append(f"K2 {u['obligation']}: unchecked (method {u['method']}); {u['reason']}")
    if gaps or not drafts:
        return None, gaps
    return ("proved" if all(d.outcome == "proved" for d in drafts) else "bounded"), gaps


def check_template(
    t: Template,
    outline: ProofOutline | None,
    *,
    restrict: Mapping[str, Any] | None = None,
    attrs: Mapping[str, Sym] | None = None,
    menus: Mapping[str, list[Any]] | None = None,
    extra: Any = None,
    ns: str = "",
    timeout_ms: int = DEFAULT_TIMEOUT_MS,
    conditions: tuple[str, ...] = ("K1", "K2", "K3"),
    skip: set[tuple[str, str]] | None = None,
) -> TemplateCheck:
    """K1, K2, K3 of ``t``. ``restrict`` fixes design variables or aliases them to a
    family variable's symbol, ``attrs`` binds p.* and ``menus`` narrows design variables
    (a chain's bindings); ``skip`` names (condition, obligation) pairs whose certificate is
    reused, which are not recomputed."""
    skip = skip or set()
    gaps: list[str] = []
    assumes = list(outline.assumes) if outline else []
    bad = assumption_issues(t, assumes)
    if bad:
        gaps += bad
        assumes = []
    try:
        sp = Space(
            t, assumes, dict(restrict or {}), dict(attrs or {}), dict(menus or {}), extra, ns
        )
    except Unsupported as exc:
        d = Draft("K1", "*", "unresolved", "not checked", reason=f"cannot build the space: {exc}")
        return TemplateCheck([d], [], None, [*gaps, d.reason])
    vac = vacuity(sp, timeout_ms)
    if vac is not None:
        return TemplateCheck([vac], [], None, [*gaps, vac.reason])
    drafts: list[Draft] = []
    if "K1" in conditions:
        drafts += [d for d in k1(sp, timeout_ms, skip)]
    unchecked: list[dict[str, Any]] = []
    cover: dict[str, list[str]] = {}
    if "K2" in conditions:
        only = {o.name for o in t.obligations if ("K2", o.name) not in skip}
        k2, unchecked, cover = _k2(sp, outline, timeout_ms, only)
        drafts += k2
    if "K3" in conditions and ("K3", "*") not in skip:
        drafts.append(k3(sp, timeout_ms))
    level, gaps = fold(drafts, unchecked, gaps)
    return TemplateCheck(drafts, unchecked, level, gaps, cover)
