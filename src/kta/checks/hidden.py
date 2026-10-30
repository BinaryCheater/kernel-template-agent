"""Hidden shapes generated from a private seed (docs/design/50 §3).

The seed lives in the checker's private directory (``<data_dir>/checker/hidden_seed``,
mode 600), outside every agent workspace. It is drawn once from the OS (``secrets``),
not from ``ctx.rng``: a seed derived from the recorded context seed would not be
private (decision D225). Check records name only a hidden shape's category.
"""

from __future__ import annotations

import hashlib
import random
import secrets
from pathlib import Path
from typing import Any

from kta.core.ids import canonical_json

SEED_FILE = "hidden_seed"
SMALL_PRIMES = (3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
CATEGORIES = ("non_tile_multiple", "unit_dimension", "small_primes")


def private_seed(private_dir: Path) -> str:
    path = private_dir / SEED_FILE
    if not path.exists():
        private_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.write_text(secrets.token_hex(16), encoding="utf-8")
        path.chmod(0o600)
    return path.read_text(encoding="utf-8").strip()


def hidden_shapes(seed: str, base: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
    """One shape per category, derived from the integer dimensions of ``base``."""
    digest = hashlib.sha256(f"{seed}:{canonical_json(base).decode()}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    dims = sorted(k for k, v in base.items() if isinstance(v, int) and not isinstance(v, bool))
    out: list[tuple[dict[str, Any], str]] = []
    if not dims:
        return out
    off = {k: max(1, int(base[k]) // 4 + rng.randrange(1, 16, 2)) for k in dims}
    out.append(({**base, **off}, "non_tile_multiple"))
    unit = rng.choice(dims)
    out.append(({**base, **{k: off[k] for k in dims}, unit: 1}, "unit_dimension"))
    out.append(({**base, **{k: rng.choice(SMALL_PRIMES) for k in dims}}, "small_primes"))
    return out
