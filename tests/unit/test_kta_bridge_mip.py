"""The decision-level MIP against direct evaluation of the same form (LEARNDEMO1)."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.kta_bridge import families, forms, mip  # noqa: E402

CONSTANTS = json.loads((ROOT / "tools/kta_bridge/constants-v4.json").read_text())


@pytest.mark.parametrize(
    ("family", "key"), [(families.fa, "fa_forward"), (families.hgemm, "hgemm")]
)
def test_mip_matches_enumeration(family, key):
    f, c = family("warm"), CONSTANTS[key]
    direct = mip.enumerate_all(f, c)
    solved = mip.solve(f, c, k=3)
    for rank, pick in enumerate(solved["picks"]):
        assert pick["objective"] == pytest.approx(pick["direct"], abs=1e-4)
        assert pick["objective"] == pytest.approx(direct[rank][0], abs=1e-4)


def test_spill_grows_with_the_excess_and_the_limit_falls_with_threads():
    assert forms.spill_share(200, 128) == 0
    assert forms.spill_share(510, 128) == pytest.approx(0.5)
    assert forms.spill_share(200, 512) == pytest.approx(0.36)
