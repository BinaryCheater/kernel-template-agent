"""SYSDEBUG-2: settings in force and annotations for supervised trials (D690, D691)."""

from __future__ import annotations

import pytest

from apihelp import request, seeded_api
from kta.api import review
from kta.api.app import create_app
from kta.api.common import Api
from kta.core.controls import definitions


@pytest.fixture(scope="module")
def api(tmp_path_factory: pytest.TempPathFactory) -> Api:
    return seeded_api(tmp_path_factory.mktemp("sd2"))


def test_every_mapped_knob_exists_and_unmapped_knobs_show_as_other() -> None:
    ids = {k.id for k in definitions().knobs}
    mapped = {k for ks in review.stage_table().values() for k in ks}
    assert mapped <= ids, mapped - ids
    stages = review._stages_of()  # pyright: ignore[reportPrivateUsage]
    assert stages["measure.quiet_thresholds"] == ["measuring"]


def test_in_force_says_who_set_each_knob_and_which_step_reads_it(api: Api) -> None:
    status, view = request(create_app(api), "GET", "/settings/in-force")
    assert status == 200
    knobs = {k["id"]: k for k in view["knobs"]}
    assert len(knobs) == len(definitions().knobs)
    q = knobs["measure.quiet_thresholds"]
    assert q["stages"] == ["measuring"] and q["who"].startswith("assistant default")
    assert not q["set_by_you"] and not q["calibrated"]
    assert knobs["measure.requeue_wait_s"]["value"] == 30.0
    assert sum(view["counts"].values()) == len(knobs)
    assert "other" in view["stages"] and "hardware" in view


def test_annotations_are_listed_and_resolved(api: Api) -> None:
    app = create_app(api)
    status, out = request(
        app,
        "POST",
        "/actions/annotate",
        {"subject": "knob:measure.quiet_thresholds", "verdict": "question", "note": "why 2 cores?"},
    )
    assert status == 200
    rid = out["annotation"]
    _, items = request(app, "GET", "/annotations?open=true")
    mine = [a for a in items if a["id"] == rid]
    assert mine and mine[0]["verdict"] == "question" and mine[0]["note"] == "why 2 cores?"
    _, view = request(app, "GET", "/settings/in-force")
    knob = next(k for k in view["knobs"] if k["id"] == "measure.quiet_thresholds")
    assert [a["id"] for a in knob["annotations"]] == [rid]
    status, _ = request(app, "POST", "/actions/resolve", {"annotation": rid, "outcome": "kept"})
    assert status == 200
    _, still_open = request(app, "GET", "/annotations?open=true")
    assert rid not in {a["id"] for a in still_open}
    status, _ = request(app, "POST", "/actions/annotate", {"subject": "knob:x", "verdict": "maybe"})
    assert status == 400  # the API maps validation errors to 400
