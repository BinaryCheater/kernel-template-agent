"""Frozen identities and receipt evidence cannot be bypassed by readiness flags."""

import json

import pytest

from kta.core.errors import ConfigError
from kta_source_replay.seal import digest, receipt, verify_chain, verify_seal


def test_boolean_only_readiness_is_not_launchable(tmp_path):
    (tmp_path / "campaign.json").write_text(json.dumps({"launch_allowed": True}))
    (tmp_path / "prelaunch.json").write_text(json.dumps({"passed": True}))
    with pytest.raises(ConfigError, match="no sealed"):
        verify_seal(tmp_path)


def test_receipt_binds_actual_artifact_bytes(tmp_path):
    artifact = tmp_path / "trace.json"
    artifact.write_text('{"real_trace":"original"}')
    path = tmp_path / "P1.json"
    path.write_text(
        json.dumps(
            {
                "stage": "P1",
                "artifacts": [{"path": str(artifact), "sha256": digest(artifact)}],
                "checks": [
                    {"name": "original identity", "status": "pass", "evidence": [str(artifact)]}
                ],
            }
        )
    )
    receipt(path, "P1")
    artifact.write_text('{"real_trace":"substituted"}')
    with pytest.raises(ConfigError, match="identity changed"):
        receipt(path, "P1")


def test_receipt_cannot_cite_unsealed_claim(tmp_path):
    path = tmp_path / "P6.json"
    path.write_text(
        json.dumps(
            {
                "stage": "P6",
                "artifacts": [{"path": str(path), "sha256": "bad"}],
                "checks": [{"name": "smoke", "status": "pass", "evidence": []}],
            }
        )
    )
    with pytest.raises(ConfigError):
        receipt(path, "P6")


def test_directory_symlink_and_symlink_ancestor_rejected(tmp_path):
    package = tmp_path / "workspace"
    package.mkdir()
    library = tmp_path / "other-library"
    library.mkdir()
    (library / "GUIDE.md").write_text("answers")
    (package / "library").symlink_to(library, target_is_directory=True)
    with pytest.raises(ConfigError, match="symlink"):
        digest(package)
    with pytest.raises(ConfigError, match="symlink"):
        digest(package / "library/GUIDE.md")


def test_chain_must_reach_computation_original_and_receipts(tmp_path):
    design = {
        "research_question": "question",
        "source": {"revision": "pinned", "translation_unit": "original.cu"},
        "arms": ["P", "S", "N"],
        "schedule": [{"slot": "one"}],
    }
    receipts = {f"P{i}": {"sha256": f"actual-{i}"} for i in range(1, 7)}
    chain = {
        "research_question": "question",
        "source_revision": "pinned",
        "translation_unit": "original.cu",
        "outputs": ["O", "LSE"],
        "comparator": "unchanged_original",
        "arms": ["P", "S", "N"],
        "schedule": [{"slot": "one"}],
        "receipts": {k: v["sha256"] for k, v in receipts.items()},
    }
    path = tmp_path / "chain.json"
    path.write_text(json.dumps(chain))
    verify_chain(path, design, receipts)
    chain["outputs"] = ["O"]
    path.write_text(json.dumps(chain))
    with pytest.raises(ConfigError, match="outputs"):
        verify_chain(path, design, receipts)


def test_resume_allows_candidate_work_but_rejects_public_contract_edits(tmp_path):
    from kta_source_replay.seal import verify_slot_inputs

    task = tmp_path / "TASK.md"
    task.write_text("frozen computation")
    entry = {
        "workspace": str(tmp_path),
        "sha256": digest(tmp_path),
        "immutable_files": {"TASK.md": digest(task)},
    }
    verify_slot_inputs(entry, True)
    (tmp_path / "candidate.py").write_text("model-authored new work")
    verify_slot_inputs(entry, False)
    with pytest.raises(ConfigError, match="fresh slot"):
        verify_slot_inputs(entry, True)
    task.write_text("changed computation")
    with pytest.raises(ConfigError, match="immutable public input"):
        verify_slot_inputs(entry, False)
