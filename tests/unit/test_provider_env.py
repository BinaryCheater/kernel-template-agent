"""Provider reuse must preserve experimental isolation and exclude host credentials."""

from __future__ import annotations

import io
import json
import tarfile
import tomllib
from pathlib import Path

import pytest

from kta.agents import hostfs, provider_env
from kta.agents.launch import pack_outputs, provider_status


def test_connection_is_pinned_without_importing_personal_context(tmp_path: Path) -> None:
    native, home = tmp_path / "native", tmp_path / "experiment"
    native.mkdir()
    home.mkdir()
    (native / "config.toml").write_text(
        'model_provider = "custom"\nmodel = "personal-model"\n'
        'developer_instructions = "personal instructions"\n'
        '[model_providers.custom]\nname = "custom"\nbase_url = "https://example.test/v1"\n'
        'experimental_bearer_token = "private-provider-value"\n'
    )
    (home / "config.toml").write_text('model = "experiment-model"\nweb_search = "disabled"\n')
    meta = provider_env.inherit_connection(home, native)
    cfg = tomllib.loads((home / "config.toml").read_text())
    assert cfg["model_provider"] == "custom" and cfg["model"] == "experiment-model"
    assert cfg["model_providers"]["custom"]["base_url"] == "https://example.test/v1"
    assert "developer_instructions" not in cfg
    assert "private-provider-value" not in json.dumps(meta)
    before = (home / "config.toml").read_bytes()
    (native / "config.toml").write_text('model_provider = "changed"\n')
    assert provider_env.inherit_connection(home, native) == meta
    assert (home / "config.toml").read_bytes() == before

    ws, out = tmp_path / "ws", tmp_path / "out"
    ws.mkdir()
    out.mkdir()
    (native / "auth.json").write_text("native-login")
    hostfs.link_credentials(home, native, ["auth.json"])
    pack_outputs(ws, home, out, ["auth.json"], False)
    archive = (out / "home.tar.gz").read_bytes()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        assert "config.toml" not in tar.getnames() and "auth.json" not in tar.getnames()
        for member in tar.getmembers():
            if member.isfile():
                handle = tar.extractfile(member)
                assert handle is not None
                assert b"private-provider-value" not in handle.read()


def test_native_source_and_cli_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    native = tmp_path / "configured-home"
    monkeypatch.setenv("CODEX_HOME", str(native))
    assert provider_env.auth_directory("") == native
    assert provider_env.auth_directory(str(tmp_path)) == tmp_path
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", "")
    binary = tmp_path / ".local/bin/codex"
    binary.parent.mkdir(parents=True)
    binary.touch(mode=0o755)
    assert provider_env.command("codex") == [str(binary)]
    assert provider_env.command("codex", ["/explicit/codex"]) == ["/explicit/codex"]


def test_missing_cli_is_unknown_instead_of_logged_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", "")
    monkeypatch.setenv("HOME", str(tmp_path))
    view = provider_status({"codex_cmd": ["/missing/kta-test-codex"]})
    assert view["providers"]["codex"]["auth"]["logged_in"] is None
    assert view["providers"]["codex"]["cli"]["version"] is None
