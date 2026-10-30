from __future__ import annotations

from pathlib import Path

import pytest

from kta.core.config import config_hash, load_config
from kta.core.context import Context, data_dir_for, parse_env
from kta.core.errors import ConfigError


def test_parse_env() -> None:
    assert parse_env("dev") == "dev"
    with pytest.raises(ConfigError):
        parse_env(None)
    with pytest.raises(ConfigError):
        parse_env("prod")


def test_data_dirs() -> None:
    assert data_dir_for("dev", {"KTA_DATA": "/d", "KTA_WP": "WP2"}) == Path("/d/dev/WP2")
    assert data_dir_for("main", {"KTA_DATA": "/d"}) == Path("/d/main")
    assert data_dir_for("test", {"KTA_TEST_DATA": "/t"}) == Path("/t")
    with pytest.raises(ConfigError, match="KTA_WP"):
        data_dir_for("dev", {"KTA_DATA": "/d"})
    with pytest.raises(ConfigError, match="KTA_DATA"):
        data_dir_for("main", {})


def test_rng_streams_are_reproducible_and_independent(ctx: Context) -> None:
    a1 = [ctx.rng("a").random() for _ in range(3)]
    a2 = [ctx.rng("a").random() for _ in range(3)]
    b = [ctx.rng("b").random() for _ in range(3)]
    assert a1 == a2 and a1 != b


def test_config_hash_ignores_formatting(tmp_path: Path) -> None:
    one = tmp_path / "one.toml"
    two = tmp_path / "two.toml"
    one.write_text('[impl]\nstore = "sqlite"\n# comment\n[toolchain]\ncuda_home = "/x"\n')
    two.write_text('[toolchain]\ncuda_home="/x"\n\n[impl]\nstore="sqlite"\n')
    assert load_config(one).hash == load_config(two).hash == config_hash(load_config(one).data)
    assert load_config(one).impl("store") == "sqlite"


def test_bad_config_is_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text("[impl\n")
    with pytest.raises(ConfigError):
        load_config(bad)
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.toml")
