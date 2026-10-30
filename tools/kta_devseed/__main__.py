"""Seed a dev store with origin="fixture" records: ``scripts/seed-dev [--wp WP9a]``.

# kta:fixture the whole package writes synthetic data for frontend development (D168).
Refuses to run outside KTA_ENV=dev or when the target store already holds records; it never
deletes or overwrites anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime, timedelta

from kta.core.config import default_config_path, load_config
from kta.core.context import Context, data_dir_for
from kta.core.registry import build_registry
from kta.store import open_ops, open_store
from kta_devseed.knowledge import Seeder, seed_knowledge
from kta_devseed.supervision import seed_supervision
from kta_devseed.world import WINDOW_MIN, SeedClock, seed_world


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seed-dev")
    parser.add_argument(
        "--wp",
        default="WP9a-fixture",
        help="dev data directory $KTA_DATA/dev/<name> (default WP9a-fixture, "
        "kept apart from real WP9a records)",
    )
    args = parser.parse_args(argv)
    if os.environ.get("KTA_ENV", "dev") != "dev":
        sys.stderr.write("seed-dev writes fixture records and runs only with KTA_ENV=dev\n")
        return 2
    end = datetime.now(UTC).replace(microsecond=0)
    clock = SeedClock(end - timedelta(minutes=WINDOW_MIN + 60))
    data_dir = data_dir_for("dev", {**os.environ, "KTA_WP": args.wp})
    data_dir.mkdir(parents=True, exist_ok=True)
    ctx = Context(
        env="dev",
        config=load_config(default_config_path()),
        data_dir=data_dir,
        registry=build_registry(),
        clock=clock,
        process="seed-dev",
    )
    store = open_store(ctx)
    if store.count() > 0:
        sys.stderr.write(
            f"{data_dir}/kta.db already holds {store.count()} records; "
            "refusing to add a second fixture world\n"
        )
        return 1
    seeder = Seeder(store, clock)
    seed_knowledge(seeder)
    ops = open_ops(ctx)
    seed_world(seeder, ops, end)
    seed_supervision(seeder, ops, data_dir)
    with store.db.read() as con:
        real = con.execute("SELECT COUNT(*) FROM records WHERE origin != 'fixture'").fetchone()[0]
    if real:
        sys.stderr.write(f"bug: {real} seeded records do not have origin fixture\n")
        return 1
    sys.stdout.write(
        f"seeded {store.count()} fixture records into {data_dir} (window ends {end.isoformat()})\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
