"""``scripts/perfmodel-measure``: one GPU batch on spark under the gb10-measure protocol
(docs/design/perfmodel/LOOP.md step 3), a thin wrapper over ``scripts/remote-run``.

    scripts/perfmodel-measure build <batch>    # deploy HEAD, compile probes and test set, get SASS
    scripts/perfmodel-measure predict <batch>  # the engine's replay of every case, before the run
    scripts/perfmodel-measure run <batch>      # the batch itself, then `import`
    scripts/perfmodel-measure import <batch>   # parse the raw outputs, validity, measured.json

A batch lives in ``perfmodel/batches/<batch>/``: ``entry.md`` (the entry block of every probe:
mechanism, why now, hypotheses with numbers), ``build/`` (SASS with control bits, resource usage,
launch configs), ``predicted.json``, after the run ``batch.json`` (commit, environment, clock,
canaries, cache state per case, validity) and ``measured.json`` (medians, IQR, counters). Raw
outputs stay in ``data/perfmodel/batches/<batch>/`` (not in git) and on spark. `run` refuses a
batch without its entry block and prediction, or from a dirty tree; an invalid batch is kept and
its measured.json is marked invalid, so the table does not import it.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shlex
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

from . import upstream

ROOT = upstream.ROOT
BATCHES = ROOT / "perfmodel" / "batches"
RAW = ROOT / "data" / "perfmodel" / "batches"
HOST, REMOTE_ROOT = "spark", "kta-work"
REMOTE_DATA = f"{REMOTE_ROOT}/data/perfmodel/batches"
CLOCK_DRIFT, CANARY_DRIFT = 0.01, 0.03  # gb10-measure: clock < 1 %; canaries inside their band
RESULT = re.compile(r"^RESULT (\S+) (\S+) median=(\S+) iqr=(\S+) unit=(\S+) n=(\d+)(.*)$")


def _run(cmd: list[str], **kw: Any) -> str:
    print("+", shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw).stdout


def _remote(cmd: str) -> str:
    return _run([str(ROOT / "scripts" / "remote-run"), "exec", HOST, "--", "sh", "-c", cmd])


def _sync() -> str:
    head, dirty = upstream.kta_commit()
    if dirty:
        raise SystemExit("refused: the tree has uncommitted changes (gb10-measure: code identity)")
    _run([str(ROOT / "scripts" / "remote-run"), "sync", HOST])
    return head


def build(batch: str, only: str = ".") -> int:
    head = _sync()
    rb = f"{REMOTE_DATA}/{batch}/build"
    _remote(
        f"mkdir -p ~/{REMOTE_DATA}/{batch} && cd src/{head} && "
        f"PATH=/usr/local/cuda/bin:$PATH ONLY={shlex.quote(only)} "
        f"bash perfmodel/remote/build.sh ~/{rb} "
        f"> ~/{REMOTE_DATA}/{batch}/build.log 2>&1"
    )
    local = BATCHES / batch / "build"
    local.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "rsync",
            "-a",
            "--delete",
            "--exclude=bin",
            "--exclude=*.cubin",
            "--exclude=common.cuh",
            f"{HOST}:{rb}/",
            f"{local}/",
        ]
    )
    doc = {
        "batch": batch,
        "commit": head,
        "host": HOST,
        "remote_build": rb,
        "only": only,
        "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "nvcc_host": (local / "nvcc-host.txt").read_text().strip(),
        "nvcc_image": (local / "nvcc-image.txt").read_text().strip(),
        "tilelang": (local / "tilelang-version.txt").read_text().strip(),
    }
    (BATCHES / batch / "build.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"built {batch} at {head[:12]}; SASS in {local}")
    return 0


_pred_cases: dict[str, Any] = {}


def _predict_one(args: tuple[str, str]) -> tuple[str, float, str]:
    from kta.models.gb10sim import probeset

    d, cid = args
    if not _pred_cases:
        _pred_cases.update((c.id, c) for c in probeset.all_cases(BATCHES / d))
    c = _pred_cases[cid]
    v, _, _ = probeset.run_case(c)
    return cid, v, c.unit


def predict(batch: str) -> int:
    """The engine's replay of every case before the run, in parallel (LOOP §8)."""
    from concurrent.futures import ProcessPoolExecutor

    from kta.models.gb10sim import probeset

    from . import table

    d = BATCHES / batch
    if (d / "measured.json").exists():
        raise SystemExit("refused: the batch is measured; a prediction must come before the run")
    ids = [c.id for c in probeset.all_cases(d)]
    out = {}
    with ProcessPoolExecutor(table.default_jobs()) as ex:
        for cid, v, unit in ex.map(_predict_one, [(batch, i) for i in ids]):
            out[cid] = round(v, 4)
            print(f"{cid:52s} {v:12.4f} {unit}", flush=True)
    out = {i: out[i] for i in ids}
    doc = {
        "engine": table.engine_hash(),
        "kta_commit": upstream.kta_commit()[0],
        "date": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "clock_GHz": "description kernel clock (no batch clock before the run)",
        "predicted": out,
    }
    (d / "predicted.json").write_text(json.dumps(doc, indent=1) + "\n")
    return 0


def run(batch: str) -> int:
    d = BATCHES / batch
    for need in ("entry.md", "predicted.json", "build.json"):
        if not (d / need).exists():
            raise SystemExit(f"refused: {d / need} missing (entry block and prediction first)")
    head = _sync()
    built = json.loads((d / "build.json").read_text())
    rb = f"~/{built['remote_build']}"
    ro = f"~/{REMOTE_DATA}/{batch}/run"
    _remote(f"test ! -e {ro} || {{ echo 'refused: {ro} exists'; exit 1; }}")
    _remote(f"cd src/{head} && python3 perfmodel/remote/run.py {rb} {ro} > {ro}.log 2>&1")
    local = RAW / batch
    local.mkdir(parents=True, exist_ok=True)
    _run(["rsync", "-a", f"{HOST}:{REMOTE_DATA}/{batch}/run/", f"{local}/"])
    (local / "commit").write_text(head + "\n")
    return import_(batch)


def _results(text: str) -> dict[str, dict[str, Any]]:
    out = {}
    for line in text.splitlines():
        m = RESULT.match(line.strip())
        if not m:
            continue
        extra = dict(kv.split("=", 1) for kv in m.group(7).split() if "=" in kv)
        extra.pop("samples", None)
        samples = re.search(r"samples=([\d.,]+)", line)
        rec: dict[str, Any] = {
            "median": float(m.group(3)),
            "iqr": float(m.group(4)),
            "unit": m.group(5),
            "n": int(m.group(6)),
            **{k: _num(v) for k, v in extra.items()},
        }
        if samples:
            rec["samples"] = [float(x) for x in samples.group(1).split(",") if x]
        out[f"{m.group(1)} {m.group(2)}"] = rec
    return out


def _num(v: str) -> Any:
    try:
        return float(v)
    except ValueError:
        return v


def _ncu(text: str, metrics: str) -> dict[str, Any]:
    rows = [r for r in csv.reader(io.StringIO(text)) if len(r) > 5]
    if len(rows) < 3:
        return {}
    head, vals = rows[0], rows[-1]
    want = set(metrics.split(","))
    return {h: _num(v.replace(",", "")) for h, v in zip(head, vals, strict=False) if h in want}


def import_(batch: str) -> int:
    d, raw = BATCHES / batch, RAW / batch
    run_doc = json.loads((raw / "run.json").read_text())
    texts = {p.stem: p.read_text() for p in (raw / "raw").glob("*.txt")}
    before = {
        **_results(texts["2-protocol-clock"]),
        **_results(texts["2-protocol-fma"]),
        **_results(texts["2-protocol-mem"]),
    }
    after = {
        **_results(texts["6-protocol-clock"]),
        **_results(texts["6-protocol-fma"]),
        **_results(texts["6-protocol-mem"]),
    }
    checks = {}
    for k, lim in (
        ("clock chain_GHz", CLOCK_DRIFT),
        ("canary fma_us", CANARY_DRIFT),
        ("canary mem_us", CANARY_DRIFT),
    ):
        b, a = before[k]["median"], after[k]["median"]
        checks[k] = {"before": b, "after": a, "drift": a / b - 1, "ok": abs(a / b - 1) < lim}
    # validity is the machine's (clock and canaries, gb10-measure); a case whose run failed has no
    # result and is listed, the rest of the batch stands
    failed = [s["step"] for s in run_doc["steps"] if s["rc"] != 0]
    valid = all(c["ok"] for c in checks.values())
    results: dict[str, Any] = {}
    counters: dict[str, Any] = {}
    for name, text in sorted(texts.items()):
        if name.startswith("4-"):
            results.update(_results(text))
        elif name.startswith("5-ncu-"):
            counters.setdefault(name[len("5-ncu-") :], {}).update(_ncu(text, run_doc["metrics"]))
        elif name.startswith("5x-ncu-"):  # round 2: the extra metrics' own pass
            x = _ncu(text, run_doc.get("metrics_x", ""))
            counters.setdefault(name[len("5x-ncu-") :], {}).update(x)
        elif name[:7] in ("5p-ncu-", "5q-ncu-"):  # round 3: probe profiler points
            key = "metrics_p" if name.startswith("5p") else "metrics_q"
            x = _ncu(text, run_doc.get(key, ""))
            counters.setdefault(name[7:], {}).update(x)
    clock = statistics.mean(
        [before["clock chain_GHz"]["median"], after["clock chain_GHz"]["median"]]
    )
    results["clock chain_GHz"] = {**before["clock chain_GHz"], "median": clock}
    cache = {k: "warm" for k in results}
    for k in results:
        if k.startswith("dram_rotate"):
            cache[k] = "rotating"
        # a test-set row is "<unit> kernel" (one launch), "<unit> b2b" or "<unit> graph" (round 4)
        unit = d / "build" / "testset" / k.rsplit(" ", 1)[0] / "launch.json"
        if unit.exists():  # the test set declares its cache state per kernel (gen.py)
            cache[k] = json.loads(unit.read_text()).get("cache", "warm")
    built = json.loads((d / "build.json").read_text())
    batch_doc = {
        "batch": batch,
        "commit": (raw / "commit").read_text().strip(),
        "build_commit": built["commit"],
        "entry": "entry.md",
        "prediction": "predicted.json",
        "clock_GHz": clock,
        "checks": checks,
        "failed_steps": failed,
        "valid": valid,
        "cache_state": cache,
        "environment": {
            k: texts[k] for k in texts if k.startswith("1-env") and not k.endswith("-smi")
        },
        "smi_before": texts.get("1-env-before-smi", "")[:6000],
        "smi_after": texts.get("1-env-after-smi", "")[:6000],
        "steps": run_doc["steps"],
        "not_owned_by_user": run_doc.get("not_owned_by_user", ""),
        "raw": str(raw.relative_to(ROOT)),
    }
    (d / "batch.json").write_text(json.dumps(batch_doc, indent=1) + "\n")
    # cases whose run is known to be wrong (a harness fault found after the run), with the reason:
    # kept out of the results, listed in the record (`invalid_cases.json` in the batch directory)
    inv_p = d / "invalid_cases.json"
    invalid = json.loads(inv_p.read_text()) if inv_p.exists() else {}
    dropped = {k: {**results.pop(k), "reason": invalid[k]} for k in list(results) if k in invalid}
    batch_doc["invalid_cases"] = invalid
    (d / "batch.json").write_text(json.dumps(batch_doc, indent=1) + "\n")
    meas = {"batch": batch, "valid": valid, "clock_GHz": clock, "results": results}
    meas["invalid_results"] = dropped
    meas["counters"] = counters
    (d / "measured.json").write_text(json.dumps(meas, indent=1) + "\n")
    print(f"{batch}: {'valid' if valid else 'INVALID'}; {len(results)} results; checks {checks}")
    if failed:
        print("failed steps:", failed)
    return 0 if valid else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="scripts/perfmodel-measure")
    ap.add_argument("command", choices=("build", "predict", "run", "import"))
    ap.add_argument("batch")
    ap.add_argument("--only", default=".", help="build: regex over probe and kernel names")
    a = ap.parse_args(argv)
    if a.command == "build":
        return build(a.batch, a.only)
    return {"predict": predict, "run": run, "import": import_}[a.command](a.batch)


if __name__ == "__main__":
    sys.exit(main())
