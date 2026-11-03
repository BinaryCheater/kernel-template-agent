"""One measurement batch on spark under the gb10-measure protocol (perfmodel-measure run).

    python3 perfmodel/remote/run.py <build_dir> <out_dir>

Standard library only (runs with the host's python3). Order, all in one exclusive job:
 1 environment record     2 clock calibration     3 compute and memory canaries
 4 every probe and test-set kernel, event/clock64 timing, no profiler attached
 5 test-set kernels under ncu, counts only, fixed metric list (one launch after warm-up)
 6 canaries and clock again                       7 environment record again
Each step's stdout goes to <out>/raw/<step>-<unit>.txt; <out>/run.json lists the steps.
The profiler runs as `sudo -n /usr/local/cuda/bin/ncu` (the one command allowed with sudo on
spark; the driver keeps RmProfilingAdminOnly=1), stdout captured here, no report files.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

NCU = "/usr/local/cuda/bin/ncu"
METRICS = ",".join(
    [
        "smsp__inst_executed.sum",
        *(
            f"smsp__inst_executed_pipe_{p}.sum"
            for p in (
                "alu",
                "fma",
                "fmaheavy",
                "fmalite",
                "lsu",
                "tensor",
                "xu",
                "uniform",
                "cbu",
                "adu",
            )
        ),
        *(f"sm__pipe_{p}_cycles_active.sum" for p in ("tensor", "fma", "alu", "fp64")),
        "sm__pipe_tensor_subpipe_hmma_cycles_active.sum",
        "sm__cycles_elapsed.max",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
        "lts__t_sectors_op_read.sum",
        "lts__t_sectors_op_write.sum",
        "lts__t_sector_hit_rate.pct",
        "launch__grid_size",
        "launch__block_size",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block_dynamic",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "gpu__time_duration.sum",
    ]
)


def sh(cmd: list[str], cwd: Path | None = None, timeout: int = 1800) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    return p.returncode, p.stdout + (("\n# stderr\n" + p.stderr) if p.stderr.strip() else "")


def main(build: Path, out: Path) -> int:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    steps: list[dict] = []

    def step(name: str, cmd: list[str], cwd: Path | None = None) -> str:
        t0 = time.time()
        rc, text = sh(cmd, cwd)
        (raw / f"{name}.txt").write_text(text)
        steps.append(
            {"step": name, "cmd": cmd, "rc": rc, "t0": t0, "s": round(time.time() - t0, 1)}
        )
        print(f"{name}: rc {rc} {time.time() - t0:.1f} s", flush=True)
        return text

    def env(tag: str) -> None:
        step(f"1-env-{tag}-smi", ["nvidia-smi", "-q", "-d", "CLOCK,TEMPERATURE,POWER,PERFORMANCE"])
        apps = step(
            f"1-env-{tag}-apps",
            ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        )
        if apps.strip():
            steps[-1]["busy"] = apps.strip()
        step(f"1-env-{tag}-uptime", ["uptime"])

    def protocol(tag: str) -> None:
        b = build / "protocol" / "bin"
        for mode in ("clock", "fma", "mem"):
            step(f"{tag}-protocol-{mode}", [str(b), mode])

    env("before")
    if any(s.get("busy") for s in steps):
        print("GPU busy: refusing the batch", flush=True)
        (out / "run.json").write_text(json.dumps({"steps": steps, "refused": "gpu busy"}, indent=1))
        return 2
    step("1-env-versions", ["sh", "-c", "nvidia-smi --version; nvcc --version | tail -2"])
    protocol("2")
    for d in sorted(build.glob("probe-*")):
        step(f"4-{d.name}", [str(d / "bin")], cwd=d)
    kernels = sorted(p for p in (build / "testset").iterdir() if (p / "bin").exists())
    for d in kernels:
        step(f"4-{d.name}", [str(d / "bin"), "30"], cwd=d)
    for d in kernels:
        entry = json.loads((d / "launch.json").read_text())["entry"]
        step(
            f"5-ncu-{d.name}",
            [
                "sudo",
                "-n",
                NCU,
                "--csv",
                "--page",
                "raw",
                "--metrics",
                METRICS,
                "-k",
                f"regex:^{entry}$",
                "--launch-skip",
                "10",
                "--launch-count",
                "1",
                str(d / "bin"),
                "1",
            ],
            cwd=d,
        )
    protocol("6")
    env("after")
    owned = sh(["find", str(out), "-not", "-user", str(os.getuid())])[1].strip()
    doc = {"steps": steps, "metrics": METRICS, "not_owned_by_user": owned}
    (out / "run.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()))
