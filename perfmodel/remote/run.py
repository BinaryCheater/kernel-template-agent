"""One measurement batch on spark under the gb10-measure protocol (perfmodel-measure run).

    python3 perfmodel/remote/run.py <build_dir> <out_dir> [no-counters regex]

Standard library only (runs with the host's python3). Order, all in one exclusive job:
 1 environment record     2 clock calibration     3 compute and memory canaries
 p every kernel launched once (reps 0), one that faults, leaves its buffer or hangs is dropped
 0 the GPU below 45 C
 4 every probe and test-set kernel, event/clock64 timing, no profiler attached
 6 canaries and clock again
 4r1, 4r2 each kernel timed again after a cool-down, with its own clock (3r*, 6r*)
 5 test-set kernels under ncu, counts only, fixed metric list (one launch after warm-up)
 7 environment record again
Each step's stdout goes to <out>/raw/<step>-<unit>.txt; <out>/run.json lists the steps.
The profiler runs as `sudo -n /usr/local/cuda/bin/ncu` (the one command allowed with sudo on
spark; the driver keeps RmProfilingAdminOnly=1), stdout captured here, no report files. From
round 2 with `--cache-control none`: the profiled launch sees the cache state the harness made
(warm or rotating), not an L2 flushed by the profiler (round 1's counters were taken after a flush).
"""

import json
import os
import re
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

# round 2: metrics not used before on this GPU, in their own profiler pass (5x-ncu-<unit>), so that
# an unknown name cannot cost the main counts
METRICS_X = ",".join(
    [
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "lts__t_sectors_lookup_hit.sum",
        "lts__t_sectors_lookup_miss.sum",
    ]
)


# round 3: probes that declare profiler points (perfmodel/probes/<name>/ncu.json: a list of
# {"tag", "args", "kernel", "skip"}) are profiled after the test set, L2 counters in one pass and
# candidate metrics not used before on this GPU in a second (5p-ncu-<tag>, 5q-ncu-<tag>)
METRICS_P = ",".join(
    [
        "lts__t_sectors_op_read.sum",
        "lts__t_sectors_op_write.sum",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
        "lts__t_sectors_lookup_hit.sum",
        "lts__t_sectors_lookup_miss.sum",
        "lts__t_sector_hit_rate.pct",
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
        "gpu__time_duration.sum",
        "lts__lts2xbar_cycles_active.sum",
    ]
)
METRICS_Q = ",".join(
    [
        "lts__t_sectors_srcunit_tex_op_read.sum",
        "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
        "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
        "lts__t_sector_op_read_hit_rate.pct",
        "dram__sectors_read.sum",
        "lts__d_sectors_fill_sysmem.sum",
        "lts__d_sectors_fill_device.sum",
    ]
)


# round 10: per-unit stall and activity counters, in their own pass (5u-ncu-<unit>); names from the
# profiler's catalog, none used before on this GPU
METRICS_U = ",".join(
    [
        *(
            f"smsp__warps_issue_stalled_{s}.sum"
            for s in (
                "long_scoreboard",
                "short_scoreboard",
                "mio_throttle",
                "lg_throttle",
                "math_pipe_throttle",
                "wait",
                "barrier",
                "not_selected",
                "selected",
                "no_instruction",
                "drain",
                "dispatch_stall",
                "branch_resolving",
                "membar",
                "misc",
            )
        ),
        "smsp__warps_active.sum",
        "smsp__warps_eligible.sum",
        "smsp__issue_active.sum",
        "sm__pipe_fmaheavy_cycles_active.sum",
        "sm__pipe_fmalite_cycles_active.sum",
        "sm__cycles_active.sum",
    ]
)
METRICS_V = ",".join(
    [
        "l1tex__m_l1tex2xbar_req_cycles_active.sum",
        "l1tex__m_l1tex2xbar_req_cycles_stalled.sum",
        "l1tex__m_xbar2l1tex_read_bytes_pipe_lsu.sum",
        "l1tex__lsu_writeback_active.sum",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
        "sm__mio2rf_writeback_active.sum",
        "sm__mio_pq_read_cycles_active.sum",
        "sm__mio_pq_write_cycles_active.sum",
        "sm__inst_pending_pipe_lsu_mem_lg_cmd_read.sum",
        "sm__inst_pending_pipe_lsu_mem_shared_cmd_read.sum",
        "lts__xbar2lts_cycles_active.sum",
        "lts__lts2xbar_cycles_active.sum",
    ]
)


def sh(cmd: list[str], cwd: Path | None = None, timeout: int = 1800) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout} s"
    return p.returncode, p.stdout + (("\n# stderr\n" + p.stderr) if p.stderr.strip() else "")


def main(build: Path, out: Path, no_counters: str = "") -> int:
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

    def cool(limit_c: int = 45, wait_s: int = 900) -> None:
        t0 = time.time()
        while time.time() - t0 < wait_s:
            q = ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"]
            if int(sh(q)[1].split()[0]) <= limit_c:
                break
            time.sleep(10)
        print(f"cooled {time.time() - t0:.0f} s", flush=True)

    # round 14: rounds 12 and 13 lost kernels and sweeps to harness faults found after the run
    kernels, dropped = [], {}
    for d in sorted(p for p in (build / "testset").iterdir() if (p / "bin").exists()):
        rc, text = sh([str(d / "bin"), "0"], d, 120)
        if rc == 0 and "PREFLIGHT ok" in text:
            kernels.append(d)
        else:
            dropped[d.name] = text.strip()[-300:]
            print(f"PREFLIGHT dropped {d.name}: rc {rc} {dropped[d.name]}", flush=True)
    print(f"preflight: {len(kernels)} kernels go on, {len(dropped)} dropped", flush=True)
    (raw / "0-preflight.txt").write_text(json.dumps(dropped, indent=1) + "\n")
    cool()
    env("before")
    if any(s.get("busy") for s in steps):
        print("GPU busy: refusing the batch", flush=True)
        (out / "run.json").write_text(json.dumps({"steps": steps, "refused": "gpu busy"}, indent=1))
        return 2
    step("1-env-versions", ["sh", "-c", "nvidia-smi --version; nvcc --version | tail -2"])
    protocol("2")
    for d in sorted(build.glob("probe-*")):
        step(f"4-{d.name}", [str(d / "bin")], cwd=d)
    for d in kernels:
        step(f"4-{d.name}", [str(d / "bin"), "30"], cwd=d)
    # round 13: the clock falls with temperature, about 0.002 GHz per degree (r13, r13a: 2.494 at
    # 40 C, 2.454 at 59 C), and three passes in a row leave the 1 % band. The batch's clock and
    # canaries bracket the first pass; each repeat, in its own process, cools down first and has its
    # own clock: 4r1 in reverse order, 4r2 in order
    protocol("6")
    for i, order in ((1, reversed(kernels)), (2, kernels)):
        cool()
        protocol(f"3r{i}")
        for d in order:
            step(f"4r{i}-{d.name}", [str(d / "bin"), "30"], cwd=d)
        protocol(f"6r{i}")

    def ncu(tag: str, d: Path, metrics: str) -> None:
        entry = json.loads((d / "launch.json").read_text())["entry"]
        step(
            f"{tag}-{d.name}",
            [
                "sudo",
                "-n",
                NCU,
                "--csv",
                "--page",
                "raw",
                "--metrics",
                metrics,
                "--cache-control",
                "none",
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

    # a sequence's kernels have their counts on their own rows (one entry per profiler run)
    single = [d for d in kernels if "kernels" not in json.loads((d / "launch.json").read_text())]
    if no_counters:  # rows an earlier batch has counters for: timing only
        single = [d for d in single if not re.search(no_counters, d.name)]
    for d in single:
        ncu("5-ncu", d, METRICS)
    for d in single:
        ncu("5x-ncu", d, METRICS_X)
    for d in single:
        ncu("5u-ncu", d, METRICS_U)
        ncu("5v-ncu", d, METRICS_V)
    for d in sorted(build.glob("probe-*")):
        spec = Path("perfmodel/probes") / d.name.removeprefix("probe-") / "ncu.json"
        if not spec.exists():
            continue
        for pt in json.loads(spec.read_text()):
            for tag, metrics in (("5p-ncu", METRICS_P), ("5q-ncu", METRICS_Q)):
                step(
                    f"{tag}-{d.name}-{pt['tag']}",
                    [
                        "sudo",
                        "-n",
                        NCU,
                        "--csv",
                        "--page",
                        "raw",
                        "--metrics",
                        metrics,
                        "--cache-control",
                        "none",
                        "-k",
                        f"regex:{pt['kernel']}",
                        "--launch-skip",
                        str(pt["skip"]),
                        "--launch-count",
                        "1",
                        str(d / "bin"),
                        *pt["args"],
                    ],
                    cwd=d,
                )
    env("after")
    owned = sh(["find", str(out), "-not", "-user", str(os.getuid())])[1].strip()
    doc = {
        "steps": steps,
        "metrics": METRICS,
        "metrics_x": METRICS_X,
        "metrics_u": METRICS_U,
        "metrics_v": METRICS_V,
        "metrics_p": METRICS_P,
        "metrics_q": METRICS_Q,
        "not_owned_by_user": owned,
        "preflight_dropped": dropped,
    }
    (out / "run.json").write_text(json.dumps(doc, indent=1) + "\n")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), *sys.argv[3:4]))
