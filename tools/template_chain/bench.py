"""Container payload: task inputs, independent references, comparators and graph timing.

python bench.py request.json RESULT.json

Runs inside a worker image, so it imports only torch, and the request's family or comparator.
A candidate exposes `prepare(shape, x, inputs) -> run`, where `run()` returns its output tensors,
uses only the GPU and may be captured in a CUDA graph. Work done by `prepare` is not timed.
"""

import importlib.util
import json
import math
import re
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch

DEVICE = "cuda"


def decode_table(shape: dict, seed: int) -> dict:
    """The page table is the planning input: fixed per case, not per step."""
    g = torch.Generator().manual_seed(seed)
    b, page = shape["B"], shape["page"]
    lengths = torch.randint(shape["min_len"], shape["max_len"], (b,), generator=g)
    pages = [(int(n) + page - 1) // page for n in lengths]
    total = sum(pages)
    return {
        "indptr": torch.tensor([0, *pages]).cumsum(0).to(DEVICE, torch.int32),
        "indices": torch.randperm(total, generator=g).to(DEVICE, torch.int32),
        "last": ((lengths - 1) % page + 1).to(DEVICE, torch.int32),
        "num_pages": total,
    }


def decode_inputs(shape: dict, seed: int) -> dict:
    t = decode_table(shape, shape["table_seed"])
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    h, hk, d, page = shape["H"], shape["Hk"], shape["D"], shape["page"]

    def rand(*size: int) -> torch.Tensor:
        return torch.randn(size, device=DEVICE, generator=g).to(torch.float16)

    return {
        **t,
        "q": rand(shape["B"], h, d),
        "k": rand(t["num_pages"], page, hk, d),
        "v": rand(t["num_pages"], page, hk, d),
    }


def decode_reference(shape: dict, i: dict) -> dict:
    h, hk, d, page = shape["H"], shape["Hk"], shape["D"], shape["page"]
    g = h // hk
    scale = 1 / math.sqrt(d)
    o = torch.zeros(shape["B"], h, d, device=DEVICE, dtype=torch.float32)
    lse = torch.zeros(shape["B"], h, device=DEVICE, dtype=torch.float32)
    for b in range(shape["B"]):
        a, e = int(i["indptr"][b]), int(i["indptr"][b + 1])
        n = (e - a - 1) * page + int(i["last"][b])
        pages = i["indices"][a:e].long()
        k = i["k"][pages].reshape(-1, hk, d)[:n].double()
        v = i["v"][pages].reshape(-1, hk, d)[:n].double()
        q = i["q"][b].double().reshape(hk, g, d)
        s = scale * torch.einsum("hgd,nhd->hgn", q, k)
        o[b] = torch.einsum("hgn,nhd->hgd", s.softmax(-1), v).reshape(h, d).float()
        lse[b] = (torch.logsumexp(s, -1) / math.log(2)).reshape(h).float()
    return {"o": o, "lse": lse}


def forward_inputs(shape: dict, seed: int) -> dict:
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    size = (1, shape["S"], shape["H"], shape["D"])
    return {
        name: torch.randn(size, device=DEVICE, generator=g).to(torch.float16)
        for name in ("q", "k", "v")
    }


def forward_reference(shape: dict, i: dict) -> dict:
    s_, h, d = shape["S"], shape["H"], shape["D"]
    o = torch.zeros(1, s_, h, d, device=DEVICE, dtype=torch.float32)
    lse = torch.zeros(1, h, s_, device=DEVICE, dtype=torch.float32)
    mask = torch.ones(s_, s_, device=DEVICE, dtype=torch.bool).tril()
    for head in range(h):
        q, k, v = (i[n][0, :, head].double() for n in ("q", "k", "v"))
        score = (q @ k.T / math.sqrt(d)).masked_fill(~mask, -math.inf)
        o[0, :, head] = (score.softmax(-1) @ v).float()
        lse[0, head] = torch.logsumexp(score, -1).float()
    return {"o": o, "lse": lse}


def decode_comparator(shape: dict, i: dict):
    import flashinfer

    workspace = torch.zeros(256 << 20, dtype=torch.uint8, device=DEVICE)
    w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        paged_kv_indptr_buffer=i["indptr"].clone(),
        paged_kv_indices_buffer=i["indices"].clone(),
        paged_kv_last_page_len_buffer=i["last"].clone(),
        backend="fa2",
    )
    w.plan(
        i["indptr"],
        i["indices"],
        i["last"],
        shape["H"],
        shape["Hk"],
        shape["D"],
        shape["page"],
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        sm_scale=1 / math.sqrt(shape["D"]),
    )
    o = torch.zeros(shape["B"], shape["H"], shape["D"], device=DEVICE, dtype=torch.float16)
    lse = torch.zeros(shape["B"], shape["H"], device=DEVICE, dtype=torch.float32)

    def run() -> dict:
        w.run(i["q"], (i["k"], i["v"]), out=o, lse=lse, return_lse=True)
        return {"o": o, "lse": lse}

    return run


def forward_comparator(shape: dict, i: dict):
    sys.path.insert(0, "/w/baseline")
    from binding import original_entry  # the pinned FlashAttention-2 build of the first campaign

    def run() -> dict:
        o, lse = original_entry(i["q"], i["k"], i["v"])
        return {"o": o, "lse": lse}

    return run


TASKS = {
    "paged_decode": {
        "inputs": decode_inputs,
        "reference": decode_reference,
        "comparator": decode_comparator,
        "dynamic": ("q", "k", "v"),
        "tolerance": {"o": (0.02, 0.01), "lse": (0.02, 0.002)},
    },
    "causal_forward": {
        "inputs": forward_inputs,
        "reference": forward_reference,
        "comparator": forward_comparator,
        "dynamic": ("q", "k", "v"),
        "tolerance": {"o": (0.01, 0.01), "lse": (0.001, 0.001)},
    },
}


def worst(out: dict, ref: dict, tolerance: dict) -> dict:
    """Largest error over each allowance `absolute + relative * |reference|`; pass means <= 1."""
    verdict = {}
    for name, r in ref.items():
        if name not in out:
            return {"missing": name}
        o = out[name].float()
        if o.shape != r.shape:
            return {"shape": f"{name}: {list(o.shape)} expected {list(r.shape)}"}
        a, b = tolerance[name]
        verdict[name] = float(((o - r).abs() / (a + b * r.abs())).nan_to_num(1e9).max())
    return verdict


def timed(run, warm: int, n: int, graph) -> list[float]:
    samples = []
    for i in range(warm + n):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        if i >= warm:
            samples.append(begin.elapsed_time(end) * 1e3)
    return samples


class Records:
    """Every kernel TileLang compiles here: its source and what the compiler reports for it."""

    def __init__(self, out: Path) -> None:
        self.out = out
        self.kernels: list[dict] = []
        import tilelang

        self.original = tilelang.compile
        env = tilelang.env
        self.includes = f"-I{env.TILELANG_TEMPLATE_PATH} -I{env.CUTLASS_INCLUDE_DIR}"
        tilelang.compile = self.compile

    def compile(self, *args, **kwargs):
        k = self.original(*args, **kwargs)
        src = k.get_kernel_source()
        host = k.get_host_source() or ""
        packed = re.findall(r"v_int64\) = \(\(int64_t\)(\d+)\)", host)
        bound = re.search(r"__launch_bounds__\((\d+)", src)
        self.kernels.append(
            {
                "smem": int(packed[-1]) if packed and "extern __shared__" in src else 0,
                "threads": int(bound.group(1)) if bound else None,
                "source": src,
            }
        )
        return k

    def usage(self) -> list[dict]:
        rows = []
        for n, k in enumerate(self.kernels):
            path = self.out / f"k{n}.cu"
            path.write_text(k["source"])
            done = subprocess.run(
                f"nvcc -arch=sm_121 -O3 --std=c++20 {self.includes} "
                f"-res-usage -c {path} -o {path}.o",
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )
            found = re.findall(r"Used (\d+) registers", done.stdout + done.stderr)
            if not found:
                (self.out / f"k{n}.log").write_text(done.stdout + done.stderr)
            rows.append(
                {
                    "regs": int(found[-1]) if found else None,
                    "smem": k["smem"],
                    "threads": k["threads"],
                }
            )
        return rows


def candidate_module(path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("family", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def one(request: dict, job: dict, records: Records | None) -> dict:
    task = TASKS[request["task"]]
    shape = job["shape"]
    i = task["inputs"](shape, job.get("seed", 1))
    began = time.time()
    if job["kind"] == "comparator":
        run = task["comparator"](shape, i)
    else:
        if records:
            records.kernels.clear()
        run = candidate_module(Path(request["family"])).prepare(shape, job["x"], i)
    out = run()
    torch.cuda.synchronize()
    result: dict = {"prepare_s": time.time() - began}
    if records and records.kernels:
        result["compiled"] = records.usage()
    ref = task["reference"](shape, i)
    result["errors"] = worst(out, ref, task["tolerance"])
    correct = all(isinstance(v, float) and v <= 1 for v in result["errors"].values())
    result["correct"] = correct
    if job.get("timing") and correct:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = run()
        samples = timed(run, request["warm"], request["n"], graph)
        second = task["inputs"](shape, job.get("seed", 1) + 1)
        for name in task["dynamic"]:
            i[name].copy_(second[name])
        graph.replay()
        torch.cuda.synchronize()
        result["replay_errors"] = worst(out, task["reference"](shape, i), task["tolerance"])
        result["replay_correct"] = all(
            isinstance(v, float) and v <= 1 for v in result["replay_errors"].values()
        )
        q = statistics.quantiles(samples, n=4, method="inclusive")
        result.update(median_us=statistics.median(samples), iqr_us=q[2] - q[0], samples_us=samples)
    return result


def main() -> None:
    request = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    work = Path(request.get("work", "."))
    records = Records(work) if request.get("records") else None
    results = {}
    for job in request["jobs"]:
        try:
            results[job["id"]] = one(request, job, records)
        except Exception as e:  # noqa: BLE001
            trace = traceback.format_exc()[-3000:]
            results[job["id"]] = {"error": f"{type(e).__name__}: {e}"[:600], "trace": trace}
        torch.cuda.empty_cache()
        out.write_text(json.dumps(results))


if __name__ == "__main__":
    main()
