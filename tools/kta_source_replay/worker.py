"""Evaluator-only BSHD O/LSE correctness and complete-entry graph timing payload."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import math
import random
import statistics
import struct
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audit_candidate(source: str) -> None:
    """Reject explicit reference delegation/host extraction; command audits remain required."""
    tree = ast.parse(source)
    forbidden = {"flash_attn", "flashinfer", "xformers", "triton"}
    torch_aliases = {"torch"}
    torch_math_calls: set[str] = set()
    tensor_math = set(
        "matmul mm bmm einsum softmax logsumexp compile "
        "scaled_dot_product_attention _scaled_dot_product_flash_attention".split()
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in forbidden:
                    raise ValueError(f"Forbidden candidate import: {alias.name}")
                if alias.name.split(".")[0] == "torch":
                    torch_aliases.add(alias.asname or "torch")
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in forbidden:
                raise ValueError(f"Forbidden candidate import: {node.module}")
            if (node.module or "").startswith("torch"):
                torch_aliases.update(a.asname or a.name for a in node.names)
                torch_math_calls.update(
                    a.asname or a.name for a in node.names if a.name in tensor_math
                )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if isinstance(fn, ast.Name) and fn.id == "open":
            raise ValueError("Candidate runtime source reads are forbidden")
        if isinstance(fn, ast.Name) and fn.id in torch_math_calls:
            raise ValueError("Prebuilt Torch attention arithmetic is forbidden")
        if isinstance(fn, ast.Attribute):
            if fn.attr in {"cpu", "numpy", "item", "tolist", "read_text", "read_bytes"}:
                raise ValueError("Host extraction/source reads are forbidden")
            root = fn
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in torch_aliases and fn.attr in tensor_math:
                raise ValueError("Prebuilt Torch attention arithmetic is forbidden")


def stats(values: list[float]) -> dict[str, Any]:
    if len(values) < 4 or not all(math.isfinite(x) and x > 0 for x in values):
        raise ValueError("Need four positive finite timing samples")
    q = statistics.quantiles(values, n=4, method="inclusive")
    return {"median_us": statistics.median(values), "iqr_us": q[2] - q[0], "samples_us": values}


def drift(values: list[float]) -> float:
    return max(values) / min(values) - 1


def window_summary(original: list[float], candidate: list[float], timing: dict) -> dict:
    a, b = stats(original), stats(candidate)
    half = len(original) // 2
    half_drift = drift([statistics.median(original[:half]), statistics.median(original[half:])])
    validity = timing["validity"]
    valid = (
        a["iqr_us"] / a["median_us"] <= validity["original_IQR_over_median_max"]
        and half_drift <= validity["original_half_window_median_drift_max"]
    )
    return {
        "original": a,
        "candidate": b,
        "ratio": a["median_us"] / b["median_us"],
        "time_difference_percent": 100 * (b["median_us"] / a["median_us"] - 1),
        "original_half_drift": half_drift,
        "valid": valid,
    }


def inputs(torch: Any, shape: list[int], seed: int, pattern: str) -> tuple:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v = [
        torch.randn(shape, dtype=torch.float32, device="cuda", generator=generator)
        for _ in range(3)
    ]
    if pattern == "magnitude":
        q *= 4
        v *= 0.25
    elif pattern == "cancellation":
        n = shape[1] // 2
        v[:, 1 : 2 * n : 2] = -v[:, : 2 * n : 2]
    elif pattern == "mask":
        v[:, -1] = 100
    elif pattern == "negative_logits":
        q.fill_(4)
        k.fill_(-4)
    return tuple(t.half().contiguous() for t in (q, k, v))


def oracle(torch: Any, xs: tuple) -> tuple:
    q, k, v = xs
    scale = struct.unpack("f", struct.pack("f", 1 / math.sqrt(128)))[0]
    scores = torch.einsum("bihd,bjhd->bhij", q.double(), k.double()) * scale
    s = q.shape[1]
    index = torch.arange(s, device=q.device)
    masked = torch.where(index[None, :] <= index[:, None], scores, float("-inf"))
    probabilities = torch.softmax(masked, dim=-1)
    output = torch.einsum("bhij,bjhd->bihd", probabilities, v.double()).half().contiguous()
    return output, torch.logsumexp(masked, dim=-1).float().contiguous()


def tensor_check(torch: Any, value: Any, expected: Any, tolerance: dict) -> dict:
    if not isinstance(value, torch.Tensor):
        return {"passed": False, "reason": "not a tensor"}
    if value.shape != expected.shape or value.dtype != expected.dtype:
        return {"passed": False, "reason": "wrong shape/dtype"}
    if value.device != expected.device or not value.is_contiguous():
        return {"passed": False, "reason": "wrong device/layout"}
    if not bool(torch.isfinite(value).all()):
        return {"passed": False, "reason": "NaN/inf"}
    error = (value.double() - expected.double()).abs()
    bad = error > tolerance["atol"] + tolerance["rtol"] * expected.double().abs()
    return {
        "passed": not bool(bad.any()),
        "max_abs_error": error.max().item(),
        "failing_elements": int(bad.sum().item()),
    }


def check(torch: Any, outputs: Any, expected: tuple, xs: tuple, before: tuple, task: dict) -> dict:
    if not isinstance(outputs, tuple | list) or len(outputs) != 2:
        return {"passed": False, "reason": "entry must return (O,LSE)"}
    result: dict[str, Any] = {
        name: tensor_check(torch, value, reference, task[name])
        for name, value, reference in zip(("O", "LSE"), outputs, expected, strict=True)
    }
    result["inputs_unchanged"] = all(torch.equal(a, b) for a, b in zip(xs, before, strict=True))
    result["new_outputs"] = all(
        value.data_ptr() != x.data_ptr()
        for value in outputs
        if isinstance(value, torch.Tensor)
        for x in xs
    )
    result["passed"] = (
        result["O"]["passed"]
        and result["LSE"]["passed"]
        and result["inputs_unchanged"]
        and result["new_outputs"]
    )
    return result


def load_entry(path: str, name: str) -> Any:
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, name)


def capture(torch: Any, entry: Any, xs: tuple) -> tuple:
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            entry(*xs)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = entry(*xs)
    torch.cuda.synchronize()
    return graph, outputs


def copy_inputs(dest: tuple, source: tuple) -> None:
    for dst, src in zip(dest, source, strict=True):
        dst.copy_(src)


def graph_validation(
    torch: Any,
    entry: Any,
    graph: Any,
    outputs: tuple,
    addresses: tuple,
    fresh: tuple,
    expected: tuple,
    task: dict,
    history: list,
) -> dict:
    before = tuple(x.clone() for x in fresh)
    eager = entry(*fresh)
    torch.cuda.synchronize()
    eager_check = check(torch, eager, expected, fresh, before, task)
    if isinstance(eager, tuple | list) and len(eager) == 2:
        preserved = all(torch.equal(out, saved) for old in history for out, saved in old)
        unique = all(
            out.data_ptr() != previous.data_ptr()
            for out in eager
            if isinstance(out, torch.Tensor)
            for old in history
            for previous, _ in old
        )
        eager_check["previous_outputs_preserved"] = preserved
        eager_check["fresh_output_allocations"] = unique
        eager_check["passed"] = eager_check["passed"] and preserved and unique
        history.append([(out, out.clone()) for out in eager if isinstance(out, torch.Tensor)])
    copy_inputs(addresses, before)
    graph.replay()
    torch.cuda.synchronize()
    graph_check = check(torch, outputs, expected, addresses, before, task)
    return {
        "eager": eager_check,
        "captured": graph_check,
        "passed": eager_check["passed"] and graph_check["passed"],
    }


def event_time(torch: Any, operation: Any, repetitions: int) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        operation()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repetitions


def measure(
    torch: Any,
    entries: tuple,
    graphs: tuple,
    addresses: tuple,
    outputs: tuple,
    timing_input: tuple,
    expected: tuple,
    campaign: dict,
) -> dict:
    timing = campaign["timing"]
    for target in addresses:
        copy_inputs(target, timing_input)
    for graph in graphs:
        for _ in range(timing["warmup_replays"]):
            graph.replay()
    torch.cuda.synchronize()
    checks = [
        check(torch, out, expected, xs, timing_input, campaign["task"])
        for out, xs in zip(outputs, addresses, strict=True)
    ]
    order = [0] * 15 + [1] * 15
    random.Random(timing["measurement_seed"]).shuffle(order)
    if campaign.get("_swap_labels"):
        order = [1 - x for x in order]
    samples: tuple[list[float], list[float]] = ([], [])
    paired = []
    for first in order:
        pair = {}
        for index in (first, 1 - first):
            value = event_time(torch, graphs[index].replay, timing["replays_per_block"])
            samples[index].append(value)
            pair[("original", "candidate")[index]] = value
        paired.append({"first": ("original", "candidate")[first], **pair})
    checks += [
        check(torch, out, expected, xs, timing_input, campaign["task"])
        for out, xs in zip(outputs, addresses, strict=True)
    ]
    eager = []
    for entry, xs in zip(entries, addresses, strict=True):
        for _ in range(10):
            entry(*xs)
        torch.cuda.synchronize()
        eager.append(
            stats([event_time(torch, lambda entry=entry, xs=xs: entry(*xs), 1) for _ in range(50)])
        )
        last = entry(*xs)
        torch.cuda.synchronize()
        checks.append(check(torch, last, expected, xs, timing_input, campaign["task"]))
    checks += [
        check(torch, out, expected, xs, timing_input, campaign["task"])
        for out, xs in zip(outputs, addresses, strict=True)
    ]
    result = window_summary(*samples, timing)
    result.update(
        {
            "paired_blocks": paired,
            "post_warmup_and_timing_checks": checks,
            "eager_diagnostic": {"original": eager[0], "candidate": eager[1]},
            "outputs_valid": all(c["passed"] for c in checks),
        }
    )
    return result


def canaries(torch: Any, original: Any, task: dict) -> dict:
    xs = inputs(torch, [1, 5, 8, 128], 601, "normal")
    expected = oracle(torch, xs)
    before = tuple(x.clone() for x in xs)
    bad = {
        "zeros": (torch.zeros_like(expected[0]), expected[1].clone()),
        "NaN": (torch.full_like(expected[0], float("nan")), expected[1].clone()),
        "wrong LSE": (expected[0].clone(), expected[1] + 1),
    }
    results = {
        name: not check(torch, value, expected, xs, before, task)["passed"]
        for name, value in bad.items()
    }
    xs[0].add_(1)
    results["input mutation"] = not check(torch, expected, expected, xs, before, task)["passed"]
    # Capture a deliberately stale output path at old values, while the captured input
    # addresses really do receive changed values; mutation must independently pass.
    xs = tuple(x.clone() for x in before)
    stale = tuple(x.clone() for x in expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for value in stale:
            value.add_(0)
    changed = inputs(torch, [1, 5, 8, 128], 602, "mask")
    copy_inputs(xs, changed)
    graph.replay()
    torch.cuda.synchronize()
    stale_check = check(torch, stale, oracle(torch, changed), xs, changed, task)
    results["stale captured graph output"] = (
        stale_check["inputs_unchanged"] and not stale_check["O"]["passed"]
    )
    return results


def run(campaign: dict, request: dict) -> dict:
    if request["mode"] not in {"preflight", "development", "final"}:
        raise ValueError("Unknown evaluator mode")
    patterns = campaign["validation"]["patterns"]
    count = 8 if request["mode"] == "final" else 2
    if set(request["seeds"]) != set(patterns) or not all(
        len(request["seeds"][p]) == count and len(set(request["seeds"][p])) == count
        for p in patterns
    ):
        raise ValueError("Incorrect frozen pattern/seed coverage")
    if not request["baseline_identity"] or not any(
        name.endswith(".so") for name in request["baseline_identity"]
    ):
        raise ValueError("Original binding/binary inventory required")
    if request["source_revision"] != campaign["source"]["revision"]:
        raise ValueError("Original revision differs from frozen campaign")
    for relative, expected_identity in request["baseline_identity"].items():
        path = Path("/baseline") / relative
        if not path.resolve().is_relative_to(Path("/baseline")):
            raise ValueError("Baseline inventory path escapes read-only mount")
        if digest(path.read_bytes()) != expected_identity:
            raise ValueError(f"Original baseline changed: {relative}")
    torch = importlib.import_module("torch")
    original = load_entry(request["baseline_module"], "original_entry")
    if request["mode"] != "preflight":
        audit_candidate(Path("candidate.py").read_text())
    candidate = original if request["mode"] == "preflight" else load_entry("candidate.py", "entry")
    campaign["_swap_labels"] = request["mode"] == "preflight" and request["window"] == 2
    result: dict[str, Any] = {
        "rows": [],
        "passed": True,
        "mode": request["mode"],
        "baseline_identity": request["baseline_identity"],
    }
    if request["mode"] == "preflight":
        result["canaries"] = canaries(torch, original, campaign["task"])
        if not all(result["canaries"].values()):
            raise ValueError(f"Fault canary accepted invalid output: {result['canaries']}")
    for shape in campaign["task"]["shapes"]:
        timing_input = inputs(torch, shape, campaign["timing"]["measurement_seed"], "normal")
        expected_timing = oracle(torch, timing_input)
        addresses = tuple(tuple(x.clone() for x in timing_input) for _ in range(2))
        entries = (original, candidate)
        start = time.monotonic()
        captured = tuple(
            capture(torch, entry, xs) for entry, xs in zip(entries, addresses, strict=True)
        )
        graphs, output = tuple(x[0] for x in captured), tuple(x[1] for x in captured)
        histories: tuple[list, list] = ([], [])
        row: dict[str, Any] = {
            "shape": shape,
            "checks": [],
            "compile_capture_s": time.monotonic() - start,
        }
        for pattern, seeds in request["seeds"].items():
            for seed in seeds:
                fresh = inputs(torch, shape, seed, pattern)
                expected = oracle(torch, fresh)
                if not all(bool(torch.isfinite(x).all()) for x in expected):
                    raise ValueError("Nonfinite mathematical oracle")
                checksum = [digest(x.cpu().numpy().tobytes()) for x in fresh]
                checked = [
                    graph_validation(
                        torch,
                        entry,
                        graph,
                        out,
                        xs,
                        tuple(x.clone() for x in fresh),
                        expected,
                        campaign["task"],
                        history,
                    )
                    for entry, graph, out, xs, history in zip(
                        entries, graphs, output, addresses, histories, strict=True
                    )
                ]
                checked_row = {
                    "shape": shape,
                    "pattern": pattern,
                    "seed": seed,
                    "input_checksums": checksum,
                    "original": checked[0],
                    "candidate": checked[1],
                }
                row["checks"].append(checked_row)
                with Path("out/checks.jsonl").open("a") as stream:
                    stream.write(json.dumps(checked_row, allow_nan=False) + "\n")
        row["correct"] = all(
            c[side]["passed"] for c in row["checks"] for side in ("original", "candidate")
        )
        if row["correct"]:
            row["timing"] = measure(
                torch, entries, graphs, addresses, output, timing_input, expected_timing, campaign
            )
            row["correct"] = row["timing"]["outputs_valid"]
            if request["mode"] == "final" and request["window"] == 1:
                trace = f"out/candidate-kernel-trace-S{shape[1]}.json"
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    graphs[1].replay()
                    torch.cuda.synchronize()
                profile.export_chrome_trace(trace)
                row["untimed_native_kernel_trace"] = trace
        result["rows"].append(row)
        with Path("out/shapes.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        result["passed"] = result["passed"] and row["correct"]
    result["window_valid"] = result["passed"] and all(r["timing"]["valid"] for r in result["rows"])
    result["meets_target"] = result["window_valid"] and all(
        r["timing"]["ratio"] >= 0.90 for r in result["rows"]
    )
    return result


def main() -> None:
    started = time.monotonic()
    Path("out").mkdir(exist_ok=True)
    request = json.loads(Path("request.json").read_text())
    try:
        result = run(json.loads(Path("campaign.json").read_text()), request)
    except Exception as exc:  # noqa: BLE001 - process boundary retains failure and exits nonzero
        result = {
            "passed": False,
            "window_valid": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
        }
    result.update(
        {
            "worker_active_s": time.monotonic() - started,
            "request_identity": digest(Path("request.json").read_bytes()),
            "worker_identity": digest(Path(__file__).read_bytes()),
        }
    )
    if Path("candidate.py").exists():
        result["candidate_identity"] = digest(Path("candidate.py").read_bytes())
    Path("out/result.json").write_text(json.dumps(result, allow_nan=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
