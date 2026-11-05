"""Evaluator-only payload. Execute exclusively inside an Executor GPU job.

This module is never copied into the implementation workspace. Its references,
seeds, generated tensors and tests are evaluator-owned.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import statistics
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def summary(samples_us: list[float]) -> dict[str, Any]:
    if len(samples_us) < 4 or not all(math.isfinite(x) and x > 0 for x in samples_us):
        raise ValueError("Timing needs at least four positive finite samples")
    q = statistics.quantiles(samples_us, n=4, method="inclusive")
    return {
        "samples_us": samples_us,
        "median_us": statistics.median(samples_us),
        "iqr_us": q[2] - q[0],
    }


def mathematical(torch: Any, case: str, inputs: tuple[Any, ...]) -> Any:
    if case == "gemm":
        a, b = inputs
        return (a.double() @ b.double()).half()
    q, k, v = inputs
    scores = (q.double() @ k.double().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).triu(1)
    scores = scores.masked_fill(mask, float("-inf"))
    return (torch.softmax(scores, dim=-1) @ v.double()).half()


def production(torch: Any, case: str, inputs: tuple[Any, ...]) -> Any:
    if case == "gemm":
        # cuBLAS FP32 accumulation with reduced precision reductions explicitly disabled.
        return torch.matmul(*inputs)
    return torch.nn.functional.scaled_dot_product_attention(*inputs, dropout_p=0.0, is_causal=True)


def inputs_for(torch: Any, case: str, shape: list[int], seed: int, pattern: str) -> tuple[Any, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    dims = [(shape[0], shape[2]), (shape[2], shape[1])] if case == "gemm" else [shape] * 3
    tensors = [
        torch.randn(d, generator=generator, device="cuda", dtype=torch.float32) for d in dims
    ]
    if pattern == "magnitude":
        tensors[0] *= 4
        tensors[-1] *= 0.25
    elif pattern == "cancellation":
        # Pair opposite terms; odd reduction tails remain active and independently random.
        axis = -1 if case == "gemm" else -2
        n = tensors[0].shape[axis] // 2
        if case == "gemm":
            tensors[0][:, 1 : 2 * n : 2] = -tensors[0][:, : 2 * n : 2]
            tensors[1][1 : 2 * n : 2] = tensors[1][: 2 * n : 2] * 1.001
        else:
            tensors[2][..., 1 : 2 * n : 2, :] = -tensors[2][..., : 2 * n : 2, :]
    elif pattern == "negative_logits" and case == "attention":
        tensors[0].fill_(4)
        tensors[1].fill_(-4)
    elif pattern == "mask" and case == "attention":
        # Dominant last value exposes accidental future attention, especially at row zero.
        tensors[2][..., -1, :] = 100
    return tuple(t.half().contiguous() for t in tensors)


def compare(torch: Any, actual: Any, expected: Any, manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(actual, torch.Tensor):
        return {"passed": False, "reason": "entry must return one Tensor"}
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return {"passed": False, "reason": "output shape or dtype differs"}
    if (
        actual.device != expected.device
        or not actual.is_contiguous()
        or not bool(torch.isfinite(actual).all())
    ):
        return {"passed": False, "reason": "output device differs or contains NaN/inf"}
    error = (actual.double() - expected.double()).abs()
    allowed = manifest["atol"] + manifest["rtol"] * expected.double().abs()
    return {
        "passed": bool((error <= allowed).all()),
        "max_abs_error": error.max().item(),
        "failing_elements": int((error > allowed).sum().item()),
    }


def timed(
    torch: Any, call: Any, inputs: tuple[Any, ...], manifest: dict[str, Any]
) -> dict[str, Any]:
    for _ in range(manifest["warmup"]):
        call(*inputs)
    torch.cuda.synchronize()
    samples = []
    for _ in range(manifest["repetitions"]):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        output = call(*inputs)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
        del output
    return summary(samples)


def evaluate(manifest: dict[str, Any], preflight: bool) -> dict[str, Any]:
    torch = importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    candidate = None
    if not preflight:
        spec = importlib.util.spec_from_file_location("candidate", "candidate.py")
        if spec is None or spec.loader is None:
            raise ValueError("Cannot load candidate.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        candidate = module.entry
    rows = []
    first_calls_s = 0.0
    validation_s = 0.0
    case = manifest["case"]
    for shape in manifest["shapes"]:
        checks = []
        production_checks = []
        first_call = True
        for seed in manifest["private_seeds"]:
            for pattern in manifest["patterns"]:
                inputs = inputs_for(torch, case, shape, seed, pattern)
                oracle = mathematical(torch, case, inputs)
                if not bool(torch.isfinite(oracle).all()):
                    raise ValueError("Oracle produced nonfinite output")
                prod = production(torch, case, inputs)
                prod_check = compare(torch, prod, oracle, manifest)
                production_checks.append({"seed": seed, "pattern": pattern, **prod_check})
                if not prod_check["passed"]:
                    raise ValueError(f"Production oracle smoke failed: {shape} {prod_check}")
                if candidate is not None:
                    snapshots = tuple(t.clone() for t in inputs)
                    start = time.monotonic()
                    actual = candidate(*inputs)
                    torch.cuda.synchronize()
                    elapsed_s = time.monotonic() - start
                    if first_call:
                        first_calls_s += elapsed_s
                        first_call = False
                    else:
                        validation_s += elapsed_s
                    check = compare(torch, actual, oracle, manifest)
                    if any(not torch.equal(a, b) for a, b in zip(inputs, snapshots, strict=True)):
                        check = {"passed": False, "reason": "candidate mutated inputs"}
                    checks.append({"seed": seed, "pattern": pattern, **check})
        row: dict[str, Any] = {
            "shape": shape,
            "checks": checks,
            "production_checks": production_checks,
            "correct": all(c["passed"] for c in checks),
        }
        if candidate is not None and row["correct"]:
            inputs = inputs_for(torch, case, shape, manifest["timing_seed"], "normal")
            row["production"] = timed(
                torch, lambda *xs: production(torch, case, xs), inputs, manifest
            )
            row["candidate"] = timed(torch, candidate, inputs, manifest)
            row["ratio"] = row["production"]["median_us"] / row["candidate"]["median_us"]
        rows.append(row)
    return {
        "preflight": preflight,
        "passed": all(r["correct"] for r in rows),
        "rows": rows,
        "first_call_compile_and_execution_s": first_calls_s,
        "validation_candidate_calls_s": validation_s,
        "compile_s": None,
        "compilation": "included in first calls; compiler-only time unavailable",
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0),
    }


def main() -> None:
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    manifest_data = Path("manifest.json").read_bytes()
    manifest = json.loads(manifest_data)
    Path("out").mkdir(exist_ok=True)
    try:
        result = evaluate(manifest, "--preflight" in sys.argv)
    except Exception as exc:  # noqa: BLE001 - persist failure details, then exit nonzero
        # Retain compiler/runtime errors without disguising infrastructure as a wrong answer.
        result = {
            "passed": False,
            "preflight": "--preflight" in sys.argv,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    result.update(
        {
            "started_at": started_at,
            "worker_wall_s": time.monotonic() - started,
            "manifest_identity": hashlib.sha256(manifest_data).hexdigest(),
        }
    )
    if Path("candidate.py").exists():
        result["candidate_identity"] = hashlib.sha256(Path("candidate.py").read_bytes()).hexdigest()
    Path("out/result.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
