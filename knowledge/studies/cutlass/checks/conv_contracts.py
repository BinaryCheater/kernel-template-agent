"""Replay implicit patch coordinates and retained convolution frontend guards."""

from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import itertools
import json
import math
import sys
from pathlib import Path

from dense_contracts import DenseSymbols
from frontend_contract import DTYPES

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell/kernel/conv/"


def load_dense_frontend(root: Path) -> tuple:
    """Load only retained pure predicates, replacing dtype symbols and exceptions."""
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    DTYPES.update({"TFloat32": 32, "Int8": 8, "Uint8": 8, "Int32": 32})
    source_path = PREFIX + "dense_implicit_gemm_fprop.py"
    data = (root / source_path).read_bytes()
    retained = next(x for x in manifest["files"] if x["path"] == source_path)
    assert hashlib.sha256(data).hexdigest() == retained["sha256"]
    source = ast.parse(data)
    parent_path = root / (
        "examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent_dynamic.py"
    )
    parent = next(x for x in ast.parse(parent_path.read_text()).body if isinstance(x, ast.ClassDef))
    parent.body = [
        x
        for x in parent.body
        if isinstance(x, ast.FunctionDef)
        and x.name in {"check_supported_dtypes", "check_mma_tiler_and_cluster_shape"}
    ]
    cls = next(x for x in source.body if isinstance(x, ast.ClassDef))
    cls.body = [
        x
        for x in cls.body
        if isinstance(x, ast.FunctionDef)
        and x.name
        in {
            "check_mma_tiler_and_cluster_shape",
            "check_preferred_cluster_problem_size",
            "can_implement",
        }
    ]
    funcs = [
        x
        for x in source.body
        if isinstance(x, ast.FunctionDef)
        and x.name
        in {
            "_check_tensor_alignment",
            "_check_im2col_descriptor_limits",
            "_check_swizzle_size",
            "compute_zpq",
        }
    ]
    helpers = ast.parse((root / "python/CuTeDSL/cutlass/utils/blackwell_helpers.py").read_text())
    arch_guard = next(
        x
        for x in helpers.body
        if isinstance(x, ast.FunctionDef) and x.name == "check_int8_mma_arch"
    )
    selected = ast.Module(body=[arch_guard, *funcs, parent, cls], type_ignores=[])
    ns: dict = {"widths": DTYPES, "math": math}
    exec(
        compile(ast.fix_missing_locations(DenseSymbols().visit(selected)), source_path, "exec"), ns
    )
    kind = ns[cls.name]
    return ns, kind, data, retained, source_path


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    ns, kind, data, retained, source_path = load_dense_frontend(root)
    obj = kind.__new__(kind)
    obj.acc_dtype, obj.use_2cta_instrs = "Float32", False
    obj.mma_tiler_mn = (128, 128)
    obj.cluster_shape_mn = obj.preferred_cluster_shape_mn = obj.fallback_cluster_shape_mn = (1, 1)
    obj.split_k, obj.use_tma_store = 1, True
    obj.swizzle_size, obj.raster_along = 1, "m"
    # Frontend source witness:125 output positions,216 reduction elements, output16 channels.
    args = [(1, 8, 5, 5, 5), 16, "Float16", "Float16", (3, 3, 3)]
    assert obj.can_implement(*args, (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1))
    assert ns["compute_zpq"]((5, 5, 5), (3, 3, 3), (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)) == (
        5,
        5,
        5,
    )
    with contextlib.redirect_stdout(io.StringIO()):
        assert not obj.can_implement(
            (1, 7, 5, 5, 5), *args[1:], (1, 1, 1), (1, 1, 1), (1, 1, 1), (1, 1, 1)
        )
    guard_cases = 0
    for flt, dil, upper, lower, stride in itertools.product(
        (1, 2, 3, 17, 33), (1, 2, 4), (0, 1, 15, 16), (0, 1, 16, 17), (1, 8, 9)
    ):
        expected = (
            -16 <= -lower <= 15
            and -16 <= upper - (flt - 1) * dil <= 15
            and (flt - 1) * dil <= 31
            and stride <= 8
        )
        try:
            ns["_check_im2col_descriptor_limits"](
                (flt,) * 3, (upper,) * 3, (lower,) * 3, (stride,) * 3, (dil,) * 3
            )
            actual = True
        except ValueError:
            actual = False
        assert actual == expected
        guard_cases += 1

    # Field-width guard alone does not enforce positive stride; the rule requires it.
    ns["_check_im2col_descriptor_limits"]((1,) * 3, (0,) * 3, (0,) * 3, (0,) * 3, (1,) * 3)

    # Source lower padding is the leading side, in reverse WHD descriptor order.
    helper_text = (root / "python/CuTeDSL/cutlass/cute/nvgpu/helpers.py").read_text()
    assert "lower_corner_whd = (-pad_lower_w, -pad_lower_h, -pad_lower_d)" in helper_text
    patch_elements = output_values = geometries = 0
    wrong_pad = wrong_tap_order = None
    for stride, dilation, low, high in itertools.product(
        ((1, 1, 1), (1, 2, 1)),
        ((1, 1, 1), (1, 1, 2)),
        ((0, 0, 0), (1, 0, 1)),
        ((0, 1, 0), (0, 0, 1)),
    ):
        spatial, taps, channels = (3, 4, 5), (2, 2, 2), 2
        out = ns["compute_zpq"](spatial, taps, stride, high, low, dilation)
        assert all(x > 0 for x in out)

        def input_at(
            b: int,
            coords: tuple[int, ...],
            c: int,
            bounds: tuple[int, ...] = spatial,
            channel_count: int = channels,
        ) -> int:
            if not all(0 <= x < extent for x, extent in zip(coords, bounds, strict=True)):
                return 0
            d, h, w = coords
            return 1 + c + channel_count * (w + 5 * (h + 4 * (d + 3 * b)))

        def weight_at(o: int, tap: tuple[int, ...], c: int) -> int:
            t, r, s = tap
            return 1 + c + 2 * s + 4 * r + 8 * t + 16 * o

        matrix_rows = []
        m_extent = 2 * math.prod(out)
        k_extent = channels * math.prod(taps)
        for m in range(m_extent):
            high_m, q = divmod(m, out[2])
            high_m, p = divmod(high_m, out[1])
            b, z = divmod(high_m, out[0])
            assert m == q + out[2] * (p + out[1] * (z + out[0] * b))
            row = []
            for k in range(k_extent):
                high_k, c = divmod(k, channels)
                high_k, s = divmod(high_k, taps[2])
                t, r = divmod(high_k, taps[1])
                coords = tuple(
                    v * st - pad + tap * dil
                    for v, st, pad, tap, dil in zip(
                        (z, p, q), stride, low, (t, r, s), dilation, strict=True
                    )
                )
                row.append(input_at(b, coords, c))
                patch_elements += 1
                if low != high:
                    bad_coords = tuple(
                        v * st - pad + tap * dil
                        for v, st, pad, tap, dil in zip(
                            (z, p, q), stride, high, (t, r, s), dilation, strict=True
                        )
                    )
                    if input_at(b, bad_coords, c) != row[-1] and wrong_pad is None:
                        wrong_pad = {
                            "correct": row[-1],
                            "swapped_padding": input_at(b, bad_coords, c),
                        }
            matrix_rows.append(row)
        weights = [
            [weight_at(o, (t, r, s), c) for t, r, s, c in itertools.product(range(2), repeat=4)]
            for o in range(2)
        ]
        for b, z, p, q, o in itertools.product(range(2), *(range(x) for x in out), range(2)):
            m = ((b * out[0] + z) * out[1] + p) * out[2] + q
            direct = 0
            for t, r, s, c in itertools.product(range(2), repeat=4):
                coords = (
                    z * stride[0] - low[0] + t * dilation[0],
                    p * stride[1] - low[1] + r * dilation[1],
                    q * stride[2] - low[2] + s * dilation[2],
                )
                direct += input_at(b, coords, c) * weight_at(o, (t, r, s), c)
            lowered = sum(x * y for x, y in zip(matrix_rows[m], weights[o], strict=True))
            assert lowered == direct
            if wrong_tap_order is None:
                bad_weights = [
                    weight_at(o, (s, r, t), c)
                    for t, r, s, c in itertools.product(range(2), repeat=4)
                ]
                wrong = sum(x * y for x, y in zip(matrix_rows[m], bad_weights, strict=True))
                if wrong != direct:
                    wrong_tap_order = {"correct": direct, "swapped_t_s": wrong}
            output_values += 1
        geometries += 1
    assert wrong_pad is not None and wrong_tap_order is not None

    # Bias vectors cover every staged column, including FP32 N256's second pass.
    bias_domains = 0
    for bits, tile_n, valid_n in itertools.product(
        (8, 16, 32), (32, 64, 128, 256), (16, 32, 144, 256)
    ):
        elems = 32 // bits
        n_active = tile_n // elems
        lanes = [
            base + lane
            for base in range(0, n_active, 128)
            for lane in range(128)
            if base + lane < n_active
        ]
        covered = [lane * elems + i for lane in lanes for i in range(elems)]
        assert covered == list(range(tile_n))
        for lane in lanes:
            vector = range(lane * elems, (lane + 1) * elems)
            assert all(x < valid_n for x in vector) or all(x >= valid_n for x in vector)
        bias_domains += 1
    assert sum(1 for _ in range(0, 256, 128)) == 2
    methods = {
        x.name: [x.lineno, x.end_lineno]
        for node in ast.parse(data).body
        if isinstance(node, ast.ClassDef)
        for x in node.body
        if isinstance(x, ast.FunctionDef)
    }
    binding = {
        "path": source_path,
        "sha256": retained["sha256"],
        "methods": methods,
        "witness": {
            "ncdhw": [1, 8, 5, 5, 5],
            "ktrs": [16, 3, 3, 3],
            "pads": [1, 1, 1],
            "stride": [1, 1, 1],
            "dilation": [1, 1, 1],
            "mma_mn": [128, 128],
            "clusters_preferred_fallback": [[1, 1], [1, 1]],
            "input_output_dtype": "Float16",
            "acc_dtype": "Float32",
        },
        "status": "Dense entry source-extracted; blockscaled companion remains active.",
    }
    result = {
        "outcome": "PASS",
        "descriptor_guard_domains": guard_cases,
        "field_guard_requires_separate_positive_stride_premise": True,
        "spatial_geometries": geometries,
        "patch_elements": patch_elements,
        "output_values": output_values,
        "wrong_leading_padding": wrong_pad,
        "wrong_tap_order": wrong_tap_order,
        "bias_vector_domains": bias_domains,
        "boundary": (
            "Retained pure predicates and scalar coordinate/arithmetic equatio"
            "ns; no compiler or GPU run."
        ),
    }
    (STUDY / "inputs/conv-dense-binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    (STUDY / "checks/conv-contracts-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: implicit patch contraction, asymmetric padding, descriptor "
        "guards and bias vectors.\n"
    )


if __name__ == "__main__":
    main()
