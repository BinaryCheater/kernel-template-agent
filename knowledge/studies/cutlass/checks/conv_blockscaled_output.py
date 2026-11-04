"Replay blockscaled-convolution frontend, scale windows and intended low-float policy."

from __future__ import annotations

import ast
import contextlib
import copy
import hashlib
import io
import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from conv_contracts import load_dense_frontend
from dense_contracts import DenseSymbols

STUDY = Path(__file__).resolve().parents[1]
SOURCE = (
    "examples/python/CuTeDSL/cute/blackwell/kernel/conv/dense_blockscaled_implicit_gemm_fprop.py"
)


def e4m3_values() -> list[tuple[int, Fraction]]:
    result = []
    for code in range(127):
        exponent, mantissa = divmod(code, 8)
        value = (
            Fraction(mantissa, 512)
            if exponent == 0
            else Fraction(8 + mantissa, 8) * Fraction(2) ** (exponent - 7)
        )
        result.append((code, value))
    assert result[-1][1] == 448
    return result


E4M3 = e4m3_values()
E2M1 = list(enumerate(map(Fraction, (0, "0.5", 1, "1.5", 2, 3, 4, 6))))
E8M0 = [(code, Fraction(2) ** (code - 127)) for code in range(255)]
FP32_MAX = Fraction(2**24 - 1) * 2**104


def nearest(value: Fraction, table: list[tuple[int, Fraction]]) -> Fraction:
    sign = -1 if value < 0 else 1
    _, decoded = min(table, key=lambda item: (abs(item[1] - abs(value)), item[0] % 2))
    return sign * decoded


def quantize(values: list[Fraction], format_name: str) -> tuple[Fraction, list[Fraction]]:
    maximum, codes = (6, E2M1) if format_name == "nvfp4" else (448, E4M3)
    raw = max(map(abs, values)) / maximum
    if format_name == "nvfp4":
        scale = nearest(raw, E4M3)
    else:
        scale = next(value for _, value in E8M0 if value >= raw)
    multiplier = min(1 / scale, FP32_MAX) if scale else FP32_MAX
    normalized = [value * multiplier for value in values]
    assert all(abs(value) <= maximum for value in normalized)
    return scale, [nearest(value, codes) for value in normalized]


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    data = (root / SOURCE).read_bytes()
    entry = next(x for x in manifest["files"] if x["path"] == SOURCE)
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    source = data.decode()
    tree = ast.parse(source)
    ns, _, _, _, _ = load_dense_frontend(root)
    cls = copy.deepcopy(
        next(
            x for x in tree.body if isinstance(x, ast.ClassDef) and x.name.startswith("Sm100Block")
        )
    )
    cls.body = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == "can_implement"]
    funcs = [
        x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "sf_k_tile_channels"
    ]
    activations = copy.deepcopy(
        next(
            x
            for x in tree.body
            if isinstance(x, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "EPILOGUE_ACTIVATIONS" for t in x.targets)
        )
    )
    # Only device-lambda identities are needed by this pure membership predicate.
    for group in activations.value.values:
        group.values = [
            value
            for key, value in zip(group.keys, group.values, strict=True)
            if key.value == "device"
        ]
        group.keys = [ast.Constant("device")]
    selected = ast.Module(body=[*funcs, activations, cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(DenseSymbols().visit(selected)), SOURCE, "exec"), ns)
    kind = ns[cls.name]
    obj = kind.__new__(kind)
    obj.acc_dtype, obj.use_2cta_instrs, obj.mma_tiler_mn = "Float32", False, (128, 128)
    obj.cluster_shape_mn = obj.preferred_cluster_shape_mn = obj.fallback_cluster_shape_mn = (1, 1)
    obj.split_k, obj.use_tma_store, obj.swizzle_size, obj.raster_along = 1, True, 1, "m"
    obj.cta_tile_k, obj.sf_vec_size = 128, 16
    obj.epilogue_warp_id = (0, 1, 2, 3)

    def accepts(
        c=192,
        k=128,
        ab="Float4E2M1FN",
        out="Float16",
        sf="Float8E4M3FN",
        output_m=125,
        leading_dim=4,
        **kwargs,
    ):
        shape = (1, 5, 5, 5, k) if output_m == 125 else (1, 1, 1, output_m, k)
        output = SimpleNamespace(shape=shape, leading_dim=leading_dim)
        with contextlib.redirect_stdout(io.StringIO()):
            return obj.can_implement(
                c,
                k,
                ab,
                out,
                sf,
                output,
                (3, 3, 3),
                (1,) * 3,
                (1,) * 3,
                (1,) * 3,
                (1,) * 3,
                **kwargs,
            )

    assert accepts()
    assert not accepts(c=160)  # C must contain whole four-scale atoms, not just16-byte operands.
    positive_tiles = 0
    for vec, ab, sf, channels, tiles in (
        (16, "Float4E2M1FN", "Float8E4M3FN", 192, (64, 128, 256)),
        (32, "Float8E4M3FN", "Float8E8M0FNU", 384, (64, 128)),
    ):
        obj.sf_vec_size = vec
        for tile in tiles:
            obj.cta_tile_k = tile
            assert accepts(c=channels, ab=ab, sf=sf)
            assert accepts(c=channels, ab=ab, sf=sf, out=ab)
            positive_tiles += 1
        obj.cta_tile_k = 192
        assert not accepts(c=channels, ab=ab, sf=sf)
    obj.cta_tile_k, obj.sf_vec_size = 128, 16
    rejected = []
    for kwargs in (
        {"sf": "Float8E8M0FNU"},
        {"ab": "Float8E5M2"},
        {"out": "Float32"},
        {"out": "Float4E2M1FN", "has_bias": True},
        {"out": "Float4E2M1FN", "c_dtype": "Float4E2M1FN"},
    ):
        assert not accepts(**kwargs)
        rejected.append(kwargs)
    assert not accepts(out="Float4E2M1FN", epilogue_op=ns["EPILOGUE_ACTIVATIONS"]["relu"]["device"])

    def custom_nonidentity(x):
        return x + 1

    assert accepts(out="Float4E2M1FN", epilogue_op=custom_nonidentity)
    assert custom_nonidentity(0) != 0
    assert accepts(c_dtype="Float16", has_bias=True)
    assert not accepts(c_dtype="BFloat16")
    assert not accepts(leading_dim=3)
    obj.use_2cta_instrs, obj.mma_tiler_mn = True, (256, 128)
    obj.cluster_shape_mn = obj.preferred_cluster_shape_mn = obj.fallback_cluster_shape_mn = (2, 1)
    assert accepts(output_m=256)
    obj.mma_tiler_mn = (128, 128)
    assert not accepts(output_m=256)
    obj.use_2cta_instrs = False
    obj.cluster_shape_mn = obj.preferred_cluster_shape_mn = obj.fallback_cluster_shape_mn = (1, 1)
    obj.mma_tiler_mn = (64, 128)
    assert not accepts()  # parent allows M64, blockscaled atom does not
    obj.mma_tiler_mn = (128, 32)
    assert accepts(k=32)
    assert not accepts(k=64)  # unsupported small-N chunk offset cannot serve a second tile
    obj.mma_tiler_mn = (128, 128)
    obj.cluster_shape_mn = obj.fallback_cluster_shape_mn = obj.preferred_cluster_shape_mn = (16, 1)
    assert not accepts(k=128)  # exceeds staged SF groups along cluster M
    obj.cta_tile_k = 256
    assert accepts(c=256, output_m=4096)
    large_cluster_readers = 32 * (1 + 16 * 10)
    assert large_cluster_readers == 5152 and large_cluster_readers > 4096
    obj.cluster_shape_mn = obj.fallback_cluster_shape_mn = obj.preferred_cluster_shape_mn = (1, 1)

    # The source helper explicitly requests upward E8M0 and otherwise default nearest-even.
    conversion_path = "python/CuTeDSL/_mlir_helpers/arith.py"
    conversion_data = (root / conversion_path).read_bytes()
    retained_conversion = next(x for x in manifest["files"] if x["path"] == conversion_path)
    assert hashlib.sha256(conversion_data).hexdigest() == retained_conversion["sha256"]
    conversion = conversion_data.decode()
    assert "arith.RoundingMode.upward if res_elem_type == T.f8E8M0FNU() else None" in conversion
    assert source.index("rSFD[(0, i_sf)] = pvscale_f32.to(self.sfd_dtype)") < source.index(
        "acc_scale = norm_const * cute.arch.rcp_approx(qpvscale_f32)"
    )
    quantizer_cases = 0
    for fmt, size, maximum in (("nvfp4", 16, 6), ("mxfp8", 32, 448)):
        for scale in map(Fraction, ("0.125", "0.5", 1, 2, 16)):
            for signs in ((1, 1), (-1, 1), (1, -1)):
                values = [maximum * scale * signs[0], scale * signs[1]] + [Fraction(0)] * (size - 2)
                encoded_scale, codes = quantize(values, fmt)
                assert encoded_scale == scale
                assert [q * encoded_scale for q in codes] == values
                quantizer_cases += 1
        encoded_scale, codes = quantize([Fraction(0)] * size, fmt)
        assert encoded_scale == (0 if fmt == "nvfp4" else Fraction(2) ** -127)
        assert codes == [0] * size
        quantizer_cases += 1
    # Nearest-even ties use encoded significand parity, not distance alone.
    for table in (E4M3, E2M1):
        for (left_code, left), (_, right) in itertools.pairwise(table):
            assert nearest((left + right) / 2, table) == (left if left_code % 2 == 0 else right)
    values = [Fraction("7.4"), Fraction("2.17")] + [Fraction(0)] * 14
    stored_scale, codes = quantize(values, "nvfp4")
    raw_scale = max(values) / 6
    wrong = [nearest(v / raw_scale, E2M1) for v in values]
    assert stored_scale == Fraction("1.25") and codes[1] == Fraction("1.5") and wrong[1] == 2
    quantizer_cases += 1
    # Source has a wider overflow/underflow regime than this finite-code candidate scope.
    raw_seven_scale = Fraction(7, 6)
    assert Fraction(7) / nearest(raw_seven_scale, E4M3) > 6

    # Same overlapping N metadata window as the prior blockscaled GEMM instance.
    assignments = {
        n.targets[0].id: n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
    }
    assert ast.unparse(assignments["x_times_3"]) == "3 * x"
    assert ast.unparse(assignments["new_stride"].elts[0].elts[1]) == "((x, x), x_times_3)"
    offsets = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "offset" for t in n.targets)
    ]
    assert any(
        isinstance(x, ast.IfExp)
        and ast.unparse(x.body) == "cutlass.Int32(2)"
        and ast.unparse(x.orelse) == "cutlass.Int32(0)"
        for x in offsets
    )
    assert any(ast.unparse(x) == "cutlass.Int32(mma_tile_coord_mnl[1] % 2 * 2)" for x in offsets)
    windows = 0
    for width, tile in itertools.product((64, 128, 192, 256), range(17)):
        if width == 64:
            base, offset = 128 * (tile // 2), 64 * (tile % 2)
        elif width == 192:
            outer = 2 * tile
            rem, rest = outer % 4, outer // 4
            base, offset = 128 * (rem % 2 + rem // 2 + 3 * rest), 64 * (tile % 2)
        else:
            base, offset = tile * width, 0
        for column in range(width):
            assert base + offset + column == tile * width + column
            windows += 1
    # Evaluate the retained producer/consumer reverse-subtile expressions separately.
    reversals = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "reverse_subtile" for t in n.targets)
    ]
    producer_reverse = next(x for x in reversals if "c_acc_shadow_state" in ast.unparse(x))
    consumer_reverse = next(x for x in reversals if ast.unparse(x) == "acc_stage_index == 0")
    real_indices = [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "real_subtile_idx" for t in n.targets)
        and isinstance(n.value, ast.BinOp)
    ]
    assert len(real_indices) == 2

    def evaluate(expr, values):
        return eval(compile(ast.Expression(body=expr), SOURCE, "eval"), values)

    residual_orders = 0
    for tiles, subtiles in itertools.product(range(1, 6), (1, 2, 4, 8)):
        obj.cta_tile_shape_mnk, obj.epi_tile_n = (128, subtiles * 32, 128), 32
        producer_order, consumer_order = [], []
        for tile in range(tiles):
            phase = tile % 2
            env = {
                "self": obj,
                "c_acc_shadow_state": SimpleNamespace(phase=phase),
                "acc_stage_index": phase,
            }
            for iteration in range(subtiles):
                env["subtile_idx"] = iteration
                producer_index = (
                    evaluate(real_indices[0], env) if evaluate(producer_reverse, env) else iteration
                )
                consumer_index = (
                    evaluate(real_indices[1], env) if evaluate(consumer_reverse, env) else iteration
                )
                producer_order.append((tile, producer_index))
                consumer_order.append((tile, consumer_index))
        expected = [
            (tile, column)
            for tile in range(tiles)
            for column in (reversed(range(subtiles)) if tile % 2 == 0 else range(subtiles))
        ]
        assert producer_order == consumer_order == expected
        assert len(set(producer_order)) == tiles * subtiles
        residual_orders += 1
    assert list(reversed(range(4))) != list(range(4))
    result = {
        "outcome": "PASS",
        "positive_format_tile_bindings": positive_tiles,
        "two_cta_frontend_witness": True,
        "rejected_frontend_neighbors": rejected,
        "custom_nonidentity_activation_accepted": True,
        "source_cluster16_reader_count": large_cluster_readers,
        "outside_clc_candidate_participant_domain": True,
        "quantizer_scalar_cases": quantizer_cases,
        "decoded_scale_counterexample": {
            "raw_scale": str(raw_scale),
            "stored_scale": str(stored_scale),
            "correct_second_code": str(codes[1]),
            "wrong_second_code": str(wrong[1]),
        },
        "n_window_coordinates": windows,
        "residual_phase_order_domains": residual_orders,
        "boundary": (
            "Retained pure predicates and rational intended-policy/window chec"
            "ks; Compiler rounding, approximate reciprocal, group lane map and"
            " extreme ranges remain open."
        ),
    }
    (STUDY / "checks/conv-blockscaled-output-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: blockscaled convolution frontend, encoded output scales and SFB N windows.\n"
    )


if __name__ == "__main__":
    main()
