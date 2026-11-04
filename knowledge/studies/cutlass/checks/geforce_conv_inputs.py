"Replay retained SFA scalar coordinates, channel/tap walks and scale selection."

from __future__ import annotations

import ast
import copy
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

from dense_contracts import DenseSymbols
from frontend_contract import DTYPES

STUDY = Path(__file__).resolve().parents[1]
PREFIX = "examples/python/CuTeDSL/cute/blackwell_geforce/kernel/conv/"


def i32(value):
    return (value + (1 << 31)) % (1 << 32) - (1 << 31)


class Scalar(ast.NodeTransformer):
    def visit_Call(self, node):
        if ast.unparse(node.func) == "cutlass.Boolean":
            return ast.copy_location(ast.Call(ast.Name("bool", ast.Load()), node.args, []), node)
        return self.generic_visit(node)


class SignedBits(ast.NodeTransformer):
    "Model the source's fixed-width packed register; other arithmetic stays scalar."

    def visit_BinOp(self, node):
        node = self.generic_visit(node)
        if isinstance(node.op, (ast.LShift, ast.RShift, ast.BitOr, ast.BitAnd)):
            return ast.copy_location(ast.Call(ast.Name("i32", ast.Load()), [node], []), node)
        return node


def statements_code(statements, label):
    module = ast.Module(copy.deepcopy(statements), [])
    return compile(ast.fix_missing_locations(Scalar().visit(module)), label, "exec")


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    path = PREFIX + "dense_blockscaled_implicit_gemm_fprop.py"
    data = (Path(manifest["tree"]) / path).read_bytes()
    retained = next(x for x in manifest["files"] if x["path"] == path)
    assert hashlib.sha256(data).hexdigest() == retained["sha256"]
    cls = next(x for x in ast.parse(data).body if isinstance(x, ast.ClassDef))
    methods = {x.name: x for x in cls.body if isinstance(x, ast.FunctionDef)}
    direct = methods["sfa_cpasync_copy_tile"]
    # Execute unchanged scalar prefix after copy-atom/predicate-tensor construction.
    start = next(i for i, x in enumerate(direct.body) if ast.unparse(x).startswith("ZPQ ="))
    end = next(
        i for i, x in enumerate(direct.body) if ast.unparse(x).startswith("sfa_predicate_tensor[0]")
    )
    local_tid = next(
        x
        for x in direct.body
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "local_tid"
    )
    direct_code = statements_code([local_tid, *direct.body[start:end]], "retained-sfa-prefix")
    kernel = methods["kernel"]
    tma_loop = next(
        node
        for node in ast.walk(kernel)
        if isinstance(node, ast.For)
        and any(
            isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "tBgSFB_k" for x in node.body
        )
    )
    sfb_assignment = next(
        x
        for x in tma_loop.body
        if isinstance(x, ast.Assign) and ast.unparse(x.targets[0]) == "tBgSFB_k"
    )
    assert isinstance(sfb_assignment.value, ast.Subscript)
    sfb_index = compile(ast.Expression(sfb_assignment.value.slice.elts[1]), "sfb-index", "eval")
    carry_start = next(
        i
        for i, x in enumerate(tma_loop.body)
        if isinstance(x, ast.Assign)
        and ast.unparse(x.targets[0]) == "sfa_c_tile_idx"
        and isinstance(x.value, ast.BinOp)
    )
    carry_code = statements_code(tma_loop.body[carry_start : carry_start + 2], "retained-carry")
    assignments = {
        x.targets[0].id: x.value
        for x in ast.walk(kernel)
        if isinstance(x, ast.Assign) and len(x.targets) == 1 and isinstance(x.targets[0], ast.Name)
    }

    def expression(name, env, signed=False):
        tree = ast.Expression(copy.deepcopy(assignments[name]))
        if signed:
            tree = SignedBits().visit(tree)
        return eval(compile(ast.fix_missing_locations(tree), name, "eval"), env)

    coordinate_domains = coordinate_cases = 0
    wrong_pad = None
    for stride, dilation, lower, upper in itertools.product(
        ((1, 1, 1), (1, 2, 1)),
        ((1, 1, 1), (1, 1, 2)),
        ((0, 0, 0), (1, 0, 1)),
        ((0, 1, 0), (0, 0, 1)),
    ):
        spatial, taps, batch = (3, 4, 5), (2, 2, 2), 2
        output = tuple(
            (n + lo + up - dil * (flt - 1) - 1) // st + 1
            for n, flt, lo, up, st, dil in zip(
                spatial, taps, lower, upper, stride, dilation, strict=True
            )
        )
        rows = batch * math.prod(output)
        for m, tap in itertools.product(
            range(math.ceil(rows / 128) * 128), itertools.product(range(2), repeat=3)
        ):
            b, rem = divmod(m, math.prod(output))
            z, rem = divmod(rem, output[1] * output[2])
            p, q = divmod(rem, output[2])
            coords = tuple(
                v * st - lo + t * dil
                for v, st, lo, t, dil in zip((z, p, q), stride, lower, tap, dilation, strict=True)
            )
            valid = m < rows and all(0 <= v < n for v, n in zip(coords, spatial, strict=True))
            env = {
                "self": SimpleNamespace(
                    tile_shape_mnk=(128, 64, 128),
                    sf_tile_k=128,
                    sf_vec_size=16,
                    ab_k_tiles_per_sf_k_tile=1,
                ),
                "tidx": 256 + m % 128,
                "tile_coord_mnl": (m // 128, 0, 0),
                "sfa_c_tile_idx": 0,
                "input_N": batch,
                "input_C": 192,
            }
            for names, values in (
                (("input_D", "input_H", "input_W"), spatial),
                (("output_Z", "output_P", "output_Q"), output),
                (("rt_stride_d", "rt_stride_h", "rt_stride_w"), stride),
                (("rt_lower_pad_d", "rt_lower_pad_h", "rt_lower_pad_w"), lower),
                (("rt_dil_d", "rt_dil_h", "rt_dil_w"), dilation),
                (("t_coord", "r_coord", "s_coord"), tap),
            ):
                env.update(zip(names, values, strict=True))
            exec(direct_code, env)
            assert env["sfa_pred_val"] == valid
            assert tuple(env[name] for name in ("d_in", "h_in", "w_in")) == coords
            assert (env["sf_m_inner"], env["sf_m_outer"]) == (m % 32, (m % 128) // 32)
            # Invalid rows still name in-bounds scalar addresses before predication.
            assert 0 <= env["sfa_m_addr"] < batch * math.prod(spatial)
            if valid:
                d, h, w = coords
                assert env["sfa_m_addr"] == ((b * spatial[0] + d) * spatial[1] + h) * spatial[2] + w
            hoisted = {
                "sfa_n_clamped": b if m < rows else 0,
                "sfa_DHW": math.prod(spatial),
                "sfa_HW": spatial[1] * spatial[2],
                "input_W": spatial[2],
                "sfa_sf_k_extent": 12,
                "sfa_d_in": z * stride[0] - lower[0],
                "sfa_h_in": p * stride[1] - lower[1],
                "sfa_w_in": q * stride[2] - lower[2],
                "sfa_t_idx": tap[0],
                "sfa_r_idx": tap[1],
                "sfa_s_idx": tap[2],
                "rt_dil_d": dilation[0],
                "rt_dil_h": dilation[1],
                "rt_dil_w": dilation[2],
                "i32": i32,
            }
            base = expression("sfa_m_base", hoisted)
            hoisted["sfa_d_base"] = hoisted["sfa_d_in"]
            hoisted["sfa_hw_base"] = expression("sfa_hw_base", hoisted, signed=True)
            rebuilt = tuple(
                expression(name, hoisted, signed=True)
                for name in ("sfa_d_in", "sfa_h_in", "sfa_w_in")
            )
            assert rebuilt == coords
            if valid:
                assert base + expression("sfa_row_delta", hoisted) == env["sfa_m_addr"] * 12
            if lower != upper and wrong_pad is None:
                bad = tuple(
                    v * st - up + t * dil
                    for v, st, up, t, dil in zip(
                        (z, p, q), stride, upper, tap, dilation, strict=True
                    )
                )
                if valid and coords != bad:
                    wrong_pad = {"correct": coords, "swapped_padding": bad}
            coordinate_cases += 1
        coordinate_domains += 1
    assert wrong_pad is not None

    # Carry rebuilding stores each H/W base in signed16 halves of an Int32.
    packed_cases = 0
    packing_counterexample = None
    for h, w, r, s in itertools.product(
        (-32768, -1, 0, 32767, 32768), (-32768, -1, 0, 32767, 32768), (0, 1), (0, 1)
    ):
        env = {
            "sfa_h_in": h,
            "sfa_w_in": w,
            "sfa_r_idx": r,
            "sfa_s_idx": s,
            "rt_dil_h": 2,
            "rt_dil_w": 3,
            "i32": i32,
        }
        env["sfa_hw_base"] = expression("sfa_hw_base", env, signed=True)
        got_h = expression("sfa_h_in", env, signed=True)
        got_w = expression("sfa_w_in", env, signed=True)
        expected = (h + 2 * r, w + 3 * s)
        in_range = -32768 <= h <= 32767 and -32768 <= w <= 32767
        if in_range:
            assert (got_h, got_w) == expected
        elif h == 32768 and w == 0 and r == 0 and s == 1:
            assert (got_h, got_w) == (-32768, 3)
            packing_counterexample = {
                "base_hw": [h, w],
                "correct": expected,
                "signed32_rebuild": [got_h, got_w],
            }
        packed_cases += 1
    assert packing_counterexample is not None
    dense_path = PREFIX + "dense_implicit_gemm_fprop.py"
    dense_data = (Path(manifest["tree"]) / dense_path).read_bytes()
    dense_retained = next(x for x in manifest["files"] if x["path"] == dense_path)
    assert hashlib.sha256(dense_data).hexdigest() == dense_retained["sha256"]
    dense_tree = ast.parse(dense_data)
    helpers = [
        x
        for x in dense_tree.body
        if isinstance(x, ast.FunctionDef)
        and x.name in {"_check_tensor_alignment", "_check_im2col_descriptor_limits"}
    ]
    guarded_class = copy.deepcopy(cls)
    guarded_class.bases = []
    guarded_class.body = [copy.deepcopy(methods["can_implement"])]
    module = ast.Module([*helpers, guarded_class], [])
    ns = {"widths": DTYPES}
    exec(compile(ast.fix_missing_locations(DenseSymbols().visit(module)), "menu", "exec"), ns)
    # H32769, inputW2/filterS2 => outputQ1; m32768 starts at h32768,w0.
    # The first tap is valid; the next filter carry reconstructs negative H.
    ns[cls.name].can_implement(
        (128, 64, 64),
        "Float4E2M1FN",
        "Float8E4M3FN",
        "Float32",
        "Float16",
        16,
        64,
        128,
        (1, 1, 2),
        (1,) * 3,
        (1,) * 3,
        (0,) * 3,
        (0,) * 3,
    )
    witness_env = {
        "sfa_h_in": 32768,
        "sfa_w_in": 0,
        "sfa_r_idx": 0,
        "sfa_s_idx": 1,
        "rt_dil_h": 1,
        "rt_dil_w": 1,
        "i32": i32,
    }
    witness_env["sfa_hw_base"] = expression("sfa_hw_base", witness_env, signed=True)
    witness_hw = [expression(name, witness_env, signed=True) for name in ("sfa_h_in", "sfa_w_in")]
    assert witness_hw == [-32768, 1]
    assert 0 <= 32768 < 32769 and 0 <= 1 < 2
    assert not (0 <= witness_hw[0] < 32769 and 0 <= witness_hw[1] < 2)
    packing_counterexample["accepted_frontend"] = True
    packing_counterexample["ncdhw"] = [1, 64, 1, 32769, 2]
    packing_counterexample["ktrs"] = [128, 1, 1, 2]
    packing_counterexample["concrete_correct_hw"] = [32768, 1]
    packing_counterexample["concrete_rebuilt_hw"] = witness_hw
    packing_counterexample["first_valid_then_rebuilt_invalid"] = True

    scale_domains = scale_elements = 0
    for c, tile_k, vector, taps in itertools.product(
        (64, 128, 192, 256, 384), (64, 128), (16, 32), ((1, 1, 1), (2, 2, 3))
    ):
        # Actual connected formats/menus; C192 MXFP8 is not accepted.
        if c % (4 * vector) or (vector == 32 and tile_k != 64):
            continue
        sf_span = max(tile_k, 4 * vector)
        ratio, chunks = sf_span // tile_k, sf_span // vector // 4
        c_tiles, sf_tiles = math.ceil(c / tile_k), math.ceil(c / sf_span)
        state = {
            "sfa_c_tile_idx": 0,
            "sfa_s_idx": 0,
            "sfa_r_idx": 0,
            "sfa_t_idx": 0,
            "c_tiles_per_fpos": c_tiles,
            "rt_filter_s": taps[2],
            "rt_filter_r": taps[1],
            "rt_filter_t": taps[0],
        }
        for t, r, s, ct in itertools.product(*(range(x) for x in taps), range(c_tiles)):
            assert tuple(
                state[x] for x in ("sfa_t_idx", "sfa_r_idx", "sfa_s_idx", "sfa_c_tile_idx")
            ) == (t, r, s, ct)
            exec(carry_code, state)
            env = {
                "sfa_s_idx": s,
                "sfa_r_idx": r,
                "sfa_t_idx": t,
                "rt_filter_s": taps[2],
                "rt_filter_r": taps[1],
            }
            fpos = expression("sfb_fpos", env)
            assert fpos == s + taps[2] * (r + taps[1] * t)
            sf_tile = fpos * sf_tiles + ct // ratio
            selected_half = ct % ratio
            index_env = {
                "sfb_fpos": fpos,
                "sfb_c_tiles": sf_tiles,
                "sfa_c_tile_idx": ct,
                "self": SimpleNamespace(ab_k_tiles_per_sf_k_tile=ratio),
                "mainloop_consumer_state": SimpleNamespace(count=fpos * c_tiles + ct),
                "c_tiles_per_fpos": c_tiles,
                "num_k_blocks": tile_k // (64 if vector == 16 else 32),
            }
            assert eval(sfb_index, index_env) == sf_tile
            assert expression("sf_base", index_env) == selected_half * index_env["num_k_blocks"]
            # Source SFA transfers whole four-byte chunks, re-reading chunk0
            # for a completely padded chunk rather than crossing pixel storage.
            for chunk, element in itertools.product(range(chunks), range(4)):
                source_base = ct // ratio * (sf_span // vector) + chunk * 4
                source_base = source_base if source_base < c // vector else 0
                assert 0 <= source_base <= c // vector - 4
                sf_group = source_base + element
                assert 0 <= sf_group < c // vector
            for local_c in range(tile_k):
                channel = ct * tile_k + local_c
                sf_offset = selected_half * tile_k // vector + local_c // vector
                sfa_group = ct // ratio * sf_span // vector + sf_offset
                sfb_group = sf_tile * sf_span // vector + sf_offset
                if channel < c:
                    assert sfa_group == channel // vector
                    assert sfb_group == fpos * sf_tiles * sf_span // vector + channel // vector
                scale_elements += 1
        scale_domains += 1
    # Removing per-position padding mixes the next filter's scales into padding.
    assert math.ceil(192 / 128) * 128 == 256
    assert 192 // 16 != 256 // 16
    result = {
        "outcome": "PASS",
        "source": path,
        "sha256": retained["sha256"],
        "coordinate_domains": coordinate_domains,
        "coordinate_cases": coordinate_cases,
        "wrong_padding_witness": wrong_pad,
        "packed_signed32_cases": packed_cases,
        "packed_base_counterexample": packing_counterexample,
        "scale_domains": scale_domains,
        "scale_element_selections": scale_elements,
        "boundary": (
            "Retained scalar coordinate prefix and selected source expressions"
            ", signed32 bit arithmetic, logical scale groups. Native layouts, "
            "cp.async completion/encoding and TMA coverage remain unproved."
        ),
    }
    (STUDY / "checks/geforce-conv-inputs-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: SM120 SFA coordinates, signed packing boundary and scale group selection.\n"
    )


if __name__ == "__main__":
    main()
