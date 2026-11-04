"Source-selected cuBLASLt descriptor equations; no library/device invocation."

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def operand(rows: int, cols: int, kmajor: bool, batches: int) -> tuple[dict[int, int], int, int]:
    ld = (cols if kmajor else rows) + 3
    span = ld * (rows if kmajor else cols) + 11
    data = {}
    for batch, row, col in itertools.product(range(batches), range(rows), range(cols)):
        index = batch * span + (row * ld + col if kmajor else row + col * ld)
        data[index] = 100 * batch + 7 * row + col + 1
    return data, ld, span


def check_dense() -> dict[str, int]:
    domains, outputs = 0, 0
    for m, n, k, a_major, b_major, batches in itertools.product(
        (1, 3), (2, 5), (1, 7), (False, True), (False, True), (1, 3)
    ):
        lhs, lda, astride = operand(m, k, a_major, batches)
        rhs, ldb, bstride = operand(n, k, b_major, batches)
        for batch, row, col in itertools.product(range(batches), range(m), range(n)):
            # Wrapper swaps operands to compute D.T as a column-major matrix.
            result = 0
            for reduce in range(k):
                # cuBLAS A is rhs: transposed for K-major, plain otherwise.
                ax, ay = (reduce, col) if b_major else (col, reduce)
                # cuBLAS B is lhs: plain for K-major, transposed otherwise.
                bx, by = (reduce, row) if a_major else (row, reduce)
                result += (
                    rhs[batch * bstride + ax + ay * ldb] * lhs[batch * astride + bx + by * lda]
                )
            oracle = sum(
                (100 * batch + 7 * row + r + 1) * (100 * batch + 7 * col + r + 1) for r in range(k)
            )
            assert result == oracle
            outputs += 1
        domains += 1
    return {"major_batch_domains": domains, "integer_outputs": outputs}


def check_einsums() -> dict[str, int]:
    b, h, r, d = 2, 3, 5, 7
    checks = 0
    # Three source descriptors, expressed in their (row,col,batch) column-major offsets.
    for token, head, out in itertools.product(range(b), range(h), range(d)):
        for red in range(r):
            rhs_offset = red + out * r + head * d * r
            lhs_offset = red + token * h * r + head * r
            dst_offset = out + token * h * d + head * d
            assert rhs_offset == (head * d + out) * r + red
            assert lhs_offset == (token * h + head) * r + red
            assert dst_offset == (token * h + head) * d + out
            checks += 1
    for token, head, out in itertools.product(range(b), range(h), range(r)):
        for red in range(d):
            rhs_offset = out + red * r + head * d * r
            lhs_offset = red + token * h * d + head * d
            assert rhs_offset == (head * d + red) * r + out
            assert lhs_offset == (token * h + head) * d + red
            checks += 1
    for head, row, col in itertools.product(range(h), range(d), range(r)):
        for red in range(b):
            rhs_offset = col + red * h * r + head * r
            lhs_offset = row + red * h * d + head * d
            dst_offset = col + row * r + head * d * r
            assert rhs_offset == (red * h + head) * r + col
            assert lhs_offset == (red * h + head) * d + row
            assert dst_offset == (head * d + row) * r + col
            checks += 1
    return {"asymmetric_descriptor_coordinates": checks}


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    body = (root / "csrc/jit_kernels/impls/smxx_cublaslt.hpp").read_text()
    for marker in (
        "1.0f, rhs_sf.data_ptr(), lhs_sf.data_ptr()",
        "const float beta = accumulate ? 1.0f : 0.0f;",
        "CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3",
    ):
        assert marker in body
    api = (root / "csrc/apis/gemm.hpp").read_text()
    assert "d.copy_(c.value())" in api
    sf_sizes = [(mn + 127) // 128 * ((96 + 63) // 64) * 512 for mn in (129, 257)]
    assert sf_sizes == [2048, 3072]
    result = {
        "outcome": "PASS",
        "dense_batched": check_dense(),
        "einsums": check_einsums(),
        "nvfp4_operand_sf_bytes": sf_sizes,
        "boundary": (
            "Integer address/contraction equations only; no cuBLAS algorithm, "
            "scale decode or floating-point execution."
        ),
    }
    (STUDY / "checks/cublas-layouts-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write("PASS: cuBLAS operand swaps, padded batch strides and einsum descriptors.\n")


if __name__ == "__main__":
    main()
