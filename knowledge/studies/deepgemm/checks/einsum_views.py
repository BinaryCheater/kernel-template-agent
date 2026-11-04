"""Check rank-three view addresses and source einsum canonicalizations."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def address(coord: tuple[int, ...], strides: tuple[int, ...]) -> int:
    return sum(i * stride for i, stride in zip(coord, strides, strict=True))


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    source = (root / "csrc/apis/einsum.hpp").read_text()
    for marker in (
        "a.first.permute({1, 0, 2})",
        "b.first.permute({0, 2, 1})",
        "a.first.permute({1, 2, 0})",
        "d_tensor.stride(0) == n and n % 128 == 0",
    ):
        assert marker in source
    shape, strides = (2, 3, 5), (43, 11, 2)
    count = 0
    for axes in itertools.permutations(range(3)):
        out_shape = tuple(shape[axis] for axis in axes)
        out_strides = tuple(strides[axis] for axis in axes)
        seen = set()
        for output in itertools.product(*(range(size) for size in out_shape)):
            original = [0, 0, 0]
            for i, axis in enumerate(axes):
                original[axis] = output[i]
            original_coord = tuple(original)
            assert address(output, out_strides) == address(original_coord, strides)
            seen.add(original_coord)
            count += 1
        assert len(seen) == 30
    # Non-equal extents and coordinate-specific values distinguish batch, output and reduction axes.
    bsize, heads, dim, rank = 2, 3, 4, 5
    a_bhr = {
        (b, h, r): 100 * b + 10 * h + r + 1
        for b, h, r in itertools.product(range(bsize), range(heads), range(rank))
    }
    w_hdr = {
        (h, d, r): 100 * h + 10 * d + r + 2
        for h, d, r in itertools.product(range(heads), range(dim), range(rank))
    }
    a_bhd = {
        (b, h, d): 100 * b + 10 * h + d + 3
        for b, h, d in itertools.product(range(bsize), range(heads), range(dim))
    }
    recovered = []
    for expr in ("bhr,hdr->bhd", "bhd,hdr->bhr", "bhd,bhr->hdr"):
        if expr == "bhr,hdr->bhd":
            original = {
                (b, h, d): sum(a_bhr[b, h, r] * w_hdr[h, d, r] for r in range(rank))
                for b, h, d in itertools.product(range(bsize), range(heads), range(dim))
            }
            perm_a = {(h, b, r): value for (b, h, r), value in a_bhr.items()}
            canonical = {
                (b, h, d): sum(perm_a[h, b, k] * w_hdr[h, d, k] for k in range(rank))
                for h, b, d in itertools.product(range(heads), range(bsize), range(dim))
            }
        elif expr == "bhd,hdr->bhr":
            original = {
                (b, h, r): sum(a_bhd[b, h, d] * w_hdr[h, d, r] for d in range(dim))
                for b, h, r in itertools.product(range(bsize), range(heads), range(rank))
            }
            perm_a = {(h, b, d): value for (b, h, d), value in a_bhd.items()}
            perm_b = {(h, r, d): value for (h, d, r), value in w_hdr.items()}
            canonical = {
                (b, h, r): sum(perm_a[h, b, k] * perm_b[h, r, k] for k in range(dim))
                for h, b, r in itertools.product(range(heads), range(bsize), range(rank))
            }
        else:
            original = {
                (h, d, r): sum(a_bhd[b, h, d] * a_bhr[b, h, r] for b in range(bsize))
                for h, d, r in itertools.product(range(heads), range(dim), range(rank))
            }
            perm_a = {(h, d, b): value for (b, h, d), value in a_bhd.items()}
            perm_b = {(h, r, b): value for (b, h, r), value in a_bhr.items()}
            canonical = {
                (h, d, r): sum(perm_a[h, d, k] * perm_b[h, r, k] for k in range(bsize))
                for h, d, r in itertools.product(range(heads), range(dim), range(rank))
            }
        assert original == canonical
        recovered.append({"expression": expr, "outputs": len(original)})
    # Packed SF [H,B,P] has strides [pitch*P,1,pitch]. Inverse permutation is a view.
    pitch, packed_k = 4, 2
    sf_post_strides = (pitch * packed_k, 1, pitch)
    sf_pre_strides = (sf_post_strides[1], sf_post_strides[0], sf_post_strides[2])
    assert tuple(sf_pre_strides[i] for i in (1, 0, 2)) == sf_post_strides
    contiguous_pre = (heads * packed_k, packed_k, 1)
    wrong_post = tuple(contiguous_pre[i] for i in (1, 0, 2))
    assert wrong_post[1] != 1
    assert address((1, 1, 1), wrong_post) != address((1, 1, 1), sf_post_strides)
    # Flattened batch/K assignment for bmk,bnk->mn; a short last split crosses a batch.
    batches, ksize, tile_k, split = 3, 192, 64, 4
    units = batches * ksize // tile_k
    actual = []
    for group in range((units + split - 1) // split):
        for local in range(min(split, units - group * split)):
            sk = (group * split + local) * tile_k
            actual.extend((sk // ksize, sk % ksize + k) for k in range(tile_k))
    assert len(actual) == batches * ksize
    assert set(actual) == set(itertools.product(range(batches), range(ksize)))
    result = {
        "outcome": "PASS",
        "view_address_checks": count,
        "canonical_contractions": recovered,
        "packed_sf_reshape_rejected": True,
        "split_batch_k_witness": {
            "batches": batches,
            "K": ksize,
            "block_k": tile_k,
            "split_factor": split,
            "units": units,
            "last_split_units": units % split,
        },
        "boundary": "Exact scalar view/axis and split-index arithmetic; no FP8/BF16 execution.",
    }
    (STUDY / "checks/einsum-views-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: 180 view addresses, 3 canonical contractions, packed-SF stride counterexample.\n"
    )


if __name__ == "__main__":
    main()
