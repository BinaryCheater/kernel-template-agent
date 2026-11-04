"""Retained Marlin packing/decode equations; no torch/CUDA execution."""

from __future__ import annotations

import hashlib
import itertools
import json
import struct
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def half(value: float) -> float:
    return struct.unpack("<e", struct.pack("<e", value))[0]


def from_bits(value: int) -> float:
    return struct.unpack("<e", struct.pack("<H", value & 0xFFFF))[0]


def decode(word: int) -> list[float]:
    lo = (word & 0x000F000F) | 0x64006400
    hi = (word & 0x00F000F0) | 0x64006400
    return [
        half(from_bits(lo) - from_bits(0x6408)),
        half(from_bits(lo >> 16) - from_bits(0x6408)),
        half(from_bits(hi) * from_bits(0x2C00) + from_bits(0xD480)),
        half(from_bits(hi >> 16) * from_bits(0x2C00) + from_bits(0xD480)),
    ]


def permutation() -> list[int]:
    result = []
    for lane in range(32):
        rows = [2 * (lane % 4), 2 * (lane % 4) + 1, 2 * (lane % 4 + 4), 2 * (lane % 4 + 4) + 1]
        local = [16 * row + lane // 4 + 8 * block for block in (0, 1) for row in rows]
        for tile in range(4):
            result.extend(local[i] + 256 * tile for i in (0, 2, 4, 6, 1, 3, 5, 7))
    return result


def check_prepack() -> dict[str, int]:
    perm = permutation()
    assert sorted(perm) == list(range(1024))
    checked = 0
    for k, n in ((128, 256), (256, 256), (256, 512)):
        # Unique original coordinates before bit packing avoid collisions between four-bit tags.
        tiled = [
            (kb * 16 + kr, nb * 16 + nc)
            for kb in range(k // 16)
            for nb in range(n // 16)
            for kr in range(16)
            for nc in range(16)
        ]
        packed = [tiled[base + index] for base in range(0, k * n, 1024) for index in perm]
        inverse = {index: pos for pos, index in enumerate(perm)}
        for kr, nc in itertools.product(range(k), range(n)):
            tiled_offset = (kr // 16) * (n * 16) + (nc // 16) * 256 + (kr % 16) * 16 + nc % 16
            base, local = divmod(tiled_offset, 1024)
            assert packed[base * 1024 + inverse[local]] == (kr, nc)
            checked += 1
    grouped_sf = [i + 8 * j for i in range(8) for j in range(8)]
    channel_sf = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    assert sorted(grouped_sf) == list(range(64))
    assert sorted(channel_sf) == list(range(32))
    assert grouped_sf[:32] != channel_sf
    return {
        "weight_coordinates": checked,
        "permutation_entries": len(perm),
        "grouped_scale_entries": 64,
        "channel_scale_entries": 32,
    }


def check_decode() -> dict[str, int]:
    cases = 0
    for q0, q1, q4, q5 in itertools.product(range(16), repeat=4):
        word = q0 | q1 << 4 | q4 << 16 | q5 << 20
        assert decode(word) == [q0 - 8, q4 - 8, q1 - 8, q5 - 8]
        cases += 1
    # Source eight-entry interleave and two decode calls restore original slot order.
    interleave = (0, 2, 4, 6, 1, 3, 5, 7)
    word = sum(interleave[i] << (4 * i) for i in range(8))
    assert decode(word) + decode(word >> 8) == [float(i - 8) for i in range(8)]
    # Per-channel scale happens after FP16 rounding, not one final FP32 scale-and-round.
    value, scale = 1.00048828125, 3.0
    assert half(half(value) * scale) != half(value * scale)
    return {"selected_nibble_combinations": cases, "interleave_decode_identity": 8}


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    actual = {}
    for entry in manifest["files"]:
        data = (root / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        actual[entry["path"]] = hashlib.sha1(
            b"blob " + str(len(data)).encode() + b"\0" + data
        ).hexdigest()
    expected = {
        line.split("\t", 1)[1]: line.split()[2]
        for line in (STUDY / "inputs/git-tree.txt").read_text().splitlines()
    }
    assert actual == expected
    assert (
        hashlib.sha256(Path(manifest["archive"]).read_bytes()).hexdigest()
        == manifest["archive_sha256"]
    )
    device = (root / "marlin/marlin_cuda_kernel.cu").read_text()
    for marker in ("0x000f000f", "0x00f000f0", "0x64086408", "0x2c002c00", "0xd480d480"):
        assert marker in device
    result = {
        "outcome": "PASS",
        "retained_files": len(actual),
        "prepack": check_prepack(),
        "decode": check_decode(),
        "boundary": (
            "Packing inverse and half arithmetic only; no compiled fragment ma"
            "pping or CUDA execution."
        ),
    }
    (STUDY / "checks/packing-decode-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: retained source, prepack inverse, exact nibble decode and rounding boundary.\n"
    )


if __name__ == "__main__":
    main()
