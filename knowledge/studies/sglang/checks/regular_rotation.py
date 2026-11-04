"Integer regular butterfly witness; native rounding and INT8 quantization are separate."

import json
from pathlib import Path


def transform(values):
    out = list(values)
    bits = (len(out)).bit_length() - 1
    for bit in range(0, bits - 1, 2):
        lo, hi = 1 << bit, 1 << (bit + 1)
        for base in range(len(out)):
            if base & (lo | hi):
                continue
            ids = [base, base | lo, base | hi, base | lo | hi]
            x = [out[i] for i in ids]
            s = (x[0] + x[1]) + (x[2] + x[3])
            for d, i in enumerate(ids):
                out[i] = s - 2 * x[d ^ 3]
    if bits % 2:
        half = 1 << (bits - 1)
        for base in range(half):
            a, b = out[base], out[base | half]
            out[base], out[base | half] = a + b, a - b
    return out


observed = []
for h in [64, 128, 256, 512]:
    a = [(i * 7) % 13 - 6 for i in range(h)]
    b = [(i * 5) % 11 - 5 for i in range(h)]
    ta, tb = transform(a), transform(b)
    assert sum(x * y for x, y in zip(ta, tb, strict=False)) == h * sum(
        x * y for x, y in zip(a, b, strict=False)
    )
    constant = transform([1] * h)
    if h in [64, 256]:
        assert len(set(constant)) == 1
    else:
        assert 0 in constant and len(set(constant)) == 2
    observed.append({"group_size": h, "constant_output_values_unnormalized": sorted(set(constant))})
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "outcome": "passed",
            "observations": observed,
            "scope": "Exact integer matrix/dot identity, not native FP32 or quantization error",
        },
        indent=2,
    )
    + "\n"
)
