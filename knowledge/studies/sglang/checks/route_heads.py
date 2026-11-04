"""Key-only lane heads plus original-ID tie recovery; no GPU/scoring check."""

import json
from itertools import product
from pathlib import Path

cases = 0
for values in product(range(3), repeat=8):
    L = 2
    lanes = 4
    original = [values[i * L : (i + 1) * L] for i in range(lanes)]
    for K in range(1, 9):
        runs = [sorted(row, reverse=True) for row in original]
        selected = []
        prev = None
        run = 0
        for _k in range(K):
            heads = [r[0] if r else -1 for r in runs]
            m = max(heads)
            winner = heads.index(m)
            runs[winner].pop(0)
            pair = (m, winner)
            run = run + 1 if pair == prev else 0
            prev = pair
            hits = [j for j, v in enumerate(original[winner]) if v == m]
            selected.append(winner * L + hits[run])
        expected = sorted(range(8), key=lambda i: (-values[i], i))[:K]
        assert selected == expected
        cases += 1
# Declared16-byte lane-vector bank groups: eight lanes per128-byte phase.
bank_cases = []
for L in (8, 16):
    pitch = L + 4
    for word in range(0, L, 4):
        groups = [((lane * pitch + word) // 4) % 8 for lane in range(8)]
        assert len(set(groups)) == 8
        bank_cases.append({"L": L, "pitch_words": pitch, "word_offset": word, "groups": groups})
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "passed": True,
            "head_cases": cases,
            "declared_vector_bank_groups": bank_cases,
            "scope": (
                "Exact key/ID merge and declared32-bank/4-byte address model for16"
                "-byte vectors; no generated-instruction or runtime performance cl"
                "aim"
            ),
        },
        indent=2,
    )
    + "\n"
)
