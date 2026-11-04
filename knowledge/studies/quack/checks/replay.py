import sys
from copy import deepcopy

for L in [1, 2, 4, 8, 16, 32]:
    for S in [1, 2, 4, 8, 16, 32]:
        if S > L:
            continue
        for E in [1, 2, 3]:
            f = [
                [10000 * lane_or_matrix + 100 * s + e for s in range(S) for e in range(E)]
                for lane_or_matrix in range(L)
            ]
            original = deepcopy(f)
            bit = S
            while bit < L:
                old = deepcopy(f)
                for g in range(L):
                    f[g] = [x + y for x, y in zip(old[g], old[g ^ bit], strict=False)]
                bit *= 2
            m = S // 2
            while m:
                for r in range(m):
                    for g in range(L):
                        if not g & m:
                            for e in range(E):
                                a, b = r * E + e, (r + m) * E + e
                                f[g][a], f[g][b] = f[g][b], f[g][a]
                    old = deepcopy(f)
                    for g in range(L):
                        for e in range(E):
                            a, b = r * E + e, (r + m) * E + e
                            f[g][a] = old[g][b] + old[g ^ m][a]
                m //= 2
            for g in range(L):
                assert f[g][:E] == [
                    sum(original[lane_or_matrix][(g % S) * E + e] for lane_or_matrix in range(L))
                    for e in range(E)
                ]
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            (
                "PASS: all power-of-two L/S up to32, E1/2/3; unique slice tags and"
                " surplus lane ownership"
            )
        ]
    )
    + "\n"
)
