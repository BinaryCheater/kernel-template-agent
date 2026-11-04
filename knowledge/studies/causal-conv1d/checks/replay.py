import sys

for W in [2, 3, 4]:
    for H in range(W - 1, 8):
        for p in range(H, 3 * H + 1):
            prefix = list(range(1, p + 1))
            ring = [None] * H
            for q, v in enumerate(prefix):
                ring[q % H] = v
            window = [ring[q % H] for q in range(p - W + 1, p)]
            weights = [11 + 2 * j for j in range(W)]
            for i, x in enumerate([31, 43, 59, 71, 89]):
                ring[(p + i) % H] = x
                window.append(x)
                got = sum(a * b for a, b in zip(weights, window, strict=False))
                allx = prefix + [31, 43, 59, 71, 89][: i + 1]
                assert got == sum(a * b for a, b in zip(weights, allx[-W:], strict=False))
                window = window[1:]
                nextpos = p + i + 1
                assert [ring[q % H] for q in range(nextpos - H, nextpos)] == allx[-H:]
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            ("PASS: widths2/3/4, H=W-1..7, multiple starting positions and repeated wraparound")
        ]
    )
    + "\n"
)
