import sys
from math import isclose

for W in [2, 3, 4]:
    for L in range(1, 7):
        h = [11 + i for i in range(W - 1)]
        x = [23 + i for i in range(L)]
        w = [31 + i for i in range(W)]
        dy = [1 + i for i in range(L)]
        df = [7 + i for i in range(W - 1)]
        z = h + x

        def objective(z, w=w, W=W, L=L, dy=dy, df=df):
            y = [sum(w[j] * z[t + j] for j in range(W)) for t in range(L)]
            return sum(a * b for a, b in zip(dy, y, strict=False)) + sum(
                a * b for a, b in zip(df, z[-W + 1 :], strict=False)
            )

        grad = [0.0] * (W - 1 + L)
        for t in range(L):
            for j in range(W):
                grad[t + j] += dy[t] * w[j]
        for j, g in enumerate(df):
            grad[L + j] += g
        for k, g in enumerate(grad):
            zp = z[:]
            zm = z[:]
            zp[k] += 0.001
            zm[k] -= 0.001
            assert isclose((objective(zp) - objective(zm)) / 0.002, g, rel_tol=1e-7, abs_tol=1e-6)
        for j in range(W - 1):
            q = L - W + 1 + j
            assert (q + W - 1) == L + j
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            (
                "PASS: suffix-history pullback plus stencil adjoint, widths2/3/4 a"
                "nd L1..6, including histories longer than invocation"
            )
        ]
    )
    + "\n"
)
