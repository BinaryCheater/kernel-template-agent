import sys
from math import gcd

for T in range(1, 8):
    for U in range(1, 8):
        serial = [[0] * U for _ in range(T)]
        serial[0][0] = 1
        for t in range(T):
            for u in range(U):
                if t or u:
                    serial[t][u] = (serial[t - 1][u] if t else 0) + (serial[t][u - 1] if u else 0)
        previous = {0: 1}
        for d in range(1, T + U - 1):
            current = {}
            for u in range(U):
                t = d - u
                if 0 <= t < T:
                    current[u] = (previous.get(u, 0) if t else 0) + (
                        previous.get(u - 1, 0) if u else 0
                    )
            previous = current
        assert previous[U - 1] == serial[-1][-1]
for V in range(1, 17):
    for G in range(1, 33):
        d = gcd(V, G)
        for start in range(0, V * G, V):
            for k in range(0, V, d):
                assert (start + k) // G == (start + k + d - 1) // G


def f(a):
    return sum(sorted(map(abs, a), reverse=True)[:2])


stripes = [[1, -2, 4, -8], [3, -5, 7, -9], [17, 19, -23, 29]]
base = sum(map(f, stripes))
for a in range(4):
    for b in range(4):
        new = [x[:] for x in stripes]
        new[0][a], new[1][b] = new[1][b], new[0][a]
        assert sum(map(f, new)) - base == f(new[0]) + f(new[1]) - f(stripes[0]) - f(stripes[1])
for P in [2, 4, 8]:
    represented = [{i} for i in range(P)]
    for k in range(P.bit_length() - 1):
        old = represented
        assert all(not old[i] & old[i ^ (1 << k)] for i in range(P))
        represented = [old[i] | old[i ^ (1 << k)] for i in range(P)]
    assert all(x == set(range(P)) for x in represented)
# Two-slot local status automaton: clear+flip alternates selected slot.
s = [0, 0]
for k in range(16):
    slot = 0 if s[0] == s[1] else 1
    assert slot == k % 2
    s[slot] ^= 2
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            (
                "PASS: wavefront vs row DP on1..7 grids; gcd microblocks; local ob"
                "jective deltas; peer-round cover and alternating-slot phase"
            )
        ]
    )
    + "\n"
)
