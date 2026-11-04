import sys
from fractions import Fraction as F
from math import exp, isclose

# Nonuniform trapezoid coefficients; compare original and shifted state/output.
a = [F(1, 2), F(2, 3), F(3, 4), F(4, 5)]
eta = [F(1, 7), F(2, 7), F(3, 7), F(4, 7), F(0)]
gamma = [F(1, 3), F(1, 4), F(1, 5), F(1, 6)]
innov = [F(11), F(13), F(17), F(19)]
s = F(5)
prev = F(7)
t = s + eta[0] * prev
for i in range(4):
    s = a[i] * s + a[i] * eta[i] * prev + gamma[i] * innov[i]
    t = a[i] * t + (gamma[i] + eta[i + 1]) * innov[i]
    assert t - eta[i + 1] * innov[i] == s
    prev = innov[i]
assert t == s
for R in [1, 2, 4]:
    for i in range(3 * R):
        for j in range(3 * R):
            past = i // R > j // R
            same = i // R == j // R
            assert past + same == (i // R >= j // R)
    if R > 1:
        assert 0 // R == 1 // R and 0 != 1
# Quotient phase identity with exact period16, mixed signed increments.
incs = [3, -7, 19, 23, -41, 17]
phase = 5
for C in [1, 2, 4, 6]:
    carry = phase
    out = []
    for start in range(0, len(incs), C):
        running = carry
        for x in incs[start : start + C]:
            running += x
            out.append(running % 16)
        carry = running % 16
    serial = []
    running = phase
    for x in incs:
        running += x
        serial.append(running % 16)
    assert out == serial
# Scalar SSD tail: identity-padded chunk state equals valid-token recurrence.
for C in [2, 3, 4]:
    for L in range(1, 10):
        alpha = [exp(-0.01 * (i + 1)) for i in range(L)]
        b = [i + 2 for i in range(L)]
        seq = [0 if i < L // 2 else 1 for i in range(L)]
        s = 0.0
        serial = []
        for i in range(L):
            if i and seq[i] != seq[i - 1]:
                s = 0.0
            s = alpha[i] * s + b[i]
            serial.append(s)
        carry = 0.0
        out = []
        for start in range(0, L, C):
            for i in range(start, start + C):
                if i >= L:
                    carry = 1.0 * carry + 0.0
                    continue
                if i and seq[i] != seq[i - 1]:
                    carry = 0.0
                carry = alpha[i] * carry + b[i]
                out.append(carry)
        assert all(isclose(x, y) for x, y in zip(serial, out, strict=False)) and isclose(
            carry, serial[-1]
        )
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            (
                "PASS: exact shifted-innovation state/output; same-time rank block"
                "s; quotient phase chunking; SSD identity tails and segment reset"
            )
        ]
    )
    + "\n"
)

# Recover chunk outputs with closed-form causal expansion, not token recurrence.
for L in range(1, 8):
    for C in [2, 3, 4]:
        aa = [F(i + 1, i + 2) for i in range(L)]
        bb = [F(i + 2) for i in range(L)]
        ids = [0 if i < L // 2 else 1 for i in range(L)]
        carry = F(3)
        out = []
        seq_state = carry
        serial = []
        for i in range(L):
            if i and ids[i] != ids[i - 1]:
                seq_state = 0
            seq_state = aa[i] * seq_state + bb[i]
            serial.append(seq_state)
        for start in range(0, L, C):
            end = min(start + C, L)
            incoming_id = ids[start - 1] if start else ids[0]
            chunkout = []
            for i in range(start, end):
                prod = F(1)
                for k in range(start, i + 1):
                    prod *= aa[k]
                value = prod * carry if ids[i] == incoming_id else F(0)
                for j in range(start, i + 1):
                    if ids[j] != ids[i]:
                        continue
                    prod = F(1)
                    for k in range(j + 1, i + 1):
                        prod *= aa[k]
                    value += prod * bb[j]
                chunkout.append(value)
            out += chunkout
            carry = chunkout[-1]
        assert out == serial
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            (
                "PASS: closed-form chunk causal expansion and outgoing state with "
                "short tails and segment boundaries"
            )
        ]
    )
    + "\n"
)
