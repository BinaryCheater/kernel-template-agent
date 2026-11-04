import sys
from functools import reduce
from itertools import product


def compose(x, y):
    return (y[0] * x[0], y[0] * x[1] + y[1])


for vals in product([-2, 0, 3], repeat=6):
    maps = list(zip(vals[::2], vals[1::2], strict=False))
    assert compose(compose(maps[0], maps[1]), maps[2]) == compose(
        maps[0], compose(maps[1], maps[2])
    )
    for init in [-7, 0, 17]:
        state = init
        for a, b in maps:
            state = a * state + b
        a, b = reduce(compose, maps, (1, 0))
        assert a * init + b == state
        assert compose((a, b), (1, 0)) == (a, b)
assert compose((2, 3), (5, 7)) != compose((5, 7), (2, 3))
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            (
                "PASS:729 affine triples, three initial states, identity tails and"
                " noncommutative witness"
            )
        ]
    )
    + "\n"
)
