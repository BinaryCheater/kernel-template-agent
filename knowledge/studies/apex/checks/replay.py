import sys
from itertools import product

for sizes in product([1, 3, 9], repeat=3):
    for C in [1, 2, 4]:
        for T in [1, 2, 3]:
            for B in [1, 2, 4]:
                metadata = []
                descriptors = []
                recovered = []
                for t, n in enumerate(sizes):
                    metadata.append(t)
                    for c in range((n + C - 1) // C):
                        descriptors.append((len(metadata) - 1, c))
                        last = c == (n + C - 1) // C - 1
                        if (
                            (len(metadata) == T and last)
                            or len(descriptors) == B
                            or (t == len(sizes) - 1 and last)
                        ):
                            assert len(metadata) <= T and len(descriptors) <= B
                            recovered.extend((metadata[slot], chunk) for slot, chunk in descriptors)
                            descriptors = []
                            metadata = [] if last else [t]
                assert recovered == [
                    (t, c) for t, n in enumerate(sizes) for c in range((n + C - 1) // C)
                ]
sys.stdout.write(
    " ".join(
        str(value)
        for value in [
            (
                "PASS: 729 bounded batch combinations; chunk carry, metadata rebas"
                "e, exact-once descriptors"
            )
        ]
    )
    + "\n"
)
