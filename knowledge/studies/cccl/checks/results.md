# Witness results

Command: `.venv/bin/python knowledge/studies/cccl/checks/replay.py`.

Domain: 1..97 tiles, every earlier inclusive anchor for each current tile, unique string tokens and ordered concatenation; a 35-item tail in a 64-thread block. This distinguishes prepend from reversed operands. Fixed 32-tile anchors are a subset of the checked anchor domain. Native status publication, CUB shuffle trees, CUDA partial loads and scheduling progress were not executed.

Observed: `PASS: 152096 ordered lookback witnesses; 35-of-64 partial block sum=630`.

Continuation command: `.venv/bin/python knowledge/studies/cccl/checks/design_witnesses.py`.
Observed: `PASS: stable merge diagonals with duplicate keys; radix key/payload scatter; decoded-bin run conservation`.
Distinct key payloads distinguish incorrect equal-key merge priority and unstable digit scatter. Histogram sample includes an invalid bin and separated equal-bin runs. These are source contract witnesses, not CUDA execution.

cudax continuation witness: `PASS: sparse three-lane logical-rank reduction preserves noncommutative ABC order` from checks/design_witnesses.py. It checks logical rank and the ordered three-member recurrence; native group shuffle/convergence remains open.
