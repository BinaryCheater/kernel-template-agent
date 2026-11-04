# Batch r9b: entry block

The two probes of r9 again, with their harness faults repaired (same questions, same hypotheses as `../r9/entry.md`; replay per hypothesis in `hypotheses.json`): the rotating walk of r9 wrapped and re-read lines the L2 held, so every rotating point past Z 4 was warm (`../r9/invalid_cases.json`); `volatile` loads went past the L1; `first_share` stopped at the driver's limit on shared memory. Now a 32 MiB read between rotating launches, plain loads, 98304 B per block (two blocks per SM, 96 × w blocks, 24 tiles).

Valid r9 points already say: warm `share` D 1–48 flat at 17.6–19.2 k (replay 21.2 k), one-block cold copy 1069 cycles, cold LDG 1102 (replay 1436, 1156), follower at Z 4 about 105 cycles sooner.
