# Batch r10c: second repair

`load_sat` stopped at an address past its 4 KiB of shared memory. `dram_spread`: the load blocks overlapped each other's lines (2000 GB/s: L2 hits), so r10b's spread is the idle spread, kept in r10b as that. Blocks now read apart. Hypotheses as in r10 and r10b.
