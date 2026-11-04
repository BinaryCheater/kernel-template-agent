# Batch r10b: harness repairs of r10

`load_sat`: r10's `load_curve` offered only 0.1–0.3 of the L2 rate (span 19 µs at G 48 Z 0: 194 GB/s): its H-queue was not tested. Same question, 16 copies per pass. H-fixed (replay) flat near 392 plus the SM's return queue; H-queue rising with ρ as in r10.
`dram_spread`: r10 stopped at a misaligned address after the idle point (`b * 104729`). Same hypotheses as r10, H-fixed against H-rising.
