# Batch r1fa: entry block

Supplementary to batch r1 (same round, same protocol). Reason: in r1 the FA-forward harness launched TileLang's kernel with 64 KB of dynamic shared memory (the generator's estimate) where the kernel needs 80 KB (TileLang's host stub: 81920), so the kernel faulted (illegal memory access) and r1 has no FA value. The generator now takes the dynamic shared memory from TileLang's host stub. Only the FA kernel and the protocol kernels are built and run (`perfmodel-measure build r1fa --only fa_fwd`).

No probe: the FA row is a test-set kernel. Its replay prediction (the generic cp.async front end of `gb10sim/probeset.py`: busiest SM, block replacement, chip shares by blocks in flight, warm L2) is in `predicted.json`, written before the run. Cache state `warm`.
