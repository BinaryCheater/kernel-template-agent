# Marlin source inventory

Status: source study complete. All 18 files match the retained Git tree. One extension registration (`mul`) reaches one CUDA device family (`Marlin`), with ten compiled specializations. [Trace](trace.md) records branch and caller dispositions; [checks](checks/README.md) retain logical replays.

| Group | Files | Final disposition |
|---|---|---|
| Public API / preparation | `marlin/__init__.py` | Layer/mul contract, packing, scale layouts, forward and replacement traced. Host packing and both scale/read compositions checked. |
| Binding / build | `marlin/marlin_cuda.cpp`, `setup.py` | One entry, one C++/CUDA source pair; pointer/group/workspace contract and missing validation recorded. |
| CUDA arithmetic / layouts / pipeline | `marlin/marlin_cuda_kernel.cu` | Decode, A/B fragments, shared swizzle/stages, scale ring, CTA tree and final output traced. Logical coordinates checked; device synchronization and emitted instructions remain obligations. |
| Dispatch / scheduler / global combine | Same CUDA source | Five geometries × per-channel/group128, auto/explicit tiles, M chunk/tail, clipped stripes and reverse-K commits reconciled. Invalid max_par nonprogress witness retained. |
| Tests / benchmark | `test.py`, `bench.py` | Shape/group/tail callers read; skipped Llama tests, broad exceptions, helper buffer overrides and zero-scale timing inputs recorded. No upstream tests or benchmarks executed. |
| GPTQ / model integration | `gptq/datautils.py`, `eval.py`, `gptq.py`, `llama2.py`, `quant.py` | Symmetric scale-only packing contract, checkpoint group selection, calibration and model/data delegation traced; no additional custom CUDA family. |
| Supporting files | `README.md`, `LICENSE`, `.gitignore`, `assets/marlin.png`, `assets/models.png`, `assets/peak.png`, `assets/sustained.png` | Documentation/license/configuration and four image assets; no build registration or device implementation. |
