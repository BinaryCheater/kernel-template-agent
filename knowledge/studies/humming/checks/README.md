# Replay and validation

Run from the primary KTA checkout. Source roots and immutable inputs are in `inputs/`.
The existing project environment supplies schema and NumPy dependencies; no package was added.

```sh
python3 knowledge/studies/humming/checks/inventory.py
python3 knowledge/studies/humming/checks/bitstream.py
python3 knowledge/studies/humming/checks/repack.py
python3 knowledge/studies/humming/checks/composition.py
python3 knowledge/studies/humming/checks/mechanisms.py
.venv/bin/python knowledge/studies/humming/checks/policy.py
.venv/bin/python knowledge/studies/humming/checks/specializations.py
.venv/bin/python knowledge/studies/humming/checks/validate.py
.venv/bin/python knowledge/studies/humming/checks/references.py
.venv/bin/python knowledge/studies/humming/checks/coverage.py
.venv/bin/ruff check knowledge/studies/humming/checks
.venv/bin/ruff format --check knowledge/studies/humming/checks
```

All commands above passed in the final replay. These are source-arithmetic and schema checks;
none executes a GPU kernel, produces a ProofResult or changes assurance.

| Check | Retained output and domain | Established boundary |
|---|---|---|
| inventory | `inventory.json`: complete pinned source, exports/builders, all CUDA headers and global entries | File integrity and source ownership, not semantic completeness |
| bitstream | `bitstream-result.json`: widths 1..8,1152 one-hot bits,24 patterns,8 final-CTA lengths | Exact code bit identity and ownership; rejects reversed order |
| repack | `repack-result.json`: one64x64 A16/B4 tile, mode3 | 4096 coordinate destinations and independent inverse; rejects wrong interleave |
| composition | `composition-result.json`: N128/K64, block 16x128x64,warp 16x32x32 | All 8192 B fragment coordinates and 48 q-z half decodes; unequal-group postscale mutation rejected |
| mechanisms | `mechanisms-result.json` | 873 atomic-completion orders/two generations; all Hadamard basis vectors at H 4/32/128/256/512,V4;16 FP4 codes/deltas0..3; selected quantizer boundaries;180 raster domains; stage ring indexing |
| policy | `policy-result.json`: SM90 W4A16,N8192,K1024,ten M values | Real source dataclasses, selection priorities, seed candidate scores and final fitting; target facts are analysis inputs |
| specializations | `specializations-result.json`: SM75/80/90/100/120 | Real source type/instruction generation and shared-memory estimate; generated declaration is not compiled code |
| validate | `schema-result.json`: 4 pairs,5 Realisations,primary Chain | Pydantic bodies and expression syntax |
| references | `references-result.json`, `constraints-result.json`: 2 chains,15 Gap bodies,4 valid/invalid premise witnesses | Frozen input integrity, identity/reference/domain checks, dependency order and invalid-choice rejection |
| coverage | `coverage-result.json`, `variant-dispositions.json` | 53 decision dispositions and every device/header owner. Branch ledger is external; each anchor remains partial with next evidence |

The policy and specialization checks extract a closed AST slice from the pinned source.
They execute original definitions with scalar target inputs, without importing a CUDA runtime.
Unavailable paths are absent and fail if reached; no substitute kernel/configuration result is
returned. The supported slice and its residual gaps are stated in the trace.

## Gate result

`scripts/gate --quick WP10a` failed on two pre-existing acceptance-lock mismatches. Nine
other static checks passed. [static-gate.txt](static-gate.txt) preserves the complete output;
[gate-baseline.json](gate-baseline.json) records that both acceptance files and LOCK match
HEAD. No locked file or lock was modified. Full WP acceptance and a commit have not run.
The user clarified that this task ends with the source-study artifacts already on disk.
WP implementation gates and commits are separate; no further gate retry is part of this study.

See [correction-pass.md](correction-pass.md) for the final semantic correction pass and
remaining next evidence. Independent review is separate from this author's pass.
