# Batch lp1: entry block

LEARNDEMO1 search pilot, not a PERFMODEL round. Written before the run. Question: can the decision-level form (`tools/kta_bridge/forms.py`, constants-v2) pick fast implementations out of a space far larger than the number of kernels measured?

Selection is frozen in `perfmodel/testset/search/lp1.json`, with each row's role and predicted time. Domains: HGEMM 2816x1760x2112 (1143 smem-legal assignments, 250 compiled, generator refusals are domain results), FA forward causal off 1920x9x112 (168), MoE routed GEMM on NVFP4 weights E 6, T 5760, N 1536, K 2048 (6688; new kernel, no compiled or measured row before this batch; its scalar constant comes from six r8 NVFP4 rows, the rest from HGEMM).

Roles: `pre` top by the precompile register estimate, `post` top after compile records, `mip` diverse MIP shortlist, `random` seeded sample of the remainder (controls), `anchor` the nearest conventional choice. Warm and rotating each.

Hypotheses: the shortlist's best beats the random sample's best at the same count; MoE's best moves far from the anchor by larger BM (the form puts dequantisation at nb·kt·BN·BK). What would refute it: random controls faster than the shortlist, or the MoE shortlist above the anchor.
