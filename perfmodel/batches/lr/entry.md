# Batch lr: entry block

LEARNDEMO1 search pilot, the first validity-checked run of the whole funnel. Written before the run; form v3, `tools/kta_bridge/constants-v3.json`, frozen at the commit that holds this file. v3 was developed on lp1, whose clock check failed; lp1 is development evidence only.

Question: on shapes new to every batch, does MIP shortlist, compile records, GPU top 8 find a fast kernel at a small measurement count?

Selection: `perfmodel/testset/search/lr.json`. Shapes: HGEMM 3200x2240x1920 (1143 legal), FA 2240x8x96 (180), MoE NVFP4 E 4, T 3840, N 2048, K 1536 (6688). Roles: `mip` the top after compile records, at most 8 (FA keeps 4: most of its shortlist was refused by the generator); `random` 12 compiled controls in draw order, before any time is seen; `anchor` the conventional choice.

Hypotheses: the `mip` set's best beats the anchor and the median of random draws of the same size. Refuted if random controls beat the `mip` set on two of three families.
