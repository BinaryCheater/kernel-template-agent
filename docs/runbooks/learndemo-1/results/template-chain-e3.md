# E3: Template information through the full chain

Status: finished as an experiment, not accepted by the user. [Protocol](template-chain-e3-protocol.md), frozen before the sessions. Figures are local, in `data/template-chain-e3/plots`, drawn by `tools/plot_template_chain.py` from `/Users/dingyi/kta-data/dev/template-chain-e3/figdata`.

## Conclusions

- **The chain produces production-class kernels on both tasks.** Agent sessions wrote families and forms, a real MIP chose shortlists, and every selected kernel passed the independent reference and changed-input graph replay.
- **No Template effect is visible.** Two repeats per arm cannot show a small effect, and the spread inside an arm is larger than any gap between arms.
- **The largest gains came from the revision turn changing structure,** after measurements, in one cell by adding placement and query order. The MIP then chose parameters inside that space.
- **The form did not rank reliably.** Pairs at least 3 % apart: 258 of 405 overall, 64 %; 206 of 306 in round 1, 67 %; 52 of 99 in round 2, 53 %, where candidates are close.

## Confirmed results

Best kernel of each cell, three fresh-seed windows, production measured around each window. All 36 windows valid. Ratio is the median of candidate over production; below 1 is faster.

| Arm | Repeat | Causal forward | Paged decode |
|---|---:|---:|---:|
| none | 1 | 0.709 | 1.016 |
| none | 2 | 0.972 | 1.017 |
| prescribed | 1 | 0.910 | 1.019 |
| prescribed | 2 | 0.980 | 0.987 |
| selection | 1 | 0.924 | 0.995 |
| selection | 2 | 1.213 | 0.964 |

Production: pinned FlashAttention-2 build about 168 µs at S 2048, H 8, D 128; FlashInfer 0.6.6 FA2 about 2360 µs at B 64, H 32, Hk 8, D 128. Decode ratios are inside the window-to-window spread of about ±5 %, so decode is parity, with no cell shown faster or slower. Causal forward has one shape, with no 512 or 513 case, so this does not repeat the first campaign's target test.

The best causal cell is BM 64, BN 32, T 128 with a register operand and descending query order, 117 to 133 µs in three windows. Its window 3 was 133 µs against 118 in the others, unexplained.

## Search against the agent's guess

The anchor is the agent's untimed guess; it compiled in 11 of 12 cells. Counting a difference under 3 % as a tie, in round 1 the MIP's first pick beat it in 5 cells, lost in 4 and tied in 2. The best of the six measured MIP picks beat it in 6, lost in 2 and tied in 3. Search space sizes were 32 to 580, so enumeration would have served, and direct measurement of the whole space would take 7 minutes to 2.3 hours per space. MIP matched enumeration in every checked round. The true optimum of each space is unknown.

## Why moves were missed

Rule counts use the E2 gold sets under leave one source out. Named in DESIGN.md is my text match, not proof of use.

| Task | Moves | No rule in view | Prescribed: named | Selection: surfaced, selected |
|---|---:|---:|---:|---:|
| Causal forward | 18 | 7 | 6, 7 of 11 | 5 and 2, 4 and 1 of 11 |
| Paged decode | 18 | 4 | 9, 11 of 14 | 4 and 3, 9 and 3 of 14 |

Coverage remains a real gap: causal forward has no rule in view for streamed normalization or causal pruning. Selection sessions chose only 1 to 3 gold rules, so in this harder task they read much less than the 58 of 64 selected in E2, where selection was the only job. Composition was not separately scored.

## Deviations and defects

- Four decode cells hit an exporter bug on families with varying kernel counts. They were rerun from round 1 on the same family, with no authoring repeated. Each pre-existing cell.json is kept as `cell-exporter-bug.json`.
- Marlin was the planned second comparator and proved invalid on GB10; FlashAttention forward replaced it. Causal forward is the case the user called too classic, kept for continuity.
- `./chain check` spelling failed in the pilot and one session patched its wrapper; fixed before the matrix. Pilot cells are not scored.
- My jobs overlapped PERFMODEL r26 on spark; that session discarded and reran its first run. Timing jobs now wait for PERFMODEL.
- Batch validity is a 5 % comparator band, looser than gb10-measure canaries. Round 1 and 2 comparator in causal cells moved between about 152 and 169 µs across batches, unexplained; confirmation windows all sat near 168.

## Costs

Per cell: 17 to 42 model minutes, 1.9 to 6.0 M input tokens, mostly cached, 29 to 79 k output tokens, 15 GPU rows, 25 to 43 compiled configurations.
