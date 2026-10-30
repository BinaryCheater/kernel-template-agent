- [S1] /Users/dingyi/Projects/agentic-codesign/kernel-lab/results/formulaic-reuse-f2-20261027-01/formal-v7
- [S2] /Users/dingyi/Projects/agentic-codesign/kernel-lab/results/formulaic-reuse-f2-20261027-01/formal-v7/export-attention-20261027-01/results.json
- [S3] /Users/dingyi/Projects/agentic-codesign/kernel-lab/results/formulaic-reuse-f2-20261027-01/formal-v5/common-assets/hardware.json

### A1. Share of end-to-end wall, fresh trajectories (medium + long)

| arm | n | end-to-end s | author ok | author timeout/failed | solve | GPU worker active | GPU transport+queue | controller wait | src |
|---|---|---|---|---|---|---|---|---|---|
| all | 18 | 43,135 | 69.1% | 10.1% | 0.9% | 4.1% | 3.1% | 12.7% | [S1] [S2] |
| reconstruct | 6 | 13,241 | 75.2% | 9.1% | 0.8% | 4.3% | 3.2% | 7.4% | [S1] [S2] |
| text | 6 | 16,026 | 63.3% | 8.5% | 0.7% | 3.6% | 2.9% | 21.1% | [S1] [S2] |
| template | 6 | 13,868 | 70.0% | 13.0% | 1.3% | 4.3% | 3.2% | 8.2% | [S1] [S2] |

Provider connection: 91 author sessions, 91 with a reconnect chain starting right after launch, 68 losing >= 60 s; retry span 5,796 s (lower bound, 13.5% of author wall), start-to-fallback 8,282 s (upper bound, 19.3%); author wall 42,995 s. [S1]

### A2. Per trajectory, median of 3 replicates (v7 stages only)

| task/arm | author s | rounds | timeouts | retry s | startup stall s | solve s | compile s | measure s | GPU wall s | wait s | input tok | output tok | reads | writes | checks | read B | knowledge B | src |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| attn-small/reconstruct | 1,056 | 2 | 0 | 106 | 142 | 12 | 2.0 | 13.9 | 48 | 8,145 | 1,079,017 | 18,820 | 30 | 10 | 7 | 327,693 | 0 | [S1] [S2] |
| attn-small/text | 969 | 2 | 0 | 162 | 225 | 11 | 1.5 | 13.8 | 48 | 8,163 | 1,083,432 | 19,500 | 28 | 11 | 4 | 250,107 | 7,912 | [S1] [S2] |
| attn-small/template | 962 | 2 | 0 | 162 | 228 | 12 | 1.4 | 13.8 | 49 | 8,212 | 1,423,844 | 18,726 | 27 | 12 | 7 | 309,169 | 117,921 | [S1] [S2] |
| attn-medium/reconstruct | 2,038 | 4 | 1 | 309 | 439 | 20 | 3.7 | 38.6 | 113 | 116 | 1,467,240 | 31,266 | 38 | 25 | 7 | 540,914 | 0 | [S1] [S2] |
| attn-medium/text | 2,058 | 4 | 0 | 325 | 457 | 21 | 4.5 | 38.9 | 121 | 195 | 2,123,090 | 39,076 | 40 | 23 | 8 | 524,075 | 71,616 | [S1] [S2] |
| attn-medium/template | 2,011 | 4 | 1 | 310 | 441 | 25 | 3.8 | 39.0 | 125 | 115 | 2,183,947 | 29,396 | 60 | 20 | 8 | 651,086 | 269,872 | [S1] [S2] |
| attn-long/reconstruct | 1,678 | 4 | 0 | 179 | 268 | 18 | 4.3 | 116.9 | 213 | 183 | 2,051,234 | 35,280 | 57 | 22 | 6 | 493,001 | 0 | [S1] [S2] |
| attn-long/text | 1,750 | 4 | 0 | 224 | 317 | 14 | 4.5 | 118.1 | 222 | 455 | 1,817,977 | 33,446 | 52 | 18 | 8 | 527,741 | 54,169 | [S1] [S2] |
| attn-long/template | 1,907 | 4 | 0 | 179 | 252 | 35 | 4.5 | 119.3 | 222 | 241 | 2,897,294 | 39,495 | 79 | 26 | 13 | 790,120 | 256,612 | [S1] [S2] |

### A3. Template minus Text (difference of medians)

| task | author s | timeouts | retry s | startup stall s | solve s | compile s | measure s | GPU wall s | wait s | input tok | output tok | reads | writes | checks | read B | knowledge B | src |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| attn-small | -7 | 0 | -0 | 3 | 1 | -0.1 | 0.0 | 0 | 49 | 340,412 | -774 | -1 | 1 | 3 | 59,062 | 110,009 | [S1] [S2] |
| attn-medium | -47 | 1 | -15 | -16 | 3 | -0.7 | 0.1 | 3 | -80 | 60,857 | -9,680 | 20 | -3 | 0 | 127,011 | 198,256 | [S1] [S2] |
| attn-long | 156 | 0 | -46 | -66 | 21 | 0.1 | 1.2 | 0 | -214 | 1,079,317 | 6,049 | 27 | 8 | 5 | 262,379 | 202,443 | [S1] [S2] |

### A4. Per author stage, median (successful provider exits)

| task/arm | n | author s | retry s | startup stall s | input tok | output tok | knowledge B | reads | reads before 1st write | src |
|---|---|---|---|---|---|---|---|---|---|---|
| attn-small/reconstruct | 5 | 498 | 81 | 113 | 589,762 | 9,651 | 0 | 14 | 13 | [S1] |
| attn-small/text | 6 | 487 | 81 | 112 | 575,842 | 9,210 | 3,956 | 13 | 12 | [S1] |
| attn-small/template | 6 | 474 | 81 | 114 | 739,354 | 10,348 | 45,119 | 14 | 12 | [S1] |
| attn-medium/reconstruct | 10 | 523 | 81 | 113 | 542,972 | 10,406 | 0 | 9 | 8 | [S1] |
| attn-medium/text | 11 | 491 | 81 | 114 | 629,632 | 9,976 | 3,956 | 10 | 9 | [S1] |
| attn-medium/template | 10 | 438 | 78 | 110 | 661,786 | 8,832 | 68,912 | 14 | 14 | [S1] |
| attn-long/reconstruct | 12 | 445 | 67 | 99 | 562,399 | 8,807 | 0 | 10 | 9 | [S1] |
| attn-long/text | 11 | 439 | 81 | 100 | 500,654 | 8,994 | 3,956 | 9 | 9 | [S1] |
| attn-long/template | 11 | 499 | 23 | 43 | 671,632 | 10,394 | 66,075 | 21 | 20 | [S1] |

### B1. Per-stage prediction error (predicted vs measured, GB10)

| group | stages | pairs | median abs err | p90 abs err | signed median | within 10 pct | unseen pairs | unseen median abs | resolved-pair order acc. | model-point regret median / p90 | model point = measured best | unqualified configs | src |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| all | 90 | 491 | 5.5% | 55.2% | 0.8% | 64.8% | 404 | 6.5% | 67.6% (556) | 1.5% / 10.1% | 25.0% | 43/540 | [S1] |
| fresh_round01 | 18 | 97 | 16.9% | 127.2% | 16.9% | 38.1% | 97 | 16.9% | 55.9% (68) | 0.9% / 3.6% | 11.1% | 11/108 | [S1] |
| fresh_round02plus | 54 | 297 | 4.1% | 49.8% | 0.5% | 70.4% | 239 | 4.6% | 60.1% (308) | 1.8% / 11.8% | 25.0% | 21/324 | [S1] |
| small_continuations | 18 | 97 | 4.2% | 20.2% | -0.6% | 74.2% | 68 | 6.2% | 85.0% (180) | 2.6% / 12.0% | 38.9% | 11/108 | [S1] |
| attn-small | 18 | 97 | 4.2% | 20.2% | -0.6% | 74.2% | 68 | 6.2% | 85.0% (180) | 2.6% / 12.0% | 38.9% | 11/108 | [S1] |
| attn-medium | 36 | 196 | 11.1% | 71.2% | 3.0% | 45.4% | 166 | 11.1% | 56.0% (218) | 2.0% / 14.5% | 17.1% | 20/216 | [S1] |
| attn-long | 36 | 198 | 1.8% | 48.0% | 1.0% | 79.3% | 170 | 2.1% | 63.9% (158) | 1.0% / 4.6% | 25.7% | 12/216 | [S1] |
| arm:reconstruct | 30 | 154 | 7.0% | 74.9% | 1.4% | 59.7% | 130 | 7.9% | 64.8% (165) | 1.6% / 18.6% | 20.7% | 26/180 | [S1] |
| arm:text | 30 | 168 | 7.2% | 63.4% | 1.0% | 56.0% | 139 | 9.4% | 63.7% (212) | 1.7% / 7.2% | 23.3% | 12/180 | [S1] |
| arm:template | 30 | 169 | 4.2% | 16.5% | 0.1% | 78.1% | 135 | 4.2% | 74.9% (179) | 1.5% / 4.0% | 31.0% | 5/180 | [S1] |

### B2. Error by measured role (all v7 stages)

| role | pairs | median abs err | p90 abs err | signed median | src |
|---|---|---|---|---|---|
| model | 88 | 5.7% | 39.3% | -0.7% | [S1] |
| default | 81 | 1.8% | 31.7% | 0.5% | [S1] |
| probe | 322 | 6.2% | 61.0% | 1.2% | [S1] |

### B3. Final recommendations in the common final batch

| task | n | median abs err | max abs err | signed median | src |
|---|---|---|---|---|---|
| all | 27 | 4.7% | 54.9% | -0.2% | [S2] |
| attn-small | 9 | 4.4% | 27.8% | -0.2% | [S2] |
| attn-medium | 9 | 15.4% | 54.9% | -14.5% | [S2] |
| attn-long | 9 | 1.2% | 9.0% | 0.0% | [S2] |

### B4. Frozen formulation size per stage, median

| task/arm | domain assignments | CPU-legal | factor rows | solver s | src |
|---|---|---|---|---|---|
| attn-small/reconstruct | 66 | 45 | 456 | 0.05 | [S1] |
| attn-small/text | 64 | 56 | 362 | 0.04 | [S1] |
| attn-small/template | 384 | 79 | 5,501 | 0.47 | [S1] |
| attn-medium/reconstruct | 96 | 93 | 856 | 0.09 | [S1] |
| attn-medium/text | 100 | 87 | 679 | 0.07 | [S1] |
| attn-medium/template | 672 | 130 | 6,695 | 0.76 | [S1] |
| attn-long/reconstruct | 64 | 54 | 442 | 0.04 | [S1] |
| attn-long/text | 60 | 60 | 435 | 0.04 | [S1] |
| attn-long/template | 896 | 189 | 13,283 | 4.22 | [S1] |

### B5. K+V working set vs GB10 L2 (hardware.json given to authors)

| task | H/HK/D/N | K+V bytes | / hw_l2_bytes | / hw_l2_effective | src |
|---|---|---|---|---|---|
| attn-small | 16/4/64/1025 | 1,049,600 | 0.0417 | 0.06 | [S1] [S3] |
| attn-medium | 32/8/128/6145 | 25,169,920 | 1.0002 | 1.50 | [S1] [S3] |
| attn-long | 64/8/128/24577 | 100,667,392 | 4.0002 | 6.00 | [S1] [S3] |

Capture A/B deviation (noise reference), all pairs: median 1.1%, p90 4.3%. [S1]
