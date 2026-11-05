# Error table

**Core table**: 890 rows, 4316 rows left out (`perfmodel/core.txt`, LOOP §8); scores below are over these rows only and do not compare with a full table's.

Produced by `scripts/perfmodel-table` from KTA commit `7083a2ed62424ac197f8444983a1a19a1a9fd854` on 2026-11-05T10:08:29Z; upstream inputs `/Users/dingyi/Projects/ai-infra-codesign-ventus` at `a628c9d963cf9c029fa596b872d6e8920ca883e8`. CPU only: replay time 3467 s summed over rows (slowest row 71 s); this run 201 s with 18 processes, 0 rows from the cache. Error = predicted / measured − 1 (positive: the model is slow). Machine-readable copy: `error-table.json`. Mechanisms only (LOOP §4.4): every fitted parameter replaced by its probe value or removed: {'launch_stream_us': 2.05, 'tma_r_GBs_alone': 80.7, 'tma_l2_aggregate_GBs': 2250, 'block_setup_us': 0.0}.

## Summary

| Set | n | excluded | median abs error | worst | median, mechanisms only | worst, mechanisms only | blind median / worst | within 10 % | Spearman | Kendall τ | τ between repeats (ceiling, a record) | best variant | inversions | reversed pairs within / above the measured spread | Spearman / inversions, mechanisms only | counts check (R1) | bottleneck agrees |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemm1024 full | 7 | 0 | 7.6 % | +12.1 % | 6.2 % | +10.9 % |  | 6 | 0.89 | 0.81 |  | no (+12.7 %) | 2 | 0 / 2 | 0.89 / 2 |  |  |
| gemm1024 nocompute | 5 | 0 | 5.2 % | -8.3 % | 6.1 % | -10.5 % |  | 5 |  |  |  |  |  |  |  |  |  |
| gemm1024 noload 1 block | 5 | 0 | 6.7 % | +14.6 % | 6.7 % | +14.6 % |  | 4 |  |  |  |  |  |  |  |  |  |
| tma full | 3 | 0 | 9.3 % | +30.2 % | 17.6 % | +29.9 % |  | 2 | 1.00 | 1.00 |  | yes | 0 | 0 / 0 | 1.00 / 0 |  |  |
| tma noload 1 block | 3 | 0 | 5.5 % | +20.2 % | 5.5 % | +20.2 % |  | 2 |  |  |  |  |  |  |  |  |  |
| probe sched.cu | 2 | 0 | 3.8 % | -6.3 % | 3.8 % | -6.3 % |  | 2 |  |  |  |  |  |  |  |  |  |
| probe alu.cu | 7 | 0 | 2.5 % | +5.1 % | 2.5 % | +5.1 % |  | 7 |  |  |  |  |  |  |  |  |  |
| probe alumix.cu | 4 | 0 | 0.7 % | -4.4 % | 0.7 % | -4.4 % |  | 4 |  |  |  |  |  |  |  |  |  |
| probe sfu.cu | 6 | 0 | 0.3 % | -3.2 % | 0.3 % | -3.2 % |  | 6 |  |  |  |  |  |  |  |  |  |
| probe cvt.cu | 1 | 0 | 0.4 % | -0.4 % | 0.4 % | -0.4 % |  | 1 |  |  |  |  |  |  |  |  |  |
| probe smem_ldsts | 7 | 0 | 1.6 % | -21.5 % | 1.6 % | -21.5 % | 18.6 % / -47.1 % | 5 |  |  |  |  |  |  |  |  |  |
| probe shfl | 3 | 0 | 0.3 % | -2.6 % | 0.3 % | -2.6 % | 48.2 % / -72.6 % | 3 |  |  |  |  |  |  |  |  |  |
| probe ex2_ffma | 3 | 0 | 0.4 % | -1.1 % | 0.4 % | -1.1 % | 0.4 % / -1.1 % | 3 |  |  |  |  |  |  |  |  |  |
| probe namedbar | 3 | 0 | 0.2 % | +0.2 % | 0.2 % | +0.2 % | 34.7 % / +191.7 % | 3 |  |  |  |  |  |  |  |  |  |
| probe l1hit | 3 | 0 | 7.0 % | -10.2 % | 7.0 % | -10.2 % | 21.1 % / +85.8 % | 2 |  |  |  |  |  |  |  |  |  |
| probe store_tail | 4 | 0 | 1.8 % | +7.5 % | 1.8 % | +7.5 % | 3.2 % / +23.9 % | 4 |  |  |  |  |  |  |  |  |  |
| probe l2share_copy | 2 | 0 | 13.1 % | -21.4 % | 13.1 % | -21.4 % | 12.3 % / -22.1 % | 1 |  |  |  |  |  |  |  |  |  |
| probe dram_rotate | 3 | 0 | 1.4 % | -12.9 % | 1.4 % | -12.9 % | 18.9 % / +38.8 % | 2 |  |  |  |  |  |  |  |  |  |
| testset fa_forward | 2 | 0 | 4.5 % | +5.3 % | 4.5 % | +5.3 % | 11.9 % / -15.1 % | 2 |  |  |  |  |  |  |  | 2/2 | 2/2 |
| testset hgemm2048 | 10 | 0 | 5.8 % | +12.2 % | 5.8 % | +12.2 % | 5.5 % / -14.2 % | 8 | 0.48 | 0.38 | 0.87 (0.73–0.96, 7 batches) | yes (+0.4 %) | 9 | 9 / 5 | 0.48 / 9 | 10/10 | 9/10 |
| probe mma_pair | 3 | 0 | 0.7 % | +2.8 % | 0.7 % | +2.8 % | 1.8 % / +2.8 % | 3 |  |  |  |  |  |  |  |  |  |
| probe l2share_stages | 4 | 0 | 11.9 % | +22.5 % | 11.9 % | +22.5 % | 12.1 % / +20.3 % | 2 |  |  |  |  |  |  |  |  |  |
| testset decode_gqa_combine | 1 | 0 | 1.4 % | +1.4 % | 1.4 % | +1.4 % | 34.4 % / +34.4 % | 1 |  |  |  |  |  |  |  | 1/1 | 0/1 |
| testset decode_gqa | 2 | 0 | 5.9 % | +8.7 % | 5.9 % | +8.7 % | 12.4 % / +15.5 % | 2 |  |  |  |  |  |  |  | 2/2 | 1/2 |
| testset decode_gqa b2b | 1 | 0 | 7.4 % | +7.4 % | 7.4 % | +7.4 % | 4.9 % / -4.9 % | 1 |  |  |  |  |  |  |  |  |  |
| testset decode_gqa graph | 1 | 0 | 4.5 % | +4.5 % | 4.5 % | +4.5 % | 10.8 % / -10.8 % | 1 |  |  |  |  |  |  |  |  |  |
| testset gemv_w4a16 | 2 | 0 | 10.4 % | -13.3 % | 10.4 % | -13.3 % | 631.3 % / +1053.8 % | 1 |  |  |  |  |  |  |  | 2/2 | 0/2 |
| testset gemv_w4a16 b2b | 1 | 0 | 10.2 % | -10.2 % | 10.2 % | -10.2 % | 6.8 % / -6.8 % | 0 |  |  |  |  |  |  |  |  |  |
| testset gemv_w4a16 graph | 1 | 0 | 7.8 % | -7.8 % | 7.8 % | -7.8 % | 7.7 % / -7.7 % | 1 |  |  |  |  |  |  |  |  |  |
| testset rmsnorm | 2 | 0 | 8.0 % | -11.6 % | 8.0 % | -11.6 % | 39.6 % / -49.1 % | 1 |  |  |  |  |  |  |  | 2/2 | 0/2 |
| testset softmax | 2 | 0 | 14.2 % | -14.7 % | 14.2 % | -14.7 % | 20.8 % / -22.8 % | 0 |  |  |  |  |  |  |  | 2/2 | 0/2 |
| probe l2reuse | 9 | 0 | 1.8 % | -76.2 % | 1.8 % | -76.2 % | 14.4 % / -88.1 % | 5 |  |  |  |  |  |  |  |  |  |
| probe launch | 7 | 0 | 5.7 % | +40.8 % | 5.7 % | +40.8 % | 26.2 % / +68.9 % | 4 |  |  |  |  |  |  |  |  |  |
| testset fp8gemm2048 | 2 | 0 | 3.9 % | +5.0 % | 3.9 % | +5.0 % | 8.8 % / -11.0 % | 2 |  |  |  |  |  |  |  | 2/2 | 2/2 |
| probe dram_rw | 3 | 0 | 4.2 % | +7.5 % | 4.2 % | +7.5 % | 53.0 % / -68.6 % | 3 |  |  |  |  |  |  |  |  |  |
| testset hgemm2048 rotating | 10 | 0 | 7.6 % | +16.2 % | 7.6 % | +16.2 % | 9.3 % / -14.3 % | 6 | 0.93 | 0.78 | 0.91 (0.82–0.91, 4 batches) | yes | 0 | 5 / 0 | 0.93 / 0 | 10/10 | 7/10 |
| testset moe_gemm | 2 | 0 | 6.6 % | +6.7 % | 6.6 % | +6.7 % | 144.0 % / +223.9 % | 2 |  |  |  |  |  |  |  | 2/2 | 0/2 |
| probe mma_types | 5 | 0 | 5.0 % | -5.0 % | 5.0 % | -5.0 % | 5.0 % / -5.1 % | 5 |  |  |  |  |  |  |  |  |  |
| probe row_reduce | 4 | 0 | 12.9 % | +137.5 % | 12.9 % | +137.5 % | 24.4 % / +137.5 % | 2 |  |  |  |  |  |  |  |  |  |
| testset w4gemm | 4 | 0 | 2.1 % | +2.6 % | 2.1 % | +2.6 % | 11.7 % / -15.9 % | 4 | 0.80 | 0.67 | 0.67 (0.67–0.67, 1 batch, samples) | yes | 0 | 1 / 0 | 0.80 / 0 | 4/4 | 4/4 |
| probe st256 | 5 | 0 | 36.5 % | -39.9 % | 36.5 % | -39.9 % | 14.6 % / +42.6 % | 2 |  |  |  |  |  |  |  |  |  |
| probe branch | 7 | 0 | 0.6 % | +19.7 % | 0.6 % | +19.7 % | 14.9 % / -24.8 % | 6 |  |  |  |  |  |  |  |  |  |
| probe mma_issue | 6 | 0 | 4.6 % | -7.9 % | 4.6 % | -7.9 % | 4.9 % / -31.9 % | 6 |  |  |  |  |  |  |  |  |  |
| probe lds_narrow | 7 | 0 | 0.5 % | +2.1 % | 0.5 % | +2.1 % | 12.2 % / -83.8 % | 7 |  |  |  |  |  |  |  |  |  |
| probe launch_short | 7 | 0 | 1.4 % | +6.3 % | 1.4 % | +6.3 % | 27.0 % / +54.5 % | 7 |  |  |  |  |  |  |  |  |  |
| testset decode_gqa_seq | 2 | 0 | 8.0 % | +12.4 % | 8.0 % | +12.4 % | 10.7 % / +14.9 % | 1 |  |  |  |  |  |  |  |  |  |
| testset decode_gqa_seq graph | 1 | 0 | 5.8 % | +5.8 % | 5.8 % | +5.8 % | 8.5 % / -8.5 % | 1 |  |  |  |  |  |  |  |  |  |
| testset hgemm 1024x1024x1024 | 3 | 0 | 4.5 % | +8.5 % | 4.5 % | +8.5 % | 5.4 % / -5.7 % | 3 | 1.00 | 1.00 | 1.00 (1.00–1.00, 2 batches) | yes | 0 | 0 / 0 | 1.00 / 0 | 3/3 | 2/3 |
| testset hgemm 1024x1024x1024 rotating | 3 | 0 | 7.9 % | +13.5 % | 7.9 % | +13.5 % | 12.6 % / -13.1 % | 2 | 1.00 | 1.00 | 1.00 (1.00–1.00, 2 batches) | yes | 0 | 0 / 0 | 1.00 / 0 | 3/3 | 0/3 |
| probe mma_pipe | 7 | 0 | 0.0 % | +2.6 % | 0.0 % | +2.6 % | 0.3 % / +5.2 % | 7 |  |  |  |  |  |  |  |  |  |
| probe dram_burst | 4 | 0 | 1.9 % | -6.2 % | 1.9 % | -6.2 % | 2.4 % / -5.9 % | 4 |  |  |  |  |  |  |  |  |  |
| probe l2mix | 8 | 0 | 26.2 % | -64.6 % | 26.2 % | -64.6 % | 45.5 % / +191.2 % | 1 |  |  |  |  |  |  |  |  |  |
| testset fa 1536x12x96 | 2 | 0 | 3.1 % | -6.1 % | 3.1 % | -6.1 % | 9.7 % / -11.3 % | 2 |  |  |  |  |  |  |  | 2/2 | 2/2 |
| testset hgemm 3072x1536x2048 | 14 | 0 | 4.6 % | -19.9 % | 4.6 % | -19.9 % | 5.6 % / -60.4 % | 12 | 0.64 | 0.47 | 0.97 (0.97–0.97, 2 batches) | yes | 14 | 11 / 13 | 0.64 / 14 | 14/14 | 10/14 |
| testset hgemm 3072x1536x2048 rotating | 14 | 0 | 4.5 % | -13.8 % | 4.5 % | -13.8 % | 7.6 % / -58.4 % | 11 | 0.92 | 0.74 | 0.96 (0.96–0.96, 2 batches) | no (+3.6 %) | 7 | 8 / 4 | 0.92 / 7 | 14/14 | 9/14 |
| testset whatif decode rotating | 10 | 0 | 3.0 % | +16.4 % | 3.0 % | +16.4 % | 6.2 % / +16.3 % | 8 |  |  |  |  |  |  |  | 5/5 | 0/5 |
| testset whatif decode warm | 7 | 0 | 7.5 % | +14.4 % | 7.5 % | +14.4 % | 6.2 % / -11.0 % | 5 |  |  |  |  |  |  |  | 3/3 | 1/3 |
| testset whatif decode warm b2b | 6 | 0 | 2.7 % | +12.3 % | 2.7 % | +12.3 % | 5.9 % / -8.3 % | 5 |  |  |  |  |  |  |  |  |  |
| testset whatif decode warm graph | 6 | 0 | 1.7 % | -8.5 % | 1.7 % | -8.5 % | 6.6 % / -11.7 % | 6 |  |  |  |  |  |  |  |  |  |
| testset whatif fa rotating | 5 | 0 | 8.7 % | -44.0 % | 8.7 % | -44.0 % | 13.4 % / -40.4 % | 3 |  |  |  |  |  |  |  | 3/5 | 5/5 |
| testset whatif fa warm | 11 | 0 | 4.9 % | -20.6 % | 4.9 % | -20.6 % | 9.4 % / -24.9 % | 9 |  |  |  |  |  |  |  | 4/5 | 4/5 |
| testset whatif gemv rotating | 5 | 0 | 13.5 % | +14.9 % | 13.5 % | +14.9 % | 11.1 % / -83.2 % | 1 |  |  |  |  |  |  |  | 5/5 | 0/5 |
| testset whatif gemv warm | 4 | 0 | 14.3 % | -22.1 % | 14.3 % | -22.1 % | 16.4 % / -21.3 % | 1 |  |  |  |  |  |  |  | 4/4 | 0/4 |
| testset whatif gemv warm b2b | 4 | 0 | 11.2 % | -17.1 % | 11.2 % | -17.1 % | 13.4 % / -17.9 % | 1 |  |  |  |  |  |  |  |  |  |
| testset whatif gemv warm graph | 4 | 0 | 12.9 % | -23.2 % | 12.9 % | -23.2 % | 14.0 % / -26.5 % | 2 |  |  |  |  |  |  |  |  |  |
| testset whatif hgemm rotating | 15 | 0 | 5.1 % | -18.9 % | 5.1 % | -18.9 % | 7.1 % / -23.2 % | 14 |  |  |  |  |  |  |  | 14/15 | 6/15 |
| testset whatif hgemm warm | 21 | 0 | 4.1 % | -23.9 % | 4.1 % | -23.9 % | 5.9 % / -47.2 % | 16 |  |  |  |  |  |  |  | 17/18 | 13/18 |
| testset whatif rmsnorm rotating | 12 | 0 | 6.4 % | +18.4 % | 6.4 % | +18.4 % | 6.7 % / +108.3 % | 9 |  |  |  |  |  |  |  | 8/8 | 0/8 |
| testset whatif rmsnorm warm | 6 | 0 | 15.3 % | -33.6 % | 15.3 % | -33.6 % | 19.7 % / -36.7 % | 2 |  |  |  |  |  |  |  | 3/3 | 0/3 |
| testset whatif rmsnorm warm b2b | 6 | 0 | 11.0 % | -31.5 % | 11.0 % | -31.5 % | 12.9 % / -32.8 % | 2 |  |  |  |  |  |  |  |  |  |
| testset whatif rmsnorm warm graph | 6 | 0 | 11.1 % | -34.9 % | 11.1 % | -34.9 % | 24.5 % / -33.7 % | 2 |  |  |  |  |  |  |  |  |  |
| testset whatif rmsnorm rotating b2b | 2 | 0 | 5.8 % | -6.2 % | 5.8 % | -6.2 % | 63.8 % / +123.0 % | 2 |  |  |  |  |  |  |  |  |  |
| testset whatif rmsnorm rotating graph | 2 | 0 | 5.6 % | -6.4 % | 5.6 % | -6.4 % | 70.4 % / +137.3 % | 2 |  |  |  |  |  |  |  |  |  |
| testset whatif softmax warm | 2 | 0 | 16.7 % | -24.4 % | 16.7 % | -24.4 % | 23.1 % / -25.5 % | 1 |  |  |  |  |  |  |  | 2/2 | 0/2 |
| testset whatif softmax warm b2b | 2 | 0 | 16.4 % | -22.1 % | 16.4 % | -22.1 % | 18.2 % / -18.6 % | 0 |  |  |  |  |  |  |  |  |  |
| testset whatif softmax warm graph | 2 | 0 | 21.1 % | -23.6 % | 21.1 % | -23.6 % | 21.7 % / -24.2 % | 0 |  |  |  |  |  |  |  |  |  |
| testset whatif w4gemm warm | 3 | 0 | 2.5 % | -16.7 % | 2.5 % | -16.7 % | 14.8 % / -15.8 % | 2 |  |  |  |  |  |  |  | 3/3 | 2/3 |
| probe rot_depth | 5 | 0 | 2.6 % | -5.6 % | 2.6 % | -5.6 % | 2.5 % / +17.4 % | 5 |  |  |  |  |  |  |  |  |  |
| probe spill | 7 | 0 | 6.8 % | -23.1 % | 6.8 % | -23.1 % | 23.1 % / -94.8 % | 5 |  |  |  |  |  |  |  |  |  |
| probe store_lines | 9 | 0 | 25.2 % | +1204.8 % | 25.2 % | +1204.8 % | 60.9 % / +417.5 % | 3 |  |  |  |  |  |  |  |  |  |
| probe warp_map | 6 | 0 | 0.6 % | +69.8 % | 0.6 % | +69.8 % | 32.3 % / +277.1 % | 5 |  |  |  |  |  |  |  |  |  |
| probe copy_dram | 5 | 0 | 2.9 % | +7.2 % | 2.9 % | +7.2 % | 31.8 % / +77.7 % | 5 |  |  |  |  |  |  |  |  |  |
| probe store_spread | 6 | 0 | 42.0 % | +74.8 % | 42.0 % | +74.8 % | 34.1 % / +431.7 % | 0 |  |  |  |  |  |  |  |  |  |
| probe arith_beside | 5 | 0 | 3.2 % | -9.0 % | 3.2 % | -9.0 % | 2.2 % / -8.8 % | 5 |  |  |  |  |  |  |  |  |  |
| probe mma_ldsm | 4 | 0 | 1.9 % | -3.1 % | 1.9 % | -3.1 % | 1.9 % / -3.1 % | 4 |  |  |  |  |  |  |  |  |  |
| probe spill_rw | 9 | 0 | 13.5 % | +99.1 % | 13.5 % | +99.1 % | 42.7 % / +192.4 % | 4 |  |  |  |  |  |  |  |  |  |
| testset nvfp4gemm | 6 | 0 | 4.0 % | -7.4 % | 4.0 % | -7.4 % | 32.7 % / -37.4 % | 6 | 0.43 | 0.20 | 0.87 (0.87–0.87, 2 batches) | yes (+0.5 %) | 0 | 6 / 0 | 0.43 / 0 | 6/6 | 6/6 |
| testset whatif rmsnorm m192 | 5 | 0 | 1.2 % | +17.5 % | 1.2 % | +17.5 % | 2.9 % / -14.1 % | 3 |  |  |  |  |  |  |  | 3/5 | 0/5 |
| testset whatif rmsnorm m192 b2b | 5 | 0 | 7.9 % | +12.5 % | 7.9 % | +12.5 % | 7.0 % / +14.8 % | 3 |  |  |  |  |  |  |  |  |  |
| testset whatif rmsnorm m192 graph | 5 | 0 | 7.1 % | -15.3 % | 7.1 % | -15.3 % | 5.4 % / -20.1 % | 4 |  |  |  |  |  |  |  |  |  |
| testset whatif rmsnorm m48 | 5 | 0 | 7.0 % | -45.5 % | 7.0 % | -45.5 % | 7.0 % / -45.5 % | 3 |  |  |  |  |  |  |  | 3/5 | 0/5 |
| testset whatif rmsnorm m48 b2b | 5 | 0 | 2.3 % | +94.2 % | 2.3 % | +94.2 % | 1.6 % / +91.0 % | 3 |  |  |  |  |  |  |  |  |  |
| testset whatif rmsnorm m48 graph | 5 | 0 | 7.0 % | -21.9 % | 7.0 % | -21.9 % | 8.8 % / -23.4 % | 3 |  |  |  |  |  |  |  |  |  |
| testset whatif softmax m192 | 2 | 0 | 11.2 % | +11.9 % | 11.2 % | +11.9 % | 7.9 % / -8.3 % | 0 |  |  |  |  |  |  |  | 2/2 | 0/2 |
| testset whatif softmax m192 b2b | 2 | 0 | 3.2 % | -4.8 % | 3.2 % | -4.8 % | 3.8 % / -5.4 % | 2 |  |  |  |  |  |  |  |  |  |
| testset whatif softmax m192 graph | 2 | 0 | 8.5 % | -10.3 % | 8.5 % | -10.3 % | 9.7 % / -12.9 % | 1 |  |  |  |  |  |  |  |  |  |
| testset whatif softmax m48 | 2 | 0 | 0.8 % | +1.1 % | 0.8 % | +1.1 % | 0.3 % / +0.5 % | 2 |  |  |  |  |  |  |  | 2/2 | 0/2 |
| testset whatif softmax m48 b2b | 2 | 0 | 7.9 % | +12.1 % | 7.9 % | +12.1 % | 6.9 % / +11.0 % | 1 |  |  |  |  |  |  |  |  |  |
| testset whatif softmax m48 graph | 2 | 0 | 2.2 % | -2.8 % | 2.2 % | -2.8 % | 2.4 % / -4.5 % | 2 |  |  |  |  |  |  |  |  |  |
| probe first_reader | 8 | 0 | 3.3 % | +4.7 % | 3.3 % | +4.7 % | 3.5 % / +36.7 % | 8 |  |  |  |  |  |  |  |  |  |
| probe first_share | 7 | 0 | 18.5 % | +27.3 % | 18.5 % | +27.3 % | 20.3 % / +102.1 % | 2 |  |  |  |  |  |  |  |  |  |
| probe load_curve | 6 | 0 | 3.6 % | +5.0 % | 3.6 % | +5.0 % | 3.2 % / -5.0 % | 6 |  |  |  |  |  |  |  |  |  |
| probe copy_depth | 8 | 0 | 30.3 % | -57.8 % | 30.3 % | -57.8 % | 41.9 % / -74.5 % | 0 |  |  |  |  |  |  |  |  |  |
| testset breadth hand | 4 | 0 | 7.7 % | -30.9 % | 7.7 % | -30.9 % | 23.1 % / +166.9 % | 2 | 0.50 | 0.40 |  | no (+42.5 %) | 1 | 0 / 1 | 0.50 / 1 | 3/4 | 0/4 |
| testset breadth hand graph | 1 | 0 | 115.7 % | +115.7 % | 115.7 % | +115.7 % | 104.0 % / +104.0 % | 0 |  |  |  |  |  |  |  |  |  |
| testset breadth stress | 4 | 0 | 2.0 % | -14.1 % | 2.0 % | -14.1 % | 2.0 % / -14.1 % | 3 |  | 0.00 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 0 | 0 / 0 |  | 1/4 | 0/4 |
| probe load_sat | 5 | 0 | 13.8 % | +55.8 % | 13.8 % | +55.8 % | 17.7 % / +768.9 % | 2 |  |  |  |  |  |  |  |  |  |
| testset breadth library | 4 | 0 | 98.9 % | -99.2 % | 98.9 % | -99.2 % | 98.9 % / -99.6 % | 0 | 0.74 | 0.55 |  | yes | 0 | 0 / 1 | 0.74 / 0 | 0/4 | 0/4 |
| probe l2_cap | 5 | 0 | 5.2 % | -32.7 % | 5.2 % | -32.7 % | 68.3 % / +73.9 % | 3 |  |  |  |  |  |  |  |  |  |
| testset atomic_bins | 7 | 0 | 14.9 % | -76.0 % | 14.9 % | -76.0 % | 76.5 % / +2998.6 % | 2 | 0.93 | 0.81 |  | yes (+0.3 %) | 1 | 1 / 1 | 0.93 / 1 | 7/7 | 0/7 |
| testset store_rewrite | 5 | 0 | 2.4 % | +44.7 % | 2.4 % | +44.7 % | 79.7 % / +892.7 % | 3 | 0.92 | 0.78 | 0.40 (0.40–0.40, 1 batch, samples) | yes (+1.2 %) | 0 | 0 / 0 | 0.92 / 0 | 0/5 | 0/5 |
| testset wr_past_l2 | 5 | 0 | 20.9 % | +52.3 % | 20.9 % | +52.3 % | 32.2 % / +124.5 % | 2 | 1.00 | 1.00 |  | yes | 0 | 0 / 0 | 1.00 / 0 | 5/5 | 0/5 |
| testset wr_rotating | 3 | 0 | 9.3 % | +9.8 % | 9.3 % | +9.8 % | 7.1 % / +7.6 % | 3 | 1.00 | 1.00 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 0 | 0 / 0 | 1.00 / 0 | 3/3 | 0/3 |
| probe l2_local | 4 | 0 | 5.4 % | -22.0 % | 5.4 % | -22.0 % | 56.8 % / +107.9 % | 3 |  |  |  |  |  |  |  |  |  |
| probe store_merge | 6 | 0 | 44.2 % | +56.8 % | 44.2 % | +56.8 % | 54.6 % / +413.1 % | 0 |  |  |  |  |  |  |  |  |  |
| testset breadth blind | 28 | 0 | 19.4 % | +115.6 % | 19.4 % | +115.6 % | 44.0 % / +254.0 % | 12 | 0.79 | 0.64 |  | yes | 65 | 4 / 63 | 0.79 / 65 | 13/18 | 0/18 |
| testset breadth blind b2b | 5 | 0 | 54.5 % | +95.1 % | 54.5 % | +95.1 % | 59.3 % / +130.0 % | 0 | -0.10 | 0.00 |  | no (+35.7 %) | 4 | 1 / 4 | -0.10 / 4 |  |  |
| testset breadth blind graph | 5 | 0 | 59.7 % | +108.9 % | 59.7 % | +108.9 % | 64.5 % / +135.4 % | 0 | -0.10 | 0.00 |  | no (+31.8 %) | 5 | 1 / 4 | -0.10 / 5 |  |  |
| testset hgemm 2560x2048x1536 | 1 | 0 | 7.7 % | +7.7 % | 7.7 % | +7.7 % | 2.6 % / -2.6 % | 1 |  |  |  |  |  |  |  |  |  |
| testset stack_path | 6 | 0 | 16.1 % | -76.1 % | 16.1 % | -76.1 % | 85.5 % / -92.5 % | 2 | 0.77 | 0.60 |  | yes | 3 | 0 / 3 | 0.77 / 3 | 0/6 | 0/6 |
| testset wr_rmw | 4 | 0 | 28.0 % | +52.1 % | 28.0 % | +52.1 % | 13.7 % / +48.4 % | 0 | 1.00 | 1.00 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 0 | 0 / 0 | 1.00 / 0 | 4/4 | 0/4 |
| testset wr_footprint | 2 | 0 | 16.3 % | -27.7 % | 16.3 % | -27.7 % | 284.6 % / +336.7 % | 1 |  |  |  |  |  |  |  | 0/2 | 0/2 |
| testset whatif decode rotating b2b | 5 | 0 | 2.2 % | -4.1 % | 2.2 % | -4.1 % | 2.8 % / -7.2 % | 5 |  |  |  |  |  |  |  |  |  |
| testset whatif decode rotating graph | 5 | 0 | 3.6 % | -4.2 % | 3.6 % | -4.2 % | 1.7 % / -8.1 % | 5 |  |  |  |  |  |  |  |  |  |
| testset decode 24x3072x96 rotating | 4 | 0 | 0.7 % | -1.5 % | 0.7 % | -1.5 % | 70.5 % / -71.9 % | 4 | 0.80 | 0.67 | 1.00 (1.00–1.00, 1 batch, samples) | yes (+1.2 %) | 0 | 1 / 0 | 0.80 / 0 | 4/4 | 0/4 |
| testset decode 24x3072x96 | 5 | 0 | 4.9 % | -13.4 % | 4.9 % | -13.4 % | 7.3 % / -14.7 % | 4 | 1.00 | 1.00 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 0 | 0 / 0 | 1.00 / 0 | 5/5 | 0/5 |
| testset decode 24x3072x96 b2b | 5 | 0 | 1.7 % | +6.6 % | 1.7 % | +6.6 % | 3.8 % / -4.8 % | 5 | 0.90 | 0.80 |  | yes (+0.2 %) | 0 | 1 / 0 | 0.90 / 0 |  |  |
| testset decode 24x3072x96 graph | 5 | 0 | 3.5 % | -4.6 % | 3.5 % | -4.6 % | 4.6 % / -7.5 % | 5 | 0.90 | 0.80 |  | no (+5.5 %) | 1 | 1 / 0 | 0.90 / 1 |  |  |
| testset fa 1792x10x80 rotating | 7 | 0 | 9.7 % | -23.5 % | 9.7 % | -23.5 % | 15.0 % / -25.0 % | 4 | 0.57 | 0.33 | 1.00 (1.00–1.00, 1 batch, samples) | no (+7.8 %) | 6 | 0 / 7 | 0.57 / 6 | 7/7 | 6/7 |
| testset fa 1792x10x80 | 5 | 0 | 6.6 % | -13.4 % | 6.6 % | -13.4 % | 13.7 % / -20.9 % | 4 | 0.00 | -0.20 | 0.80 (0.80–0.80, 1 batch, samples) | yes | 4 | 1 / 5 | 0.00 / 4 | 5/5 | 4/5 |
| testset gemv 3072x6144 rotating | 9 | 0 | 11.2 % | +17.0 % | 11.2 % | +17.0 % | 8.2 % / +13.8 % | 2 | 0.23 | 0.23 | 0.67 (0.67–0.67, 1 batch, samples) | no (+4.9 %) | 12 | 11 / 2 | 0.23 / 12 | 9/9 | 0/9 |
| testset gemv 3072x6144 | 9 | 0 | 3.8 % | -19.0 % | 3.8 % | -19.0 % | 6.4 % / -20.7 % | 7 | 0.33 | 0.24 | 0.71 (0.71–0.71, 1 batch, samples) | yes (+0.2 %) | 8 | 7 / 5 | 0.33 / 8 | 9/9 | 0/9 |
| testset gemv 3072x6144 b2b | 9 | 0 | 1.9 % | -13.5 % | 1.9 % | -13.5 % | 3.8 % / -14.7 % | 8 | 0.43 | 0.33 |  | yes (+0.0 %) | 6 | 6 / 6 | 0.43 / 6 |  |  |
| testset gemv 3072x6144 graph | 9 | 0 | 4.7 % | -12.4 % | 4.7 % | -12.4 % | 8.5 % / -13.8 % | 8 | 0.34 | 0.20 |  | no (+4.1 %) | 9 | 6 / 8 | 0.34 / 9 |  |  |
| testset hgemm 1792x2560x2304 | 1 | 0 | 0.5 % | +0.5 % | 0.5 % | +0.5 % | 6.4 % / -6.4 % | 1 |  |  |  |  |  |  |  | 1/1 | 1/1 |
| testset hgemm 1792x2560x2304 rotating | 2 | 0 | 7.8 % | +13.6 % | 7.8 % | +13.6 % | 2.3 % / +3.5 % | 1 |  |  |  |  |  |  |  | 2/2 | 1/2 |
| testset softmax 768x3072 | 2 | 0 | 16.8 % | -24.2 % | 16.8 % | -24.2 % | 18.1 % / -25.5 % | 1 |  |  |  |  |  |  |  | 2/2 | 0/2 |
| testset softmax 768x3072 b2b | 2 | 0 | 18.4 % | -22.0 % | 18.4 % | -22.0 % | 19.7 % / -23.3 % | 0 |  |  |  |  |  |  |  |  |  |
| testset softmax 768x3072 graph | 2 | 0 | 20.2 % | -23.6 % | 20.2 % | -23.6 % | 21.9 % / -25.3 % | 0 |  |  |  |  |  |  |  |  |  |
| testset hash_sweep | 4 | 0 | 6.2 % | +37.0 % | 6.2 % | +37.0 % | 12.6 % / -32.3 % | 3 | 1.00 | 1.00 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 0 | 0 / 0 | 1.00 / 0 | 1/4 | 0/4 |
| testset record_sweep | 4 | 0 | 15.3 % | -66.8 % | 15.3 % | -66.8 % | 32.1 % / -42.5 % | 0 | 1.00 | 1.00 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 0 | 0 / 0 | 1.00 / 0 | 0/4 | 0/4 |
| testset tile trace | 2 | 0 | 3.8 % | +5.5 % | 3.8 % | +5.5 % | 5.8 % / -9.8 % | 2 |  |  |  |  |  |  |  | 2/2 | 2/2 |
| testset decode 40x2560x112 rotating | 2 | 0 | 14.5 % | -22.9 % | 14.5 % | -22.9 % | 23.7 % / -44.1 % | 1 |  |  |  |  |  |  |  | 2/2 | 0/2 |
| testset decode 40x2560x112 | 2 | 0 | 9.2 % | -17.8 % | 9.2 % | -17.8 % | 16.9 % / -22.2 % | 1 |  |  |  |  |  |  |  | 2/2 | 0/2 |
| testset decode 40x2560x112 b2b | 2 | 0 | 7.6 % | -14.1 % | 7.6 % | -14.1 % | 14.4 % / -18.7 % | 1 |  |  |  |  |  |  |  |  |  |
| testset decode 40x2560x112 graph | 2 | 0 | 10.4 % | -16.3 % | 10.4 % | -16.3 % | 18.9 % / -21.5 % | 1 |  |  |  |  |  |  |  |  |  |
| testset gemv 5120x2560 rotating | 4 | 0 | 9.2 % | +25.5 % | 9.2 % | +25.5 % | 316.2 % / +707.1 % | 2 | -1.00 | -1.00 | 0.33 (0.33–0.33, 1 batch, samples) | no (+12.7 %) | 6 | 5 / 1 | -1.00 / 6 | 4/4 | 0/4 |
| testset gemv 5120x2560 | 2 | 0 | 3.6 % | -6.2 % | 3.6 % | -6.2 % | 133.8 % / +135.0 % | 2 |  |  |  |  |  |  |  | 2/2 | 0/2 |
| testset gemv 5120x2560 b2b | 2 | 0 | 2.5 % | -3.4 % | 2.5 % | -3.4 % | 141.1 % / +142.1 % | 2 |  |  |  |  |  |  |  |  |  |
| testset gemv 5120x2560 graph | 2 | 0 | 5.2 % | -5.2 % | 5.2 % | -5.2 % | 159.8 % / +166.9 % | 2 |  |  |  |  |  |  |  |  |  |
| testset hgemm 2304x1792x1792 rotating | 2 | 0 | 9.9 % | +18.8 % | 9.9 % | +18.8 % | 6.0 % / +6.7 % | 1 |  |  |  |  |  |  |  | 2/2 | 2/2 |
| testset stage_question | 5 | 0 | 5.8 % | +17.8 % | 5.8 % | +17.8 % | 9.8 % / -12.2 % | 3 | 1.00 | 1.00 |  | yes | 0 | 0 / 0 | 1.00 / 0 | 3/5 | 4/5 |
| testset rmsnorm 1280x2560 rotating | 6 | 0 | 7.0 % | +16.3 % | 7.0 % | +16.3 % | 4.0 % / +9.3 % | 5 | -0.49 | -0.33 | 0.41 (0.41–0.41, 1 batch, samples) | no (+3.5 %) | 8 | 6 / 4 | -0.49 / 8 | 6/6 | 0/6 |
| testset rmsnorm 1280x2560 | 6 | 0 | 18.5 % | -23.6 % | 18.5 % | -23.6 % | 21.4 % / -37.8 % | 0 | 0.84 | 0.69 | 0.87 (0.87–0.87, 1 batch, samples) | no (+2.0 %) | 1 | 2 / 0 | 0.84 / 1 | 6/6 | 0/6 |
| testset rmsnorm 1280x2560 b2b | 6 | 0 | 12.5 % | -18.4 % | 12.5 % | -18.4 % | 14.4 % / -33.6 % | 0 | 0.83 | 0.73 |  | yes | 0 | 2 / 0 | 0.83 / 0 |  |  |
| testset rmsnorm 1280x2560 graph | 6 | 0 | 13.0 % | -14.9 % | 13.0 % | -14.9 % | 15.4 % / -33.8 % | 2 | 0.37 | 0.33 |  | yes (+1.0 %) | 2 | 2 / 3 | 0.37 / 2 |  |  |
| testset softmax 1280x2560 | 6 | 0 | 23.1 % | -28.8 % | 23.1 % | -28.8 % | 25.4 % / -30.3 % | 1 | 1.00 | 1.00 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 0 | 0 / 0 | 1.00 / 0 | 6/6 | 1/6 |
| testset softmax 1280x2560 b2b | 5 | 0 | 16.9 % | -22.5 % | 16.9 % | -22.5 % | 19.7 % / -24.2 % | 1 | 1.00 | 1.00 |  | yes | 0 | 0 / 0 | 1.00 / 0 |  |  |
| testset softmax 1280x2560 graph | 5 | 0 | 21.5 % | -27.4 % | 21.5 % | -27.4 % | 24.6 % / -29.3 % | 1 | 1.00 | 1.00 |  | yes | 0 | 0 / 0 | 1.00 / 0 |  |  |
| testset stack_load | 3 | 0 | 98.6 % | +120.7 % | 98.6 % | +120.7 % | 61.9 % / -71.6 % | 1 | 1.00 | 1.00 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 0 | 0 / 0 | 1.00 / 0 | 0/3 | 0/3 |
| testset occupancy | 5 | 0 | 15.1 % | +24.3 % | 15.1 % | +24.3 % | 13.0 % / +22.1 % | 2 | 1.00 | 1.00 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 0 | 0 / 0 | 1.00 / 0 | 5/5 | 0/5 |
| testset turnover | 2 | 0 | 10.7 % | -13.8 % | 10.7 % | -13.8 % | 17.9 % / -18.4 % | 1 |  |  |  |  |  |  |  | 0/2 | 1/2 |
| testset decode 36x2304x144 rotating | 6 | 0 | 17.0 % | -25.6 % | 17.0 % | -25.6 % | 19.4 % / -28.0 % | 2 | -0.26 | -0.20 | 0.73 (0.73–0.73, 1 batch, samples) | no (+18.9 %) | 5 | 4 / 5 | -0.26 / 5 | 6/6 | 0/6 |
| testset decode 36x2304x144 | 6 | 0 | 14.1 % | -25.1 % | 14.1 % | -25.1 % | 18.3 % / -27.5 % | 2 | 0.37 | 0.33 | 1.00 (1.00–1.00, 1 batch, samples) | yes (+0.5 %) | 4 | 2 / 3 | 0.37 / 4 | 6/6 | 3/6 |
| testset decode 36x2304x144 b2b | 5 | 0 | 11.5 % | -22.1 % | 11.5 % | -22.1 % | 15.7 % / -25.1 % | 2 | 0.20 | 0.20 |  | yes | 4 | 0 / 4 | 0.20 / 4 |  |  |
| testset decode 36x2304x144 graph | 5 | 0 | 15.2 % | -23.0 % | 15.2 % | -23.0 % | 19.7 % / -26.4 % | 2 | -0.10 | 0.00 |  | yes (+1.1 %) | 4 | 1 / 4 | -0.10 / 4 |  |  |
| testset dram_limit | 4 | 0 | 10.5 % | +11.6 % | 10.5 % | +11.6 % | 27.0 % / -43.3 % | 2 | 1.00 | 1.00 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 0 | 0 / 0 | 1.00 / 0 | 4/4 | 0/4 |
| testset fa 2304x6x144 rotating | 5 | 0 | 12.5 % | -25.8 % | 12.5 % | -25.8 % | 18.8 % / -28.0 % | 1 | 1.00 | 1.00 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 0 | 0 / 0 | 1.00 / 0 | 5/5 | 3/5 |
| testset fa 2304x6x144 | 5 | 0 | 13.1 % | -17.3 % | 13.1 % | -17.3 % | 19.2 % / -19.7 % | 1 | 1.00 | 1.00 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 0 | 0 / 0 | 1.00 / 0 | 5/5 | 4/5 |
| testset fa_stride | 3 | 0 | 7.2 % | -16.9 % | 7.2 % | -16.9 % | 5.2 % / -19.6 % | 2 | 0.50 | 0.33 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 1 | 0 / 1 | 0.50 / 1 | 3/3 | 0/3 |
| testset gemv 6144x3072 | 3 | 0 | 9.3 % | -12.4 % | 9.3 % | -12.4 % | 10.9 % / -18.2 % | 2 | 0.00 | 0.00 | 1.00 (1.00–1.00, 1 batch, samples) | no (+12.0 %) | 1 | 0 / 1 | 0.00 / 1 | 3/3 | 0/3 |
| testset gemv 6144x3072 b2b | 3 | 0 | 7.5 % | -10.7 % | 7.5 % | -10.7 % | 9.2 % / -16.7 % | 2 | -0.50 | -0.33 |  | no (+12.4 %) | 1 | 1 / 1 | -0.50 / 1 |  |  |
| testset gemv 6144x3072 graph | 3 | 0 | 6.1 % | -9.7 % | 6.1 % | -9.7 % | 8.1 % / -16.6 % | 3 | 0.50 | 0.33 |  | no (+6.7 %) | 1 | 0 / 1 | 0.50 / 1 |  |  |
| testset hgemm 2688x1920x1728 | 8 | 0 | 5.4 % | +12.6 % | 5.4 % | +12.6 % | 5.8 % / -16.2 % | 7 | 0.83 | 0.71 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 3 | 0 / 4 | 0.83 / 3 | 8/8 | 7/8 |
| testset hgemm 2688x1920x1728 rotating | 8 | 0 | 9.4 % | +18.4 % | 9.4 % | +18.4 % | 3.4 % / -10.9 % | 5 | 0.81 | 0.57 | 1.00 (1.00–1.00, 1 batch, samples) | no (+3.5 %) | 4 | 2 / 4 | 0.81 / 4 | 8/8 | 6/8 |
| testset rmsnorm 1920x1792 rotating | 5 | 0 | 5.8 % | +8.6 % | 5.8 % | +8.6 % | 1.7 % / +6.4 % | 5 | -0.20 | -0.20 | 1.00 (1.00–1.00, 1 batch, samples) | no (+3.2 %) | 4 | 5 / 1 | -0.20 / 4 | 5/5 | 0/5 |
| testset rmsnorm 1920x1792 | 5 | 0 | 16.8 % | -34.3 % | 16.8 % | -34.3 % | 18.1 % / -43.0 % | 0 | 0.50 | 0.40 | 0.80 (0.80–0.80, 1 batch, samples) | yes | 3 | 0 / 3 | 0.50 / 3 | 5/5 | 0/5 |
| testset rmsnorm 1920x1792 b2b | 5 | 0 | 15.6 % | -25.0 % | 15.6 % | -25.0 % | 17.0 % / -35.0 % | 2 | 0.30 | 0.20 |  | yes (+2.0 %) | 2 | 0 / 4 | 0.30 / 2 |  |  |
| testset rmsnorm 1920x1792 graph | 5 | 0 | 13.7 % | -22.5 % | 13.7 % | -22.5 % | 15.5 % / -35.2 % | 1 | 0.70 | 0.60 |  | yes | 1 | 1 / 1 | 0.70 / 1 |  |  |
| testset softmax 1920x1792 | 5 | 0 | 21.8 % | -26.3 % | 21.8 % | -26.3 % | 157.8 % / +205.1 % | 0 | 1.00 | 1.00 | 1.00 (1.00–1.00, 1 batch, samples) | yes | 0 | 0 / 0 | 1.00 / 0 | 5/5 | 1/5 |
| testset softmax 1920x1792 b2b | 5 | 0 | 17.0 % | -24.1 % | 17.0 % | -24.1 % | 173.4 % / +214.3 % | 0 | 1.00 | 1.00 |  | yes | 0 | 0 / 0 | 1.00 / 0 |  |  |
| testset softmax 1920x1792 graph | 5 | 0 | 19.0 % | -28.7 % | 19.0 % | -28.7 % | 212.1 % / +258.8 % | 0 | 0.90 | 0.80 |  | yes | 0 | 0 / 1 | 0.90 / 0 |  |  |
| testset write_limit | 3 | 0 | 27.2 % | -29.5 % | 27.2 % | -29.5 % | 172.1 % / +185.7 % | 0 | -0.50 | -0.33 | 0.67 (0.67–0.67, 1 batch, samples) | no (+7.2 %) | 1 | 1 / 1 | -0.50 / 1 | 3/3 | 0/3 |

## Against 28 §1 (targets are the user's; a seen group meeting them is not enough)

| Metric | Target | Now (54 design groups) |
| --- | ---: | --- |
| Pairs measured ≥ 3 % apart, ordered right | ≥ 95 % | 549/670 (81.9 %) |
| True best in the model's top 3 | every group | 43/54 |
| First pick's loss | ≤ 3 % | worst 18.9 %, 15 groups above |
| What-if direction, changes ≥ 3 % | ≥ 95 % | 115/143 (80.4 %) |
| What-if magnitude, changes ≥ 10 % | median ≤ 20–25 % | 33 % over 88 |

Loss above 3 %: gemm1024 full: +12.7 %, testset hgemm 3072x1536x2048 rotating: +3.6 %, testset decode 24x3072x96 graph: +5.5 %, testset fa 1792x10x80 rotating: +7.8 %, testset gemv 3072x6144 rotating: +4.9 %, testset gemv 3072x6144 graph: +4.1 %, testset gemv 5120x2560 rotating: +12.7 %, testset rmsnorm 1280x2560 rotating: +3.5 %, testset decode 36x2304x144 rotating: +18.9 %, testset gemv 6144x3072: +12.0 %, testset gemv 6144x3072 b2b: +12.4 %, testset gemv 6144x3072 graph: +6.7 %, testset hgemm 2688x1920x1728 rotating: +3.5 %, testset rmsnorm 1920x1792 rotating: +3.2 %, testset write_limit: +7.2 %.
Best outside the top 3: testset hgemm2048, testset nvfp4gemm, testset gemv 3072x6144 rotating, testset gemv 3072x6144 graph, testset gemv 5120x2560 rotating, testset rmsnorm 1280x2560 rotating, testset rmsnorm 1280x2560 graph, testset decode 36x2304x144 rotating, testset decode 36x2304x144, testset decode 36x2304x144 graph, testset rmsnorm 1920x1792 rotating.

## What-if: predicted against measured change (28 §1)

One compiled change from a base in the same batch. Direction right: signs agree, or measured within 2 % of zero and predicted within 4 %. Magnitude: |predicted − measured| / |measured| on changes of at least 5 %; target 25 %. First: the prediction committed before the batch. Now: the current engine.

| Axis | n | first: direction | first: median magnitude error | first: within 25 % | now: direction | now: median magnitude error | now: within 25 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 200 | 147/200 | 45 % | 43/122 | 149/200 | 45 % | 42/122 |
| shape | 6 | 5/6 | 17 % | 3/5 | 4/6 | 15 % | 4/5 |
| rotation depth | 4 | 0/4 | 100 % | 0/4 | 2/4 | 100 % | 0/4 |
| cp.async vs direct | 4 | 4/4 | 16 % | 4/4 | 4/4 | 79 % | 0/4 |
| stores removed | 16 | 13/16 | 41 % | 3/8 | 13/16 | 37 % | 3/8 |
| threads | 52 | 37/52 | 80 % | 5/29 | 38/52 | 58 % | 8/29 |
| bytes per lane | 12 | 12/12 | 42 % | 3/8 | 10/12 | 39 % | 4/8 |
| compute removed | 17 | 11/17 | 10 % | 8/11 | 11/17 | 19 % | 8/11 |
| stages | 16 | 14/16 | 75 % | 1/6 | 15/16 | 89 % | 1/6 |
| tile | 14 | 9/14 | 18 % | 5/8 | 11/14 | 49 % | 2/8 |
| tile+threads | 4 | 4/4 | 10 % | 2/3 | 4/4 | 15 % | 2/3 |
| loads removed | 7 | 5/7 | 23 % | 3/5 | 5/7 | 33 % | 1/5 |
| raster | 4 | 4/4 | — | 0/0 | 1/4 | — | 0/0 |
| dequantisation removed | 1 | 1/1 | 4 % | 1/1 | 1/1 | 8 % | 1/1 |
| loads and stores removed | 2 | 2/2 | 24 % | 1/2 | 2/2 | 28 % | 1/2 |
| blocks | 8 | 6/8 | 18 % | 2/4 | 5/8 | 21 % | 2/4 |
| design | 33 | 20/33 | 101 % | 2/24 | 23/33 | 90 % | 5/24 |

Kernels under 30 µs read from their graph rows (50 launches): 54/66 of those changes right, 153/200 over all; the counts above stay the `kernel` rows.

| Change | Base | Axis | measured Δ | first Δ | now Δ | first direction | now direction |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| bl14_xorwide_m1 | bl14_xorwide | shape | -55.3 % | +0.0 % | +0.0 % | WRONG | WRONG |
| bl_aos_f1 | bl_aos | stores removed | -1.2 % | -63.6 % | -63.9 % | WRONG | WRONG |
| bl_mixrwa_noload | bl_mixrwa | loads removed | -4.1 % | +0.1 % | +0.0 % | WRONG | WRONG |
| dec15_bn32_warm | dec15_b_warm | tile | -14.8 % | +13.8 % | +11.4 % | WRONG | WRONG | graph: -13.6 %, +12.7 %, WRONG |
| dec15_split16_rotating | dec15_b_rotating | blocks | +10.0 % | -2.5 % | -2.1 % | WRONG | WRONG |
| dec15_split4_bn32_warm | dec15_b_warm | design | -22.7 % | +2.8 % | +0.5 % | WRONG | WRONG | graph: -23.5 %, +0.6 %, WRONG |
| fa13_c64x128_s2_t128_warm | fa13_c64x64_s2_t128_warm | tile | +0.5 % | -7.9 % | -4.3 % | WRONG | WRONG |
| fa13_c64x32_s3_t128_rotating | fa13_c64x64_s2_t128_rotating | design | -7.8 % | +7.9 % | +2.4 % | WRONG | WRONG |
| fa13_c64x32_s3_t128_warm | fa13_c64x64_s2_t128_warm | design | -9.6 % | +6.6 % | +1.4 % | WRONG | WRONG |
| fa13_noexp_rotating | fa13_c64x64_s2_t128_rotating | compute removed | +4.3 % | -7.9 % | -7.5 % | WRONG | WRONG |
| fa13_noexp_warm | fa13_c64x64_s2_t128_warm | compute removed | +5.6 % | -8.0 % | -7.8 % | WRONG | WRONG |
| fa13_nov_rotating | fa13_c64x64_s2_t128_rotating | compute removed | -0.8 % | -29.0 % | -36.9 % | WRONG | WRONG |
| gemv13_np2_rotating | gemv13_b_rotating | threads | -3.8 % | +0.0 % | +0.0 % | WRONG | WRONG |
| gemv13_np8_r16_warm | gemv13_b_warm | design | +0.2 % | +4.8 % | +4.0 % | WRONG | WRONG | graph: +1.6 %, +4.7 %, WRONG |
| gemv13_np8_rotating | gemv13_b_rotating | threads | -4.9 % | +0.0 % | +0.0 % | WRONG | WRONG |
| gemv13_r16_warm | gemv13_b_warm | threads | +0.2 % | +4.5 % | +4.2 % | WRONG | WRONG | graph: +1.5 %, +4.8 %, WRONG |
| gemv13_r64_warm | gemv13_b_warm | threads | +12.0 % | -0.1 % | -1.2 % | WRONG | WRONG | graph: +6.0 %, -1.4 %, WRONG |
| gemv14_np2_r64_rotating | gemv14_b_rotating | design | -11.2 % | +75.2 % | +12.4 % | WRONG | WRONG |
| hgemm_m1792n2560k2304_t256_rotating | hgemm_m1792n2560k2304_c128x128x32_s3_t128_rotating | threads | -1.6 % | -6.6 % | -4.3 % | WRONG | WRONG |
| hgemm_m2688n1920k1728_c128x128x32_s2_t128_rotating | hgemm_m2688n1920k1728_c128x128x32_s3_t128_rotating | design | -2.7 % | +6.6 % | +0.0 % | WRONG | WRONG |
| hgemm_m2688n1920k1728_c128x128x32_s3_t128_sw8_rotating | hgemm_m2688n1920k1728_c128x128x32_s3_t128_rotating | design | -3.4 % | +3.8 % | +2.2 % | WRONG | WRONG |
| hgemm_m2688n1920k1728_c64x128x32_s4_t128 | hgemm_m2688n1920k1728_c128x128x32_s3_t128 | design | +5.5 % | -0.4 % | -5.4 % | WRONG | WRONG |
| rms14_t256_rotating | rms14_t128_rotating | threads | -3.5 % | +0.5 % | +1.2 % | WRONG | WRONG |
| rms14_t256_warm | rms14_t128_warm | threads | +2.0 % | -7.4 % | -3.2 % | WRONG | WRONG | graph: +0.4 %, -4.1 %, WRONG |
| rms14_t512_rotating | rms14_t128_rotating | threads | -6.5 % | +0.7 % | +4.1 % | WRONG | WRONG |
| rms15_t16_rotating | rms15_t128_rotating | threads | +5.4 % | -1.1 % | -1.1 % | WRONG | WRONG |
| rms15_t32_rotating | rms15_t128_rotating | threads | +3.2 % | -1.3 % | -1.3 % | WRONG | WRONG |
| rms15_t64_warm | rms15_t128_warm | threads | +0.2 % | +7.3 % | +7.3 % | WRONG | WRONG | graph: +7.2 %, +9.2 %, right |
| rmsb_m192_nocompute | rmsb_m192_b | compute removed | -0.7 % | -9.2 % | -15.1 % | WRONG | WRONG | graph: -27.4 %, -28.7 %, right |
| rmsb_m192_noload | rmsb_m192_b | loads removed | +0.7 % | -5.8 % | -13.6 % | WRONG | WRONG | graph: -18.8 %, -22.9 %, right |
| rmsb_m192_nostore | rmsb_m192_b | stores removed | -1.4 % | -9.2 % | -15.1 % | WRONG | WRONG | graph: -27.5 %, -37.3 %, right |
| rmsb_m48_nostore | rmsb_m48_b | stores removed | +0.0 % | -45.0 % | -45.0 % | WRONG | WRONG | graph: -0.3 %, -16.3 %, WRONG |
| wi12_fa_noexp_warm | wi12_fa_b_warm | compute removed | +11.7 % | -3.9 % | -3.6 % | WRONG | WRONG |
| wi12_rms_nocompute_rotating | wi12_rms_b_rotating | compute removed | -1.7 % | -9.0 % | -9.0 % | WRONG | WRONG |
| wi12b_dec_bn32_warm | wi12b_dec_b_warm | tile | +0.9 % | +7.8 % | +4.6 % | WRONG | WRONG | graph: +5.5 %, +5.9 %, right |
| wi12b_dec_split4_rotating | wi12b_dec_b_rotating | blocks | +4.4 % | -0.1 % | -4.0 % | WRONG | WRONG | graph: +3.5 %, -4.4 %, WRONG |
| wi_dec_rot2_rotating | wi_dec_b_rotating | rotation depth | +16.4 % | +0.0 % | -0.0 % | WRONG | WRONG |
| wi_dec_rot4_rotating | wi_dec_b_rotating | rotation depth | +14.9 % | +0.0 % | -0.0 % | WRONG | WRONG |
| wi_gemv_r16_rotating | wi_gemv_b_rotating | threads | +8.0 % | -84.2 % | -0.3 % | WRONG | WRONG |
| bl_fanout_s16 | bl_fanout | shape | -2.4 % | -59.7 % | +2.7 % | right | WRONG |
| dec14_split4_bn32_warm | dec14_b_warm | design | -12.4 % | -0.4 % | +6.0 % | right | WRONG | graph: -6.5 %, +6.8 %, WRONG |
| fa13_c64x64_s3_t128_rotating | fa13_c64x64_s2_t128_rotating | stages | -4.0 % | -1.9 % | +0.6 % | right | WRONG |
| gemv14_mk16_rotating | gemv14_b_rotating | bytes per lane | -6.0 % | -2.7 % | +10.3 % | right | WRONG |
| gemv14_mk16_warm | gemv14_b_warm | bytes per lane | +0.3 % | -0.8 % | -4.9 % | right | WRONG | graph: -6.1 %, -6.0 %, right |
| gemv14_r16_rotating | gemv14_b_rotating | threads | -2.7 % | -76.4 % | +0.2 % | right | WRONG |
| hgemm_m2688n1920k1728_c128x128x64_s2_t128 | hgemm_m2688n1920k1728_c128x128x32_s3_t128 | design | +1.3 % | -2.7 % | +5.9 % | right | WRONG |
| wi12b_dec_split16_rotating | wi12b_dec_b_rotating | blocks | +4.6 % | +1.8 % | -2.0 % | right | WRONG | graph: +3.4 %, -2.2 %, WRONG |
| wi_hg_sw4_rotating | wi_hg_b_rotating | raster | -2.3 % | -0.0 % | +2.2 % | right | WRONG |
| wi_hg_sw4_warm | wi_hg_b_warm | raster | +1.4 % | -1.3 % | -9.0 % | right | WRONG |
| wi_hg_sw8_warm | wi_hg_b_warm | raster | +0.5 % | -1.3 % | -9.0 % | right | WRONG |
| wi_rms_t256_rotating | wi_rms_b_rotating | threads | -2.0 % | -0.1 % | +1.1 % | right | WRONG |
| bl13_transpose_t128 | bl13_transpose | threads | -1.9 % | +245.2 % | +0.0 % | WRONG | right |
| dec13_bn32_rotating | dec13_b_rotating | tile | -1.7 % | +7.2 % | -1.1 % | WRONG | right |
| dec13_split4_bn32_rotating | dec13_b_rotating | design | -0.5 % | +71.3 % | -1.4 % | WRONG | right |
| dec14_split4_bn32_rotating | dec14_b_rotating | design | -18.3 % | +51.0 % | -0.6 % | WRONG | right |
| fa15_c64x64_s1_t128_rotating | fa15_c64x64_s2_t128_rotating | stages | +1.9 % | +5.1 % | +1.7 % | WRONG | right |
| gemv15_np2_r64_warm | gemv15_b_warm | design | +0.0 % | -8.2 % | -3.4 % | WRONG | right | graph: -0.2 %, -3.9 %, right |
| hgemm_m2688n1920k1728_c128x128x32_s3_t128_sw8 | hgemm_m2688n1920k1728_c128x128x32_s3_t128 | design | -0.4 % | -14.6 % | -3.0 % | WRONG | right |
| hgemm_m2688n1920k1728_c128x256x32_s3_t256 | hgemm_m2688n1920k1728_c128x128x32_s3_t128 | design | +17.7 % | -0.9 % | +15.9 % | WRONG | right |
| rms15_t256_warm | rms15_t128_warm | threads | +36.3 % | -5.1 % | +7.6 % | WRONG | right | graph: +22.0 %, +9.6 %, right |
| sm15_t256_warm | sm15_t128_warm | threads | +31.7 % | -0.1 % | +29.4 % | WRONG | right | graph: +30.8 %, +37.2 %, right |
| wi12b_dec_bn32_rotating | wi12b_dec_b_rotating | tile | -3.8 % | +0.3 % | -3.6 % | WRONG | right | graph: -4.0 %, -3.9 %, right |
| wi_hg_S_s1_rotating | wi_hg_S_s4_rotating | stages | -4.9 % | +4.1 % | -9.7 % | WRONG | right |
| wi_rms_rot2_rotating | wi_rms_b_rotating | rotation depth | +5.7 % | +0.0 % | +0.1 % | WRONG | right |
| wi_rms_rot4_rotating | wi_rms_b_rotating | rotation depth | +5.5 % | +0.0 % | +0.1 % | WRONG | right |
| bl13_swaprows_l1 | bl13_swaprows | loads removed | -8.6 % | -0.3 % | -0.2 % | right | right | graph: -10.6 %, -0.3 %, right |
| bl13_ticket_noret | bl13_ticket | stores removed | -32.3 % | -18.6 % | -74.8 % | right | right | graph: -30.6 %, -78.4 %, right |
| bl14_exchring_t128 | bl14_exchring | threads | -0.5 % | -1.3 % | -1.3 % | right | right |
| bl_mixrwa_noatom | bl_mixrwa | compute removed | -84.0 % | -92.5 % | -92.6 % | right | right |
| bl_mixrwa_nostore | bl_mixrwa | stores removed | -4.3 % | -1.2 % | -0.6 % | right | right |
| dec13_bn32_warm | dec13_b_warm | tile | +16.8 % | +7.2 % | +6.6 % | right | right | graph: +6.4 %, +8.4 %, right |
| dec13_s1_warm | dec13_b_warm | stages | +0.6 % | -1.3 % | +2.2 % | right | right | graph: -5.2 %, +2.8 %, WRONG |
| dec13_s3_rotating | dec13_b_rotating | stages | +2.4 % | +3.3 % | +0.9 % | right | right |
| dec13_split16_warm | dec13_b_warm | blocks | +19.8 % | +19.6 % | +20.0 % | right | right | graph: +25.7 %, +25.6 %, right |
| dec13_split4_bn32_warm | dec13_b_warm | design | +57.8 % | +71.3 % | +73.0 % | right | right | graph: +79.4 %, +93.3 %, right |
| dec15_bn32_rotating | dec15_b_rotating | tile | -14.8 % | -2.9 % | -2.1 % | right | right |
| dec15_s1_rotating | dec15_b_rotating | stages | +1.2 % | -0.3 % | -0.3 % | right | right |
| dec15_s1_warm | dec15_b_warm | stages | -12.9 % | -1.4 % | -0.6 % | right | right | graph: -9.8 %, -0.6 %, right |
| dec15_split16_warm | dec15_b_warm | blocks | +44.9 % | +42.5 % | +41.7 % | right | right |
| dec15_split4_bn32_rotating | dec15_b_rotating | design | -14.7 % | -3.1 % | -2.6 % | right | right |
| dec15_split4_rotating | dec15_b_rotating | blocks | +1.3 % | -2.9 % | -3.0 % | right | right |
| dec15_split4_warm | dec15_b_warm | blocks | -22.4 % | -15.7 % | -14.7 % | right | right | graph: -22.6 %, -16.4 %, right |
| fa13_c128x64_s2_t128_rotating | fa13_c64x64_s2_t128_rotating | tile | -13.6 % | -11.0 % | -11.3 % | right | right |
| fa13_c128x64_s2_t256_rotating | fa13_c64x64_s2_t128_rotating | design | -16.9 % | -9.5 % | -12.0 % | right | right |
| fa13_c128x64_s2_t256_warm | fa13_c64x64_s2_t128_warm | design | -21.4 % | -11.2 % | -13.1 % | right | right |
| fa13_c64x128_s2_t128_rotating | fa13_c64x64_s2_t128_rotating | tile | -2.3 % | -7.8 % | -3.8 % | right | right |
| fa13_c64x64_s1_t128_rotating | fa13_c64x64_s2_t128_rotating | stages | -10.4 % | -19.2 % | -22.1 % | right | right |
| fa13_c64x64_s3_t128_warm | fa13_c64x64_s2_t128_warm | stages | -1.8 % | -1.9 % | +0.9 % | right | right |
| fa13_nostore_rotating | fa13_c64x64_s2_t128_rotating | stores removed | -5.6 % | -1.8 % | -2.1 % | right | right |
| fa13_nostore_warm | fa13_c64x64_s2_t128_warm | stores removed | -3.6 % | -1.8 % | -1.8 % | right | right |
| fa15_c64x32_s3_t128_rotating | fa15_c64x64_s2_t128_rotating | design | +9.2 % | +22.4 % | +19.3 % | right | right |
| fa15_c64x32_s3_t128_warm | fa15_c64x64_s2_t128_warm | design | +7.8 % | +22.9 % | +19.8 % | right | right |
| fa15_c64x64_s1_t128_warm | fa15_c64x64_s2_t128_warm | stages | -1.3 % | -2.2 % | -2.1 % | right | right |
| gemv13_mk16_rotating | gemv13_b_rotating | bytes per lane | +0.0 % | +0.0 % | +0.0 % | right | right |
| gemv13_mk16_warm | gemv13_b_warm | bytes per lane | +21.9 % | +3.2 % | +2.6 % | right | right | graph: +1.9 %, +3.1 %, right |
| gemv13_mk4_rotating | gemv13_b_rotating | bytes per lane | -0.3 % | -0.1 % | -0.1 % | right | right |
| gemv13_mk4_warm | gemv13_b_warm | bytes per lane | +12.4 % | +10.4 % | +14.9 % | right | right | graph: +15.9 %, +17.3 %, right |
| gemv13_nocompute_rotating | gemv13_b_rotating | compute removed | -7.6 % | -5.6 % | -5.7 % | right | right |
| gemv13_nocompute_warm | gemv13_b_warm | compute removed | -36.3 % | -44.4 % | -44.0 % | right | right | graph: -39.8 %, -51.0 %, right |
| gemv13_np2_r64_rotating | gemv13_b_rotating | design | +6.4 % | +0.1 % | +5.6 % | right | right |
| gemv13_np2_r64_warm | gemv13_b_warm | design | +0.2 % | -0.2 % | -2.0 % | right | right | graph: +1.0 %, -2.3 %, right |
| gemv13_np2_warm | gemv13_b_warm | threads | +1.3 % | +1.2 % | +0.4 % | right | right | graph: -3.0 %, +0.4 %, WRONG |
| gemv13_np8_r16_rotating | gemv13_b_rotating | design | +0.6 % | +0.1 % | +0.0 % | right | right |
| gemv13_np8_warm | gemv13_b_warm | threads | +0.0 % | +0.6 % | +0.0 % | right | right | graph: +1.6 %, +0.1 %, right |
| gemv13_r16_rotating | gemv13_b_rotating | threads | +12.3 % | +0.1 % | +0.0 % | right | right |
| gemv13_r64_rotating | gemv13_b_rotating | threads | +6.9 % | +0.1 % | +5.6 % | right | right |
| gemv15_mk16_warm | gemv15_b_warm | bytes per lane | -10.7 % | -2.8 % | -2.8 % | right | right | graph: -6.4 %, -3.2 %, right |
| hgemm_m1792n2560k2304_c128x128x32_s4_t128_rotating | hgemm_m1792n2560k2304_c128x128x32_s3_t128_rotating | stages | +14.5 % | +19.9 % | +27.5 % | right | right |
| hgemm_m1792n2560k2304_direct | hgemm_m1792n2560k2304_c128x128x32_s3_t128 | cp.async vs direct | +20.9 % | +24.3 % | +7.3 % | right | right |
| hgemm_m1792n2560k2304_noload | hgemm_m1792n2560k2304_c128x128x32_s3_t128 | loads removed | -16.1 % | -16.8 % | -20.5 % | right | right |
| hgemm_m1792n2560k2304_s1 | hgemm_m1792n2560k2304_c128x128x32_s3_t128 | stages | -1.6 % | -3.7 % | -3.4 % | right | right |
| hgemm_m1792n2560k2304_t64 | hgemm_m1792n2560k2304_c128x128x32_s3_t128 | threads | +232.8 % | +169.7 % | +161.3 % | right | right |
| hgemm_m1792n2560k2304_t64_rotating | hgemm_m1792n2560k2304_c128x128x32_s3_t128_rotating | threads | +223.1 % | +151.0 % | +156.9 % | right | right |
| hgemm_m2304n1792k1792_c128x128x64_s2_t128_rotating | hgemm_m2304n1792k1792_c128x128x32_s3_t128_rotating | design | +6.9 % | +20.6 % | +25.9 % | right | right |
| hgemm_m2560n2048k1536_c128x128x32_s3_t128_nocompute | hgemm_m2560n2048k1536_c128x128x32_s3_t128 | compute removed | -36.8 % | -36.3 % | -41.6 % | right | right |
| hgemm_m2560n2048k1536_c128x128x32_s3_t128_noload | hgemm_m2560n2048k1536_c128x128x32_s3_t128 | loads removed | -10.9 % | -15.1 % | -29.6 % | right | right |
| hgemm_m2560n2048k1536_c128x128x32_s3_t128_nostore | hgemm_m2560n2048k1536_c128x128x32_s3_t128 | stores removed | -10.5 % | -12.9 % | -14.5 % | right | right |
| hgemm_m2688n1920k1728_c128x128x32_s2_t128 | hgemm_m2688n1920k1728_c128x128x32_s3_t128 | design | -5.9 % | -6.6 % | -14.1 % | right | right |
| hgemm_m2688n1920k1728_c128x128x32_s4_t128 | hgemm_m2688n1920k1728_c128x128x32_s3_t128 | design | +11.3 % | +5.8 % | +9.1 % | right | right |
| hgemm_m2688n1920k1728_c128x128x32_s4_t128_rotating | hgemm_m2688n1920k1728_c128x128x32_s3_t128_rotating | design | +12.3 % | +24.1 % | +20.7 % | right | right |
| hgemm_m2688n1920k1728_c128x128x64_s2_t128_rotating | hgemm_m2688n1920k1728_c128x128x32_s3_t128_rotating | design | +6.2 % | +20.8 % | +18.7 % | right | right |
| hgemm_m2688n1920k1728_c128x256x32_s3_t256_rotating | hgemm_m2688n1920k1728_c128x128x32_s3_t128_rotating | design | +14.9 % | +8.8 % | +19.3 % | right | right |
| hgemm_m2688n1920k1728_c256x128x32_s2_t256 | hgemm_m2688n1920k1728_c128x128x32_s3_t128 | design | +19.8 % | +13.0 % | +13.6 % | right | right |
| hgemm_m2688n1920k1728_c256x128x32_s2_t256_rotating | hgemm_m2688n1920k1728_c128x128x32_s3_t128_rotating | design | +20.1 % | +25.3 % | +19.6 % | right | right |
| hgemm_m2688n1920k1728_c64x128x32_s4_t128_rotating | hgemm_m2688n1920k1728_c128x128x32_s3_t128_rotating | design | +8.4 % | +17.4 % | +7.1 % | right | right |
| rms14_t16_rotating | rms14_t128_rotating | threads | -0.2 % | -1.4 % | -1.4 % | right | right |
| rms14_t16_warm | rms14_t128_warm | threads | +9.2 % | +0.5 % | +0.5 % | right | right | graph: +5.4 %, +0.6 %, right |
| rms14_t32_rotating | rms14_t128_rotating | threads | -3.2 % | -2.0 % | -2.0 % | right | right |
| rms14_t32_warm | rms14_t128_warm | threads | +7.9 % | +2.3 % | +2.3 % | right | right | graph: +6.9 %, +2.9 %, right |
| rms14_t512_warm | rms14_t128_warm | threads | +40.9 % | +3.7 % | +24.6 % | right | right | graph: +39.5 %, +30.9 %, right |
| rms14_t64_rotating | rms14_t128_rotating | threads | -3.7 % | -0.1 % | -0.1 % | right | right |
| rms14_t64_warm | rms14_t128_warm | threads | +9.2 % | +2.7 % | +2.7 % | right | right | graph: -0.6 %, +3.4 %, right |
| rms15_t16_warm | rms15_t128_warm | threads | +18.7 % | +1.0 % | +1.0 % | right | right | graph: +3.0 %, +1.2 %, right |
| rms15_t256_rotating | rms15_t128_rotating | threads | +4.8 % | +0.1 % | +2.1 % | right | right |
| rms15_t32_warm | rms15_t128_warm | threads | +7.0 % | +8.2 % | +8.2 % | right | right | graph: +5.7 %, +10.3 %, right |
| rms15_t64_rotating | rms15_t128_rotating | threads | +0.5 % | -0.0 % | -0.1 % | right | right |
| rmsb_m192_arith | rmsb_m192_b | loads and stores removed | -36.1 % | -50.1 % | -53.3 % | right | right | graph: -59.9 %, -55.5 %, right |
| rmsb_m48_arith | rmsb_m48_b | loads and stores removed | -41.4 % | -45.0 % | -45.0 % | right | right | graph: -55.8 %, -44.2 %, right |
| rmsb_m48_nocompute | rmsb_m48_b | compute removed | -47.6 % | -45.0 % | -45.0 % | right | right | graph: -40.8 %, -32.6 %, right |
| rmsb_m48_noload | rmsb_m48_b | loads removed | -36.6 % | -45.0 % | -45.0 % | right | right | graph: -33.6 %, -23.9 %, right |
| sm13_t64_warm | sm13_t128_warm | threads | +31.8 % | +9.9 % | +10.2 % | right | right | graph: +23.9 %, +13.7 %, right |
| sm14_t16_warm | sm14_t128_warm | threads | +124.8 % | +132.5 % | +133.6 % | right | right |
| sm14_t256_warm | sm14_t128_warm | threads | +1.7 % | -1.1 % | +3.3 % | right | right | graph: +10.5 %, +4.2 %, right |
| sm14_t32_warm | sm14_t128_warm | threads | +23.6 % | +24.3 % | +24.5 % | right | right | graph: +40.8 %, +31.1 %, right |
| sm14_t512_warm | sm14_t128_warm | threads | +54.6 % | +64.4 % | +86.8 % | right | right | graph: +86.8 %, +110.4 %, right |
| sm14_t64_warm | sm14_t128_warm | threads | +15.8 % | +8.4 % | +8.5 % | right | right | graph: +27.0 %, +10.8 %, right |
| sm15_t16_warm | sm15_t128_warm | threads | +83.4 % | +8.6 % | +64.4 % | right | right | graph: +106.1 %, +81.6 %, right |
| sm15_t32_warm | sm15_t128_warm | threads | +23.9 % | +4.7 % | +17.9 % | right | right | graph: +14.5 %, +22.7 %, right |
| sm15_t64_warm | sm15_t128_warm | threads | +16.1 % | +1.4 % | +5.4 % | right | right | graph: +16.7 %, +6.8 %, right |
| smb_m192_nostore | smb_m192_b | stores removed | -30.0 % | -17.9 % | -12.3 % | right | right | graph: -22.8 %, -19.6 %, right |
| smb_m48_nostore | smb_m48_b | stores removed | -0.7 % | -0.3 % | -1.3 % | right | right | graph: -10.1 %, -13.9 %, right |
| wi12_fa_bn32_warm | wi12_fa_b_warm | tile | -12.2 % | -11.0 % | -14.1 % | right | right |
| wi12_fa_nostore_warm | wi12_fa_b_warm | stores removed | -0.5 % | -0.9 % | -1.4 % | right | right |
| wi12_fa_s1_warm | wi12_fa_b_warm | stages | -18.3 % | -18.2 % | -20.7 % | right | right |
| wi12_fa_t64_warm | wi12_fa_b_warm | threads | +61.1 % | +78.8 % | +73.6 % | right | right |
| wi12_rms_fp32_rotating | wi12_rms_b_rotating | bytes per lane | +84.0 % | +99.0 % | +99.1 % | right | right |
| wi12_rms_fp32_warm | wi12_rms_b_warm | bytes per lane | +109.3 % | +76.2 % | +85.3 % | right | right | graph: +157.8 %, +113.9 %, right |
| wi12_rms_nostore_rotating | wi12_rms_b_rotating | stores removed | -49.3 % | -57.9 % | -57.4 % | right | right |
| wi12_rms_t64_warm | wi12_rms_b_warm | threads | +2.2 % | +3.3 % | +2.2 % | right | right | graph: +1.1 %, +3.0 %, right |
| wi12b_dec_s1_rotating | wi12b_dec_b_rotating | stages | -3.2 % | -0.9 % | -4.7 % | right | right | graph: -2.8 %, -5.1 %, right |
| wi12b_dec_s1_warm | wi12b_dec_b_warm | stages | -18.7 % | -1.1 % | -2.0 % | right | right | graph: -1.7 %, -2.6 %, right |
| wi12b_dec_split16_warm | wi12b_dec_b_warm | blocks | +3.3 % | +15.3 % | +11.9 % | right | right | graph: +28.0 %, +15.5 %, right |
| wi_dec_S2048_rotating | wi_dec_b_rotating | shape | -38.4 % | -44.8 % | -44.1 % | right | right |
| wi_dec_S2048_warm | wi_dec_b_warm | shape | -37.3 % | -38.1 % | -39.9 % | right | right | graph: -39.8 %, -43.5 %, right |
| wi_dec_S8192_rotating | wi_dec_b_rotating | shape | +91.5 % | +89.8 % | +93.6 % | right | right |
| wi_dec_S8192_warm | wi_dec_b_warm | shape | +319.4 % | +412.6 % | +383.6 % | right | right |
| wi_fa_direct_warm | wi_fa_b_warm | cp.async vs direct | -5.3 % | -6.2 % | -11.6 % | right | right |
| wi_fa_nostore_rotating | wi_fa_b_rotating | stores removed | -1.4 % | -0.5 % | -1.1 % | right | right |
| wi_fa_t64_warm | wi_fa_b_warm | threads | +84.0 % | +94.8 % | +87.3 % | right | right |
| wi_gemv_mk16_rotating | wi_gemv_b_rotating | bytes per lane | -0.8 % | +0.1 % | +0.0 % | right | right |
| wi_gemv_nocompute_rotating | wi_gemv_b_rotating | compute removed | -0.3 % | -3.0 % | -3.0 % | right | right |
| wi_gemv_nocompute_warm | wi_gemv_b_warm | compute removed | -35.1 % | -41.7 % | -41.7 % | right | right | graph: -44.1 %, -49.2 %, right |
| wi_gemv_r64_warm | wi_gemv_b_warm | threads | -10.2 % | -1.3 % | -1.0 % | right | right | graph: -2.1 %, -1.1 %, right |
| wi_hg_bk64_warm | wi_hg_b_warm | tile | +3.7 % | +3.3 % | +10.3 % | right | right |
| wi_hg_bm256_rotating | wi_hg_b_rotating | tile+threads | +3.9 % | +7.4 % | +9.2 % | right | right |
| wi_hg_bm256_warm | wi_hg_b_warm | tile+threads | +5.9 % | +6.5 % | +6.8 % | right | right |
| wi_hg_bm64_rotating | wi_hg_b_rotating | tile | +10.5 % | +10.6 % | +14.5 % | right | right |
| wi_hg_bm64_warm | wi_hg_b_warm | tile | +5.6 % | +6.5 % | +1.1 % | right | right |
| wi_hg_bn256_rotating | wi_hg_b_rotating | tile+threads | +5.7 % | +5.6 % | +9.1 % | right | right |
| wi_hg_bn256_warm | wi_hg_b_warm | tile+threads | +7.8 % | +4.6 % | +6.8 % | right | right |
| wi_hg_bn64_rotating | wi_hg_b_rotating | tile | +10.7 % | +9.7 % | +13.8 % | right | right |
| wi_hg_direct_rotating | wi_hg_b_rotating | cp.async vs direct | +20.8 % | +20.0 % | +11.7 % | right | right |
| wi_hg_direct_warm | wi_hg_b_warm | cp.async vs direct | +8.8 % | +6.7 % | +0.7 % | right | right |
| wi_hg_nocompute_rotating | wi_hg_b_rotating | compute removed | -30.0 % | -26.9 % | -27.4 % | right | right |
| wi_hg_noload_rotating | wi_hg_b_rotating | loads removed | -17.7 % | -15.4 % | -23.6 % | right | right |
| wi_hg_nostore_rotating | wi_hg_b_rotating | stores removed | -8.8 % | -8.3 % | -11.0 % | right | right |
| wi_hg_s2_warm | wi_hg_b_warm | stages | -7.3 % | -2.5 % | -10.9 % | right | right |
| wi_hg_sw8_rotating | wi_hg_b_rotating | raster | -2.6 % | -0.0 % | -4.6 % | right | right |
| wi_hg_t64_warm | wi_hg_b_warm | threads | +223.6 % | +81.7 % | +137.5 % | right | right |
| wi_rms_fp32_rotating | wi_rms_b_rotating | bytes per lane | +82.9 % | +92.7 % | +93.8 % | right | right |
| wi_rms_fp32_warm | wi_rms_b_warm | bytes per lane | +134.5 % | +62.6 % | +60.0 % | right | right | graph: +155.3 %, +80.4 %, right |
| wi_rms_nocompute_rotating | wi_rms_b_rotating | compute removed | -2.0 % | -3.4 % | -3.4 % | right | right |
| wi_rms_nostore_rotating | wi_rms_b_rotating | stores removed | -49.6 % | -0.4 % | -57.4 % | right | right |
| wi_rms_t64_rotating | wi_rms_b_rotating | threads | -0.1 % | +0.2 % | -2.0 % | right | right |
| wi_rms_t64_warm | wi_rms_b_warm | threads | -1.8 % | +2.3 % | +0.8 % | right | right | graph: +0.1 %, +1.1 %, right |
| wi_softmax_t64_warm | wi_softmax_b_warm | threads | +41.6 % | +28.1 % | +17.5 % | right | right | graph: +32.1 %, +24.0 %, right |
| wi_w4_nocompute_warm | wi_w4_b_warm | compute removed | -37.1 % | -38.2 % | -48.9 % | right | right |
| wi_w4_nodequant_warm | wi_w4_b_warm | dequantisation removed | -24.3 % | -23.3 % | -26.4 % | right | right |
| wq_k32_s3_t256_nocompute | wq_k32_s3_t256 | compute removed | -49.5 % | -50.6 % | -55.7 % | right | right |
| wq_m3072_s2_nostore | wq_m3072_s2 | stores removed | -8.7 % | -14.0 % | -11.8 % | right | right |

## Bottleneck: what the model names against what removal gains

Critical-path share by kind in the base's replay (launch left out) and the change from removing that kind, replayed / measured; short bases from graph rows. Decided: two kinds removed, the best gains at least 3 %; kinds within 3 points tie.

Agreement on decided bases: 3/9 (15 bases undecided).

| Base | model's top | shares | gains, replay / measured | largest measured gain | agree |
| --- | --- | --- | --- | --- | --- |
| wi_fa_b_rotating | stores | compute 77 %, loads 19 % | stores -1.1 % / -1.4 % | stores | undecided |
| wi_gemv_b_rotating | compute | loads 87 %, compute 3 % | compute -3.0 % / -0.3 % | compute | undecided |
| wi_gemv_b_warm | compute | compute 76 %, loads 3 % | compute -49.2 % / -44.1 % | compute | undecided |
| wi_hg_b_rotating | compute | compute 46 %, loads 37 %, stores 12 % | compute -27.4 % / -30.0 %, loads -23.6 % / -17.7 %, stores -11.0 % / -8.8 % | compute | yes |
| wi_rms_b_rotating | compute | loads 49 %, compute 39 %, stores 3 % | compute -3.4 % / -2.0 %, stores -57.4 % / -49.6 % | stores | NO |
| wi_w4_b_warm | compute | compute 86 %, stores 9 %, loads 1 % | compute -48.9 % / -37.1 % | compute | undecided |
| rmsb_m192_b | stores | stores 24 %, loads 16 %, compute 6 % | compute -28.7 % / -27.4 %, loads -22.9 % / -18.8 %, stores -37.3 % / -27.5 % | stores | yes |
| rmsb_m48_b | loads | loads 17 %, stores 7 %, compute 2 % | compute -32.6 % / -40.8 %, loads -23.9 % / -33.6 %, stores -16.3 % / -0.3 % | compute | NO |
| smb_m192_b | stores | stores 14 %, loads 15 %, compute 21 % | stores -19.6 % / -22.8 % | stores | undecided |
| smb_m48_b | stores | stores 8 %, compute 23 %, loads 7 % | stores -13.9 % / -10.1 % | stores | undecided |
| wq_k32_s3_t256 | compute | compute 77 %, stores 13 %, loads 6 % | compute -55.7 % / -49.5 % | compute | undecided |
| wq_m3072_s2 | stores | compute 65 %, stores 11 %, loads 20 % | stores -11.8 % / -8.7 % | stores | undecided |
| bl_aos | stores | stores 95 %, loads 1 % | stores -63.9 % / -1.2 % | stores | undecided |
| bl_mixrwa | stores | stores 97 %, loads 1 % | compute -92.6 % / -84.0 %, loads +0.0 % / -4.1 %, stores -0.6 % / -4.3 % | compute | NO |
| hgemm_m2560n2048k1536_c128x128x32_s3_t128 | compute | compute 66 %, stores 15 %, loads 15 % | compute -41.6 % / -36.8 %, loads -29.6 % / -10.9 %, stores -14.5 % / -10.5 % | compute | yes |
| wi12_fa_b_warm | compute | compute 76 %, loads 17 % | compute -3.6 % / +11.7 %, stores -1.4 % / -0.5 % | stores | undecided |
| wi12_rms_b_rotating | compute | loads 66 %, compute 21 %, stores 4 % | compute -9.0 % / -1.7 %, stores -57.4 % / -49.3 % | stores | NO |
| bl13_swaprows | loads | stores 59 %, loads 14 % | loads -0.3 % / -10.6 % | loads | undecided |
| bl13_ticket | stores | stores 87 %, compute 6 % | stores -78.4 % / -30.6 % | stores | undecided |
| fa13_c64x64_s2_t128_rotating | compute | compute 84 %, loads 12 % | compute -36.9 % / -0.8 %, stores -2.1 % / -5.6 % | stores | NO |
| fa13_c64x64_s2_t128_warm | compute | compute 87 %, loads 10 % | compute -7.8 % / +5.6 %, stores -1.8 % / -3.6 % | stores | NO |
| gemv13_b_rotating | compute | loads 89 %, compute 2 % | compute -5.7 % / -7.6 % | compute | undecided |
| gemv13_b_warm | compute | compute 76 %, loads 4 % | compute -51.0 % / -39.8 % | compute | undecided |
| hgemm_m1792n2560k2304_c128x128x32_s3_t128 | loads | compute 69 %, loads 16 %, stores 10 % | loads -20.5 % / -16.1 % | loads | undecided |

## Dynamic opcode weight

153 opcodes with modifiers on the `kernel` rows; default rules time 38 of them, 10.4 % of the instructions on the mean row. Share of a row's instructions, mean over rows and the largest.

| Opcode | Class | Default rule | Mean | Largest |
| --- | --- | --- | ---: | ---: |
| `HMMA.16816.F32` | mma |  | 16.24 % | 72.4 % |
| `FMUL` | fp32 |  | 9.72 % | 52.7 % |
| `FADD` | fp32 |  | 5.40 % | 50.1 % |
| `HADD2.F32` | half_to_float |  | 4.69 % | 27.9 % |
| `LOP3.LUT` | alu |  | 4.40 % | 42.1 % |
| `IADD` | alu |  | 3.41 % | 46.9 % |
| `IADD.64` | alu_wide |  | 3.19 % | 30.7 % |
| `FFMA` | fp32 |  | 3.08 % | 25.3 % |
| `STG.E` | stg |  | 3.01 % | 57.7 % |
| `LDSM.16.M88.4` | ldsm |  | 2.99 % | 11.9 % |
| `LDSM.16.MT88.4` | ldsm |  | 2.42 % | 11.8 % |
| `LDG.E` | ldg |  | 2.12 % | 45.7 % |
| `LDGSTS.E.BYPASS.128` | cp_async |  | 2.04 % | 33.4 % |
| `SHF.R.U32.HI` | alu |  | 1.88 % | 21.8 % |
| `MOV` | other | yes | 1.76 % | 22.9 % |
| `UIADD3` | other | yes | 0.96 % | 8.8 % |
| `S2R` | other | yes | 0.91 % | 19.0 % |
| `LDCU.64` | other | yes | 0.78 % | 18.2 % |
| `LDC` | other | yes | 0.73 % | 29.6 % |
| `LDC.64` | other | yes | 0.65 % | 15.4 % |
| `BRA` | other | yes | 0.58 % | 5.5 % |
| `BSYNC.RECONVERGENT` | other | yes | 0.49 % | 5.4 % |
| `S2UR` | other | yes | 0.45 % | 8.7 % |
| `BSSY.RECONVERGENT` | other | yes | 0.45 % | 5.1 % |
| `FCHK` | other | yes | 0.38 % | 5.1 % |
| `CS2R` | other | yes | 0.33 % | 3.5 % |
| `USHF.L.U32` | other | yes | 0.31 % | 7.7 % |
| `DEPBAR.LE` | other | yes | 0.30 % | 5.1 % |
| `UMOV` | other | yes | 0.25 % | 11.8 % |

## Per unit: model against counters

Median of model busy cycles / counter cycles per block, on rows with the round 10 counters: tensor 1.06 (n 123), fma 0.70 (n 270), alu 1.05 (n 275), fill 1.91 (n 275), lsu 0.94 (n 229).
Stall families, mean counter share of stalled warp-cycles / model critical-path share: pipes 0.23 / 0.52, memory 0.51 / 0.46, sync 0.26 / 0.00.

| Row | Unit | Model | Counter | Ratio |
| --- | --- | ---: | ---: | ---: |
| stress_02 kernel | fill | 0 | 4 | 0.00 |
| stress_12 kernel | fill | 0 | 4 | 0.00 |
| atom_b1 kernel | fill | 0 | 1 | 0.00 |
| atom_b16_gap128 kernel | fill | 0 | 2 | 0.00 |
| atom_b256_gap4096 kernel | fill | 0 | 2 | 0.00 |
| rw_s16_o0_g48 kernel | fill | 0 | 18 | 0.00 |
| rw_s32_o0_g48 kernel | fill | 0 | 18 | 0.00 |
| rw_s32_o0_g768 kernel | fill | 0 | 2 | 0.00 |
| rw_s4_o0_g48 kernel | fill | 0 | 18 | 0.00 |
| rw_s8_o3_g768 kernel | fill | 0 | 2 | 0.00 |

## Convergence record

Per class, in the order first scored. Not a claim of convergence: first-score errors must fall and mechanisms per class must reach zero on classes from outside.

| Class | First batch | n | Median first | Worst first | New mechanisms needed |
| --- | --- | ---: | ---: | ---: | ---: |
| HGEMM (TileLang) | r1 | 85 | 6.3 % | -60.4 % | 9 |
| FA forward | r1fa | 4 | 10.0 % | -15.1 % | 4 |
| decode attention | r2b | 5 | 14.9 % | +34.4 % | 3 |
| w4a16 GEMV | r2b | 2 | 631.3 % | +1053.8 % | 3 |
| RMSNorm, softmax | r2 | 4 | 26.5 % | -49.1 % | 5 |
| FP8 GEMM | r3 | 2 | 8.8 % | -11.0 % | 0 |
| MoE GEMM | r4 | 2 | 144.0 % | +223.9 % | 1 |
| w4a16 GEMM | r5 | 4 | 11.7 % | -15.9 % | 1 |
| NVFP4 GEMM | r8 | 6 | 32.7 % | -37.4 % | 1 |
| hand-written CUDA | r10 | 4 | 23.1 % | +166.9 % | 5 |
| generated mixes | r10 | 4 | 2.0 % | -14.1 % | 0 |
| library operators | r10w | 4 | 98.9 % | -99.6 % | 0 |

## Address coverage

Memory operations of the kernel rows whose address the lanes resolve: global 99.93 % of 2536839, shared and stack 99.88 % of 2205302; 86784 more come from loaded data, a value and no gap; 45 rows have a gap.

| Broke the chain | Operations |
| --- | ---: |
| local SHF.R.U32 | 1376 |
| local guard | 996 |
| chip IMAD.WIDE.U32 | 744 |
| chip POPC | 704 |
| local LEA | 290 |
| chip guard | 208 |
| chip SHF.L.U32 | 26 |
| chip MOV | 6 |
| chip LEA | 2 |
| chip LOP3.LUT | 2 |

| Row | Unresolved share |
| --- | ---: |
| lib_E08 kernel | 100.0 % |
| lib_E07 kernel | 100.0 % |
| bl13_ticket kernel | 50.0 % |
| wi12b_dec_split16_warm kernel | 8.2 % |
| wi12b_dec_split16_rotating kernel | 8.2 % |
| gemv14_np2_r64_rotating kernel | 7.4 % |
| gd_np2_r64 kernel | 7.4 % |
| gemv15_np2_r64_warm kernel | 6.2 % |
| dec13_split16_warm kernel | 6.1 % |
| dec15_split16_warm kernel | 6.0 % |
| dec15_split16_rotating kernel | 6.0 % |
| wi12b_dec_s1_warm kernel | 5.3 % |

## Top 15 rows by absolute error

| Row | Level | Unit | Measured | Predicted | Error | Blind error | Attribution (critical path) | Defaults | Note |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| store_lines S2_W4_D128_g128 | 4 | cycles | 105.00 | 1,370.00 | +1204.8 % | +417.5 % (r7b) | store_port: queue (stores) 46 %; store_port: drain after the last instruction 30 %; store_queue: queue (stg) 24 % | default latency 54, issue-only 42 | batch r7b |
| store_lines S2_W16_D32_g128 | 4 | cycles | 287.00 | 797.41 | +177.8 % | +129.4 % (r7b) | store_port: drain after the last instruction 44 %; store_port: queue (stores) 44 %; store_queue: queue (stg) 9 % | default latency 54, issue-only 75 | batch r7b |
| row_reduce empty_g48_span | 4 | cycles | 8.00 | 19.00 | +137.5 % | +137.5 % (r5) | issue: default result latency (S2UR) 22 %; issue: default result latency (S2R) 22 %; issue: stall count after other 18 % | default latency 8, issue-only 13 | batch r5 |
| store_lines S2_W4_D32_g256 | 4 | cycles | 1,585.00 | 3,648.95 | +130.2 % | +50.5 % (r7b) | store_queue: queue (stg) 62 %; store_port: drain after the last instruction 23 %; store_port: queue (stores) 14 % | default latency 108, issue-only 78 | batch r7b |
| stl_l_n64 kernel | 7 | us | 63.90 | 141.03 | +120.7 % | -61.9 % (r14) | l1: queue 97 %; launch: launch 2 % | default latency 240, issue-only 96 | batch r14 |
| hand_mandel graph | 7 | us | 5.60 | 12.08 | +115.7 % | +104.0 % (r10) | issue: scheduler (another warp issued) 49 %; fma: stall count after fp32 16 %; alu: stall count after alu 11 % | branch fallback 11088, default latency 1056, issue-only 13376 | batch r10 |
| bl13_tilefill_p2 kernel | 7 | us | 24.00 | 51.74 | +115.6 % | +254.0 % (r13b) | store_queue: queue (stg) 93 %; launch: launch 6 %; store_port: drain after the last instruction 1 % | default latency 1056, issue-only 176 | batch r13b |
| bl13_ticket graph | 7 | us | 21.79 | 45.53 | +108.9 % | +103.0 % (r13b) | store_queue: queue (red) 63 %; store_port: queue (atomics) 15 %; store_queue: queue (stg) 14 % | default latency 1232, issue-only 880 | batch r13b |
| lib_E07 kernel | 7 | us | 609.28 | 4.60 | -99.2 % | -99.6 % (r10w) | launch: launch 72 %; l2_return: queue (chip L2 share) 13 %; lsu: queue 10 % | default latency 240, front end 32, issue-only 48 | batch r10w |
| spill_rw n256_L4_S1_H0_g1 | 4 | cycles | 72,909.00 | 145,133.40 | +99.1 % | +7.1 % (r8) | l1: queue 40 %; l1: latency 30 %; alu: stall count after alu 10 % | branch fallback 1, default latency 67, issue-only 39177 | batch r8 |
| lib_E02 kernel | 7 | us | 235.86 | 2.53 | -98.9 % | -98.9 % (r10w) | launch: launch 97 %; alu: stall count after alu 1 %; alu: queue (alu) 1 % | default latency 144, issue-only 24 | batch r10w |
| lib_E01 kernel | 7 | us | 221.85 | 2.53 | -98.9 % | -98.9 % (r10w) | launch: launch 97 %; alu: stall count after alu 1 %; alu: queue (alu) 1 % | default latency 144, issue-only 24 | batch r10w |
| stl_u_n256 kernel | 7 | us | 1,237.34 | 2,456.83 | +98.6 % | -71.6 % (r14) | l1: queue 100 % | default latency 240, issue-only 1632 | batch r14 |
| lib_E08 kernel | 7 | us | 604.03 | 14.21 | -97.6 % | -98.8 % (r10w) | l2_return: queue (chip L2 share) 32 %; l2/dram: load latency (incl. DRAM queue) 21 %; launch: launch 20 % | default latency 1496, front end 176, issue-only 264 | batch r10w |
| bl13_ticket b2b | 7 | us | 24.43 | 47.67 | +95.1 % | +89.8 % (r13b) | store_queue: queue (red) 63 %; store_port: queue (atomics) 15 %; store_queue: queue (stg) 14 % | default latency 1232, issue-only 880 | batch r13b |

## All rows

| Row | Level | Unit | Measured | Predicted | Error | Blind error | Attribution (critical path) | Defaults | Note |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| gemm1024 c128x128x32_st3_th128 full | 7 | us | 39.46 | 37.17 | -5.8 % |  | tensor: stall count after mma 28 %; tensor: queue (mma) 27 %; lsu: queue 12 % | default latency 80, default rbar 248, issue-only 16300 |  |
| gemm1024 c128x256x32_st3_th256 nocompute | 7 | us | 25.09 | 23.78 | -5.2 % |  | l2_return: queue (chip L2 share) 35 %; store_queue: queue (stg) 31 %; launch: launch 11 % | default latency 80, issue-only 282 |  |
| gemm1024 c128x256x64_st2_th256 full | 7 | us | 41.28 | 45.22 | +9.5 % |  | tensor: queue (mma) 23 %; tensor: stall count after mma 22 %; store_queue: queue (stg) 16 % | default latency 88, default rbar 8, issue-only 14122 |  |
| gemm1024 c128x64x32_st3_th128 full | 7 | us | 37.41 | 40.26 | +7.6 % |  | l2_return: queue (chip L2 share) 31 %; tensor: stall count after mma 17 %; store_queue: queue (stg) 16 % | default latency 120, issue-only 13878 |  |
| gemm1024 c256x128x32_st3_th256 nocompute | 7 | us | 24.74 | 23.78 | -3.9 % |  | l2_return: queue (chip L2 share) 35 %; store_queue: queue (stg) 31 %; launch: launch 11 % | default latency 80, issue-only 282 |  |
| gemm1024 c256x128x64_st2_th256 nocompute | 7 | us | 25.92 | 23.76 | -8.3 % |  | l2_return: queue (chip L2 share) 32 %; store_queue: queue (stg) 31 %; lsu: queue 13 % | default latency 80, issue-only 154 |  |
| gemm1024 c256x128x64_st2_th256 noload 1 block | 7 | us | 30.27 | 31.26 | +3.3 % |  | tensor: queue (mma) 37 %; tensor: stall count after mma 31 %; issue: stall count after ldsm 9 % | default latency 176, default rbar 8, issue-only 13698 |  |
| gemm1024 c256x64x32_st3_th128 full | 7 | us | 47.42 | 53.16 | +12.1 % |  | tensor: stall count after mma 38 %; lsu: queue 16 %; store_queue: queue (stg) 14 % | default latency 80, default rbar 248, issue-only 16260 |  |
| gemm1024 c256x64x32_st3_th256 nocompute | 7 | us | 33.38 | 35.43 | +6.1 % |  | l2_return: queue (chip L2 share) 33 %; store_queue: queue (stg) 22 %; l2_return: queue (SM port rate) 13 % | default latency 160, issue-only 564 |  |
| gemm1024 c256x64x32_st3_th256 noload 1 block | 7 | us | 15.68 | 16.73 | +6.7 % |  | tensor: queue (mma) 27 %; tensor: stall count after mma 23 %; issue: stall count after barrier 14 % | default latency 64, default rbar 496, issue-only 5586 |  |
| gemm1024 c64x128x32_st3_th128 nocompute | 7 | us | 35.20 | 34.51 | -1.9 % |  | l2_return: queue (chip L2 share) 53 %; store_queue: queue (stg) 19 %; l2: L2 latency 9 % | default latency 120, issue-only 426 |  |
| gemm1024 c64x128x32_st3_th256 full | 7 | us | 35.01 | 37.74 | +7.8 % |  | l2_return: queue (chip L2 share) 33 %; store_queue: queue (stg) 18 %; tensor: queue (mma) 9 % | default latency 240, default rbar 72, issue-only 6462 |  |
| gemm1024 c64x128x64_st3_th128 full | 7 | us | 40.32 | 42.85 | +6.3 % |  | tensor: stall count after mma 34 %; lsu: queue 16 %; store_queue: queue (stg) 11 % | default latency 120, default rbar 24, issue-only 11202 |  |
| gemm1024 c64x128x64_st3_th128 noload 1 block | 7 | us | 8.38 | 8.59 | +2.5 % |  | tensor: stall count after mma 56 %; issue: stall count after ldsm 15 %; issue: stall count after other 7 % | default latency 32, default rbar 4, issue-only 3902 |  |
| gemm1024 c64x256x32_st3_th256 full | 7 | us | 51.71 | 54.06 | +4.5 % |  | tensor: stall count after mma 19 %; lsu: queue 15 %; store_queue: queue (stg) 14 % | default latency 160, default rbar 16, issue-only 19940 |  |
| gemm1024 c64x64x128_st3_th128 noload 1 block | 7 | us | 3.94 | 4.26 | +8.2 % |  | tensor: stall count after mma 48 %; issue: stall count after ldsm 24 %; issue: stall count after other 6 % | default latency 32, default rbar 4, issue-only 1642 |  |
| gemm1024 c64x64x32_st3_th256 noload 1 block | 7 | us | 3.78 | 4.33 | +14.6 % |  | tensor: queue (mma) 24 %; issue: stall count after barrier 16 %; tensor: stall count after mma 14 % | default latency 64, issue-only 962 |  |
| tma 1024 full | 7 | us | 45.12 | 40.91 | -9.3 % |  | tma_engine: queue 33 %; tensor: stall count after mma 16 %; store_queue: queue (stg) 9 % | branch fallback 48, default latency 468, default rbar 444, issue-only 10755 |  |
| tma m32 full | 7 | us | 123.97 | 126.01 | +1.6 % |  | tma_engine: transfer / DRAM queue 81 %; tma_engine: queue 7 %; tma_engine: L2 latency (TMA) 2 % | branch fallback 16, default latency 372, default rbar 332, issue-only 6308 | measured warm in a session where part of B stayed in the L2; upstream compares the decode picks cold against a later flush measurement (gb10 design note §7) |
| tma m32 noload 1 block | 7 | us | 15.62 | 16.48 | +5.5 % |  | tensor: stall count after mma 42 %; issue: stall count after ldsm 27 %; issue: stall count after other 6 % | branch fallback 16, default latency 371, default rbar 204, issue-only 5553 |  |
| tma k1024 noload 1 block | 7 | us | 25.02 | 30.09 | +20.2 % |  | tensor: queue (mma) 65 %; tensor: stall count after mma 13 %; issue: stall count after other 4 % | default latency 2860, default rbar 1360, issue-only 15684 |  |
| tma m128 full | 7 | us | 162.11 | 211.14 | +30.2 % |  | tma_engine: transfer / DRAM queue 41 %; tensor: stall count after mma 20 %; tma_engine: L2 latency (TMA) 11 % | default latency 2208, default rbar 1832, issue-only 63986 | measured warm in a session where part of B stayed in the L2; upstream compares the decode picks cold against a later flush measurement (gb10 design note §7) |
| tma m128 noload 1 block | 7 | us | 60.58 | 62.13 | +2.6 % |  | tensor: stall count after mma 67 %; issue: stall count after ldsm 13 %; issue: stall count after other 7 % | default latency 981, default rbar 660, issue-only 30087 |  |
| sched adds alone | 1 | cycles/add | 5.67 | 5.31 | -6.3 % |  | alu: stall count after alu 94 %; issue: stall count after other 6 % |  |  |
| sched adds beside FFMA | 3 | cycles/add | 15.10 | 14.91 | -1.2 % |  | issue: scheduler (another warp issued) 64 %; alu: stall count after alu 34 %; issue: stall count after other 2 % | issue-only 12000 |  |
| alu LOP3 w4 | 2 | cycles | 13,451.56 | 13,598.00 | +1.1 % |  | alu: stall count after alu 86 %; issue: stall count after other 12 %; xu: xu result latency 1 % | default latency 13, default rbar 4, issue-only 1552 |  |
| alu SHF w4 | 2 | cycles | 8,393.44 | 8,815.00 | +5.0 % |  | alu: stall count after alu 96 %; issue: stall count after other 1 %; xu: xu result latency 1 % | default latency 13, default rbar 8, issue-only 112 |  |
| alu HFMA2 w4 | 2 | cycles | 8,397.74 | 8,823.00 | +5.1 % |  | fma: stall count after fma_half_rate 95 %; issue: stall count after other 2 %; xu: xu result latency 1 % | default latency 13, issue-only 112 |  |
| alu F2F.F16.F32+HADD2.F32 w4 | 2 | cycles | 15,922.25 | 15,229.00 | -4.4 % |  | issue: stall count after half_to_float 47 %; fma: stall count after fp32 28 %; alu: stall count after alu 18 % | default latency 13, issue-only 400 |  |
| alu HADD2.F32(h2f) w4 | 2 | cycles | 40,655.09 | 40,671.00 | +0.0 % |  | xu: xu result latency 90 %; alu: stall count after alu 5 %; issue: stall count after other 2 % | default latency 13, default rbar 260, issue-only 784 |  |
| alu F2FP.PACK_AB w8 | 2 | cycles | 23,076.06 | 23,659.00 | +2.5 % |  | issue: stall count after half_to_float 30 %; fma: stall count after fp32 17 %; alu: queue (alu) 17 % | default latency 25, issue-only 800 |  |
| alu F2I w4 | 2 | cycles | 4,329.81 | 4,360.00 | +0.7 % |  | fma: stall count after fp32 87 %; issue: stall count after other 5 %; xu: xu result latency 3 % | default latency 13, default rbar 8, issue-only 208 |  |
| alumix HFMA2+LOP3 | 3 | cycles | 51,505.00 | 49,251.00 | -4.4 % |  | alu: queue (alu) 83 %; alu: stall count after alu 7 %; fma: stall count after fma_half_rate 4 % | default latency 97, issue-only 12576 |  |
| alumix PRMT+IMAD | 3 | cycles | 33,480.00 | 33,479.00 | -0.0 % |  | fma: queue (fma_half_rate) 61 %; alu: queue (alu) 23 %; alu: stall count after alu 6 % | default latency 97, default rbar 64, issue-only 1664 |  |
| alumix HADD2.F32+FADD+IMAD | 3 | cycles | 64,977.00 | 65,745.00 | +1.2 % |  | fma: queue (fp32) 67 %; fma: queue (fma_half_rate) 14 %; fma: stall count after fma_half_rate 5 % | default latency 129, issue-only 4224 |  |
| alumix FFMA+FFMA | 3 | cycles | 34,117.00 | 34,066.00 | -0.1 % |  | fma: queue (fp32) 67 %; fma: stall count after fp32 12 %; fma: queue (fma_half_rate) 11 % | default latency 97, default rbar 64, issue-only 3200 |  |
| sfu ex2.f32 w1 | 2 | cycles | 262,144.00 | 264,795.00 | +1.0 % |  | xu: queue (xu) 85 %; xu: stall count after xu 12 %; xu: xu result latency 2 % | default latency 5, default rbar 1792, issue-only 781 |  |
| sfu rsqrt.f32 latency | 1 | cycles/op | 18.10 | 18.05 | -0.3 % |  | xu: xu result latency 100 % | branch fallback 1, default latency 5, default rbar 256, issue-only 793 |  |
| sfu sin.f32 latency | 1 | cycles/op | 22.20 | 22.17 | -0.1 % |  | xu: xu result latency 81 %; fma: stall count after fp32 18 %; issue: stall count after other 1 % | branch fallback 1, default latency 5, issue-only 793 |  |
| sfu lg2.f32 w8 | 2 | cycles | 524,288.00 | 524,404.00 | +0.0 % |  | xu: queue (xu) 93 %; xu: stall count after xu 6 %; issue: stall count after other 1 % | default latency 33, default rbar 14336, issue-only 6248 |  |
| sfu tanh.f32 w4 | 2 | cycles | 265,462.28 | 264,801.00 | -0.2 % |  | xu: queue (xu) 85 %; xu: stall count after xu 12 %; xu: xu result latency 2 % | default latency 17, default rbar 7168, issue-only 3124 |  |
| sfu ffma.f32 latency | 1 | cycles/op | 4.70 | 4.55 | -3.2 % |  | fma: stall count after fp32 75 %; issue: stall count after other 24 % | branch fallback 1, default latency 5, issue-only 793 |  |
| cvt int to fp32 | 2 | cycles/element/smsp | 4.03 | 4.01 | -0.4 % |  | alu: queue (alu) 87 %; alu: stall count after alu 12 % | default latency 192, front end 256, issue-only 24640 |  |
| smem_ldsts LDS.32_w8 | 2 | cycles/instr/SM | 5.21 | 5.24 | +0.6 % | -18.6 % (r1) | smem_array: LDS latency 58 %; alu: stall count after alu 28 %; alu: queue (alu) 8 % | default latency 33, issue-only 6192 | batch r3 |
| smem_ldsts LDS.64_w1 | 1 | cycles/instr/SM | 46.05 | 46.52 | +1.0 % | -25.0 % (r1) | smem_array: LDS latency 69 %; alu: stall count after alu 27 %; issue: stall count after other 4 % | default latency 5, issue-only 774 | batch r3 |
| smem_ldsts STS.32_w2 | 2 | cycles/instr/SM | 2.54 | 2.58 | +1.6 % | +1.6 % (r2) | issue: stall count after smem_op 55 %; issue: stall count after other 27 %; mio: queue (smem_op) 17 % | default latency 8, issue-only 1548 | batch r3 |
| smem_ldsts STS.32_w3 | 2 | cycles/instr/SM | 2.27 | 2.01 | -11.7 % | -11.7 % (r2) | issue: stall count after smem_op 45 %; smem_array: queue (sts) 16 %; mio: queue (smem_op) 14 % | default latency 11, issue-only 2322 | batch r3 |
| smem_ldsts STS.32_w4 | 2 | cycles/instr/SM | 2.14 | 2.01 | -6.4 % | -47.1 % (r1) | smem_array: queue (sts) 37 %; issue: stall count after smem_op 34 %; mio: queue (smem_op) 11 % | default latency 14, issue-only 3096 | batch r3 |
| smem_ldsts STS.32_w6 | 2 | cycles/instr/SM | 2.01 | 2.00 | -0.6 % | -0.6 % (r2) | smem_array: queue (sts) 56 %; issue: stall count after smem_op 23 %; mio: queue (smem_op) 9 % | default latency 20, issue-only 4644 | batch r3 |
| smem_ldsts STS.128_w5 | 2 | cycles/instr/SM | 5.62 | 4.41 | -21.5 % | -21.1 % (r2) | smem_array: store read 77 %; issue: stall count after other 23 % | default latency 17, issue-only 14115 | batch r3 |
| shfl latency_w1 | 1 | cycles/hop | 28.01 | 28.00 | -0.0 % | -28.6 % (r1) | smem_array: SHFL latency 100 % | default latency 3, issue-only 773 | batch r3 |
| shfl tput_w4 | 2 | cycles/instr/SM | 2.12 | 2.07 | -2.6 % | -48.2 % (r1) | smem_array: SHFL latency 42 %; smem_array: queue (shfl) 35 %; issue: stall count after smem_op 16 % | default latency 9, issue-only 3092 | batch r3 |
| shfl tput_w8 | 2 | cycles/instr/SM | 2.01 | 2.00 | -0.3 % | -72.6 % (r1) | smem_array: queue (shfl) 66 %; smem_array: SHFL latency 22 %; issue: stall count after smem_op 8 % | default latency 17, issue-only 6184 | batch r3 |
| ex2_ffma E8_F0 | 3 | cycles/pass/smsp | 511.55 | 512.67 | +0.2 % | +0.2 % (r1) | xu: queue (xu) 95 %; issue: stall count after other 3 %; xu: stall count after xu 2 % | default latency 65, issue-only 24704 | batch r3 |
| ex2_ffma E8_F8 | 3 | cycles/pass/smsp | 518.46 | 512.52 | -1.1 % | -1.1 % (r1) | xu: queue (xu) 93 %; issue: stall count after other 3 %; fma: stall count after fp32 2 % | default latency 65, issue-only 24704 | batch r3 |
| ex2_ffma E8_F16 | 3 | cycles/pass/smsp | 514.85 | 512.85 | -0.4 % | -0.4 % (r1) | xu: queue (xu) 91 %; fma: stall count after fp32 3 %; issue: stall count after other 3 % | default latency 65, issue-only 24704 | batch r3 |
| namedbar sync_w2 | 1 | cycles/barrier | 26.05 | 26.10 | +0.2 % | +34.7 % (r1) | issue: stall count after other 68 %; barrier: release latency 30 %; issue: default result latency (LDC) 1 % | default latency 4, issue-only 776 | batch r3 |
| namedbar sync_w4 | 1 | cycles/barrier | 30.05 | 30.10 | +0.2 % | +30.1 % (r1) | issue: stall count after other 59 %; barrier: release latency 39 %; issue: default result latency (LDC) 1 % | default latency 6, issue-only 1552 | batch r3 |
| namedbar arrive_L0 | 3 | cycles/arrive | 12.00 | 12.00 | +0.0 % | +191.7 % (r1) | barrier: release latency 48 %; issue: stall count after other 20 %; alu: stall count after alu 18 % | default latency 11, issue-only 3096 | batch r3 |
| l1hit chase | 1 | cycles/hop | 55.09 | 52.00 | -5.6 % | -4.3 % (r1) | l1: hit latency 86 %; fma: stall count after fma_half_rate 10 %; l1: hit path 2 % | default latency 3, issue-only 6341 | batch r3 |
| l1hit tput_w4 | 2 | cycles/instr/SM | 6.70 | 6.01 | -10.2 % | +21.1 % (r1) | l1: hit latency 47 %; alu: stall count after alu 30 %; fma: stall count after fma_half_rate 7 % | branch fallback 4, default latency 17, issue-only 3112 | batch r3 |
| l1hit tput_w8 | 2 | cycles/instr/SM | 4.31 | 4.01 | -7.0 % | +85.8 % (r1) | alu: queue (alu) 27 %; alu: stall count after alu 26 %; l1: hit latency 18 % | branch fallback 8, default latency 33, issue-only 6224 | batch r3 |
| store_tail tl_b1 | 4 | us | 3.04 | 3.27 | +7.5 % | +3.6 % (r1) | store_queue: queue (stg) 67 %; store_port: queue (stores) 13 %; alu: stall count after alu 7 % | default latency 49, default rbar 8, issue-only 936 | batch r3 |
| store_tail v16_b2 | 4 | us | 1.23 | 1.24 | +0.7 % | +23.9 % (r1) | store_queue: queue (stg) 53 %; store_port: queue (stores) 33 %; alu: stall count after alu 4 % | default latency 41, default rbar 8, issue-only 448 | batch r3 |
| store_tail v16_b32 | 4 | us | 1.55 | 1.57 | +0.9 % | -2.0 % (r1) | store_queue: queue (stg) 42 %; store_port: queue (stores) 27 %; store_port: drain after the last instruction 19 % | default latency 41, default rbar 8, issue-only 448 | batch r3 |
| store_tail v16_b48 | 4 | us | 2.24 | 2.31 | +2.7 % | +2.8 % (r1) | store_queue: queue (stg) 48 %; store_port: queue (stores) 31 %; store_port: drain after the last instruction 13 % | default latency 41, default rbar 8, issue-only 448 | batch r3 |
| l2share_copy b1 | 4 | us | 10.08 | 10.57 | +4.8 % | +2.5 % (r1) | l2_return: queue (SM port rate) 54 %; lsu: queue 29 %; l2: L2 latency 11 % | default latency 80, issue-only 154 | batch r3 |
| l2share_copy b32 | 4 | us | 16.07 | 12.63 | -21.4 % | -22.1 % (r1) | l2_return: queue (chip L2 share) 62 %; lsu: queue 24 %; l2: L2 latency 9 % | default latency 80, issue-only 154 | batch r3 |
| dram_rotate t64 | 5 | us | 621.09 | 629.60 | +1.4 % | +38.8 % (r1) | l2/dram: load latency (incl. DRAM queue) 96 %; l2_return: transfer 3 %; issue: stall count after other 1 % | default latency 8, issue-only 8166 | batch r3 |
| dram_rotate t128 | 5 | us | 362.98 | 316.24 | -12.9 % | +18.9 % (r1) | l2/dram: load latency (incl. DRAM queue) 96 %; l2_return: transfer 3 %; issue: stall count after other 1 % | default latency 16, issue-only 8172 | batch r3 |
| dram_rotate t256 | 5 | us | 288.13 | 291.05 | +1.0 % | -1.4 % (r1) | l2/dram: load latency (incl. DRAM queue) 98 %; l2_return: transfer 1 % | default latency 32, issue-only 8184 | batch r3 |
| fa_fwd_s2048_h8_d128 kernel | 7 | us | 253.47 | 266.87 | +5.3 % | -8.7 % (r1fa) | tensor: stall count after mma 19 %; lsu: queue 15 %; tensor: queue (mma) 14 % | branch fallback 1536, default latency 1824, default rbar 8496, issue-only 10776 | batch r4 |
| hgemm2048_c128x128x32_s3_t128 kernel | 7 | us | 223.78 | 231.52 | +3.5 % | -6.3 % (r1) | tensor: queue (mma) 39 %; tensor: stall count after mma 20 %; lsu: queue 15 % | default latency 192, default rbar 1512, issue-only 10776 | batch r10 |
| hgemm2048_c128x128x32_s3_t128_sw8 kernel | 7 | us | 223.58 | 210.58 | -5.8 % | -14.2 % (r1) | tensor: queue (mma) 35 %; tensor: stall count after mma 22 %; store_queue: queue (stg) 14 % | default latency 264, default rbar 1572, issue-only 10812 | batch r10 |
| hgemm2048_c128x128x32_s4_t128 kernel | 7 | us | 248.13 | 264.69 | +6.7 % | -3.5 % (r1) | tensor: stall count after mma 36 %; tensor: queue (mma) 19 %; lsu: queue 16 % | default latency 192, default rbar 24, issue-only 4704 | batch r10 |
| hgemm2048_c128x128x64_s2_t128 kernel | 7 | us | 228.61 | 255.98 | +12.0 % | +0.8 % (r1) | tensor: stall count after mma 35 %; tensor: queue (mma) 20 %; lsu: queue 16 % | default latency 240, default rbar 24, issue-only 3288 | batch r10 |
| hgemm2048_c128x128x64_s3_t256 kernel | 7 | us | 229.50 | 257.46 | +12.2 % | -2.7 % (r1) | tensor: queue (mma) 32 %; tensor: stall count after mma 17 %; lsu: queue 13 % | default latency 480, default rbar 48, issue-only 10752 | batch r10 |
| hgemm2048_c128x256x32_s3_t256 kernel | 7 | us | 237.92 | 247.39 | +4.0 % | -7.4 % (r1) | tensor: queue (mma) 26 %; issue: stall count after barrier 19 %; tensor: stall count after mma 19 % | default latency 216, default rbar 1512, issue-only 13776 | batch r10 |
| hgemm2048_c128x256x32_s3_t256_sw4 kernel | 7 | us | 236.96 | 235.03 | -0.8 % | -9.7 % (r1) | tensor: queue (mma) 27 %; tensor: stall count after mma 20 %; issue: stall count after barrier 15 % | default latency 264, default rbar 1568, issue-only 13808 | batch r10 |
| hgemm2048_c128x256x64_s2_t256 kernel | 7 | us | 222.69 | 241.94 | +8.6 % | -7.3 % (r1) | tensor: queue (mma) 27 %; tensor: stall count after mma 18 %; issue: stall count after barrier 17 % | default latency 240, default rbar 24, issue-only 3288 | batch r10 |
| hgemm2048_c256x128x32_s3_t256 kernel | 7 | us | 233.79 | 247.26 | +5.8 % | -4.6 % (r1) | tensor: queue (mma) 43 %; tensor: stall count after mma 19 %; store_queue: queue (stg) 12 % | default latency 216, default rbar 1512, issue-only 13752 | batch r10 |
| hgemm2048_c64x128x32_s4_t128 kernel | 7 | us | 233.60 | 223.15 | -4.5 % | -3.0 % (r1) | l2_return: queue (chip L2 share) 34 %; tensor: queue (mma) 20 %; tensor: stall count after mma 13 % | default latency 396, issue-only 7172 | batch r10 |
| mma_pair b1_span | 3 | cycles | 310,515.00 | 319,186.80 | +2.8 % | +1.8 % (r2) | tensor: stall count after mma 76 %; issue: stall count after other 9 %; mio: queue (ldsm) 9 % | branch fallback 1, default latency 23, default rbar 4, issue-only 71989 | batch r3 |
| mma_pair b96_s0_span | 3 | cycles | 529,802.00 | 533,712.60 | +0.7 % | -1.0 % (r2) | tensor: stall count after mma 45 %; tensor: queue (mma) 41 %; issue: stall count after other 6 % | branch fallback 2, default latency 46, default rbar 8, issue-only 143978 | batch r3 |
| mma_pair b96_s4096_first | 3 | cycles | 469,062.00 | 469,066.90 | +0.0 % | +2.8 % (r2) | tensor: stall count after mma 41 %; tensor: queue (mma) 27 %; issue: stall count after other 20 % | branch fallback 2, default latency 52, default rbar 8, issue-only 189046 | batch r3 |
| l2share_stages s1_b48 | 4 | us | 19.49 | 23.89 | +22.5 % | +20.3 % (r2) | l2_return: queue (chip L2 share) 51 %; lsu: queue 23 %; l2: L2 latency 19 % | default latency 72, issue-only 1338 | batch r3 |
| l2share_stages s2_b16 | 4 | us | 12.59 | 10.53 | -16.3 % | -18.1 % (r2) | l2_return: queue (SM port rate) 41 %; lsu: queue 26 %; l2: L2 latency 21 % | default latency 72, issue-only 1034 | batch r3 |
| l2share_stages s3_b1 | 4 | us | 10.05 | 10.53 | +4.8 % | +2.5 % (r2) | l2_return: queue (SM port rate) 57 %; lsu: queue 19 %; l2: L2 latency 15 % | default latency 72, issue-only 762 | batch r3 |
| l2share_stages s4_b48 | 4 | us | 17.50 | 18.79 | +7.4 % | +6.0 % (r2) | l2_return: queue (chip L2 share) 81 %; lsu: queue 10 %; l2: L2 latency 6 % | default latency 72, issue-only 738 | batch r3 |
| decode_gqa_comb_s4096_warm kernel | 7 | us | 2.50 | 2.53 | +1.4 % | +34.4 % (r2b) | launch: launch 91 %; l2/dram: load latency (incl. DRAM queue) 4 %; issue: stall count after ldg 1 % | branch fallback 4, default latency 28, issue-only 16 | batch r4 |
| decode_gqa_part_s4096_rotating kernel | 7 | us | 71.04 | 77.25 | +8.7 % | +15.5 % (r2b) | dram_queue: queue (DRAM share) 71 %; dram: DRAM latency 12 %; launch: launch 4 % | branch fallback 266, default latency 356, default rbar 496, issue-only 1604 | batch r4 |
| decode_gqa_part_s4096_warm kernel | 7 | us | 25.89 | 26.67 | +3.0 % | -9.3 % (r2b) | tensor: stall count after mma 17 %; lsu: queue 13 %; tensor: queue (mma) 12 % | branch fallback 266, default latency 356, default rbar 496, issue-only 1604 | batch r4 |
| decode_gqa_part_s4096_warm b2b | 7 | us | 24.79 | 26.61 | +7.4 % | -4.9 % (r4) | tensor: stall count after mma 19 %; lsu: queue 14 %; tensor: queue (mma) 13 % | branch fallback 266, default latency 356, default rbar 496, issue-only 1604 | batch r4 |
| decode_gqa_part_s4096_warm graph | 7 | us | 23.42 | 24.47 | +4.5 % | -10.8 % (r4) | tensor: stall count after mma 19 %; lsu: queue 14 %; tensor: queue (mma) 13 % | branch fallback 266, default latency 356, default rbar 496, issue-only 1604 | batch r4 |
| gemv_w4a16_n4096_k4096_rotating kernel | 7 | us | 37.34 | 40.16 | +7.6 % | +1053.8 % (r2b) | l2/dram: load latency (incl. DRAM queue) 87 %; launch: launch 7 %; issue: stall count after half_to_float 1 % | default latency 726, default rbar 1056, issue-only 2376 | batch r4 |
| gemv_w4a16_n4096_k4096_warm kernel | 7 | us | 16.74 | 14.51 | -13.3 % | +208.8 % (r2b) | xu: queue (xu) 32 %; alu: queue (alu) 21 %; launch: launch 20 % | default latency 726, default rbar 1056, issue-only 2376 | batch r4 |
| gemv_w4a16_n4096_k4096_warm b2b | 7 | us | 16.08 | 14.45 | -10.2 % | -6.8 % (r4) | xu: queue (xu) 40 %; alu: queue (alu) 26 %; alu: stall count after alu 4 % | default latency 726, default rbar 1056, issue-only 2376 | batch r4 |
| gemv_w4a16_n4096_k4096_warm graph | 7 | us | 13.35 | 12.31 | -7.8 % | -7.7 % (r4) | xu: queue (xu) 40 %; alu: queue (alu) 26 %; alu: stall count after alu 4 % | default latency 726, default rbar 1056, issue-only 2376 | batch r4 |
| rmsnorm_m512_n4096_rotating kernel | 7 | us | 41.22 | 43.06 | +4.5 % | -49.1 % (r2c) | l2/dram: load latency (incl. DRAM queue) 49 %; l1: hit latency 39 %; launch: launch 7 % | default latency 308, issue-only 660 | batch r4 |
| softmax_m512_n4096_warm kernel | 7 | us | 9.60 | 8.19 | -14.7 % | -22.8 % (r2) | launch: launch 35 %; store_port: drain after the last instruction 12 %; alu: queue (alu) 8 % | branch fallback 1408, default latency 1672, default rbar 440, issue-only 4488 | batch r4 |
| l2reuse n2_cg_in4 | 5 | us | 7.46 | 1.78 | -76.2 % | -70.7 % (r3) | l2/dram: load latency (incl. DRAM queue) 83 %; store_port: drain after the last instruction 4 %; lsu: queue 4 % | default latency 161, issue-only 513 | batch r3 |
| l2reuse n8_ca_cold | 5 | us | 27.23 | 27.18 | -0.2 % | -0.7 % (r3) | l2/dram: load latency (incl. DRAM queue) 99 %; lsu: queue 1 % | default latency 161, issue-only 321 | batch r3 |
| l2reuse n16_cs_warm | 5 | us | 53.41 | 54.32 | +1.7 % | -88.1 % (r3) | l2/dram: load latency (incl. DRAM queue) 99 % | default latency 161, issue-only 513 | batch r3 |
| l2reuse n20_cg_warm | 5 | us | 9.18 | 8.04 | -12.5 % | -14.4 % (r3) | l2_return: queue (chip L2 share) 86 %; l2/dram: load latency (incl. DRAM queue) 9 %; lsu: queue 2 % | default latency 161, issue-only 609 | batch r3 |
| l2reuse n22_cg_warm | 5 | us | 9.82 | 8.81 | -10.3 % | -12.3 % (r3) | l2_return: queue (chip L2 share) 86 %; l2/dram: load latency (incl. DRAM queue) 9 %; store_port: drain after the last instruction 1 % | default latency 161, issue-only 769 | batch r3 |
| l2reuse n24_cg_cold | 5 | us | 80.22 | 81.46 | +1.5 % | -0.1 % (r3) | l2/dram: load latency (incl. DRAM queue) 99 % | default latency 161, issue-only 705 | batch r3 |
| l2reuse n27_ca_warm | 5 | us | 21.15 | 11.94 | -43.5 % | -50.3 % (r3) | l2_return: queue (chip L2 share) 64 %; l2/dram: load latency (incl. DRAM queue) 32 %; lsu: queue 1 % | default latency 161, issue-only 897 | batch r3 |
| l2reuse n32_cg_cold | 5 | us | 106.66 | 108.60 | +1.8 % | +0.0 % (r3) | l2/dram: load latency (incl. DRAM queue) 100 % | default latency 161, issue-only 897 | batch r3 |
| l2reuse n43_cg_warm | 5 | us | 124.22 | 123.97 | -0.2 % | -86.7 % (r3) | l2/dram: load latency (incl. DRAM queue) 100 % | default latency 161, issue-only 1409 | batch r3 |
| launch empty_g1_t128 | 1 | us | 2.43 | 2.80 | +15.2 % | +7.0 % (r3) | issue: last instruction 50 %; issue: stall count after other 50 % | issue-only 4 | batch r4 |
| launch empty_g96_t128 | 1 | us | 2.40 | 2.80 | +16.7 % | +7.0 % (r3) | issue: scheduler (another warp issued) 50 %; issue: last instruction 25 %; issue: stall count after other 25 % | issue-only 8 | batch r4 |
| launch empty_g2048_t128 | 1 | us | 2.91 | 2.93 | +0.6 % | -9.5 % (r3) | issue: scheduler (another warp issued) 97 %; issue: last instruction 2 %; issue: stall count after other 1 % | issue-only 172 | batch r4 |
| launch empty_g4096_t128 | 1 | us | 4.67 | 4.41 | -5.7 % | -42.5 % (r3) | issue: scheduler (another warp issued) 98 %; issue: last instruction 1 %; issue: stall count after other 1 % | issue-only 344 | batch r4 |
| launch tiny_g48_t1024 | 1 | us | 2.46 | 3.47 | +40.8 % | +33.8 % (r3) | store_port: drain after the last instruction 68 %; store_queue: queue (stg) 27 %; issue: scheduler (another warp issued) 2 % | default latency 160, issue-only 32 | batch r4 |
| launch graph_g512_t128 | 1 | us | 0.60 | 0.61 | +2.4 % | -26.2 % (r4) | issue: scheduler (another warp issued) 87 %; issue: last instruction 9 %; issue: stall count after other 4 % | issue-only 44 | batch r4 |
| launch b2b_work_i0 | 1 | us | 1.90 | 1.96 | +3.2 % | +68.9 % (r4) | issue: default result latency (S2R) 49 %; alu: stall count after alu 22 %; issue: stall count after other 10 % | default latency 7, issue-only 9 | batch r4 |
| fp8gemm2048_c128x128x64_s3_t128 kernel | 7 | us | 123.23 | 129.46 | +5.0 % | -6.5 % (r3) | tensor: queue (mma) 35 %; tensor: stall count after mma 19 %; store_queue: queue (stg) 19 % | default latency 192, default rbar 24, issue-only 3360 | batch r4 |
| fp8gemm2048_c128x256x64_s3_t256 kernel | 7 | us | 137.66 | 141.50 | +2.8 % | -11.0 % (r3) | tensor: queue (mma) 30 %; store_queue: queue (stg) 22 %; tensor: stall count after mma 17 % | default latency 192, issue-only 3360 | batch r4 |
| dram_rw r2_w0 | 5 | us | 267.49 | 274.06 | +2.5 % | +0.4 % (r4) | l2/dram: load latency (incl. DRAM queue) 100 % | default latency 128, issue-only 3936 | batch r4 |
| dram_rw r1_w1 | 5 | us | 289.25 | 301.49 | +4.2 % | -53.0 % (r4) | l2/dram: load latency (incl. DRAM queue) 100 % | default latency 160, issue-only 5216 | batch r4 |
| dram_rw r1_w2 | 5 | us | 432.22 | 464.60 | +7.5 % | -68.6 % (r4) | l2/dram: load latency (incl. DRAM queue) 100 % | default latency 160, issue-only 5216 | batch r4 |
| hgemm2048_c128x128x32_s3_t128_rotating kernel | 7 | us | 229.76 | 235.20 | +2.4 % | -12.5 % (r4) | tensor: queue (mma) 24 %; dram_queue: queue (DRAM share) 22 %; tensor: stall count after mma 16 % | default latency 192, default rbar 1512, issue-only 10776 | batch r10 |
| hgemm2048_c128x128x32_s3_t128_sw8_rotating kernel | 7 | us | 223.58 | 224.29 | +0.3 % | -14.3 % (r4) | tensor: queue (mma) 27 %; dram_queue: queue (DRAM share) 19 %; tensor: stall count after mma 17 % | default latency 264, default rbar 1572, issue-only 10812 | batch r10 |
| hgemm2048_c128x128x32_s4_t128_rotating kernel | 7 | us | 267.68 | 295.82 | +10.5 % | -10.7 % (r4) | tensor: stall count after mma 28 %; dram_queue: queue (DRAM share) 17 %; tensor: queue (mma) 15 % | default latency 192, default rbar 24, issue-only 4704 | batch r10 |
| hgemm2048_c128x128x64_s2_t128_rotating kernel | 7 | us | 252.03 | 289.95 | +15.0 % | -5.8 % (r4) | tensor: stall count after mma 28 %; tensor: queue (mma) 16 %; dram_queue: queue (DRAM share) 14 % | default latency 240, default rbar 24, issue-only 3288 | batch r10 |
| hgemm2048_c128x128x64_s3_t256_rotating kernel | 7 | us | 250.30 | 290.89 | +16.2 % | -10.9 % (r4) | tensor: queue (mma) 22 %; dram_queue: queue (DRAM share) 17 %; tensor: stall count after mma 14 % | default latency 480, default rbar 48, issue-only 10752 | batch r10 |
| hgemm2048_c128x256x32_s3_t256_rotating kernel | 7 | us | 241.73 | 256.48 | +6.1 % | -7.9 % (r4) | issue: stall count after barrier 23 %; tensor: stall count after mma 15 %; dram_queue: queue (DRAM share) 14 % | default latency 216, default rbar 1512, issue-only 13776 | batch r10 |
| hgemm2048_c128x256x32_s3_t256_sw4_rotating kernel | 7 | us | 236.93 | 255.47 | +7.8 % | -9.1 % (r4) | tensor: queue (mma) 24 %; tensor: stall count after mma 16 %; issue: stall count after barrier 15 % | default latency 264, default rbar 1568, issue-only 13808 | batch r10 |
| hgemm2048_c128x256x64_s2_t256_rotating kernel | 7 | us | 228.86 | 254.68 | +11.3 % | -9.5 % (r4) | issue: stall count after barrier 20 %; tensor: queue (mma) 16 %; tensor: stall count after mma 15 % | default latency 240, default rbar 24, issue-only 3288 | batch r10 |
| hgemm2048_c256x128x32_s3_t256_rotating kernel | 7 | us | 239.07 | 256.58 | +7.3 % | -7.2 % (r4) | tensor: queue (mma) 33 %; dram_queue: queue (DRAM share) 16 %; tensor: stall count after mma 14 % | default latency 216, default rbar 1512, issue-only 13752 | batch r10 |
| hgemm2048_c64x128x32_s4_t128_rotating kernel | 7 | us | 255.14 | 266.73 | +4.5 % | -8.6 % (r4) | l2_return: queue (chip L2 share) 23 %; dram_queue: queue (DRAM share) 20 %; tensor: queue (mma) 15 % | default latency 396, issue-only 7172 | batch r10 |
| moe_e8_t2048_n1024_k2048_c128x128x32_s3_t128 kernel | 7 | us | 179.55 | 191.31 | +6.5 % | +223.9 % (r4) | dram_queue: queue (DRAM share) 55 %; tensor: stall count after mma 11 %; dram: DRAM latency 8 % | default latency 108, default rbar 756, issue-only 5388 | batch r4 |
| moe_e8_t2048_n1024_k2048_c64x128x32_s4_t128 kernel | 7 | us | 173.12 | 184.67 | +6.7 % | -64.1 % (r4) | dram_queue: queue (DRAM share) 64 %; dram: DRAM latency 14 %; store_queue: queue (stg) 5 % | default latency 216, issue-only 3912 | batch r4 |
| mma_types f32_bf16_chain | 2 | cycles/instr/subpartition | 29.05 | 29.07 | +0.1 % | +0.1 % (r5) | issue: stall count after other 95 %; tensor: stall count after mma 3 %; alu: stall count after alu 1 % | default latency 4, issue-only 775 | batch r5 |
| mma_types f32_e4m3_w1 | 2 | cycles/instr/subpartition | 17.72 | 16.83 | -5.0 % | -5.0 % (r5) | tensor: stall count after mma 78 %; issue: stall count after other 21 %; xu: xu result latency 1 % | default latency 4, issue-only 4645 | batch r5 |
| mma_types f32_e5m2_chain | 2 | cycles/instr/subpartition | 29.05 | 29.07 | +0.1 % | +0.1 % (r5) | issue: stall count after other 95 %; tensor: stall count after mma 3 %; alu: stall count after alu 1 % | default latency 4, issue-only 775 | batch r5 |
| mma_types s32_s8_w1 | 2 | cycles/instr/subpartition | 17.59 | 16.71 | -5.0 % | -5.1 % (r5) | tensor: stall count after mma 79 %; issue: stall count after other 21 % | default latency 4, issue-only 4645 | batch r5 |
| mma_types s32_s8_w4 | 2 | cycles/instr/subpartition | 17.59 | 16.71 | -5.0 % | -5.1 % (r5) | tensor: stall count after mma 79 %; issue: stall count after other 21 % | default latency 13, issue-only 18580 | batch r5 |
| row_reduce empty_g48_span | 4 | cycles | 8.00 | 19.00 | +137.5 % | +137.5 % (r5) | issue: default result latency (S2UR) 22 %; issue: default result latency (S2R) 22 %; issue: stall count after other 18 % | default latency 8, issue-only 13 | batch r5 |
| row_reduce empty_g512_span | 4 | cycles | 229.00 | 179.00 | -21.8 % | -27.9 % (r5) | issue: scheduler (another warp issued) 59 %; issue: stall count after other 8 %; issue: default result latency (S2UR) 8 % | default latency 88, issue-only 143 | batch r5 |
| row_reduce red_g48_span | 4 | cycles | 125.00 | 130.00 | +4.0 % | +8.0 % (r5) | issue: stall count after other 24 %; smem_array: LDS latency 19 %; alu: stall count after alu 14 % | branch fallback 1, default latency 15, issue-only 34 | batch r5 |
| row_reduce ld_red_st_g48_span | 4 | cycles | 1,195.00 | 1,215.08 | +1.7 % | -20.9 % (r5) | store_port: drain after the last instruction 36 %; l2/dram: load latency (incl. DRAM queue) 16 %; l2_return: queue (chip L2 share) 15 % | default latency 26, issue-only 25 | batch r5 |
| w4gemm2048_c128x128x32_s3_t128_rotating kernel | 7 | us | 286.08 | 293.39 | +2.6 % | -15.9 % (r5) | tensor: queue (mma) 22 %; tensor: stall count after mma 15 %; smem_array: LDS latency 11 % | default latency 216, default rbar 24, issue-only 15072 | batch r5 |
| w4gemm2048_c128x128x32_s3_t128_warm kernel | 7 | us | 288.00 | 294.21 | +2.2 % | -10.1 % (r5) | tensor: queue (mma) 22 %; tensor: stall count after mma 15 %; smem_array: LDS latency 11 % | default latency 216, default rbar 24, issue-only 15072 | batch r5 |
| w4gemm2048_c64x128x32_s4_t128_rotating kernel | 7 | us | 430.75 | 439.42 | +2.0 % | -10.3 % (r5) | tensor: queue (mma) 19 %; smem_array: LDS latency 15 %; xu: xu result latency 10 % | default latency 396, default rbar 2684, issue-only 9856 | batch r5 |
| w4gemm2048_c64x128x32_s4_t128_warm kernel | 7 | us | 432.29 | 439.35 | +1.6 % | -13.1 % (r5) | tensor: queue (mma) 20 %; smem_array: LDS latency 16 %; xu: xu result latency 10 % | default latency 396, default rbar 2684, issue-only 9856 | batch r5 |
| st256 st_g2 | 4 | cycles | 205.00 | 123.20 | -39.9 % | -14.6 % (r5b) | store_port: queue (stores) 56 %; issue: stall count after other 14 %; issue: default result latency (S2R) 10 % | default latency 18, issue-only 49 | batch r5b |
| st256 st_g16 | 4 | cycles | 205.00 | 123.20 | -39.9 % | -14.6 % (r5b) | store_port: drain after the last instruction 49 %; store_port: queue (stores) 31 %; issue: stall count after other 8 % | default latency 18, issue-only 49 | batch r5b |
| st256 st_g48 | 4 | cycles | 194.00 | 123.20 | -36.5 % | +32.2 % (r5b) | store_port: drain after the last instruction 81 %; store_port: queue (stores) 11 %; issue: stall count after other 3 % | default latency 18, issue-only 49 | batch r5b |
| st256 st_g96 | 4 | cycles | 463.00 | 458.32 | -1.0 % | +42.6 % (r5b) | store_port: drain after the last instruction 57 %; store_port: queue (stores) 38 %; issue: stall count after other 1 % | default latency 36, issue-only 98 | batch r5b |
| st256 ld_g1 | 4 | cycles | 564.00 | 580.00 | +2.8 % | +2.7 % (r5b) | l2/dram: load latency (incl. DRAM queue) 47 %; l2_return: queue (SM port rate) 25 %; lsu: queue 7 % | default latency 18, issue-only 17 | batch r5b |
| branch loop8_w4 | 2 | cycles | 24.01 | 24.05 | +0.2 % | -24.8 % (r6) | issue: stall count after other 54 %; fma: stall count after fp32 45 % | default latency 9, issue-only 6160 | batch r6 |
| branch loop16_w1 | 2 | cycles | 29.02 | 29.04 | +0.1 % | -17.2 % (r6) | fma: stall count after fp32 55 %; issue: stall count after other 45 % | default latency 3, issue-only 1540 | batch r6 |
| branch flat64_w1 | 2 | cycles | 1.03 | 1.23 | +19.7 % | +7.6 % (r6) | fma: stall count after fp32 48 %; alu: stall count after alu 23 %; issue: default result latency (LDC) 10 % | default latency 3, issue-only 3 | batch r6 |
| branch flat256_w2 | 2 | cycles | 1.12 | 1.16 | +3.1 % | -8.0 % (r6) | fma: stall count after fp32 75 %; alu: stall count after alu 11 %; issue: default result latency (LDC) 5 % | default latency 5, issue-only 6 | batch r6 |
| branch flat1024_w1 | 2 | cycles | 1.12 | 1.13 | +0.6 % | -10.5 % (r6) | fma: stall count after fp32 92 %; alu: stall count after alu 4 %; issue: default result latency (LDC) 2 % | default latency 3, issue-only 3 | batch r6 |
| branch flat4096_w1 | 2 | cycles | 1.18 | 1.13 | -4.3 % | -14.9 % (r6) | fma: stall count after fp32 98 %; alu: stall count after alu 1 % | default latency 3, issue-only 3 | batch r6 |
| branch fwd_taken_w1 | 2 | cycles | 43.15 | 43.05 | -0.2 % | -16.5 % (r6) | issue: stall count after other 62 %; fma: stall count after fp32 37 % | default latency 4, issue-only 3076 | batch r6 |
| mma_issue n16 | 2 | cycles/mma | 17.72 | 16.83 | -5.0 % | -5.0 % (r6) | tensor: stall count after mma 79 %; issue: stall count after other 21 % | default latency 4, issue-only 4645 | batch r6 |
| mma_issue mfma_8 | 2 | cycles/mma | 26.26 | 24.19 | -7.9 % | -31.9 % (r6) | fma: stall count after fp32 56 %; tensor: queue (mma) 26 %; issue: stall count after other 10 % | default latency 4, issue-only 1045 | batch r6 |
| mma_issue malu_2 | 2 | cycles/mma | 18.52 | 17.72 | -4.3 % | -4.7 % (r6) | tensor: stall count after mma 65 %; issue: stall count after other 32 %; alu: stall count after alu 2 % | default latency 4, issue-only 2582 | batch r6 |
| mma_issue mlds_1 | 2 | cycles/mma | 18.41 | 17.65 | -4.1 % | -4.1 % (r6) | tensor: stall count after mma 65 %; issue: stall count after other 35 % | default latency 5, issue-only 2583 | batch r6 |
| mma_issue mlds_4 | 2 | cycles/mma | 18.42 | 17.66 | -4.1 % | -4.1 % (r6) | tensor: stall count after mma 65 %; issue: stall count after other 35 % | default latency 5, issue-only 2582 | batch r6 |
| mma_issue nbr_lds | 2 | cycles/mma | 18.21 | 17.27 | -5.1 % | -5.2 % (r6) | tensor: stall count after mma 82 %; issue: stall count after other 17 %; alu: stall count after alu 1 % | default latency 17, issue-only 4767 | batch r6 |
| lds_narrow LDS.U8_s1_w32 | 2 | cycles/instr/SM | 2.00 | 2.05 | +2.1 % | +2.5 % (r6) | alu: queue (alu) 38 %; smem_array: LDS latency 23 %; alu: stall count after alu 18 % | default latency 161, issue-only 24768 | batch r6 |
| lds_narrow LDS.U8_s64_w4 | 2 | cycles/instr/SM | 17.79 | 17.86 | +0.4 % | -41.8 % (r6) | smem_array: LDS latency 81 %; alu: stall count after alu 16 %; issue: stall count after other 2 % | default latency 21, issue-only 3096 | batch r6 |
| lds_narrow LDS.U16_s16_w4 | 2 | cycles/instr/SM | 11.79 | 11.85 | +0.5 % | -12.2 % (r6) | smem_array: LDS latency 72 %; alu: stall count after alu 25 %; issue: stall count after other 3 % | default latency 21, issue-only 3096 | batch r6 |
| lds_narrow LDS.32_s4_w32 | 2 | cycles/instr/SM | 2.00 | 2.05 | +2.1 % | +2.5 % (r6) | alu: queue (alu) 38 %; smem_array: LDS latency 23 %; alu: stall count after alu 18 % | default latency 161, issue-only 24768 | batch r6 |
| lds_narrow STS.U16_s2_w1 | 2 | cycles/instr/SM | 5.20 | 5.15 | -0.9 % | -0.9 % (r6) | issue: stall count after smem_op 55 %; issue: stall count after other 27 %; mio: queue (smem_op) 17 % | default latency 5, issue-only 774 | batch r6 |
| lds_narrow STS.U16_s16_w1 | 2 | cycles/instr/SM | 8.04 | 8.03 | -0.2 % | -35.9 % (r6) | mio: queue (smem_op) 46 %; issue: stall count after smem_op 36 %; issue: stall count after other 17 % | default latency 5, issue-only 774 | batch r6 |
| lds_narrow STS.32_s64_w1 | 2 | cycles/instr/SM | 31.90 | 32.02 | +0.4 % | -83.8 % (r6) | mio: queue (smem_op) 87 %; issue: stall count after smem_op 9 %; issue: stall count after other 4 % | default latency 5, issue-only 774 | batch r6 |
| launch_short b2b_work_i0 | 1 | us | 1.84 | 1.96 | +6.3 % | +54.5 % (r6) | issue: default result latency (S2R) 49 %; alu: stall count after alu 22 %; issue: stall count after other 10 % | default latency 7, issue-only 9 | batch r6 |
| launch_short graph_work_i8 | 1 | us | 0.82 | 0.79 | -4.1 % | +2.3 % (r6) | issue: stall count after other 38 %; store_port: drain after the last instruction 18 %; store_port: queue (stores) 15 % | default latency 14, issue-only 113 | batch r6 |
| launch_short work_i16 | 1 | us | 2.50 | 2.53 | +1.4 % | +24.1 % (r6) | issue: stall count after other 53 %; store_port: drain after the last instruction 12 %; store_port: queue (stores) 11 % | default latency 14, issue-only 209 | batch r6 |
| launch_short work_i24 | 1 | us | 2.53 | 2.53 | +0.1 % | +24.7 % (r6) | issue: stall count after other 61 %; store_port: drain after the last instruction 10 %; fma: stall count after fp32 9 % | default latency 14, issue-only 305 | batch r6 |
| launch_short work_i32 | 1 | us | 2.53 | 2.53 | +0.1 % | +27.0 % (r6) | issue: stall count after other 66 %; fma: stall count after fp32 9 %; store_port: drain after the last instruction 8 % | default latency 14, issue-only 401 | batch r6 |
| launch_short work_i40 | 1 | us | 2.43 | 2.53 | +4.0 % | +34.3 % (r6) | issue: stall count after other 69 %; fma: stall count after fp32 10 %; store_port: drain after the last instruction 7 % | default latency 14, issue-only 497 | batch r6 |
| launch_short work_i48 | 1 | us | 2.56 | 2.53 | -1.2 % | +29.8 % (r6) | issue: stall count after other 71 %; fma: stall count after fp32 10 %; store_port: drain after the last instruction 6 % | default latency 14, issue-only 593 | batch r6 |
| decode_seq_s4096_rotating kernel | 7 | us | 70.94 | 79.73 | +12.4 % | +14.9 % (r6) | dram_queue: queue (DRAM share) 71 %; dram: DRAM latency 12 %; launch: launch 4 % | branch fallback 266, default latency 356, default rbar 496, issue-only 1604 | batch r6 |
| decode_seq_s4096_warm kernel | 7 | us | 28.10 | 29.14 | +3.7 % | -6.5 % (r6) | tensor: stall count after mma 17 %; lsu: queue 13 %; tensor: queue (mma) 12 % | branch fallback 266, default latency 356, default rbar 496, issue-only 1604 | batch r6 |
| decode_seq_s4096_warm graph | 7 | us | 23.97 | 25.35 | +5.8 % | -8.5 % (r6) | tensor: stall count after mma 19 %; lsu: queue 14 %; tensor: queue (mma) 13 % | branch fallback 266, default latency 356, default rbar 496, issue-only 1604 | batch r6 |
| fa_fwd_s1024_h16_d64 kernel | 7 | us | 66.05 | 63.58 | -3.7 % | -15.1 % (r6) | tensor: queue (mma) 30 %; tensor: stall count after mma 10 %; fma: stall count after fp32 10 % | branch fallback 768, default latency 984, default rbar 3048, issue-only 5448 | batch r6 |
| hgemm_m1024n1024k1024_c128x128x32_s4_t128 kernel | 7 | us | 47.52 | 49.64 | +4.5 % | -5.7 % (r6) | tensor: stall count after mma 32 %; tensor: queue (mma) 17 %; store_queue: queue (stg) 16 % | default latency 72, default rbar 8, issue-only 1048 | batch r9 |
| hgemm_m1024n1024k1024_c128x128x32_s4_t128_rotating kernel | 7 | us | 52.29 | 56.43 | +7.9 % | -11.3 % (r6) | dram_queue: queue (DRAM share) 29 %; tensor: stall count after mma 18 %; store_queue: queue (stg) 14 % | default latency 72, default rbar 8, issue-only 1048 | batch r9 |
| hgemm_m1024n1024k1024_c128x128x64_s3_t256 kernel | 7 | us | 44.64 | 48.44 | +8.5 % | -5.4 % (r6) | tensor: queue (mma) 20 %; issue: stall count after barrier 20 %; store_queue: queue (stg) 16 % | default latency 160, default rbar 16, issue-only 2048 | batch r9 |
| hgemm_m1024n1024k1024_c128x128x64_s3_t256_rotating kernel | 7 | us | 49.44 | 56.10 | +13.5 % | -13.1 % (r6) | dram_queue: queue (DRAM share) 28 %; tensor: queue (mma) 16 %; store_queue: queue (stg) 14 % | default latency 160, default rbar 16, issue-only 2048 | batch r9 |
| hgemm_m1024n1024k1024_c64x128x32_s4_t128 kernel | 7 | us | 39.14 | 39.37 | +0.6 % | -2.8 % (r6) | l2_return: queue (chip L2 share) 27 %; store_queue: queue (stg) 14 %; tensor: stall count after mma 14 % | default latency 108, issue-only 1188 | batch r9 |
| hgemm_m1024n1024k1024_c64x128x32_s4_t128_rotating kernel | 7 | us | 42.56 | 45.94 | +7.9 % | -12.6 % (r6) | dram_queue: queue (DRAM share) 37 %; store_queue: queue (stg) 12 %; tensor: stall count after mma 11 % | default latency 108, issue-only 1188 | batch r9 |
| rmsnorm_m2048_n1024_warm kernel | 7 | us | 8.70 | 7.70 | -11.6 % | -30.1 % (r6) | launch: launch 37 %; store_queue: queue (stg) 20 %; store_port: drain after the last instruction 10 % | default latency 1204, default rbar 172, issue-only 516 | batch r6 |
| softmax_m1024_n2048_warm kernel | 7 | us | 8.70 | 7.51 | -13.7 % | -18.8 % (r6) | launch: launch 38 %; issue: scheduler (another warp issued) 11 %; l2_return: queue (chip L2 share) 11 % | branch fallback 1408, default latency 1936, default rbar 528, issue-only 4576 | batch r6 |
| mma_pipe v13_0_a1_b1_w8 | 2 | cycles/mma | 16.02 | 16.02 | +0.0 % | +0.0 % (r6b) | tensor: queue (mma) 47 %; tensor: stall count after mma 41 %; issue: stall count after other 11 % | default latency 25, issue-only 37160 | batch r6b |
| mma_pipe v13_0_a1_b4_w8 | 2 | cycles/mma | 16.06 | 16.48 | +2.6 % | -0.3 % (r6b) | tensor: queue (mma) 48 %; tensor: stall count after mma 40 %; issue: stall count after other 11 % | default latency 25, issue-only 43304 | batch r6b |
| mma_pipe v13_2_a1_b1_w1 | 2 | cycles/mma | 16.84 | 16.83 | -0.1 % | +5.1 % (r6b) | tensor: stall count after mma 58 %; tensor: queue (mma) 28 %; issue: stall count after other 14 % | default latency 4, issue-only 805 | batch r6b |
| mma_pipe v13_2_a1_b1_w8 | 2 | cycles/mma | 16.02 | 16.02 | +0.0 % | +0.0 % (r6b) | tensor: queue (mma) 62 %; tensor: stall count after mma 30 %; issue: stall count after other 7 % | default latency 25, issue-only 6440 | batch r6b |
| mma_pipe v13_2_a8_b1_w1 | 2 | cycles/mma | 17.76 | 17.78 | +0.1 % | +3.2 % (r6b) | tensor: stall count after mma 50 %; tensor: queue (mma) 27 %; issue: stall count after other 23 % | default latency 4, issue-only 8229 | batch r6b |
| mma_pipe v13_2_a1_b4_w1 | 2 | cycles/mma | 17.01 | 17.01 | +0.0 % | +5.2 % (r6b) | tensor: stall count after mma 57 %; tensor: queue (mma) 27 %; issue: stall count after other 15 % | default latency 4, issue-only 1573 | batch r6b |
| mma_pipe v13_2_a1_b8_w8 | 2 | cycles/mma | 16.29 | 16.29 | -0.0 % | -0.2 % (r6b) | tensor: queue (mma) 61 %; tensor: stall count after mma 30 %; issue: stall count after other 9 % | default latency 25, issue-only 20776 | batch r6b |
| dram_burst t256_n5 | 5 | us | 20.99 | 19.69 | -6.2 % | -5.9 % (r7) | l2/dram: load latency (incl. DRAM queue) 98 %; l2_return: transfer 1 % | default latency 32, issue-only 504 | batch r7 |
| dram_burst t256_n21 | 5 | us | 74.24 | 73.99 | -0.3 % | -1.9 % (r7) | l2/dram: load latency (incl. DRAM queue) 98 %; l2_return: transfer 1 % | default latency 32, issue-only 2040 | batch r7 |
| dram_burst t256_n42 | 5 | us | 145.76 | 145.25 | -0.3 % | -2.2 % (r7) | l2/dram: load latency (incl. DRAM queue) 98 %; l2_return: transfer 1 % | default latency 32, issue-only 4056 | batch r7 |
| dram_burst t1024_n10 | 5 | us | 35.42 | 36.65 | +3.5 % | +2.6 % (r7) | l2/dram: load latency (incl. DRAM queue) 99 % | default latency 128, issue-only 1056 | batch r7 |
| l2mix L2_S1_W32_D32_P0_g96 | 4 | cycles | 1,280.00 | 1,411.04 | +10.2 % | +23.4 % (r7) | l2_return: queue (chip L2 share) 47 %; l2/dram: load latency (incl. DRAM queue) 20 %; store_port: drain after the last instruction 14 % | default latency 44, issue-only 50 | batch r7 |
| l2mix L1_S4_W32_D32_P0_g96 | 4 | cycles | 1,111.00 | 1,402.55 | +26.2 % | +112.2 % (r7) | store_port: queue (stores) 40 %; store_port: drain after the last instruction 35 %; l2_return: queue (chip L2 share) 10 % | default latency 44, issue-only 154 | batch r7 |
| l2mix L4_S1_W32_D32_P1_g96 | 4 | cycles | 2,303.00 | 2,394.08 | +4.0 % | +10.4 % (r7) | l2_return: queue (chip L2 share) 47 %; l2/dram: load latency (incl. DRAM queue) 23 %; store_port: drain after the last instruction 17 % | default latency 44, issue-only 50 | batch r7 |
| l2mix L2_S1_W32_D32_P1_g96 | 4 | cycles | 1,278.00 | 1,411.04 | +10.4 % | +23.6 % (r7) | l2_return: queue (chip L2 share) 47 %; l2/dram: load latency (incl. DRAM queue) 20 %; store_port: drain after the last instruction 14 % | default latency 44, issue-only 50 | batch r7 |
| l2mix L0_S2_W16_D16_P0_g96 | 4 | cycles | 134.00 | 220.00 | +64.2 % | +62.8 % (r7) | store_port: drain after the last instruction 62 %; store_port: queue (stores) 32 %; fma: queue (fma_half_rate) 2 % | default latency 36, issue-only 50 | batch r7 |
| l2mix L0_S2_W16_D64_P0_g96 | 4 | cycles | 132.00 | 215.00 | +62.9 % | +191.2 % (r7) | store_port: drain after the last instruction 59 %; store_port: queue (stores) 38 %; fma: queue (fma_half_rate) 1 % | default latency 36, issue-only 58 | batch r7 |
| l2mix L0_S2_W16_D64_P0_g512 | 4 | cycles | 23,057.00 | 8,159.12 | -64.6 % | -76.3 % (r7) | store_queue: queue (stg) 79 %; store_port: drain after the last instruction 11 %; store_port: queue (stores) 8 % | default latency 198, issue-only 319 | batch r7 |
| l2mix L0_S2_W4_D4_P0_g96 | 4 | cycles | 92.00 | 68.00 | -26.1 % | -28.3 % (r7) | store_port: drain after the last instruction 54 %; store_port: queue (stores) 11 %; issue: default result latency (S2UR) 7 % | default latency 36, issue-only 26 | batch r7 |
| fa_s1536_h12_d96_c64x32_s3_t128 kernel | 7 | us | 142.75 | 134.02 | -6.1 % | -11.3 % (r7) | tensor: queue (mma) 33 %; tensor: stall count after mma 9 %; fma: stall count after fp32 8 % | branch fallback 1152, default latency 1368, default rbar 3984, issue-only 15912 | batch r9 |
| fa_s1536_h12_d96_c64x64_s1_t128 kernel | 7 | us | 131.55 | 131.39 | -0.1 % | -8.1 % (r7) | tensor: queue (mma) 36 %; tensor: stall count after mma 10 %; fma: stall count after fp32 8 % | branch fallback 1152, default latency 1368, default rbar 4200, issue-only 9312 | batch r9 |
| hgemm_m3072n1536k2048_c128x128x32_s2_t128 kernel | 7 | us | 213.47 | 217.68 | +2.0 % | -1.1 % (r7) | tensor: queue (mma) 40 %; tensor: stall count after mma 18 %; store_queue: queue (stg) 10 % | default latency 240, default rbar 48, issue-only 4848 | batch r9 |
| hgemm_m3072n1536k2048_c128x128x32_s2_t128_rotating kernel | 7 | us | 224.80 | 236.50 | +5.2 % | -6.1 % (r7) | tensor: queue (mma) 21 %; dram_queue: queue (DRAM share) 14 %; l2_return: queue (chip L2 share) 14 % | default latency 240, default rbar 48, issue-only 4848 | batch r9 |
| hgemm_m3072n1536k2048_c128x128x32_s3_t128 kernel | 7 | us | 229.44 | 242.20 | +5.6 % | -7.3 % (r7) | tensor: queue (mma) 36 %; tensor: stall count after mma 20 %; lsu: queue 16 % | default latency 216, default rbar 1512, issue-only 10728 | batch r9 |
| hgemm_m3072n1536k2048_c128x128x32_s3_t128_rotating kernel | 7 | us | 232.96 | 225.44 | -3.2 % | -13.0 % (r7) | tensor: queue (mma) 32 %; tensor: stall count after mma 16 %; dram_queue: queue (DRAM share) 16 % | default latency 216, default rbar 1512, issue-only 10728 | batch r9 |
| hgemm_m3072n1536k2048_c128x128x32_s3_t128_sw8 kernel | 7 | us | 230.56 | 234.47 | +1.7 % | -6.3 % (r7) | tensor: queue (mma) 39 %; tensor: stall count after mma 20 %; store_queue: queue (stg) 15 % | default latency 264, default rbar 1568, issue-only 10808 | batch r9 |
| hgemm_m3072n1536k2048_c128x128x32_s3_t128_sw8_rotating kernel | 7 | us | 229.66 | 225.75 | -1.7 % | -5.9 % (r7) | tensor: queue (mma) 28 %; dram_queue: queue (DRAM share) 19 %; tensor: stall count after mma 17 % | default latency 264, default rbar 1568, issue-only 10808 | batch r9 |
| hgemm_m3072n1536k2048_c128x128x32_s4_t128 kernel | 7 | us | 254.37 | 268.46 | +5.5 % | -3.9 % (r7) | tensor: stall count after mma 36 %; tensor: queue (mma) 19 %; lsu: queue 16 % | default latency 216, default rbar 24, issue-only 4656 | batch r9 |
| hgemm_m3072n1536k2048_c128x128x32_s4_t128_rotating kernel | 7 | us | 269.60 | 288.94 | +7.2 % | -8.5 % (r7) | tensor: stall count after mma 29 %; tensor: queue (mma) 16 %; dram_queue: queue (DRAM share) 13 % | default latency 216, default rbar 24, issue-only 4656 | batch r9 |
| hgemm_m3072n1536k2048_c128x128x64_s2_t128 kernel | 7 | us | 238.34 | 259.80 | +9.0 % | -3.1 % (r7) | tensor: stall count after mma 35 %; tensor: queue (mma) 20 %; lsu: queue 16 % | default latency 240, default rbar 48, issue-only 3336 | batch r9 |
| hgemm_m3072n1536k2048_c128x128x64_s2_t128_rotating kernel | 7 | us | 252.38 | 283.30 | +12.2 % | -6.4 % (r7) | tensor: stall count after mma 28 %; tensor: queue (mma) 16 %; lsu: queue 14 % | default latency 240, default rbar 48, issue-only 3336 | batch r9 |
| hgemm_m3072n1536k2048_c128x128x64_s3_t256 kernel | 7 | us | 234.88 | 262.42 | +11.7 % | -3.6 % (r7) | tensor: queue (mma) 26 %; issue: stall count after barrier 17 %; tensor: stall count after mma 17 % | default latency 480, default rbar 96, issue-only 10848 | batch r9 |
| hgemm_m3072n1536k2048_c128x128x64_s3_t256_rotating kernel | 7 | us | 251.10 | 285.33 | +13.6 % | -9.2 % (r7) | tensor: queue (mma) 21 %; issue: stall count after barrier 17 %; tensor: stall count after mma 14 % | default latency 480, default rbar 96, issue-only 10848 | batch r9 |
| hgemm_m3072n1536k2048_c128x256x32_s3_t256 kernel | 7 | us | 246.08 | 251.19 | +2.1 % | -7.5 % (r7) | tensor: queue (mma) 25 %; issue: stall count after barrier 20 %; tensor: stall count after mma 19 % | default latency 216, default rbar 1512, issue-only 13728 | batch r9 |
| hgemm_m3072n1536k2048_c128x256x32_s3_t256_rotating kernel | 7 | us | 247.94 | 253.49 | +2.2 % | -7.6 % (r7) | issue: stall count after barrier 26 %; tensor: stall count after mma 16 %; store_queue: queue (stg) 14 % | default latency 216, default rbar 1512, issue-only 13728 | batch r9 |
| hgemm_m3072n1536k2048_c128x256x32_s3_t256_sw4 kernel | 7 | us | 246.02 | 240.01 | -2.4 % | -8.5 % (r7) | tensor: queue (mma) 28 %; tensor: stall count after mma 20 %; store_queue: queue (stg) 15 % | default latency 264, default rbar 1560, issue-only 13800 | batch r9 |
| hgemm_m3072n1536k2048_c128x256x32_s3_t256_sw4_rotating kernel | 7 | us | 246.91 | 246.08 | -0.3 % | -8.7 % (r7) | tensor: queue (mma) 24 %; tensor: stall count after mma 16 %; store_queue: queue (stg) 14 % | default latency 264, default rbar 1560, issue-only 13800 | batch r9 |
| hgemm_m3072n1536k2048_c128x256x64_s2_t256 kernel | 7 | us | 234.59 | 247.64 | +5.6 % | -7.4 % (r7) | tensor: queue (mma) 42 %; tensor: stall count after mma 18 %; store_queue: queue (stg) 14 % | default latency 240, default rbar 24, issue-only 3336 | batch r9 |
| hgemm_m3072n1536k2048_c128x256x64_s2_t256_rotating kernel | 7 | us | 236.67 | 252.78 | +6.8 % | -7.5 % (r7) | tensor: queue (mma) 30 %; tensor: stall count after mma 15 %; store_queue: queue (stg) 14 % | default latency 240, default rbar 24, issue-only 3336 | batch r9 |
| hgemm_m3072n1536k2048_c128x64x32_s4_t128 kernel | 7 | us | 245.82 | 254.61 | +3.6 % | +0.0 % (r7) | l2_return: queue (chip L2 share) 61 %; store_queue: queue (stg) 11 %; tensor: stall count after mma 6 % | default latency 432, issue-only 7776 | batch r9 |
| hgemm_m3072n1536k2048_c128x64x32_s4_t128_rotating kernel | 7 | us | 259.46 | 274.23 | +5.7 % | -4.3 % (r7) | l2_return: queue (chip L2 share) 49 %; dram_queue: queue (DRAM share) 16 %; store_queue: queue (stg) 10 % | default latency 432, issue-only 7776 | batch r9 |
| hgemm_m3072n1536k2048_c256x128x32_s3_t256 kernel | 7 | us | 244.80 | 251.75 | +2.8 % | -4.9 % (r7) | tensor: queue (mma) 42 %; tensor: stall count after mma 18 %; store_queue: queue (stg) 14 % | default latency 216, default rbar 1512, issue-only 13728 | batch r9 |
| hgemm_m3072n1536k2048_c256x128x32_s3_t256_rotating kernel | 7 | us | 246.72 | 254.25 | +3.1 % | -5.6 % (r7) | tensor: queue (mma) 35 %; tensor: stall count after mma 15 %; store_queue: queue (stg) 14 % | default latency 216, default rbar 1512, issue-only 13728 | batch r9 |
| hgemm_m3072n1536k2048_c256x256x32_s2_t256 kernel | 7 | us | 891.26 | 713.87 | -19.9 % | -60.4 % (r7) | l1: queue 68 %; tensor: queue (mma) 7 %; tensor: stall count after mma 5 % | branch fallback 1008, default latency 3264, front end 8064, issue-only 104416 | batch r9 |
| hgemm_m3072n1536k2048_c256x256x32_s2_t256_rotating kernel | 7 | us | 871.58 | 751.13 | -13.8 % | -58.4 % (r7) | l1: queue 65 %; tensor: queue (mma) 7 %; tensor: stall count after mma 5 % | branch fallback 1008, default latency 3264, front end 8064, issue-only 104416 | batch r9 |
| hgemm_m3072n1536k2048_c64x128x32_s4_t128 kernel | 7 | us | 246.24 | 235.57 | -4.3 % | -1.3 % (r7) | l2_return: queue (chip L2 share) 47 %; tensor: queue (mma) 14 %; tensor: stall count after mma 9 % | default latency 432, issue-only 7776 | batch r9 |
| hgemm_m3072n1536k2048_c64x128x32_s4_t128_rotating kernel | 7 | us | 261.66 | 270.88 | +3.5 % | -5.6 % (r7) | l2_return: queue (chip L2 share) 37 %; dram_queue: queue (DRAM share) 16 %; tensor: queue (mma) 11 % | default latency 432, issue-only 7776 | batch r9 |
| hgemm_m3072n1536k2048_c64x64x32_s4_t64 kernel | 7 | us | 294.14 | 308.42 | +4.9 % | +17.0 % (r7) | l2_return: queue (chip L2 share) 66 %; tensor: queue (mma) 7 %; tensor: stall count after mma 6 % | default latency 432, default rbar 48, issue-only 7776 | batch r9 |
| hgemm_m3072n1536k2048_c64x64x32_s4_t64_rotating kernel | 7 | us | 311.68 | 323.22 | +3.7 % | +11.9 % (r7) | l2_return: queue (chip L2 share) 57 %; dram_queue: queue (DRAM share) 12 %; tensor: stall count after mma 6 % | default latency 432, default rbar 48, issue-only 7776 | batch r9 |
| wi_dec_S2048_rotating kernel | 7 | us | 41.38 | 43.21 | +4.4 % | +5.2 % (r7) | dram_queue: queue (DRAM share) 64 %; dram: DRAM latency 11 %; launch: launch 7 % | branch fallback 282, default latency 388, default rbar 208, issue-only 1364 | batch r7 |
| wi_dec_S2048_warm kernel | 7 | us | 16.86 | 16.03 | -5.0 % | -10.5 % (r7) | launch: launch 18 %; tensor: stall count after mma 14 %; fma: stall count after fp32 10 % | branch fallback 282, default latency 388, default rbar 208, issue-only 1364 | batch r7 |
| wi_dec_S2048_warm b2b | 7 | us | 16.39 | 15.97 | -2.6 % | -8.3 % (r7) | tensor: stall count after mma 17 %; fma: stall count after fp32 12 %; tensor: queue (mma) 12 % | branch fallback 282, default latency 388, default rbar 208, issue-only 1364 | batch r7 |
| wi_dec_S2048_warm graph | 7 | us | 14.51 | 13.83 | -4.7 % | -11.2 % (r7) | tensor: stall count after mma 17 %; fma: stall count after fp32 12 %; tensor: queue (mma) 12 % | branch fallback 282, default latency 388, default rbar 208, issue-only 1364 | batch r7 |
| wi_dec_S8192_rotating kernel | 7 | us | 128.64 | 149.69 | +16.4 % | +16.3 % (r7) | dram_queue: queue (DRAM share) 77 %; dram: DRAM latency 11 %; launch: launch 2 % | branch fallback 266, default latency 356, default rbar 1008, issue-only 2116 | batch r7 |
| wi_dec_S8192_warm kernel | 7 | us | 112.74 | 128.97 | +14.4 % | +10.8 % (r7) | dram_queue: queue (DRAM share) 72 %; dram: DRAM latency 10 %; tensor: stall count after mma 3 % | branch fallback 266, default latency 356, default rbar 1008, issue-only 2116 | batch r7 |
| wi_dec_b_rotating kernel | 7 | us | 67.17 | 77.30 | +15.1 % | +14.3 % (r7) | dram_queue: queue (DRAM share) 71 %; dram: DRAM latency 12 %; launch: launch 4 % | branch fallback 266, default latency 356, default rbar 496, issue-only 1604 | batch r12 |
| wi_dec_b_warm kernel | 7 | us | 26.88 | 26.67 | -0.8 % | -6.2 % (r7) | tensor: stall count after mma 17 %; lsu: queue 13 %; tensor: queue (mma) 12 % | branch fallback 266, default latency 356, default rbar 496, issue-only 1604 | batch r10 |
| wi_dec_b_warm b2b | 7 | us | 26.36 | 26.61 | +0.9 % | -6.8 % (r7) | tensor: stall count after mma 19 %; lsu: queue 14 %; tensor: queue (mma) 13 % | branch fallback 266, default latency 356, default rbar 496, issue-only 1604 | batch r10 |
| wi_dec_b_warm graph | 7 | us | 24.12 | 24.47 | +1.5 % | -6.8 % (r7) | tensor: stall count after mma 19 %; lsu: queue 14 %; tensor: queue (mma) 13 % | branch fallback 266, default latency 356, default rbar 496, issue-only 1604 | batch r10 |
| wi_dec_rot2_rotating kernel | 7 | us | 78.21 | 77.29 | -1.2 % | +0.8 % (r7) | dram_queue: queue (DRAM share) 71 %; dram: DRAM latency 12 %; launch: launch 4 % | branch fallback 266, default latency 356, default rbar 496, issue-only 1604 | batch r7 |
| wi_dec_rot4_rotating kernel | 7 | us | 77.15 | 77.29 | +0.2 % | +2.1 % (r7) | dram_queue: queue (DRAM share) 71 %; dram: DRAM latency 12 %; launch: launch 4 % | branch fallback 266, default latency 356, default rbar 496, issue-only 1604 | batch r7 |
| wi_fa_b_rotating kernel | 7 | us | 264.26 | 272.86 | +3.3 % | +4.0 % (r7) | tensor: stall count after mma 19 %; lsu: queue 14 %; tensor: queue (mma) 13 % | branch fallback 1536, default latency 1824, default rbar 8496, issue-only 10776 | batch r10 |
| wi_fa_b_warm kernel | 7 | us | 257.82 | 267.06 | +3.6 % | -6.4 % (r7) | tensor: stall count after mma 19 %; lsu: queue 15 %; tensor: queue (mma) 14 % | branch fallback 1536, default latency 1824, default rbar 8496, issue-only 10776 | batch r12 |
| wi_fa_direct_warm kernel | 7 | us | 244.03 | 236.03 | -3.3 % | -7.1 % (r7) | l2_return: queue (chip L2 share) 22 %; tensor: queue (mma) 22 %; l2/dram: load latency (incl. DRAM queue) 10 % | branch fallback 1536, default latency 1776, default rbar 3936, issue-only 9528 | batch r7 |
| wi_fa_nostore_rotating kernel | 7 | us | 260.58 | 269.77 | +3.5 % | +4.6 % (r7) | tensor: stall count after mma 18 %; lsu: queue 15 %; tensor: queue (mma) 14 % | branch fallback 1536, default latency 1776, default rbar 7776, issue-only 10872 | batch r7 |
| wi_fa_t64_warm kernel | 7 | us | 474.40 | 500.15 | +5.4 % | -0.7 % (r7) | tensor: stall count after mma 28 %; tensor: queue (mma) 17 %; fma: stall count after fp32 10 % | branch fallback 1536, default latency 3516, default rbar 10320, issue-only 12108 | batch r7 |
| wi_gemv_b_rotating kernel | 7 | us | 35.26 | 40.19 | +14.0 % | +9.9 % (r7) | l2/dram: load latency (incl. DRAM queue) 87 %; launch: launch 7 %; issue: stall count after half_to_float 1 % | default latency 726, default rbar 1056, issue-only 2376 | batch r12 |
| wi_gemv_b_warm kernel | 7 | us | 16.70 | 14.51 | -13.2 % | -14.4 % (r7) | xu: queue (xu) 32 %; alu: queue (alu) 21 %; launch: launch 20 % | default latency 726, default rbar 1056, issue-only 2376 | batch r10 |
| wi_gemv_b_warm b2b | 7 | us | 16.07 | 14.45 | -10.1 % | -11.6 % (r7) | xu: queue (xu) 40 %; alu: queue (alu) 26 %; alu: stall count after alu 4 % | default latency 726, default rbar 1056, issue-only 2376 | batch r10 |
| wi_gemv_b_warm graph | 7 | us | 13.48 | 12.31 | -8.7 % | -9.4 % (r7) | xu: queue (xu) 40 %; alu: queue (alu) 26 %; alu: stall count after alu 4 % | default latency 726, default rbar 1056, issue-only 2376 | batch r10 |
| wi_gemv_mk16_rotating kernel | 7 | us | 34.98 | 40.20 | +14.9 % | +15.2 % (r7) | l2/dram: load latency (incl. DRAM queue) 86 %; launch: launch 7 %; xu: queue (xu) 1 % | default latency 638, default rbar 1056, issue-only 2024 | batch r7 |
| wi_gemv_nocompute_rotating kernel | 7 | us | 35.17 | 38.98 | +10.9 % | +11.1 % (r7) | l2/dram: load latency (incl. DRAM queue) 90 %; launch: launch 7 %; fma: stall count after fp32 1 % | default latency 550, issue-only 2728 | batch r7 |
| wi_gemv_nocompute_warm kernel | 7 | us | 10.85 | 8.45 | -22.1 % | -21.3 % (r7) | launch: launch 34 %; l2_return: queue (chip L2 share) 14 %; fma: queue (fp32) 12 % | default latency 550, issue-only 2728 | batch r8 |
| wi_gemv_nocompute_warm b2b | 7 | us | 10.13 | 8.39 | -17.1 % | -17.9 % (r7) | l2_return: queue (chip L2 share) 22 %; fma: queue (fp32) 17 %; xu: queue (xu) 15 % | default latency 550, issue-only 2728 | batch r8 |
| wi_gemv_nocompute_warm graph | 7 | us | 7.54 | 6.25 | -17.1 % | -18.4 % (r7) | l2_return: queue (chip L2 share) 22 %; fma: queue (fp32) 17 %; xu: queue (xu) 15 % | default latency 550, issue-only 2728 | batch r8 |
| wi_gemv_r16_rotating kernel | 7 | us | 38.08 | 40.08 | +5.2 % | -83.2 % (r7) | l2/dram: load latency (incl. DRAM queue) 86 %; launch: launch 7 %; issue: stall count after half_to_float 1 % | default latency 330, default rbar 1760, issue-only 1144 | batch r12 |
| wi_gemv_r64_warm kernel | 7 | us | 15.01 | 14.37 | -4.3 % | -14.7 % (r7) | xu: queue (xu) 29 %; launch: launch 20 %; alu: queue (alu) 14 % | default latency 1430, default rbar 1056, issue-only 704 | batch r8 |
| wi_gemv_r64_warm b2b | 7 | us | 14.48 | 14.31 | -1.2 % | -8.1 % (r7) | xu: queue (xu) 36 %; alu: queue (alu) 18 %; issue: scheduler (another warp issued) 6 % | default latency 1430, default rbar 1056, issue-only 704 | batch r8 |
| wi_gemv_r64_warm graph | 7 | us | 13.19 | 12.17 | -7.8 % | -9.7 % (r7) | xu: queue (xu) 36 %; alu: queue (alu) 18 %; issue: scheduler (another warp issued) 6 % | default latency 1430, default rbar 1056, issue-only 704 | batch r8 |
| wi_hg_S_s1_rotating kernel | 7 | us | 239.97 | 240.95 | +0.4 % | -2.0 % (r7) | l2_return: queue (chip L2 share) 32 %; dram: DRAM latency 14 %; store_queue: queue (stg) 11 % | default latency 396, default rbar 88, issue-only 7260 | batch r9 |
| wi_hg_S_s4_rotating kernel | 7 | us | 252.35 | 266.92 | +5.8 % | -8.7 % (r7) | l2_return: queue (chip L2 share) 23 %; dram_queue: queue (DRAM share) 20 %; tensor: queue (mma) 15 % | default latency 396, issue-only 7172 | batch r9 |
| wi_hg_b_rotating kernel | 7 | us | 231.78 | 235.20 | +1.5 % | -8.2 % (r7) | tensor: queue (mma) 24 %; dram_queue: queue (DRAM share) 22 %; tensor: stall count after mma 16 % | default latency 192, default rbar 1512, issue-only 10776 | batch r10 |
| wi_hg_b_warm kernel | 7 | us | 223.52 | 231.70 | +3.7 % | -4.7 % (r7) | tensor: queue (mma) 39 %; tensor: stall count after mma 20 %; lsu: queue 15 % | default latency 192, default rbar 1512, issue-only 10776 | batch r12 |
| wi_hg_bk64_warm kernel | 7 | us | 231.68 | 255.53 | +10.3 % | -4.2 % (r7) | tensor: stall count after mma 36 %; tensor: queue (mma) 20 %; lsu: queue 16 % | default latency 240, default rbar 1464, issue-only 6144 | batch r9 |
| wi_hg_bm256_rotating kernel | 7 | us | 240.83 | 256.75 | +6.6 % | -4.4 % (r7) | tensor: queue (mma) 33 %; dram_queue: queue (DRAM share) 16 %; tensor: stall count after mma 14 % | default latency 216, default rbar 1512, issue-only 13752 | batch r9 |
| wi_hg_bm256_warm kernel | 7 | us | 236.74 | 247.43 | +4.5 % | -4.0 % (r7) | tensor: queue (mma) 43 %; tensor: stall count after mma 19 %; store_queue: queue (stg) 12 % | default latency 216, default rbar 1512, issue-only 13752 | batch r9 |
| wi_hg_bm64_rotating kernel | 7 | us | 256.06 | 269.19 | +5.1 % | -6.9 % (r7) | l2_return: queue (chip L2 share) 34 %; dram_queue: queue (DRAM share) 18 %; tensor: stall count after mma 9 % | default latency 352, default rbar 44, issue-only 23848 | batch r9 |
| wi_hg_bm64_warm kernel | 7 | us | 236.03 | 234.34 | -0.7 % | -4.1 % (r7) | l2_return: queue (chip L2 share) 44 %; tensor: queue (mma) 13 %; tensor: stall count after mma 10 % | default latency 352, default rbar 44, issue-only 23848 | batch r9 |
| wi_hg_bn256_rotating kernel | 7 | us | 245.02 | 256.66 | +4.8 % | -7.5 % (r7) | issue: stall count after barrier 23 %; tensor: stall count after mma 15 %; dram_queue: queue (DRAM share) 14 % | default latency 216, default rbar 1512, issue-only 13776 | batch r9 |
| wi_hg_bn256_warm kernel | 7 | us | 240.96 | 247.56 | +2.7 % | -6.7 % (r7) | tensor: queue (mma) 26 %; issue: stall count after barrier 19 %; tensor: stall count after mma 19 % | default latency 216, default rbar 1512, issue-only 13776 | batch r9 |
| wi_hg_bn64_rotating kernel | 7 | us | 256.48 | 267.63 | +4.3 % | -10.7 % (r7) | l2_return: queue (chip L2 share) 38 %; dram_queue: queue (DRAM share) 18 %; store_queue: queue (stg) 9 % | default latency 396, issue-only 23936 | batch r9 |
| wi_hg_direct_rotating kernel | 7 | us | 279.94 | 262.71 | -6.2 % | -7.1 % (r7) | tensor: queue (mma) 24 %; tensor: stall count after mma 18 %; l2_return: queue (chip L2 share) 18 % | default latency 192, issue-only 6240 | batch r9 |
| wi_hg_direct_warm kernel | 7 | us | 243.23 | 233.30 | -4.1 % | -6.8 % (r7) | tensor: queue (mma) 28 %; tensor: stall count after mma 21 %; store_queue: queue (stg) 13 % | default latency 192, issue-only 6240 | batch r9 |
| wi_hg_nocompute_rotating kernel | 7 | us | 162.30 | 170.87 | +5.3 % | -7.9 % (r7) | l2_return: queue (chip L2 share) 43 %; dram_queue: queue (DRAM share) 36 %; dram: DRAM latency 6 % | default latency 216, issue-only 2472 | batch r9 |
| wi_hg_noload_rotating kernel | 7 | us | 190.69 | 179.63 | -5.8 % | -5.6 % (r7) | tensor: queue (mma) 69 %; tensor: stall count after mma 18 %; store_queue: queue (stg) 6 % | default latency 144, issue-only 6240 | batch r9 |
| wi_hg_nostore_rotating kernel | 7 | us | 211.26 | 209.33 | -0.9 % | -7.7 % (r7) | tensor: queue (mma) 29 %; dram_queue: queue (DRAM share) 24 %; tensor: stall count after mma 18 % | default latency 216, default rbar 1488, issue-only 10776 | batch r9 |
| wi_hg_s2_warm kernel | 7 | us | 207.23 | 206.47 | -0.4 % | +1.2 % (r7) | tensor: queue (mma) 40 %; tensor: stall count after mma 19 %; l2_return: queue (chip L2 share) 9 % | default latency 240, default rbar 24, issue-only 4824 | batch r9 |
| wi_hg_sw4_rotating kernel | 7 | us | 226.53 | 240.32 | +6.1 % | -6.1 % (r7) | dram_queue: queue (DRAM share) 25 %; tensor: queue (mma) 23 %; tensor: stall count after mma 16 % | default latency 264, default rbar 1568, issue-only 10808 | batch r9 |
| wi_hg_sw4_warm kernel | 7 | us | 226.62 | 210.73 | -7.0 % | -5.9 % (r7) | tensor: queue (mma) 36 %; tensor: stall count after mma 23 %; store_queue: queue (stg) 14 % | default latency 264, default rbar 1568, issue-only 10808 | batch r9 |
| wi_hg_sw8_rotating kernel | 7 | us | 225.70 | 224.44 | -0.6 % | -4.9 % (r7) | tensor: queue (mma) 27 %; dram_queue: queue (DRAM share) 19 %; tensor: stall count after mma 17 % | default latency 264, default rbar 1572, issue-only 10812 | batch r9 |
| wi_hg_sw8_warm kernel | 7 | us | 224.54 | 210.73 | -6.1 % | -6.0 % (r7) | tensor: queue (mma) 35 %; tensor: stall count after mma 22 %; store_queue: queue (stg) 14 % | default latency 264, default rbar 1572, issue-only 10812 | batch r12 |
| wi_hg_t64_warm kernel | 7 | us | 723.26 | 550.22 | -23.9 % | -47.2 % (r7) | l1: queue 57 %; tensor: stall count after mma 9 %; issue: stall count after local 5 % | default latency 924, default rbar 1512, front end 11712, issue-only 100164 | batch r12 |
| wi_rms_b_rotating kernel | 7 | us | 41.06 | 43.06 | +4.9 % | +4.9 % (r7) | l2/dram: load latency (incl. DRAM queue) 49 %; l1: hit latency 39 %; launch: launch 7 % | default latency 308, issue-only 660 | batch r10 |
| wi_rms_b_warm kernel | 7 | us | 8.90 | 8.66 | -2.6 % | -23.6 % (r7) | launch: launch 33 %; store_port: queue (stores) 13 %; store_queue: queue (stg) 12 % | default latency 308, issue-only 660 | batch r10 |
| wi_rms_b_warm b2b | 7 | us | 8.30 | 8.60 | +3.6 % | -14.0 % (r7) | store_port: queue (stores) 19 %; store_queue: queue (stg) 17 %; l2_return: queue (chip L2 share) 17 % | default latency 308, issue-only 660 | batch r10 |
| wi_rms_b_warm graph | 7 | us | 7.02 | 6.46 | -7.9 % | -27.2 % (r7) | store_port: queue (stores) 19 %; store_queue: queue (stg) 17 %; l2_return: queue (chip L2 share) 17 % | default latency 308, issue-only 660 | batch r10 |
| wi_rms_fp32_rotating kernel | 7 | us | 75.07 | 83.47 | +11.2 % | +11.2 % (r7) | l2/dram: load latency (incl. DRAM queue) 63 %; l1: hit latency 27 %; store_queue: queue (stg) 4 % | default latency 308, issue-only 660 | batch r7 |
| wi_rms_fp32_warm kernel | 7 | us | 20.86 | 13.86 | -33.6 % | -36.7 % (r7) | store_queue: queue (stg) 26 %; launch: launch 21 %; l2_return: queue (chip L2 share) 17 % | default latency 308, issue-only 1188 | batch r8 |
| wi_rms_fp32_warm b2b | 7 | us | 20.16 | 13.80 | -31.5 % | -32.8 % (r7) | store_queue: queue (stg) 32 %; l2_return: queue (chip L2 share) 22 %; store_port: queue (stores) 20 % | default latency 308, issue-only 1188 | batch r8 |
| wi_rms_fp32_warm graph | 7 | us | 17.92 | 11.66 | -34.9 % | -33.7 % (r7) | store_queue: queue (stg) 32 %; l2_return: queue (chip L2 share) 22 %; store_port: queue (stores) 20 % | default latency 308, issue-only 1188 | batch r8 |
| wi_rms_nocompute_rotating kernel | 7 | us | 40.22 | 41.58 | +3.4 % | +4.1 % (r7) | l2/dram: load latency (incl. DRAM queue) 92 %; launch: launch 7 %; lsu: queue 1 % | default latency 220, issue-only 88 | batch r7 |
| wi_rms_nostore_rotating kernel | 7 | us | 20.70 | 18.34 | -11.4 % | +108.3 % (r7) | l2/dram: load latency (incl. DRAM queue) 44 %; l1: hit latency 36 %; launch: launch 16 % | default latency 264, issue-only 132 | batch r7 |
| wi_rms_nostore_rotating b2b | 7 | us | 19.32 | 18.28 | -5.4 % | +123.0 % (r7) | l2/dram: load latency (incl. DRAM queue) 52 %; l1: hit latency 43 %; lsu: queue 1 % | default latency 264, issue-only 132 | batch r7 |
| wi_rms_nostore_rotating graph | 7 | us | 17.25 | 16.14 | -6.4 % | +137.3 % (r7) | l2/dram: load latency (incl. DRAM queue) 52 %; l1: hit latency 43 %; lsu: queue 1 % | default latency 264, issue-only 132 | batch r7 |
| wi_rms_rot2_rotating kernel | 7 | us | 43.39 | 43.09 | -0.7 % | -0.2 % (r7) | l2/dram: load latency (incl. DRAM queue) 49 %; l1: hit latency 39 %; launch: launch 7 % | default latency 308, issue-only 660 | batch r7 |
| wi_rms_rot4_rotating kernel | 7 | us | 43.30 | 43.09 | -0.5 % | +0.1 % (r7) | l2/dram: load latency (incl. DRAM queue) 49 %; l1: hit latency 39 %; launch: launch 7 % | default latency 308, issue-only 660 | batch r7 |
| wi_rms_t256_rotating kernel | 7 | us | 40.22 | 43.53 | +8.2 % | +7.6 % (r7) | l2/dram: load latency (incl. DRAM queue) 81 %; l1: hit latency 9 %; launch: launch 7 % | default latency 616, issue-only 616 | batch r7 |
| wi_rms_t64_rotating kernel | 7 | us | 41.02 | 42.22 | +2.9 % | +5.8 % (r7) | l2/dram: load latency (incl. DRAM queue) 64 %; l1: hit latency 25 %; launch: launch 7 % | default latency 154, issue-only 418 | batch r7 |
| wi_rms_t64_warm kernel | 7 | us | 8.74 | 8.74 | +0.0 % | -14.7 % (r7) | launch: launch 33 %; store_port: queue (stores) 16 %; store_queue: queue (stg) 14 % | default latency 154, issue-only 418 | batch r8 |
| wi_rms_t64_warm b2b | 7 | us | 8.56 | 8.68 | +1.3 % | -9.4 % (r7) | store_port: queue (stores) 24 %; store_queue: queue (stg) 21 %; l2_return: queue (chip L2 share) 17 % | default latency 154, issue-only 418 | batch r8 |
| wi_rms_t64_warm graph | 7 | us | 7.03 | 6.54 | -6.9 % | -23.9 % (r7) | store_port: queue (stores) 24 %; store_queue: queue (stg) 21 %; l2_return: queue (chip L2 share) 17 % | default latency 154, issue-only 418 | batch r8 |
| wi_softmax_b_warm kernel | 7 | us | 8.99 | 8.19 | -8.9 % | -25.5 % (r7) | launch: launch 35 %; store_port: drain after the last instruction 12 %; alu: queue (alu) 8 % | branch fallback 1408, default latency 1672, default rbar 440, issue-only 4488 | batch r10 |
| wi_softmax_b_warm b2b | 7 | us | 9.12 | 8.13 | -10.8 % | -18.6 % (r7) | store_port: drain after the last instruction 19 %; alu: queue (alu) 13 %; fma: stall count after fp32 12 % | branch fallback 1408, default latency 1672, default rbar 440, issue-only 4488 | batch r10 |
| wi_softmax_b_warm graph | 7 | us | 7.36 | 5.99 | -18.6 % | -24.2 % (r7) | store_port: drain after the last instruction 19 %; alu: queue (alu) 13 %; fma: stall count after fp32 12 % | branch fallback 1408, default latency 1672, default rbar 440, issue-only 4488 | batch r10 |
| wi_softmax_t64_warm kernel | 7 | us | 12.74 | 9.63 | -24.4 % | -20.7 % (r7) | launch: launch 30 %; l2_return: queue (chip L2 share) 17 %; store_port: drain after the last instruction 11 % | branch fallback 1408, default latency 1540, default rbar 440, issue-only 4378 | batch r8 |
| wi_softmax_t64_warm b2b | 7 | us | 12.27 | 9.57 | -22.1 % | -17.9 % (r7) | l2_return: queue (chip L2 share) 25 %; store_port: drain after the last instruction 16 %; store_queue: queue (stg) 12 % | branch fallback 1408, default latency 1540, default rbar 440, issue-only 4378 | batch r8 |
| wi_softmax_t64_warm graph | 7 | us | 9.72 | 7.43 | -23.6 % | -19.1 % (r7) | l2_return: queue (chip L2 share) 25 %; store_port: drain after the last instruction 16 %; store_queue: queue (stg) 12 % | branch fallback 1408, default latency 1540, default rbar 440, issue-only 4378 | batch r8 |
| wi_w4_b_warm kernel | 7 | us | 287.14 | 294.42 | +2.5 % | -14.8 % (r7) | tensor: queue (mma) 22 %; tensor: stall count after mma 15 %; smem_array: LDS latency 11 % | default latency 216, default rbar 24, issue-only 15072 | batch r10 |
| wi_w4_nocompute_warm kernel | 7 | us | 180.70 | 150.44 | -16.7 % | -15.8 % (r7) | smem_array: LDS latency 21 %; xu: xu result latency 15 %; fma: stall count after fma_half_rate 9 % | default latency 216, issue-only 22296 | batch r7 |
| wi_w4_nodequant_warm kernel | 7 | us | 217.34 | 216.82 | -0.2 % | -13.1 % (r7) | tensor: queue (mma) 45 %; tensor: stall count after mma 21 %; store_queue: queue (stg) 10 % | default latency 216, default rbar 24, issue-only 6168 | batch r7 |
| rot_depth n10_D32 | 5 | us | 30.98 | 29.25 | -5.6 % | +17.4 % (r7b) | l2/dram: load latency (incl. DRAM queue) 80 %; l2_return: queue (chip L2 share) 19 % | default latency 128, issue-only 1056 | batch r7b |
| rot_depth n10_D64 | 5 | us | 35.20 | 36.11 | +2.6 % | +3.3 % (r7b) | l2/dram: load latency (incl. DRAM queue) 99 % | default latency 128, issue-only 1056 | batch r7b |
| rot_depth n10_D96 | 5 | us | 36.16 | 36.11 | -0.1 % | +0.5 % (r7b) | l2/dram: load latency (incl. DRAM queue) 99 % | default latency 128, issue-only 1056 | batch r7b |
| rot_depth n10_D192 | 5 | us | 36.19 | 36.11 | -0.2 % | +0.5 % (r7b) | l2/dram: load latency (incl. DRAM queue) 99 % | default latency 128, issue-only 1056 | batch r7b |
| rot_depth n21_D48 | 5 | us | 71.10 | 72.99 | +2.7 % | +2.5 % (r7b) | l2/dram: load latency (incl. DRAM queue) 99 % | default latency 128, issue-only 2112 | batch r7b |
| spill t64_n16_g1_s0 | 4 | cycles | 1,911.00 | 1,470.00 | -23.1 % | -23.1 % (r7b) | issue: stall count after other 41 %; alu: stall count after alu 37 %; alu: stall count after alu_wide 12 % | branch fallback 1, default latency 9, issue-only 569 | batch r7b |
| spill t64_n16_g1_s96 | 4 | cycles | 1,911.00 | 1,470.00 | -23.1 % | -23.1 % (r7b) | issue: stall count after other 41 %; alu: stall count after alu 37 %; alu: stall count after alu_wide 12 % | branch fallback 1, default latency 9, issue-only 569 | batch r7b |
| spill t64_n16_g48_s0 | 4 | cycles | 1,911.00 | 2,045.64 | +7.0 % | -23.1 % (r7b) | l1: queue 44 %; issue: stall count after other 27 %; l1: local path 11 % | branch fallback 1, default latency 9, issue-only 569 | batch r7b |
| spill t64_n16_g192_s96 | 4 | cycles | 9,111.00 | 8,492.56 | -6.8 % | -32.3 % (r7b) | l1: queue 44 %; issue: stall count after other 27 %; l1: local path 11 % | branch fallback 4, default latency 36, issue-only 2276 | batch r7b |
| spill t64_n256_g48_s0 | 4 | cycles | 30,598.00 | 31,536.84 | +3.1 % | -29.3 % (r7b) | l1: queue 51 %; issue: stall count after other 23 %; l1: local path 13 % | branch fallback 1, default latency 9, issue-only 8249 | batch r7b |
| spill t256_n64_g1_s0 | 4 | cycles | 8,111.00 | 8,310.00 | +2.5 % | +2.3 % (r7b) | l1: queue 51 %; issue: stall count after other 23 %; l1: local path 12 % | branch fallback 1, default latency 27, issue-only 8417 | batch r7b |
| spill t256_n1024_g48_s0 | 4 | cycles | 2,516,207 | 2,497,715 | -0.7 % | -94.8 % (r7b) | l1: queue 98 %; issue: stall count after other 1 %; l1: local path 1 % | branch fallback 1, default latency 27, issue-only 131297 | batch r7b |
| store_lines S3_W32_D32_g128 | 4 | cycles | 1,733.00 | 1,576.39 | -9.0 % | +44.4 % (r7b) | store_queue: queue (stg) 39 %; store_port: drain after the last instruction 30 %; store_port: queue (stores) 28 % | default latency 54, issue-only 183 | batch r7b |
| store_lines S6_W32_D32_g128 | 4 | cycles | 5,818.00 | 5,532.30 | -4.9 % | +10.4 % (r7b) | store_queue: queue (stg) 54 %; store_port: queue (stores) 30 %; store_port: drain after the last instruction 15 % | default latency 54, issue-only 339 | batch r7b |
| store_lines S2_W16_D32_g128 | 4 | cycles | 287.00 | 797.41 | +177.8 % | +129.4 % (r7b) | store_port: drain after the last instruction 44 %; store_port: queue (stores) 44 %; store_queue: queue (stg) 9 % | default latency 54, issue-only 75 | batch r7b |
| store_lines S4_W16_D32_g128 | 4 | cycles | 3,900.00 | 2,918.92 | -25.2 % | -60.9 % (r7b) | store_queue: queue (stg) 55 %; store_port: drain after the last instruction 22 %; store_port: queue (stores) 20 % | default latency 54, issue-only 147 | batch r7b |
| store_lines S1_W32_D64_g256 | 4 | cycles | 1,196.00 | 1,009.38 | -15.6 % | +116.7 % (r7b) | store_port: drain after the last instruction 44 %; store_queue: queue (stg) 33 %; store_port: queue (stores) 18 % | default latency 108, issue-only 150 | batch r7b |
| store_lines S8_W4_D4_g512 | 4 | cycles | 4,054.00 | 3,799.77 | -6.3 % | +372.3 % (r7b) | store_queue: queue (stg) 72 %; store_port: drain after the last instruction 22 %; store_port: queue (stores) 3 % | default latency 198, issue-only 143 | batch r7b |
| store_lines S2_W4_D32_g256 | 4 | cycles | 1,585.00 | 3,648.95 | +130.2 % | +50.5 % (r7b) | store_queue: queue (stg) 62 %; store_port: drain after the last instruction 23 %; store_port: queue (stores) 14 % | default latency 108, issue-only 78 | batch r7b |
| store_lines S1_W4_D128_g128 | 4 | cycles | 99.00 | 66.00 | -33.3 % | -35.4 % (r7b) | store_port: drain after the last instruction 51 %; store_port: queue (stores) 47 %; issue: default result latency (S2R) 1 % | default latency 54, issue-only 42 | batch r7b |
| store_lines S2_W4_D128_g128 | 4 | cycles | 105.00 | 1,370.00 | +1204.8 % | +417.5 % (r7b) | store_port: queue (stores) 46 %; store_port: drain after the last instruction 30 %; store_queue: queue (stg) 24 % | default latency 54, issue-only 42 | batch r7b |
| warp_map ex2_t32_n1 | 4 | cycles | 9,465.00 | 9,481.00 | +0.2 % | +0.2 % (r8) | xu: queue (xu) 61 %; xu: xu result latency 24 %; xu: stall count after xu 12 % | branch fallback 1, default latency 5, issue-only 392 | batch r8 |
| warp_map ex2_t32_n8 | 4 | cycles | 17,411.00 | 17,366.00 | -0.3 % | +277.1 % (r8) | xu: queue (xu) 79 %; xu: xu result latency 10 %; xu: stall count after xu 7 % | branch fallback 8, default latency 40, issue-only 3136 | batch r8 |
| warp_map ex2_t96_n1 | 4 | cycles | 9,465.00 | 9,485.00 | +0.2 % | +0.2 % (r8) | xu: queue (xu) 61 %; xu: xu result latency 24 %; xu: stall count after xu 12 % | branch fallback 1, default latency 9, issue-only 1174 | batch r8 |
| warp_map ex2_t96_n2 | 4 | cycles | 10,233.00 | 17,372.00 | +69.8 % | +60.5 % (r8) | xu: queue (xu) 79 %; xu: xu result latency 10 %; xu: stall count after xu 7 % | branch fallback 2, default latency 18, issue-only 2348 | batch r8 |
| warp_map mma_t64_n4 | 4 | cycles | 33,171.00 | 32,832.10 | -1.0 % | +98.0 % (r8) | tensor: queue (mma) 45 %; tensor: stall count after mma 36 %; issue: stall count after other 19 % | branch fallback 4, default latency 28, issue-only 10440 | batch r8 |
| warp_map mma_t128_n1 | 4 | cycles | 18,863.00 | 18,093.00 | -4.1 % | -4.1 % (r8) | tensor: stall count after mma 65 %; issue: stall count after other 35 % | branch fallback 1, default latency 11, issue-only 5218 | batch r8 |
| copy_dram warm_z0_g1 | 4 | cycles | 14,447.00 | 14,759.20 | +2.2 % | +0.8 % (r8) | l2: L2 latency 38 %; lsu: queue 29 %; l2_return: queue (SM port rate) 22 % | default latency 22, issue-only 281 | batch r8 |
| copy_dram warm_z0_g16 | 4 | cycles | 14,472.00 | 14,759.20 | +2.0 % | +0.7 % (r8) | l2: L2 latency 38 %; lsu: queue 29 %; l2_return: queue (SM port rate) 22 % | default latency 22, issue-only 281 | batch r8 |
| copy_dram rot_z64_g16 | 4 | cycles | 58,894.00 | 57,202.21 | -2.9 % | +60.0 % (r8) | dram: DRAM latency 39 %; issue: stall count after other 38 %; lsu: queue 8 % | default latency 82, issue-only 12889 | batch r8 |
| copy_dram rot_z256_g48 | 4 | cycles | 143,510.00 | 152,199.31 | +6.1 % | +77.7 % (r8) | issue: stall count after other 57 %; dram: DRAM latency 21 %; fma: stall count after fp32 8 % | default latency 82, issue-only 49753 | batch r8 |
| copy_dram rot_z1024_g48 | 4 | cycles | 417,205.00 | 447,111.31 | +7.2 % | +31.8 % (r8) | issue: stall count after other 77 %; fma: stall count after fp32 11 %; dram: DRAM latency 7 % | default latency 82, issue-only 197209 | batch r8 |
| store_spread L4_R128_S16_g512 | 4 | cycles | 14,115.00 | 20,097.19 | +42.4 % | +8.9 % (r8) | store_queue: queue (stg) 90 %; store_port: drain after the last instruction 6 %; store_port: queue (stores) 3 % | default latency 198, issue-only 143 | batch r8 |
| store_spread L4_R1024_S16_g256 | 4 | cycles | 5,267.00 | 9,177.93 | +74.3 % | +10.7 % (r8) | store_queue: queue (stg) 82 %; store_port: drain after the last instruction 11 %; store_port: queue (stores) 5 % | default latency 108, issue-only 78 | batch r8 |
| store_spread L4_R2048_S16_g256 | 4 | cycles | 5,250.00 | 9,177.93 | +74.8 % | +11.1 % (r8) | store_queue: queue (stg) 82 %; store_port: drain after the last instruction 11 %; store_port: queue (stores) 5 % | default latency 108, issue-only 78 | batch r8 |
| store_spread L2_R4096_S8_g256 | 4 | cycles | 13,648.00 | 19,327.11 | +41.6 % | -57.1 % (r8) | store_queue: queue (stg) 88 %; store_port: drain after the last instruction 6 %; store_port: queue (stores) 5 % | default latency 108, issue-only 84 | batch r8 |
| store_spread L8_R4096_S32_g256 | 4 | cycles | 1,164.00 | 1,307.04 | +12.3 % | +401.1 % (r8) | store_port: drain after the last instruction 44 %; store_queue: queue (stg) 43 %; store_port: queue (stores) 5 % | default latency 108, issue-only 78 | batch r8 |
| store_spread L16_R4096_S64_g256 | 4 | cycles | 1,097.00 | 1,307.04 | +19.1 % | +431.7 % (r8) | store_port: drain after the last instruction 44 %; store_queue: queue (stg) 43 %; store_port: queue (stores) 5 % | default latency 108, issue-only 78 | batch r8 |
| arith_beside L2_S0_A0_g96 | 4 | cycles | 1,163.00 | 1,199.72 | +3.2 % | +1.8 % (r8) | l2_return: queue (chip L2 share) 56 %; l2/dram: load latency (incl. DRAM queue) 24 %; lsu: queue 7 % | default latency 36, issue-only 26 | batch r8 |
| arith_beside L2_S0_A32_g96 | 4 | cycles | 1,196.00 | 1,196.72 | +0.1 % | -1.2 % (r8) | l2_return: queue (chip L2 share) 51 %; l2/dram: load latency (incl. DRAM queue) 22 %; fma: stall count after fp32 8 % | default latency 36, issue-only 26 | batch r8 |
| arith_beside L2_S0_A128_g512 | 4 | cycles | 5,022.00 | 5,463.41 | +8.8 % | +6.8 % (r8) | l2_return: queue (chip L2 share) 72 %; lsu: queue 9 %; fma: stall count after fp32 8 % | default latency 198, issue-only 143 | batch r8 |
| arith_beside L0_S2_A32_g96 | 4 | cycles | 372.00 | 338.40 | -9.0 % | -8.8 % (r8) | store_port: drain after the last instruction 54 %; store_port: queue (stores) 34 %; fma: stall count after fp32 6 % | default latency 36, issue-only 90 | batch r8 |
| arith_beside L0_S2_A128_g96 | 4 | cycles | 751.00 | 732.40 | -2.5 % | -2.2 % (r8) | store_port: drain after the last instruction 46 %; store_port: queue (stores) 29 %; fma: stall count after fp32 20 % | default latency 36, issue-only 90 | batch r8 |
| mma_ldsm t128_L0_M64 | 2 | cycles/pass | 1,105.88 | 1,081.17 | -2.2 % | -2.2 % (r8) | tensor: stall count after mma 76 %; issue: stall count after other 21 %; tensor: queue (mma) 2 % | default latency 9, issue-only 16468 | batch r8 |
| mma_ldsm t128_L8_M64 | 2 | cycles/pass | 1,111.53 | 1,076.85 | -3.1 % | -3.1 % (r8) | tensor: stall count after mma 67 %; issue: stall count after other 19 %; issue: stall count after ldsm 11 % | default latency 13, issue-only 15192 | batch r8 |
| mma_ldsm t256_L4_M32 | 2 | cycles/pass | 1,079.16 | 1,089.95 | +1.0 % | +1.0 % (r8) | tensor: queue (mma) 48 %; tensor: stall count after mma 32 %; issue: stall count after other 11 % | default latency 25, issue-only 17072 | batch r8 |
| mma_ldsm t512_L2_M16 | 2 | cycles/pass | 1,099.41 | 1,117.41 | +1.6 % | +1.6 % (r8) | tensor: queue (mma) 68 %; tensor: stall count after mma 15 %; issue: stall count after other 7 % | default latency 49, issue-only 20848 | batch r8 |
| spill_rw n64_L1_S0_H0_g48 | 4 | cycles | 16,282.00 | 18,935.88 | +16.3 % | +187.7 % (r8) | l1: latency 42 %; l1: queue 27 %; alu: stall count after alu 15 % | branch fallback 1, default latency 67, issue-only 7913 | batch r8 |
| spill_rw n64_L0_S1_H0_g1 | 4 | cycles | 9,669.00 | 8,361.00 | -13.5 % | -13.5 % (r8) | l1: queue 33 %; issue: stall count after other 23 %; alu: stall count after alu 23 % | default latency 60, issue-only 7913 | batch r8 |
| spill_rw n64_L0_S1_H0_g48 | 4 | cycles | 31,749.00 | 31,580.16 | -0.5 % | +47.4 % (r8) | l1: queue 84 %; issue: stall count after other 6 %; alu: stall count after alu 6 % | default latency 60, issue-only 7913 | batch r8 |
| spill_rw n64_L1_S1_H0_g1 | 4 | cycles | 18,124.00 | 19,717.00 | +8.8 % | +8.8 % (r8) | l1: latency 52 %; alu: stall count after alu 18 %; issue: stall count after other 7 % | branch fallback 1, default latency 67, issue-only 7945 | batch r8 |
| spill_rw n64_L4_S1_H0_g48 | 4 | cycles | 31,956.00 | 31,638.80 | -1.0 % | +192.4 % (r8) | l1: queue 48 %; l1: latency 29 %; alu: stall count after alu 10 % | branch fallback 1, default latency 67, issue-only 9993 | batch r8 |
| spill_rw n256_L1_S0_H0_g1 | 4 | cycles | 351,714.00 | 112,393.30 | -68.0 % | -78.7 % (r8) | l1: latency 38 %; l1: queue 27 %; alu: stall count after alu 13 % | branch fallback 1, default latency 67, issue-only 30953 | batch r8 |
| spill_rw n256_L1_S0_H0_g48 | 4 | cycles | 325,706.00 | 309,311.82 | -5.0 % | -42.7 % (r8) | l1: queue 68 %; l1: latency 13 %; l1: local path 11 % | branch fallback 1, default latency 67, issue-only 30953 | batch r8 |
| spill_rw n256_L1_S1_H0_g48 | 4 | cycles | 396,050.00 | 517,146.20 | +30.6 % | -5.8 % (r8) | l1: queue 80 %; l1: latency 8 %; l1: local path 7 % | branch fallback 1, default latency 67, issue-only 30985 | batch r8 |
| spill_rw n256_L4_S1_H0_g1 | 4 | cycles | 72,909.00 | 145,133.40 | +99.1 % | +7.1 % (r8) | l1: queue 40 %; l1: latency 30 %; alu: stall count after alu 10 % | branch fallback 1, default latency 67, issue-only 39177 | batch r8 |
| nvfp4gemm2048_c128x128x32_s2_t128_rotating kernel | 7 | us | 701.66 | 671.22 | -4.3 % | -32.6 % (r8) | alu: stall count after alu 21 %; issue: stall count after other 21 %; xu: xu result latency 13 % | branch fallback 49152, default latency 240, default rbar 24, issue-only 349536 | batch r9 |
| nvfp4gemm2048_c128x128x32_s2_t128_warm kernel | 7 | us | 699.87 | 674.85 | -3.6 % | -32.8 % (r8) | alu: stall count after alu 21 %; issue: stall count after other 21 %; xu: xu result latency 13 % | branch fallback 49152, default latency 240, default rbar 24, issue-only 349536 | batch r9 |
| nvfp4gemm2048_c128x128x32_s3_t128_rotating kernel | 7 | us | 698.46 | 696.30 | -0.3 % | -30.0 % (r8) | alu: stall count after alu 21 %; issue: stall count after other 20 %; xu: xu result latency 12 % | branch fallback 49152, default latency 240, default rbar 24, issue-only 356520 | batch r9 |
| nvfp4gemm2048_c128x128x32_s3_t128_warm kernel | 7 | us | 698.59 | 696.14 | -0.4 % | -30.3 % (r8) | alu: stall count after alu 21 %; issue: stall count after other 20 %; xu: xu result latency 12 % | branch fallback 49152, default latency 240, default rbar 24, issue-only 356520 | batch r9 |
| nvfp4gemm2048_c64x128x32_s4_t128_rotating kernel | 7 | us | 1,270.62 | 1,176.94 | -7.4 % | -37.4 % (r8) | alu: stall count after alu 25 %; issue: stall count after other 24 %; xu: xu result latency 15 % | branch fallback 90112, default latency 440, issue-only 637912 | batch r9 |
| nvfp4gemm2048_c64x128x32_s4_t128_warm kernel | 7 | us | 1,273.95 | 1,182.24 | -7.2 % | -36.7 % (r8) | alu: stall count after alu 25 %; issue: stall count after other 24 %; xu: xu result latency 15 % | branch fallback 90112, default latency 440, issue-only 637912 | batch r9 |
| rmsb_m192_arith kernel | 7 | us | 2.94 | 2.53 | -14.1 % | -14.1 % (r8) | launch: launch 77 %; xu: queue (xu) 6 %; alu: queue (alu) 5 % | default latency 32, default rbar 128, issue-only 48 | batch r8 |
| rmsb_m192_arith b2b | 7 | us | 3.65 | 4.07 | +11.4 % | +11.4 % (r8) | xu: queue (xu) 27 %; alu: queue (alu) 24 %; fma: stall count after fp32 8 % | default latency 32, default rbar 128, issue-only 48 | batch r8 |
| rmsb_m192_arith graph | 7 | us | 1.32 | 1.43 | +8.7 % | +4.7 % (r8) | xu: queue (xu) 27 %; alu: queue (alu) 24 %; fma: stall count after fp32 8 % | default latency 32, default rbar 128, issue-only 48 | batch r8 |
| rmsb_m192_b kernel | 7 | us | 4.61 | 5.42 | +17.5 % | +9.9 % (r8) | launch: launch 52 %; store_port: drain after the last instruction 16 %; l2_return: queue (chip L2 share) 12 % | default latency 112, issue-only 240 | batch r8 |
| rmsb_m192_b b2b | 7 | us | 5.01 | 5.36 | +7.0 % | -0.0 % (r8) | store_port: drain after the last instruction 33 %; l2_return: queue (chip L2 share) 26 %; store_port: queue (stores) 16 % | default latency 112, issue-only 240 | batch r8 |
| rmsb_m192_b graph | 7 | us | 3.28 | 3.22 | -2.1 % | -12.8 % (r8) | store_port: drain after the last instruction 33 %; l2_return: queue (chip L2 share) 26 %; store_port: queue (stores) 16 % | default latency 112, issue-only 240 | batch r8 |
| rmsb_m192_nocompute kernel | 7 | us | 4.58 | 4.60 | +0.5 % | +0.5 % (r8) | launch: launch 63 %; store_port: drain after the last instruction 16 %; l2_return: queue (chip L2 share) 15 % | default latency 80, issue-only 32 | batch r8 |
| rmsb_m192_nocompute b2b | 7 | us | 4.11 | 4.43 | +7.9 % | +7.0 % (r8) | store_port: drain after the last instruction 44 %; l2_return: queue (chip L2 share) 39 %; l2/dram: load latency (incl. DRAM queue) 8 % | default latency 80, issue-only 32 | batch r8 |
| rmsb_m192_nocompute graph | 7 | us | 2.39 | 2.29 | -3.9 % | -5.4 % (r8) | store_port: drain after the last instruction 44 %; l2_return: queue (chip L2 share) 39 %; l2/dram: load latency (incl. DRAM queue) 8 % | default latency 80, issue-only 32 | batch r8 |
| rmsb_m192_noload kernel | 7 | us | 4.64 | 4.68 | +0.8 % | +2.9 % (r8) | launch: launch 60 %; store_port: drain after the last instruction 13 %; store_queue: queue (stg) 9 % | default latency 80, default rbar 80, issue-only 64 | batch r8 |
| rmsb_m192_noload b2b | 7 | us | 4.11 | 4.62 | +12.5 % | +14.8 % (r8) | store_port: drain after the last instruction 34 %; store_queue: queue (stg) 23 %; xu: queue (xu) 18 % | default latency 80, default rbar 80, issue-only 64 | batch r8 |
| rmsb_m192_noload graph | 7 | us | 2.67 | 2.48 | -7.1 % | -3.5 % (r8) | store_port: drain after the last instruction 34 %; store_queue: queue (stg) 23 %; xu: queue (xu) 18 % | default latency 80, default rbar 80, issue-only 64 | batch r8 |
| rmsb_m192_nostore kernel | 7 | us | 4.54 | 4.60 | +1.2 % | +1.2 % (r8) | launch: launch 67 %; l2_return: queue (chip L2 share) 16 %; l2/dram: load latency (incl. DRAM queue) 3 % | default latency 96, issue-only 48 | batch r8 |
| rmsb_m192_nostore b2b | 7 | us | 4.12 | 4.16 | +0.9 % | -1.2 % (r8) | l2_return: queue (chip L2 share) 47 %; l2/dram: load latency (incl. DRAM queue) 10 %; lsu: queue 6 % | default latency 96, issue-only 48 | batch r8 |
| rmsb_m192_nostore graph | 7 | us | 2.38 | 2.02 | -15.3 % | -20.1 % (r8) | l2_return: queue (chip L2 share) 47 %; l2/dram: load latency (incl. DRAM queue) 10 %; lsu: queue 6 % | default latency 96, issue-only 48 | batch r8 |
| rmsb_m48_arith kernel | 7 | us | 2.72 | 2.53 | -7.0 % | -7.0 % (r8) | launch: launch 89 %; fma: stall count after fp32 2 %; xu: xu result latency 2 % | default latency 8, default rbar 32, issue-only 12 | batch r8 |
| rmsb_m48_arith b2b | 7 | us | 1.95 | 1.99 | +2.3 % | +1.6 % (r8) | fma: stall count after fp32 20 %; xu: xu result latency 18 %; smem_array: SHFL latency 16 % | default latency 8, default rbar 32, issue-only 12 | batch r8 |
| rmsb_m48_arith graph | 7 | us | 0.82 | 0.96 | +17.6 % | +15.9 % (r8) | fma: stall count after fp32 20 %; xu: xu result latency 18 %; smem_array: SHFL latency 16 % | default latency 8, default rbar 32, issue-only 12 | batch r8 |
| rmsb_m48_b kernel | 7 | us | 4.64 | 4.60 | -0.9 % | -0.9 % (r8) | launch: launch 72 %; l2_return: queue (chip L2 share) 8 %; store_port: drain after the last instruction 7 % | default latency 28, issue-only 60 | batch r8 |
| rmsb_m48_b b2b | 7 | us | 4.06 | 4.07 | +0.1 % | +0.1 % (r8) | l2_return: queue (chip L2 share) 27 %; store_port: drain after the last instruction 26 %; l2/dram: load latency (incl. DRAM queue) 25 % | default latency 28, issue-only 60 | batch r8 |
| rmsb_m48_b graph | 7 | us | 1.85 | 1.72 | -7.0 % | -8.8 % (r8) | l2_return: queue (chip L2 share) 27 %; store_port: drain after the last instruction 26 %; l2/dram: load latency (incl. DRAM queue) 25 % | default latency 28, issue-only 60 | batch r8 |
| rmsb_m48_nocompute kernel | 7 | us | 2.43 | 2.53 | +4.0 % | +4.0 % (r8) | launch: launch 84 %; store_port: drain after the last instruction 6 %; l2_return: queue (chip L2 share) 4 % | default latency 20, issue-only 8 | batch r8 |
| rmsb_m48_nocompute b2b | 7 | us | 2.10 | 4.07 | +94.2 % | +4.0 % (r8) | store_port: drain after the last instruction 36 %; l2_return: queue (chip L2 share) 27 %; l2/dram: load latency (incl. DRAM queue) 25 % | default latency 20, issue-only 8 | batch r8 |
| rmsb_m48_nocompute graph | 7 | us | 1.10 | 1.16 | +5.8 % | +4.6 % (r8) | store_port: drain after the last instruction 36 %; l2_return: queue (chip L2 share) 27 %; l2/dram: load latency (incl. DRAM queue) 25 % | default latency 20, issue-only 8 | batch r8 |
| rmsb_m48_noload kernel | 7 | us | 2.94 | 2.53 | -14.1 % | -14.1 % (r8) | launch: launch 80 %; store_port: drain after the last instruction 9 %; xu: xu result latency 2 % | default latency 20, default rbar 20, issue-only 16 | batch r8 |
| rmsb_m48_noload b2b | 7 | us | 2.13 | 4.07 | +91.0 % | +91.0 % (r8) | store_port: drain after the last instruction 46 %; xu: xu result latency 10 %; fma: stall count after fp32 10 % | default latency 20, default rbar 20, issue-only 16 | batch r8 |
| rmsb_m48_noload graph | 7 | us | 1.23 | 1.31 | +6.6 % | +5.7 % (r8) | store_port: drain after the last instruction 46 %; xu: xu result latency 10 %; fma: stall count after fp32 10 % | default latency 20, default rbar 20, issue-only 16 | batch r8 |
| rmsb_m48_nostore kernel | 7 | us | 4.64 | 2.53 | -45.5 % | -45.5 % (r8) | launch: launch 77 %; l2_return: queue (chip L2 share) 8 %; l2/dram: load latency (incl. DRAM queue) 7 % | default latency 24, issue-only 12 | batch r8 |
| rmsb_m48_nostore b2b | 7 | us | 4.07 | 4.07 | +0.1 % | +0.1 % (r8) | l2_return: queue (chip L2 share) 36 %; l2/dram: load latency (incl. DRAM queue) 33 %; l2_return: transfer 6 % | default latency 24, issue-only 12 | batch r8 |
| rmsb_m48_nostore graph | 7 | us | 1.85 | 1.44 | -21.9 % | -23.4 % (r8) | l2_return: queue (chip L2 share) 36 %; l2/dram: load latency (incl. DRAM queue) 33 %; l2_return: transfer 6 % | default latency 24, issue-only 12 | batch r8 |
| smb_m192_b kernel | 7 | us | 6.62 | 5.92 | -10.6 % | -8.3 % (r8) | launch: launch 48 %; store_port: drain after the last instruction 14 %; l2_return: queue (chip L2 share) 11 % | branch fallback 512, default latency 608, default rbar 160, issue-only 1632 | batch r8 |
| smb_m192_b b2b | 7 | us | 6.15 | 5.86 | -4.8 % | -2.2 % (r8) | store_port: drain after the last instruction 26 %; l2_return: queue (chip L2 share) 21 %; fma: stall count after fp32 10 % | branch fallback 512, default latency 608, default rbar 160, issue-only 1632 | batch r8 |
| smb_m192_b graph | 7 | us | 4.15 | 3.72 | -10.3 % | -6.6 % (r8) | store_port: drain after the last instruction 26 %; l2_return: queue (chip L2 share) 21 %; fma: stall count after fp32 10 % | branch fallback 512, default latency 608, default rbar 160, issue-only 1632 | batch r8 |
| smb_m192_nostore kernel | 7 | us | 4.64 | 5.19 | +11.9 % | +7.5 % (r8) | launch: launch 55 %; l2_return: queue (chip L2 share) 13 %; fma: stall count after fp32 6 % | branch fallback 512, default latency 592, default rbar 160, issue-only 1584 | batch r8 |
| smb_m192_nostore b2b | 7 | us | 5.21 | 5.13 | -1.5 % | -5.4 % (r8) | l2_return: queue (chip L2 share) 28 %; fma: stall count after fp32 13 %; xu: xu result latency 10 % | branch fallback 512, default latency 592, default rbar 160, issue-only 1584 | batch r8 |
| smb_m192_nostore graph | 7 | us | 3.20 | 2.99 | -6.6 % | -12.9 % (r8) | l2_return: queue (chip L2 share) 28 %; fma: stall count after fp32 13 %; xu: xu result latency 10 % | branch fallback 512, default latency 592, default rbar 160, issue-only 1584 | batch r8 |
| smb_m48_b kernel | 7 | us | 4.61 | 4.66 | +1.1 % | +0.2 % (r8) | launch: launch 61 %; store_port: drain after the last instruction 8 %; fma: stall count after fp32 7 % | branch fallback 128, default latency 152, default rbar 40, issue-only 408 | batch r8 |
| smb_m48_b b2b | 7 | us | 4.11 | 4.60 | +12.1 % | +11.0 % (r8) | store_port: drain after the last instruction 20 %; fma: stall count after fp32 17 %; xu: xu result latency 13 % | branch fallback 128, default latency 152, default rbar 40, issue-only 408 | batch r8 |
| smb_m48_b graph | 7 | us | 2.42 | 2.46 | +1.6 % | -0.3 % (r8) | store_port: drain after the last instruction 20 %; fma: stall count after fp32 17 %; xu: xu result latency 13 % | branch fallback 128, default latency 152, default rbar 40, issue-only 408 | batch r8 |
| smb_m48_nostore kernel | 7 | us | 4.58 | 4.60 | +0.5 % | +0.5 % (r8) | launch: launch 65 %; fma: stall count after fp32 7 %; xu: xu result latency 6 % | branch fallback 128, default latency 148, default rbar 40, issue-only 396 | batch r8 |
| smb_m48_nostore b2b | 7 | us | 4.11 | 4.26 | +3.7 % | +2.8 % (r8) | fma: stall count after fp32 21 %; xu: xu result latency 16 %; issue: stall count after other 14 % | branch fallback 128, default latency 148, default rbar 40, issue-only 396 | batch r8 |
| smb_m48_nostore graph | 7 | us | 2.18 | 2.12 | -2.8 % | -4.5 % (r8) | fma: stall count after fp32 21 %; xu: xu result latency 16 %; issue: stall count after other 14 % | branch fallback 128, default latency 148, default rbar 40, issue-only 396 | batch r8 |
| first_reader warm_ldg_z0_alone | 4 | cycles | 385.14 | 397.70 | +3.3 % | +3.5 % (r9b) | l2/dram: load latency (incl. DRAM queue) 76 %; issue: stall count after other 6 %; issue: default result latency (LDC) 5 % | default latency 77, issue-only 1676 | batch r9c |
| first_reader warm_ldg_z8_alone | 4 | cycles | 411.28 | 424.70 | +3.3 % | +3.5 % (r9b) | l2/dram: load latency (incl. DRAM queue) 49 %; issue: stall count after other 29 %; alu: stall count after alu 5 % | default latency 137, issue-only 2756 | batch r9c |
| first_reader warm_ldg_z24_alone | 4 | cycles | 411.28 | 424.70 | +3.3 % | +3.5 % (r9b) | issue: stall count after other 50 %; l2/dram: load latency (incl. DRAM queue) 32 %; fma: stall count after fp32 7 % | default latency 137, issue-only 5828 | batch r9c |
| first_reader warm_cp_z32_alone | 4 | cycles | 380.30 | 398.00 | +4.7 % | +3.4 % (r9) | issue: stall count after other 57 %; l2: L2 latency 28 %; fma: stall count after fp32 8 % | default latency 77, issue-only 7372 | batch r9c |
| first_reader warm_ldg_z48_alone | 4 | cycles | 411.22 | 424.70 | +3.3 % | +3.5 % (r9b) | issue: stall count after other 63 %; l2/dram: load latency (incl. DRAM queue) 21 %; fma: stall count after fp32 9 % | default latency 137, issue-only 10436 | batch r9c |
| first_reader rot_cp_z4_lead | 4 | cycles | 972.22 | 991.00 | +1.9 % | +36.7 % (r9c) | l2: wait on a fill 81 %; issue: stall count after other 10 %; issue: default result latency (LDCU) 2 % | default latency 269, issue-only 2508 | batch r9c |
| first_reader rot_ldg_z16_alone | 4 | cycles | 1,137.56 | 1,182.70 | +4.0 % | +4.0 % (r9c) | l2/dram: load latency (incl. DRAM queue) 67 %; issue: stall count after other 22 %; fma: stall count after fp32 3 % | default latency 137, issue-only 4292 | batch r9c |
| first_reader rot_cp_z64_alone | 4 | cycles | 1,106.62 | 1,123.00 | +1.5 % | +32.1 % (r9c) | issue: stall count after other 50 %; dram: DRAM latency 39 %; fma: stall count after fp32 7 % | default latency 77, issue-only 13516 | batch r9c |
| first_share warm_share_d1_z0 | 4 | cycles | 17,597.00 | 21,383.30 | +21.5 % | +20.3 % (r9) | l2_return: queue (chip L2 share) 46 %; l2: L2 latency 26 %; lsu: queue 20 % | default latency 22, issue-only 281 | batch r9c |
| first_share warm_share_d16_z0 | 4 | cycles | 18,751.00 | 21,383.30 | +14.0 % | +15.1 % (r9) | l2_return: queue (chip L2 share) 46 %; l2: L2 latency 26 %; lsu: queue 20 % | default latency 22, issue-only 281 | batch r9c |
| first_share rot_share_d8_z0 | 4 | cycles | 29,451.00 | 34,905.82 | +18.5 % | +30.1 % (r9c) | dram: DRAM latency 55 %; l2_return: queue (chip L2 share) 28 %; lsu: queue 12 % | default latency 22, issue-only 281 | batch r9c |
| first_share warm_share_d48_z256 | 4 | cycles | 110,270.00 | 120,138.85 | +8.9 % | +8.9 % (r9) | issue: stall count after other 72 %; fma: stall count after fp32 10 %; l2_return: queue (chip L2 share) 8 % | default latency 82, issue-only 49753 | batch r9c |
| first_share rot_pieces_s64 | 4 | cycles | 7,333.31 | 7,336.80 | +0.0 % | -2.7 % (r9c) | issue: stall count after other 73 %; dram: DRAM latency 15 %; fma: stall count after fp32 10 % | default latency 269, issue-only 198168 | batch r9c |
| first_share warm_seq_w4_first | 4 | cycles | 17,568.00 | 21,380.30 | +21.7 % | +75.3 % (r9b) | l2_return: queue (chip L2 share) 46 %; l2: L2 latency 26 %; lsu: queue 20 % | default latency 176, issue-only 2248 | batch r9c |
| first_share rot_seqd2_w2_last | 4 | cycles | 16,792.00 | 21,380.30 | +27.3 % | +102.1 % (r9c) | dram_queue: queue (DRAM share) 41 %; dram: DRAM latency 28 %; l2_return: queue (chip L2 share) 11 % | default latency 88, issue-only 1124 | batch r9c |
| load_curve g1_z0 | 4 | cycles | 379.66 | 398.03 | +4.8 % | +3.3 % (r10) | issue: stall count after other 58 %; l2: L2 latency 29 %; fma: stall count after fp32 8 % | default latency 52, issue-only 4290 | batch r10 |
| load_curve g12_z256 | 4 | cycles | 379.66 | 398.03 | +4.8 % | +3.3 % (r10) | issue: stall count after other 87 %; fma: stall count after fp32 12 % | default latency 229, issue-only 142530 | batch r10 |
| load_curve g24_z64 | 4 | cycles | 380.16 | 399.12 | +5.0 % | +3.1 % (r10) | issue: stall count after other 86 %; fma: stall count after fp32 12 %; issue: default result latency (LDCU) 1 % | default latency 307, issue-only 54216 | batch r10 |
| load_curve g48_z16 | 4 | cycles | 390.28 | 399.53 | +2.4 % | +0.4 % (r10) | issue: stall count after other 82 %; fma: stall count after fp32 11 %; issue: default result latency (LDCU) 5 % | default latency 337, issue-only 18654 | batch r10 |
| load_curve g96_z16 | 4 | cycles | 399.72 | 399.84 | +0.0 % | -1.9 % (r10) | issue: stall count after other 80 %; fma: stall count after fp32 11 %; issue: default result latency (LDCU) 4 % | default latency 673, issue-only 37308 | batch r10 |
| load_curve g144_z16 | 4 | cycles | 415.19 | 406.44 | -2.1 % | -5.0 % (r10) | issue: stall count after other 78 %; fma: stall count after fp32 10 %; issue: scheduler (another warp issued) 7 % | default latency 1009, issue-only 55962 | batch r10 |
| copy_depth warm_t32_k4 | 4 | cycles | 293.00 | 155.00 | -47.1 % | -56.7 % (r10) | l2: L2 latency 62 %; issue: stall count after other 9 %; lsu: address stage 7 % | default latency 5, issue-only 20 | batch r10 |
| copy_depth warm_t32_k4_landed | 4 | cycles | 611.00 | 500.00 | -18.2 % | -22.6 % (r10) | l2: L2 latency 62 %; issue: stall count after other 9 %; lsu: address stage 7 % | default latency 5, issue-only 20 | batch r10 |
| copy_depth warm_t128_k4 | 4 | cycles | 288.00 | 192.00 | -33.3 % | -55.2 % (r10) | l2: L2 latency 50 %; l2_return: queue (SM port rate) 14 %; lsu: queue 9 % | default latency 17, issue-only 80 | batch r10 |
| copy_depth warm_t32_k16 | 4 | cycles | 737.00 | 491.00 | -33.4 % | -48.6 % (r10) | l2: L2 latency 39 %; lsu: address stage 18 %; issue: stall count after other 13 % | default latency 5, issue-only 56 | batch r10 |
| copy_depth warm_t32_k64 | 4 | cycles | 3,026.00 | 2,347.00 | -22.4 % | -35.1 % (r10) | issue: stall count after other 38 %; lsu: address stage 23 %; l2: L2 latency 13 % | default latency 5, issue-only 328 | batch r10 |
| copy_depth warm_t32_k256 | 4 | cycles | 11,666.00 | 9,259.00 | -20.6 % | -33.8 % (r10) | issue: stall count after other 43 %; lsu: address stage 26 %; alu: stall count after alu 13 % | default latency 5, issue-only 1288 | batch r10 |
| copy_depth rot_t128_k8 | 4 | cycles | 834.00 | 352.00 | -57.8 % | -74.5 % (r10) | dram: DRAM latency 63 %; l2_return: queue (SM port rate) 12 %; lsu: queue 7 % | default latency 17, issue-only 128 | batch r10 |
| copy_depth rot_t32_k16_landed | 4 | cycles | 2,144.00 | 1,561.00 | -27.2 % | -32.4 % (r10) | dram: DRAM latency 66 %; lsu: address stage 10 %; issue: stall count after other 7 % | default latency 5, issue-only 56 | batch r10 |
| hand_bcast kernel | 7 | us | 10.72 | 9.24 | -13.8 % | -15.3 % (r10) | launch: launch 31 %; smem_array: queue (shfl) 28 %; l2_return: queue (chip L2 share) 16 % | default latency 440, issue-only 1672 | batch r10 |
| hand_mandel graph | 7 | us | 5.60 | 12.08 | +115.7 % | +104.0 % (r10) | issue: scheduler (another warp issued) 49 %; fma: stall count after fp32 16 %; alu: stall count after alu 11 % | branch fallback 11088, default latency 1056, issue-only 13376 | batch r10 |
| hand_oddeven kernel | 7 | us | 6.66 | 4.60 | -30.9 % | -30.9 % (r10) | launch: launch 63 %; mio: queue (smem_op) 13 %; issue: stall count after barrier 6 % | default latency 528, issue-only 264 | batch r10 |
| stress_00 kernel | 7 | us | 2.46 | 2.53 | +2.7 % | +2.7 % (r10) | launch: launch 87 %; l2/dram: load latency (incl. DRAM queue) 4 %; l2_return: queue (chip L2 share) 3 % | default latency 10, issue-only 28 | batch r10 |
| stress_02 kernel | 7 | us | 2.50 | 2.53 | +1.4 % | +1.4 % (r10) | launch: launch 97 %; store_port: drain after the last instruction 1 % | default latency 40, issue-only 32 | batch r10 |
| stress_10 kernel | 7 | us | 2.94 | 2.53 | -14.1 % | -14.1 % (r10) | launch: launch 83 %; fma: stall count after fp32 5 %; l2/dram: load latency (incl. DRAM queue) 4 % | branch fallback 8, default latency 28, default rbar 20, issue-only 116 | batch r10 |
| stress_12 kernel | 7 | us | 2.50 | 2.53 | +1.4 % | +1.4 % (r10) | launch: launch 95 %; store_port: drain after the last instruction 2 % | default latency 80, issue-only 80 | batch r10 |
| wq_k32_s3_t256 kernel | 7 | us | 238.27 | 247.39 | +3.8 % | -7.2 % (r10) | tensor: queue (mma) 26 %; issue: stall count after barrier 19 %; tensor: stall count after mma 19 % | default latency 216, default rbar 1512, issue-only 13776 | batch r10 |
| wq_k32_s3_t256_nocompute kernel | 7 | us | 120.32 | 109.55 | -8.9 % | -9.2 % (r10) | l2_return: queue (chip L2 share) 70 %; lsu: queue 10 %; l2: L2 latency 8 % | default latency 216, issue-only 1704 | batch r10 |
| wq_m3072_s2 kernel | 7 | us | 211.14 | 217.53 | +3.0 % | +3.5 % (r10) | tensor: queue (mma) 40 %; tensor: stall count after mma 18 %; store_queue: queue (stg) 10 % | default latency 240, default rbar 48, issue-only 4848 | batch r10 |
| wq_m3072_s2_nostore kernel | 7 | us | 192.70 | 191.78 | -0.5 % | -2.5 % (r10) | tensor: queue (mma) 43 %; tensor: stall count after mma 20 %; l2_return: queue (chip L2 share) 11 % | default latency 216, default rbar 48, issue-only 4848 | batch r10 |
| load_sat g12_z0 | 4 | cycles | 389.44 | 532.28 | +36.7 % | +1.0 % (r10c) | issue: stall count after other 52 %; l2: L2 latency 21 %; l2_return: queue (SM port rate) 10 % | default latency 229, issue-only 9510 | batch r10c |
| load_sat g24_z64 | 4 | cycles | 383.44 | 404.06 | +5.4 % | +17.7 % (r10c) | issue: stall count after other 65 %; fma: stall count after fp32 9 %; alu: stall count after alu 8 % | default latency 217, issue-only 41526 | batch r10c |
| load_sat g48_z0 | 4 | cycles | 395.81 | 616.75 | +55.8 % | +4.4 % (r10c) | issue: stall count after other 49 %; l2: L2 latency 20 %; l2_return: queue (chip L2 share) 15 % | default latency 229, issue-only 9510 | batch r10c |
| load_sat g96_z256 | 4 | cycles | 397.41 | 422.52 | +6.3 % | +132.1 % (r10c) | issue: stall count after other 79 %; fma: stall count after fp32 11 %; alu: stall count after alu 3 % | default latency 379, issue-only 233226 | batch r10c |
| load_sat g144_z64 | 4 | cycles | 461.44 | 525.25 | +13.8 % | +768.9 % (r10c) | issue: stall count after other 52 %; issue: scheduler (another warp issued) 9 %; fma: stall count after fp32 7 % | default latency 649, issue-only 124578 | batch r10c |
| lib_E01 kernel | 7 | us | 221.85 | 2.53 | -98.9 % | -98.9 % (r10w) | launch: launch 97 %; alu: stall count after alu 1 %; alu: queue (alu) 1 % | default latency 144, issue-only 24 | batch r10w |
| lib_E02 kernel | 7 | us | 235.86 | 2.53 | -98.9 % | -98.9 % (r10w) | launch: launch 97 %; alu: stall count after alu 1 %; alu: queue (alu) 1 % | default latency 144, issue-only 24 | batch r10w |
| lib_E07 kernel | 7 | us | 609.28 | 4.60 | -99.2 % | -99.6 % (r10w) | launch: launch 72 %; l2_return: queue (chip L2 share) 13 %; lsu: queue 10 % | default latency 240, front end 32, issue-only 48 | batch r10w |
| lib_E08 kernel | 7 | us | 604.03 | 14.21 | -97.6 % | -98.8 % (r10w) | l2_return: queue (chip L2 share) 32 %; l2/dram: load latency (incl. DRAM queue) 21 %; launch: launch 20 % | default latency 1496, front end 176, issue-only 264 | batch r10w |
| l2_cap ld4_g12 | 4 | us | 21.02 | 22.12 | +5.2 % | +3.2 % (r11) | l2/dram: load latency (incl. DRAM queue) 91 %; issue: stall count after other 3 %; alu: stall count after alu 2 % | default latency 32, issue-only 4112 | batch r11 |
| l2_cap ld32_g144 | 4 | us | 70.11 | 78.93 | +12.6 % | -68.3 % (r11) | l2_return: queue (chip L2 share) 70 %; l2/dram: load latency (incl. DRAM queue) 23 %; l2_return: transfer 4 % | default latency 96, issue-only 12336 | batch r11 |
| l2_cap ld128_g480 | 4 | us | 463.26 | 455.10 | -1.8 % | -71.5 % (r11) | l2_return: queue (chip L2 share) 88 %; l2/dram: load latency (incl. DRAM queue) 8 %; l2_return: transfer 2 % | default latency 320, issue-only 41120 | batch r11 |
| l2_cap cp16_g48 | 4 | us | 8.67 | 8.31 | -4.2 % | +73.9 % (r11) | l2: L2 latency 41 %; l2_return: queue (chip L2 share) 34 %; issue: stall count after other 7 % | default latency 288, issue-only 5144 | batch r11 |
| l2_cap cp64_g12 | 4 | us | 14.66 | 9.87 | -32.7 % | -33.7 % (r11) | l2_return: queue (SM port rate) 29 %; l2: L2 latency 26 %; lsu: queue 24 % | default latency 288, issue-only 5144 | batch r11 |
| atom_b1 kernel | 7 | us | 61.73 | 40.53 | -34.3 % | -38.0 % (r11) | store_queue: queue (red) 92 %; launch: launch 7 %; store_port: drain after the last instruction 1 % | default latency 528, issue-only 528 | batch r11 |
| atom_b16_gap128 kernel | 7 | us | 1,030.37 | 1,053.37 | +2.2 % | -96.3 % (r11) | store_queue: queue (red) 100 % | default latency 704, issue-only 528 | batch r11 |
| atom_b256_gap4096 kernel | 7 | us | 131.20 | 144.73 | +10.3 % | -70.8 % (r11) | store_queue: queue (red) 98 %; launch: launch 2 % | default latency 704, issue-only 528 | batch r11 |
| hand_smear kernel | 7 | us | 4.67 | 4.60 | -1.5 % | +166.9 % (r11) | launch: launch 64 %; store_queue: queue (stg) 30 %; fma: queue (fma_half_rate) 2 % | default latency 128, issue-only 448 | batch r11 |
| hand_stencil512 kernel | 7 | us | 4.67 | 4.66 | -0.3 % | -1.5 % (r11) | launch: launch 62 %; l2_return: queue (chip L2 share) 18 %; l2/dram: load latency (incl. DRAM queue) 12 % | default latency 1232, issue-only 176 | batch r11 |
| rw_s16_o0_g48 kernel | 7 | us | 2.59 | 2.53 | -2.4 % | -2.4 % (r11) | launch: launch 92 %; store_queue: queue (stg) 4 %; store_port: drain after the last instruction 1 % | default latency 32, issue-only 72 | batch r11 |
| rw_s32_o0_g48 kernel | 7 | us | 2.59 | 2.53 | -2.4 % | -2.4 % (r11) | launch: launch 92 %; store_queue: queue (stg) 4 %; store_port: drain after the last instruction 2 % | default latency 32, issue-only 72 | batch r11 |
| rw_s32_o0_g768 kernel | 7 | us | 6.62 | 7.88 | +19.0 % | +383.1 % (r11) | store_queue: queue (stg) 54 %; launch: launch 36 %; store_port: drain after the last instruction 6 % | default latency 512, issue-only 1152 | batch r11 |
| rw_s4_o0_g48 kernel | 7 | us | 2.56 | 2.53 | -1.2 % | +79.7 % (r11) | launch: launch 92 %; store_queue: queue (stg) 5 %; store_port: queue (stores) 1 % | default latency 32, issue-only 72 | batch r11 |
| rw_s8_o3_g768 kernel | 7 | us | 4.67 | 6.76 | +44.7 % | +892.7 % (r11) | store_queue: queue (stg) 53 %; launch: launch 42 %; store_port: drain after the last instruction 1 % | default latency 512, issue-only 1152 | batch r11 |
| wr_r1w0_8m kernel | 7 | us | 7.58 | 7.14 | -5.8 % | -7.1 % (r11) | l2_return: queue (chip L2 share) 50 %; launch: launch 40 %; l2/dram: load latency (incl. DRAM queue) 4 % | default latency 96, issue-only 48 | batch r11 |
| wr_r1w0_8m_rot4 kernel | 7 | us | 30.21 | 33.00 | +9.3 % | +7.1 % (r11) | l2/dram: load latency (incl. DRAM queue) 90 %; launch: launch 9 %; lsu: queue 1 % | default latency 96, issue-only 48 | batch r11 |
| wr_r1w0_8m_rot8 kernel | 7 | us | 34.18 | 37.52 | +9.8 % | +7.6 % (r11) | l2/dram: load latency (incl. DRAM queue) 91 %; launch: launch 8 %; lsu: queue 1 % | default latency 96, issue-only 48 | batch r11 |
| wr_r1w1_8m_rot2 kernel | 7 | us | 79.49 | 75.93 | -4.5 % | -6.5 % (r11) | l2/dram: load latency (incl. DRAM queue) 96 %; launch: launch 4 % | default latency 96, issue-only 48 | batch r11 |
| wr_r1w2_32m kernel | 7 | us | 408.83 | 494.40 | +20.9 % | +56.5 % (r11) | l2/dram: load latency (incl. DRAM queue) 97 %; store_queue: queue (stg) 2 %; launch: launch 1 % | default latency 352, issue-only 264 | batch r11 |
| wr_r1w4_4m kernel | 7 | us | 91.68 | 139.67 | +52.3 % | +124.5 % (r11) | store_queue: queue (stg) 88 %; l2/dram: load latency (incl. DRAM queue) 6 %; launch: launch 2 % | default latency 64, issue-only 448 | batch r11 |
| wr_r2w1_16m kernel | 7 | us | 180.54 | 231.67 | +28.3 % | +32.2 % (r11) | l2/dram: load latency (incl. DRAM queue) 98 %; launch: launch 1 % | default latency 192, issue-only 480 | batch r11 |
| l2_local scatter_g48 | 4 | us | 8.80 | 8.31 | -5.5 % | -11.2 % (r11b) | l2: L2 latency 41 %; l2_return: queue (chip L2 share) 34 %; issue: stall count after other 7 % | default latency 288, issue-only 5144 | batch r11b |
| l2_local walk_g48 | 4 | us | 8.77 | 8.31 | -5.2 % | +72.1 % (r11b) | l2: L2 latency 41 %; l2_return: queue (chip L2 share) 33 %; issue: stall count after other 7 % | default latency 288, issue-only 5144 | batch r11b |
| l2_local lockstep_g48 | 4 | us | 10.66 | 8.31 | -22.0 % | +41.5 % (r11b) | l2: L2 latency 41 %; l2_return: queue (chip L2 share) 33 %; issue: stall count after other 8 % | default latency 280, issue-only 4904 | batch r11b |
| l2_local page_g144 | 4 | us | 18.91 | 18.98 | +0.4 % | +107.9 % (r11b) | l2_return: queue (chip L2 share) 70 %; l2: L2 latency 13 %; lsu: queue 7 % | default latency 864, issue-only 15432 | batch r11b |
| wr_r1w1_5m kernel | 7 | us | 12.80 | 12.13 | -5.3 % | -6.9 % (r11b) | l2/dram: load latency (incl. DRAM queue) 72 %; launch: launch 23 %; l2_return: transfer 1 % | default latency 64, issue-only 32 | batch r11b |
| store_merge L1_R4096_S4_g128 | 4 | cycles | 12,270.00 | 17,659.98 | +43.9 % | -78.7 % (r12) | store_queue: queue (stg) 80 %; store_port: queue (stores) 11 %; store_port: drain after the last instruction 9 % | default latency 54, issue-only 42 | batch r12b |
| store_merge L2_R4096_S8_g128 | 4 | cycles | 5,628.00 | 8,313.94 | +47.7 % | -54.0 % (r12) | store_queue: queue (stg) 74 %; store_port: drain after the last instruction 14 %; store_port: queue (stores) 11 % | default latency 54, issue-only 42 | batch r12b |
| store_merge L2_R512_S8_g128 | 4 | cycles | 5,754.00 | 8,313.94 | +44.5 % | -55.1 % (r12) | store_queue: queue (stg) 74 %; store_port: drain after the last instruction 14 %; store_port: queue (stores) 11 % | default latency 54, issue-only 42 | batch r12b |
| store_merge L16_R4096_S16_g128 | 4 | cycles | 293.00 | 459.28 | +56.8 % | +413.1 % (r12) | store_queue: queue (stg) 49 %; store_port: drain after the last instruction 21 %; alu: queue (alu) 5 % | default latency 54, issue-only 39 | batch r12b |
| store_merge L16_R4096_S32_g128 | 4 | cycles | 286.00 | 370.56 | +29.6 % | +37.7 % (r12) | store_port: drain after the last instruction 46 %; store_queue: queue (stg) 29 %; issue: stall count after stg 4 % | default latency 54, issue-only 39 | batch r12b |
| store_merge L32_R4096_S32_g128 | 4 | cycles | 267.00 | 370.56 | +38.8 % | +48.1 % (r12) | store_queue: queue (stg) 43 %; store_port: drain after the last instruction 21 %; issue: stall count after stg 6 % | default latency 54, issue-only 39 | batch r12b |
| bl_aos kernel | 7 | us | 171.20 | 75.27 | -56.0 % | +39.4 % (r12) | store_queue: queue (stg) 88 %; store_port: queue (stores) 5 %; launch: launch 4 % | default latency 96, issue-only 24 | batch r13b |
| bl_aos_f1 kernel | 7 | us | 169.22 | 27.17 | -83.9 % | -48.7 % (r12) | store_queue: queue (stg) 82 %; launch: launch 11 %; store_port: queue (stores) 3 % | default latency 96, issue-only 24 | batch r13b |
| bl_axpy_m8 kernel | 7 | us | 34.66 | 11.61 | -66.5 % | +198.1 % (r12) | l2/dram: load latency (incl. DRAM queue) 38 %; l2_return: queue (chip L2 share) 32 %; launch: launch 25 % | default latency 192, issue-only 96 | batch r13b |
| bl_fanout kernel | 7 | us | 88.48 | 79.58 | -10.1 % | -44.9 % (r12) | store_queue: queue (stg) 92 %; launch: launch 4 %; store_port: queue (stores) 3 % | default latency 352, issue-only 88 | batch r13b |
| bl_fanout_s16 kernel | 7 | us | 86.37 | 81.69 | -5.4 % | -77.3 % (r12) | store_queue: queue (stg) 92 %; launch: launch 4 %; store_port: queue (stores) 3 % | default latency 352, issue-only 88 | batch r13b |
| bl_maskstore_all kernel | 7 | us | 136.77 | 150.90 | +10.3 % | +5.9 % (r12) | l2/dram: load latency (incl. DRAM queue) 97 %; launch: launch 2 %; store_queue: queue (stg) 1 % | default latency 352, issue-only 176 | batch r13b |
| bl_mixrwa kernel | 7 | us | 162.82 | 154.77 | -4.9 % | -8.4 % (r12) | store_queue: queue (red) 89 %; store_port: queue (stores) 5 %; store_queue: queue (stg) 3 % | default latency 440, issue-only 880 | batch r13b |
| bl_mixrwa_noatom kernel | 7 | us | 25.98 | 11.52 | -55.7 % | -59.8 % (r12) | l2/dram: load latency (incl. DRAM queue) 38 %; store_queue: queue (stg) 33 %; launch: launch 25 % | default latency 440, issue-only 176 | batch r13b |
| bl_mixrwa_noatom b2b | 7 | us | 25.16 | 11.46 | -54.5 % | -59.3 % (r12) | l2/dram: load latency (incl. DRAM queue) 51 %; store_queue: queue (stg) 44 %; l2_return: queue (chip L2 share) 2 % | default latency 440, issue-only 176 | batch r13b |
| bl_mixrwa_noatom graph | 7 | us | 23.28 | 9.32 | -60.0 % | -64.5 % (r12) | l2/dram: load latency (incl. DRAM queue) 51 %; store_queue: queue (stg) 44 %; l2_return: queue (chip L2 share) 2 % | default latency 440, issue-only 176 | batch r13b |
| bl_mixrwa_noload kernel | 7 | us | 156.13 | 154.82 | -0.8 % | -3.6 % (r12) | store_queue: queue (red) 76 %; store_port: queue (atomics) 7 %; store_queue: queue (stg) 7 % | default latency 440, issue-only 792 | batch r13b |
| bl_mixrwa_nostore kernel | 7 | us | 155.78 | 153.86 | -1.2 % | -4.7 % (r12) | store_queue: queue (red) 88 %; store_port: queue (atomics) 8 %; launch: launch 2 % | default latency 440, issue-only 880 | batch r13b |
| hgemm_m2560n2048k1536_c128x128x32_s3_t128 kernel | 7 | us | 208.42 | 224.44 | +7.7 % | -2.6 % (r12) | tensor: queue (mma) 38 %; tensor: stall count after mma 21 %; store_queue: queue (stg) 15 % | default latency 252, default rbar 1316, issue-only 9912 | batch r13b |
| hgemm_m2560n2048k1536_c128x128x32_s3_t128_nocompute kernel | 7 | us | 131.71 | 131.00 | -0.5 % | -3.0 % (r12) | l2_return: queue (chip L2 share) 79 %; l2: L2 latency 7 %; lsu: queue 7 % | default latency 252, issue-only 1568 | batch r13b |
| hgemm_m2560n2048k1536_c128x128x32_s3_t128_noload kernel | 7 | us | 185.79 | 157.99 | -15.0 % | -7.1 % (r12) | tensor: queue (mma) 72 %; tensor: stall count after mma 15 %; store_queue: queue (stg) 5 % | default latency 168, issue-only 5936 | batch r13b |
| hgemm_m2560n2048k1536_c128x128x32_s3_t128_nostore kernel | 7 | us | 186.56 | 191.91 | +2.9 % | -4.7 % (r12) | tensor: queue (mma) 44 %; tensor: stall count after mma 25 %; lsu: queue 15 % | default latency 224, default rbar 1288, issue-only 9884 | batch r13b |
| stk_t128_n256_g48 kernel | 7 | us | 467.04 | 469.73 | +0.6 % | -92.5 % (r12) | l1: queue 94 %; l1: latency 4 %; alu: stall count after alu 1 % | default latency 8, issue-only 1188 | batch r12 |
| stk_t128_n64_g144 kernel | 7 | us | 319.84 | 352.99 | +10.4 % | -91.9 % (r12) | l1: queue 97 %; l1: latency 1 %; launch: launch 1 % | default latency 24, issue-only 12 | batch r12 |
| stk_t64_n64_g48 kernel | 7 | us | 61.86 | 61.10 | -1.2 % | -83.8 % (r12) | l1: queue 84 %; l1: latency 8 %; launch: launch 5 % | default latency 4, issue-only 2 | batch r12 |
| wi12_fa_b_warm kernel | 7 | us | 99.46 | 97.41 | -2.1 % | -9.9 % (r12) | tensor: stall count after mma 18 %; lsu: queue 13 %; tensor: queue (mma) 12 % | branch fallback 1024, default latency 1216, default rbar 2848, issue-only 5392 | batch r13b |
| wi12_fa_bn32_warm kernel | 7 | us | 87.33 | 83.70 | -4.2 % | -9.4 % (r12) | tensor: queue (mma) 32 %; tensor: stall count after mma 10 %; lsu: queue 8 % | branch fallback 1024, default latency 1168, default rbar 2672, issue-only 8176 | batch r13b |
| wi12_fa_noexp_warm kernel | 7 | us | 111.14 | 93.88 | -15.5 % | -23.6 % (r12) | tensor: stall count after mma 21 %; tensor: queue (mma) 14 %; lsu: queue 13 % | branch fallback 1024, default latency 1216, default rbar 832, issue-only 5840 | batch r13b |
| wi12_fa_nostore_warm kernel | 7 | us | 98.98 | 96.03 | -3.0 % | -9.4 % (r12) | tensor: stall count after mma 18 %; lsu: queue 13 %; tensor: queue (mma) 13 % | branch fallback 1024, default latency 1184, default rbar 2624, issue-only 5456 | batch r13b |
| wi12_fa_s1_warm kernel | 7 | us | 81.22 | 77.23 | -4.9 % | -10.4 % (r12) | tensor: queue (mma) 37 %; tensor: stall count after mma 12 %; fma: stall count after fp32 7 % | branch fallback 1024, default latency 1168, default rbar 1760, issue-only 5440 | batch r13b |
| wi12_fa_t64_warm kernel | 7 | us | 160.22 | 169.07 | +5.5 % | -1.3 % (r12) | tensor: stall count after mma 28 %; tensor: queue (mma) 19 %; fma: stall count after fp32 13 % | branch fallback 1024, default latency 1096, default rbar 4512, issue-only 9288 | batch r13b |
| wi12_rms_b_rotating kernel | 7 | us | 41.31 | 45.22 | +9.5 % | +12.1 % (r12) | l2/dram: load latency (incl. DRAM queue) 66 %; l1: hit latency 21 %; launch: launch 6 % | default latency 168, issue-only 456 | batch r13b |
| wi12_rms_b_warm kernel | 7 | us | 10.37 | 8.77 | -15.4 % | -8.6 % (r12) | launch: launch 33 %; store_port: queue (stores) 20 %; l2_return: queue (chip L2 share) 13 % | default latency 168, issue-only 456 | batch r13b |
| wi12_rms_b_warm b2b | 7 | us | 9.78 | 8.71 | -10.9 % | -9.6 % (r12) | store_port: queue (stores) 29 %; l2_return: queue (chip L2 share) 20 %; l1: hit latency 11 % | default latency 168, issue-only 456 | batch r13b |
| wi12_rms_b_warm graph | 7 | us | 7.46 | 6.57 | -11.9 % | -12.0 % (r12) | store_port: queue (stores) 29 %; l2_return: queue (chip L2 share) 20 %; l1: hit latency 11 % | default latency 168, issue-only 456 | batch r13b |
| wi12_rms_fp32_rotating kernel | 7 | us | 76.03 | 90.03 | +18.4 % | +14.8 % (r12) | l2/dram: load latency (incl. DRAM queue) 87 %; store_port: queue (stores) 6 %; launch: launch 3 % | default latency 168, issue-only 1104 | batch r13b |
| wi12_rms_fp32_warm kernel | 7 | us | 21.70 | 16.26 | -25.0 % | -26.6 % (r12) | store_port: queue (stores) 31 %; l2_return: queue (chip L2 share) 24 %; launch: launch 18 % | default latency 168, issue-only 1104 | batch r13b |
| wi12_rms_fp32_warm b2b | 7 | us | 21.14 | 16.20 | -23.4 % | -23.9 % (r12) | store_port: queue (stores) 37 %; l2_return: queue (chip L2 share) 29 %; store_queue: queue (stg) 13 % | default latency 168, issue-only 1104 | batch r13b |
| wi12_rms_fp32_warm graph | 7 | us | 19.24 | 14.06 | -26.9 % | -25.1 % (r12) | store_port: queue (stores) 37 %; l2_return: queue (chip L2 share) 29 %; store_queue: queue (stg) 13 % | default latency 168, issue-only 1104 | batch r13b |
| wi12_rms_nocompute_rotating kernel | 7 | us | 40.61 | 41.13 | +1.3 % | -0.8 % (r12) | l2/dram: load latency (incl. DRAM queue) 92 %; launch: launch 7 %; lsu: queue 1 % | default latency 120, issue-only 48 | batch r13b |
| wi12_rms_nostore_rotating kernel | 7 | us | 20.93 | 19.27 | -7.9 % | -11.2 % (r12) | l2/dram: load latency (incl. DRAM queue) 51 %; l1: hit latency 28 %; launch: launch 15 % | default latency 144, issue-only 72 | batch r13b |
| wi12_rms_nostore_rotating b2b | 7 | us | 20.49 | 19.21 | -6.2 % | -4.7 % (r12) | l2/dram: load latency (incl. DRAM queue) 61 %; l1: hit latency 32 %; fma: stall count after fp32 1 % | default latency 144, issue-only 72 | batch r13b |
| wi12_rms_nostore_rotating graph | 7 | us | 17.93 | 17.07 | -4.8 % | -3.5 % (r12) | l2/dram: load latency (incl. DRAM queue) 61 %; l1: hit latency 32 %; fma: stall count after fp32 1 % | default latency 144, issue-only 72 | batch r13b |
| wi12_rms_t64_warm kernel | 7 | us | 10.59 | 8.97 | -15.3 % | -15.8 % (r12) | launch: launch 32 %; l2_return: queue (chip L2 share) 21 %; store_port: queue (stores) 13 % | default latency 84, issue-only 324 | batch r13b |
| wi12_rms_t64_warm b2b | 7 | us | 10.02 | 8.91 | -11.1 % | -11.9 % (r12) | l2_return: queue (chip L2 share) 31 %; store_port: queue (stores) 20 %; store_port: drain after the last instruction 13 % | default latency 84, issue-only 324 | batch r13b |
| wi12_rms_t64_warm graph | 7 | us | 7.55 | 6.77 | -10.3 % | -10.3 % (r12) | l2_return: queue (chip L2 share) 31 %; store_port: queue (stores) 20 %; store_port: drain after the last instruction 13 % | default latency 84, issue-only 324 | batch r13b |
| atw_l32_b1048576 kernel | 7 | us | 78.18 | 74.25 | -5.0 % | -7.2 % (r12b) | store_queue: queue (red) 91 %; launch: launch 4 %; store_port: queue (atomics) 4 % | default latency 704, issue-only 1056 | batch r12b |
| atw_l8_b64 kernel | 7 | us | 1,407.81 | 337.75 | -76.0 % | -76.5 % (r12b) | store_queue: queue (red) 95 %; store_port: queue (atomics) 4 %; launch: launch 1 % | default latency 704, issue-only 1056 | batch r12b |
| rmw_e6 kernel | 7 | us | 15.81 | 9.46 | -40.2 % | +8.4 % (r12b) | l2/dram: load latency (incl. DRAM queue) 46 %; launch: launch 30 %; l2_return: queue (chip L2 share) 19 % | default latency 128, issue-only 64 | batch r12b |
| stu_t64_n1024_g48 kernel | 7 | us | 573.63 | 137.12 | -76.1 % | -87.1 % (r12b) | l1: latency 54 %; l1: queue 16 %; alu: stall count after alu 12 % | default latency 4, issue-only 510 | batch r12b |
| stu_t64_n256_g48 kernel | 7 | us | 41.34 | 32.28 | -21.9 % | -23.7 % (r12b) | l1: latency 58 %; alu: stall count after alu 13 %; launch: launch 9 % | default latency 4, issue-only 16 | batch r12b |
| stu_t64_n256_g96 kernel | 7 | us | 74.08 | 35.53 | -52.0 % | -53.1 % (r12b) | l1: latency 52 %; l1: queue 17 %; alu: stall count after alu 12 % | default latency 8, issue-only 32 | batch r12b |
| wf_m4 kernel | 7 | us | 71.97 | 68.42 | -4.9 % | +336.7 % (r12b) | store_queue: queue (stg) 85 %; store_port: queue (stores) 9 %; launch: launch 4 % | default latency 704, issue-only 6864 | batch r12b |
| wf_m8 kernel | 7 | us | 94.59 | 68.42 | -27.7 % | +232.6 % (r12b) | store_queue: queue (stg) 85 %; store_port: queue (stores) 9 %; launch: launch 4 % | default latency 704, issue-only 6864 | batch r12b |
| wi12b_dec_b_rotating kernel | 7 | us | 23.84 | 23.83 | -0.0 % | -3.3 % (r12b) | dram_queue: queue (DRAM share) 64 %; launch: launch 12 %; dram: DRAM latency 10 % | branch fallback 141, default latency 194, default rbar 104, issue-only 682 | batch r13b |
| wi12b_dec_b_rotating b2b | 7 | us | 23.23 | 23.77 | +2.3 % | -1.2 % (r12b) | dram_queue: queue (DRAM share) 73 %; dram: DRAM latency 11 %; fma: stall count after fp32 3 % | branch fallback 141, default latency 194, default rbar 104, issue-only 682 | batch r13b |
| wi12b_dec_b_rotating graph | 7 | us | 20.87 | 21.63 | +3.6 % | -1.6 % (r12b) | dram_queue: queue (DRAM share) 73 %; dram: DRAM latency 11 %; fma: stall count after fp32 3 % | branch fallback 141, default latency 194, default rbar 104, issue-only 682 | batch r13b |
| wi12b_dec_b_warm kernel | 7 | us | 10.62 | 9.48 | -10.8 % | +0.6 % (r12b) | launch: launch 30 %; tensor: stall count after mma 12 %; lsu: queue 9 % | branch fallback 141, default latency 194, default rbar 104, issue-only 682 | batch r13b |
| wi12b_dec_b_warm b2b | 7 | us | 9.34 | 9.42 | +0.8 % | +5.2 % (r12b) | tensor: stall count after mma 17 %; lsu: queue 13 %; fma: stall count after fp32 12 % | branch fallback 141, default latency 194, default rbar 104, issue-only 682 | batch r13b |
| wi12b_dec_b_warm graph | 7 | us | 7.18 | 7.28 | +1.4 % | -6.4 % (r12b) | tensor: stall count after mma 17 %; lsu: queue 13 %; fma: stall count after fp32 12 % | branch fallback 141, default latency 194, default rbar 104, issue-only 682 | batch r13b |
| wi12b_dec_bn32_rotating kernel | 7 | us | 22.94 | 22.98 | +0.2 % | -2.5 % (r12b) | dram_queue: queue (DRAM share) 60 %; dram: DRAM latency 14 %; launch: launch 13 % | branch fallback 133, default latency 170, default rbar 112, issue-only 802 | batch r13b |
| wi12b_dec_bn32_rotating b2b | 7 | us | 22.43 | 22.92 | +2.2 % | -0.8 % (r12b) | dram_queue: queue (DRAM share) 68 %; dram: DRAM latency 15 %; fma: stall count after fp32 3 % | branch fallback 133, default latency 170, default rbar 112, issue-only 802 | batch r13b |
| wi12b_dec_bn32_rotating graph | 7 | us | 20.04 | 20.78 | +3.7 % | +0.4 % (r12b) | dram_queue: queue (DRAM share) 68 %; dram: DRAM latency 15 %; fma: stall count after fp32 3 % | branch fallback 133, default latency 170, default rbar 112, issue-only 802 | batch r13b |
| wi12b_dec_bn32_warm kernel | 7 | us | 10.72 | 9.91 | -7.6 % | -11.0 % (r12b) | launch: launch 29 %; tensor: stall count after mma 10 %; fma: stall count after fp32 9 % | branch fallback 133, default latency 170, default rbar 112, issue-only 802 | batch r13b |
| wi12b_dec_bn32_warm b2b | 7 | us | 10.14 | 9.85 | -2.9 % | -5.5 % (r12b) | tensor: stall count after mma 14 %; fma: stall count after fp32 13 %; tensor: queue (mma) 11 % | branch fallback 133, default latency 170, default rbar 112, issue-only 802 | batch r13b |
| wi12b_dec_bn32_warm graph | 7 | us | 7.57 | 7.71 | +1.9 % | -1.7 % (r12b) | tensor: stall count after mma 14 %; fma: stall count after fp32 13 %; tensor: queue (mma) 11 % | branch fallback 133, default latency 170, default rbar 112, issue-only 802 | batch r13b |
| wi12b_dec_s1_rotating kernel | 7 | us | 23.07 | 22.72 | -1.5 % | -7.2 % (r12b) | dram_queue: queue (DRAM share) 50 %; dram: DRAM latency 20 %; launch: launch 13 % | branch fallback 133, default latency 170, default rbar 108, issue-only 682 | batch r13b |
| wi12b_dec_s1_rotating b2b | 7 | us | 22.68 | 22.66 | -0.1 % | -2.9 % (r12b) | dram_queue: queue (DRAM share) 58 %; dram: DRAM latency 23 %; fma: stall count after fp32 4 % | branch fallback 133, default latency 170, default rbar 108, issue-only 682 | batch r13b |
| wi12b_dec_s1_rotating graph | 7 | us | 20.28 | 20.52 | +1.2 % | -1.7 % (r12b) | dram_queue: queue (DRAM share) 58 %; dram: DRAM latency 23 %; fma: stall count after fp32 4 % | branch fallback 133, default latency 170, default rbar 108, issue-only 682 | batch r13b |
| wi12b_dec_s1_warm kernel | 7 | us | 8.64 | 9.29 | +7.5 % | +0.9 % (r12b) | launch: launch 31 %; tensor: stall count after mma 12 %; fma: stall count after fp32 9 % | branch fallback 133, default latency 170, default rbar 108, issue-only 682 | batch r13b |
| wi12b_dec_s1_warm b2b | 7 | us | 8.22 | 9.23 | +12.3 % | +6.3 % (r12b) | tensor: stall count after mma 17 %; fma: stall count after fp32 14 %; tensor: queue (mma) 14 % | branch fallback 133, default latency 170, default rbar 108, issue-only 682 | batch r13b |
| wi12b_dec_s1_warm graph | 7 | us | 7.05 | 7.09 | +0.6 % | -6.1 % (r12b) | tensor: stall count after mma 17 %; fma: stall count after fp32 14 %; tensor: queue (mma) 14 % | branch fallback 133, default latency 170, default rbar 108, issue-only 682 | batch r13b |
| wi12b_dec_split16_rotating kernel | 7 | us | 24.93 | 23.36 | -6.3 % | -8.8 % (r12b) | dram_queue: queue (DRAM share) 52 %; dram: DRAM latency 13 %; launch: launch 12 % | branch fallback 266, default latency 346, default rbar 56, issue-only 962 | batch r13b |
| wi12b_dec_split16_rotating b2b | 7 | us | 23.82 | 23.30 | -2.2 % | -2.8 % (r12b) | dram_queue: queue (DRAM share) 60 %; dram: DRAM latency 14 %; fma: stall count after fp32 5 % | branch fallback 266, default latency 346, default rbar 56, issue-only 962 | batch r13b |
| wi12b_dec_split16_rotating graph | 7 | us | 21.58 | 21.16 | -2.0 % | -2.6 % (r12b) | dram_queue: queue (DRAM share) 60 %; dram: DRAM latency 14 %; fma: stall count after fp32 5 % | branch fallback 266, default latency 346, default rbar 56, issue-only 962 | batch r13b |
| wi12b_dec_split16_warm kernel | 7 | us | 10.98 | 10.60 | -3.4 % | -5.0 % (r12b) | launch: launch 27 %; fma: stall count after fp32 10 %; tensor: stall count after mma 9 % | branch fallback 266, default latency 346, default rbar 56, issue-only 962 | batch r13b |
| wi12b_dec_split16_warm b2b | 7 | us | 11.08 | 10.54 | -4.9 % | -5.0 % (r12b) | fma: stall count after fp32 13 %; tensor: stall count after mma 13 %; l2_return: queue (chip L2 share) 11 % | branch fallback 266, default latency 346, default rbar 56, issue-only 962 | batch r13b |
| wi12b_dec_split16_warm graph | 7 | us | 9.18 | 8.40 | -8.5 % | -11.7 % (r12b) | fma: stall count after fp32 13 %; tensor: stall count after mma 13 %; l2_return: queue (chip L2 share) 11 % | branch fallback 266, default latency 346, default rbar 56, issue-only 962 | batch r13b |
| wi12b_dec_split4_rotating kernel | 7 | us | 24.90 | 22.88 | -8.1 % | -7.3 % (r12b) | dram_queue: queue (DRAM share) 55 %; dram: DRAM latency 14 %; launch: launch 13 % | branch fallback 133, default latency 181, default rbar 376, issue-only 799 | batch r13b |
| wi12b_dec_split4_rotating b2b | 7 | us | 23.78 | 22.82 | -4.1 % | -7.2 % (r12b) | dram_queue: queue (DRAM share) 63 %; dram: DRAM latency 16 %; fma: stall count after fp32 4 % | branch fallback 133, default latency 181, default rbar 376, issue-only 799 | batch r13b |
| wi12b_dec_split4_rotating graph | 7 | us | 21.59 | 20.68 | -4.2 % | -8.1 % (r12b) | dram_queue: queue (DRAM share) 63 %; dram: DRAM latency 16 %; fma: stall count after fp32 4 % | branch fallback 133, default latency 181, default rbar 376, issue-only 799 | batch r13b |
| wr_e12 kernel | 7 | us | 87.26 | 100.98 | +15.7 % | +12.9 % (r12b) | l2_return: queue (chip L2 share) 87 %; l2/dram: load latency (incl. DRAM queue) 9 %; launch: launch 3 % | default latency 256, issue-only 128 | batch r12b |
| wr_e4 kernel | 7 | us | 8.70 | 7.57 | -13.0 % | -14.4 % (r12b) | l2/dram: load latency (incl. DRAM queue) 58 %; launch: launch 38 %; l2_return: transfer 1 % | default latency 96, issue-only 48 | batch r12b |
| wr_e8 kernel | 7 | us | 30.21 | 45.93 | +52.1 % | +48.4 % (r12b) | store_queue: queue (stg) 83 %; l2/dram: load latency (incl. DRAM queue) 10 %; launch: launch 6 % | default latency 192, issue-only 96 | batch r12b |
| bl13_interleave_t128 kernel | 7 | us | 144.58 | 195.34 | +35.1 % | +59.3 % (r13b) | l2/dram: load latency (incl. DRAM queue) 96 %; store_queue: queue (stg) 2 %; launch: launch 1 % | default latency 528, issue-only 88 | batch r13b |
| bl13_recordfill_w4 kernel | 7 | us | 84.29 | 52.43 | -37.8 % | +108.9 % (r13b) | store_queue: queue (stg) 92 %; launch: launch 6 %; store_port: drain after the last instruction 2 % | default latency 704, issue-only 176 | batch r13b |
| bl13_swaprows kernel | 7 | us | 24.93 | 11.19 | -55.1 % | -56.2 % (r13b) | store_queue: queue (stg) 58 %; launch: launch 26 %; l2/dram: load latency (incl. DRAM queue) 11 % | default latency 352, issue-only 176 | batch r13b |
| bl13_swaprows b2b | 7 | us | 24.50 | 11.13 | -54.6 % | -55.6 % (r13b) | store_queue: queue (stg) 78 %; l2/dram: load latency (incl. DRAM queue) 15 %; l2_return: queue (chip L2 share) 4 % | default latency 352, issue-only 176 | batch r13b |
| bl13_swaprows graph | 7 | us | 22.31 | 8.99 | -59.7 % | -60.9 % (r13b) | store_queue: queue (stg) 78 %; l2/dram: load latency (incl. DRAM queue) 15 %; l2_return: queue (chip L2 share) 4 % | default latency 352, issue-only 176 | batch r13b |
| bl13_swaprows_l1 kernel | 7 | us | 22.78 | 11.16 | -51.0 % | -52.2 % (r13b) | store_queue: queue (stg) 51 %; launch: launch 26 %; l2/dram: load latency (incl. DRAM queue) 20 % | default latency 352, issue-only 176 | batch r13b |
| bl13_swaprows_l1 b2b | 7 | us | 22.27 | 11.10 | -50.2 % | -51.3 % (r13b) | store_queue: queue (stg) 69 %; l2/dram: load latency (incl. DRAM queue) 27 %; l2_return: queue (chip L2 share) 1 % | default latency 352, issue-only 176 | batch r13b |
| bl13_swaprows_l1 graph | 7 | us | 19.94 | 8.96 | -55.1 % | -56.4 % (r13b) | store_queue: queue (stg) 69 %; l2/dram: load latency (incl. DRAM queue) 27 %; l2_return: queue (chip L2 share) 1 % | default latency 352, issue-only 176 | batch r13b |
| bl13_ticket kernel | 7 | us | 24.86 | 47.73 | +92.0 % | +86.8 % (r13b) | store_queue: queue (red) 59 %; store_port: queue (atomics) 14 %; store_queue: queue (stg) 14 % | default latency 1232, issue-only 880 | batch r13b |
| bl13_ticket b2b | 7 | us | 24.43 | 47.67 | +95.1 % | +89.8 % (r13b) | store_queue: queue (red) 63 %; store_port: queue (atomics) 15 %; store_queue: queue (stg) 14 % | default latency 1232, issue-only 880 | batch r13b |
| bl13_ticket graph | 7 | us | 21.79 | 45.53 | +108.9 % | +103.0 % (r13b) | store_queue: queue (red) 63 %; store_port: queue (atomics) 15 %; store_queue: queue (stg) 14 % | default latency 1232, issue-only 880 | batch r13b |
| bl13_ticket_noret kernel | 7 | us | 16.83 | 12.04 | -28.4 % | +124.6 % (r13b) | store_queue: queue (red) 73 %; launch: launch 24 %; store_port: drain after the last instruction 2 % | default latency 528, issue-only 528 | batch r13b |
| bl13_ticket_noret b2b | 7 | us | 16.41 | 11.98 | -27.0 % | +130.0 % (r13b) | store_queue: queue (red) 97 %; store_port: drain after the last instruction 2 % | default latency 528, issue-only 528 | batch r13b |
| bl13_ticket_noret graph | 7 | us | 15.12 | 9.84 | -34.9 % | +135.4 % (r13b) | store_queue: queue (red) 97 %; store_port: drain after the last instruction 2 % | default latency 528, issue-only 528 | batch r13b |
| bl13_tilefill_p2 kernel | 7 | us | 24.00 | 51.74 | +115.6 % | +254.0 % (r13b) | store_queue: queue (stg) 93 %; launch: launch 6 %; store_port: drain after the last instruction 1 % | default latency 1056, issue-only 176 | batch r13b |
| bl13_transpose kernel | 7 | us | 74.05 | 75.43 | +1.9 % | -77.3 % (r13b) | store_queue: queue (stg) 83 %; store_port: queue (stores) 11 %; launch: launch 4 % | default latency 440, issue-only 176 | batch r13b |
| bl13_transpose_t128 kernel | 7 | us | 72.67 | 75.46 | +3.8 % | -20.3 % (r13b) | store_queue: queue (stg) 83 %; store_port: queue (stores) 11 %; launch: launch 4 % | default latency 440, issue-only 264 | batch r13b |
| dec13_b_rotating kernel | 7 | us | 35.36 | 35.32 | -0.1 % | -71.9 % (r13b) | dram_queue: queue (DRAM share) 71 %; dram: DRAM latency 11 %; launch: launch 8 % | branch fallback 101, default latency 141, default rbar 240, issue-only 895 | batch r13b |
| dec13_b_warm kernel | 7 | us | 10.69 | 10.14 | -5.1 % | -7.1 % (r13b) | launch: launch 28 %; tensor: stall count after mma 12 %; fma: stall count after fp32 10 % | branch fallback 101, default latency 141, default rbar 240, issue-only 895 | batch r13b |
| dec13_b_warm b2b | 7 | us | 10.25 | 10.08 | -1.7 % | -3.8 % (r13b) | tensor: stall count after mma 17 %; fma: stall count after fp32 14 %; tensor: queue (mma) 12 % | branch fallback 101, default latency 141, default rbar 240, issue-only 895 | batch r13b |
| dec13_b_warm graph | 7 | us | 8.31 | 7.94 | -4.5 % | -7.1 % (r13b) | tensor: stall count after mma 17 %; fma: stall count after fp32 14 %; tensor: queue (mma) 12 % | branch fallback 101, default latency 141, default rbar 240, issue-only 895 | batch r13b |
| dec13_bn32_rotating kernel | 7 | us | 34.75 | 34.92 | +0.5 % | -69.4 % (r13b) | dram_queue: queue (DRAM share) 70 %; dram: DRAM latency 13 %; launch: launch 8 % | branch fallback 101, default latency 141, default rbar 72, issue-only 1079 | batch r13b |
| dec13_bn32_warm kernel | 7 | us | 12.48 | 10.81 | -13.4 % | -14.7 % (r13b) | launch: launch 26 %; tensor: stall count after mma 12 %; fma: stall count after fp32 10 % | branch fallback 101, default latency 141, default rbar 72, issue-only 1079 | batch r13b |
| dec13_bn32_warm b2b | 7 | us | 10.27 | 10.75 | +4.7 % | +3.1 % (r13b) | tensor: stall count after mma 16 %; fma: stall count after fp32 13 %; tensor: queue (mma) 10 % | branch fallback 101, default latency 141, default rbar 72, issue-only 1079 | batch r13b |
| dec13_bn32_warm graph | 7 | us | 8.84 | 8.61 | -2.7 % | -4.6 % (r13b) | tensor: stall count after mma 16 %; fma: stall count after fp32 13 %; tensor: queue (mma) 10 % | branch fallback 101, default latency 141, default rbar 72, issue-only 1079 | batch r13b |
| dec13_s1_warm kernel | 7 | us | 10.75 | 10.36 | -3.6 % | -8.9 % (r13b) | launch: launch 28 %; tensor: stall count after mma 11 %; fma: stall count after fp32 9 % | branch fallback 101, default latency 141, default rbar 156, issue-only 707 | batch r13b |
| dec13_s1_warm b2b | 7 | us | 10.23 | 10.30 | +0.7 % | -4.8 % (r13b) | tensor: stall count after mma 15 %; fma: stall count after fp32 12 %; lsu: queue 11 % | branch fallback 101, default latency 141, default rbar 156, issue-only 707 | batch r13b |
| dec13_s1_warm graph | 7 | us | 7.88 | 8.16 | +3.5 % | -3.6 % (r13b) | tensor: stall count after mma 15 %; fma: stall count after fp32 12 %; lsu: queue 11 % | branch fallback 101, default latency 141, default rbar 156, issue-only 707 | batch r13b |
| dec13_s3_rotating kernel | 7 | us | 36.19 | 35.65 | -1.5 % | -71.7 % (r13b) | dram_queue: queue (DRAM share) 73 %; dram: DRAM latency 9 %; launch: launch 8 % | branch fallback 101, default latency 165, default rbar 168, issue-only 919 | batch r13b |
| dec13_split16_warm kernel | 7 | us | 12.80 | 12.17 | -4.9 % | -7.3 % (r13b) | launch: launch 24 %; l2_return: queue (chip L2 share) 12 %; tensor: stall count after mma 10 % | branch fallback 202, default latency 316, default rbar 112, issue-only 1080 | batch r13b |
| dec13_split16_warm b2b | 7 | us | 12.31 | 12.11 | -1.6 % | -4.1 % (r13b) | l2_return: queue (chip L2 share) 16 %; tensor: stall count after mma 13 %; fma: stall count after fp32 12 % | branch fallback 202, default latency 316, default rbar 112, issue-only 1080 | batch r13b |
| dec13_split16_warm graph | 7 | us | 10.45 | 9.97 | -4.6 % | -7.5 % (r13b) | l2_return: queue (chip L2 share) 16 %; tensor: stall count after mma 13 %; fma: stall count after fp32 12 % | branch fallback 202, default latency 316, default rbar 112, issue-only 1080 | batch r13b |
| dec13_split4_bn32_rotating kernel | 7 | us | 35.17 | 34.84 | -0.9 % | -51.7 % (r13b) | dram_queue: queue (DRAM share) 59 %; dram: DRAM latency 20 %; launch: launch 8 % | branch fallback 101, default latency 141, default rbar 120, issue-only 1655 | batch r13b |
| dec13_split4_bn32_warm kernel | 7 | us | 16.86 | 17.55 | +4.0 % | +0.8 % (r13b) | launch: launch 16 %; tensor: stall count after mma 15 %; fma: stall count after fp32 11 % | branch fallback 101, default latency 141, default rbar 120, issue-only 1655 | batch r13b |
| dec13_split4_bn32_warm b2b | 7 | us | 16.41 | 17.49 | +6.6 % | +3.2 % (r13b) | tensor: stall count after mma 18 %; fma: stall count after fp32 13 %; tensor: queue (mma) 11 % | branch fallback 101, default latency 141, default rbar 120, issue-only 1655 | batch r13b |
| dec13_split4_bn32_warm graph | 7 | us | 14.92 | 15.35 | +2.9 % | -0.8 % (r13b) | tensor: stall count after mma 18 %; fma: stall count after fp32 13 %; tensor: queue (mma) 11 % | branch fallback 101, default latency 141, default rbar 120, issue-only 1655 | batch r13b |
| fa13_c128x64_s2_t128_rotating kernel | 7 | us | 166.27 | 150.18 | -9.7 % | -14.2 % (r13b) | tensor: stall count after mma 23 %; tensor: queue (mma) 16 %; fma: stall count after fp32 14 % | branch fallback 960, default latency 1116, default rbar 8724, issue-only 8664 | batch r13b |
| fa13_c128x64_s2_t256_rotating kernel | 7 | us | 159.97 | 149.00 | -6.9 % | -9.4 % (r13b) | tensor: queue (mma) 37 %; tensor: stall count after mma 9 %; fma: stall count after fp32 8 % | branch fallback 960, default latency 1176, default rbar 4152, issue-only 13728 | batch r13b |
| fa13_c128x64_s2_t256_warm kernel | 7 | us | 142.82 | 143.70 | +0.6 % | -2.5 % (r13b) | tensor: queue (mma) 38 %; tensor: stall count after mma 9 %; fma: stall count after fp32 8 % | branch fallback 960, default latency 1176, default rbar 4152, issue-only 13728 | batch r13b |
| fa13_c64x128_s2_t128_rotating kernel | 7 | us | 188.00 | 162.86 | -13.4 % | -21.5 % (r13b) | tensor: stall count after mma 17 %; tensor: queue (mma) 13 %; fma: stall count after fp32 12 % | branch fallback 960, default latency 1224, default rbar 7128, issue-only 13464 | batch r13b |
| fa13_c64x128_s2_t128_warm kernel | 7 | us | 182.59 | 158.13 | -13.4 % | -20.9 % (r13b) | tensor: stall count after mma 17 %; lsu: queue 14 %; tensor: queue (mma) 13 % | branch fallback 960, default latency 1224, default rbar 7128, issue-only 13464 | batch r13b |
| fa13_c64x32_s3_t128_rotating kernel | 7 | us | 177.50 | 173.44 | -2.3 % | -2.6 % (r13b) | l2_return: queue (chip L2 share) 62 %; dram: DRAM latency 12 %; tensor: queue (mma) 5 % | branch fallback 960, default latency 1176, default rbar 10536, issue-only 22536 | batch r13b |
| fa13_c64x32_s3_t128_warm kernel | 7 | us | 164.26 | 167.55 | +2.0 % | +1.8 % (r13b) | l2_return: queue (chip L2 share) 71 %; l2: L2 latency 5 %; tensor: queue (mma) 4 % | branch fallback 960, default latency 1176, default rbar 10536, issue-only 22536 | batch r13b |
| fa13_c64x64_s1_t128_rotating kernel | 7 | us | 172.48 | 132.01 | -23.5 % | -25.0 % (r13b) | tensor: queue (mma) 43 %; dram: DRAM latency 12 %; l2/dram: load latency (incl. DRAM queue) 6 % | branch fallback 960, default latency 1200, default rbar 5400, issue-only 9288 | batch r13b |
| fa13_c64x64_s2_t128_rotating kernel | 7 | us | 192.51 | 169.37 | -12.0 % | -16.8 % (r13b) | tensor: stall count after mma 16 %; fma: stall count after fp32 14 %; tensor: queue (mma) 12 % | branch fallback 960, default latency 1176, default rbar 4152, issue-only 16776 | batch r13b |
| fa13_c64x64_s2_t128_warm kernel | 7 | us | 181.76 | 165.29 | -9.1 % | -13.7 % (r13b) | tensor: stall count after mma 17 %; fma: stall count after fp32 14 %; tensor: queue (mma) 12 % | branch fallback 960, default latency 1176, default rbar 4152, issue-only 16776 | batch r13b |
| fa13_c64x64_s3_t128_rotating kernel | 7 | us | 184.80 | 170.36 | -7.8 % | -15.0 % (r13b) | tensor: stall count after mma 16 %; fma: stall count after fp32 14 %; tensor: queue (mma) 11 % | branch fallback 960, default latency 1272, default rbar 6288, issue-only 10608 | batch r13b |
| fa13_c64x64_s3_t128_warm kernel | 7 | us | 178.50 | 166.79 | -6.6 % | -13.8 % (r13b) | tensor: stall count after mma 17 %; fma: stall count after fp32 14 %; tensor: queue (mma) 11 % | branch fallback 960, default latency 1272, default rbar 6288, issue-only 10608 | batch r13b |
| fa13_noexp_rotating kernel | 7 | us | 200.86 | 156.67 | -22.0 % | -26.5 % (r13b) | tensor: stall count after mma 21 %; tensor: queue (mma) 15 %; fma: stall count after fp32 10 % | branch fallback 960, default latency 1176, default rbar 96, issue-only 16776 | batch r13b |
| fa13_noexp_warm kernel | 7 | us | 191.97 | 152.41 | -20.6 % | -24.9 % (r13b) | tensor: stall count after mma 22 %; tensor: queue (mma) 16 %; fma: stall count after fp32 11 % | branch fallback 960, default latency 1176, default rbar 96, issue-only 16776 | batch r13b |
| fa13_nostore_rotating kernel | 7 | us | 181.70 | 165.87 | -8.7 % | -13.4 % (r13b) | tensor: stall count after mma 19 %; fma: stall count after fp32 13 %; tensor: queue (mma) 13 % | branch fallback 960, default latency 1152, default rbar 6648, issue-only 16176 | batch r13b |
| fa13_nostore_warm kernel | 7 | us | 175.30 | 162.30 | -7.4 % | -12.2 % (r13b) | tensor: stall count after mma 19 %; fma: stall count after fp32 13 %; tensor: queue (mma) 13 % | branch fallback 960, default latency 1152, default rbar 6648, issue-only 16176 | batch r13b |
| fa13_nov_rotating kernel | 7 | us | 190.88 | 106.86 | -44.0 % | -40.4 % (r13b) | fma: stall count after fp32 14 %; tensor: queue (mma) 13 %; tensor: stall count after mma 10 % | branch fallback 3864, default latency 4848, default rbar 3312, issue-only 73128 | batch r13b |
| gemv13_b_rotating kernel | 7 | us | 41.47 | 46.14 | +11.2 % | +8.2 % (r13b) | l2/dram: load latency (incl. DRAM queue) 74 %; l2_return: queue (chip L2 share) 14 %; launch: launch 6 % | default latency 528, default rbar 1152, issue-only 2560 | batch r13b |
| gemv13_b_warm kernel | 7 | us | 16.83 | 16.19 | -3.8 % | -6.3 % (r13b) | xu: queue (xu) 36 %; alu: queue (alu) 22 %; launch: launch 18 % | default latency 528, default rbar 1152, issue-only 2560 | batch r13b |
| gemv13_b_warm b2b | 7 | us | 16.41 | 16.13 | -1.7 % | -4.2 % (r13b) | xu: queue (xu) 44 %; alu: queue (alu) 26 %; l2/dram: load latency (incl. DRAM queue) 5 % | default latency 528, default rbar 1152, issue-only 2560 | batch r13b |
| gemv13_b_warm graph | 7 | us | 14.84 | 13.99 | -5.8 % | -8.5 % (r13b) | xu: queue (xu) 44 %; alu: queue (alu) 26 %; l2/dram: load latency (incl. DRAM queue) 5 % | default latency 528, default rbar 1152, issue-only 2560 | batch r13b |
| gemv13_mk16_rotating kernel | 7 | us | 41.47 | 46.14 | +11.3 % | +8.2 % (r13b) | l2/dram: load latency (incl. DRAM queue) 60 %; alu: queue (alu) 19 %; l2_return: queue (chip L2 share) 8 % | default latency 464, default rbar 1152, issue-only 2112 | batch r13b |
| gemv13_mk16_warm kernel | 7 | us | 20.51 | 16.61 | -19.0 % | -20.7 % (r13b) | xu: queue (xu) 32 %; alu: queue (alu) 21 %; launch: launch 17 % | default latency 464, default rbar 1152, issue-only 2112 | batch r13b |
| gemv13_mk16_warm b2b | 7 | us | 16.49 | 16.55 | +0.4 % | -1.7 % (r13b) | xu: queue (xu) 39 %; alu: queue (alu) 26 %; l2/dram: load latency (incl. DRAM queue) 7 % | default latency 464, default rbar 1152, issue-only 2112 | batch r13b |
| gemv13_mk16_warm graph | 7 | us | 15.12 | 14.41 | -4.6 % | -6.9 % (r13b) | xu: queue (xu) 39 %; alu: queue (alu) 26 %; l2/dram: load latency (incl. DRAM queue) 7 % | default latency 464, default rbar 1152, issue-only 2112 | batch r13b |
| gemv13_mk4_rotating kernel | 7 | us | 41.34 | 46.08 | +11.5 % | +8.4 % (r13b) | l2/dram: load latency (incl. DRAM queue) 86 %; launch: launch 6 %; l2_return: queue (chip L2 share) 3 % | default latency 528, default rbar 1536, issue-only 5632 | batch r13b |
| gemv13_mk4_warm kernel | 7 | us | 18.91 | 18.60 | -1.6 % | -7.9 % (r13b) | alu: queue (alu) 36 %; xu: queue (xu) 18 %; launch: launch 15 % | default latency 528, default rbar 1536, issue-only 5632 | batch r13b |
| gemv13_mk4_warm b2b | 7 | us | 18.45 | 18.54 | +0.5 % | -5.9 % (r13b) | alu: queue (alu) 42 %; xu: queue (xu) 22 %; l2/dram: load latency (incl. DRAM queue) 10 % | default latency 528, default rbar 1536, issue-only 5632 | batch r13b |
| gemv13_mk4_warm graph | 7 | us | 17.20 | 16.40 | -4.7 % | -11.5 % (r13b) | alu: queue (alu) 42 %; xu: queue (xu) 22 %; l2/dram: load latency (incl. DRAM queue) 10 % | default latency 528, default rbar 1536, issue-only 5632 | batch r13b |
| gemv13_nocompute_rotating kernel | 7 | us | 38.34 | 43.52 | +13.5 % | +10.5 % (r13b) | l2/dram: load latency (incl. DRAM queue) 91 %; launch: launch 7 %; fma: stall count after fp32 1 % | default latency 400, issue-only 2880 | batch r13b |
| gemv13_nocompute_warm kernel | 7 | us | 10.72 | 9.06 | -15.5 % | -18.2 % (r13b) | launch: launch 32 %; l2_return: queue (chip L2 share) 16 %; xu: queue (xu) 14 % | default latency 400, issue-only 2880 | batch r13b |
| gemv13_nocompute_warm b2b | 7 | us | 10.27 | 9.00 | -12.3 % | -15.2 % (r13b) | l2_return: queue (chip L2 share) 23 %; xu: queue (xu) 20 %; l2/dram: load latency (incl. DRAM queue) 18 % | default latency 400, issue-only 2880 | batch r13b |
| gemv13_nocompute_warm graph | 7 | us | 8.93 | 6.86 | -23.2 % | -26.5 % (r13b) | l2_return: queue (chip L2 share) 23 %; xu: queue (xu) 20 %; l2/dram: load latency (incl. DRAM queue) 18 % | default latency 400, issue-only 2880 | batch r13b |
| gemv13_np2_r64_rotating kernel | 7 | us | 44.13 | 48.73 | +10.4 % | +1.8 % (r13b) | l2/dram: load latency (incl. DRAM queue) 87 %; launch: launch 6 %; issue: stall count after half_to_float 1 % | default latency 1184, default rbar 1920, issue-only 2176 | batch r13b |
| gemv13_np2_r64_warm kernel | 7 | us | 16.86 | 15.87 | -5.9 % | -6.7 % (r13b) | xu: queue (xu) 37 %; launch: launch 18 %; alu: queue (alu) 17 % | default latency 1184, default rbar 1920, issue-only 2176 | batch r13b |
| gemv13_np2_r64_warm b2b | 7 | us | 16.41 | 15.81 | -3.7 % | -4.4 % (r13b) | xu: queue (xu) 45 %; alu: queue (alu) 20 %; issue: scheduler (another warp issued) 5 % | default latency 1184, default rbar 1920, issue-only 2176 | batch r13b |
| gemv13_np2_r64_warm graph | 7 | us | 14.99 | 13.67 | -8.8 % | -9.6 % (r13b) | xu: queue (xu) 45 %; alu: queue (alu) 20 %; issue: scheduler (another warp issued) 5 % | default latency 1184, default rbar 1920, issue-only 2176 | batch r13b |
| gemv13_np2_rotating kernel | 7 | us | 39.90 | 46.16 | +15.7 % | +12.5 % (r13b) | l2/dram: load latency (incl. DRAM queue) 88 %; launch: launch 6 %; issue: stall count after half_to_float 1 % | default latency 544, default rbar 1920, issue-only 1792 | batch r13b |
| gemv13_np2_warm kernel | 7 | us | 17.06 | 16.25 | -4.7 % | -6.4 % (r13b) | xu: queue (xu) 41 %; launch: launch 18 %; alu: queue (alu) 14 % | default latency 544, default rbar 1920, issue-only 1792 | batch r13b |
| gemv13_np2_warm b2b | 7 | us | 16.41 | 16.19 | -1.4 % | -3.1 % (r13b) | xu: queue (xu) 50 %; alu: queue (alu) 17 %; l2/dram: load latency (incl. DRAM queue) 9 % | default latency 544, default rbar 1920, issue-only 1792 | batch r13b |
| gemv13_np2_warm graph | 7 | us | 14.40 | 14.05 | -2.4 % | -4.5 % (r13b) | xu: queue (xu) 50 %; alu: queue (alu) 17 %; l2/dram: load latency (incl. DRAM queue) 9 % | default latency 544, default rbar 1920, issue-only 1792 | batch r13b |
| gemv13_np8_r16_rotating kernel | 7 | us | 41.73 | 46.14 | +10.6 % | +7.6 % (r13b) | l2/dram: load latency (incl. DRAM queue) 89 %; launch: launch 6 %; issue: stall count after half_to_float 1 % | default latency 264, default rbar 1152, issue-only 2432 | batch r13b |
| gemv13_np8_r16_warm kernel | 7 | us | 16.86 | 16.84 | -0.1 % | -2.0 % (r13b) | xu: queue (xu) 45 %; launch: launch 17 %; alu: queue (alu) 15 % | default latency 264, default rbar 1152, issue-only 2432 | batch r13b |
| gemv13_np8_r16_warm b2b | 7 | us | 16.44 | 16.78 | +2.1 % | +0.1 % (r13b) | xu: queue (xu) 54 %; alu: queue (alu) 18 %; l2/dram: load latency (incl. DRAM queue) 6 % | default latency 264, default rbar 1152, issue-only 2432 | batch r13b |
| gemv13_np8_r16_warm graph | 7 | us | 15.08 | 14.64 | -2.9 % | -5.0 % (r13b) | xu: queue (xu) 54 %; alu: queue (alu) 18 %; l2/dram: load latency (incl. DRAM queue) 6 % | default latency 264, default rbar 1152, issue-only 2432 | batch r13b |
| gemv13_np8_rotating kernel | 7 | us | 39.42 | 46.14 | +17.0 % | +13.8 % (r13b) | l2/dram: load latency (incl. DRAM queue) 80 %; alu: queue (alu) 8 %; launch: launch 6 % | default latency 520, default rbar 1152, issue-only 2560 | batch r13b |
| gemv13_np8_warm kernel | 7 | us | 16.83 | 16.20 | -3.8 % | -5.7 % (r13b) | xu: queue (xu) 41 %; launch: launch 18 %; alu: queue (alu) 16 % | default latency 520, default rbar 1152, issue-only 2560 | batch r13b |
| gemv13_np8_warm b2b | 7 | us | 16.44 | 16.14 | -1.9 % | -3.8 % (r13b) | xu: queue (xu) 50 %; alu: queue (alu) 19 %; l2/dram: load latency (incl. DRAM queue) 6 % | default latency 520, default rbar 1152, issue-only 2560 | batch r13b |
| gemv13_np8_warm graph | 7 | us | 15.08 | 14.00 | -7.2 % | -9.3 % (r13b) | xu: queue (xu) 50 %; alu: queue (alu) 19 %; l2/dram: load latency (incl. DRAM queue) 6 % | default latency 520, default rbar 1152, issue-only 2560 | batch r13b |
| gemv13_r16_rotating kernel | 7 | us | 46.56 | 46.14 | -0.9 % | -3.5 % (r13b) | l2/dram: load latency (incl. DRAM queue) 89 %; launch: launch 6 %; issue: stall count after half_to_float 1 % | default latency 272, default rbar 1152, issue-only 2432 | batch r13b |
| gemv13_r16_warm kernel | 7 | us | 16.86 | 16.86 | -0.0 % | -2.3 % (r13b) | xu: queue (xu) 43 %; launch: launch 17 %; alu: queue (alu) 12 % | default latency 272, default rbar 1152, issue-only 2432 | batch r13b |
| gemv13_r16_warm b2b | 7 | us | 16.44 | 16.80 | +2.2 % | -0.1 % (r13b) | xu: queue (xu) 52 %; alu: queue (alu) 14 %; l2/dram: load latency (incl. DRAM queue) 8 % | default latency 272, default rbar 1152, issue-only 2432 | batch r13b |
| gemv13_r16_warm graph | 7 | us | 15.06 | 14.66 | -2.7 % | -5.2 % (r13b) | xu: queue (xu) 52 %; alu: queue (alu) 14 %; l2/dram: load latency (incl. DRAM queue) 8 % | default latency 272, default rbar 1152, issue-only 2432 | batch r13b |
| gemv13_r64_rotating kernel | 7 | us | 44.35 | 48.73 | +9.9 % | +1.3 % (r13b) | l2/dram: load latency (incl. DRAM queue) 86 %; launch: launch 6 %; issue: stall count after barrier 2 % | default latency 1168, default rbar 1152, issue-only 3072 | batch r13b |
| gemv13_r64_warm kernel | 7 | us | 18.85 | 15.99 | -15.2 % | -16.4 % (r13b) | xu: queue (xu) 29 %; launch: launch 18 %; alu: queue (alu) 17 % | default latency 1168, default rbar 1152, issue-only 3072 | batch r13b |
| gemv13_r64_warm b2b | 7 | us | 18.41 | 15.93 | -13.5 % | -14.7 % (r13b) | xu: queue (xu) 35 %; alu: queue (alu) 21 %; l2/dram: load latency (incl. DRAM queue) 6 % | default latency 1168, default rbar 1152, issue-only 3072 | batch r13b |
| gemv13_r64_warm graph | 7 | us | 15.74 | 13.79 | -12.4 % | -13.8 % (r13b) | xu: queue (xu) 35 %; alu: queue (alu) 21 %; l2/dram: load latency (incl. DRAM queue) 6 % | default latency 1168, default rbar 1152, issue-only 3072 | batch r13b |
| hgemm_m1792n2560k2304_c128x128x32_s3_t128 kernel | 7 | us | 254.46 | 255.65 | +0.5 % | -6.4 % (r13b) | tensor: queue (mma) 39 %; tensor: stall count after mma 21 %; lsu: queue 15 % | default latency 240, default rbar 1728, issue-only 12000 | batch r13b |
| hgemm_m1792n2560k2304_c128x128x32_s3_t128_rotating kernel | 7 | us | 266.66 | 272.11 | +2.0 % | -1.1 % (r13b) | dram_queue: queue (DRAM share) 26 %; tensor: queue (mma) 25 %; tensor: stall count after mma 16 % | default latency 240, default rbar 1728, issue-only 12000 | batch r13b |
| hgemm_m1792n2560k2304_c128x128x32_s4_t128_rotating kernel | 7 | us | 305.41 | 346.86 | +13.6 % | +3.5 % (r13b) | tensor: stall count after mma 27 %; dram_queue: queue (DRAM share) 20 %; tensor: queue (mma) 15 % | default latency 240, default rbar 48, issue-only 5160 | batch r13b |
| hgemm_m1792n2560k2304_direct kernel | 7 | us | 307.65 | 274.42 | -10.8 % | -3.8 % (r13b) | l2_return: queue (chip L2 share) 27 %; tensor: stall count after mma 20 %; tensor: queue (mma) 13 % | default latency 192, issue-only 6816 | batch r13b |
| hgemm_m1792n2560k2304_noload kernel | 7 | us | 213.41 | 203.34 | -4.7 % | -7.1 % (r13b) | tensor: queue (mma) 69 %; tensor: stall count after mma 18 %; store_queue: queue (stg) 5 % | default latency 144, issue-only 6816 | batch r13b |
| hgemm_m1792n2560k2304_s1 kernel | 7 | us | 250.27 | 247.03 | -1.3 % | -8.3 % (r13b) | tensor: queue (mma) 22 %; l2_return: queue (chip L2 share) 18 %; tensor: stall count after mma 16 % | default latency 216, default rbar 24, issue-only 8568 | batch r13b |
| hgemm_m1792n2560k2304_t256_rotating kernel | 7 | us | 262.43 | 260.49 | -0.7 % | -6.2 % (r13b) | dram_queue: queue (DRAM share) 27 %; tensor: queue (mma) 23 %; issue: stall count after barrier 13 % | default latency 480, default rbar 48, issue-only 8832 | batch r13b |
| hgemm_m1792n2560k2304_t64 kernel | 7 | us | 846.85 | 667.92 | -21.1 % | -24.2 % (r13b) | l1: queue 63 %; tensor: stall count after mma 8 %; issue: stall count after local 5 % | default latency 1032, default rbar 1680, front end 13248, issue-only 112632 | batch r13b |
| hgemm_m1792n2560k2304_t64_rotating kernel | 7 | us | 861.50 | 698.93 | -18.9 % | -23.2 % (r13b) | l1: queue 59 %; tensor: stall count after mma 7 %; dram_queue: queue (DRAM share) 6 % | default latency 1032, default rbar 1680, front end 13248, issue-only 112632 | batch r13b |
| sm13_t128_warm kernel | 7 | us | 9.66 | 8.76 | -9.4 % | -10.7 % (r13b) | launch: launch 33 %; issue: scheduler (another warp issued) 14 %; store_port: drain after the last instruction 9 % | branch fallback 1536, default latency 1920, default rbar 448, issue-only 4800 | batch r13b |
| sm13_t128_warm b2b | 7 | us | 10.23 | 8.70 | -14.9 % | -16.2 % (r13b) | issue: scheduler (another warp issued) 21 %; store_port: drain after the last instruction 14 %; store_queue: queue (stg) 11 % | branch fallback 1536, default latency 1920, default rbar 448, issue-only 4800 | batch r13b |
| sm13_t128_warm graph | 7 | us | 7.88 | 6.56 | -16.8 % | -18.4 % (r13b) | issue: scheduler (another warp issued) 21 %; store_port: drain after the last instruction 14 %; store_queue: queue (stg) 11 % | branch fallback 1536, default latency 1920, default rbar 448, issue-only 4800 | batch r13b |
| sm13_t64_warm kernel | 7 | us | 12.74 | 9.66 | -24.2 % | -25.5 % (r13b) | launch: launch 30 %; l2_return: queue (chip L2 share) 21 %; store_port: drain after the last instruction 12 % | branch fallback 1536, default latency 1728, default rbar 512, issue-only 4832 | batch r13b |
| sm13_t64_warm b2b | 7 | us | 12.29 | 9.60 | -22.0 % | -23.3 % (r13b) | l2_return: queue (chip L2 share) 30 %; store_port: drain after the last instruction 18 %; store_queue: queue (stg) 10 % | branch fallback 1536, default latency 1728, default rbar 512, issue-only 4832 | batch r13b |
| sm13_t64_warm graph | 7 | us | 9.76 | 7.46 | -23.6 % | -25.3 % (r13b) | l2_return: queue (chip L2 share) 30 %; store_port: drain after the last instruction 18 %; store_queue: queue (stg) 10 % | branch fallback 1536, default latency 1728, default rbar 512, issue-only 4832 | batch r13b |
| atb_w1024 kernel | 7 | us | 18.98 | 21.79 | +14.9 % | +2988.1 % (r13c) | store_queue: queue (red) 85 %; launch: launch 13 %; store_port: drain after the last instruction 1 % | default latency 704, issue-only 1056 | batch r13c |
| atk_p2 kernel | 7 | us | 18.91 | 21.80 | +15.3 % | +2998.6 % (r13c) | store_queue: queue (red) 85 %; launch: launch 13 %; store_port: drain after the last instruction 1 % | default latency 704, issue-only 2640 | batch r13c |
| hb_f4 kernel | 7 | us | 123.23 | 127.96 | +3.8 % | -8.3 % (r13c) | store_queue: queue (stg) 96 %; launch: launch 2 %; store_port: drain after the last instruction 1 % | default latency 704, issue-only 176 | batch r13c |
| hb_f64 kernel | 7 | us | 516.29 | 707.53 | +37.0 % | -32.3 % (r13c) | store_queue: queue (stg) 99 % | default latency 704, issue-only 176 | batch r13c |
| hl_f4096 kernel | 7 | us | 63.49 | 60.96 | -4.0 % | -6.5 % (r13c) | l2_return: queue (chip L2 share) 81 %; lsu: queue 12 %; launch: launch 5 % | default latency 880, issue-only 880 | batch r13c |
| hs_f1024 kernel | 7 | us | 135.33 | 146.79 | +8.5 % | -16.9 % (r13c) | store_queue: queue (stg) 93 %; store_port: queue (stores) 4 %; launch: launch 2 % | default latency 704, issue-only 1056 | batch r13c |
| rf_w1_m4 kernel | 7 | us | 6.66 | 5.64 | -15.2 % | -30.5 % (r14) | launch: launch 50 %; store_port: drain after the last instruction 27 %; store_queue: queue (stg) 22 % | default latency 192, issue-only 48 | batch r14 |
| rf_w2_m16 kernel | 7 | us | 83.30 | 27.67 | -66.8 % | +6.5 % (r14) | store_queue: queue (stg) 84 %; launch: launch 10 %; store_port: drain after the last instruction 5 % | default latency 704, issue-only 176 | batch r14 |
| rf_w2_m4 kernel | 7 | us | 9.82 | 8.58 | -12.6 % | -33.7 % (r14) | store_queue: queue (stg) 50 %; launch: launch 33 %; store_port: drain after the last instruction 16 % | default latency 192, issue-only 48 | batch r14 |
| rf_w8_m4 kernel | 7 | us | 31.01 | 26.27 | -15.3 % | -42.5 % (r14) | store_queue: queue (stg) 85 %; launch: launch 11 %; store_port: drain after the last instruction 4 % | default latency 192, issue-only 48 | batch r14 |
| tr_c2048_s3 kernel | 7 | us | 219.52 | 231.62 | +5.5 % | -1.8 % (r13c) | tensor: queue (mma) 41 %; tensor: stall count after mma 21 %; lsu: queue 12 % | default latency 282, default rbar 1512, issue-only 11562 | batch r13c |
| tr_c2048_sw8 kernel | 7 | us | 220.67 | 216.28 | -2.0 % | -9.8 % (r13c) | tensor: queue (mma) 37 %; tensor: stall count after mma 22 %; store_queue: queue (stg) 14 % | default latency 726, default rbar 1572, issue-only 11622 | batch r13c |
| bl14_countblock_nl kernel | 7 | us | 37.31 | 21.88 | -41.4 % | -43.1 % (r14) | store_queue: queue (red) 85 %; launch: launch 13 %; store_port: drain after the last instruction 1 % | default latency 1056, issue-only 1232 | batch r14 |
| bl14_exchring kernel | 7 | us | 39.30 | 39.37 | +0.2 % | -2.7 % (r14) | store_queue: queue (red) 52 %; mio: queue (red) 22 %; store_port: queue (atomics) 9 % | default latency 880, issue-only 704 | batch r14 |
| bl14_exchring_t128 kernel | 7 | us | 39.10 | 38.86 | -0.6 % | -3.5 % (r14) | store_queue: queue (red) 49 %; mio: queue (red) 25 %; store_port: queue (atomics) 11 % | default latency 860, issue-only 688 | batch r14 |
| bl14_histlane_b64 kernel | 7 | us | 510.85 | 482.16 | -5.6 % | -8.4 % (r14) | store_queue: queue (red) 88 %; store_port: queue (atomics) 11 %; launch: launch 1 % | default latency 704, issue-only 1408 | batch r14 |
| bl14_xorwide kernel | 7 | us | 176.54 | 74.54 | -57.8 % | -59.0 % (r14) | store_queue: queue (red) 91 %; launch: launch 4 %; store_port: queue (atomics) 4 % | default latency 704, issue-only 1056 | batch r14 |
| bl14_xorwide_m1 kernel | 7 | us | 78.98 | 74.54 | -5.6 % | -8.4 % (r14) | store_queue: queue (red) 91 %; launch: launch 4 %; store_port: queue (atomics) 4 % | default latency 704, issue-only 1056 | batch r14 |
| dec14_b_rotating kernel | 7 | us | 71.90 | 55.44 | -22.9 % | -44.1 % (r14) | dram_queue: queue (DRAM share) 67 %; dram: DRAM latency 15 %; launch: launch 5 % | branch fallback 234, default latency 378, default rbar 336, issue-only 1826 | batch r14 |
| dec14_b_warm kernel | 7 | us | 21.50 | 17.67 | -17.8 % | -22.2 % (r14) | launch: launch 16 %; tensor: stall count after mma 13 %; fma: stall count after fp32 10 % | branch fallback 234, default latency 378, default rbar 336, issue-only 1826 | batch r14 |
| dec14_b_warm b2b | 7 | us | 20.50 | 17.61 | -14.1 % | -18.7 % (r14) | tensor: stall count after mma 15 %; fma: stall count after fp32 12 %; tensor: queue (mma) 11 % | branch fallback 234, default latency 378, default rbar 336, issue-only 1826 | batch r14 |
| dec14_b_warm graph | 7 | us | 18.50 | 15.47 | -16.3 % | -21.5 % (r14) | tensor: stall count after mma 15 %; fma: stall count after fp32 12 %; tensor: queue (mma) 11 % | branch fallback 234, default latency 378, default rbar 336, issue-only 1826 | batch r14 |
| dec14_split4_bn32_rotating kernel | 7 | us | 58.72 | 55.10 | -6.2 % | +3.4 % (r14) | dram_queue: queue (DRAM share) 72 %; dram: DRAM latency 14 %; launch: launch 5 % | branch fallback 117, default latency 157, default rbar 228, issue-only 2075 | batch r14 |
| dec14_split4_bn32_warm kernel | 7 | us | 18.85 | 18.73 | -0.6 % | -11.6 % (r14) | lsu: queue 18 %; launch: launch 15 %; tensor: stall count after mma 10 % | branch fallback 117, default latency 157, default rbar 228, issue-only 2075 | batch r14 |
| dec14_split4_bn32_warm b2b | 7 | us | 18.45 | 18.67 | +1.2 % | -10.1 % (r14) | lsu: queue 22 %; tensor: stall count after mma 12 %; fma: stall count after fp32 11 % | branch fallback 117, default latency 157, default rbar 228, issue-only 2075 | batch r14 |
| dec14_split4_bn32_warm graph | 7 | us | 17.29 | 16.53 | -4.4 % | -16.4 % (r14) | lsu: queue 22 %; tensor: stall count after mma 12 %; fma: stall count after fp32 11 % | branch fallback 117, default latency 157, default rbar 228, issue-only 2075 | batch r14 |
| gemv14_b_rotating kernel | 7 | us | 35.04 | 34.72 | -0.9 % | +308.9 % (r14) | l2/dram: load latency (incl. DRAM queue) 78 %; alu: queue (alu) 9 %; launch: launch 8 % | default latency 891, default rbar 540, issue-only 2484 | batch r14 |
| gemv14_b_warm kernel | 7 | us | 12.70 | 12.57 | -1.0 % | +135.0 % (r14) | alu: queue (alu) 25 %; launch: launch 23 %; xu: queue (xu) 22 % | default latency 891, default rbar 540, issue-only 2484 | batch r14 |
| gemv14_b_warm b2b | 7 | us | 12.31 | 12.51 | +1.6 % | +142.1 % (r14) | alu: queue (alu) 32 %; xu: queue (xu) 29 %; l2/dram: load latency (incl. DRAM queue) 7 % | default latency 891, default rbar 540, issue-only 2484 | batch r14 |
| gemv14_b_warm graph | 7 | us | 10.94 | 10.37 | -5.2 % | +152.7 % (r14) | alu: queue (alu) 32 %; xu: queue (xu) 29 %; l2/dram: load latency (incl. DRAM queue) 7 % | default latency 891, default rbar 540, issue-only 2484 | batch r14 |
| gemv14_mk16_rotating kernel | 7 | us | 32.93 | 38.28 | +16.3 % | +323.6 % (r14) | l2/dram: load latency (incl. DRAM queue) 77 %; alu: queue (alu) 10 %; launch: launch 8 % | default latency 783, default rbar 1188, issue-only 216 | batch r14 |
| gemv14_mk16_warm kernel | 7 | us | 12.74 | 11.95 | -6.2 % | +132.5 % (r14) | xu: queue (xu) 29 %; launch: launch 24 %; alu: queue (alu) 19 % | default latency 783, default rbar 1188, issue-only 216 | batch r14 |
| gemv14_mk16_warm b2b | 7 | us | 12.31 | 11.89 | -3.4 % | +140.1 % (r14) | xu: queue (xu) 38 %; alu: queue (alu) 25 %; alu: stall count after alu 7 % | default latency 783, default rbar 1188, issue-only 216 | batch r14 |
| gemv14_mk16_warm graph | 7 | us | 10.27 | 9.75 | -5.1 % | +166.9 % (r14) | xu: queue (xu) 38 %; alu: queue (alu) 25 %; alu: stall count after alu 7 % | default latency 783, default rbar 1188, issue-only 216 | batch r14 |
| gemv14_np2_r64_rotating kernel | 7 | us | 31.10 | 39.03 | +25.5 % | +707.1 % (r14) | l2/dram: load latency (incl. DRAM queue) 63 %; l1: hit latency 23 %; launch: launch 7 % | default latency 1782, issue-only 864 | batch r14 |
| gemv14_r16_rotating kernel | 7 | us | 34.08 | 34.78 | +2.1 % | -0.7 % (r14) | l2/dram: load latency (incl. DRAM queue) 85 %; launch: launch 8 %; issue: stall count after half_to_float 1 % | default latency 459, default rbar 810, issue-only 1836 | batch r14 |
| hgemm_m2304n1792k1792_c128x128x32_s3_t128_rotating kernel | 7 | us | 203.33 | 205.13 | +0.9 % | -5.4 % (r14) | tensor: queue (mma) 26 %; dram_queue: queue (DRAM share) 22 %; tensor: stall count after mma 16 % | default latency 240, default rbar 1344, issue-only 9648 | batch r14 |
| hgemm_m2304n1792k1792_c128x128x64_s2_t128_rotating kernel | 7 | us | 217.38 | 258.33 | +18.8 % | +6.7 % (r14) | tensor: stall count after mma 28 %; tensor: queue (mma) 16 %; dram_queue: queue (DRAM share) 14 % | default latency 240, default rbar 48, issue-only 3096 | batch r14 |
| qs1_s1 kernel | 7 | us | 51.49 | 60.65 | +17.8 % | +11.6 % (r14) | l2_return: queue (chip L2 share) 21 %; tensor: stall count after mma 20 %; l2: L2 latency 15 % | default latency 36, default rbar 4, issue-only 1292 | batch r14 |
| qs1_s3 kernel | 7 | us | 47.46 | 49.08 | +3.4 % | -8.9 % (r14) | tensor: stall count after mma 33 %; tensor: queue (mma) 17 %; lsu: queue 14 % | default latency 36, default rbar 252, issue-only 1788 | batch r14 |
| rms14_t128_rotating kernel | 7 | us | 62.88 | 65.69 | +4.5 % | +1.5 % (r14) | l2/dram: load latency (incl. DRAM queue) 60 %; l1: hit latency 32 %; launch: launch 4 % | branch fallback 108, default latency 864, issue-only 1080 | batch r14 |
| rms14_t128_warm kernel | 7 | us | 12.51 | 10.81 | -13.6 % | -15.5 % (r14) | store_queue: queue (stg) 38 %; launch: launch 26 %; l1: hit latency 9 % | branch fallback 108, default latency 864, issue-only 1080 | batch r14 |
| rms14_t128_warm b2b | 7 | us | 12.31 | 10.75 | -12.7 % | -14.6 % (r14) | store_queue: queue (stg) 51 %; l1: hit latency 12 %; l2_return: queue (chip L2 share) 12 % | branch fallback 108, default latency 864, issue-only 1080 | batch r14 |
| rms14_t128_warm graph | 7 | us | 9.49 | 8.61 | -9.3 % | -11.7 % (r14) | store_queue: queue (stg) 51 %; l1: hit latency 12 %; l2_return: queue (chip L2 share) 12 % | branch fallback 108, default latency 864, issue-only 1080 | batch r14 |
| rms14_t16_rotating kernel | 7 | us | 62.75 | 64.78 | +3.2 % | +0.4 % (r14) | l2/dram: load latency (incl. DRAM queue) 66 %; l2_return: queue (chip L2 share) 24 %; launch: launch 4 % | branch fallback 27, default latency 189, issue-only 1296 | batch r14 |
| rms14_t16_warm kernel | 7 | us | 13.66 | 10.86 | -20.5 % | -22.3 % (r14) | launch: launch 26 %; l2_return: queue (chip L2 share) 16 %; store_port: queue (stores) 16 % | branch fallback 27, default latency 189, issue-only 1296 | batch r14 |
| rms14_t16_warm b2b | 7 | us | 12.31 | 10.80 | -12.3 % | -14.2 % (r14) | l2_return: queue (chip L2 share) 22 %; store_port: queue (stores) 22 %; fma: stall count after fp32 14 % | branch fallback 27, default latency 189, issue-only 1296 | batch r14 |
| rms14_t16_warm graph | 7 | us | 10.00 | 8.66 | -13.4 % | -15.8 % (r14) | l2_return: queue (chip L2 share) 22 %; store_port: queue (stores) 22 %; fma: stall count after fp32 14 % | branch fallback 27, default latency 189, issue-only 1296 | batch r14 |
| rms14_t256_rotating kernel | 7 | us | 60.67 | 66.46 | +9.5 % | +5.7 % (r14) | l2/dram: load latency (incl. DRAM queue) 81 %; l1: hit latency 11 %; launch: launch 4 % | branch fallback 216, default latency 1728, issue-only 2160 | batch r14 |
| rms14_t256_warm kernel | 7 | us | 12.77 | 10.46 | -18.1 % | -23.3 % (r14) | launch: launch 29 %; store_queue: queue (stg) 26 %; l2_return: queue (chip L2 share) 13 % | branch fallback 216, default latency 1728, issue-only 2160 | batch r14 |
| rms14_t256_warm b2b | 7 | us | 12.28 | 10.40 | -15.3 % | -20.7 % (r14) | store_queue: queue (stg) 37 %; l2_return: queue (chip L2 share) 18 %; l2/dram: load latency (incl. DRAM queue) 10 % | branch fallback 216, default latency 1728, issue-only 2160 | batch r14 |
| rms14_t256_warm graph | 7 | us | 9.53 | 8.26 | -13.3 % | -20.3 % (r14) | store_queue: queue (stg) 37 %; l2_return: queue (chip L2 share) 18 %; l2/dram: load latency (incl. DRAM queue) 10 % | branch fallback 216, default latency 1728, issue-only 2160 | batch r14 |
| rms14_t32_rotating kernel | 7 | us | 60.90 | 64.34 | +5.7 % | +2.7 % (r14) | l2/dram: load latency (incl. DRAM queue) 80 %; l1: hit latency 13 %; launch: launch 4 % | branch fallback 27, default latency 189, issue-only 837 | batch r14 |
| rms14_t32_warm kernel | 7 | us | 13.50 | 11.06 | -18.1 % | -19.9 % (r14) | store_port: queue (stores) 35 %; launch: launch 26 %; l2_return: queue (chip L2 share) 13 % | branch fallback 27, default latency 189, issue-only 837 | batch r14 |
| rms14_t32_warm b2b | 7 | us | 12.31 | 11.00 | -10.7 % | -12.6 % (r14) | store_port: queue (stores) 47 %; l2_return: queue (chip L2 share) 17 %; store_port: drain after the last instruction 8 % | branch fallback 27, default latency 189, issue-only 837 | batch r14 |
| rms14_t32_warm graph | 7 | us | 10.14 | 8.86 | -12.7 % | -15.1 % (r14) | store_port: queue (stores) 47 %; l2_return: queue (chip L2 share) 17 %; store_port: drain after the last instruction 8 % | branch fallback 27, default latency 189, issue-only 837 | batch r14 |
| rms14_t512_rotating kernel | 7 | us | 58.82 | 68.39 | +16.3 % | +9.3 % (r14) | l2/dram: load latency (incl. DRAM queue) 89 %; launch: launch 4 %; issue: stall count after barrier 2 % | branch fallback 432, default latency 3456, issue-only 4320 | batch r14 |
| rms14_t512_warm kernel | 7 | us | 17.63 | 13.47 | -23.6 % | -37.8 % (r14) | launch: launch 26 %; l2_return: queue (chip L2 share) 13 %; l2/dram: load latency (incl. DRAM queue) 11 % | branch fallback 432, default latency 3456, issue-only 4320 | batch r14 |
| rms14_t512_warm b2b | 7 | us | 16.43 | 13.41 | -18.4 % | -33.6 % (r14) | l2_return: queue (chip L2 share) 18 %; l2/dram: load latency (incl. DRAM queue) 15 %; issue: stall count after barrier 13 % | branch fallback 432, default latency 3456, issue-only 4320 | batch r14 |
| rms14_t512_warm graph | 7 | us | 13.24 | 11.27 | -14.9 % | -33.8 % (r14) | l2_return: queue (chip L2 share) 18 %; l2/dram: load latency (incl. DRAM queue) 15 %; issue: stall count after barrier 13 % | branch fallback 432, default latency 3456, issue-only 4320 | batch r14 |
| rms14_t64_rotating kernel | 7 | us | 60.54 | 65.63 | +8.4 % | +5.4 % (r14) | l2/dram: load latency (incl. DRAM queue) 83 %; l1: hit latency 7 %; launch: launch 4 % | branch fallback 54, default latency 432, issue-only 540 | batch r14 |
| rms14_t64_warm kernel | 7 | us | 13.66 | 11.10 | -18.8 % | -20.5 % (r14) | store_queue: queue (stg) 33 %; launch: launch 26 %; l2_return: queue (chip L2 share) 9 % | branch fallback 54, default latency 432, issue-only 540 | batch r14 |
| rms14_t64_warm b2b | 7 | us | 12.31 | 11.04 | -10.4 % | -12.3 % (r14) | store_queue: queue (stg) 45 %; l2_return: queue (chip L2 share) 12 %; l1: hit latency 10 % | branch fallback 54, default latency 432, issue-only 540 | batch r14 |
| rms14_t64_warm graph | 7 | us | 9.43 | 8.90 | -5.6 % | -8.2 % (r14) | store_queue: queue (stg) 45 %; l2_return: queue (chip L2 share) 12 %; l1: hit latency 10 % | branch fallback 54, default latency 432, issue-only 540 | batch r14 |
| sm14_t128_warm kernel | 7 | us | 13.54 | 10.29 | -24.0 % | -25.6 % (r14) | launch: launch 28 %; store_queue: queue (stg) 19 %; issue: scheduler (another warp issued) 9 % | branch fallback 2160, default latency 2808, default rbar 540, issue-only 6804 | batch r14 |
| sm14_t128_warm b2b | 7 | us | 12.31 | 10.23 | -16.9 % | -18.7 % (r14) | store_queue: queue (stg) 26 %; issue: scheduler (another warp issued) 12 %; l2_return: queue (chip L2 share) 9 % | branch fallback 2160, default latency 2808, default rbar 540, issue-only 6804 | batch r14 |
| sm14_t128_warm graph | 7 | us | 9.72 | 8.09 | -16.8 % | -19.0 % (r14) | store_queue: queue (stg) 26 %; issue: scheduler (another warp issued) 12 %; l2_return: queue (chip L2 share) 9 % | branch fallback 2160, default latency 2808, default rbar 540, issue-only 6804 | batch r14 |
| sm14_t16_warm kernel | 7 | us | 30.43 | 24.03 | -21.0 % | -23.1 % (r14) | fma: stall count after fp32 25 %; xu: xu result latency 19 %; issue: stall count after other 18 % | branch fallback 4320, default latency 4455, default rbar 810, issue-only 13014 | batch r14 |
| sm14_t256_warm kernel | 7 | us | 13.76 | 10.63 | -22.7 % | -27.6 % (r14) | launch: launch 28 %; l2_return: queue (chip L2 share) 9 %; issue: stall count after barrier 7 % | branch fallback 2160, default latency 3456, default rbar 216, issue-only 7128 | batch r14 |
| sm14_t256_warm b2b | 7 | us | 12.34 | 10.57 | -14.3 % | -19.7 % (r14) | l2_return: queue (chip L2 share) 12 %; issue: stall count after barrier 10 %; l2/dram: load latency (incl. DRAM queue) 9 % | branch fallback 2160, default latency 3456, default rbar 216, issue-only 7128 | batch r14 |
| sm14_t256_warm graph | 7 | us | 10.74 | 8.43 | -21.5 % | -27.7 % (r14) | l2_return: queue (chip L2 share) 12 %; issue: stall count after barrier 10 %; l2/dram: load latency (incl. DRAM queue) 9 % | branch fallback 2160, default latency 3456, default rbar 216, issue-only 7128 | batch r14 |
| sm14_t32_warm kernel | 7 | us | 16.74 | 12.81 | -23.5 % | -25.2 % (r14) | launch: launch 22 %; store_queue: queue (stg) 14 %; fma: stall count after fp32 12 % | branch fallback 2160, default latency 2295, default rbar 621, issue-only 6561 | batch r14 |
| sm14_t32_warm b2b | 7 | us | 16.00 | 12.75 | -20.3 % | -22.1 % (r14) | store_queue: queue (stg) 18 %; fma: stall count after fp32 15 %; l2_return: queue (chip L2 share) 14 % | branch fallback 2160, default latency 2295, default rbar 621, issue-only 6561 | batch r14 |
| sm14_t32_warm graph | 7 | us | 13.69 | 10.61 | -22.5 % | -24.6 % (r14) | store_queue: queue (stg) 18 %; fma: stall count after fp32 15 %; l2_return: queue (chip L2 share) 14 % | branch fallback 2160, default latency 2295, default rbar 621, issue-only 6561 | batch r14 |
| sm14_t512_warm kernel | 7 | us | 20.93 | 19.22 | -8.1 % | -20.9 % (r14) | launch: launch 17 %; issue: stall count after barrier 12 %; l2_return: queue (chip L2 share) 9 % | branch fallback 2160, default latency 4752, issue-only 7776 | batch r14 |
| sm14_t512_warm b2b | 7 | us | 20.51 | 19.16 | -6.6 % | -19.6 % (r14) | issue: stall count after barrier 14 %; l2_return: queue (chip L2 share) 11 %; alu: queue (alu) 9 % | branch fallback 2160, default latency 4752, issue-only 7776 | batch r14 |
| sm14_t512_warm graph | 7 | us | 18.16 | 17.02 | -6.2 % | -20.9 % (r14) | issue: stall count after barrier 14 %; l2_return: queue (chip L2 share) 11 %; alu: queue (alu) 9 % | branch fallback 2160, default latency 4752, issue-only 7776 | batch r14 |
| sm14_t64_warm kernel | 7 | us | 15.68 | 11.17 | -28.8 % | -30.3 % (r14) | launch: launch 26 %; store_queue: queue (stg) 16 %; issue: scheduler (another warp issued) 10 % | branch fallback 2160, default latency 2484, default rbar 486, issue-only 6642 | batch r14 |
| sm14_t64_warm b2b | 7 | us | 14.33 | 11.11 | -22.5 % | -24.2 % (r14) | store_queue: queue (stg) 22 %; issue: scheduler (another warp issued) 14 %; l2_return: queue (chip L2 share) 11 % | branch fallback 2160, default latency 2484, default rbar 486, issue-only 6642 | batch r14 |
| sm14_t64_warm graph | 7 | us | 12.34 | 8.97 | -27.4 % | -29.3 % (r14) | store_queue: queue (stg) 22 %; issue: scheduler (another warp issued) 14 %; l2_return: queue (chip L2 share) 11 % | branch fallback 2160, default latency 2484, default rbar 486, issue-only 6642 | batch r14 |
| stl_l_n64 kernel | 7 | us | 63.90 | 141.03 | +120.7 % | -61.9 % (r14) | l1: queue 97 %; launch: launch 2 % | default latency 240, issue-only 96 | batch r14 |
| stl_u_n16 kernel | 7 | us | 2.46 | 2.53 | +2.7 % | +2.7 % (r14) | launch: launch 92 %; store_port: drain after the last instruction 4 %; issue: scheduler (another warp issued) 2 % | default latency 192, issue-only 96 | batch r14 |
| stl_u_n256 kernel | 7 | us | 1,237.34 | 2,456.83 | +98.6 % | -71.6 % (r14) | l1: queue 100 % | default latency 240, issue-only 1632 | batch r14 |
| occ_t1024_k1 kernel | 7 | us | 12.77 | 15.72 | +23.1 % | +20.9 % (r14b) | l2/dram: load latency (incl. DRAM queue) 56 %; launch: launch 18 %; l2_return: transfer 18 % | default latency 129, front end 63, issue-only 255 | batch r14b |
| occ_t128_k1 kernel | 7 | us | 12.64 | 15.72 | +24.3 % | +22.1 % (r14b) | l2/dram: load latency (incl. DRAM queue) 56 %; launch: launch 18 %; l2_return: transfer 18 % | default latency 17, front end 63, issue-only 31 | batch r14b |
| occ_t256_k4 kernel | 7 | us | 14.78 | 15.85 | +7.2 % | +5.3 % (r14b) | l2/dram: load latency (incl. DRAM queue) 55 %; launch: launch 18 %; l2_return: transfer 18 % | default latency 132, front end 252, issue-only 252 | batch r14b |
| occ_t512_k2 kernel | 7 | us | 13.70 | 15.76 | +15.1 % | +13.0 % (r14b) | l2/dram: load latency (incl. DRAM queue) 56 %; launch: launch 18 %; l2_return: transfer 18 % | default latency 130, front end 126, issue-only 254 | batch r14b |
| occ_t64_k8 kernel | 7 | us | 24.99 | 25.60 | +2.4 % | +0.4 % (r14b) | l2_return: queue (chip L2 share) 38 %; l2/dram: load latency (incl. DRAM queue) 34 %; launch: launch 11 % | default latency 72, front end 504, issue-only 120 | batch r14b |
| qs1_s2_nocopy kernel | 7 | us | 39.36 | 41.08 | +4.4 % | -1.4 % (r14b) | tensor: stall count after mma 39 %; tensor: queue (mma) 21 %; store_queue: queue (stg) 12 % | default latency 24, default rbar 8, issue-only 1788 | batch r14b |
| qs2_s3_nocopy kernel | 7 | us | 71.84 | 67.68 | -5.8 % | -9.8 % (r14b) | tensor: queue (mma) 50 %; tensor: stall count after mma 24 %; store_queue: queue (stg) 8 % | default latency 48, default rbar 512, issue-only 4528 | batch r14b |
| qs_s2_late kernel | 7 | us | 211.84 | 190.01 | -10.3 % | -12.2 % (r14b) | tensor: queue (mma) 47 %; tensor: stall count after mma 22 %; store_queue: queue (stg) 8 % | default latency 240, default rbar 48, issue-only 4848 | batch r14b |
| bl15_chunked_w6 kernel | 7 | us | 111.46 | 76.72 | -31.2 % | -24.1 % (r15) | l2/dram: load latency (incl. DRAM queue) 73 %; l2_return: queue (chip L2 share) 20 %; launch: launch 4 % | default latency 860, issue-only 344 | batch r15 |
| bl15_gather3 kernel | 7 | us | 6.69 | 6.85 | +2.4 % | -0.3 % (r15) | launch: launch 42 %; l2_return: queue (chip L2 share) 42 %; l2/dram: load latency (incl. DRAM queue) 8 % | default latency 880, issue-only 352 | batch r15 |
| bl15_morton_k1 kernel | 7 | us | 4.61 | 4.60 | -0.2 % | -0.2 % (r15) | launch: launch 71 %; alu: queue (alu) 9 %; store_port: drain after the last instruction 9 % | default latency 880, issue-only 352 | batch r15 |
| ct_k480 kernel | 7 | us | 72.22 | 66.67 | -7.7 % | -18.4 % (r15) | store_queue: queue (stg) 34 %; tensor: queue (mma) 33 %; tensor: stall count after mma 17 % | default latency 144, default rbar 48, issue-only 3672 | batch r15 |
| ct_k960 kernel | 7 | us | 108.96 | 93.94 | -13.8 % | -17.5 % (r15) | tensor: queue (mma) 43 %; tensor: stall count after mma 24 %; store_queue: queue (stg) 17 % | default latency 144, default rbar 48, issue-only 5832 | batch r15 |
| dec15_b_rotating kernel | 7 | us | 91.46 | 76.47 | -16.4 % | -18.8 % (r15) | dram_queue: queue (DRAM share) 72 %; dram: DRAM latency 14 %; launch: launch 4 % | branch fallback 298, default latency 416, default rbar 280, issue-only 2270 | batch r15 |
| dec15_b_warm kernel | 7 | us | 27.01 | 20.70 | -23.3 % | -26.3 % (r15) | tensor: stall count after mma 14 %; launch: launch 14 %; tensor: queue (mma) 10 % | branch fallback 298, default latency 416, default rbar 280, issue-only 2270 | batch r15 |
| dec15_b_warm b2b | 7 | us | 26.49 | 20.64 | -22.1 % | -25.1 % (r15) | tensor: stall count after mma 16 %; tensor: queue (mma) 12 %; fma: stall count after fp32 12 % | branch fallback 298, default latency 416, default rbar 280, issue-only 2270 | batch r15 |
| dec15_b_warm graph | 7 | us | 24.04 | 18.50 | -23.0 % | -26.4 % (r15) | tensor: stall count after mma 16 %; tensor: queue (mma) 12 %; fma: stall count after fp32 12 % | branch fallback 298, default latency 416, default rbar 280, issue-only 2270 | batch r15 |
| dec15_bn32_rotating kernel | 7 | us | 77.92 | 74.84 | -4.0 % | -7.4 % (r15) | dram_queue: queue (DRAM share) 72 %; dram: DRAM latency 13 %; launch: launch 4 % | branch fallback 298, default latency 372, default rbar 224, issue-only 3184 | batch r15 |
| dec15_bn32_warm kernel | 7 | us | 23.01 | 23.06 | +0.2 % | -1.5 % (r15) | l2_return: queue (chip L2 share) 17 %; lsu: queue 15 %; launch: launch 12 % | branch fallback 298, default latency 372, default rbar 224, issue-only 3184 | batch r15 |
| dec15_bn32_warm b2b | 7 | us | 22.55 | 23.00 | +2.0 % | +0.2 % (r15) | l2_return: queue (chip L2 share) 19 %; lsu: queue 17 %; fma: stall count after fp32 9 % | branch fallback 298, default latency 372, default rbar 224, issue-only 3184 | batch r15 |
| dec15_bn32_warm graph | 7 | us | 20.76 | 20.86 | +0.5 % | -1.5 % (r15) | l2_return: queue (chip L2 share) 19 %; lsu: queue 17 %; fma: stall count after fp32 9 % | branch fallback 298, default latency 372, default rbar 224, issue-only 3184 | batch r15 |
| dec15_s1_rotating kernel | 7 | us | 92.54 | 76.27 | -17.6 % | -20.0 % (r15) | dram_queue: queue (DRAM share) 76 %; dram: DRAM latency 7 %; launch: launch 4 % | branch fallback 298, default latency 372, default rbar 488, issue-only 1726 | batch r15 |
| dec15_s1_warm kernel | 7 | us | 23.52 | 20.59 | -12.5 % | -16.6 % (r15) | tensor: stall count after mma 14 %; launch: launch 14 %; tensor: queue (mma) 11 % | branch fallback 298, default latency 372, default rbar 488, issue-only 1726 | batch r15 |
| dec15_s1_warm b2b | 7 | us | 23.21 | 20.53 | -11.5 % | -15.7 % (r15) | tensor: stall count after mma 16 %; tensor: queue (mma) 13 %; lsu: queue 12 % | branch fallback 298, default latency 372, default rbar 488, issue-only 1726 | batch r15 |
| dec15_s1_warm graph | 7 | us | 21.68 | 18.39 | -15.2 % | -19.7 % (r15) | tensor: stall count after mma 16 %; tensor: queue (mma) 13 %; lsu: queue 12 % | branch fallback 298, default latency 372, default rbar 488, issue-only 1726 | batch r15 |
| dec15_split16_rotating kernel | 7 | us | 100.61 | 74.83 | -25.6 % | -28.0 % (r15) | dram_queue: queue (DRAM share) 54 %; dram: DRAM latency 23 %; launch: launch 4 % | branch fallback 596, default latency 888, default rbar 128, issue-only 2992 | batch r15 |
| dec15_split16_warm kernel | 7 | us | 39.14 | 29.33 | -25.1 % | -27.5 % (r15) | tensor: stall count after mma 13 %; tensor: queue (mma) 11 %; smem_array: ldsm latency 10 % | branch fallback 596, default latency 888, default rbar 128, issue-only 2992 | batch r15 |
| dec15_split4_bn32_rotating kernel | 7 | us | 78.02 | 74.49 | -4.5 % | -7.8 % (r15) | dram_queue: queue (DRAM share) 77 %; dram: DRAM latency 11 %; launch: launch 4 % | branch fallback 146, default latency 186, default rbar 228, issue-only 2468 | batch r15 |
| dec15_split4_bn32_warm kernel | 7 | us | 20.86 | 20.81 | -0.3 % | -2.0 % (r15) | launch: launch 14 %; lsu: queue 14 %; tensor: stall count after mma 11 % | branch fallback 146, default latency 186, default rbar 228, issue-only 2468 | batch r15 |
| dec15_split4_bn32_warm b2b | 7 | us | 20.50 | 20.75 | +1.2 % | -0.5 % (r15) | lsu: queue 16 %; tensor: stall count after mma 12 %; l2_return: queue (chip L2 share) 11 % | branch fallback 146, default latency 186, default rbar 228, issue-only 2468 | batch r15 |
| dec15_split4_bn32_warm graph | 7 | us | 18.40 | 18.61 | +1.1 % | -0.8 % (r15) | lsu: queue 16 %; tensor: stall count after mma 12 %; l2_return: queue (chip L2 share) 11 % | branch fallback 146, default latency 186, default rbar 228, issue-only 2468 | batch r15 |
| dec15_split4_rotating kernel | 7 | us | 92.64 | 74.15 | -20.0 % | -22.1 % (r15) | dram_queue: queue (DRAM share) 78 %; dram: DRAM latency 10 %; launch: launch 4 % | branch fallback 146, default latency 208, default rbar 304, issue-only 1508 | batch r15 |
| dec15_split4_warm kernel | 7 | us | 20.96 | 17.66 | -15.7 % | -20.0 % (r15) | launch: launch 16 %; tensor: stall count after mma 15 %; tensor: queue (mma) 11 % | branch fallback 146, default latency 208, default rbar 304, issue-only 1508 | batch r15 |
| dec15_split4_warm b2b | 7 | us | 20.50 | 17.60 | -14.2 % | -18.5 % (r15) | tensor: stall count after mma 17 %; tensor: queue (mma) 14 %; lsu: queue 12 % | branch fallback 146, default latency 208, default rbar 304, issue-only 1508 | batch r15 |
| dec15_split4_warm graph | 7 | us | 18.60 | 15.46 | -16.9 % | -21.6 % (r15) | tensor: stall count after mma 17 %; tensor: queue (mma) 14 %; lsu: queue 12 % | branch fallback 146, default latency 208, default rbar 304, issue-only 1508 | batch r15 |
| dl_w16_t128_rotating kernel | 7 | us | 68.13 | 76.03 | +11.6 % | -43.3 % (r15) | l2/dram: load latency (incl. DRAM queue) 95 %; launch: launch 4 %; lsu: queue 1 % | default latency 440, issue-only 440 | batch r15 |
| dl_w4_t32_rotating kernel | 7 | us | 66.82 | 74.49 | +11.5 % | -43.3 % (r15) | l2/dram: load latency (incl. DRAM queue) 94 %; launch: launch 4 %; lsu: queue 2 % | default latency 430, issue-only 430 | batch r15 |
| dl_w8_t512_warm kernel | 7 | us | 8.67 | 7.85 | -9.5 % | -10.8 % (r15) | lsu: queue 57 %; launch: launch 36 %; l2/dram: load latency (incl. DRAM queue) 3 % | default latency 480, issue-only 480 | batch r15 |
| fa15_c32x64_s2_t64_rotating kernel | 7 | us | 572.74 | 493.10 | -13.9 % | -19.3 % (r15) | tensor: stall count after mma 19 %; tensor: queue (mma) 14 %; issue: stall count after ldsm 13 % | branch fallback 1296, default latency 1494, default rbar 3366, issue-only 35334 | batch r15 |
| fa15_c32x64_s2_t64_warm kernel | 7 | us | 567.52 | 489.59 | -13.7 % | -19.2 % (r15) | tensor: stall count after mma 19 %; tensor: queue (mma) 15 %; issue: stall count after ldsm 13 % | branch fallback 1296, default latency 1494, default rbar 3366, issue-only 35334 | batch r15 |
| fa15_c64x32_s3_t128_rotating kernel | 7 | us | 353.63 | 338.58 | -4.3 % | -9.0 % (r15) | l2_return: queue (chip L2 share) 40 %; lsu: queue 11 %; dram: DRAM latency 8 % | branch fallback 1440, default latency 1680, default rbar 8680, issue-only 24440 | batch r15 |
| fa15_c64x32_s3_t128_warm kernel | 7 | us | 342.11 | 333.39 | -2.5 % | -7.4 % (r15) | l2_return: queue (chip L2 share) 45 %; lsu: queue 11 %; tensor: stall count after mma 8 % | branch fallback 1440, default latency 1680, default rbar 8680, issue-only 24440 | batch r15 |
| fa15_c64x64_s1_t128_rotating kernel | 7 | us | 329.86 | 288.73 | -12.5 % | -16.2 % (r15) | dram: DRAM latency 18 %; tensor: stall count after mma 15 %; tensor: queue (mma) 11 % | branch fallback 1440, default latency 1620, default rbar 6600, issue-only 11320 | batch r15 |
| fa15_c64x64_s1_t128_warm kernel | 7 | us | 313.50 | 272.38 | -13.1 % | -19.7 % (r15) | tensor: stall count after mma 19 %; tensor: queue (mma) 16 %; lsu: queue 15 % | branch fallback 1440, default latency 1620, default rbar 6600, issue-only 11320 | batch r15 |
| fa15_c64x64_s2_t128_rotating kernel | 7 | us | 323.74 | 283.77 | -12.3 % | -18.8 % (r15) | tensor: stall count after mma 18 %; tensor: queue (mma) 15 %; lsu: queue 15 % | branch fallback 1440, default latency 1680, default rbar 7240, issue-only 24480 | batch r15 |
| fa15_c64x64_s2_t128_warm kernel | 7 | us | 317.50 | 278.30 | -12.3 % | -18.9 % (r15) | tensor: stall count after mma 18 %; tensor: queue (mma) 15 %; lsu: queue 15 % | branch fallback 1440, default latency 1680, default rbar 7240, issue-only 24480 | batch r15 |
| fa15_c64x64_s2_t64_rotating kernel | 7 | us | 879.81 | 653.17 | -25.8 % | -28.0 % (r15) | tensor: stall count after mma 23 %; tensor: queue (mma) 15 %; l1: queue 11 % | branch fallback 1440, default latency 3240, default rbar 3260, front end 12240, issue-only 28680 | batch r15 |
| fa15_c64x64_s2_t64_warm kernel | 7 | us | 785.89 | 649.72 | -17.3 % | -19.7 % (r15) | tensor: stall count after mma 23 %; tensor: queue (mma) 15 %; l1: queue 11 % | branch fallback 1440, default latency 3240, default rbar 3260, front end 12240, issue-only 28680 | batch r15 |
| fad_d64_t128 kernel | 7 | us | 12.70 | 13.62 | +7.2 % | +2.3 % (r15) | launch: launch 21 %; fma: stall count after fp32 12 %; tensor: stall count after mma 12 % | branch fallback 128, default latency 164, default rbar 380, issue-only 796 | batch r15 |
| fad_d80_t128 kernel | 7 | us | 18.82 | 15.64 | -16.9 % | -19.6 % (r15) | launch: launch 18 %; tensor: stall count after mma 12 %; fma: stall count after fp32 12 % | branch fallback 160, default latency 196, default rbar 308, issue-only 1516 | batch r15 |
| fad_d96_t128 kernel | 7 | us | 17.76 | 17.64 | -0.7 % | -5.2 % (r15) | launch: launch 16 %; tensor: stall count after mma 13 %; fma: stall count after fp32 10 % | branch fallback 192, default latency 228, default rbar 168, issue-only 1728 | batch r15 |
| gd_np2_r64 kernel | 7 | us | 36.00 | 38.76 | +7.7 % | -5.3 % (r15) | l2/dram: load latency (incl. DRAM queue) 63 %; l1: hit latency 23 %; launch: launch 7 % | default latency 1782, issue-only 864 | batch r15 |
| gemv15_b_warm kernel | 7 | us | 18.85 | 17.10 | -9.3 % | -10.9 % (r15) | alu: queue (alu) 30 %; xu: queue (xu) 23 %; launch: launch 17 % | default latency 928, default rbar 2304, issue-only 3456 | batch r15 |
| gemv15_b_warm b2b | 7 | us | 18.42 | 17.04 | -7.5 % | -9.2 % (r15) | alu: queue (alu) 36 %; xu: queue (xu) 28 %; l2/dram: load latency (incl. DRAM queue) 8 % | default latency 928, default rbar 2304, issue-only 3456 | batch r15 |
| gemv15_b_warm graph | 7 | us | 15.87 | 14.90 | -6.1 % | -8.1 % (r15) | alu: queue (alu) 36 %; xu: queue (xu) 28 %; l2/dram: load latency (incl. DRAM queue) 8 % | default latency 928, default rbar 2304, issue-only 3456 | batch r15 |
| gemv15_mk16_warm kernel | 7 | us | 16.83 | 16.62 | -1.3 % | -3.1 % (r15) | xu: queue (xu) 32 %; alu: queue (alu) 20 %; launch: launch 17 % | default latency 928, default rbar 1152, issue-only 2816 | batch r15 |
| gemv15_mk16_warm b2b | 7 | us | 16.41 | 16.56 | +0.9 % | -0.9 % (r15) | xu: queue (xu) 39 %; alu: queue (alu) 24 %; l2/dram: load latency (incl. DRAM queue) 5 % | default latency 928, default rbar 1152, issue-only 2816 | batch r15 |
| gemv15_mk16_warm graph | 7 | us | 14.86 | 14.42 | -3.0 % | -5.0 % (r15) | xu: queue (xu) 39 %; alu: queue (alu) 24 %; l2/dram: load latency (incl. DRAM queue) 5 % | default latency 928, default rbar 1152, issue-only 2816 | batch r15 |
| gemv15_np2_r64_warm kernel | 7 | us | 18.85 | 16.51 | -12.4 % | -18.2 % (r15) | alu: queue (alu) 28 %; launch: launch 18 %; xu: queue (xu) 15 % | default latency 2112, default rbar 1280, issue-only 1280 | batch r15 |
| gemv15_np2_r64_warm b2b | 7 | us | 18.43 | 16.45 | -10.7 % | -16.7 % (r15) | alu: queue (alu) 35 %; xu: queue (xu) 18 %; fma: queue (fma_half_rate) 9 % | default latency 2112, default rbar 1280, issue-only 1280 | batch r15 |
| gemv15_np2_r64_warm graph | 7 | us | 15.85 | 14.31 | -9.7 % | -16.6 % (r15) | alu: queue (alu) 35 %; xu: queue (xu) 18 %; fma: queue (fma_half_rate) 9 % | default latency 2112, default rbar 1280, issue-only 1280 | batch r15 |
| hgemm_m2688n1920k1728_c128x128x32_s2_t128 kernel | 7 | us | 213.25 | 209.95 | -1.5 % | -1.2 % (r15) | tensor: queue (mma) 28 %; tensor: stall count after mma 19 %; l2_return: queue (chip L2 share) 15 % | default latency 280, default rbar 56, issue-only 5096 | batch r15 |
| hgemm_m2688n1920k1728_c128x128x32_s2_t128_rotating kernel | 7 | us | 227.55 | 247.72 | +8.9 % | +3.0 % (r15) | tensor: queue (mma) 19 %; dram_queue: queue (DRAM share) 16 %; tensor: stall count after mma 15 % | default latency 280, default rbar 56, issue-only 5096 | batch r15 |
| hgemm_m2688n1920k1728_c128x128x32_s3_t128 kernel | 7 | us | 226.62 | 244.29 | +7.8 % | -0.5 % (r15) | tensor: queue (mma) 39 %; tensor: stall count after mma 22 %; lsu: queue 13 % | default latency 280, default rbar 1512, issue-only 10976 | batch r15 |
| hgemm_m2688n1920k1728_c128x128x32_s3_t128_rotating kernel | 7 | us | 233.76 | 247.68 | +6.0 % | -6.0 % (r15) | tensor: queue (mma) 23 %; dram_queue: queue (DRAM share) 21 %; tensor: stall count after mma 18 % | default latency 280, default rbar 1512, issue-only 10976 | batch r15 |
| hgemm_m2688n1920k1728_c128x128x32_s3_t128_sw8 kernel | 7 | us | 225.70 | 237.07 | +5.0 % | -14.7 % (r15) | tensor: queue (mma) 38 %; tensor: stall count after mma 22 %; store_queue: queue (stg) 14 % | default latency 280, default rbar 1548, issue-only 10928 | batch r15 |
| hgemm_m2688n1920k1728_c128x128x32_s3_t128_sw8_rotating kernel | 7 | us | 225.79 | 253.07 | +12.1 % | +1.1 % (r15) | tensor: queue (mma) 27 %; tensor: stall count after mma 18 %; dram_queue: queue (DRAM share) 17 % | default latency 280, default rbar 1548, issue-only 10928 | batch r15 |
| hgemm_m2688n1920k1728_c128x128x32_s4_t128 kernel | 7 | us | 252.16 | 266.46 | +5.7 % | -5.4 % (r15) | tensor: stall count after mma 35 %; tensor: queue (mma) 19 %; lsu: queue 16 % | default latency 280, default rbar 56, issue-only 5012 | batch r15 |
| hgemm_m2688n1920k1728_c128x128x32_s4_t128_rotating kernel | 7 | us | 262.62 | 298.87 | +13.8 % | +3.9 % (r15) | tensor: stall count after mma 28 %; dram_queue: queue (DRAM share) 16 %; tensor: queue (mma) 15 % | default latency 280, default rbar 56, issue-only 5012 | batch r15 |
| hgemm_m2688n1920k1728_c128x128x64_s2_t128 kernel | 7 | us | 229.63 | 258.64 | +12.6 % | -4.5 % (r15) | tensor: stall count after mma 34 %; tensor: queue (mma) 19 %; lsu: queue 16 % | default latency 280, default rbar 56, issue-only 3584 | batch r15 |
| hgemm_m2688n1920k1728_c128x128x64_s2_t128_rotating kernel | 7 | us | 248.32 | 293.94 | +18.4 % | +7.0 % (r15) | tensor: stall count after mma 27 %; tensor: queue (mma) 15 %; dram_queue: queue (DRAM share) 14 % | default latency 280, default rbar 56, issue-only 3584 | batch r15 |
| hgemm_m2688n1920k1728_c128x256x32_s3_t256 kernel | 7 | us | 266.78 | 283.24 | +6.2 % | -16.2 % (r15) | tensor: queue (mma) 25 %; issue: stall count after barrier 19 %; tensor: stall count after mma 19 % | default latency 320, default rbar 1696, issue-only 15840 | batch r15 |
| hgemm_m2688n1920k1728_c128x256x32_s3_t256_rotating kernel | 7 | us | 268.64 | 295.48 | +10.0 % | -10.9 % (r15) | issue: stall count after barrier 26 %; tensor: stall count after mma 15 %; store_queue: queue (stg) 14 % | default latency 320, default rbar 1696, issue-only 15840 | batch r15 |
| hgemm_m2688n1920k1728_c256x128x32_s2_t256 kernel | 7 | us | 271.55 | 277.50 | +2.2 % | -6.1 % (r15) | tensor: queue (mma) 24 %; tensor: stall count after mma 19 %; issue: stall count after barrier 17 % | default latency 320, issue-only 22528 | batch r15 |
| hgemm_m2688n1920k1728_c256x128x32_s2_t256_rotating kernel | 7 | us | 280.80 | 296.16 | +5.5 % | -1.9 % (r15) | tensor: queue (mma) 21 %; issue: stall count after barrier 19 %; tensor: stall count after mma 16 % | default latency 320, issue-only 22528 | batch r15 |
| hgemm_m2688n1920k1728_c64x128x32_s4_t128 kernel | 7 | us | 239.10 | 231.02 | -3.4 % | -6.1 % (r15) | l2_return: queue (chip L2 share) 37 %; tensor: queue (mma) 18 %; tensor: stall count after mma 11 % | default latency 560, issue-only 8232 | batch r15 |
| hgemm_m2688n1920k1728_c64x128x32_s4_t128_rotating kernel | 7 | us | 253.38 | 265.14 | +4.6 % | +1.8 % (r15) | l2_return: queue (chip L2 share) 28 %; dram_queue: queue (DRAM share) 19 %; tensor: queue (mma) 14 % | default latency 560, issue-only 8232 | batch r15 |
| rms15_t128_rotating kernel | 7 | us | 62.62 | 68.03 | +8.6 % | +6.4 % (r15) | l2/dram: load latency (incl. DRAM queue) 74 %; l1: hit latency 18 %; launch: launch 4 % | branch fallback 160, default latency 1280, issue-only 1600 | batch r15 |
| rms15_t128_warm kernel | 7 | us | 12.80 | 10.65 | -16.8 % | -18.1 % (r15) | store_queue: queue (stg) 39 %; launch: launch 27 %; l2_return: queue (chip L2 share) 9 % | branch fallback 160, default latency 1280, issue-only 1600 | batch r15 |
| rms15_t128_warm b2b | 7 | us | 12.56 | 10.59 | -15.6 % | -17.0 % (r15) | store_queue: queue (stg) 53 %; l2_return: queue (chip L2 share) 12 %; l2/dram: load latency (incl. DRAM queue) 7 % | branch fallback 160, default latency 1280, issue-only 1600 | batch r15 |
| rms15_t128_warm graph | 7 | us | 9.80 | 8.45 | -13.7 % | -15.5 % (r15) | store_queue: queue (stg) 53 %; l2_return: queue (chip L2 share) 12 %; l2/dram: load latency (incl. DRAM queue) 7 % | branch fallback 160, default latency 1280, issue-only 1600 | batch r15 |
| rms15_t16_rotating kernel | 7 | us | 65.98 | 67.31 | +2.0 % | -0.1 % (r15) | l2/dram: load latency (incl. DRAM queue) 90 %; launch: launch 4 %; fma: stall count after fp32 2 % | branch fallback 40, default latency 280, issue-only 2520 | batch r15 |
| rms15_t16_warm kernel | 7 | us | 15.20 | 10.76 | -29.2 % | -30.4 % (r15) | launch: launch 26 %; store_port: queue (stores) 24 %; fma: stall count after fp32 10 % | branch fallback 40, default latency 280, issue-only 2520 | batch r15 |
| rms15_t16_warm b2b | 7 | us | 12.93 | 10.70 | -17.3 % | -18.6 % (r15) | store_port: queue (stores) 32 %; fma: stall count after fp32 13 %; l2_return: queue (chip L2 share) 12 % | branch fallback 40, default latency 280, issue-only 2520 | batch r15 |
| rms15_t16_warm graph | 7 | us | 10.09 | 8.56 | -15.2 % | -16.9 % (r15) | store_port: queue (stores) 32 %; fma: stall count after fp32 13 %; l2_return: queue (chip L2 share) 12 % | branch fallback 40, default latency 280, issue-only 2520 | batch r15 |
| rms15_t256_rotating kernel | 7 | us | 65.60 | 69.43 | +5.8 % | +1.6 % (r15) | l2/dram: load latency (incl. DRAM queue) 86 %; l1: hit latency 4 %; launch: launch 4 % | branch fallback 320, default latency 2560, issue-only 3200 | batch r15 |
| rms15_t256_warm kernel | 7 | us | 17.44 | 11.46 | -34.3 % | -43.0 % (r15) | launch: launch 28 %; store_queue: queue (stg) 19 %; l2_return: queue (chip L2 share) 10 % | branch fallback 320, default latency 2560, issue-only 3200 | batch r15 |
| rms15_t256_warm b2b | 7 | us | 15.20 | 11.40 | -25.0 % | -35.0 % (r15) | store_queue: queue (stg) 27 %; l2_return: queue (chip L2 share) 14 %; l2/dram: load latency (incl. DRAM queue) 13 % | branch fallback 320, default latency 2560, issue-only 3200 | batch r15 |
| rms15_t256_warm graph | 7 | us | 11.95 | 9.26 | -22.5 % | -35.2 % (r15) | store_queue: queue (stg) 27 %; l2_return: queue (chip L2 share) 14 %; l2/dram: load latency (incl. DRAM queue) 13 % | branch fallback 320, default latency 2560, issue-only 3200 | batch r15 |
| rms15_t32_rotating kernel | 7 | us | 64.61 | 67.14 | +3.9 % | +1.7 % (r15) | l2/dram: load latency (incl. DRAM queue) 72 %; l1: hit latency 19 %; launch: launch 4 % | branch fallback 40, default latency 280, issue-only 360 | batch r15 |
| rms15_t32_warm kernel | 7 | us | 13.70 | 11.52 | -15.9 % | -17.2 % (r15) | store_queue: queue (stg) 30 %; launch: launch 25 %; l2_return: queue (chip L2 share) 14 % | branch fallback 40, default latency 280, issue-only 360 | batch r15 |
| rms15_t32_warm b2b | 7 | us | 12.73 | 11.46 | -9.9 % | -11.4 % (r15) | store_queue: queue (stg) 39 %; l2_return: queue (chip L2 share) 19 %; store_port: queue (stores) 13 % | branch fallback 40, default latency 280, issue-only 360 | batch r15 |
| rms15_t32_warm graph | 7 | us | 10.35 | 9.32 | -9.9 % | -11.8 % (r15) | store_queue: queue (stg) 39 %; l2_return: queue (chip L2 share) 19 %; store_port: queue (stores) 13 % | branch fallback 40, default latency 280, issue-only 360 | batch r15 |
| rms15_t64_rotating kernel | 7 | us | 62.91 | 67.99 | +8.1 % | +5.8 % (r15) | l2/dram: load latency (incl. DRAM queue) 80 %; l1: hit latency 10 %; launch: launch 4 % | branch fallback 80, default latency 640, issue-only 800 | batch r15 |
| rms15_t64_warm kernel | 7 | us | 12.83 | 11.43 | -10.9 % | -12.4 % (r15) | store_queue: queue (stg) 36 %; launch: launch 25 %; l2_return: queue (chip L2 share) 14 % | branch fallback 80, default latency 640, issue-only 800 | batch r15 |
| rms15_t64_warm b2b | 7 | us | 12.32 | 11.37 | -7.6 % | -9.2 % (r15) | store_queue: queue (stg) 48 %; l2_return: queue (chip L2 share) 19 %; store_port: drain after the last instruction 7 % | branch fallback 80, default latency 640, issue-only 800 | batch r15 |
| rms15_t64_warm graph | 7 | us | 10.50 | 9.23 | -12.1 % | -13.9 % (r15) | store_queue: queue (stg) 48 %; l2_return: queue (chip L2 share) 19 %; store_port: drain after the last instruction 7 % | branch fallback 80, default latency 640, issue-only 800 | batch r15 |
| sm15_t128_warm kernel | 7 | us | 12.70 | 10.45 | -17.7 % | +205.1 % (r15) | launch: launch 27 %; store_queue: queue (stg) 13 %; issue: scheduler (another warp issued) 12 % | branch fallback 2240, default latency 3200, default rbar 480, issue-only 7200 | batch r15 |
| sm15_t128_warm b2b | 7 | us | 12.31 | 10.39 | -15.6 % | +214.3 % (r15) | store_queue: queue (stg) 18 %; issue: scheduler (another warp issued) 17 %; l2_return: queue (chip L2 share) 8 % | branch fallback 2240, default latency 3200, default rbar 480, issue-only 7200 | batch r15 |
| sm15_t128_warm graph | 7 | us | 10.19 | 8.25 | -19.0 % | +258.8 % (r15) | store_queue: queue (stg) 18 %; issue: scheduler (another warp issued) 17 %; l2_return: queue (chip L2 share) 8 % | branch fallback 2240, default latency 3200, default rbar 480, issue-only 7200 | batch r15 |
| sm15_t16_warm kernel | 7 | us | 23.30 | 17.18 | -26.3 % | +80.8 % (r15) | fma: stall count after fp32 18 %; launch: launch 17 %; xu: xu result latency 14 % | branch fallback 4480, default latency 4680, default rbar 1440, issue-only 13600 | batch r15 |
| sm15_t16_warm b2b | 7 | us | 22.55 | 17.12 | -24.1 % | +86.4 % (r15) | fma: stall count after fp32 22 %; xu: xu result latency 17 %; issue: stall count after other 16 % | branch fallback 4480, default latency 4680, default rbar 1440, issue-only 13600 | batch r15 |
| sm15_t16_warm graph | 7 | us | 21.00 | 14.98 | -28.7 % | +90.0 % (r15) | fma: stall count after fp32 22 %; xu: xu result latency 17 %; issue: stall count after other 16 % | branch fallback 4480, default latency 4680, default rbar 1440, issue-only 13600 | batch r15 |
| sm15_t256_warm kernel | 7 | us | 16.74 | 13.52 | -19.2 % | +131.5 % (r15) | launch: launch 23 %; l2_return: queue (chip L2 share) 10 %; l2/dram: load latency (incl. DRAM queue) 8 % | branch fallback 2240, default latency 4160, issue-only 7680 | batch r15 |
| sm15_t256_warm b2b | 7 | us | 16.23 | 13.46 | -17.0 % | +138.3 % (r15) | l2_return: queue (chip L2 share) 13 %; l2/dram: load latency (incl. DRAM queue) 10 %; issue: stall count after barrier 9 % | branch fallback 2240, default latency 4160, issue-only 7680 | batch r15 |
| sm15_t256_warm graph | 7 | us | 13.32 | 11.32 | -15.0 % | +174.2 % (r15) | l2_return: queue (chip L2 share) 13 %; l2/dram: load latency (incl. DRAM queue) 10 %; issue: stall count after barrier 9 % | branch fallback 2240, default latency 4160, issue-only 7680 | batch r15 |
| sm15_t32_warm kernel | 7 | us | 15.74 | 12.32 | -21.8 % | +157.8 % (r15) | launch: launch 23 %; store_queue: queue (stg) 19 %; l2_return: queue (chip L2 share) 11 % | branch fallback 2240, default latency 2440, default rbar 480, issue-only 6800 | batch r15 |
| sm15_t32_warm b2b | 7 | us | 14.39 | 12.26 | -14.8 % | +181.7 % (r15) | store_queue: queue (stg) 25 %; l2_return: queue (chip L2 share) 14 %; fma: stall count after fp32 11 % | branch fallback 2240, default latency 2440, default rbar 480, issue-only 6800 | batch r15 |
| sm15_t32_warm graph | 7 | us | 11.67 | 10.12 | -13.3 % | +229.0 % (r15) | store_queue: queue (stg) 25 %; l2_return: queue (chip L2 share) 14 %; fma: stall count after fp32 11 % | branch fallback 2240, default latency 2440, default rbar 480, issue-only 6800 | batch r15 |
| sm15_t64_warm kernel | 7 | us | 14.75 | 11.01 | -25.4 % | +166.5 % (r15) | launch: launch 26 %; store_queue: queue (stg) 14 %; issue: scheduler (another warp issued) 12 % | branch fallback 2240, default latency 2720, default rbar 400, issue-only 6960 | batch r15 |
| sm15_t64_warm b2b | 7 | us | 14.36 | 10.95 | -23.7 % | +173.4 % (r15) | store_queue: queue (stg) 19 %; issue: scheduler (another warp issued) 16 %; fma: stall count after fp32 10 % | branch fallback 2240, default latency 2720, default rbar 400, issue-only 6960 | batch r15 |
| sm15_t64_warm graph | 7 | us | 11.89 | 8.81 | -25.9 % | +212.1 % (r15) | store_queue: queue (stg) 19 %; issue: scheduler (another warp issued) 16 %; fma: stall count after fp32 10 % | branch fallback 2240, default latency 2720, default rbar 400, issue-only 6960 | batch r15 |
| wl_m1920 kernel | 7 | us | 14.82 | 10.45 | -29.5 % | +161.6 % (r15b) | launch: launch 27 %; store_queue: queue (stg) 13 %; issue: scheduler (another warp issued) 12 % | branch fallback 2240, default latency 3200, default rbar 480, issue-only 7200 | batch r15b |
| wl_m1960 kernel | 7 | us | 13.82 | 10.60 | -23.3 % | +185.7 % (r15b) | launch: launch 27 %; store_queue: queue (stg) 13 %; issue: scheduler (another warp issued) 10 % | branch fallback 2296, default latency 3280, default rbar 492, issue-only 7380 | batch r15b |
| wl_m2000 kernel | 7 | us | 14.78 | 10.76 | -27.2 % | +172.1 % (r15b) | launch: launch 27 %; l2_return: queue (chip L2 share) 12 %; store_queue: queue (stg) 10 % | branch fallback 2352, default latency 3360, default rbar 504, issue-only 7560 | batch r15b |
