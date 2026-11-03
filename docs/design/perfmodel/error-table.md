# Error table

Produced by `scripts/perfmodel-table` from KTA commit `fb66f5370b472426ccd53d9fe3bf7fa7741a9780` on 2026-11-03T17:34:46Z; upstream inputs `/Users/dingyi/Projects/ai-infra-codesign-ventus` at `a628c9d963cf9c029fa596b872d6e8920ca883e8`. CPU only, 0 s. Error = predicted / measured − 1 (positive: the model is slow). Machine-readable copy: `error-table.json`.

## Summary

| Set | n | excluded | median abs error | worst | within 10 % | Spearman |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gemm1024 full | 22 | 0 | 5.3 % | -15.5 % | 21 | 0.83 |
| gemm1024 nocompute | 22 | 0 | 6.7 % | -23.7 % | 15 |  |
| gemm1024 noload 1 block | 22 | 0 | 3.4 % | +17.1 % | 20 |  |
| tma full | 7 | 0 | 11.1 % | +36.7 % | 3 | 0.96 |
| tma noload 1 block | 7 | 0 | 4.4 % | +15.4 % | 5 |  |
| probe sched.cu | 2 | 0 | 3.8 % | -6.3 % | 2 |  |
| probe alu.cu | 39 | 9 | 0.5 % | -11.8 % | 38 |  |
| probe alumix.cu | 13 | 0 | 0.2 % | -11.0 % | 12 |  |
| probe sfu.cu | 36 | 6 | 0.3 % | -10.0 % | 35 |  |
| probe cvt.cu | 7 | 0 | 0.8 % | +100.4 % | 6 |  |

## Top 15 rows by absolute error

| Row | Level | Unit | Measured | Predicted | Error | Attribution (critical path) | Defaults | Note |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| cvt int to fp32 | 2 | cycles/element/smsp | 4.03 | 8.08 | +100.4 % |  |  |  |
| tma m128 full | 7 | us | 162.11 | 221.59 | +36.7 % |  |  | measured warm in a session where part of B stayed in the L2; upstream compares the decode picks cold against a later flush measurement (gb10 design note §7) |
| gemm1024 c128x256x64_st2_th256 nocompute | 7 | us | 27.71 | 21.14 | -23.7 % |  |  |  |
| tma m32 full | 7 | us | 123.97 | 148.76 | +20.0 % |  |  | measured warm in a session where part of B stayed in the L2; upstream compares the decode picks cold against a later flush measurement (gb10 design note §7) |
| gemm1024 c256x128x64_st2_th256 nocompute | 7 | us | 25.92 | 21.14 | -18.4 % |  |  |  |
| gemm1024 c64x64x32_st3_th256 noload 1 block | 7 | us | 3.78 | 4.42 | +17.1 % |  |  |  |
| gemm1024 c128x256x32_st3_th256 nocompute | 7 | us | 25.09 | 21.14 | -15.7 % |  |  |  |
| gemm1024 c128x128x32_st3_th256 full | 7 | us | 41.38 | 34.94 | -15.5 % |  |  |  |
| tma k1024 noload 1 block | 7 | us | 25.02 | 28.87 | +15.4 % |  |  |  |
| gemm1024 c64x128x64_st3_th128 nocompute | 7 | us | 38.98 | 33.22 | -14.8 % |  |  |  |
| gemm1024 c256x128x32_st3_th256 nocompute | 7 | us | 24.74 | 21.16 | -14.5 % |  |  |  |
| gemm1024 c128x128x32_st3_th128 nocompute | 7 | us | 29.12 | 24.98 | -14.2 % |  |  |  |
| gemm1024 c64x128x64_st3_th256 nocompute | 7 | us | 38.34 | 33.25 | -13.3 % |  |  |  |
| alu F2F.F16.F32+HADD2.F32 w32 | 2 | cycles | 102,963.08 | 90,776.00 | -11.8 % |  |  |  |
| tma 1024 full | 7 | us | 45.12 | 40.07 | -11.2 % |  |  |  |

## All rows

| Row | Level | Unit | Measured | Predicted | Error | Attribution (critical path) | Defaults | Note |
| --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- |
| gemm1024 c128x128x32_st3_th128 full | 7 | us | 39.46 | 36.07 | -8.6 % |  |  |  |
| gemm1024 c128x128x32_st3_th128 nocompute | 7 | us | 29.12 | 24.98 | -14.2 % |  |  |  |
| gemm1024 c128x128x32_st3_th128 noload 1 block | 7 | us | 17.54 | 16.99 | -3.1 % |  |  |  |
| gemm1024 c128x128x32_st3_th256 full | 7 | us | 41.38 | 34.94 | -15.5 % |  |  |  |
| gemm1024 c128x128x32_st3_th256 nocompute | 7 | us | 27.14 | 25.00 | -7.9 % |  |  |  |
| gemm1024 c128x128x32_st3_th256 noload 1 block | 7 | us | 15.49 | 16.06 | +3.7 % |  |  |  |
| gemm1024 c128x128x64_st3_th128 full | 7 | us | 46.85 | 42.50 | -9.3 % |  |  |  |
| gemm1024 c128x128x64_st3_th128 nocompute | 7 | us | 30.34 | 29.34 | -3.3 % |  |  |  |
| gemm1024 c128x128x64_st3_th128 noload 1 block | 7 | us | 16.26 | 15.61 | -3.9 % |  |  |  |
| gemm1024 c128x128x64_st3_th256 full | 7 | us | 44.45 | 42.43 | -4.5 % |  |  |  |
| gemm1024 c128x128x64_st3_th256 nocompute | 7 | us | 30.34 | 29.40 | -3.1 % |  |  |  |
| gemm1024 c128x128x64_st3_th256 noload 1 block | 7 | us | 15.30 | 15.78 | +3.1 % |  |  |  |
| gemm1024 c128x256x32_st3_th256 full | 7 | us | 43.58 | 41.23 | -5.4 % |  |  |  |
| gemm1024 c128x256x32_st3_th256 nocompute | 7 | us | 25.09 | 21.14 | -15.7 % |  |  |  |
| gemm1024 c128x256x32_st3_th256 noload 1 block | 7 | us | 32.48 | 32.39 | -0.3 % |  |  |  |
| gemm1024 c128x256x64_st2_th256 full | 7 | us | 41.28 | 38.95 | -5.6 % |  |  |  |
| gemm1024 c128x256x64_st2_th256 nocompute | 7 | us | 27.71 | 21.14 | -23.7 % |  |  |  |
| gemm1024 c128x256x64_st2_th256 noload 1 block | 7 | us | 30.30 | 30.26 | -0.2 % |  |  |  |
| gemm1024 c128x64x32_st3_th128 full | 7 | us | 37.41 | 37.40 | -0.0 % |  |  |  |
| gemm1024 c128x64x32_st3_th128 nocompute | 7 | us | 33.09 | 32.71 | -1.1 % |  |  |  |
| gemm1024 c128x64x32_st3_th128 noload 1 block | 7 | us | 8.35 | 8.86 | +6.1 % |  |  |  |
| gemm1024 c128x64x32_st3_th256 full | 7 | us | 37.09 | 36.13 | -2.6 % |  |  |  |
| gemm1024 c128x64x32_st3_th256 nocompute | 7 | us | 32.93 | 32.58 | -1.1 % |  |  |  |
| gemm1024 c128x64x32_st3_th256 noload 1 block | 7 | us | 8.64 | 8.98 | +3.9 % |  |  |  |
| gemm1024 c256x128x32_st3_th256 full | 7 | us | 43.26 | 41.36 | -4.4 % |  |  |  |
| gemm1024 c256x128x32_st3_th256 nocompute | 7 | us | 24.74 | 21.16 | -14.5 % |  |  |  |
| gemm1024 c256x128x32_st3_th256 noload 1 block | 7 | us | 30.02 | 30.57 | +1.9 % |  |  |  |
| gemm1024 c256x128x64_st2_th256 full | 7 | us | 41.31 | 37.95 | -8.1 % |  |  |  |
| gemm1024 c256x128x64_st2_th256 nocompute | 7 | us | 25.92 | 21.14 | -18.4 % |  |  |  |
| gemm1024 c256x128x64_st2_th256 noload 1 block | 7 | us | 30.27 | 30.51 | +0.8 % |  |  |  |
| gemm1024 c256x64x32_st3_th128 full | 7 | us | 47.42 | 48.42 | +2.1 % |  |  |  |
| gemm1024 c256x64x32_st3_th128 nocompute | 7 | us | 33.15 | 34.09 | +2.8 % |  |  |  |
| gemm1024 c256x64x32_st3_th128 noload 1 block | 7 | us | 16.03 | 16.31 | +1.7 % |  |  |  |
| gemm1024 c256x64x32_st3_th256 full | 7 | us | 48.26 | 44.98 | -6.8 % |  |  |  |
| gemm1024 c256x64x32_st3_th256 nocompute | 7 | us | 33.38 | 34.13 | +2.2 % |  |  |  |
| gemm1024 c256x64x32_st3_th256 noload 1 block | 7 | us | 15.68 | 16.30 | +3.9 % |  |  |  |
| gemm1024 c64x128x32_st3_th128 full | 7 | us | 39.26 | 37.26 | -5.1 % |  |  |  |
| gemm1024 c64x128x32_st3_th128 nocompute | 7 | us | 35.20 | 32.68 | -7.1 % |  |  |  |
| gemm1024 c64x128x32_st3_th128 noload 1 block | 7 | us | 9.22 | 9.49 | +3.0 % |  |  |  |
| gemm1024 c64x128x32_st3_th256 full | 7 | us | 35.01 | 34.68 | -0.9 % |  |  |  |
| gemm1024 c64x128x32_st3_th256 nocompute | 7 | us | 33.89 | 32.81 | -3.2 % |  |  |  |
| gemm1024 c64x128x32_st3_th256 noload 1 block | 7 | us | 8.29 | 8.80 | +6.2 % |  |  |  |
| gemm1024 c64x128x64_st3_th128 full | 7 | us | 40.32 | 37.45 | -7.1 % |  |  |  |
| gemm1024 c64x128x64_st3_th128 nocompute | 7 | us | 38.98 | 33.22 | -14.8 % |  |  |  |
| gemm1024 c64x128x64_st3_th128 noload 1 block | 7 | us | 8.38 | 8.54 | +1.9 % |  |  |  |
| gemm1024 c64x128x64_st3_th256 full | 7 | us | 39.26 | 37.21 | -5.2 % |  |  |  |
| gemm1024 c64x128x64_st3_th256 nocompute | 7 | us | 38.34 | 33.25 | -13.3 % |  |  |  |
| gemm1024 c64x128x64_st3_th256 noload 1 block | 7 | us | 8.00 | 8.39 | +4.8 % |  |  |  |
| gemm1024 c64x256x32_st3_th128 full | 7 | us | 49.63 | 47.29 | -4.7 % |  |  |  |
| gemm1024 c64x256x32_st3_th128 nocompute | 7 | us | 36.29 | 34.10 | -6.0 % |  |  |  |
| gemm1024 c64x256x32_st3_th128 noload 1 block | 7 | us | 17.54 | 17.33 | -1.2 % |  |  |  |
| gemm1024 c64x256x32_st3_th256 full | 7 | us | 51.71 | 47.83 | -7.5 % |  |  |  |
| gemm1024 c64x256x32_st3_th256 nocompute | 7 | us | 36.54 | 34.10 | -6.7 % |  |  |  |
| gemm1024 c64x256x32_st3_th256 noload 1 block | 7 | us | 17.18 | 17.04 | -0.9 % |  |  |  |
| gemm1024 c64x64x128_st3_th128 full | 7 | us | 50.37 | 46.97 | -6.7 % |  |  |  |
| gemm1024 c64x64x128_st3_th128 nocompute | 7 | us | 42.40 | 42.94 | +1.3 % |  |  |  |
| gemm1024 c64x64x128_st3_th128 noload 1 block | 7 | us | 3.94 | 4.21 | +7.0 % |  |  |  |
| gemm1024 c64x64x128_st3_th256 full | 7 | us | 50.46 | 46.71 | -7.4 % |  |  |  |
| gemm1024 c64x64x128_st3_th256 nocompute | 7 | us | 42.46 | 43.00 | +1.3 % |  |  |  |
| gemm1024 c64x64x128_st3_th256 noload 1 block | 7 | us | 3.94 | 4.23 | +7.4 % |  |  |  |
| gemm1024 c64x64x32_st3_th128 full | 7 | us | 38.88 | 40.04 | +3.0 % |  |  |  |
| gemm1024 c64x64x32_st3_th128 nocompute | 7 | us | 37.22 | 40.02 | +7.5 % |  |  |  |
| gemm1024 c64x64x32_st3_th128 noload 1 block | 7 | us | 4.51 | 4.97 | +10.2 % |  |  |  |
| gemm1024 c64x64x32_st3_th256 full | 7 | us | 39.20 | 39.75 | +1.4 % |  |  |  |
| gemm1024 c64x64x32_st3_th256 nocompute | 7 | us | 37.34 | 39.84 | +6.7 % |  |  |  |
| gemm1024 c64x64x32_st3_th256 noload 1 block | 7 | us | 3.78 | 4.42 | +17.1 % |  |  |  |
| tma 1024 full | 7 | us | 45.12 | 40.07 | -11.2 % |  |  |  |
| tma 1024 noload 1 block | 7 | us | 7.58 | 8.41 | +10.9 % |  |  |  |
| tma 4096 full | 7 | us | 1,510.56 | 1,568.87 | +3.9 % |  |  |  |
| tma 4096 noload 1 block | 7 | us | 108.61 | 111.27 | +2.5 % |  |  |  |
| tma m32 full | 7 | us | 123.97 | 148.76 | +20.0 % |  |  | measured warm in a session where part of B stayed in the L2; upstream compares the decode picks cold against a later flush measurement (gb10 design note §7) |
| tma m32 noload 1 block | 7 | us | 15.62 | 16.31 | +4.4 % |  |  |  |
| tma 2048 full | 7 | us | 203.30 | 190.95 | -6.1 % |  |  |  |
| tma 2048 noload 1 block | 7 | us | 52.54 | 55.77 | +6.1 % |  |  |  |
| tma 8192 full | 7 | us | 15,136.06 | 15,813.60 | +4.5 % |  |  |  |
| tma 8192 noload 1 block | 7 | us | 211.33 | 211.48 | +0.1 % |  |  |  |
| tma k1024 full | 7 | us | 448.99 | 398.97 | -11.1 % |  |  |  |
| tma k1024 noload 1 block | 7 | us | 25.02 | 28.87 | +15.4 % |  |  |  |
| tma m128 full | 7 | us | 162.11 | 221.59 | +36.7 % |  |  | measured warm in a session where part of B stayed in the L2; upstream compares the decode picks cold against a later flush measurement (gb10 design note §7) |
| tma m128 noload 1 block | 7 | us | 60.58 | 60.24 | -0.6 % |  |  |  |
| sched adds alone | 1 | cycles/add | 5.67 | 5.31 | -6.3 % |  |  |  |
| sched adds beside FFMA | 3 | cycles/add | 15.10 | 14.91 | -1.2 % |  |  |  |
| alu FFMA w4 | 2 | cycles | 4,616.51 | 4,513.00 | -2.2 % |  |  |  |
| alu FFMA w8 | 2 | cycles | 8,526.67 | 8,683.00 | +1.8 % |  |  |  |
| alu FFMA w16 | 2 | cycles | 16,868.98 | 17,118.00 | +1.5 % |  |  |  |
| alu FFMA w32 | 2 | cycles | 33,947.68 | 34,056.00 | +0.3 % |  |  |  |
| alu IMAD w4 | 2 | cycles | 8,346.41 | 8,305.00 | -0.5 % |  |  |  |
| alu IMAD w8 | 2 | cycles | 16,425.06 | 16,432.00 | +0.0 % |  |  |  |
| alu IMAD w16 | 2 | cycles | 32,800.80 | 32,836.00 | +0.1 % |  |  |  |
| alu IMAD w32 | 2 | cycles | 65,601.60 | 65,636.00 | +0.1 % |  |  |  |
| alu IADD3 w4 | 2 | cycles | 86.00 | 93.00 | +8.1 % |  |  | excluded: loop folded by the compiler (no loop in the SASS) |
| alu IADD3 w8 | 2 | cycles | 107.00 | 112.00 | +4.7 % |  |  | excluded: loop folded by the compiler (no loop in the SASS) |
| alu IADD3 w16 | 2 | cycles | 187.00 | 127.00 | -32.1 % |  |  | excluded: loop folded by the compiler (no loop in the SASS) |
| alu IADD3 w32 | 2 | cycles | 350.00 | 175.00 | -50.0 % |  |  | excluded: loop folded by the compiler (no loop in the SASS) |
| alu LOP3 w4 | 2 | cycles | 13,451.56 | 12,961.00 | -3.6 % |  |  |  |
| alu LOP3 w8 | 2 | cycles | 24,824.24 | 24,624.00 | -0.8 % |  |  |  |
| alu LOP3 w16 | 2 | cycles | 49,164.29 | 49,220.00 | +0.1 % |  |  |  |
| alu LOP3 w32 | 2 | cycles | 98,402.40 | 98,404.00 | +0.0 % |  |  |  |
| alu SHF w4 | 2 | cycles | 8,393.44 | 8,265.00 | -1.5 % |  |  |  |
| alu SHF w8 | 2 | cycles | 16,441.55 | 16,432.00 | -0.1 % |  |  |  |
| alu SHF w16 | 2 | cycles | 32,751.62 | 32,836.00 | +0.3 % |  |  |  |
| alu SHF w32 | 2 | cycles | 65,568.78 | 65,636.00 | +0.1 % |  |  |  |
| alu PRMT w4 | 2 | cycles | 8,489.12 | 8,385.00 | -1.2 % |  |  |  |
| alu PRMT w8 | 2 | cycles | 16,474.61 | 16,432.00 | -0.3 % |  |  |  |
| alu PRMT w16 | 2 | cycles | 32,833.67 | 32,836.00 | +0.0 % |  |  |  |
| alu PRMT w32 | 2 | cycles | 65,339.98 | 65,636.00 | +0.5 % |  |  |  |
| alu HFMA2 w4 | 2 | cycles | 8,397.74 | 8,265.00 | -1.6 % |  |  |  |
| alu HFMA2 w8 | 2 | cycles | 16,408.61 | 16,432.00 | +0.1 % |  |  |  |
| alu HFMA2 w16 | 2 | cycles | 32,751.62 | 32,836.00 | +0.3 % |  |  |  |
| alu HFMA2 w32 | 2 | cycles | 65,437.84 | 65,636.00 | +0.3 % |  |  |  |
| alu F2F.F16.F32+HADD2.F32 w4 | 2 | cycles | 15,922.25 | 15,041.00 | -5.5 % |  |  |  |
| alu F2F.F16.F32+HADD2.F32 w8 | 2 | cycles | 26,489.89 | 24,487.00 | -7.6 % |  |  |  |
| alu F2F.F16.F32+HADD2.F32 w16 | 2 | cycles | 51,481.54 | 46,584.00 | -9.5 % |  |  |  |
| alu F2F.F16.F32+HADD2.F32 w32 | 2 | cycles | 102,963.08 | 90,776.00 | -11.8 % |  |  |  |
| alu HADD2.F32(h2f) w4 | 2 | cycles | 40,655.09 | 40,353.00 | -0.7 % |  |  |  |
| alu HADD2.F32(h2f) w8 | 2 | cycles | 66,064.52 | 66,237.00 | +0.3 % |  |  |  |
| alu HADD2.F32(h2f) w16 | 2 | cycles | 131,598.39 | 132,056.00 | +0.3 % |  |  |  |
| alu HADD2.F32(h2f) w32 | 2 | cycles | 260,580.52 | 263,701.00 | +1.2 % |  |  |  |
| alu F2FP.PACK_AB w4 | 2 | cycles | 14,641.64 | 14,465.00 | -1.2 % |  |  |  |
| alu F2FP.PACK_AB w8 | 2 | cycles | 23,076.06 | 23,601.00 | +2.3 % |  |  |  |
| alu F2FP.PACK_AB w16 | 2 | cycles | 45,134.99 | 45,377.00 | +0.5 % |  |  |  |
| alu F2FP.PACK_AB w32 | 2 | cycles | 89,898.49 | 87,907.00 | -2.2 % |  |  |  |
| alu I2F w4 | 2 | cycles | 311.00 | 318.00 | +2.3 % |  |  | excluded: loop folded by the compiler (no loop in the SASS) |
| alu I2F w8 | 2 | cycles | 343.00 | 347.00 | +1.2 % |  |  | excluded: loop folded by the compiler (no loop in the SASS) |
| alu I2F w16 | 2 | cycles | 533.00 | 348.00 | -34.7 % |  |  | excluded: loop folded by the compiler (no loop in the SASS) |
| alu I2F w32 | 2 | cycles | 1,042.00 | 394.00 | -62.2 % |  |  | excluded: loop folded by the compiler (no loop in the SASS) |
| alu F2I w4 | 2 | cycles | 4,329.81 | 4,257.00 | -1.7 % |  |  |  |
| alu F2I w8 | 2 | cycles | 8,302.00 | 8,336.00 | +0.4 % |  |  |  |
| alu F2I w16 | 2 | cycles | 8,300.95 | 16,646.00 | +100.5 % |  |  | excluded: measured rate 7.895 exceeds the issue limit 4.0/clk/SM |
| alu F2I w32 | 2 | cycles | 33,250.13 | 33,258.00 | +0.0 % |  |  |  |
| alumix IMAD+LOP3 | 3 | cycles | 49,419.00 | 49,624.00 | +0.4 % |  |  |  |
| alumix IMAD+FFMA | 3 | cycles | 49,601.00 | 49,508.00 | -0.2 % |  |  |  |
| alumix LOP3+FFMA | 3 | cycles | 64,070.00 | 64,108.00 | +0.1 % |  |  |  |
| alumix HFMA2+LOP3 | 3 | cycles | 51,505.00 | 49,247.00 | -4.4 % |  |  |  |
| alumix HFMA2+IMAD | 3 | cycles | 65,069.00 | 65,631.00 | +0.9 % |  |  |  |
| alumix HFMA2+FFMA | 3 | cycles | 55,908.00 | 49,764.00 | -11.0 % |  |  |  |
| alumix PRMT+IMAD | 3 | cycles | 33,480.00 | 33,467.00 | -0.0 % |  |  |  |
| alumix PRMT+LOP3 | 3 | cycles | 81,985.00 | 82,020.00 | +0.0 % |  |  |  |
| alumix HADD2.F32+FADD+IMAD | 3 | cycles | 64,977.00 | 65,925.00 | +1.5 % |  |  |  |
| alumix HADD2.F32+FADD+LOP3 | 3 | cycles | 76,227.00 | 81,407.00 | +6.8 % |  |  |  |
| alumix IMAD+IMAD | 3 | cycles | 65,596.00 | 65,636.00 | +0.1 % |  |  |  |
| alumix LOP3+LOP3 | 3 | cycles | 98,396.00 | 98,404.00 | +0.0 % |  |  |  |
| alumix FFMA+FFMA | 3 | cycles | 34,117.00 | 34,056.00 | -0.2 % |  |  |  |
| sfu ex2.f32 latency | 1 | cycles/op | 18.20 | 18.04 | -0.9 % |  |  |  |
| sfu ex2.f32 w1 | 2 | cycles | 262,144.00 | 264,779.00 | +1.0 % |  |  |  |
| sfu ex2.f32 w4 | 2 | cycles | 265,462.28 | 264,785.00 | -0.3 % |  |  |  |
| sfu ex2.f32 w8 | 2 | cycles | 524,288.00 | 524,396.00 | +0.0 % |  |  |  |
| sfu ex2.f32 w16 | 2 | cycles | 1,048,576 | 1,048,750 | +0.0 % |  |  |  |
| sfu ex2.f32 w32 | 2 | cycles | 2,097,152 | 2,097,684 | +0.0 % |  |  |  |
| sfu rcp.f32 latency | 1 | cycles/op | 0.10 | 0.02 | -81.7 % |  |  | excluded: withdrawn upstream: the rcp loop was folded away (gb10 design note §9.4) |
| sfu rcp.f32 w1 | 2 | cycles | 201.00 | 69.00 | -65.7 % |  |  | excluded: withdrawn upstream: the rcp loop was folded away (gb10 design note §9.4) |
| sfu rcp.f32 w4 | 2 | cycles | 746.00 | 75.00 | -89.9 % |  |  | excluded: withdrawn upstream: the rcp loop was folded away (gb10 design note §9.4) |
| sfu rcp.f32 w8 | 2 | cycles | 505.00 | 133.00 | -73.7 % |  |  | excluded: withdrawn upstream: the rcp loop was folded away (gb10 design note §9.4) |
| sfu rcp.f32 w16 | 2 | cycles | 455.00 | 213.00 | -53.2 % |  |  | excluded: withdrawn upstream: the rcp loop was folded away (gb10 design note §9.4) |
| sfu rcp.f32 w32 | 2 | cycles | 381.00 | 560.00 | +47.0 % |  |  | excluded: withdrawn upstream: the rcp loop was folded away (gb10 design note §9.4) |
| sfu rsqrt.f32 latency | 1 | cycles/op | 18.10 | 18.04 | -0.3 % |  |  |  |
| sfu rsqrt.f32 w1 | 2 | cycles | 262,144.00 | 264,779.00 | +1.0 % |  |  |  |
| sfu rsqrt.f32 w4 | 2 | cycles | 265,462.28 | 264,785.00 | -0.3 % |  |  |  |
| sfu rsqrt.f32 w8 | 2 | cycles | 524,288.00 | 524,396.00 | +0.0 % |  |  |  |
| sfu rsqrt.f32 w16 | 2 | cycles | 1,048,576 | 1,048,750 | +0.0 % |  |  |  |
| sfu rsqrt.f32 w32 | 2 | cycles | 2,097,152 | 2,097,684 | +0.0 % |  |  |  |
| sfu sin.f32 latency | 1 | cycles/op | 22.20 | 22.16 | -0.2 % |  |  |  |
| sfu sin.f32 w1 | 2 | cycles | 268,865.64 | 266,827.00 | -0.8 % |  |  |  |
| sfu sin.f32 w4 | 2 | cycles | 267,153.12 | 266,833.00 | -0.1 % |  |  |  |
| sfu sin.f32 w8 | 2 | cycles | 524,288.00 | 525,461.00 | +0.2 % |  |  |  |
| sfu sin.f32 w16 | 2 | cycles | 1,048,576 | 1,050,870 | +0.2 % |  |  |  |
| sfu sin.f32 w32 | 2 | cycles | 2,097,152 | 2,102,913 | +0.3 % |  |  |  |
| sfu lg2.f32 latency | 1 | cycles/op | 18.10 | 18.04 | -0.3 % |  |  |  |
| sfu lg2.f32 w1 | 2 | cycles | 262,144.00 | 264,779.00 | +1.0 % |  |  |  |
| sfu lg2.f32 w4 | 2 | cycles | 265,462.28 | 264,785.00 | -0.3 % |  |  |  |
| sfu lg2.f32 w8 | 2 | cycles | 524,288.00 | 524,396.00 | +0.0 % |  |  |  |
| sfu lg2.f32 w16 | 2 | cycles | 1,048,576 | 1,048,750 | +0.0 % |  |  |  |
| sfu lg2.f32 w32 | 2 | cycles | 2,097,152 | 2,097,684 | +0.0 % |  |  |  |
| sfu tanh.f32 latency | 1 | cycles/op | 18.10 | 18.04 | -0.3 % |  |  |  |
| sfu tanh.f32 w1 | 2 | cycles | 262,144.00 | 264,779.00 | +1.0 % |  |  |  |
| sfu tanh.f32 w4 | 2 | cycles | 265,462.28 | 264,785.00 | -0.3 % |  |  |  |
| sfu tanh.f32 w8 | 2 | cycles | 524,288.00 | 524,396.00 | +0.0 % |  |  |  |
| sfu tanh.f32 w16 | 2 | cycles | 1,048,576 | 1,048,750 | +0.0 % |  |  |  |
| sfu tanh.f32 w32 | 2 | cycles | 2,097,152 | 2,097,684 | +0.0 % |  |  |  |
| sfu ffma.f32 latency | 1 | cycles/op | 4.70 | 4.23 | -10.0 % |  |  |  |
| sfu ffma.f32 w1 | 2 | cycles | 37,183.55 | 35,927.00 | -3.4 % |  |  |  |
| sfu ffma.f32 w4 | 2 | cycles | 37,449.14 | 35,933.00 | -4.0 % |  |  |  |
| sfu ffma.f32 w8 | 2 | cycles | 68,871.99 | 69,263.00 | +0.6 % |  |  |  |
| sfu ffma.f32 w16 | 2 | cycles | 138,654.68 | 136,641.00 | -1.5 % |  |  |  |
| sfu ffma.f32 w32 | 2 | cycles | 271,475.99 | 272,064.00 | +0.2 % |  |  |  |
| cvt scaffold only (mov) | 2 | cycles/element/smsp | 4.04 | 4.01 | -0.8 % |  |  |  |
| cvt I2F.F16 from s32 | 2 | cycles/element/smsp | 8.06 | 8.02 | -0.4 % |  |  |  |
| cvt I2F.F16 from u32 | 2 | cycles/element/smsp | 8.05 | 8.02 | -0.3 % |  |  |  |
| cvt int to fp32 | 2 | cycles/element/smsp | 4.03 | 8.08 | +100.4 % |  |  |  |
| cvt fp32 to fp16 | 2 | cycles/element/smsp | 8.04 | 8.01 | -0.4 % |  |  |  |
| cvt MUFU.EX2 ruler | 2 | cycles/element/smsp | 8.23 | 8.07 | -1.9 % |  |  |  |
| cvt I2F.F16 + MUFU.EX2 alternating | 3 | cycles/element/smsp | 8.13 | 8.01 | -1.5 % |  |  |  |
