# Performance model: state

The single state file of the GB10 programme (28 §7). Updated 2026-11-04 at the end of P0/P1. Table: [error-table.md](error-table.md) (KTA commit `8631cdd`, upstream inputs `a628c9d`). Inventory: [29](../29-mechanism-inventory.md). Nothing here was measured on the GPU; no parameter was tuned.

## Where things stand

| Item | State |
| --- | --- |
| Baseline replay | Reproduces upstream. 22 cp.async GEMMs at 1024³, whole kernel: median 5.3 %, worst −15.5 %, 21/22 within 10 %, Spearman 0.835. This is upstream's code at `0eef497` rerun bit for bit. Upstream's saved `sim_kernel_level.json` (median 5.3 %, worst 15.6 %, Spearman 0.857) comes from the earlier commit `df99b05`; `8ddfa04` changed the engine after it was written. The 7 TMA picks equal upstream `runs/gb10/ws_validate.txt` |
| Engine | Builds from `machine.yaml` + `gb10.json`; all 192 rows were bit-identical before and after the move. One upstream mechanism ported since then: I2FP on the ALU (D746), which changes only probe rows |
| Analysis | Critical-path attribution (covers 100 % of the predicted cycles on every row), what-if (`scripts/perfmodel-whatif`), assumption ledger per row |
| Table runtime | 73 s cold on 16 processes, under 1 s when cached. The slowest row is 58–61 s (cvt.cu with tracing). `tma 8192 full` is not run by default (D749, user request): 150 s untraced, 300 s traced |

## Error table: top rows

Rows with a measured value, by absolute error. Attribution is the share of the predicted time on the critical path.

| Row | Error | Attribution (top) | Assumed/fitted on path |
| --- | ---: | --- | --- |
| tma m128 full | +36.7 % | TMA transfer / DRAM queue 45 %, stall after mma 19 %, TMA L2 latency 10 %, TMA engine queue 7 % | launch 1 % |
| gemm1024 c128x256x64_st2_th256 nocompute | −23.7 % | chip-L2 share queue (cp.async return) 46 %, store queue 28 %, launch 12 %, L2 latency 5 %, LSU queue 5 % | launch (fitted) 12 % |
| tma m32 full | +20.0 % | TMA transfer / DRAM queue 84 %, TMA engine queue 6 % | launch 2 % |
| gemm1024 c256x128x64_st2_th256 nocompute | −18.4 % | chip-L2 share queue 46 %, store queue 28 %, launch 12 % | launch 12 % |
| gemm1024 c64x64x32_st3_th256 noload 1 block | +17.1 % | tensor queue 21 %, stall after barrier 20 %, stall after mma 13 %, stall after ldsm 11 %, MIO queue (ldsm) 10 % | default latency 0.4 % |

The two decode rows (m128, m32) compare a warm prediction with a session in which part of B stayed in the L2. Upstream measured them again after a flush (146 and 203 µs) and got +1 % / +7 % cold. These two rows are therefore not mechanism evidence until they are re-measured under a declared cache state (28 §4.3). That is a measurement task, not a probe. The first mechanism row is the loads-only 128×256×64 kernel.

## Open questions

1. **Two clocks.** `validate` converts at 2.52 GHz (`clock_GHz`); `kernel`, `ws` and `ws_validate` at 2.54 (`clock_under_load_GHz`). The description names both (`clock:` in machine.yaml). The measurement protocol (28 §4.2) should settle a single effective clock.
2. **Launch overhead is fitted and sits on the path.** `launch_stream_us` 2.6 is 12 % of the small loads-only rows.
3. **Decode rows' cache state** (above).
4. **Silent defaults that should name a missing primitive** (27 §2.3). A TMA kernel without warp specialisation gets the default latency for `UTMALDG`. `BAR.ARV`/`BAR.RED` are treated as blocking. No test-set kernel triggers either yet.
5. **Issue-only and default-latency instructions.** Every row uses them: thousands of issue-only instructions (S2R, MOV, uniform ops, …) and default-latency instructions (LDC, S2R, …). Their share of the critical path is below 1 % on every GEMM row, so R2 does not let them be refined now. They will matter for FA and softmax (inventory §3).
6. **Excluded probe rows**, kept in the table with their reasons: alu IADD3 and I2F (loop folded), F2I at 16 warps (measured 7.9 per clock, above the issue limit), sfu rcp (withdrawn upstream).
7. Not done in this step: the 8192³ row (by request), any GPU measurement, the program-representation and machine-description *schemas* of P0 as separate documents (machine.yaml is the instance; no schema file yet).

## Frozen parameters

All of `gb10.json` is unchanged from upstream `0eef497` apart from the WP26 additions; nothing was tuned. Values marked `fitted` that the table depends on: `launch_stream_us`, `tma_r_GBs_alone`, `tma_l2_aggregate_GBs`, `block_setup_us`. Values marked `assumed`: `sim.default_var_latency_clk` 20, `sim.rbar_release_clk` 8.

## Residual ledger

Empty. No residual term exists.

## Proposed next probes (written per `write-probe`, not run)

### P-1 C-store path at partial and full grids (for the top mechanism row)

1. **Mechanism**: store port shared by a TPC, and the chip store limit (inventory §6, rows 1–2: 15 cycles per 4-byte-lane STG per pair, chip 137.8 B/clk; "partner busy whenever the share is ≥ 2" is a structural assumption).
2. **Why now**: the store queue holds 28 % of the critical path of the two worst loads-only rows (−23.7 %, −18.4 %). A what-if that halves the chip store rate moves that row +27.5 %. Upstream's own ablation records already show the gap. For 128×256×64, the store-only variant minus its no-store twin (`kt0` − `kt0_nostore`) is 8.2 µs at 32 blocks against about 6.0 µs from the model's rule. At 1 block it is 4.1 µs against 3.0. Store width and tile size are search-space variables (staged vs direct C store); the store-port sharing is a hardware design variable.
3. **Competing hypotheses**, for a probe that stores a 64 KB tile per block with 4-byte lanes (512 STG per block), blocks ∈ {1, 2, 8, 16, 32, 48}, one block per SM, time per block:
   - H-a, the current rule, per instruction max(15 × pair share, 128 B × SM-equivalents / 137.8): 3.0, 6.0, 6.0, 6.0, 6.0, 9.0 µs.
   - H-b, a lower effective chip limit (≈ 102 B/clk, the value that fits the 32-block gap): 3.0, 6.0, 6.0, 6.0, 8.1, 12.1 µs.
   - H-c, the current rule plus a fixed per-block store tail (≈ 1.1–2 µs, e.g. drain and write acknowledgement before the block's slot frees): 4.1–5.0, 7.1–8.0, 7.1–8.0, 7.1–8.0, 7.1–8.0, 10.1–11.0 µs.

   They separate at 1 block (H-c vs H-a/H-b) and at 48 blocks (H-b vs H-c). Upstream's 1-block record (4.1 µs) already leans to H-c. A 16-byte-lane variant (128 STG per block; H-a predicts 0.7, 1.4, 1.4, 1.4, 1.45, 2.2 µs) tells a per-instruction cost from a per-byte one.
4. **Model prediction**: the numbers under H-a come from the engine's own rule. They must be replaced by a replay of the probe's own SASS before it is run (the probe is not written yet; `storepat.cu` upstream is the pattern to copy). The replay needs only `SM` with `stg_share` and `active_sms` set as for a grid, as `kernel_us` does.

### P-2 cp.async chip-L2 delivery at 32 blocks with one block per SM (second component of the same row)

1. **Mechanism**: chip L2 rate for copy traffic shared by blocks in flight (inventory §5, row 6: 817.7 B/clk from `loopsteps.cu`, `l2share.cu`).
2. **Why now**: this component holds 46 % of the same rows' critical path. A what-if at 600 B/clk moves the row +20.9 %. Removing the 8.2 µs measured store share leaves about 19.5 µs of loads measured against about 15.1 µs predicted. Bytes in flight per SM are a design variable (stages).
3. **Competing hypotheses** (copy-only loop without the C store, `nocompute_nostore`, 48 KB per k-tile, 2 stages, blocks ∈ {1, 16, 32, 48}):
   - H-a, the chip rate is flat in blocks in flight (current): time per k-tile = max(SM port, 48 KB × share / 817.7 B/clk).
   - H-b, delivery is bounded by bytes in flight per SM (Little's law with the 350-cycle L2 latency): it rises with stages, not with blocks.
   - H-c, the chip rate falls with fewer, larger copies in flight (the 128×256 tiles issue B rows of 512 B, 4 lines per copy): it depends on lines per copy.
4. **Model prediction**: replaying the `nocompute_nostore` SASS needs that SASS. Upstream saved only `full`, `nocompute` and `noload`, so the SASS must be compiled first (spark, `gb10-measure`). Its 1-block measured value (12.8 µs) exists in `ablation.jsonl`.

The order is P-1, then P-2. P-1 alone is expected to close about 2 µs of the 6.6 µs gap on the two worst loads-only rows. P-2 accounts for the rest.
