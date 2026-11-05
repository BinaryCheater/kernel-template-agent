# Decision-level form and MIP search: first results

Status: research pilot, not accepted by the user. Numbers below come from `lr` and `lr2`, both valid under the gb10-measure checks, on spark-6951. `lp1` failed its clock check (−1.55 %) and is development evidence only.

## Conclusions

- **A simulator level between L3 and MIP was not needed.** The searchable level is a closed form over decision variables. It takes no SASS, and it uses no L2 in the old sense.
- **On new shapes the funnel found kernels faster than the conventional choices with few GPU runs,** on one new routed quantized kernel and two earlier families.
- **One blind model round is not safe by itself.** Form v3 failed on MoE: the MIP exploited a missing mechanism. Round two, after refitting on eight shortlisted and twelve control rows, repaired it. Measuring a few and refitting is part of the method.
- **Hardware constants copied from `gb10.json` hold a fixed anchor for extrapolation.** Fitting them freely lost ranking on variants left out.

## Search results

Times in µs. Anchor is a conventional choice picked before the run. Random is 12 compiled assignments drawn before any time was seen. The true optimum of each domain is unknown: only a small part was measured.

| Kernel | Domain | GPU rows, shortlist | Anchor | Random median, best | Round 1 best | Round 2 best | Against anchor |
|---|---:|---:|---:|---:|---:|---:|---:|
| HGEMM 3200×2240×1920 | 1143 | 8 + 8 | 338.5 | 434.5, 349.4 | 334.2 | 323.1 | −4.6 % |
| FA forward 2240×8×96 | 180 | 4 + 1 | 230.5 | 234.7, 194.8 | 185.3 | 209.3 | −19.6 % |
| MoE GEMM on NVFP4 weights, E 4, T 3840, N 2048, K 1536 | 6688 | 8 + 8 | 786.8, 1141.0 | 1680, 852.4 | 995.6 | 596.2 | −24.2 %, −47.8 % |

MoE round 2: seven of eight beat the better anchor, median error of the eight was about 14 %, worst 36 %. Round 1 predicted 373 to 384 µs and measured 996 to 3676. A generated 160×48×64 kernel matched a torch reference to 0.05 % on the GPU (`perfmodel/testset/check_moe.py`), as did two other tiles; other shortlisted tiles were timed only.

MIP translation: top picks equal direct enumeration on HGEMM, FA and MoE. The check caught one real defect, a term read from the wrong factor on FA, before any selection. On these single-kernel domains enumeration of the form takes about 30 ms and the MIP about 30 s for one solve, so the MIP adds nothing yet over enumeration. Its case rests on composed spaces too large to enumerate and on hardware variables; neither is tested.

## Form against L3 on retained rows

Round 21 was never used to fit. It was scored once on v1; later versions saw only the score. Pairs measured ≥ 3 % apart, right direction:

| Family | Form v4 | L3 blind | Best in top 3, form, L3 |
|---|---:|---:|---:|
| HGEMM | 64/69 | 61/69 | 2/2, 2/2 |
| FA | 45/50 | 47/50 | 2/2, 2/2 |
| Decode | 20/35 | 32/35 | 1/2, 2/2 |

That is one new shape, two groups a family. Variant left out across all shapes, v2, hardware frozen: HGEMM 85 % pairs, 29/29 top 3; FA 86 %, 20/20, against L3's 92 % and 91 % on the same rows. Fitting every constant freely gave 74 % and 64 %. Decode is weak and has had no search run.

Constant sources on held-out shapes, FA and HGEMM: zero fit keeps the best in the top 3 but ranks only 47 to 78 % of pairs; a few software constants reach 81 to 88 %; calibrating them against L3 instead of the GPU gave the same pair score, so that step needs no GPU, though level error doubles.

## Not done

- No Template, chain or agent drives this yet. I wrote each family's amounts by hand from the generator. Three-arm runs have not started.
- Kernels are TileLang generator families with shared memory and `cp.async`. Other implementation styles are not modelled.
- L3 replay of the MoE kernels had not finished after 55 minutes and was stopped, so L3 has no MoE comparison. On `lp1` warm rows it had median error 6.1 % on HGEMM and 8.5 % on FA.
- Cost accounting is partial: no token count for this session; about two hours of wall time from freeze to `lr2`. MoE's calibration used 44 `lp1` rows on another shape, 22 `lr` rows and six old NVFP4 rows, so 16 is the funnel's GPU count on the final shape, not the whole preparation.
