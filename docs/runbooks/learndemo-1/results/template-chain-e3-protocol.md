# E3: does Template information help the full chain? Protocol

Frozen before any model session of this experiment. Status: finished, not accepted by the user. [Results](template-chain-e3.md).

**Question.** With the same form, MIP, funnel and budgets, does Template information change the kernel the chain finds, and when it does not, is the Template missing, unfound, or found and not composed?

**Arms.** `none`; `prescribed`, the rules in the E2 gold sets that survive leave one source out (19 for decode, 16 for forward); `selection`, the progressive-disclosure access path over the same leave-one-source-out view. All arms get TASK, FORM, the form API, TileLang docs and one plain GEMM example.

**Kernels and comparators.** Both are production implementations running unchanged on GB10, and both passed an independent reference.

| Task | Comparator | Shape | Comparator time |
|---|---|---|---:|
| `paged_decode` | FlashInfer 0.6.6, FA2 backend, in the sglang 26.03 image | B 64, H 32, Hk 8, D 128, page 16, lengths 512 to 4096 | about 2436 µs |
| `causal_forward` | pinned FlashAttention-2 build of the first campaign | S 2048, H 8, D 128 | about 150 to 170 µs, spread under review |

Rejected as comparators: Marlin at the pinned revision gives wrong blocks at M 64 and an illegal instruction at M 16 on GB10; FlashInfer's `grouped_gemm_nt_masked` and group MXFP4 GEMM support SM100 only; `mm_fp4` runs through cuDNN, on hardware FP4 that TileLang cannot reach through ptxas.

**Cell.** One Codex `gpt-6.1-sol` session at high effort, as in E2, 45 minutes, 30 compile-and-check calls, no timing. It writes `family.py` and `form.py`. Then, with no model: MIP shortlist of 12, up to three passes when compiles reject picks, compile records, rerank, measure 6 plus the anchor plus 2 random controls. One report goes back to the same session for one revision. Round two repeats the search with refit software constants and measured points cut, measuring 6 more. Repeats: 2 per cell, 12 cells.

**Measures.** Best correct measured time against the comparator and against the session's own anchor; rank of the best; form error; MIP against enumeration on each form; tokens, wall time, GPU rows. Each timed row passes the correctness check at full shape and a changed-input graph replay.

**Causes.** For each gold move: absent from the view, not surfaced, surfaced and not selected, selected and not used in the family. Attribution uses the access logs, DESIGN.md and the source. It is my judgement and is labelled as such.

**Limits fixed in advance.** Two repeats is too few to claim small differences. Batches are valid when no PERFMODEL run overlaps and the comparator moves at most 5 % within the batch, a looser check than gb10-measure's canaries. Hardware constants stay frozen; only software constants refit.
