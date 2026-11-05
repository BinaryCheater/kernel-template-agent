# Template replay results

**Scope correction:** these are generic implementation results. They do not establish reconstruction of a studied FlashAttention/FlashInfer kernel. No pinned studied source kernel was used as the direct comparator. The earlier completion claim for that objective is withdrawn; see [correction and missing evidence](scope-correction.md).

**Comparator identification:** a later trace in the original environment identifies PyTorch's integrated fused FlashAttention kernel on all three original attention shapes. These ratios are against SDPA's optimized backend, not an explicitly forced unfused math implementation. [Backend evidence](attention-backend-audit.md).

The file library maps 344 source proposals to 312 versioned paired entries; incomplete/conflicting proposals remain explicit. Four basic runs, category disclosure, a matched DSH run and two separate guide-repair replays are complete. All final candidates pass their full numerical suites. Packages exclude upstream kernels and prior answers; native access was audited.

| Run | Ratios, shape order | Active min | GPU wait s | API token estimate |
|---|---|---:|---:|---:|
| [GEMM prescribed](gemm-prescribed-v1.md) | 0.877 / 0.992 / 1.560 | 10.67 | 0.04 | $0.545 |
| [GEMM self-selection](gemm-self-selection-v1.md) | 0.961 / 1.018 / 1.238 | 9.90 | 6.32 | $0.586 |
| [Attention prescribed](attention-prescribed-v1.md) | 1.048 / 1.082 / 1.207 | 8.16 | 14.75 | $0.517 |
| [Attention self-selection](attention-self-selection-v1.md) | 1.098 / 1.197 / 1.280 | 9.35 | 0.04 | $0.604 |
| [Attention categories](attention-category-disclosure-v1.md) | 1.108 / 1.166 / 1.216 | 9.59 | 99.89 | $0.650 |
| [Attention DSH](attention-dsh-protocol-v2.md) | 1.047 / 1.101 / 1.162 | 14.26 | 10.02 | $0.067–0.134 |

Ratios are production latency / candidate latency. GEMM order: 1024³, 32×4096×4096, 127×1000×769. Attention order: S512/D64, S2048/D128, S513/D64 (B1/H8). The 0.90 per-shape aim is missed by prescribed GEMM's square. Shared GPU locks, pinned toolchain, ten warmups, fifty event samples and fresh validation inputs apply throughout. Detailed reports retain every sample and IQR.

The matched attention Codex/DSH sessions used 3.234M/6.629M input tokens, including 3.180M/6.540M cached tokens, and 17,850/56,962 output tokens. DSH needed more time and tokens; its published-API token estimate is lower. Estimates use [OpenAI](https://developers.openai.com/api/docs/models/gpt-6.1-sol) and [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing/) rate cards; native account charges were not returned. [Accounting method](accounting-method.md) separates model cost, GPU waiting and session wall time.

Category disclosure was usable, but active time was 575 s versus 561 s and estimated token cost $0.650 versus $0.604. Initial native context grew 11,831→12,189 tokens. Ordinary catalog search already worked; this run supplies no efficiency gain from categories.

The physical-storage guide now records an observed compiler-pipelined K/V alias trap. A fresh self-selection replay omitted the changed guide and repeated the failure. A bounded prescribed replay read it and used independent buffers without that failure; its long-shape ratio 0.895 still misses the aim. Automatic lookup when a new optimization changes storage/layout/scheduling remains the useful next improvement. [Repair evidence](failures-and-library-delta.md) keeps these outcomes separate.

An earlier DSH exploration also produced a correct kernel, but untracked manual sweeps lacked raw samples and queue accounting; it is excluded from the match and preserved separately. Final DSH S513 timing varied substantially across batches; the cause is unestablished, so no clock explanation or strong stability claim is adopted.

[Reviewable kernels and replay commands](../../../../examples/template_replay/README.md), [full results index](../replay-results.md), [library](../../../../knowledge/library/README.md). Next: fixed-code unseen-tail tests, then one W4A16 variant with a frozen packing/scaling contract; [transfer proposal](transfer-next.md).
