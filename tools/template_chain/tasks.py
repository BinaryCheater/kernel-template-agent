"""The two production comparisons of the experiment."""

TASKS = {
    "paged_decode": {
        "source_study": "flashinfer",
        "gold": "paged_batch_decode",
        "image": "nvcr.io/nvidia/sglang:26.03-py3",
        "constants": "decode_gqa",
        "dev_shape": {
            "B": 6,
            "H": 8,
            "Hk": 2,
            "D": 128,
            "page": 16,
            "min_len": 40,
            "max_len": 300,
            "table_seed": 7,
        },
        "shape": {
            "B": 64,
            "H": 32,
            "Hk": 8,
            "D": 128,
            "page": 16,
            "min_len": 512,
            "max_len": 4096,
            "table_seed": 7,
        },
        "statement": """# Batched attention decode over a paged cache

Write a TileLang family for one decode step of attention for a batch of requests.

For each request b and query head h, with g = H / Hk and scale = 1 / sqrt(D):

    s = scale * q[b, h] @ K_b[:, h // g].T
    o[b, h] = softmax(s) @ V_b[:, h // g]
    lse[b, h] = log2(sum(exp(s)))

- `inputs` holds q FP16 [B, H, D]; k and v FP16 [num_pages, page, Hk, D]; and the page table:
  indptr int32 [B + 1], indices int32, last int32 [B]. Request b owns pages
  indices[indptr[b]:indptr[b + 1]] in order. Its last page holds last[b] tokens, the others
  `page` tokens. Pages are scattered: adjacent pages of a request are not adjacent in memory.
- `run()` returns {"o": FP16 [B, H, D], "lse": FP32 [B, H]}.
- The page table is fixed when `prepare` runs, which may plan and build workspace on the GPU or
  in Python. The contents of q, k and v change between calls, in place: `run()` must read them
  every time. Requests have very different lengths.
- Accuracy: |o - ref| <= 0.02 + 0.01 |ref|, |lse - ref| <= 0.02 + 0.002 |ref|.
- Search shape: B 64, H 32, Hk 8, D 128, page 16, lengths uniform in [512, 4096). The check
  command uses a smaller shape. `shape` carries the same names plus min_len, max_len.

Target: NVIDIA GB10, CUDA, through TileLang. Objective: lowest GPU time per call.
""",
    },
    "causal_forward": {
        "source_study": "flash-attention",
        "gold": "attention_causal_fp16",
        "image": "kta-worker:tilelang-0.1.15",
        "constants": "fa_forward",
        "dev_shape": {"S": 256, "H": 2, "D": 128},
        "shape": {"S": 2048, "H": 8, "D": 128},
        "statement": """# Causal attention forward with log-sum-exp

Write a TileLang family for causal multi-head attention, forward only.

For each head h and query i, with scale = 1 / sqrt(D) and visible keys j <= i:

    s[j] = scale * Q[0, i, h] @ K[0, j, h]
    O[0, i, h] = softmax(s) @ V[0, :i + 1, h]
    LSE[0, h, i] = log(sum(exp(s)))

- `inputs` holds q, k, v, FP16 [1, S, H, D].
- `run()` returns {"o": FP16 [1, S, H, D], "lse": FP32 [1, H, S]}.
- The contents of q, k and v change between calls, in place: `run()` must read them each time.
- Accuracy: |O - ref| <= 0.01 + 0.01 |ref|, |LSE - ref| <= 0.001 + 0.001 |ref|. Accumulation
  and normalisation use FP32.
- Search shape: S 2048, H 8, D 128. The check command uses a smaller shape.

Target: NVIDIA GB10, CUDA, through TileLang. Objective: lowest GPU time per call.
""",
    },
}
