# Batched attention decode over a paged cache

Implement one decode step of attention for a batch of requests.

For each request b and query head h:

o[b, h] = softmax(scale * q[b, h] @ K_b[:, h / g].T) @ V_b[:, h / g]

Also return lse[b, h], the base-2 log-sum-exp of the score row.

- q: FP16 [B, H, 128], one query position per request.
- Keys and values live in a page cache: FP16 [num_pages, page_size, Hk, 128]. Pages of
  one request are not adjacent. A page table gives each request's page list through
  kv_indptr (B + 1 entries), kv_indices and kv_last_page_len.
- Requests have very different lengths, from zero to tens of thousands of tokens, and the
  mix changes every step.
- g = H / Hk is 1, 2, 3, 4 or 8.
- o: FP16 [B, H, 128]. lse: FP32 [B, H].
- A host planning stage may run before the kernel and may reuse workspace across steps.

Target: NVIDIA GB10, compute capability 12.1, CUDA C++ with inline PTX allowed.
Objective: lowest GPU time per step for mixed batches, with output correct within FP16
tolerance against a float32 reference.
