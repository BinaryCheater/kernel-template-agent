# Causal scaled dot-product attention

Implement one forward attention operator for inference.

O = softmax(Q @ K.transpose(-2,-1) / sqrt(D) + causal_mask) @ V

Also return LSE, the natural-log log-sum-exp of each masked score row.

- Q, K, V: contiguous FP16 tensors of shape [B, H, S, D], with equal head counts and D = 128.
- O: new contiguous FP16 tensor of the same shape. LSE: FP32 tensor [B, H, S].
- The mask excludes key positions greater than the query position.
- No dropout, no backward pass.
- Representative shapes: B = 1, H = 8, S in {512, 513, 2048}.
- Inputs must stay unchanged. Accumulation and statistics may use FP32.

Target: NVIDIA GB10, compute capability 12.1, CUDA C++ with inline PTX allowed.
Objective: lowest GPU time per call on every shape, with the output correct within
FP16 accumulation tolerance.
