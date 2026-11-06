# The decision-level form

A family is judged by search, not by a single chosen configuration. You write:

- `family.py`: `prepare(shape, x, inputs) -> run`. It emits the TileLang kernels for decision
  assignment `x` and returns a zero-argument `run()`. `run()` launches on the GPU and returns a
  dict of output tensors. `prepare` may do untimed planning and load-time packing.
- `form.py`: what the search reads. It runs on a laptop without a GPU, a compiler or torch, and
  it imports only `form_api` and the standard library.

```python
FACTORS = {"BM": (64, 96, 128), "s": (1, 2, 3)}  # every decision and its menu, 2 to 12 values
STRUCTURE = ("BM",)                 # factors that define distinct designs, for a varied shortlist
ANCHOR = {"BM": 64, "s": 2}         # the configuration you would ship without any search

def legal(shape, x) -> bool: ...    # False for assignments your generator cannot build
def amounts(shape, x, records):     # one Amounts, or a list with one per launched kernel
    ...
```

`records` is `None` before compilation: estimate registers and shared memory from `x`. After
compilation it is a list in launch order of `{"regs", "smem", "threads"}`. Use the real values
when present. The domain holds at most 20000 legal assignments.

## What the search computes

For each kernel, with `c` a table of constants:

```text
T = launch + stores + stage + turnover + barrier + trip
    + max(compute, l2_copies, dram, recurrence)
```

- Issue regime from `warps * occ / 4`, warps sharing one tensor pipe per SM: `w1` up to 1,
  `w15` below 2, `w2` below 3, `w3` above. Use is `min(1, wps)`.
- `compute = (tensor_gflop * c.tensor + scalar_mop * c.scalar + spill * tensor_gflop * c.spill) / use`
- `l2_copies = copy_mb * c.copy`; `dram = dram_mb * c.dram`, only when `rotating`.
- `recurrence = compute / stages + latency * waves * trips / stages`, the loop-carried limit.
- `waves = blocks / min(blocks, 48 * occ)`.
- `turnover = c.wave * waves`, `stage = c.stage * stages`, `store = store_mb * c.store`.
- `trip = c.trip * waves * trips / 100`, `barrier = c.barrier * max(0, warps - 4) * waves * trips / 10`.

Hardware constants are fixed from the machine description: 48 SMs at 2.5 GHz, tensor about 19
µs per Gflop at full use, L2 to shared copy 0.44 µs/MB, store 1.6 µs/MB, DRAM 4.2 µs/MB.
Software constants are refit by the harness on measured rows.

The form has no queue state, so anything you want searched must appear as an amount or a regime.
A mechanism missing from `amounts` is invisible to the search, which will happily exploit it:
if a choice costs time that no amount shows, the optimiser will choose it. Common omissions are
work per block that repeats with tiling, register need beside accumulators, and work the kernel
does outside the main loop. `example/` shows a complete plain GEMM family.

The constants come from kernels compiled with the pass settings in `example/family.py`, where
all threads run the same code. Other settings are allowed and can be a factor, but then the
form has no evidence for them until the first measurements.

## What happens to your files

`./chain check '<json assignment>'` compiles the kernels for that assignment, runs them on a
GPU at a small shape, and reports errors and compile records. It gives no timing.

When you finish, the harness solves a real MIP over your form, compiles a shortlist, reranks it
on compile records, and measures a few on the GPU along with your `ANCHOR`. You then get one
report with predicted and measured times and may revise both files once. A second search follows.
The result is the fastest correct kernel it measured.
