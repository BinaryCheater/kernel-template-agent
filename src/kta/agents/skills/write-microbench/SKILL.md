---
name: write-microbench
description: How to write a microbenchmark that measures one hardware parameter — isolate one unit, state the work of one launch, make it repeatable, and deliver microbench.yaml plus the CUDA source that KTA builds and measures five times. Used for task T5.
---

# Writing a microbenchmark

A T5 task names a hardware fact the system lacks (a bandwidth, a latency, a capacity, an
issue rate). You write one small CUDA kernel whose time is governed by that one
parameter, and state the work of one launch. KTA builds it, measures it five times under
the measurement protocol on an idle GPU, and accepts it when the repeats agree (their
coefficient of variation is below twice the protocol's noise floor). The accepted value
becomes a measured HwFact other tasks can use.

## Design

1. **One unit.** The kernel must spend its time on the unit being measured and as
   little as possible on others. For a bandwidth, move many bytes with wide, coalesced
   accesses and keep arithmetic trivial; for a latency, make each access depend on the
   previous one (a dependent chain) so accesses cannot overlap; for a throughput of
   instructions, use several independent chains per thread so the pipeline stays full.
2. **Defeat the compiler.** Every result must reach memory (write a reduction of the
   values to an output), and loop trip counts must come from kernel parameters, or the
   compiler removes the work.
3. **Count the work exactly.** Write the work of one launch as an expression over the
   shape variables (bytes, loads, instructions, CTAs). The value is either work per
   second (a rate) or seconds per work item (a latency).
4. **Size it.** One launch should take well above the timer's resolution (tens of
   microseconds or more) and below a second. Know what the protocol does around each
   launch: it flushes the L2 cache and fills outputs before every timed launch, and it
   fills inputs with random values. A benchmark that needs data in a cache must touch
   it many times per launch so the first, cold pass is a small share.
5. **Fill the machine or not, on purpose.** A peak needs every SM busy (grid a multiple
   of the SM count, enough warps per SM); a latency needs one thread. Say which.

## Deliverable

- `out/kernel.cu`: one `extern "C" __global__` function. Pointer parameters are `f32*`
  buffers sized by shape variables; scalar parameters are `i32` shape variables.
- `out/microbench.yaml`:

```yaml
quantity: <name of the fact, e.g. what it measures and its unit>
unit: <hardware unit the fact belongs to, as the model names them>
entry: <kernel function name>
source: kernel.cu
grid: <CTAs>
block: <threads per CTA>
params: [{name: X, type: "f32*"}, {name: n, type: i32}]
inputs: [[X, n]]          # (buffer, element-count variable)
outputs: [[Y, m]]
shape: {n: 8388608, m: 1}
work: "4 * n"            # expression over the shape variables
value: work_per_s        # or s_per_work
meaning: <one sentence: what the kernel does and what the work counts>
```

- `out/DELIVERY.md`: what was done (the design and why it isolates the unit), the
  simplifications (what else the time includes, e.g. TLB misses), known issues.

Check that the kernel compiles with `kta build` and runs with `kta run` before you
submit; do not call `kta measure` repeatedly to select a lucky number — the done-check
measures it itself.

## What not to do

- Do not hard-code or post-process the result; the value comes only from KTA's
  measurement.
- Do not measure two parameters in one benchmark. If the fact needs two (for example a
  bandwidth at several working-set sizes), write one benchmark per point.
