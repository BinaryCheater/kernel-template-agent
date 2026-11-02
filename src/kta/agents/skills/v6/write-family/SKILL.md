---
name: write-family
description: How to build or extend an executable family — a chain of Template steps, the generator that emits code for each assignment, and the code markers that tie code regions to chain items — and how to check it with kta validate-family. Used for tasks T1 and T6.
---

# Writing a family

A family is an executable chain: the chain lists the refinement steps (each binds one
Template), and the generator turns an assignment of the variables into kernel source.
The system, not you, enumerates assignments, compiles, checks and measures them.

## Start from what exists

1. When the case has a family, `out/` already holds it: change only what you decided to
   change. Bump the version; keep the name. Without one, start from the example below
   and the reference chains (`kta query chain --case <case>`).
2. For each chain item, check whether its Template's Γ and preconditions still hold for
   the new case (skill `find-knowledge`). Only the items that fail need work.

## The family directory

```
family.yaml        # Family body: name, version, backend, variables, glue_constraints, dispatch, entry
chain.yaml         # Chain body: workload, items
gen.py             # generate(assignment: dict, case: dict) -> GeneratedKernel
edge_cases.yaml
README.md          # one page at most
```

For every chain item:

- `key`: unique name inside the chain;
- `template`: the Template it applies (`name@version`);
- `applies_to`: the concrete transaction it refines;
- `bindings`: each Template variable -> a family variable, a constant, or an expression
  over family variables, `case.*` and `hw.*` (`p.<name>` for Pattern parameters);
- `after`: keys of the items it depends on (a DAG);
- `code_markers`: names of the marked regions in the generated code that implement it.

In the generated source, wrap the code of each item between its markers, so a checker
can relate code to steps. A region without an item, or an item without a region, is an
error. Every glue constraint's `meaning` must say why it cannot belong to one step.

## A complete minimal example (the formats the checks expect)

`family.yaml` — `variables` is a list; `dispatch` is a mapping with a `rule`; `entry` is
`<file>:<function>`; there is no `constraints` key (constraints come from the Templates and
`glue_constraints`):

```yaml
name: gemm_fp16_example
version: 1
backend: tilelang
variables:
  - {name: BM, kind: numeric, binding: compile, role: software, domain: [64, 128], meaning: "CTA tile rows"}
  - {name: S, kind: numeric, binding: compile, role: software, domain: {lo: 2, hi: 4}, meaning: "pipeline stages"}
  - {name: KT, kind: numeric, binding: runtime, role: software, domain: {lo: 1, hi: 65536}, meaning: "k-tiles per tile"}
glue_constraints:
  - {id: g_acc_regs, expr: "BM * BN <= threads * 128", cls: resource, meaning: "why no single step owns it"}
dispatch:
  runtime_vars: [KT]
  buckets: []
  rule: "KT = ceil_div(case.K, BK) for each shape"
entry: gen.py:generate
```

`chain.yaml` — every Template variable is bound, and so is **every parameter `p.<name>`
the Template uses** (`kta show <template id>` lists them; the self-check names any that
is missing). Bind it to a constant or an expression over family variables, `case.*`, `hw.*`:

```yaml
workload: "fp16 GEMM C[M,N] = A[M,K] @ B[K,N]"
items:
  - key: tiles
    template: output_tiling@2
    applies_to: "the M x N output, one BM x BN tile per CTA"
    bindings:
      tile_m: BM
      tile_n: BN
      executor: cta
      index_map: identity
      p.M: "case.M"
      p.N: "case.N"
      p.independent_outputs: true
      p.input_bytes_per_tile: "(BM + BN) * KT * BK * 2"
      p.compulsory_bytes: "(case.M * case.K + case.K * case.N + case.M * case.N) * 2"
    after: []
    code_markers: [tiles]
```

`gen.py` — `generate(assignment, case)` returns a mapping. `launch.params` lists the
kernel arguments in order (a launch without them fails on the GPU), and `smem_bytes` is
the dynamic shared memory the kernel needs:

```python
def generate(assignment, case):
    a, s = assignment, case.get("shape", case)
    source = KERNEL.format(**s, **a)          # the kernel, with kta:step markers per item
    smem = a["S"] * (a["BM"] + a["BN"]) * a["BK"] * 2
    return {
        "files": {"kernel.py": source},
        "entry": "main",                       # the kernel function in the source
        "launch": {
            "grid": [s["N"] // a["BN"], s["M"] // a["BM"], 1],
            "block": [a["threads"], 1, 1],
            "smem_bytes": smem,
            "params": [{"name": "A", "type": "f16*"}, {"name": "B", "type": "f16*"},
                       {"name": "C", "type": "f16*"}],
        },
        "markers": ["tiles"],
    }
```

TileLang in this harness: kernels are launched from the cubin, so TMA and warp
specialization are not available; the build turns them off for every kernel. Copies stay
`cp.async` (TileLang's `T.copy` lowers to it when TMA is off).

## Changing one step

1. Say in `NOTES.md` which item changes and why (which condition failed).
2. Pick the replacement Template (or a new version of the same one). If none exists,
   stop: describe the missing step as a Gap in the delivery note rather than writing
   untracked code.
3. Change the bindings and only the marked regions of that item.
4. Re-check the constraints that mention the changed variables.

## Self-check before submit

- `kta validate-family out`: chain consistency (every item bound and marked, every
  constraint evaluable), the formulation assembled on the case, a feasible point, and one
  generated kernel. It runs what verify_space runs before compiling; fix every issue.
- `kta instantiate` and `kta build` accept existing Family record IDs with assignment/shape JSON.
  A draft under out/ is compiled by verify_space after submission; it has no Family ID yet.
- Do not tune by running many candidates yourself; the system does that.

## Delivery note

List the items you changed, the Templates (IDs) each now binds, the conditions that
made the change necessary, every constraint you added, and anything you could not
express (as a Gap description).
