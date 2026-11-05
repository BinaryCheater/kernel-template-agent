"""Common public instructions for every arm of the controlled reconstruction case."""

COMMON_PROMPT = (
    "Implement TASK.md. Use ENVIRONMENT.md and the supplied evaluation interface. Validate"
    " and optimize within the fixed budget, preserve the best correct self-contained "
    "candidate and a short DESIGN.md. Preserve failed versions. Do not read original "
    "target implementations, study traces, previous solutions or evaluator code. Run all "
    "target correctness and timing checks through the evaluator. Stop according to the "
    "shared stopping rule."
    "\n"
)

CLAUSES = {
    "prescribed": "Read and use the supplied optimization rules in library/. Record the rules"
    " used and choices supplied from your own knowledge.\n",
    "self-selection": "Select relevant optimization rules from library/catalog.md, read and use "
    "them, and record the rules used and choices supplied from your own "
    "knowledge.\n",
    "no-template": "",
}

TASK = """# Causal attention forward

Implement `entry(q, k, v) -> (O, LSE)` in one self-contained `candidate.py` using
TileLang 0.1.15 for GB10 sm_121. Q, K and V are contiguous FP16 CUDA tensors in
[B,S,H,D] order. B=1, H=8, D=128. All three sequence lengths S=512,2048,513 are
required. Q and KV have equal head counts.

For every batch b, head h and query i, form score(i,j) = dot(Q[b,i,h,:],
K[b,j,h,:]) * scale, where scale is FP32(1/sqrt(128)). A key j is visible iff
j<=i. Softmax normalizes over all visible keys. O[b,i,h,:] is the probability-
weighted sum of V[b,j,h,:]. LSE[b,h,i] is the natural logarithm of the sum of
exponentials of the visible scaled scores. Both outputs are required.

Return a new contiguous FP16 O tensor of shape [B,S,H,D] and a new contiguous
FP32 LSE tensor of shape [B,H,S], on the input device. Accumulation and normalization
state use FP32. Numerical refinements must pass the fixed checks. No dropout,
backward, ALiBi, window, softcap, paging, GQA, returned probability matrix or KV
update is requested. Input values change between invocations.

Elementwise acceptance: O absolute error <=0.01+0.01*abs(reference); LSE absolute
error <=0.001+0.001*abs(reference). The evaluator uses an independent mathematical
reference and checks complete outputs, ABI, finiteness, unchanged inputs, changed-
data repeat invocation and captured CUDA-graph replay. Tolerances are fixed.

Torch tensor allocation and metadata inspection are allowed. Attention arithmetic
must execute in your own TileLang kernels. No input mutation, CPU tensor arithmetic,
input/result cache, prebuilt attention delegation, reading target source/answers,
or untracked target tests/timing. Cache only compiled executables. Ship only
candidate.py; private helpers cannot hide computation. Include every necessary
GPU conversion, packing and normalization on each invocation.

Primary timing covers the complete GPU computation under CUDA graph replay,
including both outputs and all necessary GPU preparation. Use harness/evaluate
for every target correctness/timing request. It supplies per-shape measurements
against the fixed original implementation. The performance target is original_us /
candidate_us >=0.90 on every shape (candidate at most about11.1% more time).

Budget: 1800 active seconds and20 target submissions, whichever is reached first.
Perform all model work in this one recorded session. Do not spawn other agents
or invoke another model, model CLI or model API. The same single-session rule
applies to every arm; native child-agent features are disabled.
Verified GPU-queue wait is excluded; reading, thinking, compilation, transport and
ordinary tool work count. A compile/runtime/numerical failure consumes a submission.
Finish early only after two consecutive unchanged-source submissions pass every
check and meet the target on every shape; voluntary finish is allowed. Otherwise
stop at budget. Preserve failed versions in candidate_versions/ before replacing
candidate.py. End with selected candidate.py and DESIGN.md naming the source selected.
Final hidden validation and confirmation are coordinator-only, with no repairs.

Keep DESIGN.md concise: implementation choices, experiments, best observed result
and source selected. Use scratch/ for retained compiler logs and static API checks.
Do not modify supplied task, environment, docs, examples or harness files. The
coordinator audits native commands and accesses. Native full access is instruction-
only isolation, not an operating-system sandbox. Do not use arbitrary network
searches or installed attention implementation examples as documentation.
"""

INTERFACE = """# Common evaluator interface

From the workspace run:

```sh
harness/evaluate candidate.py
```

The default candidate path is candidate.py. The launcher supplies
KTA_SOURCE_REPLAY_ROOT, KTA_SOURCE_REPLAY_SLOT and SOURCE_REPLAY_CLI. The common
wrapper invokes the private controller's `evaluate ROOT --slot SLOT --candidate FILE`.
All target correctness and timing runs use this interface. Submit a self-contained
candidate; scratch and support files are not shipped. Requests are synchronous:
no background evaluator calls, simultaneous model work during queue wait or other
target execution. The controller owns the fixed submission and active-time budgets.

Public feedback includes `passed`, `window_valid`, `meets_target` and `rows`.
Each row identifies the shape, correctness, per-pattern eager/captured O/LSE checks,
original/candidate medians and samples, production/candidate ratio, percent time
difference and measurement validity. Compile/runtime/numerical failures remain
outcomes. Private input seeds, reference code and hidden checks are not exposed.
Fresh development inputs are generated on each submission. An invalid external
measurement may be retried only under the common controller policy; editing source
is not a free retry. Preserve every response.

Two consecutive unchanged-source submissions must be correct and meet target on
all shapes to use the early stopping rule. Final hidden validation and three
confirmation windows happen after source selection and model closure, outside
discovery, with no model feedback or repair. They do not authorize source changes.

Direct compilation and static public-API inspection remain available through the
fixed locked scratch recipe in ENVIRONMENT.md. Retain source and logs there.
That route does not permit target correctness tests, timing, answer source searches
or access to other slots' artifacts. The allowlisted language docs and unrelated
example are identical in every arm.
"""

EVALUATE = """#!/bin/sh
set -eu
exec "${SOURCE_REPLAY_CLI:?launcher must set SOURCE_REPLAY_CLI}" evaluate \\
  "${KTA_SOURCE_REPLAY_ROOT:?launcher must set KTA_SOURCE_REPLAY_ROOT}" \\
  --slot "${KTA_SOURCE_REPLAY_SLOT:?launcher must set KTA_SOURCE_REPLAY_SLOT}" \\
  --candidate "${1:-candidate.py}"
"""
