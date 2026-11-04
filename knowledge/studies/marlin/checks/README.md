# Marlin source-equation replays

Run from the primary checkout:

```sh
.venv/bin/python knowledge/studies/marlin/checks/packing_decode.py
.venv/bin/python knowledge/studies/marlin/checks/striped_combine.py
.venv/bin/python knowledge/studies/marlin/checks/validate_chain.py
.venv/bin/python knowledge/studies/marlin/checks/output_callers.py
.venv/bin/ruff check knowledge/studies/marlin/checks
.venv/bin/ruff format --check knowledge/studies/marlin/checks
```

All six pass. [Retained result](packing-decode-result.json) verifies all18 retained
source files against Git blob identities and their archive,229376 coordinate tags
through the host inverse,65536 selected-nibble decode combinations and a binary16
cast-before-scale counterexample. Logical read/fragment maps and scale ownership are covered by the composed replays.
Compiled ISA, numerical tolerance and GPU execution remain open. No ProofResult or
Store admission is produced. No installed dependencies or GPU work were required.

[Stripe/fragment results](striped-combine-result.json) cover1800 ownership domains,
29341 slice segments,262144 B-fragment coordinates,4096 scale-column bindings,
1536 shared int4 addresses and160 scale-ring positions. The reference PTX fragment
mapping is linked from trace.md. Reverse-K versus increasing-K FP16 commit order and
pre-final-store lock reset are retained numerical/lifetime boundaries.

[Chain validation](chain-result.json) checks four concrete reused-rule identities,
pinned bodies, schema, variable domains and dependency order. Proof premises and
statistical/numerical/device obligations are not admitted by those shape checks.
Correction pass: fixed YAML key spacing, preserved reverse-K order, and replaced the
incorrect self-inverse XOR assumption with actual logical write/read composition.


[A/output/caller results](output-callers-result.json) cover45056 A-fragment
coordinates,1020928 live output coordinates,21504 per-channel scale bindings and4100
M/parallel launch domains. A one-step host replay preserves the max_par0/M65
nonprogress witness without running the nonterminating source loop. PTX logical
ldmatrix/MMA layout is a reference relation, not a device execution result.
