# File-based Template library

Start with [the catalog](catalog.md) (or [JSON cards](catalog.json)), choose an
explicit name and version, then read its GUIDE.md, template.yaml and proof.yaml.
Each rule describes a local design transformation; select and compose rules
against the workload rather than treating a source decomposition as an answer.

1. Match the computation and structural cue. Category names do not restrict use.
2. Bind caller-fixed extents, dtypes, strides, masks and numerical policy first.
   Select legal optimizer choices and derive dependent sizes. Record those bindings.
3. Check scope, constraints and preconditions, including target/compiler support.
   A source menu and an abstract variable domain are different things.
4. Match producer/consumer coordinates, numerical frames, storage lifetimes and
   synchronization at every interface. Test tails and failure cases from the guide.
5. Keep a brief design note: versions used, bindings, implementation choices and
   missing rules or unsupported extensions. Validate the actual output and measure
   the actual implementation separately.

For dense FP16 GEMM, useful independent retrieval cues are output ownership,
reduction grouping, instruction ABI, physical storage, movement and buffer reuse.
For causal attention, look for streamed normalization and structural mask domains;
keep caller-fixed causal validity and scale separate from tunable tile geometry.
The catalog supplies alternatives, not a prescribed composition or whole kernel.

All entries retain proposed status. File publication adds no solver or GPU
assurance. An empty requires list does not establish backend support. Read the
outline's open assumptions even when a guide gives a concrete scalar example.

[mapping.json](mapping.json) accounts for every discovered seed/reference/study
candidate. [unresolved.md](unresolved.md) lists incomplete proposals and conflicting
identities excluded from the catalog. Originals remain in place. Multiple versions
are addressable; the highest version is not an automatic default. Source paths in
guides/mapping are provenance and are not required to instantiate the local rule.
The source-study files are excluded from isolated replay packages.

Regenerate from the repository root:

```sh
.venv/bin/python tools/template_library.py
.venv/bin/python tools/template_library.py --check
```

The builder prunes inputs, pool-records, snapshot(s) and this library from discovery.
It validates Template and ProofOutline schemas, preserves template bytes, and
withholds different contracts sharing an identity. Curated explanations live in
curation.yaml. The single new outline lives in curated-proofs/ and is explicitly
marked as consolidation-authored. Generated catalogs and rule directories must be
changed through their source proposal or curation input, then regenerated.
