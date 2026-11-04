# TileLang NVIDIA design study

Assignment: continue the full NVIDIA catalog after user-paused partial CUTLASS.
Survey distinct computation, layout, pipeline, quantization and scheduling
choices; deep-read representatives and inspect relatives only for differences.
Do not resume CUTLASS in this catalog pass.

Pinned source and license: [manifest](inputs/sources.json), full retained
TileLang top-level archive/tree with MIT text and additional dated notices.
External submodules were not materialized. The
[starting pool](inputs/pool-snapshot.json) includes the 23 registered CUTLASS
proposals and excludes its saved unregistered mask draft; admission is separate.

Status: source-design pass complete. The [inventory](inventory.md) has a
representative-backed disposition for every NVIDIA design group and accounts
for the immediate example directories. [Review](review.md) links six provisional
Template/ProofOutline pairs, source comparisons and validation. Closure follows
the user's distinct-design scope, not an entry-by-entry implementation audit.

Native layouts/capabilities, numerical admission, unmaterialized submodules and
named algebra/realization gaps remain explicit. No compiler/GPU run or benchmark
was needed to resolve the recorded extraction choices. End-to-end replay stays
deferred. Next catalog library: NVIDIA/TransformerEngine.
