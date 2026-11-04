# DeepGEMM NVIDIA source study

Assignment: continue the NVIDIA catalog using the extraction skill and library playbook. Primary checkout output: `knowledge/studies/deepgemm/`. One library at a time.

Pinned source: DeepGEMM `5081f8513af57e61696de4f4f04b0e707ffb503e`, MIT; retained complete superproject archive and tree in [sources.json](inputs/sources.json). Both direct CUTLASS and DeepJIT submodule revisions are retained separately in [submodule-sources.json](inputs/submodule-sources.json); nested submodules are not implied. No downloaded source is deleted by this study.

Starting pool: [file manifest](inputs/pool-snapshot.json), repository seed/reference versions plus provisional Humming candidates. [Reviewed index](../reviewed.yaml) is empty. Reused reference candidates remain provisional; a name/version collision must be resolved by the pinned file path.

Scope: every public operation, registration, dispatch family and device implementation in this revision, including SM80 legacy Triton, SM90 and SM100 paths; helpers and wrappers receive explicit dispositions. Trace one concrete witness per mechanism group and account for variants. Separate source arithmetic, legality, heuristics and measured behavior.

Stopping condition: inventory reconciled to the retained source; observed decisions mapped to versioned rules or explicit gaps; candidates paired with outlines, schema checks and discriminating source-binding replays; one correction pass and a short review. End at proposals; admission, GPU execution, WP delivery and commits are separate assignments. Then continue to Marlin in catalog reading order.

Status: source study complete; pending independent review/admission. All160 pinned superproject files are retained, both direct submodules are retained, and all71 API registrations and28 discovered device sources have source-backed trace assignments in [coverage.json](inputs/coverage.json). Nine Template/outline pairs, eight CUDA Realisations and five partial Chains validate; the complete22-command replay/lint batch passes. No GPU kernel or external backend has run. Implementation/proof gaps remain explicitly declared in the family traces.

Closure: [short rule menu and worked composition](review.md), [inventory dispositions](inventory.md), [configuration/host and coverage closure](trace-configuration.md), [checks and retained results](checks/README.md). Source extraction, relevant bounded replays and one final correction pass are recorded. No WP gate, commit, formal promotion or Store admission is part of this study.

Next catalog library: Marlin. Reuse this study's candidates only as provisional dependencies alongside the pinned seed/reference and Humming baseline; reviewed.yaml remains empty until independent review.
