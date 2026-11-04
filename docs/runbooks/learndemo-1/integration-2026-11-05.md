# Template source-study integration, 2026-11-05

This commit preserves the extraction workflow and source-study proposals on main. It does not admit candidates to the runtime Store, establish GPU performance, or replace the earlier feature merge.

Included: the extraction skill, three operating guides, the retained RMSNorm worked example, knowledge navigation, study notes, Template/ProofOutline proposals, source manifests and reusable local check scripts. The primary checkout is authoritative; its later reading-depth guidance is retained. The RMSNorm example was copied from learndemo-1. Existing seed rules are unchanged.

Excluded: supervision-ux (separate work), .codex local settings, downloaded upstream sources, input/pool snapshots, generated result JSON, file inventories, caches and logs. Existing files were not removed. Excluded study files have an additional retained copy and hash manifest at `/Users/dingyi/kta-data/dev/operator-sources/integration-2026-11-05/manifest.json`. Study links to these local input/result files refer to retained evidence, not files shipped by this commit. Source manifests retain the original archive/extraction locations. Checks that use these inputs require restoring the listed workspace-relative paths from that manifest.

All 306 current study Template/ProofOutline pairs passed schema loading at integration. One name/version collision remains: `chunked_decayed_outer_product_state@2` is proposed by both Mamba and FlashInfer. Both source proposals remain addressable; reconcile their contents before presenting a unique shared rule or importing them. No semantic deduplication or kernel reconstruction is claimed.

Workers' study status and unfinished families are recorded in knowledge/studies/README.md and the individual study pages. No worktree is deleted. Frontend build/full feature validation from the earlier LEARNDEMO merge is outside this source-study commit.

## Integration validation and fixes

The study scripts now pass repository formatting and lint rules. Changes include explicit zip truncation policy, bound loop parameters, readable identifiers, equivalent string-literal wrapping and stdout writes for script summaries. Fifteen local witness/replay scripts across Apex, causal-conv1d, CCCL, Fuser, Mamba, QuACK and Triton passed after edits.

The acceptance lock mismatch predates this integration: SYSDEBUG-2 decision D704 migrated multi_buffer_rotation from version 1 to version 2 and updated the WP4/WP5 tests accordingly, but left their lock entries unchanged. Reviewed the committed diff: it updates version selection, binds late/private to zero, and updates the expected symbolic constraint. No assertions were removed by this integration. Both test files are unchanged; their ten tests passed. Only their two lock digests were refreshed to match the already-merged tests.

The gate has no LEARNDEMO1 selector; the source-decomposition package WP10a was used for its shared quick checks. Full feature/GPU validation remains outside this source-study integration.

```text
gate WP10a --quick: PASSED (10 pass, 0 warn, 0 fail)
  PASS ruff format
  PASS ruff check
  PASS import layers
  PASS residue
  PASS test doubles
  PASS error handling
  PASS print
  PASS subprocess outside Executor
  PASS secrets
  PASS acceptance lock — 7 locked files unchanged
```
