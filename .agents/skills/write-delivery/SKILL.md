---
name: write-delivery
description: Write deliveries/<WP>.md and deliveries/<WP>.evidence.json for a finished work package so the gate and the coordinator can verify them.
---

# Write a delivery

## Delivery note: `deliveries/<WP>.md`

English, with exactly these `##` sections (the gate checks the headings):

1. `## What was done` — each connected capability, one bullet each, with where its real run is recorded (record IDs, evidence entry, or output path). Separate what is designed, what has an interface only (`NotSupported`), what ran on fixtures, and what ran for real.
2. `## Fixtures and simplifications` — every file containing `# kta:fixture`, with the reason; every simplification the WP allows that you used. Write "None" if there are none.
3. `## Known issues`
4. `## New decisions` — decision IDs added to `docs/decisions.md` (e.g. D80, D81).
5. `## Needed changes` — spec or other-WP changes you need; "None" if none.
6. `## Gate output` — the summary block of `scripts/gate <WP>`, pasted verbatim, from the final run.

Explain any skipped or xfailed test by naming it in the note (the gate warns otherwise).

## Evidence manifest: `deliveries/<WP>.evidence.json`

A JSON list following `deliveries/evidence.schema.json`. One entry per claim:

```json
{"claim": "...", "command": "the exact command", "git_commit": "<40-hex of the code that ran>",
 "dirty": false, "env": "dev", "control_host": "<hostname>", "workers": ["spark"],
 "record_ids": [], "blob_ids": [], "env_manifests": [], "started_at": "2026-10-30T05:00:00+00:00",
 "requires_gpu": false, "output": "where the captured output is kept"}
```

Rules:

- Run claims from a committed, clean tree. Commit code first, run, then commit the delivery files (the evidence commit is the code commit, not the delivery commit).
- Store-backed WPs list record IDs that exist in `$KTA_DATA/dev/<WP>/kta.db` with origin `real`. A number that only appears in the note or in stdout is not evidence.
- Claims that need a GPU set `requires_gpu: true`; their records must come from a worker labelled `gpu`.
- Never paste secrets, tokens or credential paths' contents.
