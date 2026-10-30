# Review queue

Decisions made without the user that need review. Newest last. Each entry: question, choice made, rationale, cost of changing it later, link to the decision ID.


## D100 — Seed reference files are bodies; refs implied by layout

- **Question:** WP10a must ship YAML that "parses with the pydantic models", but no seed file format existed (WP4 not started) and refs are content-hash IDs that do not exist for seed files.
- **Choice:** one body per file, kind from file / directory name (compatible with the gate's D87 rules), refs implied by the directory layout; Template and Pattern directories are `templates/` and `patterns/`.
- **Rationale:** same convention as family directories; each file validates directly against its model; `kta init` (WP4) can derive refs when importing.
- **Cost of changing later:** low; a mechanical rewrite of 7 directories.

## D101 — Source has no schema

- **Question:** Reference refs `source` and D59 lists Source as a node, but spec/schemas.md defines no Source fields.
- **Choice:** `source.yaml` with repo, commit, commit_date, license, paths, retrieved_at, note.
- **Rationale:** the minimum needed to freeze and re-fetch an upstream source.
- **Cost of changing later:** low; needs a spec/schemas.md addition (listed under "Needed changes" in deliveries/WP10a.md).

## D102 — How match rate and unmatched transactions are counted

- **Question:** schemas.md defines `matched` / `total` and `unmatched_transactions` but not how Templates proposed during a decomposition count, or how partial matches and step-less transactions are represented. These numbers feed the coverage page (21 §2).
- **Choice:** fixed decomposition order defines the pool; only Templates that existed before the reference count as matched; a partial match is not matched; each unmatched step or step-less transaction is one entry with exactly one Gap.
- **Rationale:** gives a discovery curve that depends only on the order and makes the gate's one-to-one check meaningful.
- **Cost of changing later:** low for data (recount from chain.yaml); medium once the coverage page shows these numbers.

## D103 — Variable names of the five seed Templates were assumed

- **Question:** chains must bind seed Template variables before WP4 defines them.
- **Choice:** names from paper.md §3.3 (num_groups, group_size, assignment; level, buffer, share_order; mechanism, issue_point, complete_point, granularity_bytes; S, P, beta; binding, queue_depth).
- **Rationale:** closest authoritative source; keeps WP10a and WP4 parallel.
- **Cost of changing later:** low; renaming keys in about 60 chain items. Three findings (register double buffering, FlashAttention-2 phase-offset slot, runtime num_groups) depend on these assumed variables.

## D105 — Three new Patterns proposed

- **Question:** D50 says new Patterns need your confirmation; the decompositions needed transactions the 8 seed Patterns cannot describe.
- **Choice:** proposed `work_assign` (assignment and order of work units), `issue_order` (issue order within / across executors) and `scan` (prefix computation), all with category `other` because the schema's category enum has no matching entry; they stay `proposed`.
- **Rationale:** 7 chain steps over 5 references needed work_assign, 2 needed issue_order (plus one unmatched), 1 needed scan.
- **Cost of changing later:** low while proposed; if the category enum gains `schedule` / `scan`, the three files change one field.
