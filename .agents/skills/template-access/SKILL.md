---
name: template-access
description: Find optimisation rules in the file Template library by symptom, bottleneck resource, pattern or transformation, then open them one layer at a time. Use when choosing Templates for a kernel or formulation. Every read is logged per layer.
---

# Finding Templates by progressive disclosure

The library holds 312 proposed rules. Do not read it whole. Search by what you need,
open only what looks relevant, and go deeper only for rules you may use.

Run the tool as `scripts/template-access` in this repository, or `./template-access` in a
frozen run package. Replace `TA` below with that command.

## The four layers

| Layer | Command | What you get | Read it when |
| --- | --- | --- | --- |
| 1 Index card | `TA search`, `TA card <id>` | When, choice, boundary | Always start here |
| 2 GUIDE | `TA guide <id>` | Computation before and after, bindings, misuse | The card may fit |
| 3 Template with proof | `TA template <id>` | Variables, constraints, obligations, open assumptions | You intend to apply it |
| 4 Realisation with witness | `TA realisation <id>` | Backend code facts and evidence, or the recorded gap | You need to emit or check it |

## Searching

Describe the problem, not a rule name. Run several searches with different angles.

```sh
TA facets                                   # list tags with counts
TA search "row statistic needed before output, large key axis"
TA search --symptom shared-memory-capacity --resource registers
TA search "avoid storing scores" --pattern reduce
TA search --transformation "fold scale into exponent"
```

- `--symptom` and `--resource` filter when the value is a listed tag. Other words are
  searched as text.
- `--pattern` is exact: a category or a refined pattern.
- Tags come from a regular-expression table (`tools/template_access/lexicon.json`), so
  treat them as cues. An entry tagged `other` matched no rule.
- `--limit N` changes how many cards you see. The default is 15.

## Judging

A card is a summary. Before relying on a rule, check its GUIDE scope and the proof
outline's open assumptions. All entries have proposed status. An empty requires list does not
establish backend support. A gap printed by `TA realisation` means no backend mapping is
recorded for that version.

## Recording

`TA select <id> --reason "..."` records a rule you would apply. `TA need "..."` records a
need that no rule satisfied. Do not rewrite library files.

## The read log

Each read appends one JSON line to `$KTA_ACCESS_LOG`, default `access-log.jsonl` in the
current directory, with event, layer, identity, bytes and content hash. `--session` or
`$KTA_ACCESS_SESSION` labels the rows. `TA status` counts them. `TA gaps` counts missing
layer content across the library.

`--exclude-source <study>` hides rules whose source lineage includes that study, for
leave-one-source-out work. Frozen packages built by `tools/template_recall` already apply it.
