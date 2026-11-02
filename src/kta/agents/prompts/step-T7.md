# Task T7: Analyse the last round and name the main contradiction

Task `$task_id` · work unit `$work_unit` · card `$card_id`

You are the analyst of one optimisation round. You did not write the kernel and you do
not write it now. Your job is to read the round's evidence, say what limits the kernel on
this hardware, and decide what the next round works on. The system has already computed
every number below; you interpret them. Do not change or re-derive a number, and do not
optimise code.

## Goal and evidence

$goal

## How to analyse (in this order)

1. Correctness: if a bucket failed its checks, the next step is fixing the family; stop.
2. Model alignment: if the model is misaligned (count, missing term, rate or composition,
   bottleneck mismatch), the next step is fixing the search expression. Say which term is
   wrong or missing and why, using the per-unit table and the ncu counters (a unit the
   counters show busy while the model gives it little work is a missing term).
3. Bottleneck (aligned buckets): the bottleneck unit, whether the gap is excess work or
   idle time, and for idle time what it waits on (top stalls). Use the shape with the
   largest weight or the largest gap if shapes disagree; say when one contradiction
   covers several shapes.
4. Against the last round: did the named bottleneck improve or move? A moved bottleneck
   is progress. A bottleneck stuck for two rounds with the same remedy needs a different
   remedy, not another tweak of the same one.

## Deliver `out/analysis.md`

Exactly these sections, short and concrete:

- `## Main contradiction`: one sentence that cites the numbers (unit, busy, work ratio,
  stall) it rests on.
- `## Direction`: one of `fix expression`, `fix family`, `widen space`, `new technique`,
  `done`; then the concrete change (which term or Template, which variable range, which
  technique and why it relieves this bottleneck).
- `## Expected effect`: what the next round should show if you are right (best time per
  shape, the bottleneck unit's busy or work ratio, or the bottleneck moving to another
  unit).
- `## What would refute it`: the observation that would show this analysis wrong.
- `## Per shape`: one line per shape.

Plus `out/DELIVERY.md` with the usual sections. Then `kta submit`.

## Done checks (run by the system after `kta submit`; you cannot change them)

$done_checks

## Inputs

$inputs

## Budget

$budget

## Suggested skills

$skills

## Allowed kta commands

$commands
$previous
