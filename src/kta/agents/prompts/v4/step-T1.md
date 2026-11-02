# Task T1: Build or extend a family

Task `$task_id` · work unit `$work_unit` · card `$card_id`

## Goal

$goal

## Where this task sits

KTA optimises one case (an operator, its shapes, a target GPU and a correctness contract) in
rounds. A round takes one **family** — a chain of Templates with bindings, plus a generator
(`gen.py`) that turns one assignment of the family's variables into a kernel — and then, with
no agent involved:

1. assembles the family's **formulation** (variables, constraints and modelled cost) from the
   chain's Templates and the hardware numbers;
2. solves it per shape bucket and keeps the best-predicted candidates; a candidate a proof
   obligation rules out never reaches the compiler;
3. compiles the best-predicted ones, runs a quick correctness check (it rejects only results
   that are clearly wrong), measures the passing ones on the GPU, plus random candidates as a
   control;
4. judges: the fastest measured candidate of each bucket gets a strict full check, and the
   round is compared with the best earlier round (the incumbent) and the strongest reference.

You are here because judging could not accept the round. The case header above gives what the
round measured for each shape, the reference, the lower bound (SOL), the model's prediction and
the family it used. Your delivery becomes the family the next round solves first; the incumbent
stays as the baseline it is compared against.

## What to change

- Change **one mechanism** per delivery (one Template step added, replaced or re-bound, or one
  generator change). A gain is credited to the mechanism that changed; two changes at once
  cannot be told apart, and a round that is slower than the incumbent is rolled back.
- Start from the shape with the largest gap to its reference. Ask what the kernel spends that
  time on (memory traffic, a pipeline that does not overlap, too few blocks per SM, an
  instruction choice) and look for Templates that address it (skill `find-knowledge`;
  `kta query tmpl`, `kta query pool --case <case>`, `kta why <id>`).
- Keep the family general over its variables: the solver chooses the values, not you. Widen
  a domain or add a variable when the best value may lie outside what the family allows now.
- Prefer Templates whose scope holds on this case and whose assurance is higher; a step whose
  scope or proof is open is allowed, but say so in the delivery note.

## Deliver

A family at the root of `out/`: `out/family.yaml`, `out/chain.yaml`, `out/gen.py` (plus any
code markers), and `out/DELIVERY.md`.

- Copy the current family from `inputs/` and edit it; keep its `name` and raise `version` by
  one, so the lineage stays visible.
- `kta validate-family out` must pass; answer every warning in the delivery note.
- `kta instantiate`, `kta build` and `kta sass` work only on families already in the store
  (the current one): use them to read what the current generator emits, not to search values.
- In `DELIVERY.md` under `## Done`: the one mechanism you changed, which shape(s) it should
  help, why (cite the record IDs you used), and the effect you expect.

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
