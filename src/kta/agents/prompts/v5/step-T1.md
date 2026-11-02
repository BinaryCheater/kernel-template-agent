# Task T1: Build or extend a family

Task `$task_id` · work unit `$work_unit` · card `$card_id`

## Goal

$goal

## Where this task sits

KTA optimises one case (an operator, its shapes, a target GPU and a correctness contract) in
rounds. A round takes one **family** of this case — a chain of Templates with bindings, the
family's variables and constraints, plus a generator (`gen.py`) that turns one assignment of
the variables into a kernel — and then, with no agent involved:

1. assembles the family's **formulation** (variables, constraints and modelled cost) from the
   chain's Templates and the hardware numbers, and solves it per shape bucket as a MIP (finite
   menus one-hot, products of menu variables tabulated); a candidate a proof obligation rules
   out never reaches the compiler;
2. compiles the best-predicted candidates, runs a quick correctness check (it rejects only
   results that are clearly wrong) and measures the passing ones on the GPU;
3. judges the round by its gain per shape against the previous round and the strongest
   reference; a slower round is recorded, not an error.

Templates contribute decision variables and known tricks; they do not write the program.
The program is yours: `gen.py` must emit a correct kernel for every assignment the family
allows, and the family must say which Templates it realises and with which variables.

If the case has no family yet, your delivery is its first one (the baseline): a correct
kernel with whatever structure you judge sensible, described as a family so the solver can
search it. Otherwise the case header above gives what the last round measured per shape,
the reference, the lower bound (SOL), the model's prediction and the family it used.

## What to change

- Make the change you judge most worth measuring. Several related mechanisms at once are
  fine (a deeper pipeline with the shared memory and swizzle it needs, for example); a
  single small change (one variable's range) only when you think it is really the better
  move. Avoid circling on small points round after round.
- Start from where the time goes: the shape with the largest gap, what the kernel spends it
  on (memory traffic, a pipeline that does not overlap, too few blocks per SM, an
  instruction choice), and the Templates that address it (skill `find-knowledge`;
  `kta query tmpl`, `kta query chain --case <case>` for the Template chains of reference
  kernels, `kta query pool --case <case>`, `kta why <id>`). Nothing is preselected for you:
  which Templates, which chain and which search space is your choice.
- Keep the family general over its variables: the solver chooses the values, not you.
- Prefer Templates whose scope holds on this case and whose assurance is higher; a step whose
  scope or proof is open is allowed, but say so in the delivery note.
- TileLang reference: the TileLang package source used by the build worker is under
  `refs/tilelang/` in your workspace when present; read it as documentation.

## Deliver

A family at the root of `out/`: `out/family.yaml`, `out/chain.yaml`, `out/gen.py` (plus any
code markers), and `out/DELIVERY.md`.

- When the case has a family, `out/` already holds it: edit it in place, keep its `name` and
  raise `version` by one. The system records the diff against the previous version; do not
  rewrite files you do not change.
- `kta validate-family out` must pass; answer every warning in the delivery note.
- `kta instantiate`, `kta build` and `kta sass` work only on families already in the store:
  use them to read what the current generator emits, not to search values.
- `DELIVERY.md` under `## Done`, short: what you changed, which shape(s) it should help, why
  (cite the record IDs you used), and the effect you expect.

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
