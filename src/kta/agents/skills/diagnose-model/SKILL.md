---
name: diagnose-model
description: How to explain why the cost model mispredicts measured kernels — pair predictions with measurements, locate the error by unit and by regime, propose a revised model with its evidence, and check it on points you did not fit. Used for task T4.
---

# Diagnosing the cost model

The cost model predicts a candidate's time from per-unit work (bytes on DRAM and L2,
shared-memory wavefronts, tensor and FMA operations, issue slots, launches) divided by
each unit's peak, an efficiency per unit, a serial term and a few pairwise interaction
terms. A T4 task gives you the current model version and (Prediction, Measurement) pairs
where it is wrong. Your job is to say *why*, and to deliver a revised model that is
better on points it was not fitted on.

## Inputs

`TASK.md` lists the model version first, then Prediction and Measurement records in
pairs (each Prediction is followed by the Measurement of the same candidate). Read them
with `kta show`; `kta why <measurement>` gives the build, check and protocol behind a
number. Never use a Measurement taken under another protocol in the same comparison.

## Method

1. **Tabulate.** One row per pair: predicted and measured time, signed relative error
   `(pred − meas) / meas`, the per-unit predicted times, which unit the model calls the
   bottleneck, and the candidate's assignment and shape.
2. **Locate the error.** Group rows by bottleneck unit, by family variable values and by
   shape regime (for example working set below, near and above a cache capacity, waves
   below and above one). An error that is uniform within a group points to that unit's
   efficiency; one that changes sign across a boundary points to a missing regime.
3. **Name the cause.** Pick the smallest explanation the rows support: a wrong
   efficiency, a missing serial or launch term, an interaction between two units that
   overlap less than assumed, a work count that is wrong for this family (then the cost
   term of a Template is wrong: say which one and why), or a hardware fact the model
   lacks (then say which microbenchmark would measure it; that is a T5, not your task).
4. **Revise.** Change only what the cause requires. Keep the model's form: efficiencies
   per unit, the serial coefficient and constant, interaction gains with their basis.
   Record every change with the rows that support it.
5. **Hold out.** Before submitting, split the pairs: fit or tune on some, and compare
   the old and the new model on the rest. The done-check does the same on the pairs in
   the card, so a change that only fits the points you tuned on will be rejected.

## Deliverable

- `out/model.json`: the revised model in the model's own dictionary form (the fields of
  the input version: `target`, `eff`, `serial_coef`, `serial_const_us`, `interactions`,
  `interaction_basis`, `occupancy_knee`, `parent` = the input version's ID, `fit_info`
  with your hold-out numbers). Omit `version`; it is computed from the parameters.
- `out/DELIVERY.md`: what was done (the table summary, the cause, each change and its
  evidence), simplifications (none, or which groups you could not explain), known
  issues (rows the new model still gets wrong, and why you think so).

## What not to do

- Do not tune parameters until the numbers fit without a cause you can state; an
  unexplained fit does not transfer to new cases.
- Do not drop pairs because they disagree; explain them or list them as unexplained.
- Do not change a Template here. If a cost term is wrong, say so in the delivery note;
  the system opens a separate task for it.
