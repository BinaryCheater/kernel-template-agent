# Witness results

Command: `.venv/bin/python knowledge/studies/fuser/checks/replay.py`

Observed: `PASS: 11826 phase arrival orders; four non-power-of-two Y-reduction segments`.

Domain: all arrival permutations for 1..7 CTAs, initial phase zero or high bit set; four independent three-element reduction segments. Covers phase_join arithmetic and the Y-stride example. Native visibility, scheduler progress and generated CUDA behavior remain open. These are source-explanation checks, not GPU measurements.

Continuation command: `.venv/bin/python knowledge/studies/fuser/checks/design_witnesses.py`.
Observed: `PASS: two padded digit-layout bijections and 2/4/8-lane reduce-scatter membership`.
This checks the new digit address inverse and component exact-once ownership using distinct coordinate tokens; it does not simulate GPU execution or establish device visibility.
