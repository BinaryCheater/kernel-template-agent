> Historical first-pass record. Current status and dispositions are in study.md, inventory.md and designs.md.

# quack source study

Assignment: Worker C, QuACK → apex → Mamba → causal-conv1d. No shared-file edits or commits.

Source: https://github.com/Dao-AILab/quack, revision `bb0e6b220f6b9ee6b36a638d3d6c6150d52db8d8`. [Retained inputs](inputs/sources.json). License: upstream LICENSE files.

Scope: distinct NVIDIA kernel designs across the library, representative reading and variant differences. Stop after assigned queue.

Pool: repository seed versions and empty reviewed.yaml read 2026-11-04; study proposals provisional.

Status: representative extraction pass closed; one candidate with paired outline. Quantization/layout/scheduler gaps named in inventory and trace; pending independent review.

License files inspected: LICENSE.

Final checks: [bounded structural replay and schema validation](checks/replay.md). No commits, GPU experiments, gate run or shared-index edits were part of this source-study assignment.


Continuation status (2026-11-04): reopened after depth review. Prior closure was premature: major distinct designs require concrete representative traces and field-level bindings. Existing proposals and checks are retained. Queue order remains QuACK → apex → Mamba → causal-conv1d.
