> Historical first-pass record. Current status and dispositions are in study.md, inventory.md and designs.md.

# causal-conv1d source study

Assignment: Worker C, QuACK → apex → Mamba → causal-conv1d. No shared-file edits or commits.

Source: https://github.com/Dao-AILab/causal-conv1d, revision `cd81f0413cad2fc1e6f17e785ac39f59aae690cd`. [Retained inputs](inputs/sources.json). License: upstream LICENSE files.

Scope: distinct NVIDIA kernel designs across the library, representative reading and variant differences. Stop after assigned queue.

Pool: repository seed versions and empty reviewed.yaml read 2026-11-04; study proposals provisional.

Status: representative extraction pass closed; one candidate with paired outline. Vector-coordinate and state-gradient gaps named. Worker C queue exhausted; stop here.

License files inspected: LICENSE.

Final checks: [bounded structural replay and schema validation](checks/replay.md). No commits, GPU experiments, gate run or shared-index edits were part of this source-study assignment.


Continuation status (2026-11-04): reopened after depth review. Prior closure was premature: major distinct designs require concrete representative traces and field-level bindings. Existing proposals and checks are retained. Queue order remains QuACK → apex → Mamba → causal-conv1d.
