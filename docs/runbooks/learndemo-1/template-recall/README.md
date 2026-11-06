# Template recall evaluation inputs

Inputs for the offline recall evaluation. Results are in
[template-recall-e2.md](../results/template-recall-e2.md).

- `tasks/<id>.md`: the only kernel text an agent or a retrieval query sees. A task
  names the operator contract and the target. It does not name the source library.
- `gold/<id>.yaml`: moves the source implementation actually uses, with provenance.
  Gold sets are never placed in a run package.

Each gold move has:

- `need`: the need stated without a rule name, for the need-to-rule retrieval check.
- `source_rule`: the identity the decomposition or trace cites.
- `accept`: catalog identities that are the same rule (same name, other version).
- `equivalent`: other catalog identities that express the same mechanism. This list is
  the author's judgement and is scored separately.
- `role`: `distinctive` when the move shapes this kernel's computation or schedule,
  `generic` when any GPU kernel of this class needs it.
- `provenance`: where the source decomposition or study records the move.

Authoring rule: gold sets were written from the source studies and committed before any
retrieval or agent run was scored. Changing one after a run needs a new file name.
