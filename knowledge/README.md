# Finding and reusing Templates

Start here when analyzing another kernel. The shared identity of a rule is its mechanism name and version. A Chain records that identity and its variable bindings. Source and hardware names belong on instances and Realisations unless the mechanism itself requires a particular capability.

Browse the [Humming candidate rules by kernel need](studies/humming/review.md).

1. For a live KTA study, query the starting Store snapshot by refined pattern, scope and target capability. Record the returned versions and assurance. The repository's seed inputs live in [templates](seed/templates/) with available [proof outlines](seed/proofs/). [Reference packages](seed/references/) also contain proposed rules; use the versions present in the selected snapshot.
2. For a repository-only study, start with the declared seed baseline and the [reviewed study index](studies/reviewed.yaml). Record the chosen baseline and index versions in study.md. Search by refined pattern, introduced object and workload/capability conditions. Follow the linked YAML, proof and source trace. Check bindings and scope on the concrete case before reusing a rule. Review-pending candidates can be located under studies/*/delta/ to detect duplicate work; mark any provisional dependency explicitly.
3. Author new candidates at studies/<library>/delta/<name>@<version>/template.yaml with proof.yaml beside it. Keep source bindings and chain in the study. Link rules across studies by identity; one YAML pair can serve multiple kernels.
4. After independent review, add one entry per accepted version to studies/reviewed.yaml with name, version, refines, introduced object, concise scope, and repository-relative paths to the Template, outline and review. Check identity uniqueness, paths, schema and replay evidence. An identity with different content requires reconciliation or a new version. Keep superseded entries for old chains. Repository review is separate from live Store admission and proof assurance.

The index is a navigation manifest for file-based work. Live retrieval uses admitted KTA records. Editing the manifest alone does not import them.
