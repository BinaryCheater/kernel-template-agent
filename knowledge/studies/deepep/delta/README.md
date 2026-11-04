# Proposed delta

- destination_quotient_transfer@1: one source payload per transport locality, retaining all logical deliveries. Distinct from sorted shared-index union: no sorted-list premise, and transport classes do not merge arithmetic contributions.
- partial_batch_issue_retention@1: leave a nonempty final request batch so aggregation ends with an explicit doorbell. Distinct from generic buffering: eager exact-width draining violates this protocol premise.

Both have paired ProofOutlines. Logical witnesses live in ../checks; native Gin ordering and progress remain open. No promotion requested.
