# Version1 to version2

Version1 remains addressable and unchanged. Its source/check bindings do not move.

| Old field or obligation | New owner/translation | Replay |
|---|---|---|
| descriptor_count, constraints, preconditions and scope predicate | Identical fields; add update_site=shared_shadow | Re-evaluate old scope/constraints and both storage amounts at counts1..5. |
| Shared shadow update then retire/publish/acquire | shared_shadow branch preserves ordering/mapping | Existing grouped/K0 metadata witness stays inside its original abstract scope. |
| descriptor_shadow bytes | Same expression, conditional on shared_shadow | Equal old/new projected bytes; direct_global allocates no shadow. |
| descriptor_published bytes | Same expression for either mode | Private global slots remain live. |
| All four obligations | Same owners; publication also covers direct release-fence | Direct path requires retirement before first store. Native ordering remains open. |

The new mode intentionally extends the implementation domain. No old proof is made
to establish it. Both source K branches provide the shared initialization partner; native
initialization progress remains a source-binding obligation in both versions.
