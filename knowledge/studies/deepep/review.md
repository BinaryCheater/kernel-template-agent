# DeepEP: fanout and issue batching

| Kernel need | Proposed rule | Useful choice |
|---|---|---|
| Several expert recipients share a transport locality | [destination_quotient_transfer@1](delta/destination_quotient_transfer@1/template.yaml) | Send one payload per locality, then expand using preserved expert metadata |
| Aggregated sparse requests need a final doorbell | [partial_batch_issue_retention@1](delta/partial_batch_issue_retention@1/template.yaml) | Retain one full batch and issue it explicitly at end |

Worked composition: expert IDs[1,2,6] in two four-expert localities require two remote payloads and three expert deliveries. Combining expert outputs still needs three weighted contributions; deduplicating those by locality is invalid.

Status: partial; paused by user direction. Additional distinct families remain to be read in [inventory](inventory.md). [Trace](trace.md) records direct/hybrid, notify, Engram and bucket distinctions. [Replay](checks/replay.json) checks finite incidence and queue witnesses; [script](checks/replay.py) is reproducible. Source is retained in [manifest](inputs/sources.json). Native communication visibility/progress, full combine, PP and additional bucket schedules remain open. Two candidates are pending independent review; no measurements or admission.
