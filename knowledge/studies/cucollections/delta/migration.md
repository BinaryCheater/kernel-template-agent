# Packed-claim scope widening

| Old field/obligation | New owner | Translation and replay |
|---|---|---|
| cooperative_probe_atomic_claim@1 group_size | @2 group_size | Identity |
| @1 slot_bytes4/8 | @2 slot_bytes4/8 | Identity; all old source bindings retained |
| @1 linearization,unique_key,coverage,publication | @2 same obligation names | Same logical slot claim; old proof premises retained |
| No16-byte domain | @2 slot_bytes16 and pre_wide | New scope requires packable representation, sm90+ and supported native128 CAS; not claimed from two64-bit stores |

Version1 remains available. The abstract compare-exchange collision witness applies unchanged to old widths;16-byte device lowering remains an open realization obligation. Split-slot publication is a different rule, not a wide packed realization.
