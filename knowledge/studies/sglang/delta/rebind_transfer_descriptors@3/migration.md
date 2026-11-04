# Version2 to3 translation

All variables, constraints, preconditions, program, effects and obligations retain their meaning. Every version2 binding maps identically to version3. Only descriptor_count domain upper bound changes from5 to6. Old identities and their source instances remain at knowledge/studies/cutlass/delta/rebind_transfer_descriptors@2; no file there is modified.

New binding: persistent KDA input tuple descriptor_count6, update_site=direct_global, publisher_tuple_size6. CTA-linear index selects private slots0..5 forAB,KS,V,QS,AQC,KG; output has independent slot6, descriptor_count1 and its own store issuer. Bounded tensors set sequence end before update; first use fences the updated descriptor. Native old-reader retirement and TensorMapManager release/acquire semantics remain open implementation premises. Source: kda_nvidia_prefill/fuse_k4_only_persistent.py:199–238,680–775,902–945.

The source tuple could not satisfy version2's declared maximum5. No previous proof is invalidated by keeping version2; new six-descriptor uses must refer to version3 and discharge the same four obligations. No admission requested.
