# Batch r2b: entry block

Supplementary to batch r2 (same round, same protocol). Reason: TileLang 0.1.15 orders a kernel's parameters by name, and r2's harness passed the buffers in the prim_func's order. HGEMM (A, B, C), FA (equal sizes) and softmax (X, Y) were unaffected; decode partial, decode combine, GEMV and RMSNorm got buffers of the wrong size (RMSNorm faulted; the others read out of bounds), so their r2 values are listed in `r2/invalid_cases.json` and not scored. The generator now passes buffers by parameter name. Only these kernels and the protocol kernels are built and run (`perfmodel-measure build r2b --only 'decode|gemv|rmsnorm'`).

No probe. The kernels, shapes, cache states and front end are those of r2's entry block ("Test set growth"); their replay predictions are in `predicted.json`, written before the run.
