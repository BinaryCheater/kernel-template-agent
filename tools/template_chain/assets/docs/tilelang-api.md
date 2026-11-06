# Installed TileLang API docstrings

Version: 0.1.15

## T.Kernel

Construct a kernel launch frame for CUDA: a grid of thread blocks.

Code inside the launch operates at the block level: ``T.Parallel``,
``T.copy`` and friends are mapped onto threads by the compiler.
``T.get_thread_binding()`` exposes ``threadIdx`` for thread-level code.
The keyword arguments are recorded at trace time and materialized by the
CUDA pipeline once the target is known.

Parameters
----------
*blocks : int | PrimExpr
    Grid extent along each axis (1-3 dimensions, ``gridDim.(x|y|z)``). The
    launch yields one block index per axis (``blockIdx.(x|y|z)``).
threads : int | list[int] | tuple[int, ...], optional
    Threads per block: a count for ``blockDim.x`` or up to three
    per-dimension extents for ``blockDim.(x|y|z)``. Defaults to 128 when
    omitted.
prelude : str, optional
    CUDA source injected before the generated kernel, e.g. ``#include``
    lines or helper functions.
cluster_dims : int | tuple[int, int, int] | list[int], optional
    Thread block cluster shape (SM90+). ``2`` or ``(2, 1, 1)`` launches
    2-CTA clusters via ``cudaLaunchKernelEx``. ``T.ClusterKernel`` is the
    same launch with a required ``cluster_dims``.

Examples
--------
.. code-block:: python

    with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
        ...

    with T.Kernel(grid_x, grid_y, threads=(64, 2)) as (bx, by):
        tx, ty = T.get_thread_bindings()
        ...

## T.Parallel

Construct a nested parallel loop, with CUDA lowering hints.

Same semantics as the common :func:`tilelang.language.loop.Parallel`.
``prefer_async`` requests the PTX cp.async rewrite for copies in this loop
subtree even outside pipelined loops; it is a performance hint ignored by
targets without async copy.

Parameters
----------
extents : int | PrimExpr
    Extents of the parallel loop nest.
coalesced_width : Optional[int]
    Width for coalesced memory access.
loop_layout : Optional[Fragment]
    Layout annotation for the parallel loop nest.
prefer_async : Optional[bool]
    When True, requests cp.async injection for this subtree; when False,
    forbids it. Lowered as the ``"parallel_prefer_async"`` annotation.
annotations : Optional[Dict[str, Any]]
    Additional loop annotations; values in it take precedence.

## T.serial

The serial For statement.

Parameters
----------
start : PrimExpr
    The minimum value of iteration.

stop : PrimExpr
    The maximum value of iteration.

step : PrimExpr
    The step size of the iteration.

annotations : Dict[str, Any]
    The optional annotations of the For statement.

Returns
-------
res : frame.ForFrame
    The ForFrame.

## T.Pipelined

Tools to construct pipelined for loop.

Parameters
----------
start : PrimExpr
    The minimum value of iteration.
stop : PrimExpr
    The maximum value of iteration.
num_stages : int
    The max number of buffer used between pipeline producers and consumers.
    For compiler-inferred pipelines, if ``num_stages`` is 0, pipeline will
    not be enabled. For manually scheduled pipelines, prefer specifying
    ``order`` and ``stage`` without ``num_stages``; the pipeline depth is
    inferred as ``max(stage) + 1``.
order : Optional[List[int]]
    Optional manual emission order for scheduled statements in the loop
    body. This list should describe executable pipeline statements such as
    copies, fills, GEMMs, reductions, stores, waits, or commits.
stage : Optional[List[int]]
    Optional manual pipeline stage for each scheduled statement in the loop
    body. The list is aligned with ``order`` and follows the same statement
    counting rule.
annotations : Optional[Dict[str, Any]]
    Additional loop annotations.

Notes
-----
Replayable scalar ``Bind`` statements created by local aliases are not
scheduled pipeline operations and should not consume entries in ``order``
or ``stage``. For example, in the body below only the copy and store need
annotation entries; ``base`` is replayed automatically at each use:

.. code-block:: python

    for i in T.Pipelined(n, order=[1, 0], stage=[0, 1]):
        base: T.int32 = i * block
        T.copy(A[base], shared)
        T.copy(shared, B[base])

A replayable ``Bind`` may also read a buffer that is not written by the
pipeline body, such as ``idx = Ids[i]``. If a ``Bind`` reads a buffer that
is written inside the same pipeline body, it is kept as a scheduled
statement because the load has a pipeline dependency and cannot be freely
replayed.

Older code that included replayable scalar ``Bind`` entries in
``order``/``stage`` is accepted for compatibility, but those entries are
ignored by the pipeline passes. New code should annotate only the scheduled
statements. Avoid setting ``num_stages`` together with manual
``order``/``stage`` unless you intentionally need an explicit depth
override.
Returns
-------
res : frame.ForFrame
    The ForFrame.

## T.alloc_shared

Allocate a shared memory buffer for inter-thread communication.

Args:
    shape (tuple): The shape of the buffer to allocate
    dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
    scope (str, optional): The memory scope. Defaults to "shared.dyn"

Returns:
    T.Buffer: A TVM buffer object allocated in shared memory

## T.alloc_fragment

Allocate a fragment memory buffer for specialized operations.

Args:
    shape (tuple): The shape of the buffer to allocate
    dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
    scope (str, optional): The memory scope. Defaults to "local.fragment"

Returns:
    T.Buffer: A TVM buffer object allocated in fragment memory

## T.alloc_local

Allocate a local memory buffer for thread-private storage.

Args:
    shape (tuple): The shape of the buffer to allocate
    dtype (str): The data type of the buffer (e.g., 'float32', 'int32')
    scope (str, optional): The memory scope. Defaults to "local"

Returns:
    T.Buffer: A TVM buffer object allocated in local memory

## T.copy

Copy data between memory regions, with CUDA lowering hints.

Same semantics as the common :func:`tilelang.language.copy_op.copy`; the
extra keywords steer how the CUDA backend lowers the copy. They are
performance hints recorded on the tile op: compiling the same kernel for a
target that has no use for them leaves the result unchanged.

Args:
    src: Source memory region (Buffer, BufferLoad or BufferRegion).
    dst: Destination memory region.
    coalesced_width (Optional[int], keyword-only): Width for coalesced
        memory access. Defaults to None.
    disable_tma (bool, keyword-only): Never lower this copy through TMA
        even when the shape and scopes qualify. Defaults to False.
    eviction_policy (Optional[str], keyword-only): L2 cache eviction
        priority for the generated load/store or TMA instruction, one of
        ``"evict_normal"``, ``"evict_first"``, ``"evict_last"``.
    prefer_instruction (Optional[str], keyword-only): Preferred lowering
        instruction category: ``"tma"``, ``"cp_async"`` or ``"sync"``. For
        ``"tma"``, T.copy keeps synchronous copy semantics; global ->
        shared copies lower through TMA with an automatically allocated
        barrier and wait when constraints are satisfied.
    annotations (Optional[dict], keyword-only): Additional annotations
        dict; values in it take precedence over the individual keywords.
    loop_layout (Optional[Fragment], keyword-only): Parallel loop layout
        hint for the SIMT copy path.

Returns:
    tirx.Call: A handle to the copy operation.

## T.gemm

TileLang GEMM operator for CUDA.

Same semantics as the common :func:`tilelang.language.gemm_op.gemm`: the
default synchronous GEMM. On Hopper, if the compiler selects WGMMA
lowering, TileLang inserts the corresponding wait implicitly. On Blackwell
TCGEN5MMA, TileLang inserts the corresponding
``mbarrier_wait_parity(...)`` implicitly after issue.

For manual asynchronous scheduling, use ``T.wgmma_gemm(...)`` with
``T.wait_wgmma(...)`` on Hopper, or ``T.tcgen05_gemm(...)`` with
``T.mbarrier_wait_parity(...)`` on Blackwell.

Args:
    A (BufferLikeType, i.e. Buffer | BufferLoad | BufferRegion, or Var): Input buffer A.
    B (BufferLikeType): Input buffer B.
    C (BufferLikeType): Output buffer C.
    transpose_A (bool): Whether to transpose A. Defaults to False.
    transpose_B (bool): Whether to transpose B. Defaults to False.
    policy (GemmWarpPolicy): GEMM warp partition policy.
    clear_accum (bool): Whether to clear the accumulator.
    mbar (BarrierType, i.e. Buffer | BufferLoad, or Var, optional): Mbarrier in Blackwell.
        Required when this GEMM lowers to TCGEN5MMA. Defaults to None.
    annotations (Optional[dict]): Additional annotations.

Returns:
    tirx.Call: A handle to the GEMM operation.

## T.reduce_max

Perform reduce max, with the CUDA NaN-propagation knob.

Same semantics as the common :func:`tilelang.language.reduce_op.reduce_max`.
``nan_propagate=True`` requires float16 or bfloat16 output. Other output
dtypes raise a compilation error. With NaN propagation enabled, the
reduction lowers to ``__hmax_nan`` so NaNs propagate; when False (default)
``__hmax`` returns the non-NaN operand. Targets without these intrinsics
reject the annotation at compile time.

## T.reduce_sum

Perform reduce sum on input buffer, store the result to output buffer.

Args:
    buffer (tirx.Buffer): The input buffer
    out (tirx.Buffer): The output buffer
    dim (int): The dimension to perform reduce on
    clear (bool, optional): If True, output buffer will be cleared before reduction.
                          If False, results will be accumulated on existing values.
                          Defaults to True.
    batch (int): Number of output elements per batched AllReduce call (default 1).
    annotations (dict, optional): On CUDA SM100+, set
        ``{"enable_fadd2": False}`` to disable packed FP32x2 accumulation
        for this reduction. It is enabled by default, unless the
        ``tl.enable_fp32x2_reduction`` pass config is False.
Note: When clear=True, reduce_sum will not compute directly on the output buffer. This is because
      during warp reduction, the same value would be accumulated multiple times (number of threads
      in the warp). Therefore, the implementation with clear=True follows these steps:
    1. create a temp buffer with same shape and dtype as out
    2. copy out to temp buffer
    3. call reduce_sum with temp buffer and out
    4. Add temp buffer to out

Returns:
    tirx.Call: Handle to the reduction operation

## T.fill

Fill a buffer or buffer region with a specified value.

Args:
    buffer: Either a TVM buffer or buffer region to be filled
    value: The value to fill the buffer with
    annotations: Optional annotations to attach to the fill call

Returns:
    A TVM intrinsic call that performs the fill operation

## T.clear

Clear a buffer by filling it with zeros.

Args:
    buffer: Either a TVM buffer or a variable that contains a buffer region
    annotations: Optional annotations forwarded to the underlying fill

Returns:
    A fill operation that sets the buffer contents to zero

Raises:
    ValueError: If the buffer variable contains an invalid buffer region

## T.exp

Take exponential of input x.

Parameters
----------
x : PrimExpr
    Input argument.

Returns
-------
y : PrimExpr
    The result.

## T.exp2

Calculate 2**x

Parameters
----------
x : PrimExpr
    Input argument.

Returns
-------
y : PrimExpr
    The result.

## T.log

Take log of input x.

Parameters
----------
x : PrimExpr
    Input argument.

Returns
-------
y : PrimExpr
    The result.

## T.cast

Cast ``value`` to ``dtype`` with optional PTX-style rounding hints.

Parameters
----------
value : object
    The source operand.
dtype : str
    The target data type.
round : str, optional
    PTX rounding modifier (e.g. ``"rn"``, ``"rz"``, ``"rp"``, ``"rm"``,
    ``"rs"``). Empty string means use the backend default. Currently
    only ``""`` and ``"rs"`` are lowered by the CUDA backend; the other
    modifiers are reserved.
sat : bool, optional
    Saturate to finite (``True`` = PTX ``.satfinite``, default).
rbits : PrimExpr, optional
    Random bits operand for stochastic rounding (``round="rs"``).
span : Optional[Span]
    The location of this operator in the source.

Returns
-------
op : tvm.tir.PrimExpr
    The cast expression.

## T.annotate_layout

Annotate the layout of the buffer.

## T.use_swizzle

Annotate a kernel to use a specific threadblock swizzle pattern.

## T.infinity

infinity value of dtype

Parameters
----------
dtype : str
    The data type.

span : Optional[Span]
    The location of this operator in the source code.

Returns
-------
value : tvm.Expr
    The infinity value of dtype.

## T.if_then_else

Conditional selection expression.

Parameters
----------
cond : PrimExpr
    The condition

t : PrimExpr
    The result expression if cond is true.

f : PrimExpr
    The result expression if cond is false.

span : Optional[Span]
    The location of this operator in the source.

Returns
-------
result : Node
    The result of conditional expression.

Note
----
Unlike Select, if_then_else will not execute
the branch that does not satisfy the condition.
You can use it to guard against out of bound access.
Unlike Select, if_then_else cannot be vectorized
if some lanes in the vector have different conditions.

## tilelang.jit

JIT compiler decorator for TileLang functions.

Supports two execution modes (automatically inferred):
- **lazy**: Function returns PrimFunc explicitly. Returns compiled kernel object.
- **eager**: Function uses DSL builder pattern. Executes kernel immediately.

Parameters
----------
out_idx : list[int] | int | None
    Output tensor index(es). Only supported in lazy mode.
target : str | tvm.target.Target | None
    Compilation target (e.g., "cuda", "llvm", "auto").
target_host : str | tvm.target.Target | None
    Host target for cross-compilation.
execution_backend : ExecutionBackend | None
    Backend for kernel execution.
verbose : bool | None
    Enable verbose compilation output.
pass_configs : dict[str, Any] | None
    TVM pass configuration options.
debug_root_path : str | None
    Directory to save compiled kernel source for debugging.
compile_flags : list[str] | str | None
    Additional compiler flags.
