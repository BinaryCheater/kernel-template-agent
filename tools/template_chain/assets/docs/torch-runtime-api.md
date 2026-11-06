# Installed allocation and graph interfaces

## torch.empty

empty(*size, *, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False, pin_memory=False, memory_format=torch.contiguous_format) -> Tensor

Returns a tensor filled with uninitialized data. The shape of the tensor is
defined by the variable argument :attr:`size`.

.. note::
    If :func:`torch.use_deterministic_algorithms()` and
    :attr:`torch.utils.deterministic.fill_uninitialized_memory` are both set to
    ``True``, the output tensor is initialized to prevent any possible
    nondeterministic behavior from using the data as an input to an operation.
    Floating point and complex tensors are filled with NaN, and integer tensors
    are filled with the maximum value.

Args:
    size (int...): a sequence of integers defining the shape of the output tensor.
        Can be a variable number of arguments or a collection like a list or tuple.

Keyword args:
    out (Tensor, optional): the output tensor.
    dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
        Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
    layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
        Default: ``torch.strided``.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, uses the current device for the default tensor type
        (see :func:`torch.set_default_device`). :attr:`device` will be the CPU
        for CPU tensor types and the current CUDA device for CUDA tensor types.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    pin_memory (bool, optional): If set, returned tensor would be allocated in
        the pinned memory. Works only for CPU tensors. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.contiguous_format``.

Example::

    >>> torch.empty((2,3), dtype=torch.int64)
    tensor([[ 9.4064e+13,  2.8000e+01,  9.3493e+13],
            [ 7.5751e+18,  7.1428e+18,  7.5955e+18]])

## torch.empty_like

empty_like(input, *, dtype=None, layout=None, device=None, requires_grad=False, memory_format=torch.preserve_format) -> Tensor

Returns an uninitialized tensor with the same size as :attr:`input`.
``torch.empty_like(input)`` is equivalent to
``torch.empty(input.size(), dtype=input.dtype, layout=input.layout, device=input.device)``.

.. note::
    If :func:`torch.use_deterministic_algorithms()` and
    :attr:`torch.utils.deterministic.fill_uninitialized_memory` are both set to
    ``True``, the output tensor is initialized to prevent any possible
    nondeterministic behavior from using the data as an input to an operation.
    Floating point and complex tensors are filled with NaN, and integer tensors
    are filled with the maximum value.

    When ``torch.preserve_format`` is used:
    If the input tensor is dense (i.e., non-overlapping strided),
    its memory format (including strides) is retained.
    Otherwise (e.g., a non-dense view like a stepped slice),
    the output is converted to the dense format.

Args:
    input (Tensor): the size of :attr:`input` will determine size of the output tensor.

Keyword args:
    dtype (:class:`torch.dtype`, optional): the desired data type of returned Tensor.
        Default: if ``None``, defaults to the dtype of :attr:`input`.
    layout (:class:`torch.layout`, optional): the desired layout of returned tensor.
        Default: if ``None``, defaults to the layout of :attr:`input`.
    device (:class:`torch.device`, optional): the desired device of returned tensor.
        Default: if ``None``, defaults to the device of :attr:`input`.
    requires_grad (bool, optional): If autograd should record operations on the
        returned tensor. Default: ``False``.
    memory_format (:class:`torch.memory_format`, optional): the desired memory format of
        returned Tensor. Default: ``torch.preserve_format``.

Example::

    >>> a=torch.empty((2,3), dtype=torch.int32, device = 'cuda')
    >>> torch.empty_like(a)
    tensor([[0, 0, 0],
            [0, 0, 0]], device='cuda:0', dtype=torch.int32)

## torch.cuda.CUDAGraph

Wrapper around a CUDA graph.

Arguments:
    keep_graph (bool, optional): If ``keep_graph=False``, the
        cudaGraphExec_t will be instantiated on GPU at the end of
        ``capture_end`` and the underlying cudaGraph_t will be
        destroyed. Users who want to query or otherwise modify the
        underlying cudaGraph_t before instantiation can set
        ``keep_graph=True`` and access it via ``raw_cuda_graph`` after
        ``capture_end``. Note that the cudaGraphExec_t will not be
        instantiated at the end of ``capture_end`` in this
        case. Instead, it will be instantiated via an explicit called
        to ``instantiate`` or automatically on the first call to
        ``replay`` if ``instantiate`` was not already called. Calling
        ``instantiate`` manually before ``replay`` is recommended to
        prevent increased latency on the first call to ``replay``. It
        is allowed to modify the raw cudaGraph_t after first calling
        ``instantiate``, but the user must call ``instantiate`` again
        manually to make sure the instantiated graph has these
        changes. Pytorch has no means of tracking these changes.

.. warning::
    This API is in beta and may change in future releases.

## torch.cuda.graph

Context-manager that captures CUDA work into a :class:`torch.cuda.CUDAGraph` object for later replay.

See :ref:`CUDA Graphs <cuda-graph-semantics>` for a general introduction,
detailed use, and constraints.

Arguments:
    cuda_graph (torch.cuda.CUDAGraph): Graph object used for capture.
    pool (optional): Opaque token (returned by a call to :func:`~torch.cuda.graph_pool_handle()` or
        :meth:`other_Graph_instance.pool()<torch.cuda.CUDAGraph.pool>`) hinting this graph's capture
        may share memory from the specified pool. See :ref:`Graph memory management<graph-memory-management>`.
    stream (torch.cuda.Stream, optional): If supplied, will be set as the current stream in the context.
        If not supplied, ``graph`` sets its own internal side stream as the current stream in the context.
    capture_error_mode (str, optional): specifies the cudaStreamCaptureMode for the graph capture stream.
        Can be "global", "thread_local" or "relaxed". During cuda graph capture, some actions, such as cudaMalloc,
        may be unsafe. "global" will error on actions in other threads, "thread_local" will only error for
        actions in the current thread, and "relaxed" will not error on actions. Do NOT change this setting
        unless you're familiar with `cudaStreamCaptureMode <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85>`_

.. note::
    For effective memory sharing, if you pass a ``pool`` used by a previous capture and the previous capture
    used an explicit ``stream`` argument, you should pass the same ``stream`` argument to this capture.

.. warning::
    This API is in beta and may change in future releases.

.. _cudaStreamCaptureMode:
    https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
