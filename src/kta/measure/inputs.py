"""Input generation shared by measurement and checks.

Tensor shapes in a Problem are expressions over the shape variables (``["M", "K"]``),
evaluated with ``kta.core.expr``. Distributions form an open table; checks adds its
operator-specific ones on top (``kta.checks.inputs``). Every generator takes a
``random.Random`` from ``ctx.rng(name)``, so a seed recorded with the result
reproduces the inputs.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np

from kta.core.errors import NotSupported
from kta.core.expr import Expr
from kta.core.schemas.evaluation import TensorSpec

DTYPES: dict[str, Any] = {
    "f16": np.float16,
    "float16": np.float16,
    "f32": np.float32,
    "float32": np.float32,
    "f64": np.float64,
    "float64": np.float64,
    "i32": np.int32,
    "int32": np.int32,
}
# Magnitude of the "extreme" entries: large, yet far from overflow in sums of a few
# hundred products of the dtype.
EXTREME_SCALE = {"f16": 16.0, "float16": 16.0, "f32": 1e3, "float32": 1e3}
EXTREME_FRACTION = 0.01

Generator = Callable[[tuple[int, ...], str, np.random.Generator], np.ndarray[Any, Any]]


def np_dtype(dtype: str) -> Any:
    if dtype not in DTYPES:
        raise NotSupported(f"dtype {dtype!r} has no host representation (bf16 not yet)")
    return DTYPES[dtype]


def dims(spec: TensorSpec, shape: dict[str, Any]) -> tuple[int, ...]:
    return tuple(int(Expr.parse(d, set(shape)).eval(shape)) for d in spec.shape)


def nbytes(spec: TensorSpec, shape: dict[str, Any]) -> int:
    return int(np.prod(dims(spec, shape), dtype=np.int64)) * np.dtype(np_dtype(spec.dtype)).itemsize


def _normal(d: tuple[int, ...], dtype: str, g: np.random.Generator) -> np.ndarray[Any, Any]:
    return g.standard_normal(d)


def _zeros(d: tuple[int, ...], dtype: str, g: np.random.Generator) -> np.ndarray[Any, Any]:
    return np.zeros(d)


def _uniform(d: tuple[int, ...], dtype: str, g: np.random.Generator) -> np.ndarray[Any, Any]:
    return g.uniform(-1.0, 1.0, d)


def _extreme(d: tuple[int, ...], dtype: str, g: np.random.Generator) -> np.ndarray[Any, Any]:
    x = g.standard_normal(d)
    mask = g.random(d) < EXTREME_FRACTION
    return np.where(mask, x * EXTREME_SCALE.get(dtype, 1e3), x)


BASE: dict[str, Generator] = {
    "normal": _normal,
    "zeros": _zeros,
    "uniform": _uniform,
    "extreme": _extreme,
}


def generate(
    spec: TensorSpec,
    shape: dict[str, Any],
    dist: str,
    rng: random.Random,
    table: dict[str, Generator] | None = None,
) -> np.ndarray[Any, Any]:
    gens = table if table is not None else BASE
    if dist not in gens:
        raise NotSupported(f"input distribution {dist!r} (known: {sorted(gens)})")
    g = np.random.default_rng(rng.getrandbits(63))
    return np.ascontiguousarray(gens[dist](dims(spec, shape), spec.dtype, g)).astype(
        np_dtype(spec.dtype)
    )
