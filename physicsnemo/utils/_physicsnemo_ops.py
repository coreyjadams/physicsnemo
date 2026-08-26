# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional-dependency gate for the ``physicsnemo_ops`` accelerated kernels.

`physicsnemo-ops <https://github.com/NVIDIA/physicsnemo-ops>`_ provides
CPU (OpenMP) and CUDA kernels for segment reductions, bounded-integer row
deduplication/lookup, and spatial neighbor searches, registered as
``torch.library`` custom ops. It is an optional dependency: every call site in
physicsnemo that uses it must keep a pure-torch fallback.

This module is the single place that decides whether those kernels are used:

- :func:`physicsnemo_ops_torch` returns the ``physicsnemo_ops.torch`` module
  when it is importable and not disabled, else ``None``. The result is
  resolved once and cached in a module global, so calls inside hot loops (or
  ``torch.compile`` regions) reduce to a constant lookup.
- :func:`physicsnemo_ops_for` additionally applies the device policy: the
  accelerated paths engage for CUDA tensors only. Benchmarking on x86 showed
  the OpenMP CPU kernels losing 2-3x to torch's radix-pack + ``torch.unique``
  and ``scatter_add_`` compositions at mesh-typical sizes, so CPU tensors
  keep the pure-torch paths unless
  ``PHYSICSNEMO_PHYSICSNEMO_OPS_CPU=1`` opts in (used by parity tests).
- Setting the environment variable ``PHYSICSNEMO_DISABLE_PHYSICSNEMO_OPS=1``
  before process start disables the accelerated paths for debugging and
  A/B comparison. (Tests may combine ``monkeypatch.setenv`` with
  :func:`_reset_cache` to toggle it after start.)
"""

import math
import os
from types import ModuleType

import torch

_ENV_DISABLE = "PHYSICSNEMO_DISABLE_PHYSICSNEMO_OPS"
_ENV_ENABLE_CPU = "PHYSICSNEMO_PHYSICSNEMO_OPS_CPU"

_UNSET = object()
_cached: ModuleType | None | object = _UNSET
_cpu_enabled: bool | object = _UNSET


def physicsnemo_ops_torch() -> ModuleType | None:
    """Return ``physicsnemo_ops.torch`` if available and enabled, else ``None``.

    The import is attempted once per process; any failure (package not
    installed, incompatible torch, broken compiled extension) results in a
    cached ``None`` so callers silently use their pure-torch fallbacks.
    """
    global _cached
    if _cached is _UNSET:
        if os.environ.get(_ENV_DISABLE, "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            _cached = None
        else:
            try:
                import physicsnemo_ops.torch as _physicsnemo_ops_torch

                _cached = _physicsnemo_ops_torch
            except Exception:
                _cached = None
    return _cached  # type: ignore[return-value]


def physicsnemo_ops_for(tensor: torch.Tensor) -> ModuleType | None:
    """Return ``physicsnemo_ops.torch`` if it should serve ops on ``tensor``.

    Applies the device policy on top of :func:`physicsnemo_ops_torch`:
    CUDA tensors use the accelerated kernels; CPU tensors keep the
    pure-torch fallbacks (measured faster at mesh-typical sizes) unless
    ``PHYSICSNEMO_PHYSICSNEMO_OPS_CPU=1`` opts in. Compiled regions always
    use the pure-torch paths: inductor fuses native scatter/unique into
    surrounding element-wise work, which measured 5-15% faster end-to-end
    than routing through an opaque custom op (and avoids its graph break).
    """
    if torch.compiler.is_compiling():
        return None
    ops = physicsnemo_ops_torch()
    if ops is None or tensor.is_cuda:
        return ops
    global _cpu_enabled
    if _cpu_enabled is _UNSET:
        _cpu_enabled = os.environ.get(_ENV_ENABLE_CPU, "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
    return ops if _cpu_enabled else None


def _reset_cache() -> None:
    """Forget the cached import/policy decisions. Intended for tests only."""
    global _cached, _cpu_enabled
    _cached = _UNSET
    _cpu_enabled = _UNSET


#: Row-count/width/bound constraints of ``physicsnemo_ops`` integer row ops
#: (``unique_rows`` / ``lookup_rows``): int32/int64 rows, 1..8 columns, and
#: ``index_bound ** n_columns`` must fit in signed int64.
def int_rows_ok(rows: torch.Tensor, index_bound: int) -> bool:
    """Whether ``rows`` satisfies the ``unique_rows``/``lookup_rows`` constraints."""
    if rows.dtype not in (torch.int32, torch.int64):
        return False
    if rows.ndim != 2 or not (1 <= rows.shape[1] <= 8):
        return False
    if index_bound < 1:
        return False
    if index_bound <= 1:
        return True
    return math.log2(index_bound) * rows.shape[1] < 63


#: Dtypes accepted by the COO segment ops (``segment_sum``/``segment_min``/
#: ``segment_max``) per device type. CUDA additionally accepts half precision
#: for ``segment_sum``.
_SEGMENT_COO_CPU_DTYPES = frozenset(
    {torch.float32, torch.float64, torch.int32, torch.int64}
)
_SEGMENT_SUM_CUDA_DTYPES = frozenset(
    {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
        torch.int32,
        torch.int64,
    }
)


def segment_sum_dtype_ok(data: torch.Tensor) -> bool:
    """Whether ``data`` has a dtype supported by ``physicsnemo_ops`` segment_sum."""
    if data.is_cuda:
        return data.dtype in _SEGMENT_SUM_CUDA_DTYPES
    return data.dtype in _SEGMENT_COO_CPU_DTYPES


def segment_cmp_dtype_ok(data: torch.Tensor) -> bool:
    """Whether ``data`` is supported by segment_min/segment_max (both devices)."""
    return data.dtype in _SEGMENT_COO_CPU_DTYPES


#: Dtypes accepted by the CSR segment reductions per operation and device.
_CSR_SUM_CPU_DTYPES = _SEGMENT_COO_CPU_DTYPES
_CSR_SUM_CUDA_DTYPES = _SEGMENT_SUM_CUDA_DTYPES
_CSR_MEAN_CPU_DTYPES = frozenset({torch.float32, torch.float64})
_CSR_MEAN_CUDA_DTYPES = frozenset(
    {torch.float16, torch.bfloat16, torch.float32, torch.float64}
)
_CSR_CMP_DTYPES = _SEGMENT_COO_CPU_DTYPES


def csr_sum_dtype_ok(values: torch.Tensor) -> bool:
    """Whether ``values`` is supported by ``segment_sum_csr``."""
    if values.is_cuda:
        return values.dtype in _CSR_SUM_CUDA_DTYPES
    return values.dtype in _CSR_SUM_CPU_DTYPES


def csr_mean_dtype_ok(values: torch.Tensor) -> bool:
    """Whether ``values`` is supported by ``segment_mean_csr``."""
    if values.is_cuda:
        return values.dtype in _CSR_MEAN_CUDA_DTYPES
    return values.dtype in _CSR_MEAN_CPU_DTYPES


def csr_cmp_dtype_ok(values: torch.Tensor) -> bool:
    """Whether ``values`` is supported by ``segment_min_csr``/``segment_max_csr``."""
    return values.dtype in _CSR_CMP_DTYPES
