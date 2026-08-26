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

"""Scatter operation utilities for aggregating data across mesh elements.

This module provides unified scatter-based aggregation operations that are
commonly used throughout physicsnemo.mesh for transferring data between different
mesh entities (points, cells, facets).
"""

import torch
from jaxtyping import Float, Int, Shaped

from physicsnemo.mesh.utilities._tolerances import safe_eps
from physicsnemo.utils._physicsnemo_ops import (
    csr_cmp_dtype_ok,
    csr_mean_dtype_ok,
    csr_sum_dtype_ok,
    physicsnemo_ops_for,
    segment_cmp_dtype_ok,
    segment_sum_dtype_ok,
)


def scatter_sum_coo(
    src: Shaped[torch.Tensor, "n_src ..."],
    index: Int[torch.Tensor, " n_src"],
    n_dst: int,
    *,
    init: Shaped[torch.Tensor, "n_dst ..."] | None = None,
) -> Shaped[torch.Tensor, "n_dst ..."]:
    """Segmented sum: ``out[j] = init[j] + sum(src[i] for index[i] == j)``.

    Drop-in for the ``zeros().scatter_add_`` pattern, accelerated by the
    optional ``physicsnemo_ops.segment_sum`` kernel (autograd-capable,
    including double backward). Falls back to ``torch.scatter_add_``.
    """
    ops = physicsnemo_ops_for(src)
    if (
        ops is not None
        and src.shape[0] > 0
        and not src.is_complex()
        and segment_sum_dtype_ok(src)
        and index.dtype in (torch.int32, torch.int64)
        and (init is None or init.dtype == src.dtype)
    ):
        return ops.segment_sum(src.contiguous(), index.contiguous(), n_dst, init)

    if init is not None:
        out = init.clone()
    else:
        out = torch.zeros((n_dst, *src.shape[1:]), dtype=src.dtype, device=src.device)
    if src.shape[0] == 0:
        return out
    expanded_index = index.view(-1, *([1] * (src.ndim - 1))).expand_as(src)
    return out.scatter_add(0, expanded_index.long(), src)


def _scatter_cmp_coo(
    src: Shaped[torch.Tensor, "n_src ..."],
    index: Int[torch.Tensor, " n_src"],
    n_dst: int,
    init: Shaped[torch.Tensor, "n_dst ..."],
    reduce: str,
) -> Shaped[torch.Tensor, "n_dst ..."]:
    ops = physicsnemo_ops_for(src)
    if (
        ops is not None
        and src.shape[0] > 0
        and segment_cmp_dtype_ok(src)
        and index.dtype in (torch.int32, torch.int64)
        and init.dtype == src.dtype
        and not src.requires_grad
        and not init.requires_grad
    ):
        op = ops.segment_min if reduce == "amin" else ops.segment_max
        return op(src.contiguous(), index.contiguous(), n_dst, init.contiguous())

    out = init.clone()
    if src.shape[0] == 0:
        return out
    expanded_index = index.view(-1, *([1] * (src.ndim - 1))).expand_as(src)
    return out.scatter_reduce_(
        0, expanded_index.long(), src, reduce=reduce, include_self=True
    )


def scatter_min_coo(
    src: Shaped[torch.Tensor, "n_src ..."],
    index: Int[torch.Tensor, " n_src"],
    n_dst: int,
    *,
    init: Shaped[torch.Tensor, "n_dst ..."],
) -> Shaped[torch.Tensor, "n_dst ..."]:
    """Segmented min with include-self semantics: destinations that receive no
    source keep ``init``. No autograd (index-producing/label use only)."""
    return _scatter_cmp_coo(src, index, n_dst, init, "amin")


def scatter_max_coo(
    src: Shaped[torch.Tensor, "n_src ..."],
    index: Int[torch.Tensor, " n_src"],
    n_dst: int,
    *,
    init: Shaped[torch.Tensor, "n_dst ..."],
) -> Shaped[torch.Tensor, "n_dst ..."]:
    """Segmented max with include-self semantics (see :func:`scatter_min_coo`)."""
    return _scatter_cmp_coo(src, index, n_dst, init, "amax")


def _csr_segment_ids(
    offsets: Int[torch.Tensor, " n_segments_plus_1"],
    n_elements: int,
) -> Int[torch.Tensor, " n_elements"]:
    """Expand CSR offsets to per-element segment ids (fallback path helper)."""
    n_segments = offsets.shape[0] - 1
    return torch.repeat_interleave(
        torch.arange(n_segments, device=offsets.device),
        offsets.diff(),
        output_size=n_elements,
    )


def segment_sum_csr(
    values: Shaped[torch.Tensor, "n_elements ..."],
    offsets: Int[torch.Tensor, " n_segments_plus_1"],
) -> Shaped[torch.Tensor, "n_segments ..."]:
    """CSR segmented sum; empty segments are 0. Deterministic on the
    accelerated path; autograd-capable on both paths."""
    ops = physicsnemo_ops_for(values)
    if (
        ops is not None
        and values.shape[0] > 0
        and not values.is_complex()
        and csr_sum_dtype_ok(values)
        and offsets.dtype in (torch.int32, torch.int64)
    ):
        return ops.segment_sum_csr(values.contiguous(), offsets.contiguous())

    n_segments = offsets.shape[0] - 1
    out = torch.zeros(
        (n_segments, *values.shape[1:]), dtype=values.dtype, device=values.device
    )
    if values.shape[0] == 0:
        return out
    seg_ids = _csr_segment_ids(offsets, values.shape[0])
    expanded = seg_ids.view(-1, *([1] * (values.ndim - 1))).expand_as(values)
    return out.scatter_add(0, expanded, values)


def segment_mean_csr(
    values: Float[torch.Tensor, "n_elements ..."],
    offsets: Int[torch.Tensor, " n_segments_plus_1"],
) -> Float[torch.Tensor, "n_segments ..."]:
    """CSR segmented mean; empty segments are 0. Autograd-capable."""
    ops = physicsnemo_ops_for(values)
    if (
        ops is not None
        and values.shape[0] > 0
        and not values.is_complex()
        and csr_mean_dtype_ok(values)
        and offsets.dtype in (torch.int32, torch.int64)
    ):
        return ops.segment_mean_csr(values.contiguous(), offsets.contiguous())

    sums = segment_sum_csr(values, offsets)
    counts = offsets.diff().clamp_min(1)
    return sums / counts.view(-1, *([1] * (values.ndim - 1))).to(sums.dtype)


def _segment_cmp_csr(
    values: Shaped[torch.Tensor, "n_elements ..."],
    offsets: Int[torch.Tensor, " n_segments_plus_1"],
    reduce: str,
) -> Shaped[torch.Tensor, "n_segments ..."]:
    ops = physicsnemo_ops_for(values)
    if (
        ops is not None
        and values.shape[0] > 0
        and csr_cmp_dtype_ok(values)
        and offsets.dtype in (torch.int32, torch.int64)
        and not values.requires_grad
    ):
        op = ops.segment_min_csr if reduce == "amin" else ops.segment_max_csr
        return op(values.contiguous(), offsets.contiguous())

    n_segments = offsets.shape[0] - 1
    fill = float("inf") if reduce == "amin" else float("-inf")
    if not values.is_floating_point():
        info = torch.iinfo(values.dtype)
        fill = info.max if reduce == "amin" else info.min
    out = torch.full(
        (n_segments, *values.shape[1:]), fill, dtype=values.dtype, device=values.device
    )
    counts = offsets.diff()
    if values.shape[0] > 0:
        seg_ids = _csr_segment_ids(offsets, values.shape[0])
        expanded = seg_ids.view(-1, *([1] * (values.ndim - 1))).expand_as(values)
        out.scatter_reduce_(0, expanded, values, reduce=reduce, include_self=True)
    ### Empty segments are defined to be 0 (torch_scatter.segment_csr parity,
    ### matching the physicsnemo_ops CSR kernels).
    empty = (counts == 0).view(-1, *([1] * (values.ndim - 1)))
    return torch.where(empty, torch.zeros_like(out), out)


def segment_min_csr(
    values: Shaped[torch.Tensor, "n_elements ..."],
    offsets: Int[torch.Tensor, " n_segments_plus_1"],
) -> Shaped[torch.Tensor, "n_segments ..."]:
    """CSR segmented min; empty segments are 0 (torch_scatter parity)."""
    return _segment_cmp_csr(values, offsets, "amin")


def segment_max_csr(
    values: Shaped[torch.Tensor, "n_elements ..."],
    offsets: Int[torch.Tensor, " n_segments_plus_1"],
) -> Shaped[torch.Tensor, "n_segments ..."]:
    """CSR segmented max; empty segments are 0 (torch_scatter parity)."""
    return _segment_cmp_csr(values, offsets, "amax")


def scatter_aggregate(
    src_data: Shaped[torch.Tensor, "n_src ..."],
    src_to_dst_mapping: Int[torch.Tensor, " n_src"],
    n_dst: int,
    weights: Float[torch.Tensor, " n_src"] | None = None,
    aggregation: str = "mean",
) -> Shaped[torch.Tensor, "n_dst ..."]:
    """Aggregate source data to destination using scatter operations.

    This is the core scatter-based aggregation pattern used throughout physicsnemo.mesh
    for operations like:

    - Aggregating cell data to points
    - Aggregating parent cell data to facets
    - Merging duplicate point data

    The pattern is:
    1. Initialize destination tensor with zeros
    2. Scatter-add weighted source data to destinations
    3. Scatter-add weights to compute normalization
    4. Divide aggregated data by total weights

    Parameters
    ----------
    src_data : torch.Tensor
        Source data to aggregate, shape (n_src, *data_shape).
    src_to_dst_mapping : torch.Tensor
        Mapping from each source to its destination index,
        shape (n_src,). Each value should be in [0, n_dst).
    n_dst : int
        Number of destination elements.
    weights : torch.Tensor or None
        Optional weights for each source element, shape (n_src,).
        If None, uses uniform weights of 1.0.
    aggregation : str
        Aggregation mode:

        - "mean": Weighted mean (uses weights if provided, uniform otherwise)
        - "sum": Weighted sum (no normalization)

    Returns
    -------
    torch.Tensor
        Aggregated data at destinations, shape (n_dst, *data_shape).
        For "mean" mode, values are weighted averages.
        For "sum" mode, values are weighted sums.

    Notes
    -----
    The output dtype follows ``src_data``, with one exception: a ``"mean"`` of
    an integer or boolean ``src_data`` is promoted to ``torch.float64``. A mean
    of integers is generally non-integral, so computing it in the source integer
    dtype would truncate (e.g. ``(1 + 2) // 2 == 1``); promoting to a floating
    dtype avoids this. A ``"sum"`` always preserves the source dtype.

    Examples
    --------
    >>> # Aggregate cell data to points
    >>> src_data = torch.tensor([[1.0], [2.0], [3.0]])  # 3 cells
    >>> src_to_dst = torch.tensor([0, 0, 1])  # map to 2 points
    >>> result = scatter_aggregate(src_data, src_to_dst, n_dst=2)
    >>> # result = [[1.5], [3.0]]  # point 0 gets mean of cells 0,1
    """
    device = src_data.device
    dtype = src_data.dtype

    ### Get data shape beyond the first dimension
    data_shape = src_data.shape[1:]

    if aggregation not in ("mean", "sum"):
        raise ValueError(f"Invalid {aggregation=}. Must be 'mean' or 'sum'.")

    ### Choose the compute dtype. A "mean" of integer/bool data must be computed in a
    ### floating dtype: integer division truncates (e.g. (1 + 2) // 2 == 1), and the
    ### division guard ``safe_eps()`` -> ``torch.finfo`` raises on integer dtypes. A
    ### "sum" preserves the native (possibly integer) dtype.
    if aggregation == "mean" and not torch.is_floating_point(src_data):
        compute_dtype = torch.float64
    else:
        compute_dtype = dtype

    ### Fast path: unweighted sum is a single segmented sum with no extra work
    if weights is None and aggregation == "sum":
        return scatter_sum_coo(src_data, src_to_dst_mapping, n_dst)

    ### Initialize weights if not provided
    if weights is None:
        weights = torch.ones(
            len(src_to_dst_mapping), dtype=compute_dtype, device=device
        )

    ### Ensure weights share the compute dtype (avoid dtype mismatch in multiplication)
    if weights.dtype != compute_dtype:
        weights = weights.to(compute_dtype)

    ### Weight the source data
    # Broadcast weights to match data shape: (n_src, *data_shape)
    weight_shape = [len(weights)] + [1] * len(data_shape)
    weighted_data = src_data.to(compute_dtype) * weights.view(weight_shape)

    ### Scatter-add weighted data to destinations
    aggregated_data = scatter_sum_coo(weighted_data, src_to_dst_mapping, n_dst)

    ### Normalize weighted sum to weighted mean
    if aggregation == "mean":
        ### Compute sum of weights at each destination
        weight_sums = scatter_sum_coo(weights, src_to_dst_mapping, n_dst)

        ### Normalize by total weight (avoid division by zero)
        weight_sums = weight_sums.clamp(min=safe_eps(weight_sums.dtype))
        aggregated_data = aggregated_data / weight_sums.view(
            -1, *([1] * len(data_shape))
        )

    return aggregated_data
