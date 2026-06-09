# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import triton
import triton.language as tl


@triton.jit
def load_input_block(
    pid,
    input_ptr,
    input_stride_M,
    input_stride_K,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
):
    """
    Load a block of input-like data.  Also used for grad_output (which is input in the backward pass)
    """

    # M represents the "batch" like index, we take groups of rows.
    m_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # And the entirety of the K (feature) dimension:
    k_offsets = tl.arange(0, BLOCK_SIZE_K)

    # Use shaping to make this 2D:
    input_offsets = (
        m_offsets[:, None] * input_stride_M + k_offsets[None, :] * input_stride_K
    )

    # Make sure we stay in bounds:
    row_masks = m_offsets < M
    col_masks = k_offsets < K
    input_masks = row_masks[:, None] & col_masks[None, :]

    input_ptrs = input_ptr + input_offsets

    # Load the input block:
    input_block = tl.load(input_ptrs, mask=input_masks, other=0.0, cache_modifier=".cg")

    return input_block


# Write the outputs:
@triton.jit
def write_output_block(
    pid,
    output_ptr,
    output,
    output_stride_M: tl.constexpr,
    output_stride_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_OUT_N: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):

    out_row_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_row_masks = out_row_offsets < M

    # The outputs have a potentially different column layout from the inputs:
    out_col_offsets = tl.arange(0, BLOCK_SIZE_OUT_N)
    out_col_masks = out_col_offsets < N

    # We have to use the output strides of course, though we can reuse the row offsets
    out_offsets = (
        out_row_offsets[:, None] * output_stride_M
        + out_col_offsets[None, :] * output_stride_N
    )
    out_masks = out_row_masks[:, None] & out_col_masks[None, :]

    # Write the output:
    tl.store(output_ptr + out_offsets, output, mask=out_masks, cache_modifier=".wt")


@triton.jit
def load_input_tile(
    pid,
    input_ptr,
    input_stride_M,
    input_stride_K,
    k_offset,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
):
    """Load a ``[BLOCK_SIZE_M, BLOCK_SIZE_K]`` input tile starting at column ``k_offset``.

    This is the column-tiled generalization of :func:`load_input_block`, used by
    the forward kernels to stream the contraction dimension in chunks while the
    accumulator stays resident. ``k_offset`` selects which slice of the feature
    dimension to read; out-of-range columns are masked to zero.
    """

    # Same row-block as load_input_block; only the column window is offset.
    m_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    k_offsets = k_offset + tl.arange(0, BLOCK_SIZE_K)

    input_offsets = (
        m_offsets[:, None] * input_stride_M + k_offsets[None, :] * input_stride_K
    )

    row_masks = m_offsets < M
    col_masks = k_offsets < K
    input_masks = row_masks[:, None] & col_masks[None, :]

    return tl.load(
        input_ptr + input_offsets, mask=input_masks, other=0.0, cache_modifier=".cg"
    )


@triton.jit
def load_weight_tile(
    weight_ptr,
    weight_stride_K,
    weight_stride_N,
    n_offset,
    k_offset,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
):
    """Load a ``[BLOCK_SIZE_N, BLOCK_SIZE_K]`` tile of a ``[N, K]`` weight matrix.

    Mirrors :func:`load_weight_matrix` but reads only the window starting at
    ``(n_offset, k_offset)``, so the forward kernels can stream weight slices
    instead of holding the whole matrix resident. The tile is returned in the
    native ``[N, K]`` (PyTorch ``nn.Linear``) layout and must be transposed by
    the caller before the dot.
    """

    n_offsets = n_offset + tl.arange(0, BLOCK_SIZE_N)
    k_offsets = k_offset + tl.arange(0, BLOCK_SIZE_K)

    offsets = (
        n_offsets[:, None] * weight_stride_N + k_offsets[None, :] * weight_stride_K
    )

    mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
    return tl.load(weight_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")


@triton.jit
def write_output_tile(
    pid,
    output_ptr,
    output,
    output_stride_M: tl.constexpr,
    output_stride_N: tl.constexpr,
    n_offset,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
):
    """Write a ``[BLOCK_SIZE_M, BLOCK_SIZE_N]`` output tile at column ``n_offset``.

    Column-tiled generalization of :func:`write_output_block` so the last layer
    can emit its output one N-slice at a time.
    """

    out_row_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_row_masks = out_row_offsets < M

    out_col_offsets = n_offset + tl.arange(0, BLOCK_SIZE_N)
    out_col_masks = out_col_offsets < N

    out_offsets = (
        out_row_offsets[:, None] * output_stride_M
        + out_col_offsets[None, :] * output_stride_N
    )
    out_masks = out_row_masks[:, None] & out_col_masks[None, :]

    tl.store(output_ptr + out_offsets, output, mask=out_masks, cache_modifier=".wt")


@triton.jit
def load_weight_matrix(
    weight_ptr,
    weight_stride_K,  # Stride for in_features (second dim in PyTorch)
    weight_stride_N,  # Stride for out_features (first dim in PyTorch)
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    """
    Load a weight matrix from PyTorch Linear layer format [N, K].
    Loads in native PyTorch layout for better coalescing.

    **NOTE** Returns matrix for matmul that needs to be transposed.

    Args:
        weight_ptr: Pointer to weight matrix in PyTorch format [N, K]
        weight_stride_N: Stride for out_features (first dim in PyTorch)
        weight_stride_K: Stride for in_features (second dim in PyTorch)
        K: Number of input features (second dim in PyTorch)
        N: Number of output features (first dim in PyTorch)

    Returns:
        weight_matrix: A [K, N] matrix
    """
    # Load in PyTorch's native [N, K] layout.  No thread id, get the whole thing
    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)

    offsets = (
        n_offsets[:, None] * weight_stride_N
        + k_offsets[None, :] * weight_stride_K  # First dim stride  # Second dim stride
    )

    mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
    weight_matrix = tl.load(
        weight_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg"
    )

    return weight_matrix


@triton.jit
def load_weight_matrix_kn(
    weight_ptr,
    weight_stride_K,  # Stride for in_features (second dim in PyTorch [N, K])
    weight_stride_N,  # Stride for out_features (first dim in PyTorch [N, K])
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    """Load a ``[N, K]`` PyTorch weight directly into ``[K, N]`` (dot) layout.

    Equivalent in value to ``tl.trans(load_weight_matrix(...))`` -- it produces
    the operand for ``input[M, K] @ weight[K, N]`` -- but it materializes only
    the single ``[K, N]`` tile instead of staging the native ``[N, K]`` copy
    *and* its transpose. For the persistent kernels the weight is loop-invariant
    and lives in shared memory for the whole grid-stride loop, so avoiding the
    second copy roughly halves the resident weight footprint (the binding
    constraint that caps how wide a layer can stay weight-resident).

    The read is strided along ``N`` (``weight_stride_N``), so it is less
    coalesced than :func:`load_weight_matrix`; this is a deliberate trade since
    the load happens once per program, not once per row-block.
    """

    k_offsets = tl.arange(0, BLOCK_SIZE_K)
    n_offsets = tl.arange(0, BLOCK_SIZE_N)

    # [K, N] tile: row index walks in-features (K), column index walks
    # out-features (N), using the native [N, K] strides.
    offsets = (
        k_offsets[:, None] * weight_stride_K + n_offsets[None, :] * weight_stride_N
    )

    mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
    return tl.load(weight_ptr + offsets, mask=mask, other=0.0, cache_modifier=".cg")


@triton.jit
def atomic_update_weight(
    weight_ptr,
    weight_updates,
    weight_stride_K,
    weight_stride_N,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    """
    Perform an atomic add to a weight-shaped matrix.  Used for gradient accumulation.

    Matrix is expected to be contiguous in [N, K] format.

    """

    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)

    # K is moving fastest (and stride 1) - it's the interior index
    offsets = (
        n_offsets[:, None] * weight_stride_N + k_offsets[None, :] * weight_stride_K
    )

    mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)

    # Use atomics to sum it to the grad_weight tensor:
    # (Remember that the weight is stored transposed!)
    tl.atomic_add(weight_ptr + offsets, weight_updates, mask=mask, scope="gpu")


@triton.jit
def write_weight_buffer_chunk(
    pid,
    weight_ptr,
    weight_updates,
    weight_stride_K: tl.constexpr,
    weight_stride_N: tl.constexpr,
    weight_buffer_stride: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
):

    # Write a chunk of weight (gradients, usually) to a buffer.
    # The buffer has shape [num_pids, *weight.shape].
    # We do this to avoid costly atomic operations

    n_offsets = tl.arange(0, BLOCK_SIZE_N)
    k_offsets = tl.arange(0, BLOCK_SIZE_K)

    # K is moving fastest (and stride 1) - it's the interior index
    offsets = (
        n_offsets[:, None] * weight_stride_N + k_offsets[None, :] * weight_stride_K
    )

    mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)

    # Use atomics to sum it to the grad_weight tensor:
    # (Remember that the weight is stored transposed!)
    tl.store(
        weight_ptr + pid * weight_buffer_stride + offsets,
        weight_updates,
        mask=mask,
        eviction_policy="evict_first",
    )