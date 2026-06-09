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

# Check if triton is available
import triton
import triton.language as tl

from .activations import activation_dispatch
from .load_store import (
    load_input_block,
    load_input_tile,
    load_weight_matrix,
    load_weight_tile,
)
from .utils import PRECISION


@triton.jit
def fwd_out_tile_reduce_k(
    pid,
    input_ptr,
    input_stride_M,
    input_stride_K,
    weight_ptr,
    weight_stride_K,
    weight_stride_N,
    bias_ptr,
    n_offset,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_KR: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    """Compute one ``[BLOCK_SIZE_M, BLOCK_SIZE_N]`` output tile, reducing over K.

    Reads the input from DRAM in ``BLOCK_SIZE_KR``-wide chunks and accumulates
    ``input[M, K] @ weight[N, K].T`` for the output columns ``[n_offset,
    n_offset + BLOCK_SIZE_N)``. Only a ``[BLOCK_SIZE_M, BLOCK_SIZE_KR]`` input
    slice and a ``[BLOCK_SIZE_N, BLOCK_SIZE_KR]`` weight slice are resident at a
    time, so the contraction width K can grow without holding the whole weight.

    Used for the first layer (call once with ``BLOCK_SIZE_N`` = full hidden width
    to produce the resident hidden tile) and for single-layer MLPs (call per
    output tile in an N loop).
    """

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Stream the contraction dimension; static (unrolled) loop over constexprs.
    for k_offset in range(0, K, BLOCK_SIZE_KR):
        input_tile = load_input_tile(
            pid,
            input_ptr,
            input_stride_M,
            input_stride_K,
            k_offset,
            BLOCK_SIZE_M,
            BLOCK_SIZE_KR,
            M,
            K,
        )
        weight_tile = load_weight_tile(
            weight_ptr,
            weight_stride_K,
            weight_stride_N,
            n_offset,
            k_offset,
            BLOCK_SIZE_N,
            BLOCK_SIZE_KR,
            N,
            K,
        )
        accumulator += tl.dot(
            input_tile, tl.trans(weight_tile), input_precision=PRECISION
        )

    if bias_ptr is not None:
        bias_idx = n_offset + tl.arange(0, BLOCK_SIZE_N)
        bias_block = tl.load(bias_ptr + bias_idx, mask=bias_idx < N)
        accumulator += bias_block

    return accumulator


@triton.jit
def fwd_consume_resident_out_tile(
    resident_input,
    weight_ptr,
    weight_stride_K,
    weight_stride_N,
    bias_ptr,
    n_offset,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    """Compute one ``[BLOCK_SIZE_M, BLOCK_SIZE_N]`` output tile from a resident input.

    The contraction dimension K (the hidden width) is already fully resident in
    ``resident_input`` ``[BLOCK_SIZE_M, BLOCK_SIZE_K]``, so no reduction loop is
    needed -- a single dot against the ``[n_offset, n_offset + BLOCK_SIZE_N)``
    slice of the weight produces the tile. Used by the last layer (called per
    output tile in an N loop) so the output width N can grow while the hidden
    activation stays on chip.
    """

    weight_tile = load_weight_tile(
        weight_ptr,
        weight_stride_K,
        weight_stride_N,
        n_offset,
        0,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        N,
        K,
    )
    output = tl.dot(
        resident_input, tl.trans(weight_tile), input_precision=PRECISION
    )

    if bias_ptr is not None:
        bias_idx = n_offset + tl.arange(0, BLOCK_SIZE_N)
        bias_block = tl.load(bias_ptr + bias_idx, mask=bias_idx < N)
        output += bias_block

    return output


@triton.jit
def fwd_tiled_middle_out_tile(
    resident_hidden1,
    weight1_ptr,
    weight1_stride_K,
    weight1_stride_N,
    bias1_ptr,
    weight2_ptr,
    weight2_stride_K,
    weight2_stride_N,
    bias2_ptr,
    n_offset,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
    BLOCK_SIZE_KR2: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    N: tl.constexpr,
):
    """Compute one final-output tile of a fused middle+last layer, tiling K2.

    Given the resident first-hidden tile ``resident_hidden1`` ``[BLOCK_SIZE_M,
    K1]``, this produces the output columns ``[n_offset, n_offset +
    BLOCK_SIZE_N)`` by a double reduction over the second hidden width ``K2``:
    for each ``BLOCK_SIZE_KR2`` chunk it builds that slice of ``hidden2 =
    act(hidden1 @ W1.T + b1)`` on the fly and immediately contracts it into the
    output with the matching ``W2`` slice. Only a ``[BLOCK_SIZE_M,
    BLOCK_SIZE_KR2]`` hidden2 chunk and the two weight slices are resident at a
    time, so neither the full ``hidden2`` nor the full middle weight ``W1``
    ``[K2, K1]`` is ever held -- which is what lets the three-layer kernel run at
    wide hidden widths (e.g. 256) without exceeding shared memory.

    The middle bias ``b1`` is applied per chunk (inside the hidden2 build); the
    final bias ``b2`` is applied once after the K2 reduction. ``hidden2`` is
    recomputed once per output tile, which is cheap because the output width
    ``N`` is small in the tall-skinny regime (typically 1-2 tiles).
    """

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k2_offset in range(0, K2, BLOCK_SIZE_KR2):
        # Build a [BLOCK_SIZE_M, BLOCK_SIZE_KR2] slice of hidden2 from the
        # resident hidden1 (contraction over K1, output slice over K2).
        hidden2_chunk = fwd_consume_resident_out_tile(
            resident_hidden1,
            weight1_ptr,
            weight1_stride_K,
            weight1_stride_N,
            bias1_ptr,
            k2_offset,
            BLOCK_SIZE_K1,
            BLOCK_SIZE_KR2,
            K1,
            K2,
        )
        hidden2_chunk = activation_dispatch(hidden2_chunk, ACTIVATION)

        # Contract the chunk into the output tile (reduction over K2).
        weight2_tile = load_weight_tile(
            weight2_ptr,
            weight2_stride_K,
            weight2_stride_N,
            n_offset,
            k2_offset,
            BLOCK_SIZE_N,
            BLOCK_SIZE_KR2,
            N,
            K2,
        )
        accumulator += tl.dot(
            hidden2_chunk, tl.trans(weight2_tile), input_precision=PRECISION
        )

    if bias2_ptr is not None:
        bias_idx = n_offset + tl.arange(0, BLOCK_SIZE_N)
        bias2_block = tl.load(bias2_ptr + bias_idx, mask=bias_idx < N)
        accumulator += bias2_block

    return accumulator


@triton.jit
def forward_matmul_no_activation(
    pid,
    input_ptr,
    input_stride_M,
    input_stride_K,
    weight_ptr,
    weight_stride_K: tl.constexpr,
    weight_stride_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    bias_ptr,
):
    """
    Compute the forward pass of a matmul, without activation

    The activation is computed elsewhere, in the caller, since behavior is different fwd/bwd
    """

    output = load_input_block(
        pid, input_ptr, input_stride_M, input_stride_K, BLOCK_SIZE_M, BLOCK_SIZE_K, M, K
    )

    ######## Load weight matrix
    weight_matrix = load_weight_matrix(
        weight_ptr,
        weight_stride_K,
        weight_stride_N,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
        K,
        N,
    )

    # weight_matrix = weight_matrix.to(output.dtype)

    # Load the bias if its not none:
    if bias_ptr is not None:
        block_idx = tl.arange(0, BLOCK_SIZE_N)
        block_mask = block_idx < N
        bias_block = tl.load(bias_ptr + block_idx, mask=block_mask)

    output = tl.dot(output, tl.trans(weight_matrix), input_precision=PRECISION)

    if bias_ptr is not None:
        output += bias_block

    return output


@triton.jit
def forward_matmul_with_input(
    input_block,
    weight_ptr,
    weight_stride_K,
    weight_stride_N,
    BLOCK_SIZE_K,
    BLOCK_SIZE_N,
    K,
    N,
    bias_ptr,
):

    ######## Load weight matrix
    weight_matrix = load_weight_matrix(
        weight_ptr,
        weight_stride_K,
        weight_stride_N,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
        K,
        N,
    )

    # weight_matrix = weight_matrix.to(output.dtype)

    # Load the bias if its not none:
    if bias_ptr is not None:
        block_idx = tl.arange(0, BLOCK_SIZE_N)
        block_mask = block_idx < N
        bias_block = tl.load(bias_ptr + block_idx, mask=block_mask)

    output = tl.dot(input_block, tl.trans(weight_matrix), input_precision=PRECISION)

    if bias_ptr is not None:
        output += bias_block

    return output


@triton.jit
def forward_matmul_no_activation_with_weights_and_inputs(
    pid,
    input_ptr,
    input_stride_M,
    input_stride_K,
    weight_ptr,
    weight_stride_K,
    weight_stride_N,
    BLOCK_SIZE_M,
    BLOCK_SIZE_K,
    BLOCK_SIZE_N,
    M,
    K,
    N,
    bias_ptr,
):
    """
    Compute the forward pass of a matmul, without activation

    The activation is computed elsewhere, in the caller, since behavior is different fwd/bwd
    """

    inputs = load_input_block(
        pid, input_ptr, input_stride_M, input_stride_K, BLOCK_SIZE_M, BLOCK_SIZE_K, M, K
    )

    ######## Load weight matrix
    weight_matrix = load_weight_matrix(
        weight_ptr,
        weight_stride_K,
        weight_stride_N,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
        K,
        N,
    )

    # weight_matrix = weight_matrix.to(inputs.dtype)

    # Load the bias if its not none:
    if bias_ptr is not None:
        block_idx = tl.arange(0, BLOCK_SIZE_N)
        block_mask = block_idx < N
        bias_block = tl.load(bias_ptr + block_idx, mask=block_mask)

    output = tl.dot(inputs, tl.trans(weight_matrix), input_precision=PRECISION)

    if bias_ptr is not None:
        output += bias_block

    return output, weight_matrix, inputs


@triton.jit
def forward_matmul_with_input_and_weights(
    input_block,
    weight_ptr,
    weight_stride_K,
    weight_stride_N,
    BLOCK_SIZE_K,
    BLOCK_SIZE_N,
    K,
    N,
    bias_ptr,
):

    ######## Load weight matrix
    weight_matrix = load_weight_matrix(
        weight_ptr,
        weight_stride_K,
        weight_stride_N,
        BLOCK_SIZE_K,
        BLOCK_SIZE_N,
        K,
        N,
    )

    # weight_matrix = weight_matrix.to(output.dtype)

    # Load the bias if its not none:
    if bias_ptr is not None:
        block_idx = tl.arange(0, BLOCK_SIZE_N)
        block_mask = block_idx < N
        bias_block = tl.load(bias_ptr + block_idx, mask=block_mask)

    output = tl.dot(input_block, tl.trans(weight_matrix), input_precision=PRECISION)

    if bias_ptr is not None:
        output += bias_block

    return output, weight_matrix