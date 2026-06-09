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

import torch
import triton
import triton.language as tl

from physicsnemo.utils.profiling import profile

from .activations import (
    Activation,
    activation_dispatch,
)
from .load_store import (
    load_input_block,
    load_weight_matrix_kn,
    write_output_block,
    write_output_tile,
)
from .primitives import (
    fwd_out_tile_reduce_k,
    fwd_tiled_middle_out_tile,
)
from .utils import (
    PRECISION,
    get_cuda_autotune_config_fwd_persistent,
    get_cuda_autotune_config_fwd_tiled,
    persistent_max_width,
)

# The three-layer persistent kernel holds all three weights resident, so its
# per-GPU width threshold (see persistent_max_width) is naturally lower than the
# two-layer kernel's; wider problems use the (middle-tiled) width-tiled kernel.
_PERSISTENT_N_WEIGHTS = 3


@triton.autotune(
    configs=get_cuda_autotune_config_fwd_tiled(),
    # Re-tune per problem shape (see matmul1_kernel for the rationale).
    key=["M", "K0", "K1", "K2", "N"],
)
@triton.jit
def matmul3_kernel(
    input_ptr,
    input_stride_M: tl.constexpr,
    input_stride_K: tl.constexpr,
    weight0_ptr,
    weight0_stride_K: tl.constexpr,
    weight0_stride_N: tl.constexpr,
    weight1_ptr,
    weight1_stride_K: tl.constexpr,
    weight1_stride_N: tl.constexpr,
    weight2_ptr,
    weight2_stride_K: tl.constexpr,
    weight2_stride_N: tl.constexpr,
    bias0_ptr,
    bias1_ptr,
    bias2_ptr,
    output_ptr,
    output_stride_M: tl.constexpr,
    output_stride_N: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAST_ACTIVATION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_KR: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    M: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    N: tl.constexpr,
):
    """
    Fused three-layer MLP forward.

    ``X`` is ``[M, K0]``; hidden widths are ``K1`` then ``K2``; output is
    ``[M, N]``. Designed for tall-skinny matrices (huge ``M``, small widths).

    Only the first hidden tile ``[BLOCK_SIZE_M, K1]`` is held fully resident.
    Layer 0 streams the input width ``K0`` (in ``BLOCK_SIZE_KR`` chunks) to build
    it. The middle and last layers are then fused as a double reduction over the
    second hidden width ``K2`` (see :func:`fwd_tiled_middle_out_tile`): for each
    ``BLOCK_SIZE_N`` output tile, ``hidden2`` is built one ``BLOCK_SIZE_KR``
    chunk at a time and immediately contracted into the output, so neither the
    full ``hidden2`` nor the full middle weight ``W1`` is ever resident. The
    intermediates never touch DRAM, and wide hidden widths (e.g. 256) no longer
    exceed shared memory.

    Inputs:
        input_ptr: Pointer to the input tensor
        input_stride_M: Stride for the input tensor along the M dimension (first)
        input_stride_K: Stride for the input tensor along the K dimension (second)
        weight0_ptr/weight1_ptr/weight2_ptr: Weight pointers ([N, K] layout)
        weight*_stride_K: Stride along the in-features (K) dimension
        weight*_stride_N: Stride along the out-features (N) dimension
        bias0_ptr/bias1_ptr/bias2_ptr: Optional bias pointers (can be None)
        output_ptr: Pointer to the output tensor
        output_stride_M: Stride for the output tensor along the M dimension (first)
        output_stride_N: Stride for the output tensor along the N dimension (second)
        ACTIVATION: Activation function to apply (enum value)
        LAST_ACTIVATION: Whether to apply the activation to the final output
        BLOCK_SIZE_M: Rows processed per program
        BLOCK_SIZE_KR: Contraction tile streamed in both the layer-0 K0 reduction
            and the middle/last K2 reduction
        BLOCK_SIZE_K1: Full (resident) first hidden width
        BLOCK_SIZE_N: Output column tile for layer 2
        M: Number of input rows
        K0/K1/K2: Input and hidden widths
        N: Output feature width
    """

    pid = tl.program_id(axis=0)

    # Layer 0: reduce over input width K0 -> resident hidden tile 1 [BM, K1].
    hidden1 = fwd_out_tile_reduce_k(
        pid,
        input_ptr,
        input_stride_M,
        input_stride_K,
        weight0_ptr,
        weight0_stride_K,
        weight0_stride_N,
        bias0_ptr,
        0,
        BLOCK_SIZE_M,
        BLOCK_SIZE_KR,
        BLOCK_SIZE_K1,
        M,
        K0,
        K1,
    )
    hidden1 = activation_dispatch(hidden1, ACTIVATION)

    # Layers 1+2 fused: for each output tile, build hidden2 in BLOCK_SIZE_KR
    # chunks from the resident hidden1 and contract straight into the output
    # (reduction over K2). Neither hidden2 nor the middle weight is held whole.
    for n_offset in range(0, N, BLOCK_SIZE_N):
        output = fwd_tiled_middle_out_tile(
            hidden1,
            weight1_ptr,
            weight1_stride_K,
            weight1_stride_N,
            bias1_ptr,
            weight2_ptr,
            weight2_stride_K,
            weight2_stride_N,
            bias2_ptr,
            n_offset,
            ACTIVATION,
            BLOCK_SIZE_M,
            BLOCK_SIZE_K1,
            BLOCK_SIZE_KR,
            BLOCK_SIZE_N,
            K1,
            K2,
            N,
        )

        if LAST_ACTIVATION:
            output = activation_dispatch(output, ACTIVATION)

        write_output_tile(
            pid,
            output_ptr,
            output,
            output_stride_M,
            output_stride_N,
            n_offset,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            M,
            N,
        )


@triton.autotune(
    configs=get_cuda_autotune_config_fwd_persistent(),
    key=["M", "K0", "K1", "K2", "N"],
)
@triton.jit
def matmul3_kernel_persistent(
    input_ptr,
    input_stride_M: tl.constexpr,
    input_stride_K: tl.constexpr,
    weight0_ptr,
    weight0_stride_K: tl.constexpr,
    weight0_stride_N: tl.constexpr,
    weight1_ptr,
    weight1_stride_K: tl.constexpr,
    weight1_stride_N: tl.constexpr,
    weight2_ptr,
    weight2_stride_K: tl.constexpr,
    weight2_stride_N: tl.constexpr,
    bias0_ptr,
    bias1_ptr,
    bias2_ptr,
    output_ptr,
    output_stride_M: tl.constexpr,
    output_stride_N: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAST_ACTIVATION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K0: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
    BLOCK_SIZE_K2: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    M: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    K2: tl.constexpr,
    N: tl.constexpr,
):
    """
    Weight-stationary persistent three-layer MLP forward (narrow widths).

    Same math as :func:`matmul3_kernel`, but the grid is fixed at ~one program
    per SM. Each program loads all three weights (and biases) once into registers
    and grid-strides over its share of the row-blocks, streaming the activation
    through the resident weights so each weight is read once per program rather
    than once per row-block. All widths are held resident, so only
    ``BLOCK_SIZE_M`` is autotuned. Used only for widths
    <= ``_PERSISTENT_MAX_WIDTH``; wider problems use the (middle-tiled) tiled
    kernel.

    Inputs:
        input_ptr: Pointer to the input tensor
        input_stride_M/input_stride_K: Input strides (row, col)
        weight0_ptr/weight1_ptr/weight2_ptr: Weight pointers ([N, K] layout)
        weight*_stride_K/weight*_stride_N: Weight strides (in-features, out-features)
        bias0_ptr/bias1_ptr/bias2_ptr: Optional bias pointers (can be None)
        output_ptr: Pointer to the output tensor
        output_stride_M/output_stride_N: Output strides (row, col)
        ACTIVATION: Activation function to apply (enum value)
        LAST_ACTIVATION: Whether to apply the activation to the final output
        BLOCK_SIZE_M: Rows streamed per grid-stride iteration
        BLOCK_SIZE_K0: Full (resident) input width
        BLOCK_SIZE_K1: Full (resident) first hidden width
        BLOCK_SIZE_K2: Full (resident) second hidden width
        BLOCK_SIZE_N: Full (resident) output width
        M: Number of input rows
        K0/K1/K2: Input and hidden widths
        N: Output feature width
    """

    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_row_blocks = tl.cdiv(M, BLOCK_SIZE_M)

    # Hoist all three weights once, directly in [K, N] (dot) layout so only one
    # copy of each is staged in shared memory (half the footprint of
    # load_weight_matrix + tl.trans).
    weight0_t = load_weight_matrix_kn(
        weight0_ptr,
        weight0_stride_K,
        weight0_stride_N,
        BLOCK_SIZE_K0,
        BLOCK_SIZE_K1,
        K0,
        K1,
    )  # [K0, K1]
    weight1_t = load_weight_matrix_kn(
        weight1_ptr,
        weight1_stride_K,
        weight1_stride_N,
        BLOCK_SIZE_K1,
        BLOCK_SIZE_K2,
        K1,
        K2,
    )  # [K1, K2]
    weight2_t = load_weight_matrix_kn(
        weight2_ptr,
        weight2_stride_K,
        weight2_stride_N,
        BLOCK_SIZE_K2,
        BLOCK_SIZE_N,
        K2,
        N,
    )  # [K2, N]

    if bias0_ptr is not None:
        bias0_idx = tl.arange(0, BLOCK_SIZE_K1)
        bias0 = tl.load(bias0_ptr + bias0_idx, mask=bias0_idx < K1)
    if bias1_ptr is not None:
        bias1_idx = tl.arange(0, BLOCK_SIZE_K2)
        bias1 = tl.load(bias1_ptr + bias1_idx, mask=bias1_idx < K2)
    if bias2_ptr is not None:
        bias2_idx = tl.arange(0, BLOCK_SIZE_N)
        bias2 = tl.load(bias2_ptr + bias2_idx, mask=bias2_idx < N)

    for row_block in range(pid, num_row_blocks, num_programs):
        input_block = load_input_block(
            row_block,
            input_ptr,
            input_stride_M,
            input_stride_K,
            BLOCK_SIZE_M,
            BLOCK_SIZE_K0,
            M,
            K0,
        )

        hidden1 = tl.dot(input_block, weight0_t, input_precision=PRECISION)
        if bias0_ptr is not None:
            hidden1 += bias0
        hidden1 = activation_dispatch(hidden1, ACTIVATION)

        hidden2 = tl.dot(hidden1, weight1_t, input_precision=PRECISION)
        if bias1_ptr is not None:
            hidden2 += bias1
        hidden2 = activation_dispatch(hidden2, ACTIVATION)

        output = tl.dot(hidden2, weight2_t, input_precision=PRECISION)
        if bias2_ptr is not None:
            output += bias2
        if LAST_ACTIVATION:
            output = activation_dispatch(output, ACTIVATION)

        write_output_block(
            row_block,
            output_ptr,
            output,
            output_stride_M,
            output_stride_N,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            M,
            N,
        )


@profile
def matmul3_launcher(
    input_tensor: torch.Tensor,
    weight0: torch.Tensor,
    weight1: torch.Tensor,
    weight2: torch.Tensor,
    bias0: torch.Tensor | None = None,
    bias1: torch.Tensor | None = None,
    bias2: torch.Tensor | None = None,
    activation: Activation = Activation.NONE,
    last_activation: bool = False,
):
    """
    This is not meant to be used for full workloads - it exercises the matmul
    kernel for unit tests.

    """

    # We flatten the input matrix along all but the first dimension.
    # Save the shape and use it for the outputs:
    original_shape = input_tensor.shape

    K0 = input_tensor.shape[-1]
    # Remember that the weights are stored transposed:
    K1 = weight0.shape[0]
    K2 = weight1.shape[0]

    if len(input_tensor.shape) > 2:
        # Flatten the batch dimension if needed:
        input_tensor = input_tensor.reshape((-1, K0)).contiguous()
    # Gather matrix shapes:
    M = input_tensor.shape[0]
    N = weight2.shape[0]

    # Use the last N for output shape:
    output_shape = input_tensor.shape[:-1] + (N,)

    # Initialize the output:
    output_mat = torch.empty(
        output_shape, dtype=input_tensor.dtype, device=input_tensor.device
    )

    max_resident_width = persistent_max_width(
        input_tensor.device, _PERSISTENT_N_WEIGHTS, input_tensor.element_size()
    )
    if max(K0, K1, K2, N) <= max_resident_width:
        # Narrow widths: weight-stationary persistent kernel (all widths
        # resident, grid ~ one program per SM).
        BLOCK_SIZE_K0 = max(triton.next_power_of_2(K0), 16)
        BLOCK_SIZE_K1 = max(triton.next_power_of_2(K1), 16)
        BLOCK_SIZE_K2 = max(triton.next_power_of_2(K2), 16)
        BLOCK_SIZE_N = max(triton.next_power_of_2(N), 16)
        num_sm = torch.cuda.get_device_properties(
            input_tensor.device
        ).multi_processor_count

        def grid(META):
            return (min(triton.cdiv(M, META["BLOCK_SIZE_M"]), num_sm),)

        matmul3_kernel_persistent[grid](
            input_ptr=input_tensor,
            input_stride_M=input_tensor.stride(0),
            input_stride_K=input_tensor.stride(1),
            weight0_ptr=weight0,
            weight0_stride_K=weight0.stride(1),
            weight0_stride_N=weight0.stride(0),
            weight1_ptr=weight1,
            weight1_stride_K=weight1.stride(1),
            weight1_stride_N=weight1.stride(0),
            weight2_ptr=weight2,
            weight2_stride_K=weight2.stride(1),
            weight2_stride_N=weight2.stride(0),
            bias0_ptr=bias0,
            bias1_ptr=bias1,
            bias2_ptr=bias2,
            output_ptr=output_mat.view((-1, N)),
            output_stride_M=output_mat.stride(0),
            output_stride_N=output_mat.stride(1),
            ACTIVATION=activation.value,
            LAST_ACTIVATION=last_activation,
            BLOCK_SIZE_K0=BLOCK_SIZE_K0,
            BLOCK_SIZE_K1=BLOCK_SIZE_K1,
            BLOCK_SIZE_K2=BLOCK_SIZE_K2,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            M=M,
            K0=K0,
            K1=K1,
            K2=K2,
            N=N,
        )
        return output_mat.reshape(original_shape[:-1] + (N,))

    # Wide widths: width-tiled kernel. Only the first hidden width K1 is held
    # fully resident; the middle and last layers are tiled over K2 (see
    # matmul3_kernel). The input-reduction tile (BLOCK_SIZE_KR) and the layer-2
    # output tile (BLOCK_SIZE_N) come from the autotuner per problem shape.
    BLOCK_SIZE_K1 = max(triton.next_power_of_2(K1), 16)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

    selected_kernel = matmul3_kernel[grid]

    # The kernel will now automatically select the best configuration based on M and N
    # Remember that the weight matrix is stored [N, K], not [K,N], so the strides are reversed
    selected_kernel(
        input_ptr=input_tensor,
        input_stride_M=input_tensor.stride(0),
        input_stride_K=input_tensor.stride(1),
        weight0_ptr=weight0,
        weight0_stride_K=weight0.stride(1),
        weight0_stride_N=weight0.stride(0),
        weight1_ptr=weight1,
        weight1_stride_K=weight1.stride(1),
        weight1_stride_N=weight1.stride(0),
        weight2_ptr=weight2,
        weight2_stride_K=weight2.stride(1),
        weight2_stride_N=weight2.stride(0),
        bias0_ptr=bias0,
        bias1_ptr=bias1,
        bias2_ptr=bias2,
        output_ptr=output_mat.view((-1, N)),
        output_stride_M=output_mat.stride(0),
        output_stride_N=output_mat.stride(1),
        ACTIVATION=activation.value,
        LAST_ACTIVATION=last_activation,
        BLOCK_SIZE_K1=BLOCK_SIZE_K1,
        M=M,
        K0=K0,
        K1=K1,
        K2=K2,
        N=N,
    )

    return output_mat.reshape(original_shape[:-1] + (N,))


def matmul3_launcher_backward(*args, **kwargs):
    """Backward pass for the fused three-layer matmul (not yet implemented).

    The fused three-layer backward kernel is deferred. Use a one- or two-layer
    fused MLP when gradients are required, or run the three-layer fused MLP
    forward-only.
    """
    raise NotImplementedError(
        "Backward pass for the fused three-layer matmul is not yet implemented."
    )