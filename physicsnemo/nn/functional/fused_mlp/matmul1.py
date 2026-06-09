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

from .activations import (
    Activation,
    activation_dispatch,
    activation_dispatch_bwd,
)
from .load_store import (
    atomic_update_weight,
    load_input_block,
    write_output_block,
    write_output_tile,
)
from .primitives import (
    forward_matmul_no_activation_with_weights_and_inputs,
    fwd_out_tile_reduce_k,
)
from .utils import (
    PRECISION,
    get_cuda_autotune_config_bwd,
    get_cuda_autotune_config_fwd_tiled,
    grad_partials,
)


@triton.autotune(
    configs=get_cuda_autotune_config_fwd_tiled(),
    # Re-tune per problem shape so each width picks its own tiling. With key=[]
    # the autotuner benchmarks once for the first shape and reuses that config
    # for every other shape, which is catastrophic when the best tile sizes
    # track the feature dims.
    key=["M", "K", "N"],
)
@triton.jit
def matmul1_kernel(
    input_ptr,
    input_stride_M: tl.constexpr,
    input_stride_K: tl.constexpr,
    weight_ptr,
    weight_stride_K: tl.constexpr,
    weight_stride_N: tl.constexpr,
    bias_ptr,
    output_ptr,
    output_stride_M: tl.constexpr,
    output_stride_N: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAST_ACTIVATION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_KR: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    """
    Matrix multiplication kernel for A @ B.T, where:
        - A has shape [M, K]
        - B has shape [N, K]
        - Output has shape [M, N]

    Optimized for small matmuls on tall, skinny matrices. The contraction
    dimension K is streamed in ``BLOCK_SIZE_KR`` chunks and the output columns N
    are emitted in ``BLOCK_SIZE_N`` tiles, so neither the input nor the weight is
    held resident at full width. When the autotuner selects tiles >= the feature
    dims the loops run once, reproducing a single full-width matmul.

    Parameters:
        input_ptr:         Pointer to the input tensor A.
        input_stride_M:    Stride of A along the M (row) dimension.
        input_stride_K:    Stride of A along the K (column) dimension.
        weight_ptr:        Pointer to the weight tensor B.
        weight_stride_K:   Stride of B along the K (column) dimension.
        weight_stride_N:   Stride of B along the N (row) dimension.
        bias_ptr:          Pointer to the bias tensor (can be None).
        output_ptr:        Pointer to the output tensor.
        output_stride_M:   Stride of the output along the M (row) dimension.
        output_stride_N:   Stride of the output along the N (column) dimension.
        ACTIVATION:        Activation function to apply to the output (enum value).
        LAST_ACTIVATION:   Whether to apply the activation function to the output (bool).
        BLOCK_SIZE_M:      Block size for the M (row) dimension.
        BLOCK_SIZE_KR:     Contraction tile streamed in the K-reduction loop.
        BLOCK_SIZE_N:      Output column tile.
        M:                 Number of rows in the input tensor A.
        K:                 Number of columns in the input tensor A (and B).
        N:                 Number of rows in the weight tensor B (and output columns).
    """

    pid = tl.program_id(axis=0)

    # Emit the output one N-tile at a time; each tile reduces over the full K.
    for n_offset in range(0, N, BLOCK_SIZE_N):
        output = fwd_out_tile_reduce_k(
            pid,
            input_ptr,
            input_stride_M,
            input_stride_K,
            weight_ptr,
            weight_stride_K,
            weight_stride_N,
            bias_ptr,
            n_offset,
            BLOCK_SIZE_M,
            BLOCK_SIZE_KR,
            BLOCK_SIZE_N,
            M,
            K,
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


def matmul1_launcher(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: Activation = Activation.NONE,
    last_activation: bool = False,
):
    """
    Launches a fused matrix multiplication (matmul) operation with optional bias addition and activation,
    using a custom Triton kernel. This function mimics the behavior of `torch.nn.Linear` but with
    additional support for fused activations and optimized execution for small, tall-skinny matrices.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (..., K), where ... denotes optional batch dimensions.
        weight (torch.Tensor): Weight matrix of shape (N, K).
        bias (torch.Tensor, optional): Optional bias tensor of shape (N,). Defaults to None.
        activation (Activation, optional): Activation function to apply after matmul. Defaults to Activation.NONE.
        last_activation (bool, optional): Whether to apply the activation function to the output. Defaults to False.

    Returns:
        torch.Tensor: Output tensor of shape (..., N), matching the batch dimensions of the input.
    """

    # Replicate torch.Linear() functionality + fused activation

    # Flatten the input matrix along all but the last dimension if needed.
    # Save the original shape for reshaping the output later.
    original_shape = input_tensor.shape

    if len(input_tensor.shape) > 2:
        # Flatten the batch dimensions into a single leading dimension.
        input_tensor = input_tensor.reshape((-1, input_tensor.shape[-1])).contiguous()

    # Ensure the bias is contiguous in memory if present.
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    # Gather matrix shapes.
    K = input_tensor.shape[-1]
    M = input_tensor.shape[0]
    N = weight.shape[0]

    # Compute the output shape and allocate the output tensor.
    output_shape = input_tensor.shape[:-1] + (N,)

    output_mat = torch.empty(
        output_shape, dtype=input_tensor.dtype, device=input_tensor.device
    ).view((-1, N))

    # Block sizes (BLOCK_SIZE_M / BLOCK_SIZE_KR / BLOCK_SIZE_N) are chosen by the
    # autotuner per problem shape; only the true shapes are passed here.
    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

    selected_kernel = matmul1_kernel[grid]

    # Launch the Triton kernel with the appropriate arguments.
    # Note: The weight matrix is stored as [N, K], so strides are reversed.
    selected_kernel(
        input_ptr=input_tensor,
        input_stride_M=input_tensor.stride(0),
        input_stride_K=input_tensor.stride(1),
        weight_ptr=weight,
        weight_stride_K=weight.stride(1),
        weight_stride_N=weight.stride(0),
        bias_ptr=bias,
        output_ptr=output_mat,
        output_stride_M=output_mat.stride(0),
        output_stride_N=output_mat.stride(1),
        ACTIVATION=activation.value,
        LAST_ACTIVATION=last_activation,
        M=M,
        K=K,
        N=N,
    )

    # Reshape the output to match the original input batch dimensions.
    return output_mat.reshape(original_shape[:-1] + (N,))


def _reset_matmul1_bwd_grads(nargs, reset_only=False):
    """Zero the atomic-accumulated grad buffers before an autotune trial.

    ``grad_weight``/``grad_bias`` are accumulated with ``tl.atomic_add``. The
    autotuner re-runs the kernel many times against the real output buffers when
    benchmarking configs, so without this hook the gradients would be summed
    once per trial. ``grad_input`` uses ``tl.store`` (overwrite) and needs no
    reset. Buffers may be ``None`` when their gradient is not requested.
    """
    for name in ("grad_weight_ptr", "grad_bias_ptr"):
        buffer = nargs.get(name)
        if buffer is not None:
            buffer.zero_()


@triton.autotune(
    configs=get_cuda_autotune_config_bwd(),
    key=["M", "K", "N"],
    pre_hook=_reset_matmul1_bwd_grads,
)
@triton.jit
def matmul1_kernel_backward(
    input_ptr,
    input_stride_M,
    input_stride_K,
    weight_ptr,
    weight_stride_K,
    weight_stride_N,
    bias_ptr,
    grad_output_ptr,
    grad_output_stride_M,
    grad_output_stride_N,
    grad_input_ptr,
    grad_input_stride_M,
    grad_input_stride_K,
    grad_weight_ptr,
    grad_weight_stride_K,
    grad_weight_stride_N,
    grad_weight_buffer_stride: tl.constexpr,
    grad_bias_ptr,
    grad_bias_buffer_stride: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    GRAD_INPUT: tl.constexpr,
    GRAD_WEIGHT: tl.constexpr,
    GRAD_BIAS: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAST_ACTIVATION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    """
    Triton kernel for the backward pass of a fused matrix multiplication with optional bias and activation.

    This kernel computes gradients with respect to the input, weight, and bias for a single fused matmul layer,
    optionally applying the backward of a fused activation function. It is designed for tall-skinny matrices and
    uses atomic accumulation for weight and bias gradients.

    Args:
        input_ptr:         Pointer to the input tensor A from the forward pass.
        input_stride_M:    Stride of A along the M (row) dimension.
        input_stride_K:    Stride of A along the K (column) dimension.
        weight_ptr:        Pointer to the weight tensor B from the forward pass.
        weight_stride_K:   Stride of B along the K (column) dimension.
        weight_stride_N:   Stride of B along the N (row) dimension.
        bias_ptr:          Pointer to the bias tensor (can be None).
        grad_output_ptr:   Pointer to the gradient of the output tensor (from upstream).
        grad_output_stride_M: Stride of grad_output along the M (row) dimension.
        grad_output_stride_N: Stride of grad_output along the N (column) dimension.
        grad_input_ptr:    Pointer to the gradient tensor for the input (to be written, can be None).
        grad_input_stride_M: Stride for grad_input along M.
        grad_input_stride_K: Stride for grad_input along K.
        grad_weight_ptr:   Pointer to the gradient tensor for the weights (to be written, can be None).
        grad_weight_stride_K: Stride for grad_weight along K.
        grad_weight_stride_N: Stride for grad_weight along N.
        grad_bias_ptr:     Pointer to the gradient tensor for the bias (to be written, can be None).
        GRAD_INPUT:        Whether to compute input gradients (bool).
        GRAD_WEIGHT:       Whether to compute weight gradients (bool).
        GRAD_BIAS:         Whether to compute bias gradients (bool).
        ACTIVATION:        Enum value for the activation function used in the forward pass.
        LAST_ACTIVATION:   Whether the activation was applied in the forward pass (bool).
        BLOCK_SIZE_M:      Block size for the M (row) dimension.
        BLOCK_SIZE_K:      Block size for the K (column) dimension.
        BLOCK_SIZE_N:      Block size for the N (column/output) dimension.
        M:                 Number of rows in the input tensor A.
        K:                 Number of columns in the input tensor A (and B).
        N:                 Number of rows in the weight tensor B (and output columns).
    """

    # Recompute the forward pass output and relevant blocks for backward computation.
    pid = tl.program_id(axis=0)

    # Route this program's gradient contributions to one of NUM_PARTIALS buckets
    # to break up the all-programs-hit-one-buffer atomic contention. The final
    # gradient is the sum over buckets, reduced by the launcher after the kernel.
    bucket = pid % NUM_PARTIALS

    (
        matmul_output,
        weight_matrix,
        input_block,
    ) = forward_matmul_no_activation_with_weights_and_inputs(
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
    )
    # We've now caught up to the forward pass, start going backwards

    # For a single fused matmul, it's not necessary to compute the activation directly.
    # Instead, we feed the input of the actication and it's output grads to
    # the grad computation

    grad_output_block = load_input_block(
        pid,
        grad_output_ptr,
        grad_output_stride_M,
        grad_output_stride_N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        M,
        N,
    )

    # If an activation was applied in the forward pass, apply its backward function.
    if LAST_ACTIVATION:
        grad_output_block = activation_dispatch_bwd(
            matmul_output, grad_output_block, ACTIVATION
        )

    # Compute and write gradients with respect to the input, if requested.
    if GRAD_INPUT:
        grad_input_block = tl.dot(
            grad_output_block, weight_matrix, input_precision=PRECISION
        )
        write_output_block(
            pid,
            grad_input_ptr,
            grad_input_block,
            grad_input_stride_M,
            grad_input_stride_K,
            BLOCK_SIZE_M,
            BLOCK_SIZE_K,
            M,
            K,
        )

    # Compute and atomically accumulate gradients with respect to the weights, if requested.
    if GRAD_WEIGHT:

        # Computing gradients for the weights involves
        # grad_output_block.T @ input_block
        # This has to be summed across all blocks, however.

        grad_weight_block = tl.dot(
            tl.trans(grad_output_block), input_block, input_precision=PRECISION
        )

        # Use atomics to ensure no race-conditions in the accumulation, but spread
        # the contention across NUM_PARTIALS buckets via the per-bucket pointer.
        atomic_update_weight(
            weight_ptr=grad_weight_ptr + bucket * grad_weight_buffer_stride,
            weight_updates=grad_weight_block,
            weight_stride_K=grad_weight_stride_K,
            weight_stride_N=grad_weight_stride_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            K=K,
            N=N,
        )

    # Compute and atomically accumulate gradients with respect to the bias, if requested.
    if GRAD_BIAS:
        # The bias is the sum of the grad_output (not including activation) over the batch
        # (non-channel) dimension.  Like the weights, we compute on the block and use atomics to store it:

        grad_bias_block = tl.sum(grad_output_block, axis=0)

        grad_bias_offsets = tl.arange(0, BLOCK_SIZE_N)
        grad_bias_masks = grad_bias_offsets < N
        tl.atomic_add(
            grad_bias_ptr + bucket * grad_bias_buffer_stride + grad_bias_offsets,
            grad_bias_block,
            mask=grad_bias_masks,
        )

    # DONE!


def matmul1_launcher_backward(
    input_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    grad_input: bool = False,
    grad_weight: bool = False,
    grad_bias: bool = False,
    activation: Activation = Activation.NONE,
    last_activation: bool = False,
):
    """
    Launches the backward pass for the fused matmul operation, computing gradients as requested.

    This function prepares and launches the Triton backward kernel for a fused matmul layer with optional
    bias and activation, mimicking the backward pass of `torch.nn.Linear` with fused activation.
    It is primarily intended for unit testing and not for full workloads.

    Args:
        input_tensor (torch.Tensor): Input tensor from the forward pass, shape (..., K).
        grad_output (torch.Tensor): Gradient of the output tensor, shape (..., N).
        weight (torch.Tensor): Weight matrix from the forward pass, shape (N, K).
        bias (torch.Tensor, optional): Bias tensor from the forward pass, shape (N,). Defaults to None.
        grad_input (bool, optional): Whether to compute gradients with respect to the input. Defaults to False.
        grad_weight (bool, optional): Whether to compute gradients with respect to the weights. Defaults to False.
        grad_bias (bool, optional): Whether to compute gradients with respect to the bias. Defaults to False.
        activation (Activation, optional): Activation function used in the forward pass. Defaults to Activation.NONE.
        last_activation (bool, optional): Whether the activation was applied in the forward pass. Defaults to False.

    Returns:
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            Gradients with respect to input, weight, and bias, in that order. Any not requested will be None.
            - grad_input: shape (..., K) if grad_input is True, else None
            - grad_weight: shape (N, K) if grad_weight is True, else None
            - grad_bias: shape (N,) if grad_bias is True, else None
    """

    # Save the original input shape for reshaping the input gradient later.
    original_shape = input_tensor.shape
    K = input_tensor.shape[-1]
    N = weight.shape[0]

    # Flatten batch dimensions if present for kernel compatibility.
    if len(input_tensor.shape) > 2:
        input_tensor = input_tensor.view((-1, K)).contiguous()
    # Also flatten the grad_output tensor, if needed:
    if len(grad_output.shape) > 2:
        grad_output = grad_output.view((-1, N))

    # Gather matrix shapes.
    M = input_tensor.shape[0]
    B = bias.shape[0] if bias is not None else None

    # rank-2 weight matrix
    if len(weight.shape) != 2:
        raise ValueError(f"Weight matrix must be rank-2, got shape {weight.shape}")
    if B is not None and B != N:
        raise ValueError(f"Bias shape {B} must match weight output dimension {N}")

    # Ensure bias is contiguous if present.
    if bias is not None and not bias.is_contiguous():
        bias = bias.contiguous()

    # Feature-dim block sizes; BLOCK_SIZE_M (the row tile) is chosen by the
    # autotuner per problem shape.
    BLOCK_SIZE_N = max(triton.next_power_of_2(N), 16)
    BLOCK_SIZE_K = max(triton.next_power_of_2(max(N, K)), 16)

    if bias is None:
        grad_bias = False

    # The weight/bias gradients are atomic-accumulated across all row-block
    # programs and reduced at the end. Spreading the accumulation across
    # NUM_PARTIALS buckets (program ``pid`` -> ``pid % NUM_PARTIALS``) mitigates
    # the atomic contention; grad_input uses tl.store, so it stays single. Size
    # the bucket count from the densest grid the autotuner can pick (smallest
    # BLOCK_SIZE_M=16) so per-bucket contention stays bounded for any row tile.
    num_programs = triton.cdiv(M, 16)
    itemsize = weight.element_size()
    slot_bytes = 0
    if grad_weight:
        slot_bytes += weight.numel() * itemsize
    if grad_bias:
        slot_bytes += bias.numel() * itemsize
    NUM_PARTIALS = grad_partials(num_programs, slot_bytes)

    # Now, create output tensors as needed:
    if grad_input:
        grad_input_mat = torch.zeros_like(input_tensor)
    if grad_weight:
        grad_weight_buf = torch.zeros(
            (NUM_PARTIALS,) + tuple(weight.shape),
            dtype=weight.dtype,
            device=weight.device,
        )
    if grad_bias:
        grad_bias_buf = torch.zeros(
            (NUM_PARTIALS,) + tuple(bias.shape),
            dtype=bias.dtype,
            device=bias.device,
        )

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

    # Launch the Triton backward kernel.
    matmul1_kernel_backward[grid](
        input_ptr=input_tensor,
        input_stride_M=input_tensor.stride(0),
        input_stride_K=input_tensor.stride(1),
        weight_ptr=weight,
        weight_stride_K=weight.stride(1),
        weight_stride_N=weight.stride(0),
        bias_ptr=bias,
        grad_output_ptr=grad_output,
        grad_output_stride_M=grad_output.stride(0),
        grad_output_stride_N=grad_output.stride(1),
        grad_input_ptr=grad_input_mat if grad_input else None,
        grad_input_stride_M=grad_input_mat.stride(0) if grad_input else None,
        grad_input_stride_K=grad_input_mat.stride(1) if grad_input else None,
        grad_weight_ptr=grad_weight_buf if grad_weight else None,
        grad_weight_stride_K=grad_weight_buf.stride(2) if grad_weight else None,
        grad_weight_stride_N=grad_weight_buf.stride(1) if grad_weight else None,
        grad_weight_buffer_stride=grad_weight_buf.stride(0) if grad_weight else 0,
        grad_bias_ptr=grad_bias_buf if grad_bias else None,
        grad_bias_buffer_stride=grad_bias_buf.stride(0) if grad_bias else 0,
        NUM_PARTIALS=NUM_PARTIALS,
        GRAD_INPUT=grad_input,
        GRAD_WEIGHT=grad_weight,
        GRAD_BIAS=grad_bias,
        ACTIVATION=activation.value,
        LAST_ACTIVATION=last_activation,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        M=M,
        K=K,
        N=N,
    )

    # Reshape grad_input to match the original input shape, if computed; reduce
    # the per-bucket partials back down to the final gradient for the rest.
    return (
        grad_input_mat.reshape(original_shape[:-1] + (K,)) if grad_input else None,
        grad_weight_buf.sum(dim=0) if grad_weight else None,
        grad_bias_buf.sum(dim=0) if grad_bias else None,
    )