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

# Check if triton is available
import triton
import triton.language as tl

from physicsnemo.utils.profiling import profile


def _out_of_resources_errors() -> tuple:
    """Triton out-of-resources exception types, tolerant of version layout.

    A persistent-backward autotune config can exceed the device shared-memory
    budget; this Triton raises ``OutOfResources`` at launch rather than pruning
    it, so the launcher catches it and falls back to the tiled backward.
    """
    types: list = []
    try:
        from triton.runtime.errors import OutOfResources

        types.append(OutOfResources)
    except Exception:  # noqa: BLE001 - older/newer Triton may move this
        pass
    try:
        from triton.runtime.autotuner import OutOfResources as _AutoOOR

        types.append(_AutoOOR)
    except Exception:  # noqa: BLE001
        pass
    return tuple(types)


_OUT_OF_RESOURCES = _out_of_resources_errors()

from .activations import (
    Activation,
    activation_dispatch,
    activation_dispatch_bwd,
)
from .load_store import (
    atomic_update_weight,
    load_input_block,
    load_weight_matrix,
    load_weight_matrix_kn,
    write_output_block,
    write_output_tile,
)
from .primitives import (
    forward_matmul_no_activation_with_weights_and_inputs,
    forward_matmul_with_input_and_weights,
    fwd_consume_resident_out_tile,
    fwd_out_tile_reduce_k,
)
from .utils import (
    PRECISION,
    get_cuda_autotune_config_bwd,
    get_cuda_autotune_config_bwd_persistent,
    get_cuda_autotune_config_fwd_persistent,
    get_cuda_autotune_config_fwd_tiled,
    grad_partials,
    persistent_max_width,
)

# Number of resident weight matrices in the two-layer persistent kernel. The
# usable width threshold is derived per-GPU from this and the device's
# shared-memory budget (see persistent_max_width); wider layers fall back to the
# width-tiled kernel, which streams weight slices and never spills.
_PERSISTENT_N_WEIGHTS = 2

# The persistent *backward* keeps both weights resident in *both* orientations:
# the [K, N] (dot) layout for the forward recompute and the transposed [N, K]
# layout for grad_input/grad_hidden. That is ~4 resident weight tiles, so the
# usable-width threshold is derived with n_weights=4 (more conservative than the
# forward's 2). On top of the shared-memory budget the kernel also holds the
# grad_weight accumulators (2 * width^2) register-resident across the grid-stride
# loop, so this threshold is an upper bound -- Triton prunes any config that
# still overflows at autotune time, and wider layers use the tiled backward.
_PERSISTENT_BWD_N_WEIGHTS = 4


@triton.autotune(
    configs=get_cuda_autotune_config_fwd_tiled(),
    # Re-tune per problem shape (see matmul1_kernel for the rationale).
    key=["M", "K0", "K1", "N"],
)
@triton.jit
def matmul2_kernel(
    input_ptr,
    input_stride_M: tl.constexpr,
    input_stride_K: tl.constexpr,
    weight0_ptr,
    weight0_stride_K: tl.constexpr,
    weight0_stride_N: tl.constexpr,
    weight1_ptr,
    weight1_stride_K: tl.constexpr,
    weight1_stride_N: tl.constexpr,
    bias0_ptr,
    bias1_ptr,
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
    N: tl.constexpr,
):
    """
    Fused two-layer MLP forward: ``Y = act?(act(X @ W0.T + b0) @ W1.T + b1)``.

    ``X`` is ``[M, K0]``, ``W0`` is ``[K1, K0]``, ``W1`` is ``[N, K1]`` and the
    output is ``[M, N]``. Designed for tall-skinny matrices (huge ``M``, small
    feature widths).

    The hidden activation tile ``[BLOCK_SIZE_M, K1]`` is the only fully-resident
    state: layer 0 streams the input width ``K0`` in ``BLOCK_SIZE_KR`` chunks to
    build it, then layer 1 streams the output width ``N`` in ``BLOCK_SIZE_N``
    tiles. Neither weight is held resident at full width, so wide layers no
    longer spill; the intermediate never touches DRAM.

    Inputs:
        input_ptr: Pointer to the input tensor
        input_stride_M: Stride for the input tensor along the M dimension (first)
        input_stride_K: Stride for the input tensor along the K dimension (second)
        weight0_ptr/weight1_ptr: Pointers to the two weight tensors ([N, K] layout)
        weight*_stride_K: Stride along the in-features (K) dimension
        weight*_stride_N: Stride along the out-features (N) dimension
        bias0_ptr/bias1_ptr: Optional bias pointers (can be None)
        output_ptr: Pointer to the output tensor
        output_stride_M: Stride for the output tensor along the M dimension (first)
        output_stride_N: Stride for the output tensor along the N dimension (second)
        ACTIVATION: Activation function to apply (enum value)
        LAST_ACTIVATION: Whether to apply the activation to the final output
        BLOCK_SIZE_M: Rows processed per program
        BLOCK_SIZE_KR: Contraction tile streamed in the layer-0 K0 reduction
        BLOCK_SIZE_K1: Full (resident) hidden width
        BLOCK_SIZE_N: Output column tile for layer 1
        M: Number of input rows
        K0: Input feature width
        K1: Hidden width
        N: Output feature width
    """

    pid = tl.program_id(axis=0)

    # Layer 0: reduce over the input width K0 to build the full hidden tile,
    # kept resident across the layer-1 output loop.
    hidden = fwd_out_tile_reduce_k(
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

    hidden = activation_dispatch(hidden, ACTIVATION)

    # Layer 1: H is resident (contraction K1); stream the output width N.
    for n_offset in range(0, N, BLOCK_SIZE_N):
        output = fwd_consume_resident_out_tile(
            hidden,
            weight1_ptr,
            weight1_stride_K,
            weight1_stride_N,
            bias1_ptr,
            n_offset,
            BLOCK_SIZE_K1,
            BLOCK_SIZE_N,
            K1,
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
    key=["M", "K0", "K1", "N"],
)
@triton.jit
def matmul2_kernel_persistent(
    input_ptr,
    input_stride_M: tl.constexpr,
    input_stride_K: tl.constexpr,
    weight0_ptr,
    weight0_stride_K: tl.constexpr,
    weight0_stride_N: tl.constexpr,
    weight1_ptr,
    weight1_stride_K: tl.constexpr,
    weight1_stride_N: tl.constexpr,
    bias0_ptr,
    bias1_ptr,
    output_ptr,
    output_stride_M: tl.constexpr,
    output_stride_N: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAST_ACTIVATION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K0: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    M: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    N: tl.constexpr,
):
    """
    Weight-stationary persistent two-layer MLP forward (narrow widths).

    Same math as :func:`matmul2_kernel`, but the grid is fixed at roughly one
    program per SM. Each program loads both weights (and biases) once into
    registers and then walks its share of the row-blocks in a grid-stride loop,
    streaming the huge activation through the resident weights. This turns the
    weight traffic from "once per row-block" (the tiled kernel) into "once per
    program", which is the dominant cost for multi-layer tall-skinny problems.

    All feature widths are held resident, so ``BLOCK_SIZE_K0`` / ``BLOCK_SIZE_K1``
    / ``BLOCK_SIZE_N`` are the (padded) full widths; only ``BLOCK_SIZE_M`` (the
    streamed row tile) is autotuned. Used only for widths
    <= ``_PERSISTENT_MAX_WIDTH``; wider problems use the tiled kernel.

    Inputs:
        input_ptr: Pointer to the input tensor
        input_stride_M/input_stride_K: Input strides (row, col)
        weight0_ptr/weight1_ptr: Weight pointers ([N, K] layout)
        weight*_stride_K/weight*_stride_N: Weight strides (in-features, out-features)
        bias0_ptr/bias1_ptr: Optional bias pointers (can be None)
        output_ptr: Pointer to the output tensor
        output_stride_M/output_stride_N: Output strides (row, col)
        ACTIVATION: Activation function to apply (enum value)
        LAST_ACTIVATION: Whether to apply the activation to the final output
        BLOCK_SIZE_M: Rows streamed per grid-stride iteration
        BLOCK_SIZE_K0: Full (resident) input width
        BLOCK_SIZE_K1: Full (resident) hidden width
        BLOCK_SIZE_N: Full (resident) output width
        M: Number of input rows
        K0/K1: Input and hidden widths
        N: Output feature width
    """

    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_row_blocks = tl.cdiv(M, BLOCK_SIZE_M)

    # Hoist the weights once; they stay resident (shared memory) for every
    # row-block this program handles. Load directly in [K, N] (dot) layout so
    # only one copy is staged -- halving the resident weight footprint vs
    # load_weight_matrix + tl.trans.
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
        BLOCK_SIZE_N,
        K1,
        N,
    )  # [K1, N]

    if bias0_ptr is not None:
        bias0_idx = tl.arange(0, BLOCK_SIZE_K1)
        bias0 = tl.load(bias0_ptr + bias0_idx, mask=bias0_idx < K1)
    if bias1_ptr is not None:
        bias1_idx = tl.arange(0, BLOCK_SIZE_N)
        bias1 = tl.load(bias1_ptr + bias1_idx, mask=bias1_idx < N)

    # Grid-stride over row-blocks; weights stay put, activations stream through.
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

        hidden = tl.dot(input_block, weight0_t, input_precision=PRECISION)
        if bias0_ptr is not None:
            hidden += bias0
        hidden = activation_dispatch(hidden, ACTIVATION)

        output = tl.dot(hidden, weight1_t, input_precision=PRECISION)
        if bias1_ptr is not None:
            output += bias1
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
def matmul2_launcher(
    input_tensor: torch.Tensor,
    weight0: torch.Tensor,
    weight1: torch.Tensor,
    bias0: torch.Tensor | None = None,
    bias1: torch.Tensor | None = None,
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
    # Remember the weights are stored transposed:
    K1 = weight0.shape[0]

    if len(input_tensor.shape) > 2:
        # Flatten the batch dimension if needed:
        input_tensor = input_tensor.reshape((-1, K0)).contiguous()
    # Gather matrix shapes:
    M = input_tensor.shape[0]
    N = weight1.shape[0]

    # Use the last N for output shape:
    output_shape = input_tensor.shape[:-1] + (N,)

    # Initialize the output:
    output_mat = torch.empty(
        output_shape, dtype=input_tensor.dtype, device=input_tensor.device
    )

    max_resident_width = persistent_max_width(
        input_tensor.device, _PERSISTENT_N_WEIGHTS, input_tensor.element_size()
    )
    if max(K0, K1, N) <= max_resident_width:
        # Narrow widths: weight-stationary persistent kernel. All widths are
        # resident, so pass the full padded widths; the grid is ~one program per
        # SM and each program grid-strides over the row-blocks.
        BLOCK_SIZE_K0 = max(triton.next_power_of_2(K0), 16)
        BLOCK_SIZE_K1 = max(triton.next_power_of_2(K1), 16)
        BLOCK_SIZE_N = max(triton.next_power_of_2(N), 16)
        num_sm = torch.cuda.get_device_properties(
            input_tensor.device
        ).multi_processor_count

        def grid(META):
            return (min(triton.cdiv(M, META["BLOCK_SIZE_M"]), num_sm),)

        matmul2_kernel_persistent[grid](
            input_ptr=input_tensor,
            input_stride_M=input_tensor.stride(0),
            input_stride_K=input_tensor.stride(1),
            weight0_ptr=weight0,
            weight0_stride_K=weight0.stride(1),
            weight0_stride_N=weight0.stride(0),
            weight1_ptr=weight1,
            weight1_stride_K=weight1.stride(1),
            weight1_stride_N=weight1.stride(0),
            bias0_ptr=bias0,
            bias1_ptr=bias1,
            output_ptr=output_mat.view((-1, N)),
            output_stride_M=output_mat.stride(0),
            output_stride_N=output_mat.stride(1),
            ACTIVATION=activation.value,
            LAST_ACTIVATION=last_activation,
            BLOCK_SIZE_K0=BLOCK_SIZE_K0,
            BLOCK_SIZE_K1=BLOCK_SIZE_K1,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            M=M,
            K0=K0,
            K1=K1,
            N=N,
        )
        return output_mat.reshape(original_shape[:-1] + (N,))

    # Wide widths: width-tiled kernel. The hidden width K1 is held fully
    # resident (the activation tile), so its block size is the full padded
    # width. The input-reduction tile (BLOCK_SIZE_KR) and the layer-1 output
    # tile (BLOCK_SIZE_N) are chosen by the autotuner per problem shape.
    BLOCK_SIZE_K1 = max(triton.next_power_of_2(K1), 16)

    def grid(META):
        return (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

    selected_kernel = matmul2_kernel[grid]

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
        bias0_ptr=bias0,
        bias1_ptr=bias1,
        output_ptr=output_mat.view((-1, N)),
        output_stride_M=output_mat.stride(0),
        output_stride_N=output_mat.stride(1),
        ACTIVATION=activation.value,
        LAST_ACTIVATION=last_activation,
        BLOCK_SIZE_K1=BLOCK_SIZE_K1,
        M=M,
        K0=K0,
        K1=K1,
        N=N,
    )

    return output_mat.reshape(original_shape[:-1] + (N,))


def matmul2_launcher_backward(
    input_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    weight0: torch.Tensor,
    weight1: torch.Tensor,
    bias0: torch.Tensor | None = None,
    bias1: torch.Tensor | None = None,
    grad_input: bool = False,
    grad_weight0: bool = False,
    grad_bias0: bool = False,
    grad_weight1: bool = False,
    grad_bias1: bool = False,
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
    K0 = input_tensor.shape[-1]
    K1 = weight0.shape[0]
    N = weight1.shape[0]

    # Flatten batch dimensions if present for kernel compatibility.
    if len(input_tensor.shape) > 2:
        input_tensor = input_tensor.view((-1, K0)).contiguous()
    # Also flatten the grad_output tensor, if needed:
    if len(grad_output.shape) > 2:
        grad_output = grad_output.view((-1, N))

    # Gather matrix shapes.
    M = input_tensor.shape[0]

    # rank-2 weight matrix
    if len(weight0.shape) != 2 or len(weight1.shape) != 2:
        raise ValueError(f"Weight matrix must be rank-2, got shape {weight0.shape}")

    # Ensure bias is contiguous if present.
    if bias0 is not None and not bias0.is_contiguous():
        bias0 = bias0.contiguous()
    if bias1 is not None and not bias1.is_contiguous():
        bias1 = bias1.contiguous()

    # Feature-dim block sizes; BLOCK_SIZE_M (the row tile) is chosen by the
    # autotuner per problem shape.
    BLOCK_SIZE_N = max(triton.next_power_of_2(N), 16)
    BLOCK_SIZE_K0 = max(triton.next_power_of_2(K0), 16)
    BLOCK_SIZE_K1 = max(triton.next_power_of_2(K1), 16)

    if bias0 is None:
        grad_bias0 = False
    if bias1 is None:
        grad_bias1 = False

    itemsize = weight0.element_size()
    num_sm = torch.cuda.get_device_properties(
        input_tensor.device
    ).multi_processor_count

    def _alloc_and_launch(backward_kernel, grid, NUM_PARTIALS):
        """Allocate the per-bucket grad buffers and launch ``backward_kernel``.

        Returns the (possibly ``None``) partial buffers so the caller can reduce
        them. Raised :data:`_OUT_OF_RESOURCES` errors propagate to the caller so
        the persistent path can fall back to the tiled path.
        """
        grad_input_mat = torch.zeros_like(input_tensor) if grad_input else None
        grad_weight0_buf = (
            torch.zeros(
                (NUM_PARTIALS,) + tuple(weight0.shape),
                dtype=weight0.dtype,
                device=weight0.device,
            )
            if grad_weight0
            else None
        )
        grad_weight1_buf = (
            torch.zeros(
                (NUM_PARTIALS,) + tuple(weight1.shape),
                dtype=weight1.dtype,
                device=weight1.device,
            )
            if grad_weight1
            else None
        )
        grad_bias0_buf = (
            torch.zeros(
                (NUM_PARTIALS,) + tuple(bias0.shape),
                dtype=bias0.dtype,
                device=bias0.device,
            )
            if grad_bias0
            else None
        )
        grad_bias1_buf = (
            torch.zeros(
                (NUM_PARTIALS,) + tuple(bias1.shape),
                dtype=bias1.dtype,
                device=bias1.device,
            )
            if grad_bias1
            else None
        )

        backward_kernel[grid](
            input_ptr=input_tensor,
            input_stride_M=input_tensor.stride(0),
            input_stride_K=input_tensor.stride(1),
            weight0_ptr=weight0,
            weight0_stride_K=weight0.stride(1),
            weight0_stride_N=weight0.stride(0),
            weight1_ptr=weight1,
            weight1_stride_K=weight1.stride(1),
            weight1_stride_N=weight1.stride(0),
            bias0_ptr=bias0,
            bias1_ptr=bias1,
            grad_output_ptr=grad_output,
            grad_output_stride_M=grad_output.stride(0),
            grad_output_stride_N=grad_output.stride(1),
            grad_input_ptr=grad_input_mat if grad_input else None,
            grad_input_stride_M=grad_input_mat.stride(0) if grad_input else None,
            grad_input_stride_K=grad_input_mat.stride(1) if grad_input else None,
            grad_weight0_ptr=grad_weight0_buf if grad_weight0 else None,
            grad_weight0_stride_K=grad_weight0_buf.stride(2) if grad_weight0 else None,
            grad_weight0_stride_N=grad_weight0_buf.stride(1) if grad_weight0 else None,
            grad_weight0_buffer_stride=(
                grad_weight0_buf.stride(0) if grad_weight0 else 0
            ),
            grad_weight1_ptr=grad_weight1_buf if grad_weight1 else None,
            grad_weight1_stride_K=grad_weight1_buf.stride(2) if grad_weight1 else None,
            grad_weight1_stride_N=grad_weight1_buf.stride(1) if grad_weight1 else None,
            grad_weight1_buffer_stride=(
                grad_weight1_buf.stride(0) if grad_weight1 else 0
            ),
            grad_bias0_ptr=grad_bias0_buf if grad_bias0 else None,
            grad_bias0_buffer_stride=grad_bias0_buf.stride(0) if grad_bias0 else 0,
            grad_bias1_ptr=grad_bias1_buf if grad_bias1 else None,
            grad_bias1_buffer_stride=grad_bias1_buf.stride(0) if grad_bias1 else 0,
            NUM_PARTIALS=NUM_PARTIALS,
            GRAD_INPUT=grad_input,
            GRAD_WEIGHT0=grad_weight0,
            GRAD_WEIGHT1=grad_weight1,
            GRAD_BIAS0=grad_bias0,
            GRAD_BIAS1=grad_bias1,
            ACTIVATION=activation.value,
            LAST_ACTIVATION=last_activation,
            BLOCK_SIZE_K0=BLOCK_SIZE_K0,
            BLOCK_SIZE_K1=BLOCK_SIZE_K1,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            M=M,
            K0=K0,
            K1=K1,
            N=N,
        )
        return (
            grad_input_mat,
            grad_weight0_buf,
            grad_weight1_buf,
            grad_bias0_buf,
            grad_bias1_buf,
        )

    # Narrow widths: weight-stationary persistent backward. The grid is ~one
    # program per SM, each program keeps the weights resident, grid-strides over
    # the row-blocks, and accumulates grad_weight in registers -- so weights are
    # read once per program (not per row-block) and the grad_weight reduction
    # depth spans many row-blocks (good tensor-core use) instead of a single BM.
    # With <= num_sm programs each owns a unique partial slot (NUM_PARTIALS =
    # num_sm), so the closing atomic_add is uncontended.
    max_resident_width = persistent_max_width(
        input_tensor.device, _PERSISTENT_BWD_N_WEIGHTS, itemsize
    )
    buffers = None
    if max(K0, K1, N) <= max_resident_width:

        def persistent_grid(META):
            return (min(triton.cdiv(M, META["BLOCK_SIZE_M"]), num_sm),)

        try:
            buffers = _alloc_and_launch(
                matmul2_kernel_backward_persistent, persistent_grid, num_sm
            )
        except _OUT_OF_RESOURCES:
            # A persistent config still overflowed shared memory on this device;
            # fall back to the tiled backward below instead of crashing.
            buffers = None

    if buffers is None:
        # Tiled backward. The weight/bias gradients are atomic-accumulated across
        # all row-block programs; splitting that accumulation across NUM_PARTIALS
        # buckets (program ``pid`` -> ``pid % NUM_PARTIALS``) breaks up the
        # contention, and the buckets are summed afterwards. Size the bucket
        # count from the densest grid the autotuner can pick (smallest
        # BLOCK_SIZE_M=16) so per-bucket contention stays bounded for any tile.
        num_programs = triton.cdiv(M, 16)
        slot_bytes = 0
        if grad_weight0:
            slot_bytes += weight0.numel() * itemsize
        if grad_weight1:
            slot_bytes += weight1.numel() * itemsize
        if grad_bias0:
            slot_bytes += bias0.numel() * itemsize
        if grad_bias1:
            slot_bytes += bias1.numel() * itemsize
        tiled_partials = grad_partials(num_programs, slot_bytes)

        def tiled_grid(META):
            return (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

        buffers = _alloc_and_launch(
            matmul2_kernel_backward, tiled_grid, tiled_partials
        )

    (
        grad_input_mat,
        grad_weight0_buf,
        grad_weight1_buf,
        grad_bias0_buf,
        grad_bias1_buf,
    ) = buffers

    # Reshape grad_input to match the original input shape, if computed; reduce
    # the per-bucket partials back down to the final gradient for the rest.
    return (
        grad_input_mat.reshape(original_shape[:-1] + (K0,)) if grad_input else None,
        grad_weight0_buf.sum(dim=0) if grad_weight0 else None,
        grad_weight1_buf.sum(dim=0) if grad_weight1 else None,
        grad_bias0_buf.sum(dim=0) if grad_bias0 else None,
        grad_bias1_buf.sum(dim=0) if grad_bias1 else None,
    )


def _reset_matmul2_bwd_grads(nargs, reset_only=False):
    """Zero the atomic-accumulated grad buffers before an autotune trial.

    The weight/bias gradients for both layers are accumulated with
    ``tl.atomic_add`` into the per-bucket partial buffers, so they must be reset
    before each autotuning trial to avoid double counting across the autotuner's
    repeated benchmark launches. ``grad_input`` uses ``tl.store`` and needs no
    reset. Buffers may be ``None`` when their gradient is not requested.
    """
    for name in (
        "grad_weight0_ptr",
        "grad_weight1_ptr",
        "grad_bias0_ptr",
        "grad_bias1_ptr",
    ):
        buffer = nargs.get(name)
        if buffer is not None:
            buffer.zero_()


@triton.autotune(
    configs=get_cuda_autotune_config_bwd(),
    key=["M", "K0", "K1", "N"],
    pre_hook=_reset_matmul2_bwd_grads,
)
@triton.jit
def matmul2_kernel_backward(
    input_ptr,
    input_stride_M: tl.constexpr,
    input_stride_K: tl.constexpr,
    weight0_ptr,
    weight0_stride_K: tl.constexpr,
    weight0_stride_N: tl.constexpr,
    weight1_ptr,
    weight1_stride_K: tl.constexpr,
    weight1_stride_N: tl.constexpr,
    bias0_ptr,
    bias1_ptr,
    grad_output_ptr,
    grad_output_stride_M: tl.constexpr,
    grad_output_stride_N: tl.constexpr,
    grad_input_ptr,
    grad_input_stride_M: tl.constexpr,
    grad_input_stride_K: tl.constexpr,
    grad_weight0_ptr,
    grad_weight0_stride_K: tl.constexpr,
    grad_weight0_stride_N: tl.constexpr,
    grad_weight0_buffer_stride: tl.constexpr,
    grad_weight1_ptr,
    grad_weight1_stride_K: tl.constexpr,
    grad_weight1_stride_N: tl.constexpr,
    grad_weight1_buffer_stride: tl.constexpr,
    grad_bias0_ptr,
    grad_bias0_buffer_stride: tl.constexpr,
    grad_bias1_ptr,
    grad_bias1_buffer_stride: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    GRAD_INPUT: tl.constexpr,
    GRAD_WEIGHT0: tl.constexpr,
    GRAD_WEIGHT1: tl.constexpr,
    GRAD_BIAS0: tl.constexpr,
    GRAD_BIAS1: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAST_ACTIVATION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K0: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    M: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    N: tl.constexpr,
):
    """
    Backward pass for the matmul2 kernel.
    """

    # Recompute the forward pass output and relevant blocks for backward computation.
    pid = tl.program_id(axis=0)

    # Route this program's gradient contributions to one of NUM_PARTIALS buckets
    # to break up the all-programs-hit-one-buffer atomic contention. The final
    # gradient is the sum over buckets, reduced by the launcher after the kernel.
    bucket = pid % NUM_PARTIALS

    (
        matmul0_output,
        weight0_matrix,
        input0_block,
    ) = forward_matmul_no_activation_with_weights_and_inputs(
        pid,
        input_ptr,
        input_stride_M,
        input_stride_K,
        weight0_ptr,
        weight0_stride_K,
        weight0_stride_N,
        BLOCK_SIZE_M,
        BLOCK_SIZE_K0,
        BLOCK_SIZE_K1,
        M,
        K0,
        K1,
        bias0_ptr,
    )

    # Apply the middle activation
    input1_block = activation_dispatch(matmul0_output, ACTIVATION)

    # The second matmul does not need to read the input block:
    matmul_output, weight1_matrix = forward_matmul_with_input_and_weights(
        input1_block,
        weight1_ptr,
        weight1_stride_K,
        weight1_stride_N,
        BLOCK_SIZE_K1,
        BLOCK_SIZE_N,
        K1,
        N,
        bias1_ptr,
    )

    # Load the grad_outputs block:
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

    # If a last activation was applied in the forward pass, apply its backward function.
    if LAST_ACTIVATION:
        grad_output_block = activation_dispatch_bwd(
            matmul_output, grad_output_block, ACTIVATION
        )

    # Update weights1, bias 1, and compute grad_outputs for the first matmul

    # Compute and atomically accumulate gradients with respect to the weights, if requested.
    if GRAD_WEIGHT1:

        # Computing gradients for the weights involves
        # grad_output_block.T @ input_block
        # This has to be summed across all blocks, however

        # With multiple fused matmuls, we have to get the right input_block.
        # This is the second layer, so the "input" is the output from the
        # first activation

        grad_weight_block = tl.dot(
            tl.trans(grad_output_block), input1_block, input_precision=PRECISION
        )

        # Use atomics to ensure no race-conditions in the accumulation, but spread
        # the contention across NUM_PARTIALS buckets via the per-bucket pointer.
        atomic_update_weight(
            weight_ptr=grad_weight1_ptr + bucket * grad_weight1_buffer_stride,
            weight_updates=grad_weight_block,
            weight_stride_K=grad_weight1_stride_K,
            weight_stride_N=grad_weight1_stride_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K1,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            K=K1,
            N=N,
        )

    # Compute and atomically accumulate gradients with respect to the bias, if requested.
    if GRAD_BIAS1:
        # The bias is the sum of the grad_output (not including activation) over the batch
        # (non-channel) dimension.  Like the weights, we compute on the block and use atomics to store it:

        grad_bias_block = tl.sum(grad_output_block, axis=0)

        grad_bias_offsets = tl.arange(0, BLOCK_SIZE_N)
        grad_bias_masks = grad_bias_offsets < N
        tl.atomic_add(
            grad_bias1_ptr + bucket * grad_bias1_buffer_stride + grad_bias_offsets,
            grad_bias_block,
            mask=grad_bias_masks,
        )

    # We have finished the updates for the weight1 and bias1

    # Now, to update weight0 and bias0, and get grad_inputs
    # Compute the intermediate grad outputs, for the first matmul

    grad_output_block = tl.dot(
        grad_output_block, weight1_matrix, input_precision=PRECISION
    )
    grad_output_block = activation_dispatch_bwd(
        matmul0_output, grad_output_block, ACTIVATION
    )

    if GRAD_INPUT:

        # Load weight0:

        weight0_matrix = load_weight_matrix(
            weight_ptr=weight0_ptr,
            weight_stride_K=weight0_stride_K,
            weight_stride_N=weight0_stride_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K0,
            BLOCK_SIZE_N=BLOCK_SIZE_K1,
            K=K0,
            N=K1,
        )
        grad_input_block = tl.dot(
            grad_output_block, weight0_matrix, input_precision=PRECISION
        )
        write_output_block(
            pid,
            grad_input_ptr,
            grad_input_block,
            grad_input_stride_M,
            grad_input_stride_K,
            BLOCK_SIZE_M,
            BLOCK_SIZE_K0,
            M,
            K0,
        )

    # Now, do the gradients for weight0, bias1

    if GRAD_WEIGHT0:

        # Computing gradients for the weights involves
        # grad_output_block.T @ input_block
        # This has to be summed across all blocks, however.

        grad_weight_block = tl.dot(
            tl.trans(grad_output_block), input0_block, input_precision=PRECISION
        )

        # Use atomics to ensure no race-conditions in the accumulation, but spread
        # the contention across NUM_PARTIALS buckets via the per-bucket pointer.
        atomic_update_weight(
            weight_ptr=grad_weight0_ptr + bucket * grad_weight0_buffer_stride,
            weight_updates=grad_weight_block,
            weight_stride_K=grad_weight0_stride_K,
            weight_stride_N=grad_weight0_stride_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K0,
            BLOCK_SIZE_N=BLOCK_SIZE_K1,
            K=K0,
            N=K1,
        )

    # Compute and atomically accumulate gradients with respect to the bias, if requested.
    if GRAD_BIAS0:
        # The bias is the sum of the grad_output (not including activation) over the batch
        # (non-channel) dimension.  Like the weights, we compute on the block and use atomics to store it:

        grad_bias_block = tl.sum(grad_output_block, axis=0)

        grad_bias_offsets = tl.arange(0, BLOCK_SIZE_K1)
        grad_bias_masks = grad_bias_offsets < K1
        tl.atomic_add(
            grad_bias0_ptr + bucket * grad_bias0_buffer_stride + grad_bias_offsets,
            grad_bias_block,
            mask=grad_bias_masks,
        )

    # DONE!


@triton.autotune(
    configs=get_cuda_autotune_config_bwd_persistent(),
    key=["M", "K0", "K1", "N"],
    pre_hook=_reset_matmul2_bwd_grads,
)
@triton.jit
def matmul2_kernel_backward_persistent(
    input_ptr,
    input_stride_M: tl.constexpr,
    input_stride_K: tl.constexpr,
    weight0_ptr,
    weight0_stride_K: tl.constexpr,
    weight0_stride_N: tl.constexpr,
    weight1_ptr,
    weight1_stride_K: tl.constexpr,
    weight1_stride_N: tl.constexpr,
    bias0_ptr,
    bias1_ptr,
    grad_output_ptr,
    grad_output_stride_M: tl.constexpr,
    grad_output_stride_N: tl.constexpr,
    grad_input_ptr,
    grad_input_stride_M: tl.constexpr,
    grad_input_stride_K: tl.constexpr,
    grad_weight0_ptr,
    grad_weight0_stride_K: tl.constexpr,
    grad_weight0_stride_N: tl.constexpr,
    grad_weight0_buffer_stride: tl.constexpr,
    grad_weight1_ptr,
    grad_weight1_stride_K: tl.constexpr,
    grad_weight1_stride_N: tl.constexpr,
    grad_weight1_buffer_stride: tl.constexpr,
    grad_bias0_ptr,
    grad_bias0_buffer_stride: tl.constexpr,
    grad_bias1_ptr,
    grad_bias1_buffer_stride: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    GRAD_INPUT: tl.constexpr,
    GRAD_WEIGHT0: tl.constexpr,
    GRAD_WEIGHT1: tl.constexpr,
    GRAD_BIAS0: tl.constexpr,
    GRAD_BIAS1: tl.constexpr,
    ACTIVATION: tl.constexpr,
    LAST_ACTIVATION: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K0: tl.constexpr,
    BLOCK_SIZE_K1: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    M: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    N: tl.constexpr,
):
    """Weight-stationary persistent two-layer backward (narrow widths).

    Same gradients as :func:`matmul2_kernel_backward`, but the grid is fixed at
    ~one program per SM. Each program:

    - loads each weight once, keeping only the ``[K, N]`` (dot) orientation
      resident. The gradient contractions that would normally need the
      transposed weight (grad_hidden, grad_input) are reorganized so the
      transpose falls on the *per-iteration activation tile* instead -- those are
      transient, whereas a transposed weight is loop-invariant and would be
      hoisted resident, doubling the (budget-binding) weight footprint;
    - grid-strides over its share of the row-blocks, recomputing the forward and
      contracting the gradients block by block;
    - accumulates ``grad_weight0`` / ``grad_weight1`` (and the biases) in
      *registers* across all of its row-blocks, writing a single partial at the
      end.

    This fixes the two costs the tiled backward cannot: weights are read once per
    program (not once per row-block), and the ``grad_weight`` reduction
    (``grad^T @ activation``) accumulates over many row-blocks instead of a single
    ``BLOCK_SIZE_M``, so it uses the tensor cores efficiently while a *small*
    ``BLOCK_SIZE_M`` keeps occupancy high. With <= ``num_sm`` programs each owns a
    unique partial slot (``pid % NUM_PARTIALS == pid``), so the closing
    ``atomic_add`` is uncontended; the launcher reduces the partials.

    All widths are resident, so ``BLOCK_SIZE_K0`` / ``BLOCK_SIZE_K1`` /
    ``BLOCK_SIZE_N`` are the (padded) full widths; only ``BLOCK_SIZE_M`` is
    autotuned. Used only for widths <= the backward resident-width threshold.
    """

    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_row_blocks = tl.cdiv(M, BLOCK_SIZE_M)

    # grid <= num_sm == NUM_PARTIALS, so each program owns a unique bucket.
    bucket = pid % NUM_PARTIALS

    # Resident weights, loaded once in the [K, N] (dot) orientation only. The
    # grad_hidden / grad_input dots are reorganized below so they consume this
    # same orientation (transposing the activation, not the weight), keeping just
    # one resident copy per weight.
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
        BLOCK_SIZE_N,
        K1,
        N,
    )  # [K1, N]

    if bias0_ptr is not None:
        bias0_idx = tl.arange(0, BLOCK_SIZE_K1)
        bias0 = tl.load(bias0_ptr + bias0_idx, mask=bias0_idx < K1)
    if bias1_ptr is not None:
        bias1_idx = tl.arange(0, BLOCK_SIZE_N)
        bias1 = tl.load(bias1_ptr + bias1_idx, mask=bias1_idx < N)

    # Register-resident gradient accumulators, summed across this program's
    # row-blocks. grad_weight0 is [K1, K0] (out, in) and grad_weight1 is [N, K1].
    if GRAD_WEIGHT0:
        acc_grad_weight0 = tl.zeros((BLOCK_SIZE_K1, BLOCK_SIZE_K0), dtype=tl.float32)
    if GRAD_WEIGHT1:
        acc_grad_weight1 = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_K1), dtype=tl.float32)
    if GRAD_BIAS0:
        acc_grad_bias0 = tl.zeros((BLOCK_SIZE_K1,), dtype=tl.float32)
    if GRAD_BIAS1:
        acc_grad_bias1 = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    # Grid-stride over row-blocks; weights stay put, activations stream through.
    for row_block in range(pid, num_row_blocks, num_programs):
        # --- Recompute the forward for this row-block ---
        input_block = load_input_block(
            row_block,
            input_ptr,
            input_stride_M,
            input_stride_K,
            BLOCK_SIZE_M,
            BLOCK_SIZE_K0,
            M,
            K0,
        )  # [BM, K0]

        hidden_pre = tl.dot(input_block, weight0_t, input_precision=PRECISION)
        if bias0_ptr is not None:
            hidden_pre += bias0
        hidden = activation_dispatch(hidden_pre, ACTIVATION)  # [BM, K1]

        # --- Gradient of the output (pre last-activation if applicable) ---
        grad_output_block = load_input_block(
            row_block,
            grad_output_ptr,
            grad_output_stride_M,
            grad_output_stride_N,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            M,
            N,
        )  # [BM, N]

        if LAST_ACTIVATION:
            output_pre = tl.dot(hidden, weight1_t, input_precision=PRECISION)
            if bias1_ptr is not None:
                output_pre += bias1
            grad_output_block = activation_dispatch_bwd(
                output_pre, grad_output_block, ACTIVATION
            )

        # Transpose the (transient) grad-output tile once; reused below.
        grad_output_t = tl.trans(grad_output_block)  # [N, BM]

        # --- Layer 1 weight/bias grads: grad_out^T @ hidden ---
        if GRAD_WEIGHT1:
            acc_grad_weight1 += tl.dot(
                grad_output_t, hidden, input_precision=PRECISION
            )  # [N, K1]
        if GRAD_BIAS1:
            acc_grad_bias1 += tl.sum(grad_output_block, axis=0)

        # --- Backprop into the hidden activation ---
        # grad_hidden = grad_out @ W1. Computed as (W1^T grad_out^T)^T using the
        # resident [K1, N] weight and the transposed activation, so no transposed
        # weight is materialized: tl.dot(weight1_t, grad_out^T) = grad_hidden^T.
        grad_hidden_t = tl.dot(
            weight1_t, grad_output_t, input_precision=PRECISION
        )  # [K1, BM]
        grad_hidden = tl.trans(grad_hidden_t)  # [BM, K1]
        grad_hidden_pre = activation_dispatch_bwd(hidden_pre, grad_hidden, ACTIVATION)

        # Transpose the hidden-grad tile once; reused for grad_input and grad_W0.
        grad_hidden_pre_t = tl.trans(grad_hidden_pre)  # [K1, BM]

        # --- grad_input = grad_hidden_pre @ W0 (written per row-block) ---
        # Same trick: tl.dot(weight0_t, grad_hidden_pre^T) = grad_input^T.
        if GRAD_INPUT:
            grad_input_t = tl.dot(
                weight0_t, grad_hidden_pre_t, input_precision=PRECISION
            )  # [K0, BM]
            grad_input_block = tl.trans(grad_input_t)  # [BM, K0]
            write_output_block(
                row_block,
                grad_input_ptr,
                grad_input_block,
                grad_input_stride_M,
                grad_input_stride_K,
                BLOCK_SIZE_M,
                BLOCK_SIZE_K0,
                M,
                K0,
            )

        # --- Layer 0 weight/bias grads: grad_hidden_pre^T @ input ---
        if GRAD_WEIGHT0:
            acc_grad_weight0 += tl.dot(
                grad_hidden_pre_t, input_block, input_precision=PRECISION
            )  # [K1, K0]
        if GRAD_BIAS0:
            acc_grad_bias0 += tl.sum(grad_hidden_pre, axis=0)

    # --- Write this program's single partial (uncontended atomic) ---
    if GRAD_WEIGHT1:
        atomic_update_weight(
            weight_ptr=grad_weight1_ptr + bucket * grad_weight1_buffer_stride,
            weight_updates=acc_grad_weight1,
            weight_stride_K=grad_weight1_stride_K,
            weight_stride_N=grad_weight1_stride_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K1,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            K=K1,
            N=N,
        )
    if GRAD_BIAS1:
        grad_bias1_offsets = tl.arange(0, BLOCK_SIZE_N)
        tl.atomic_add(
            grad_bias1_ptr + bucket * grad_bias1_buffer_stride + grad_bias1_offsets,
            acc_grad_bias1,
            mask=grad_bias1_offsets < N,
        )
    if GRAD_WEIGHT0:
        atomic_update_weight(
            weight_ptr=grad_weight0_ptr + bucket * grad_weight0_buffer_stride,
            weight_updates=acc_grad_weight0,
            weight_stride_K=grad_weight0_stride_K,
            weight_stride_N=grad_weight0_stride_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K0,
            BLOCK_SIZE_N=BLOCK_SIZE_K1,
            K=K0,
            N=K1,
        )
    if GRAD_BIAS0:
        grad_bias0_offsets = tl.arange(0, BLOCK_SIZE_K1)
        tl.atomic_add(
            grad_bias0_ptr + bucket * grad_bias0_buffer_stride + grad_bias0_offsets,
            acc_grad_bias0,
            mask=grad_bias0_offsets < K1,
        )

    # DONE!