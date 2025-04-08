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

from typing import List

import torch
import torch.fx as fx

from ..mlp import FusedMLP

# Check if warp is available
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


def get_cuda_autotune_config() -> List[triton.Config]:
    """
    Returns a list of Triton configurations for auto-tuning the CUDA kernels.

    Returns:
        List[triton.Config]: A list of Triton configurations to use for auto-tuning.
    """
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": 16,
            },
            num_stages=1,
            num_warps=4,
        ),
    ]


@triton.autotune(configs=get_cuda_autotune_config(), key=["M", "NK"])
@triton.jit
# This implementation explicitly assumes N == K
def matmul_kernel(
    # Pointers to matrices
    input_ptr,
    output_ptr,
    w0_ptr,
    b0_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    w3_ptr,
    b3_ptr,
    w4_ptr,
    b4_ptr,
    num_layers: tl.constexpr,
    # Matrix dimensions
    M,
    NK,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_NK: tl.constexpr,
    ACTIVATION: tl.constexpr,  #
    OUTPUT_DTYPE: tl.constexpr,
):
    """
    Triton kernel for fused MLP operations with multiple matrix multiplications.

    This kernel performs multiple matrix multiplications in sequence with the given weights
    and biases, applying activations as needed.

    Args:
        input_ptr: Pointer to input tensor
        output_ptr: Pointer to output tensor
        w0_ptr through w4_ptr: Pointers to weight matrices
        b0_ptr through b4_ptr: Pointers to bias vectors
        num_layers: Number of layers to process
        M: Batch dimension size
        NK: Feature dimension size
        stride_am: Stride for moving along the batch dimension in input
        stride_ak: Stride for moving along the feature dimension in input
        stride_bk: Stride for moving along the input dimension in weights
        stride_bn: Stride for moving along the output dimension in weights
        BLOCK_SIZE_M: Block size for batch dimension
        BLOCK_SIZE_NK: Block size for feature dimension
        ACTIVATION: Type of activation function to use
        OUTPUT_DTYPE: Data type for the output tensor
    """
    pid = tl.program_id(axis=0)

    # We load blocks of A that take the full K size, for BLOCK_SIZE_M rows
    # Start at pid*BLOCK_SIZE_M
    input_batchdim_offsets = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    # Use the entire NK size:
    input_feature_offsets = tl.arange(0, BLOCK_SIZE_NK) * stride_ak
    # Create 1D masks:
    m_masks = input_batchdim_offsets < M
    k_masks = input_feature_offsets < NK

    # Use shaping to make this 2D:
    input_offsets = (
        input_batchdim_offsets[:, None] * stride_am
        + input_feature_offsets[None, :] * stride_ak
    )
    a_masks = m_masks[:, None] & k_masks[None, :]

    input_ptrs = input_ptr + input_offsets

    # Load the input block:
    out = tl.load(input_ptrs, mask=a_masks, other=0.0)

    # for i in tl.staticrange(num_layers):
    # weight_ptr = weight_ptrs[i]
    # out = single_matmul(out, weight_ptr, stride_bk, stride_bn, BLOCK_SIZE_NK, NK)

    # This is an unrolled loop over the weights:
    out = single_matmul(out, w0_ptr, b0_ptr, stride_bk, stride_bn, BLOCK_SIZE_NK, NK)

    # if num_layers > 1:
    #     out = single_matmul(out, w1_ptr, b1_ptr, stride_bk, stride_bn, BLOCK_SIZE_NK, NK)
    # if num_layers > 2:
    #     out = single_matmul(out, w2_ptr, b2_ptr, stride_bk, stride_bn, BLOCK_SIZE_NK, NK)
    # if num_layers > 3:
    #     out = single_matmul(out, w3_ptr, b3_ptr, stride_bk, stride_bn, BLOCK_SIZE_NK, NK)
    # if num_layers > 4:
    #     out = single_matmul(out, w4_ptr, b4_ptr, stride_bk, stride_bn, BLOCK_SIZE_NK, NK)

    # This can eventually be updated with the right output feature offsets
    output_offsets = (
        input_batchdim_offsets[:, None] * stride_am
        + input_feature_offsets[None, :] * stride_ak
    )
    # Write the output to C:
    output_ptrs = output_ptr + output_offsets
    tl.store(output_ptrs, out, mask=a_masks)


@triton.jit
def single_matmul(
    input_block, weight_ptr, bias_ptr, stride_bk, stride_bn, BLOCK_SIZE_NK, NK
):
    """
    Compute a single matmul of the input block with the transposed weight block.

    Note that the linear layer is applying a transpose in pytorch.  We do the same here.
    """

    count = tl.cdiv(NK, BLOCK_SIZE_NK)

    # allocate an accumulator for the output
    sum = tl.zeros_like(input_block)

    # Iterate over the weights matrix and add to the output.
    # This goes row by row because it's transposed.
    for k in range(count):

        row_offsets = k * BLOCK_SIZE_NK + tl.arange(0, BLOCK_SIZE_NK)
        col_offsets = tl.arange(0, BLOCK_SIZE_NK)
        row_starts = row_offsets[None, :] * stride_bn
        col_starts = col_offsets[:, None] * stride_bk
        # For each row, access all cols (full NK width)
        row_ptrs = weight_ptr + (row_starts + col_starts)

        row_mask = row_offsets[None, :] < NK
        col_mask = col_offsets[:, None] < NK
        weight_mask = row_mask & col_mask

        tl.static_print("row_ptrs shape", row_ptrs.shape)
        tl.static_print("weight_mask shape", weight_mask.shape)

        weight_block = tl.load(row_ptrs, mask=weight_mask, other=0.0)

        # perform gemm + accumulate
        sum += tl.dot(input_block, tl.trans(weight_block))

    if bias_ptr is not None:
        bias_block = tl.load(bias_ptr, mask=weight_mask, other=0.0)
        sum += bias_block

    return sum
    # Perform ReLU:
    # return relu(sum)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    """
    Leaky ReLU activation function.

    Args:
        x: Input tensor

    Returns:
        Tensor with leaky ReLU applied (0.01 slope for negative values)
    """
    return tl.where(x >= 0, x, 0.01 * x)


@triton.jit
def relu(x):
    """
    ReLU activation function.

    Args:
        x: Input tensor

    Returns:
        Tensor with ReLU applied (0 for negative values)
    """
    return tl.where(x >= 0, x, 0.0)


class TritonFusedMLP(FusedMLP):
    """
    Triton implementation of a fused MLP.

    This class provides a CUDA-accelerated implementation of a multi-layer perceptron
    using Triton for GPU acceleration.
    """

    def __init__(self, traced_module: fx.GraphModule):
        """
        Initialize a TritonFusedMLP.

        Args:
            traced_module: An fx.GraphModule containing the traced MLP operations
        """
        super().__init__(traced_module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the fused MLP.

        Args:
            x: Input tensor of shape [batch_size, input_features]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_features]

        Raises:
            AssertionError: If inputs or weights are not CUDA tensors or have incompatible dimensions
        """
        # Constraints checks
        if not x.is_cuda:
            raise ValueError("Inputs must be CUDA tensors")

        # Check input dtypes and get output dtype
        output_dtype = str(x.dtype).split(".")[-1]  # extract 'float32', 'float16', etc.

        first_weight = self.weights[0]

        if x.shape[1] != first_weight.shape[0]:
            raise ValueError("Incompatible dimensions for matrix multiplication")

        for weight in self.weights:
            if weight.shape != first_weight.shape:
                raise ValueError("Only same shape weights are supported")
            if not weight.is_cuda:
                raise ValueError("Weights must be CUDA tensors")

        # How many are active this time?
        active_layers = len(self.weights)

        # Kinda hacky.  Forcing exactly 5 weights.
        if len(self.weights) != self.max_layers:
            self.weights = self.weights + [
                torch.empty((), device=x.device, dtype=x.dtype)
                for _ in range(self.max_layers - len(self.weights))
            ]

        # Force 5 biases too:
        if len(self.biases) != self.max_layers:
            self.biases = self.biases + [
                torch.empty((), device=x.device, dtype=x.dtype)
                for _ in range(self.max_layers - len(self.biases))
            ]

        wb = [val for pair in zip(self.weights, self.biases) for val in pair]

        print(f"Are weights[0] contiguous? {self.weights[0].is_contiguous()}")

        # Create output tensor
        M, K = x.shape
        _, N = first_weight.shape
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # Launch kernel with dtype parameter
        def grid(META):
            return (triton.cdiv(M, META["BLOCK_SIZE_M"]),)

        matmul_kernel[grid](
            x,
            output,
            *wb,
            active_layers,
            M,
            N,
            x.stride(0),
            x.stride(1),
            first_weight.stride(0),
            first_weight.stride(1),
            BLOCK_SIZE_NK=N,
            ACTIVATION="relu",
            OUTPUT_DTYPE=output_dtype,  # Pass the dtype to the kernel
        )

        print(self.weights)

        print(f"hand output: {torch.matmul(x, self.weights[0].T)}")

        return output

        # Each thread handles one row of the input/output

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if Triton is available for this implementation.

        Returns:
            bool: True if Triton is installed and available, False otherwise
        """
        return TRITON_AVAILABLE
