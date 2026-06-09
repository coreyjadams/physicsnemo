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

from enum import Enum

import torch
import triton
import triton.language as tl


class Activation(Enum):
    """Enumeration of supported activation functions."""

    NONE = -1
    RELU = 0
    LEAKY_RELU = 1
    SILU = 2


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
    return tl.where(x > 0, x, 0.01 * x)


@triton.jit
def relu(x):
    """
    ReLU activation function.

    Args:
        x: Input tensor

    Returns:
        Tensor with ReLU applied (0 for negative values)
    """
    return tl.where(x > 0, x, 0.0)


@triton.jit
def silu(x):
    """
    SiLU activation function.
    """
    return x * tl.sigmoid(x)


@triton.jit
def silu_bwd(x, grad_output):
    """
    SiLU backward pass.
    """
    sigmoid_x = tl.sigmoid(x)
    grad = sigmoid_x + x * sigmoid_x * (1 - sigmoid_x)
    return grad_output * grad


@triton.jit
def relu_bwd(x, grad_output):
    """
    ReLU backward pass.
    """
    return tl.where(x > 0, grad_output, 0.0)


@triton.jit
def leaky_relu_bwd(x, grad_output):
    """
    Leaky ReLU backward pass.
    """
    return grad_output * tl.where(x > 0, 1.0, 0.01)


@triton.jit
def activation_dispatch(x, ACTIVATION: tl.constexpr):
    """
    Dispatch to the appropriate activation function.
    No memory management here - this is meant to embed
    in a larger kernel.
    """

    # Call the kernel:
    if ACTIVATION == 0:
        output = relu(x)
    elif ACTIVATION == 1:
        output = leaky_relu(x)
    elif ACTIVATION == 2:
        output = silu(x)
    else:
        output = x

    return output


@triton.jit
def activation_dispatch_bwd(x, grad_output, ACTIVATION: tl.constexpr):
    """
    Dispatch to the appropriate activation grad function.
    No memory management here - this is meant to embed
    in a larger kernel.
    """

    # Call the kernel:
    if ACTIVATION == 0:
        grad_input = relu_bwd(x, grad_output)
    elif ACTIVATION == 1:
        grad_input = leaky_relu_bwd(x, grad_output)
    elif ACTIVATION == 2:
        grad_input = silu_bwd(x, grad_output)
    else:
        grad_input = grad_output

    return grad_input


@triton.jit
def activation_kernel(
    x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, ACTIVATION: tl.constexpr
):
    """
    Activation full kernel, with memory load and store.

    Assumes flattened, 1D input and output.
    """

    pid = tl.program_id(axis=0)

    # Load the block:
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < n_elements
    x = tl.load(x_ptr + block_start, mask=mask, other=0.0)

    # Call the kernel:
    output = activation_dispatch(x, ACTIVATION)

    tl.store(output_ptr + block_start, output, mask=mask)


@triton.jit
def activation_kernel_bwd(
    x_ptr,
    grad_output_ptr,
    grad_input_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """
    Activation backward full kernel, with memory load and store.

    Assumes flattened, 1D input and output.
    """

    pid = tl.program_id(axis=0)

    # Load the block:
    block_start = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block_start < n_elements
    x = tl.load(x_ptr + block_start, mask=mask, other=0.0)
    grad_output = tl.load(grad_output_ptr + block_start, mask=mask, other=0.0)

    grad_input = activation_dispatch_bwd(x, grad_output, ACTIVATION)

    tl.store(grad_input_ptr + block_start, grad_input, mask=mask)


def activation_triton_forward(x: torch.Tensor, activation: Activation) -> torch.Tensor:
    """
    Apply an activation function forward using Triton kernels.
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be a CUDA tensor")

    flat_x = x.view(-1).contiguous()
    flat_output = torch.empty_like(flat_x)

    n_elements = flat_x.shape[0]

    def grid(META):
        return (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)

    activation_kernel[grid](
        flat_x, flat_output, n_elements, BLOCK_SIZE=1024, ACTIVATION=activation.value
    )

    return flat_output.view(x.shape)


def activation_triton_backward(
    x: torch.Tensor, grad_output: torch.Tensor, activation: Activation
) -> torch.Tensor:
    """
    Apply an activation function backward using Triton kernels.
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be a CUDA tensor")

    flat_x = x.view(-1).contiguous()
    flat_grad_output = grad_output.view(-1).contiguous()
    flat_grad_input = torch.empty_like(flat_x)

    n_elements = flat_x.shape[0]

    def grid(META):
        return (triton.cdiv(n_elements, META["BLOCK_SIZE"]),)

    activation_kernel_bwd[grid](
        flat_x,
        flat_grad_output,
        flat_grad_input,
        n_elements,
        BLOCK_SIZE=1024,
        ACTIVATION=activation.value,
    )

    return flat_grad_input.view(x.shape)


class TritonActivationFunction(torch.autograd.Function):
    """
    Custom autograd function for Triton-accelerated activation functions.

    Like other components in this file, it's designed for unit testing of
    components of the fused MLP.

    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, activation: Activation) -> torch.Tensor:
        """Forward pass of the activation function.

        Args:
            ctx: Autograd context
            x: Input tensor
            activation: Enum specifying which activation function to use

        Returns:
            torch.Tensor: Result of applying the activation function
        """
        # Save for backward pass
        ctx.save_for_backward(x)
        ctx.activation = activation

        # Call Triton kernel for forward activation
        result = activation_triton_forward(x, activation)
        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Backward pass computing gradients.

        Args:
            ctx: Autograd context
            grad_output: Gradient from subsequent layer

        Returns:
            tuple[torch.Tensor, None]: Gradient for input tensor and None for activation enum
        """
        (x,) = ctx.saved_tensors
        activation = ctx.activation

        # Call Triton kernel for backward pass
        grad_input = activation_triton_backward(x, grad_output, activation)
        return grad_input, None


def apply_triton_activation(x: torch.Tensor, activation: Activation) -> torch.Tensor:
    """
    Apply an activation function using Triton kernels.

    This function wraps the Triton kernel implementation of various activation
    functions in a differentiable interface using PyTorch's autograd.

    It is not really meant to be used directly - it's for testing the triton
    implementation of the activation functions, which in turn are embedded in the
    mlp fusion tool.

    Args:
        x: Input tensor to apply activation to
        activation: Which activation function to apply, from the Activation enum

    Returns:
        torch.Tensor: Result of applying the specified activation function

    Raises:
        ValueError: If input tensor is not a CUDA tensor
        ValueError: If activation type is not supported
    """
    if not x.is_cuda:
        raise ValueError("Input tensor must be a CUDA tensor")

    if activation not in Activation:
        raise ValueError(f"Unsupported activation type: {activation}")

    return TritonActivationFunction.apply(x, activation)