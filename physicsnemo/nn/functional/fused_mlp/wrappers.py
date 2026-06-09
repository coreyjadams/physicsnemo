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

from physicsnemo.utils.profiling import profile

from .activations import Activation
from .matmul1 import (
    matmul1_launcher,
    matmul1_launcher_backward,
)
from .matmul2 import (
    matmul2_launcher,
    matmul2_launcher_backward,
)
from .matmul3 import (
    matmul3_launcher,
)


@profile
def fused_mlp(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    activation: Activation,
    last_activation: bool,
):
    """ """

    # Basic checks:
    if not x.is_cuda:
        raise ValueError("Input tensor must be a CUDA tensor")
    for weight in weights:
        if not weight.is_cuda:
            raise ValueError("Weight tensor must be a CUDA tensor")

    if len(weights) != len(biases):
        raise ValueError("Number of weights and biases must match")

    if len(weights) > 3:
        raise ValueError("Only up to 2 hidden layers are supported")

    if len(weights) == 0:
        raise ValueError("At least one weight matrix is required")

    # rank-2 weight matrix consistently:
    for weight in weights:
        if len(weight.shape) != 2:
            raise ValueError(f"Weight matrix must be rank-2, got shape {weight.shape}")

    for weight, bias in zip(weights, biases):
        if bias is not None and weight.shape[0] != bias.shape[0]:
            raise ValueError(
                f"Weight and bias shapes must match, got {weight.shape[0]} and {bias.shape[0]}"
            )

    if len(weights) == 1:
        return TritonSingleLayerFusedMLP.apply(
            x, weights[0], biases[0], activation, last_activation
        )
    elif len(weights) == 2:
        return TritonTwoLayerFusedMLP.apply(
            x, weights[0], biases[0], weights[1], biases[1], activation, last_activation
        )
    elif len(weights) == 3:
        return TritonThreeLayerFusedMLP.apply(
            x,
            weights[0],
            biases[0],
            weights[1],
            biases[1],
            weights[2],
            biases[2],
            activation,
            last_activation,
        )
    else:
        raise ValueError(f"Only up to 3 layers are supported, got {len(weights)}")


class TritonSingleLayerFusedMLP(torch.autograd.Function):
    """
    Differentiable class for fused MLPs via triton.

    Enables selecting the right forward pass, caching only the necessary variables for backwards.

    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight0: torch.Tensor,
        bias0: torch.Tensor,
        activation: Activation,
        last_activation: bool,
    ) -> torch.Tensor:
        """
        Apply the forward pass of the fused matmuls.
        """

        # Apply the fusion:
        outputs = matmul1_launcher(x, weight0, bias0, activation, last_activation)
        ctx.save_for_backward(x, weight0, bias0)

        ctx.n_layers = 1
        ctx.activation = activation
        ctx.last_activation = last_activation
        ctx.input_grads = x.requires_grad
        ctx.weight_grads = weight0.requires_grad
        ctx.bias_grads = bias0.requires_grad if bias0 is not None else False

        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass, to be implemented
        """

        x, weights, biases = ctx.saved_tensors
        grad_input, grad_weight, grad_bias = matmul1_launcher_backward(
            input_tensor=x,
            grad_output=grad_output,
            weight=weights,
            bias=biases,
            grad_input=ctx.input_grads,
            grad_weight=ctx.weight_grads,
            grad_bias=ctx.bias_grads,
            activation=ctx.activation,
            last_activation=ctx.last_activation,
        )
        grad_weight = grad_weight
        grad_bias = grad_bias

        return grad_input, grad_weight, grad_bias, None, None


class TritonTwoLayerFusedMLP(torch.autograd.Function):
    """
    Differentiable class for fused MLPs via triton.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight0: torch.Tensor,
        bias0: torch.Tensor,
        weight1: torch.Tensor,
        bias1: torch.Tensor,
        activation: Activation,
        last_activation: bool,
    ) -> torch.Tensor:
        """
        Apply the forward pass of the fused matmuls.
        """

        # Apply the fusion:
        outputs = matmul2_launcher(
            x, weight0, weight1, bias0, bias1, activation, last_activation
        )
        ctx.save_for_backward(x, weight0, weight1, bias0, bias1)

        ctx.activation = activation
        ctx.last_activation = last_activation
        ctx.input_grads = x.requires_grad
        ctx.weight0_grads = weight0.requires_grad
        ctx.bias0_grads = bias0.requires_grad if bias0 is not None else False
        ctx.weight1_grads = weight1.requires_grad
        ctx.bias1_grads = bias1.requires_grad if bias1 is not None else False
        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the two-layer fused MLP.
        """

        x, weight0, weight1, bias0, bias1 = ctx.saved_tensors
        (
            grad_input,
            grad_weight0,
            grad_weight1,
            grad_bias0,
            grad_bias1,
        ) = matmul2_launcher_backward(
            input_tensor=x,
            grad_output=grad_output,
            weight0=weight0,
            weight1=weight1,
            bias0=bias0,
            bias1=bias1,
            grad_input=ctx.input_grads,
            grad_weight0=ctx.weight0_grads,
            grad_bias0=ctx.bias0_grads if ctx.bias0_grads else False,
            grad_weight1=ctx.weight1_grads,
            grad_bias1=ctx.bias1_grads if ctx.bias1_grads else False,
            activation=ctx.activation,
            last_activation=ctx.last_activation,
        )

        return (
            grad_input,
            grad_weight0,
            grad_bias0,
            grad_weight1,
            grad_bias1,
            None,
            None,
        )


class TritonThreeLayerFusedMLP(torch.autograd.Function):
    """
    Differentiable class for fused MLPs via triton.
    """

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight0: torch.Tensor,
        bias0: torch.Tensor,
        weight1: torch.Tensor,
        bias1: torch.Tensor,
        weight2: torch.Tensor,
        bias2: torch.Tensor,
        activation: Activation,
        last_activation: bool,
    ) -> torch.Tensor:
        """
        Apply the forward pass of the fused matmuls.
        """

        # Apply the fusion:
        outputs = matmul3_launcher(
            x,
            weight0,
            weight1,
            weight2,
            bias0,
            bias1,
            bias2,
            activation,
            last_activation,
        )
        ctx.save_for_backward(x, weight0, weight1, weight2, bias0, bias1, bias2)

        ctx.activation = activation
        ctx.last_activation = last_activation
        ctx.input_grads = x.requires_grad
        ctx.weight0_grads = weight0.requires_grad
        ctx.bias0_grads = bias0.requires_grad if bias0 is not None else False
        ctx.weight1_grads = weight1.requires_grad
        ctx.bias1_grads = bias1.requires_grad if bias1 is not None else False
        ctx.weight2_grads = weight2.requires_grad
        ctx.bias2_grads = bias2.requires_grad if bias2 is not None else False

        return outputs

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the three-layer fused MLP.

        The fused three-layer backward kernel is not yet implemented. Use a one-
        or two-layer fused MLP for training, or restrict the three-layer path to
        inference (forward-only) workloads.
        """

        raise NotImplementedError(
            "Backward pass for the fused three-layer MLP is not yet implemented. "
            "Use a one- or two-layer fused MLP if gradients are required, or run "
            "the three-layer fused MLP forward-only (e.g. under torch.no_grad())."
        )