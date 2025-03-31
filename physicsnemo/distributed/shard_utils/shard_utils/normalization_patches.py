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

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import wrapt
from modulus.distributed import ShardTensor, ShardTensorSpec
from torch.distributed.tensor import DTensor

__all__ = [
    "layer_norm_wrapper",
]


from modulus.distributed.shard_utils.patch_core import (
    UndeterminedShardingError,
)

aten = torch.ops.aten


class PartialLayerNorm(torch.autograd.Function):
    """Sharded convolution operation that uses halo message passing for distributed computation.

    This class implements a distributed convolution primitive that operates on sharded tensors.
    It handles both forward and backward passes while managing communication between shards.

    Leverages torch.ops.aten.convolution.default for generic convolutions.
    """

    @staticmethod
    def forward(
        ctx,
        inputs: torch.Tensor,
        normalized_shape,
        weights: torch.nn.Parameter,
        bias: Optional[torch.nn.Parameter],
        eps: float,
        spec: "ShardTensorSpec",
    ) -> "ShardTensor":
        """Forward pass of the distributed convolution.

        Args:
            ctx: Context object for saving tensors needed in backward pass
            inputs: Input tensor to convolve
            weights: Convolution filter weights
            bias: Optional bias tensor
            output_spec: Specification for output ShardTensor
            conv_kwargs: Dictionary of convolution parameters (stride, padding, etc.)

        Returns:
            ShardTensor containing the convolution result
        """
        # Save spec for backward pass
        ctx.spec = spec
        ctx.norm_shape = normalized_shape
        ctx.eps = eps

        # Perform layer normalization
        local_chunk, mean, rstd = aten.native_layer_norm(
            inputs, normalized_shape, weights, bias, eps
        )

        # Save tensors needed for backward pass
        ctx.save_for_backward(inputs, weights, bias)
        ctx.mean = mean
        ctx.rstd = rstd

        ctx.grad_mask = (
            inputs.requires_grad,
            weights is not None and weights.requires_grad,
            bias is not None and bias.requires_grad,
        )

        # Wrap result in ShardTensor
        output = ShardTensor.from_local(local_chunk, spec.mesh, spec.placements)

        return output

    @staticmethod
    def backward(
        ctx, grad_output: "ShardTensor"
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], None, None, None]:
        """Backward pass for distributed convolution.

        Args:
            ctx: Context object containing saved tensors
            grad_output: Gradient of the loss with respect to convolution output

        Returns:
            Tuple containing gradients for inputs, weights, and bias (plus None values for other args)
        """
        spec = ctx.spec
        normalized_shape = ctx.norm_shape
        local_chunk, weight, bias = ctx.saved_tensors

        output_mask = ctx.grad_mask

        # Compute local gradients
        local_grad_output = grad_output._local_tensor
        grad_input, grad_weight, grad_bias = aten.native_layer_norm_backward(
            local_grad_output,
            local_chunk,
            normalized_shape,
            ctx.mean,
            ctx.rstd,
            weight,
            bias,
            output_mask=output_mask,
        )

        # Synchronize weight and bias gradients across all ranks
        group = spec.mesh.get_group()
        dist.all_reduce(grad_weight, group=group)
        if grad_bias is not None:
            dist.all_reduce(grad_bias, group=group)

        return grad_input, None, grad_weight, grad_bias, None, None


@wrapt.patch_function_wrapper("torch.nn.functional", "layer_norm")
def layer_norm_wrapper(wrapped, instance, args, kwargs):

    input, normalized_shape, weight, bias, eps = repackage_layer_norm_args(
        *args, **kwargs
    )

    # Handle regular torch tensor inputs
    if (
        type(input) == torch.Tensor
        and (
            type(weight) == torch.nn.parameter.Parameter or type(weight) == torch.Tensor
        )
        and (
            bias is None
            or (
                type(bias) == torch.nn.parameter.Parameter or type(bias) == torch.Tensor
            )
        )
    ):
        return wrapped(*args, **kwargs)

    # Handle distributed ShardTensor inputs
    elif type(input) == ShardTensor:

        # Gather any distributed weights/bias
        if isinstance(weight, (ShardTensor, DTensor)):
            weight = weight.full_tensor()
        if isinstance(bias, (ShardTensor, DTensor)):
            bias = bias.full_tensor()

        output_spec = input._spec
        x = PartialLayerNorm.apply(
            input.to_local(), normalized_shape, weight, bias, eps, output_spec
        )

        return x

    else:
        msg = (
            "input, weight, bias (if not None) must all be the valid types "
            "(torch.Tensor or ShardTensor), but got "
            f"{type(input)}, "
            f"{type(weight)}, "
            f"{type(bias)}, "
        )
        raise UndeterminedShardingError(msg)


def repackage_layer_norm_args(
    input: torch.Tensor,
    normalized_shape: List[int],
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-05,
    *args,
    **kwargs,
):

    return input, normalized_shape, weight, bias, eps
