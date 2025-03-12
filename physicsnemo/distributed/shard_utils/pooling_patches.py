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

from typing import Any, Dict, Optional, Tuple, Union

import torch
import wrapt
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.distributed import ShardTensor
from physicsnemo.distributed.shard_utils.patch_core import (
    MissingShardPatch,
    UndeterminedShardingError,
)

aten = torch.ops.aten

__all__ = [
    "avg_pool3d_wrapper",
]


def compute_output_shape(input_shape, pool_kwargs):
    """Compute the output shape of a pooling operation.

    Args:
        input_shape: Shape of the input tensor
        pool_kwargs: Keyword arguments for the pooling operation

    Returns:
        tuple: Output shape after pooling operation
    """
    # Extract pooling parameters
    kernel_size = pool_kwargs.get("kernel_size")
    stride = pool_kwargs.get("stride", kernel_size)
    padding = pool_kwargs.get("padding", 0)

    # Handle scalar parameters
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * (len(input_shape) - 2)
    if isinstance(stride, int):
        stride = (stride,) * (len(input_shape) - 2)
    if isinstance(padding, int):
        padding = (padding,) * (len(input_shape) - 2)

    # Batch and channel dimensions remain unchanged
    output_shape = list(input_shape[:2])

    # Compute spatial dimensions
    for i, (size, k, s, p) in enumerate(
        zip(input_shape[2:], kernel_size, stride, padding)
    ):
        output_size = ((size + 2 * p - k) // s) + 1
        output_shape.append(output_size)

    return tuple(output_shape)


@wrapt.patch_function_wrapper(
    "torch.nn.functional", "avg_pool3d", enabled=ShardTensor.patches_enabled
)
def avg_pool3d_wrapper(wrapped, instance, args, kwargs):
    return generic_avg_pool_nd_wrapper(wrapped, instance, args, kwargs)


@wrapt.patch_function_wrapper(
    "torch.nn.functional", "avg_pool2d", enabled=ShardTensor.patches_enabled
)
def avg_pool2d_wrapper(wrapped, instance, args, kwargs):
    return generic_avg_pool_nd_wrapper(wrapped, instance, args, kwargs)


@wrapt.patch_function_wrapper(
    "torch.nn.functional", "avg_pool1d", enabled=ShardTensor.patches_enabled
)
def avg_pool1d_wrapper(wrapped, instance, args, kwargs):
    return generic_avg_pool_nd_wrapper(wrapped, instance, args, kwargs)


def repackage_pool_args(
    input: Union[torch.Tensor, ShardTensor],
    kernel_size: Union[int, Tuple[int, ...]],
    stride: Union[int, Tuple[int, ...]] = None,
    padding: Union[int, Tuple[int, ...]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
    *args,
    **kwargs,
) -> Tuple[Union[torch.Tensor, ShardTensor], Dict[str, Any]]:
    """Repackages pooling arguments into standard format.

    Takes the full set of arguments that could be passed to an avg_pool operation
    and separates them into the input tensor and configuration parameters
    packaged as a kwargs dict.

    Args:
        input: Input tensor to pool
        kernel_size: Size of the pooling window
        stride: Stride of the pooling window, defaults to kernel_size
        padding: Padding added to both sides of the input
        ceil_mode: When True, will use ceil instead of floor to compute the output shape
        count_include_pad: When True, will include the zero-padding in the averaging calculation
        divisor_override: If specified, will be used as divisor, otherwise kernel_size is used
        *args: Additional positional args (unused)
        **kwargs: Additional keyword args (unused)

    Returns:
        Tuple containing:
        - Input tensor
        - Dict of pooling configuration parameters
    """
    # Handle stride=None case (defaults to kernel_size)
    if stride is None:
        stride = kernel_size

    # Package all non-tensor parameters into a kwargs dictionary
    return_kwargs = {
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "ceil_mode": ceil_mode,
        "count_include_pad": count_include_pad,
    }

    # Only add divisor_override if it's not None
    if divisor_override is not None:
        return_kwargs["divisor_override"] = divisor_override

    return input, return_kwargs


def generic_avg_pool_nd_wrapper(wrapped, instance, args, kwargs):
    """Generic wrapper for torch N-dimensional pooling operations.

    Handles both regular torch.Tensor inputs and distributed ShardTensor inputs.
    For regular tensors, passes through to the wrapped pooling function.
    For ShardTensor inputs, handles applying distributed pooling.

    Args:
        wrapped: Original pooling function being wrapped
        instance: Instance the wrapped function is bound to
        args: Positional arguments for pooling
        kwargs: Keyword arguments for pooling

    Returns:
        Pooling result as either torch.Tensor or ShardTensor

    Raises:
        UndeterminedShardingError: If input tensor types are invalid
    """

    # Extract the input tensor and package the remaining arguments
    input, pool_kwargs = repackage_pool_args(*args, **kwargs)

    # Handle regular torch tensor inputs
    if type(input) == torch.Tensor:
        return wrapped(*args, **kwargs)

    # Handle distributed ShardTensor inputs
    elif type(input) == ShardTensor:

        # For pooling, the main challenge is to predict the output shape

        # Get the local tensor:
        local_input = input.to_local()

        local_pooled_output = wrapped(local_input, **pool_kwargs)

        # Reject cases where stride != kernel_size
        if pool_kwargs.get("stride") != pool_kwargs.get("kernel_size"):
            raise MissingShardPatch(
                "Stride must equal kernel_size for pooling operations"
            )

        # Check divisibility by stride only for sharded dimensions
        stride = pool_kwargs.get("stride")
        if isinstance(stride, int):
            # Assuming channels first ...
            stride = (stride,) * (len(local_input.shape) - 2)

        for mesh_dim, placement in enumerate(input._spec.placements):
            if isinstance(placement, Shard):
                # This dimension is sharded on this mesh dimension
                shard_dim = placement.dim
                # Skip batch and channel dimensions (first two dims)
                if shard_dim >= 2:
                    spatial_dim = shard_dim - 2  # Convert to spatial dimension index
                    # Get the sizes for this mesh dimension
                    shard_shapes = input._spec.sharding_sizes()[mesh_dim]
                    for shard_shape in shard_shapes:
                        if (
                            spatial_dim < len(shard_shape) - 2
                        ):  # Check if dimension is valid
                            spatial_size = shard_shape[shard_dim]
                            stride_for_dim = stride[spatial_dim]
                            if spatial_size % stride_for_dim != 0:
                                raise UndeterminedShardingError(
                                    f"Sharded dimension {shard_dim} with local size {spatial_size} "
                                    f"must be divisible by stride {stride_for_dim}"
                                )

        # Compute the sharding shapes:
        updated_placements = {}
        for mesh_dim, shard_shapes in input._spec.sharding_sizes().items():
            updated_shard_shapes = [
                compute_output_shape(shard_shape, pool_kwargs)
                for shard_shape in shard_shapes
            ]
            print(f"Updated shard shapes: {updated_shard_shapes}")
            updated_placements[mesh_dim] = updated_shard_shapes

        print(f"local pooled output shape: {local_pooled_output.shape}")

        output = ShardTensor.from_local(
            local_pooled_output,
            input._spec.mesh,
            input._spec.placements,
            sharding_shapes=updated_placements,
        )
        return output
        # Use the convolution args to compute the sharded halo

    else:
        msg = (
            "input must be a valid type "
            "(torch.Tensor or ShardTensor), but got "
            f"{type(input)}"
        )
        raise UndeterminedShardingError(msg)


# Write a function to extract the default args for avg_pool_nd


# This will become the future implementation, or similar.
# Why not today?  Because the backwards pass in DTensor has an explicit (and insufficient)
# hard coded implementation for the backwards pass.
# When that switch happens, the order in the arg repackaging will need to be updated.
# ShardTensor.register_function_handler(aten.convolution.default, generic_conv_nd_wrapper)
