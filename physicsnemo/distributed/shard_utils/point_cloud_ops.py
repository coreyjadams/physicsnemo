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

from typing import Any, Tuple, Union

import torch
import wrapt

from physicsnemo.models.layers.ball_query import BallQueryLayer
from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")

from physicsnemo.distributed import ShardTensor  # noqa: E402
from physicsnemo.distributed.shard_utils.patch_core import (  # noqa: E402
    UndeterminedShardingError,
)

__all__ = ["ball_query_wrapper"]


@wrapt.patch_function_wrapper(
    "physicsnemo.models.layers.ball_query",
    "BallQueryLayer.forward",
    enabled=ShardTensor.patches_enabled,
)
def ball_query_wrapper(
    wrapped: Any, instance: BallQueryLayer, args: tuple, kwargs: dict
) -> Union[
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    Tuple[ShardTensor, ShardTensor, ShardTensor],
]:
    """Wrapper for BallQueryLayer.forward to support sharded tensors.

    This initial prototype handles parallel processing of ball query operations
    when sharded tensors are provided as input.

    Args:
        wrapped: Original forward method
        instance: BallQueryLayer instance
        args: Positional arguments (points1, points2, lengths1, lengths2)
        kwargs: Keyword arguments

    Returns:
        Tuple of (mapping, num_neighbors, outputs) as torch.Tensor or ShardTensor
    """

    # Extract the arguments
    if len(args) >= 4:
        points1, points2, lengths1, lengths2 = args[:4]
    else:
        points1 = kwargs.get("points1", args[0] if len(args) > 0 else None)
        points2 = kwargs.get("points2", args[1] if len(args) > 1 else None)
        lengths1 = kwargs.get("lengths1", args[2] if len(args) > 2 else None)
        lengths2 = kwargs.get("lengths2", args[3] if len(args) > 3 else None)

    # If inputs are ShardTensors, handle them appropriately
    if all(isinstance(t, ShardTensor) for t in (points1, points2)):

        # Convert ShardTensors to local tensors
        local_points1 = points1.to_local()
        local_points2 = points2.to_local()

        # Call the original function with local tensors
        mapping, num_neighbors, outputs = wrapped(
            local_points1, local_points2, lengths1, lengths2, *args[4:], **kwargs
        )

        # The output values should be sharded according to the input sharding.
        # Convert the results back to ShardTensors with the same sharding spec
        mapping_shard = ShardTensor.from_local(
            mapping,
            points1._spec.mesh,
            points1._spec.placements,
            points1._spec.sharding_sizes(),
        )

        num_neighbors_shard = ShardTensor.from_local(
            num_neighbors,
            points1._spec.mesh,
            points1._spec.placements,
            points1._spec.sharding_sizes(),
        )

        outputs_shard = ShardTensor.from_local(
            outputs,
            points1._spec.mesh,
            points1._spec.placements,
            points1._spec.sharding_sizes(),
        )

        return mapping_shard, num_neighbors_shard, outputs_shard

    # If inputs are regular torch tensors, just call the original function
    elif all(isinstance(t, torch.Tensor) for t in (points1, points2)):
        return wrapped(*args, **kwargs)

    # If inputs are mixed types, raise an error
    else:
        raise UndeterminedShardingError(
            "points1 and points2 must be the same types (torch.Tensor or ShardTensor)"
        )
