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
import torch.distributed as dist
import warp as wp
import wrapt

from physicsnemo.models.layers.ball_query import BallQueryLayer
from physicsnemo.utils.version_check import check_module_requirements

check_module_requirements("physicsnemo.distributed.shard_tensor")

from torch.distributed.tensor.placement_types import (  # noqa: E402
    Shard,
)

from physicsnemo.distributed import ShardTensor  # noqa: E402
from physicsnemo.distributed.shard_utils.patch_core import (  # noqa: E402
    MissingShardPatch,
    UndeterminedShardingError,
)
from physicsnemo.distributed.shard_utils.ring import (  # noqa: E402
    RingPassingConfig,
    perform_ring_iteration,
)

wp.config.quiet = True

__all__ = ["ball_query_wrapper"]


def ring_ball_query(
    points1: ShardTensor,
    points2: ShardTensor,
    lengths1: Union[torch.Tensor, ShardTensor],
    lengths2: Union[torch.Tensor, ShardTensor],
    wrapped: Any,
    *args: Any,
    **kwargs: Any,
) -> Tuple[ShardTensor, ShardTensor, ShardTensor]:
    """
    Performs ball query operation on points distributed across ranks in a ring configuration.

    Args:
        points1: First set of points as a ShardTensor
        points2: Second set of points as a ShardTensor
        lengths1: Lengths of each batch in points1
        lengths2: Lengths of each batch in points2
        wrapped: The original ball query function to call on each rank
        *args: Additional positional arguments to pass to the wrapped function
        **kwargs: Additional keyword arguments to pass to the wrapped function

    Returns:
        Tuple of (mapping, num_neighbors, outputs) as ShardTensors
    """
    mesh = points1._spec.mesh
    # We can be confident of this because 1D meshes are enforced
    mesh_dim = 0

    local_group = mesh.get_group(mesh_dim)
    local_size = dist.get_world_size(group=local_group)

    # Create a config object to simplify function args for message passing:
    ring_config = RingPassingConfig(
        mesh_dim=mesh_dim,
        mesh_size=local_size,
        communication_method="p2p",
        ring_direction="forward",
    )

    # Now, get the inputs locally:
    local_points1 = points1.to_local()
    local_points2 = points2.to_local()
    # local_lengths1 = lengths1.to_local()
    # local_lengths2 = lengths2.to_local()

    # Get the shard sizes for the point cloud going around the ring.
    # We've already checked that the mesh is 1D so call the '0' index.
    shard_sizes = points2._spec.sharding_sizes()[0]
    print(f"shard_sizes: {shard_sizes}")

    # Call the differentiable version of the ring-ball-query:
    mapping_shard, num_neighbors_shard, outputs_shard = RingBallQuery.apply(
        local_points1,
        local_points2,
        lengths1,
        lengths2,
        mesh,
        ring_config,
        shard_sizes,
        wrapped,
        *args,
        **kwargs,
    )

    # TODO
    # the output shapes can be computed directly from the input sharding of points1
    # Requires a little work to fish out parameters but that's it.
    # For now, using blocking inference to get the output shapes.

    # Convert back to ShardTensor
    mapping_shard = ShardTensor.from_local(
        mapping_shard, points1._spec.mesh, points1._spec.placements, "infer"
    )
    num_neighbors_shard = ShardTensor.from_local(
        num_neighbors_shard, points1._spec.mesh, points1._spec.placements, "infer"
    )
    outputs_shard = ShardTensor.from_local(
        outputs_shard, points1._spec.mesh, points1._spec.placements, "infer"
    )
    return mapping_shard, num_neighbors_shard, outputs_shard


def merge_outputs(
    current_mapping: Union[torch.Tensor, None],
    current_num_neighbors: Union[torch.Tensor, None],
    current_outputs: Union[torch.Tensor, None],
    incoming_mapping: torch.Tensor,
    incoming_num_neighbors: torch.Tensor,
    incoming_outputs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform a gather/scatter operation on the mapping and outputs tensors.
    This is an _inplace_ operation on the current tensors, assuming they are not None

    Args:
        current_mapping: Current mapping tensor or None
        current_num_neighbors: Current number of neighbors tensor or None
        current_outputs: Current outputs tensor or None
        incoming_mapping: Incoming mapping tensor to merge
        incoming_num_neighbors: Incoming number of neighbors tensor to merge
        incoming_outputs: Incoming outputs tensor to merge

    Returns:
        Tuple of merged (mapping, num_neighbors, outputs) tensors
    """

    @wp.kernel
    def merge_mapping_and_outputs(
        current_m: wp.array3d(dtype=wp.int32),
        current_nn: wp.array2d(dtype=wp.int32),
        current_o: wp.array4d(dtype=wp.float32),
        incoming_m: wp.array3d(dtype=wp.int32),
        incoming_nn: wp.array2d(dtype=wp.int32),
        incoming_o: wp.array4d(dtype=wp.float32),
        max_neighbors: int,
    ):
        # This is a kernel that is essentially doing a gather/scatter operation.

        # Which points are we looking at?
        tid = wp.tid()

        # How many neighbors do we have?
        num_neighbors = current_nn[0, tid]
        available_space = max_neighbors - num_neighbors

        # How many neighbors do we have in the incoming tensor?
        incoming_num_neighbors = incoming_nn[0, tid]

        # Can't add more neighbors than we have space for:
        neighbors_to_add = min(incoming_num_neighbors, available_space)

        # Now, copy the incoming neighbors to offset locations in the current tensor:
        for i in range(neighbors_to_add):

            # incoming has no offset
            # current has offset of num_neighbors
            current_m[0, tid, num_neighbors + i] = incoming_m[0, tid, i]
            current_o[0, tid, num_neighbors + i, 0] = incoming_o[0, tid, i, 0]
            current_o[0, tid, num_neighbors + i, 1] = incoming_o[0, tid, i, 1]
            current_o[0, tid, num_neighbors + i, 2] = incoming_o[0, tid, i, 2]

        # Finally, update the number of neighbors:
        current_nn[0, tid] = num_neighbors + incoming_num_neighbors
        return

    if (
        current_mapping is None
        and current_num_neighbors is None
        and current_outputs is None
    ):
        return incoming_mapping, incoming_num_neighbors, incoming_outputs

    _, n_points, max_neighbors = current_mapping.shape

    # This is a gather/scatter operation:
    # We need to merge the incoming values into the current arrays.  The arrays
    # are essentially a ragged tensor that has been padded to a consistent shape.
    # What happens here is:
    # - Compare the available space in current tensors to the number of incoming values.
    #   - If there are more values coming in than there is space, they are truncated.
    # - Using the available space, determine the section in the incoming tensor to gather.
    # - Using the (trucated) size of incoming values, determine the region of the current tensor for scatter.
    # - gather / scatter from incoming to current.
    # - Update the current num neighbors correctly

    wp.launch(
        merge_mapping_and_outputs,
        dim=n_points,
        inputs=[
            wp.from_torch(current_mapping, return_ctype=True),
            wp.from_torch(current_num_neighbors, return_ctype=True),
            wp.from_torch(current_outputs, return_ctype=True),
            wp.from_torch(incoming_mapping, return_ctype=True),
            wp.from_torch(incoming_num_neighbors, return_ctype=True),
            wp.from_torch(incoming_outputs, return_ctype=True),
            max_neighbors,
        ],
    )

    return current_mapping, current_num_neighbors, current_outputs


class RingBallQuery(torch.autograd.Function):
    """
    Custom autograd function for performing ball query operations in a distributed ring configuration.

    Handles the forward pass of ball queries across multiple ranks, enabling distributed computation
    of nearest neighbors for point clouds.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        points1: torch.Tensor,
        points2: torch.Tensor,
        lengths1: torch.Tensor,
        lengths2: torch.Tensor,
        mesh: Any,
        ring_config: RingPassingConfig,
        shard_sizes: list,
        wrapped: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for distributed ball query computation.

        Args:
            ctx: Context for saving variables for backward pass
            points1: First set of points
            points2: Second set of points
            lengths1: Lengths of each batch in points1
            lengths2: Lengths of each batch in points2
            mesh: Distribution mesh specification
            ring_config: Configuration for ring passing
            shard_sizes: Sizes of each shard across ranks
            wrapped: The original ball query function
            *args: Additional positional arguments for the wrapped function
            **kwargs: Additional keyword arguments for the wrapped function

        Returns:
            Tuple of (mapping, num_neighbors, outputs) tensors
        """
        ctx.mesh = mesh
        ctx.ring_config = ring_config

        # Create buffers to store outputs
        current_mapping = None
        current_num_neighbors = None
        current_outputs = None

        # For the first iteration, use local tensors
        current_p1, current_p2, current_l1, current_l2 = (
            points1,
            points2,
            lengths1,
            lengths2,
        )

        rank = dist.get_rank(group=ctx.mesh.get_group(0))
        world_size = ring_config.mesh_size

        # Store results from each rank to merge in the correct order
        rank_results = [None] * world_size
        # For uneven point clouds, the global stide is important:
        strides = [s[1] for s in shard_sizes]

        for i in range(world_size):

            # Calculate which source rank this data is from
            source_rank = (rank - i) % world_size

            local_mapping, local_num_neighbors, local_outputs = wrapped(
                current_p1, current_p2, current_l1, current_l2, *args, **kwargs
            )

            # Store the result with its source rank
            rank_results[source_rank] = (
                local_mapping,
                local_num_neighbors,
                local_outputs,
            )
            # strides.append(current_p2.shape[1])

            # For point clouds, we need to pass the size of the incoming shard.
            next_source_rank = (source_rank - 1) % world_size

            # TODO - this operation should be done async and checked for completion at the start of the next loop.
            if i != world_size - 1:
                # Don't do a ring on the last iteration.
                current_p2 = perform_ring_iteration(
                    current_p2,
                    ctx.mesh,
                    ctx.ring_config,
                    recv_shape=shard_sizes[next_source_rank],
                )

        # Now merge the results in rank order (0, 1, 2, ...)
        stride = 0
        for r in range(world_size):
            if rank_results[r] is not None:
                local_mapping, local_num_neighbors, local_outputs = rank_results[r]

                current_mapping, current_num_neighbors, current_outputs = merge_outputs(
                    current_mapping,
                    current_num_neighbors,
                    current_outputs,
                    local_mapping + stride,
                    local_num_neighbors,
                    local_outputs,
                )

                stride += strides[r]

        return current_mapping, current_num_neighbors, current_outputs

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: Any,
    ) -> Tuple[None, ...]:
        """
        Backward pass for distributed ball query computation.

        Args:
            ctx: Context containing saved variables from forward pass
            grad_output: Gradients from subsequent layers

        Returns:
            Gradients for inputs (currently not implemented)
        """
        raise NotImplementedError("Backward pass not implemented")
        return None, None, None, None, None, None, None, None


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
    """
    Wrapper for BallQueryLayer.forward to support sharded tensors.

    Handles 4 situations, based on the sharding of points 1 and points 2:
    - Points 2 is sharded: a ring computation is performed.
        - Points 1 is sharded: each rank contains a partial output,
          which is returned sharded like Points 1.
        - Points 1 is replicated: each rank returns the full output,
          even though the input points 2 is sharded.
    - Points 1 is replicated: No ring is needed.
        - Points 1 is sharded: each rank contains a partial output,
          which is returned sharded like Points 1.
        - Points 1 is replicated: each rank returns the full output,
          even though the input points 2 is sharded.

    All input sharding has to be over a 1D mesh.  2D Point cloud sharding
    is not supported at this time.

    Regardless of the input sharding, the output will always be sharded like
    points 1, and the output points will always have queried every input point
    like in the non-sharded case.

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

        # Make sure all meshes are the same
        if points1._spec.mesh != points2._spec.mesh:
            raise MissingShardPatch(
                "point_cloud_ops.ball_query_wrapper: All point inputs must be on the same mesh"
            )

        # make sure all meshes are 1D
        if points1._spec.mesh.ndim != 1:
            raise MissingShardPatch(
                "point_cloud_ops.ball_query_wrapper: All point inputs must be on 1D meshes"
            )

        # Do we need a ring?
        points2_placement = points2._spec.placements[0]
        if isinstance(points2_placement, Shard):
            # We need a ring
            mapping, num_neighbors, outputs = ring_ball_query(
                points1, points2, lengths1, lengths2, wrapped
            )
        else:
            # No ring is needed
            # Call the original function with local tensors

            mapping, num_neighbors, outputs = wrapped(
                points1.to_local(),
                points2.to_local(),
                lengths1,
                lengths2,
                *args[4:],
                **kwargs,
            )

            mapping = ShardTensor.from_local(
                mapping, points1._spec.mesh, points1._spec.placements, "infer"
            )
            num_neighbors = ShardTensor.from_local(
                num_neighbors, points1._spec.mesh, points1._spec.placements, "infer"
            )
            outputs = ShardTensor.from_local(
                outputs, points1._spec.mesh, points1._spec.placements, "infer"
            )

        return mapping, num_neighbors, outputs

    # If inputs are regular torch tensors, just call the original function
    elif all(isinstance(t, torch.Tensor) for t in (points1, points2)):
        return wrapped(*args, **kwargs)

    # If inputs are mixed types, raise an error
    else:
        raise UndeterminedShardingError(
            "points1 and points2 must be the same types (torch.Tensor or ShardTensor)"
        )
