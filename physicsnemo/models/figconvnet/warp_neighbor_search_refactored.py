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

# ruff: noqa: S101,F722,F821
from typing import Tuple, Union

import torch
import warp as wp
from jaxtyping import Float
from torch import Tensor


@wp.kernel
def _radius_search_count(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    result_count: wp.array(dtype=wp.int32),
    radius: wp.float32,
):
    """
    Loop through the queries and count the number of points within a radius of each query.
    """
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    # This creates a query into the hashgrid.
    # Below it will iterate over all the points with `radius` around qp
    query = wp.hash_grid_query(hashgrid, qp, radius)
    index = int(0)
    result_count_tid = int(0)

    # the index of each neighbor is stored in `index`
    while wp.hash_grid_query_next(query, index):
        # Get the actual neighbor:
        neighbor = points[index]

        # compute distance to neighbor point
        dist = wp.length(qp - neighbor)
        if dist <= radius:
            result_count_tid += 1
    # Store the number of neighbors for this query
    result_count[tid] = result_count_tid


@wp.kernel
def _radius_search_query(
    hashgrid: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    result_offset: wp.array(dtype=wp.int32),
    result_point_idx: wp.array(dtype=wp.int32),
    result_point_dist: wp.array(dtype=wp.float32),
    radius: wp.float32,
):
    """
    Loop through the queries and find the neighbors within a radius of each query.
    Return, via result_*, the following:
    - The offset of the first neighbor for each query
    """
    tid = wp.tid()

    # create grid query around point
    qp = queries[tid]
    query = wp.hash_grid_query(hashgrid, qp, radius)

    index = int(0)
    result_count = int(0)
    offset_tid = result_offset[tid]

    while wp.hash_grid_query_next(query, index):
        # Get the neighbor
        neighbor = points[index]

        # compute distance to neighbor point
        dist = wp.length(qp - neighbor)
        if dist <= radius:
            result_point_idx[offset_tid + result_count] = index
            result_point_dist[offset_tid + result_count] = dist
            result_count += 1


def _create_hashgrid_and_count_neighbors(
    points: torch.Tensor,
    queries: torch.Tensor,
    result_count: torch.Tensor,
    radius: float,
    grid_dim: Union[int, Tuple[int, int, int]] = (128, 128, 128),
    device: str = "cuda",
    stream: torch.cuda.Stream = None,
) -> Tuple[wp.HashGrid, wp.array(dtype=wp.int32)]:

    # convert grid_dim to Tuple if it is int
    if isinstance(grid_dim, int):
        grid_dim = (grid_dim, grid_dim, grid_dim)

    if stream is not None:
        warp_stream = wp.Stream(device, cuda_stream=stream.cuda_stream)
    else:
        warp_stream = wp.Stream(device)

    # Convert from torch to warp
    points_wp = wp.from_torch(points, dtype=wp.vec3)
    queries_wp = wp.from_torch(queries, dtype=wp.vec3)

    # The first step is

    grid = wp.HashGrid(
        dim_x=grid_dim[0],
        dim_y=grid_dim[1],
        dim_z=grid_dim[2],
        device=device,
    )
    grid.build(points=points_wp, radius=2 * radius)
    # For 10M radius search, the result can overflow and fail
    # This function will fill the result_count with the number of neighbors for each query
    wp.launch(
        kernel=_radius_search_count,
        dim=len(queries),
        inputs=[grid.id, points_wp, queries_wp, wp.from_torch(result_count), radius],
        stream=warp_stream,
    )

    return grid, warp_stream


def _fill_result_tensors(
    grid: wp.HashGrid,
    points: wp.array(dtype=wp.vec3),
    queries: wp.array(dtype=wp.vec3),
    torch_offset: torch.Tensor,
    result_point_idx: torch.Tensor,
    result_point_dist: torch.Tensor,
    radius: float,
    device: str,
    stream: torch.cuda.Stream = None,
):

    if stream is not None:
        warp_stream = wp.Stream(device, cuda_stream=stream.cuda_stream)
    else:
        warp_stream = wp.Stream(device)

    wp.launch(
        kernel=_radius_search_query,
        dim=len(queries),
        inputs=[
            grid.id,
            points,
            queries,
            wp.from_torch(torch_offset),
            result_point_idx,
            result_point_dist,
            radius,
        ],
        stream=warp_stream,
    )
    return warp_stream


# CJA - this is not safe to use with more than one stream yet.
# I struggled with warp and torch dual concurrency, it's not worth
# spending more time on it right now.  Don't make this greater than 1
# and expect it to work yet ...
MAX_STREAMS = 1
stream_cache = {}


def get_or_create_stream(device: str, b: int) -> torch.cuda.Stream:
    """
    Get or create a stream for the given index (used here as batch index).
    """
    # return torch.cuda.default_stream(device)
    if b == 0:
        if b not in stream_cache:
            stream_cache[b] = torch.cuda.default_stream(device)
        return stream_cache[b]
    else:
        stream_id = b % MAX_STREAMS
        if stream_id not in stream_cache:
            print(f"Creating stream {stream_id} for device {device}")
            stream_cache[stream_id] = torch.cuda.Stream(device=device)
        return stream_cache[stream_id]


def batched_radius_search_warp(
    points: Float[Tensor, "B N 3"],
    queries: Float[Tensor, "B M 3"],
    radius: float,
    grid_dim: Union[int, Tuple[int, int, int]] = (128, 128, 128),
    device: str = "cuda",
) -> Tuple[
    Float[Tensor, "Q"], Float[Tensor, "Q"], Float[Tensor, "B*M + 1"]
]:  # noqa: F821
    """
    This is a re-implementation of the radius_search_warp function.
    It has more synchronization-aware behavior and consolidates
    sync points as much as possible.

    For batch_size 1, it's the same.  For larger batches, where
    the sync point happens once per example, this is more efficient.

    Args:
        points: [B, N, 3]
        queries: [B, M, 3]
        radius: float
        grid_dim: Union[int, Tuple[int, int, int]]
        device: str

    Returns:n
        neighbor_index: [Q]
        neighbor_distance: [Q]
        neighbor_split: [B*M + 1]
    """
    B, N, _ = points.shape

    assert points.is_contiguous(), "points must be contiguous"
    assert queries.is_contiguous(), "queries must be contiguous"

    # Create a stream for each batch item except the first one
    # For batch size 1, this implicitly uses the default stream

    # The order of operations below is important for performance.
    # First, loop once to create the hashgrid and count the neighbors.

    # Then, loop again to copy the total points count to CPU, and allocate
    # the final output.

    # Finally, loop again to perform the final neighbor search.

    cuda = torch.cuda.current_device()
    default_stream = torch.cuda.default_stream(cuda)
    # First loop: create hashgrid and count neighbors

    # Allocate output storage:
    # This is storage for the output counts:
    counts = [
        torch.zeros(len(queries[b]), device=device, dtype=torch.int32) for b in range(B)
    ]
    offsets = [
        torch.zeros(len(queries[b]) + 1, device=device, dtype=torch.int32)
        for b in range(B)
    ]

    points = torch.unbind(points, dim=0)
    queries = torch.unbind(queries, dim=0)

    ##################################################################
    # Warning
    # There is no stream syncing yet.  Don't use more than one stream.
    ##################################################################

    grids = []
    used_streams = []
    warp_streams = []
    for b in range(B):
        stream = get_or_create_stream(device, b)
        stream.wait_stream(default_stream)
        # with torch.cuda.stream(stream):

        grid, warp_stream = _create_hashgrid_and_count_neighbors(
            points=points[b],
            queries=queries[b],
            result_count=counts[b],
            radius=radius,
            grid_dim=grid_dim,
            device=device,
            stream=stream,
        )
        grids.append(grid)
        used_streams.append(stream)
        warp_streams.append(warp_stream)

        counts[b].record_stream(stream)
        points[b].record_stream(stream)
        queries[b].record_stream(stream)

    # Make sure all of this is on the default stream:
    with torch.cuda.stream(default_stream):

        for b in range(B):
            torch.cumsum(counts[b], dim=0, out=offsets[b][1:])

        # Consolidate the offsets to a single tensor:
        all_offsets = torch.cat([o[-1:] for o in offsets])
        # We will need these numbers on the GPU, too, so compute them and leave them:
        boundaries_gpu = torch.cumsum(all_offsets, 0)
        # Explicitly copy to CPU:
        all_offsets_cpu = all_offsets.cpu().long()

        total_points = all_offsets_cpu.sum()
        # Now, allocate the output tensors and chunk them into the right size per batch:
        result_point_idx = torch.zeros((total_points,), dtype=torch.int32, device=cuda)
        result_point_dist = torch.zeros(
            (total_points,), dtype=torch.float32, device=cuda
        )

        boundaries_cpu = torch.cumsum(all_offsets_cpu, 0)[:-1]
        result_point_idx_list = list(
            torch.tensor_split(result_point_idx, boundaries_cpu)
        )
        result_point_dist_list = list(
            torch.tensor_split(result_point_dist, boundaries_cpu)
        )

        running_offsets = []

    # The previous work MUST complete before progressing.

    # And now, go fill the result point tensors:
    used_streams = []
    warp_streams = []
    for b in range(B):
        stream = get_or_create_stream(device, b)
        # print(f"got stream {stream} for b {b}")
        # stream.wait_stream(default_stream)
        with torch.cuda.stream(stream):
            warp_stream = _fill_result_tensors(
                grid=grids[b],
                points=points[b],
                queries=queries[b],
                torch_offset=offsets[b],
                result_point_idx=result_point_idx_list[b],
                result_point_dist=result_point_dist_list[b],
                radius=radius,
                device=device,
                stream=stream,
            )
        warp_streams.append(warp_stream)
        # Mark the tensors used by this stream:
        points[b].record_stream(stream)
        queries[b].record_stream(stream)
        offsets[b].record_stream(stream)
        result_point_idx_list[b].record_stream(stream)
        result_point_dist_list[b].record_stream(stream)

    for b in range(B):

        result_point_idx_list[b] += N * b
        if b != 0:
            this_offset = offsets[b] + boundaries_gpu[b - 1]
        else:
            this_offset = offsets[b]
        if b != B - 1:
            running_offsets.append(this_offset[:-1])
        else:
            running_offsets.append(this_offset)

    # Neighbor index, Neighbor Distance, Neighbor Split
    return (
        result_point_idx,
        result_point_dist,
        torch.cat(running_offsets),
        total_points,
    )


_WARP_NEIGHBOR_SEARCH_INIT = False
if not _WARP_NEIGHBOR_SEARCH_INIT:
    wp.init()
    _WARP_NEIGHBOR_SEARCH_INIT = True


if __name__ == "__main__":
    torch.manual_seed(42)

    # Test search
    B = 5
    N = 100_000
    M = 200_000
    points = torch.rand(B, N, 3).cuda()
    queries = torch.rand(B, M, 3).cuda()

    radii = [0.05, 0.01, 0.005]
    for radius in radii:
        print(f"Testing radius: {radius}")
        (
            result_point_idx,
            result_point_dist,
            torch_offset,
            total_count,
        ) = batched_radius_search_warp(points=points, queries=queries, radius=radius)
        print(result_point_idx.shape)
        print(result_point_dist.shape)
        print(torch_offset.shape)
        print(f"total_count: {total_count}")

        import ipdb

        ipdb.set_trace()
