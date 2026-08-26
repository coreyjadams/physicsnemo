# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Radius search via the optional ``physicsnemo_ops`` compact-cell kernels.

Only the fixed-shape mode (``max_points is not None``) is supported: the
physicsnemo_ops kernel returns capped, zero-padded neighbor indices plus
per-query counts, which maps directly onto physicsnemo's static-shape output
contract. The dynamic mode keeps its existing warp/torch backends.
"""

import torch

from physicsnemo.utils._physicsnemo_ops import physicsnemo_ops_torch

from .utils import format_returns, validate_inputs


def radius_search_eligible(
    points: torch.Tensor,
    queries: torch.Tensor,
    max_points: int | None,
) -> bool:
    """Whether the physicsnemo_ops radius-search kernel can serve this call.

    Requires the fixed-shape mode, plain floating tensors with a trailing dim
    of 3, and rank 2 or 3. Reduced precision (bf16/f16) is accepted and
    cast-restored around the kernel, matching the warp backend's documented
    behavior. Auto-dispatch restricts to CUDA: the CPU kernel is a
    brute-force fallback that would regress against the warp backend.
    Gradients are supported because neighbor points/distances are recomputed
    with differentiable torch ops from the (discrete) indices.
    """
    if max_points is None:
        return False
    if physicsnemo_ops_torch() is None:
        return False
    if type(points) is not torch.Tensor or type(queries) is not torch.Tensor:
        return False
    supported = (torch.float32, torch.bfloat16, torch.float16)
    if points.dtype not in supported or queries.dtype != points.dtype:
        return False
    if points.ndim not in (2, 3) or queries.ndim not in (2, 3):
        return False
    if points.ndim != queries.ndim:
        return False
    if points.shape[-1] != 3 or queries.shape[-1] != 3:
        return False
    if points.device != queries.device:
        return False
    if points.shape[0] == 0 or queries.shape[0] == 0:
        return False
    return True


def radius_search(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int | None = None,
    return_dists: bool = False,
    return_points: bool = False,
):
    """Fixed-shape radius search using physicsnemo_ops compact spatial cells.

    Output contract matches the torch backend's ``max_points`` mode:
    ``indices`` is ``(B?, Q, max_points)`` int64 with unused slots set to 0;
    optional ``points``/``distances`` are zero-filled in unused slots.
    """
    ops = physicsnemo_ops_torch()
    if ops is None:
        raise ImportError(
            "physicsnemo_ops is not available; install physicsnemo-ops or use "
            "another radius_search implementation."
        )
    if max_points is None:
        raise ValueError(
            "The physicsnemo_ops radius_search implementation requires "
            "max_points to be set (fixed-shape mode)."
        )

    points_b, queries_b, was_unbatched = validate_inputs(points, queries)

    # The kernel is index-producing only and float32-only; detach so
    # requires_grad inputs are legal, and cast reduced-precision inputs up
    # (results below are computed from the original-precision tensors, so
    # outputs keep the input dtype — same contract as the warp backend).
    kernel_points = points_b.detach().contiguous()
    kernel_queries = queries_b.detach().contiguous()
    if kernel_points.dtype != torch.float32:
        kernel_points = kernel_points.to(torch.float32)
        kernel_queries = kernel_queries.to(torch.float32)
    raw_indices, counts = ops.radius_search(
        kernel_points,
        kernel_queries,
        float(radius),
        int(max_points),
    )

    # valid: (B, Q, max_points) — slots below each query's neighbor count.
    slot = torch.arange(max_points, device=points_b.device)
    valid = slot < counts.unsqueeze(-1).to(slot.dtype)
    indices = raw_indices.long()  # already zero-padded in unused slots

    if return_points or return_dists:
        # Gather neighbor coordinates: (B, Q*max_points, 3) -> (B, Q, max_points, 3)
        B, Q = queries_b.shape[0], queries_b.shape[1]
        flat_idx = indices.reshape(B, Q * max_points, 1).expand(-1, -1, 3)
        gathered = torch.gather(points_b, 1, flat_idx).reshape(B, Q, max_points, 3)
    else:
        gathered = None

    if return_dists:
        deltas = gathered - queries_b.unsqueeze(2)
        dists_out = torch.linalg.vector_norm(deltas, dim=-1) * valid
    else:
        dists_out = torch.empty(0, dtype=points_b.dtype, device=points_b.device)

    if return_points:
        pts_out = gathered * valid.unsqueeze(-1)
    else:
        pts_out = torch.empty(
            0, max_points, 3, device=points_b.device, dtype=points_b.dtype
        )

    if was_unbatched:
        indices = indices.squeeze(0)
        if return_dists:
            dists_out = dists_out.squeeze(0)
        if return_points:
            pts_out = pts_out.squeeze(0)

    return format_returns(indices, pts_out, dists_out, return_dists, return_points)
