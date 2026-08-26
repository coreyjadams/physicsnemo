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

"""kNN via the optional ``physicsnemo_ops`` compact-cell-grid kernels."""

import torch

from physicsnemo.utils._physicsnemo_ops import physicsnemo_ops_torch

#: Hard limit of the physicsnemo_ops kNN kernel.
_MAX_K = 32


def knn_eligible(points: torch.Tensor, queries: torch.Tensor, k: int) -> bool:
    """Whether the physicsnemo_ops kNN kernel can serve this call.

    The kernel is exact but constrained: float32 inputs of shape ``(N, 3)``,
    ``1 <= k <= 32``, and ``k <= N`` (so no ``-1`` padding appears in the
    output). It produces no gradients, so inputs requiring grad fall back.
    Auto-dispatch additionally restricts to CUDA: the CPU kernel is a
    brute-force fallback that would regress against SciPy's KDTree.
    """
    if physicsnemo_ops_torch() is None:
        return False
    if type(points) is not torch.Tensor or type(queries) is not torch.Tensor:
        return False
    if points.dtype != torch.float32 or queries.dtype != torch.float32:
        return False
    if points.ndim != 2 or queries.ndim != 2:
        return False
    if points.shape[-1] != 3 or queries.shape[-1] != 3:
        return False
    if points.device != queries.device:
        return False
    if not (1 <= k <= _MAX_K) or points.shape[0] < k:
        return False
    if points.requires_grad or queries.requires_grad:
        return False
    return True


def knn_impl(
    points: torch.Tensor,
    queries: torch.Tensor,
    k: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact kNN using the physicsnemo_ops two-level compact-cell grid.

    Args:
        points (torch.Tensor): Reference points, shape (N, 3), float32.
        queries (torch.Tensor): Query points, shape (Q, 3), float32.
        k (int): Number of neighbors, 1 <= k <= 32.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - indices (torch.Tensor): int64 indices, shape (Q, k).
            - distances (torch.Tensor): Euclidean distances, shape (Q, k).
    """
    ops = physicsnemo_ops_torch()
    if ops is None:
        raise ImportError(
            "physicsnemo_ops is not available; install physicsnemo-ops or use "
            "another knn implementation."
        )
    indices, squared_distances = ops.knn(points.contiguous(), queries.contiguous(), k)
    return indices.long(), torch.sqrt(squared_distances)
