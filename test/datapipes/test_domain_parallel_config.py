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

r"""Pure-logic tests for the domain-parallel reader configuration.

Placement resolution and the auto gate read only global metadata and mesh
size, so they are testable single-rank with a stub device mesh; the
distributed read/wrap behavior is covered in
``test/domain_parallel/datapipes/``.
"""

import pytest
import torch

from physicsnemo.datapipes._domain_parallel import (
    auto_shard,
    resolve_placements,
    validate_domain_parallel_config,
)


class _StubMesh:
    """Duck-typed 1-D device mesh: just a world size."""

    def __init__(self, world_size: int):
        self._world_size = world_size

    @property
    def ndim(self) -> int:
        return 1

    def size(self, dim: int = 0) -> int:
        return self._world_size


def test_validate_requires_pairing():
    validate_domain_parallel_config(None, None)  # both absent: fine
    with pytest.raises(ValueError, match="together"):
        validate_domain_parallel_config({"placements": "auto"}, None)
    with pytest.raises(ValueError, match="together"):
        validate_domain_parallel_config(None, _StubMesh(2))


def test_validate_rejects_bad_config():
    mesh = _StubMesh(2)
    with pytest.raises(ValueError, match="placements"):
        validate_domain_parallel_config({"placements": 7}, mesh)
    with pytest.raises(ValueError, match="shard"):
        validate_domain_parallel_config({"placements": {"x": "banana"}}, mesh)
    with pytest.raises(ValueError, match="unit"):
        validate_domain_parallel_config(
            {"placements": "auto", "auto_threshold": {"value": 1, "unit": "acres"}},
            mesh,
        )
    validate_domain_parallel_config(
        {"placements": {"x": "shard"}, "auto_threshold": {"unit": "bytes"}}, mesh
    )


@pytest.mark.parametrize(
    ("shape", "threshold", "world", "expected"),
    [
        # Default gate: rows >= world_size.
        ((4, 3), None, 4, True),
        ((3, 3), None, 4, False),
        ((), None, 2, False),  # scalars always replicate
        # world_size unit with a multiplier.
        ((64, 3), {"value": 32, "unit": "world_size"}, 2, True),
        ((63, 3), {"value": 32, "unit": "world_size"}, 2, False),
        # rows unit.
        ((1000, 3), {"value": 1000, "unit": "rows"}, 2, True),
        ((999, 3), {"value": 1000, "unit": "rows"}, 2, False),
        # bytes unit (float32 = 4 bytes/element).
        ((100, 3), {"value": 1200, "unit": "bytes"}, 2, True),
        ((100, 3), {"value": 1201, "unit": "bytes"}, 2, False),
        # The world-size floor applies in every unit.
        ((3, 3), {"value": 1, "unit": "bytes"}, 4, False),
    ],
)
def test_auto_shard_gate(shape, threshold, world, expected):
    assert auto_shard(shape, torch.float32, _StubMesh(world), threshold) is expected


def test_resolve_placements_manual():
    mesh = _StubMesh(2)
    meta = {
        "coords": ((100, 3), torch.float32),
        "params": ((7,), torch.float32),
    }
    decisions = resolve_placements(meta, {"placements": {"coords": "shard"}}, mesh)
    assert decisions == {"coords": True, "params": False}

    # Manually sharding below the world-size floor is an error, not a
    # degenerate split.
    with pytest.raises(ValueError, match="world size"):
        resolve_placements(
            {"tiny": ((1, 3), torch.float32)},
            {"placements": {"tiny": "shard"}},
            mesh,
        )


def test_resolve_placements_auto():
    mesh = _StubMesh(4)
    meta = {
        "big": ((1000, 3), torch.float32),
        "small": ((3,), torch.float32),
    }
    decisions = resolve_placements(meta, {"placements": "auto"}, mesh)
    assert decisions == {"big": True, "small": False}
