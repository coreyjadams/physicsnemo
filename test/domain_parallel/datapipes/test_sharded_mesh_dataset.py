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

r"""Rank-local sharded reading through ``MeshDataset(device_mesh=...)``.

Rank 0 writes seeded ``.pmsh`` samples to a shared tmp dir; every rank then
reads through the datapipe with ``device_mesh`` set and must see a ``Mesh``
of ``Shard(0)`` ShardTensors whose gathered values match the full on-disk
sample -- construction, both load paths (sync and producer/consumer), and
the recipe-style transform chain against an unsharded reference.
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.datapipes import MeshDataset
from physicsnemo.datapipes.readers.mesh import MeshReader
from physicsnemo.datapipes.transforms.mesh.transforms import (
    CenterMesh,
    NormalizeMeshFields,
)
from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.mesh import Mesh

pytestmark = [pytest.mark.multigpu_static, pytest.mark.timeout(300)]

# Uneven on 2/4/8 ranks for both batch dims (n_points = 3 * _N_CELLS).
_N_CELLS = 431
_N_SAMPLES = 2


def _build_full_mesh(sample: int) -> Mesh:
    r"""Seeded triangle soup, distinct per sample, identical on all ranks."""
    torch.manual_seed(101 + sample)
    base = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    offsets = torch.zeros(_N_CELLS, 1, 3)
    offsets[:, 0, 0] = 2.0 * torch.arange(_N_CELLS)
    points = (base.unsqueeze(0) + offsets).reshape(-1, 3)
    points = points + 0.01 * torch.randn_like(points)
    cells = torch.arange(3 * _N_CELLS, dtype=torch.int64).reshape(-1, 3)
    mesh = Mesh(points=points, cells=cells)
    mesh.point_data["velocity"] = torch.randn(mesh.n_points, 3)
    mesh.cell_data["pressure"] = torch.randn(mesh.n_cells)
    mesh.cell_data["wss"] = torch.randn(mesh.n_cells, 3)
    mesh.global_data["Re"] = torch.tensor(1.0e6)
    return mesh


@pytest.fixture(scope="module")
def pmsh_root(tmp_path_factory, distributed_mesh):
    r"""Shared directory of .pmsh samples; rank 0 writes, path broadcast."""
    dm = DistributedManager()
    if dm.rank == 0:
        root = tmp_path_factory.mktemp("sharded_pmsh")
        for i in range(_N_SAMPLES):
            _build_full_mesh(i).save(root / f"sample_{i}.pmsh")
        holder = [str(root)]
    else:
        holder = [None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def _make_dataset(pmsh_root, distributed_mesh, transforms=None):
    dm = DistributedManager()
    return MeshDataset(
        MeshReader(
            pmsh_root,
            domain_parallel={"placements": "auto"},
            device_mesh=distributed_mesh,
        ),
        transforms=transforms,
        device=dm.device,
    )


def _assert_sharded_matches_full(mesh, sample: int, distributed_mesh):
    full = _build_full_mesh(sample)
    world_size = distributed_mesh.size(0)

    assert mesh.n_points == full.n_points
    assert mesh.n_cells == full.n_cells
    assert isinstance(mesh.points, ShardTensor)
    assert mesh.points._spec.placements == (Shard(0),)
    assert mesh.points._local_tensor.shape[0] <= -(-full.n_points // world_size)
    assert not isinstance(mesh.cells, ShardTensor)

    device = mesh.points._local_tensor.device
    torch.testing.assert_close(mesh.points.full_tensor(), full.points.to(device))
    torch.testing.assert_close(mesh.cells, full.cells.to(device))
    torch.testing.assert_close(
        mesh.point_data["velocity"].full_tensor(),
        full.point_data["velocity"].to(device),
    )
    for key in ("pressure", "wss"):
        assert isinstance(mesh.cell_data[key], ShardTensor)
        torch.testing.assert_close(
            mesh.cell_data[key].full_tensor(), full.cell_data[key].to(device)
        )
    torch.testing.assert_close(
        mesh.global_data["Re"], full.global_data["Re"].to(device)
    )


def test_sharded_read_sync_path(pmsh_root, distributed_mesh):
    r"""dataset[i] (synchronous _load): sharded Mesh matches the full sample."""
    dataset = _make_dataset(pmsh_root, distributed_mesh)
    try:
        for i in range(_N_SAMPLES):
            mesh, metadata = dataset[i]
            _assert_sharded_matches_full(mesh, i, distributed_mesh)
            assert metadata["index"] == i
    finally:
        dataset.close()


def test_sharded_read_producer_consumer_path(pmsh_root, distributed_mesh):
    r"""_load_host -> _consume (the prefetch stages, no stream): the slice
    happens host-side, the ShardTensor wrap after device transfer."""
    dataset = _make_dataset(pmsh_root, distributed_mesh)
    try:
        payload = dataset._load_host(0)
        assert payload.error is None
        # Host payload carries local rows only.
        local_rows = payload.data.tensors["points"].shape[0]
        assert local_rows < 3 * _N_CELLS or distributed_mesh.size(0) == 1
        assert payload.data.tensors["points"].device.type == "cpu"

        mesh, _ = dataset._consume(payload)
        _assert_sharded_matches_full(mesh, 0, distributed_mesh)
    finally:
        dataset.close()


def test_sharded_read_with_transforms(pmsh_root, distributed_mesh):
    r"""Recipe-style transform chain on the sharded pipe matches the same
    chain applied to the full mesh: CenterMesh is the global reduction,
    NormalizeMeshFields the elementwise cell_data op."""
    fields = {
        "pressure": {"type": "scalar", "mean": 101325.0, "std": 250.0},
        "wss": {"type": "vector", "mean": [1.0, 0.0, 0.0], "std": 0.5},
    }

    def make_transforms():
        return [
            CenterMesh(use_area_weighting=False),
            NormalizeMeshFields(association="cell_data", fields=fields),
        ]

    dm = DistributedManager()
    dataset = _make_dataset(pmsh_root, distributed_mesh, transforms=make_transforms())
    try:
        mesh, _ = dataset[0]
    finally:
        dataset.close()

    reference = _build_full_mesh(0).to(dm.device)
    for t in make_transforms():
        if hasattr(t, "to"):
            t.to(dm.device)
        reference = t(reference)

    assert isinstance(mesh.points, ShardTensor)
    # 1e-4: CenterMesh's COM is a per-rank partial sum resolved by an
    # all-reduce; fp32 summation-order jitter vs the single-device
    # reference is a few 1e-5 on coordinates spanning O(1e3) units.
    torch.testing.assert_close(
        mesh.points.full_tensor(), reference.points, atol=1e-4, rtol=1e-4
    )
    for key in ("pressure", "wss"):
        torch.testing.assert_close(
            mesh.cell_data[key].full_tensor(),
            reference.cell_data[key],
            atol=1e-5,
            rtol=1e-5,
        )
