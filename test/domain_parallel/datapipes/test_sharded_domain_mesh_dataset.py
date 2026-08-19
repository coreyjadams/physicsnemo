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

r"""Rank-local sharded reading of DomainMesh through ``MeshDataset``.

Same skeleton as ``test_sharded_mesh_dataset.py``, over ``.pdmsh`` samples:
interior and a large boundary shard, a tiny boundary is size-gated to a
plain replicated mesh, and the ``apply_to_domain`` transform path is
value-checked against the unsharded reference.
"""

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.datapipes import MeshDataset
from physicsnemo.datapipes.readers.mesh import DomainMeshReader
from physicsnemo.datapipes.transforms.mesh.transforms import (
    CenterMesh,
    NormalizeMeshFields,
)
from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor
from physicsnemo.mesh import DomainMesh, Mesh

pytestmark = [pytest.mark.multigpu_static, pytest.mark.timeout(300)]

# Uneven on 2/4/8 ranks for both batch dims of the sharded sub-meshes.
_N_INTERIOR_CELLS = 137
_N_SURFACE_CELLS = 61


def _triangle_soup(n_cells: int, seed: int, x_offset: float = 0.0) -> Mesh:
    r"""One well-conditioned triangle per cell, copies offset along x."""
    torch.manual_seed(seed)
    base = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    offsets = torch.zeros(n_cells, 1, 3)
    offsets[:, 0, 0] = x_offset + 2.0 * torch.arange(n_cells)
    points = (base.unsqueeze(0) + offsets).reshape(-1, 3)
    points = points + 0.01 * torch.randn_like(points)
    cells = torch.arange(3 * n_cells, dtype=torch.int64).reshape(-1, 3)
    return Mesh(points=points, cells=cells)


def _build_full_domain(sample: int) -> DomainMesh:
    r"""Seeded domain, identical on all ranks: sizeable interior + one large
    boundary (shards) + one 3-point probe boundary (size-gated, replicates)."""
    interior = _triangle_soup(_N_INTERIOR_CELLS, seed=211 + sample)
    interior.point_data["temperature"] = torch.randn(interior.n_points)
    interior.cell_data["pressure"] = torch.randn(interior.n_cells)

    surface = _triangle_soup(_N_SURFACE_CELLS, seed=307 + sample, x_offset=500.0)
    surface.cell_data["wss"] = torch.randn(surface.n_cells, 3)

    probe = _triangle_soup(1, seed=401 + sample, x_offset=-500.0)
    probe.cell_data["flux"] = torch.randn(probe.n_cells)

    return DomainMesh(
        interior=interior,
        boundaries={"surface": surface, "probe": probe},
        global_data={"Re": torch.tensor(1.0e6), "AoA": torch.tensor(5.0)},
    )


@pytest.fixture(scope="module")
def pdmsh_root(tmp_path_factory, distributed_mesh):
    r"""Shared directory of .pdmsh samples; rank 0 writes, path broadcast."""
    dm = DistributedManager()
    if dm.rank == 0:
        root = tmp_path_factory.mktemp("sharded_pdmsh")
        for i in range(2):
            _build_full_domain(i).save(root / f"sample_{i}.pdmsh")
        holder = [str(root)]
    else:
        holder = [None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def _make_dataset(pdmsh_root, distributed_mesh, transforms=None):
    dm = DistributedManager()
    return MeshDataset(
        DomainMeshReader(
            pdmsh_root,
            domain_parallel={"placements": "auto"},
            device_mesh=distributed_mesh,
        ),
        transforms=transforms,
        device=dm.device,
    )


def _assert_sharded_submesh(mesh: Mesh, full: Mesh, device) -> None:
    assert isinstance(mesh.points, ShardTensor)
    assert mesh.points._spec.placements == (Shard(0),)
    assert mesh.n_points == full.n_points
    torch.testing.assert_close(mesh.points.full_tensor(), full.points.to(device))
    torch.testing.assert_close(mesh.cells, full.cells.to(device))
    for key, value in full.point_data.items():
        assert isinstance(mesh.point_data[key], ShardTensor)
        torch.testing.assert_close(mesh.point_data[key].full_tensor(), value.to(device))
    for key, value in full.cell_data.items():
        assert isinstance(mesh.cell_data[key], ShardTensor)
        torch.testing.assert_close(mesh.cell_data[key].full_tensor(), value.to(device))


def _assert_replicated_submesh(mesh: Mesh, full: Mesh, device) -> None:
    assert not isinstance(mesh.points, ShardTensor)
    torch.testing.assert_close(mesh.points, full.points.to(device))
    torch.testing.assert_close(mesh.cells, full.cells.to(device))
    for key, value in full.cell_data.items():
        assert not isinstance(mesh.cell_data[key], ShardTensor)
        torch.testing.assert_close(mesh.cell_data[key], value.to(device))


def test_sharded_domain_read_sync_path(pdmsh_root, distributed_mesh):
    r"""dataset[i]: interior + large boundary shard, probe boundary is
    size-gated to a plain replicated mesh, domain global_data plain."""
    dm = DistributedManager()
    dataset = _make_dataset(pdmsh_root, distributed_mesh)
    try:
        for i in range(2):
            domain, metadata = dataset[i]
            full = _build_full_domain(i)

            assert isinstance(domain, DomainMesh)
            assert domain.boundary_names == ["probe", "surface"]
            _assert_sharded_submesh(domain.interior, full.interior, dm.device)
            _assert_sharded_submesh(
                domain.boundaries["surface"], full.boundaries["surface"], dm.device
            )
            _assert_replicated_submesh(
                domain.boundaries["probe"], full.boundaries["probe"], dm.device
            )
            for key in ("Re", "AoA"):
                assert not isinstance(domain.global_data[key], ShardTensor)
                torch.testing.assert_close(
                    domain.global_data[key], full.global_data[key].to(dm.device)
                )
    finally:
        dataset.close()


def test_sharded_domain_read_producer_consumer_path(pdmsh_root, distributed_mesh):
    r"""_load_host -> _consume: host payload carries local rows for the
    sharded sub-meshes and full rows for the gated one."""
    from physicsnemo.datapipes._sharded_proto_mesh import ShardedProtoDomainMesh

    dataset = _make_dataset(pdmsh_root, distributed_mesh)
    try:
        payload = dataset._load_host(0)
        assert payload.error is None
        assert isinstance(payload.data, ShardedProtoDomainMesh)
        assert payload.data.interior.sharded
        assert payload.data.boundaries["surface"].sharded
        assert not payload.data.boundaries["probe"].sharded
        interior_local = payload.data.interior.tensors["points"].shape[0]
        assert interior_local < 3 * _N_INTERIOR_CELLS or distributed_mesh.size(0) == 1
        assert payload.data.interior.tensors["points"].device.type == "cpu"

        domain, _ = dataset._consume(payload)
        full = _build_full_domain(0)
        dm = DistributedManager()
        _assert_sharded_submesh(domain.interior, full.interior, dm.device)
        _assert_replicated_submesh(
            domain.boundaries["probe"], full.boundaries["probe"], dm.device
        )
    finally:
        dataset.close()


def test_sharded_domain_transforms(pdmsh_root, distributed_mesh):
    r"""apply_to_domain chain: CenterMesh reduces the COM from the sharded
    interior and translates every sub-mesh (including the replicated probe);
    NormalizeMeshFields is elementwise on cell_data. Both must match the
    unsharded reference."""
    fields = {"pressure": {"type": "scalar", "mean": 101325.0, "std": 250.0}}

    def make_transforms():
        return [
            CenterMesh(use_area_weighting=False),
            NormalizeMeshFields(association="cell_data", fields=fields),
        ]

    dm = DistributedManager()
    dataset = _make_dataset(pdmsh_root, distributed_mesh, transforms=make_transforms())
    try:
        domain, _ = dataset[0]
    finally:
        dataset.close()

    reference = _build_full_domain(0).to(dm.device)
    for t in make_transforms():
        if hasattr(t, "to"):
            t.to(dm.device)
        reference = t.apply_to_domain(reference)

    assert isinstance(domain.interior.points, ShardTensor)
    # 1e-4 on points: CenterMesh's COM is a per-rank partial sum resolved
    # by an all-reduce; fp32 summation-order jitter vs the single-device
    # reference is a few 1e-5 on coordinates spanning O(1e2) units.
    torch.testing.assert_close(
        domain.interior.points.full_tensor(),
        reference.interior.points,
        atol=1e-4,
        rtol=1e-4,
    )
    torch.testing.assert_close(
        domain.interior.cell_data["pressure"].full_tensor(),
        reference.interior.cell_data["pressure"],
        atol=1e-5,
        rtol=1e-5,
    )
    # The replicated probe must receive the same translation as the
    # sharded interior.
    torch.testing.assert_close(
        domain.boundaries["probe"].points,
        reference.boundaries["probe"].points,
        atol=1e-4,
        rtol=1e-4,
    )
    surface = domain.boundaries["surface"].points
    torch.testing.assert_close(
        surface.full_tensor() if hasattr(surface, "full_tensor") else surface,
        reference.boundaries["surface"].points,
        atol=1e-4,
        rtol=1e-4,
    )


def test_size_gate(distributed_mesh):
    r"""_should_shard boundary conditions against the live world size."""
    from physicsnemo.datapipes._sharded_proto_mesh import _should_shard

    world_size = distributed_mesh.size(0)

    at_gate = _triangle_soup(world_size, seed=1)
    assert _should_shard(at_gate, distributed_mesh)

    if world_size > 1:
        below_gate = _triangle_soup(world_size - 1, seed=2)
        assert not _should_shard(below_gate, distributed_mesh)

    # A point cloud (no cells) gates on n_points only.
    torch.manual_seed(3)
    cloud = Mesh(points=torch.randn(world_size, 3))
    assert _should_shard(cloud, distributed_mesh)
