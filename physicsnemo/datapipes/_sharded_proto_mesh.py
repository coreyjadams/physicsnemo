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

r"""This class exists to bridge the gap between loading a mesh partially,
on the CPU, and transfering it to GPU, vs. building it fully into a domain
parallel mesh.

Rank-local slicing and ShardTensor wrapping for mesh datapipes.

Splits "read a sharded mesh" into the two stages MeshDataset already has:

- :func:`shard_slice_mesh` (host, worker thread): slice this rank's rows out
  of a full mesh. On a memmap-backed mesh (``Mesh.load``) the slice is the
  actual disk read -- each rank faults in only its own pages.
- :func:`wrap_sharded_mesh` (device, consumer thread): rebuild the ``Mesh``
  with ``Shard(0)`` ShardTensors via the communication-free chunk path.

Layout follows the domain-parallel mesh model: ``points`` / ``point_data``
sharded over n_points, ``cell_data`` sharded over n_cells, ``cells`` and
``global_data`` replicated. Caches are dropped (they recompute lazily
through ShardTensor ops). Both stages use the same chunk arithmetic as
``ShardTensor.from_local(sharding_shapes="chunk")``, so the wrap needs no
collectives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.datapipes._domain_parallel import auto_shard, chunk_bounds
from physicsnemo.mesh import DomainMesh, Mesh

_PLACEMENTS = (Shard(0),)


def _should_shard(
    mesh: Mesh,
    device_mesh: DeviceMesh,
    threshold: dict[str, Any] | None = None,
) -> bool:
    r"""Size gate: shard a (sub-)mesh only if every batch dim splits sensibly.

    Applies the shared auto gate to the points array (respecting the
    configurable ``auto_threshold``) and additionally requires the cell
    count -- when the mesh has cells -- to reach the world size: below that
    the chunk arithmetic degenerates and empty local shards poison
    reductions. Deterministic across ranks: reads only global sizes.

    Parameters
    ----------
    mesh : Mesh
        The (sub-)mesh under consideration; only its shapes are read.
    device_mesh : DeviceMesh
        1-D device mesh the rows would be sharded over.
    threshold : dict, optional
        ``auto_threshold`` dict (see ``_domain_parallel``); default one row
        per rank.

    Returns
    -------
    bool
        ``True`` if the mesh should be sharded, ``False`` to replicate it.
    """
    if not auto_shard(
        (mesh.n_points, *mesh.points.shape[1:]),
        mesh.points.dtype,
        device_mesh,
        threshold,
    ):
        return False
    if mesh.n_cells > 0 and mesh.n_cells < device_mesh.size(0):
        return False
    return True


@dataclass
class ShardedProtoMesh:
    r"""This rank's rows of one mesh sample, plus the global batch sizes.

    Not a valid ``Mesh``: the replicated cells reference global point
    indices. :func:`wrap_sharded_mesh` restores the global index space by
    wrapping the rows as ShardTensors.

    Parameters
    ----------
    tensors : TensorDict
        Local rows: ``points``, ``cells`` (full), and the ``point_data`` /
        ``cell_data`` / ``global_data`` sub-TensorDicts.
    global_n_points : int
        Global point count of the full sample.
    global_n_cells : int
        Global cell count of the full sample.
    device_mesh : DeviceMesh
        1-D device mesh the rows were sliced against; the wrap reuses it.
    sharded : bool
        ``True`` when the rows are this rank's chunk (wrap to ShardTensors);
        ``False`` when the size gate replicated the mesh (rows are the full
        sample and the wrap builds a plain ``Mesh``).
    """

    tensors: TensorDict
    global_n_points: int
    global_n_cells: int
    device_mesh: DeviceMesh
    sharded: bool = True

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "ShardedProtoMesh":
        r"""Return a copy with the local rows moved to *device*.

        Matches the ``.to`` signature MeshDataset uses on its samples, so
        the payload flows through the device-transfer seam unchanged.

        Parameters
        ----------
        device : torch.device
            Target device for the local row tensors.
        non_blocking : bool, default=False
            Passed through to ``TensorDict.to`` for async H2D copies.

        Returns
        -------
        ShardedProtoMesh
            Payload on *device*; global batch sizes are unchanged.
        """
        return ShardedProtoMesh(
            tensors=self.tensors.to(device, non_blocking=non_blocking),
            global_n_points=self.global_n_points,
            global_n_cells=self.global_n_cells,
            device_mesh=self.device_mesh,
            sharded=self.sharded,
        )

    def pin_memory(self) -> "ShardedProtoMesh":
        r"""Return a copy with the local rows in pinned host memory.

        Returns
        -------
        ShardedProtoMesh
            Payload whose tensors are page-locked, enabling asynchronous
            host-to-device copies.
        """
        return ShardedProtoMesh(
            tensors=self.tensors.pin_memory(),
            global_n_points=self.global_n_points,
            global_n_cells=self.global_n_cells,
            device_mesh=self.device_mesh,
            sharded=self.sharded,
        )


def shard_slice_mesh(
    mesh: Mesh,
    device_mesh: DeviceMesh,
    threshold: dict[str, Any] | None = None,
    *,
    force_replicate: bool = False,
) -> ShardedProtoMesh:
    r"""Slice this rank's rows out of a full (possibly lazy) mesh.

    Shape queries and row slices are the only accesses, so a memmap-backed
    mesh reads only this rank's byte ranges from disk. Caches are dropped;
    they recompute lazily through ShardTensor ops after the wrap.

    Parameters
    ----------
    mesh : Mesh
        Full mesh sample, typically the lazy memmap-backed result of
        ``Mesh.load``.
    device_mesh : DeviceMesh
        1-D device mesh to shard over. Every rank must call with the same
        sample so the chunk bounds agree.
    threshold : dict, optional
        ``auto_threshold`` dict for the size gate; default one row per rank.
    force_replicate : bool, default=False
        Skip the gate and read the full mesh (``sharded=False``) regardless
        of size.

    Returns
    -------
    ShardedProtoMesh
        This rank's rows in plain host memory, plus the global batch sizes
        needed by :func:`wrap_sharded_mesh`. When the size gate rejects the
        mesh, the payload carries ALL rows with ``sharded=False`` and wraps
        to a plain replicated ``Mesh``.
    """
    sharded = not force_replicate and _should_shard(mesh, device_mesh, threshold)
    if sharded:
        p_lo, p_hi = chunk_bounds(mesh.n_points, device_mesh)
        c_lo, c_hi = chunk_bounds(mesh.n_cells, device_mesh)
    else:
        p_lo, p_hi = 0, mesh.n_points
        c_lo, c_hi = 0, mesh.n_cells

    # clone() materializes memmap rows into plain host memory NOW, on the
    # calling (worker) thread -- this is the actual disk read. Batch sizes
    # are LOCAL here; the global sizes ride alongside for the wrap.
    def read(t: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(t).clone()

    tensors = TensorDict(
        {
            "points": read(mesh.points[p_lo:p_hi]),
            "cells": read(mesh.cells),
            "point_data": TensorDict(
                {k: read(v[p_lo:p_hi]) for k, v in mesh.point_data.items()},
                batch_size=[p_hi - p_lo],
            ),
            "cell_data": TensorDict(
                {k: read(v[c_lo:c_hi]) for k, v in mesh.cell_data.items()},
                batch_size=[c_hi - c_lo],
            ),
            "global_data": TensorDict(
                {k: read(v) for k, v in mesh.global_data.items()},
                batch_size=[],
            ),
        },
        batch_size=[],
    )
    return ShardedProtoMesh(
        tensors=tensors,
        global_n_points=mesh.n_points,
        global_n_cells=mesh.n_cells,
        device_mesh=device_mesh,
        sharded=sharded,
    )


def wrap_sharded_mesh(payload: ShardedProtoMesh) -> Mesh:
    r"""Rebuild a ``Mesh`` from local rows as ``Shard(0)`` ShardTensors.

    Uses the chunk sharding-shape path of ``ShardTensor.from_local``, which
    needs no communication: every rank derives identical shard shapes from
    the global batch sizes carried in *payload*. A ``sharded=False`` payload
    (size-gated) rebuilds a plain replicated ``Mesh`` instead.

    Parameters
    ----------
    payload : ShardedProtoMesh
        This rank's rows, already moved to the target device; carries the
        device mesh the rows were sliced against.

    Returns
    -------
    Mesh
        Mesh with global batch sizes whose ``points`` / ``point_data`` /
        ``cell_data`` are ``Shard(0)`` ShardTensors; ``cells`` and
        ``global_data`` stay plain (replicated).
    """
    from physicsnemo.domain_parallel import ShardTensor

    td = payload.tensors

    if not payload.sharded:
        return Mesh(
            points=td["points"],
            cells=td["cells"],
            point_data=dict(td["point_data"]),
            cell_data=dict(td["cell_data"]),
            global_data=dict(td["global_data"]),
        )

    def wrap(local: torch.Tensor, global_n: int) -> ShardTensor:
        return ShardTensor.from_local(
            local,
            payload.device_mesh,
            _PLACEMENTS,
            sharding_shapes="chunk",
            global_shape=(global_n, *local.shape[1:]),
        )

    return Mesh(
        points=wrap(td["points"], payload.global_n_points),
        cells=td["cells"],
        point_data={
            k: wrap(v, payload.global_n_points) for k, v in td["point_data"].items()
        },
        cell_data={
            k: wrap(v, payload.global_n_cells) for k, v in td["cell_data"].items()
        },
        global_data=dict(td["global_data"]),
    )


@dataclass
class ShardedProtoDomainMesh:
    r"""This rank's rows of one DomainMesh sample.

    Each sub-mesh (interior and every named boundary) carries its own
    :class:`ShardedProtoMesh`, independently size-gated. The domain-level
    ``global_data`` stays replicated.

    Parameters
    ----------
    interior : ShardedProtoMesh
        The interior sub-mesh payload.
    boundaries : dict[str, ShardedProtoMesh]
        Payload per named boundary.
    global_data : TensorDict
        Domain-level global data (replicated rows).
    """

    interior: ShardedProtoMesh
    boundaries: dict[str, ShardedProtoMesh]
    global_data: TensorDict

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "ShardedProtoDomainMesh":
        r"""Return a copy with all local rows moved to *device*.

        Parameters
        ----------
        device : torch.device
            Target device for the local row tensors.
        non_blocking : bool, default=False
            Passed through for async H2D copies.

        Returns
        -------
        ShardedProtoDomainMesh
            Payload on *device*.
        """
        return ShardedProtoDomainMesh(
            interior=self.interior.to(device, non_blocking=non_blocking),
            boundaries={
                k: v.to(device, non_blocking=non_blocking)
                for k, v in self.boundaries.items()
            },
            global_data=self.global_data.to(device, non_blocking=non_blocking),
        )

    def pin_memory(self) -> "ShardedProtoDomainMesh":
        r"""Return a copy with all local rows in pinned host memory.

        Returns
        -------
        ShardedProtoDomainMesh
            Payload whose tensors are page-locked.
        """
        return ShardedProtoDomainMesh(
            interior=self.interior.pin_memory(),
            boundaries={k: v.pin_memory() for k, v in self.boundaries.items()},
            global_data=self.global_data.pin_memory(),
        )


def shard_slice_domain_mesh(
    domain: DomainMesh,
    device_mesh: DeviceMesh,
    threshold: dict[str, Any] | None = None,
    replicate_names: set[str] | None = None,
) -> ShardedProtoDomainMesh:
    r"""Slice this rank's rows out of a full (possibly lazy) DomainMesh.

    Applies :func:`shard_slice_mesh` to the interior and to each boundary
    independently -- each sub-mesh is size-gated on its own global counts,
    so large surfaces shard while small patches replicate. On a
    memmap-backed domain (``DomainMesh.load``) only this rank's byte ranges
    are read.

    Parameters
    ----------
    domain : DomainMesh
        Full domain sample, typically the lazy result of ``DomainMesh.load``.
    device_mesh : DeviceMesh
        1-D device mesh to shard over. Every rank must call with the same
        sample so the chunk bounds agree.
    threshold : dict, optional
        ``auto_threshold`` dict for the per-sub-mesh size gate.
    replicate_names : set[str], optional
        Boundary names to exclude from sharding regardless of size. Used
        for geometry that downstream ops need whole on every rank (e.g.
        SDF reference surfaces).

    Returns
    -------
    ShardedProtoDomainMesh
        Per-sub-mesh payloads plus the replicated domain global_data.
    """
    replicate_names = replicate_names or set()
    return ShardedProtoDomainMesh(
        interior=shard_slice_mesh(domain.interior, device_mesh, threshold),
        boundaries={
            name: shard_slice_mesh(
                boundary,
                device_mesh,
                threshold,
                force_replicate=name in replicate_names,
            )
            for name, boundary in domain.boundaries.items()
        },
        global_data=TensorDict(
            {k: torch.as_tensor(v).clone() for k, v in domain.global_data.items()},
            batch_size=[],
        ),
    )


def wrap_sharded_domain_mesh(payload: ShardedProtoDomainMesh) -> DomainMesh:
    r"""Rebuild a ``DomainMesh`` from per-sub-mesh payloads.

    Parameters
    ----------
    payload : ShardedProtoDomainMesh
        This rank's rows, already moved to the target device; each sub-mesh
        payload carries the device mesh it was sliced against.

    Returns
    -------
    DomainMesh
        Domain whose sufficiently large sub-meshes carry ``Shard(0)``
        ShardTensors; size-gated sub-meshes and the domain ``global_data``
        stay plain (replicated).
    """
    return DomainMesh(
        interior=wrap_sharded_mesh(payload.interior),
        boundaries={
            name: wrap_sharded_mesh(proto) for name, proto in payload.boundaries.items()
        },
        global_data=dict(payload.global_data),
    )
