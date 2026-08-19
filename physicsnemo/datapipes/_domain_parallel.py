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

r"""Shared machinery for domain-parallel (rank-local) reading in datapipes.

Readers that support domain-parallel reading accept a ``domain_parallel``
configuration dict plus a 1-D ``device_mesh`` (constructed and injected at
runtime in Python -- it is not serializable config). Each rank reads only
its chunk of the sharded keys; the local pieces move to the GPU; and the
sample is assembled with ``Shard(0)`` ShardTensors (or plain replicated
tensors) at TensorDict creation time via the communication-free chunk path
of ``ShardTensor.from_local``.

The ``domain_parallel`` dict schema::

    domain_parallel = {
        # per-key placement, or "auto" to decide from metadata shapes
        "placements": {key: "shard" | "replicate"} | "auto",
        # read only in auto mode; default {"value": 1, "unit": "world_size"}
        "auto_threshold": {"value": K, "unit": "world_size" | "rows" | "bytes"},
    }

Auto-mode gate: shard a key iff its dim-0 length reaches ``K * world_size``
(unit ``world_size``), ``K`` rows (unit ``rows``), or its global size
reaches ``K`` bytes (unit ``bytes``) -- and, in every unit, the dim-0
length is at least ``world_size`` (a chunk split below world size is
degenerate). The decision reads only global metadata, so it is identical on
every rank.

This module hosts the chunk arithmetic, the placement resolution, the
generic host-stage payload (:class:`ShardedProtoTensorDict`), and the
GPU-side wrap. Mesh-specific counterparts live in ``_sharded_proto_mesh``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.placement_types import Shard

_PLACEMENTS = (Shard(0),)

_DEFAULT_AUTO_THRESHOLD = {"value": 1, "unit": "world_size"}
_VALID_UNITS = ("world_size", "rows", "bytes")


def validate_domain_parallel_config(
    config: dict[str, Any] | None, device_mesh: DeviceMesh | None
) -> None:
    r"""Validate the ``domain_parallel`` dict / ``device_mesh`` pairing.

    Parameters
    ----------
    config : dict or None
        The ``domain_parallel`` configuration dict.
    device_mesh : DeviceMesh or None
        The device mesh the rows would be sharded over.

    Raises
    ------
    ValueError
        On a missing/extra pairing, a non-1-D mesh, an unknown placements
        spec, or an unknown auto-threshold unit.
    """
    if config is None and device_mesh is None:
        return
    if (config is None) != (device_mesh is None):
        raise ValueError("domain_parallel and device_mesh must be provided together")
    if device_mesh.ndim != 1:
        raise ValueError(f"device_mesh must be 1-D, got {device_mesh.ndim} dims")
    placements = config.get("placements", "auto")
    if placements != "auto" and not isinstance(placements, dict):
        raise ValueError(
            f'placements must be "auto" or a dict of key -> '
            f'"shard"|"replicate", got {placements!r}'
        )
    if isinstance(placements, dict):
        bad = {k: v for k, v in placements.items() if v not in ("shard", "replicate")}
        if bad:
            raise ValueError(f'placements values must be "shard"|"replicate": {bad}')
    threshold = config.get("auto_threshold", _DEFAULT_AUTO_THRESHOLD)
    if threshold.get("unit", "world_size") not in _VALID_UNITS:
        raise ValueError(
            f"auto_threshold unit must be one of {_VALID_UNITS}, "
            f"got {threshold.get('unit')!r}"
        )


def chunk_bounds(global_n: int, device_mesh: DeviceMesh) -> tuple[int, int]:
    r"""This rank's ``[start, stop)`` row range under chunk semantics.

    Uses the same shard-shape arithmetic as
    ``ShardTensor.from_local(sharding_shapes="chunk")``, so rows sliced with
    these bounds are exactly the local shard the later wrap declares.

    Parameters
    ----------
    global_n : int
        Global length of the batch dimension being sharded.
    device_mesh : DeviceMesh
        1-D device mesh the rows are sharded over.

    Returns
    -------
    tuple[int, int]
        Half-open ``[start, stop)`` row range owned by this rank.
    """
    from physicsnemo.domain_parallel._shard_tensor_spec import (
        compute_sharding_shapes_from_chunking_global_shape,
    )

    shapes = compute_sharding_shapes_from_chunking_global_shape(
        device_mesh, _PLACEMENTS, (global_n,)
    )
    sizes = [s[0] for s in shapes[0]]
    rank = device_mesh.get_local_rank(0)
    start = sum(sizes[:rank])
    return start, start + sizes[rank]


def auto_shard(
    global_shape: tuple[int, ...],
    dtype: torch.dtype,
    device_mesh: DeviceMesh,
    threshold: dict[str, Any] | None = None,
) -> bool:
    r"""Auto-mode gate: should a tensor of this global shape shard on dim 0?

    Parameters
    ----------
    global_shape : tuple[int, ...]
        Global tensor shape (from store metadata; no data read).
    dtype : torch.dtype
        Tensor dtype (used by the ``bytes`` unit).
    device_mesh : DeviceMesh
        1-D device mesh the rows would be sharded over.
    threshold : dict, optional
        ``{"value": K, "unit": "world_size" | "rows" | "bytes"}``; defaults
        to one row per rank (``{"value": 1, "unit": "world_size"}``).

    Returns
    -------
    bool
        ``True`` to shard, ``False`` to replicate. Scalars (0-dim) always
        replicate, and any tensor with fewer dim-0 rows than the world size
        replicates regardless of the threshold.
    """
    threshold = threshold or _DEFAULT_AUTO_THRESHOLD
    world_size = device_mesh.size(0)
    if len(global_shape) == 0:
        return False
    rows = global_shape[0]
    if rows < world_size:
        return False

    value = threshold.get("value", 1)
    unit = threshold.get("unit", "world_size")
    if unit == "world_size":
        return rows >= value * world_size
    if unit == "rows":
        return rows >= value
    # unit == "bytes" (validated upstream)
    n_elements = 1
    for s in global_shape:
        n_elements *= s
    return n_elements * dtype.itemsize >= value


def resolve_placements(
    meta: dict[str, tuple[tuple[int, ...], torch.dtype]],
    config: dict[str, Any],
    device_mesh: DeviceMesh,
) -> dict[str, bool]:
    r"""Decide shard-vs-replicate per key from config and global metadata.

    Parameters
    ----------
    meta : dict[str, tuple[shape, dtype]]
        Per-key global shape and dtype (from store metadata; no data read).
    config : dict
        The ``domain_parallel`` dict (see module docstring).
    device_mesh : DeviceMesh
        1-D device mesh the rows would be sharded over.

    Returns
    -------
    dict[str, bool]
        Per-key sharding decision. In manual mode, keys absent from the
        placements map replicate; a manually sharded key with fewer dim-0
        rows than the world size raises rather than degenerating.
    """
    placements = config.get("placements", "auto")
    world_size = device_mesh.size(0)

    if placements == "auto":
        threshold = config.get("auto_threshold", _DEFAULT_AUTO_THRESHOLD)
        return {
            key: auto_shard(shape, dtype, device_mesh, threshold)
            for key, (shape, dtype) in meta.items()
        }

    decisions: dict[str, bool] = {}
    for key, (shape, _dtype) in meta.items():
        shard = placements.get(key, "replicate") == "shard"
        if shard and (len(shape) == 0 or shape[0] < world_size):
            raise ValueError(
                f"key {key!r} is configured to shard but has dim-0 length "
                f"{shape[0] if shape else 0} < world size {world_size}"
            )
        decisions[key] = shard
    return decisions


@dataclass
class ShardedProtoTensorDict:
    r"""This rank's rows of one sample, before GPU-side ShardTensor assembly.

    The host-stage payload of a domain-parallel read: local rows per key,
    the global dim-0 length of each sharded key (``None`` for replicated
    keys -- their rows are already complete), and the device mesh the rows
    were sliced against.

    Parameters
    ----------
    tensors : TensorDict
        Per-key local rows (replicated keys carry full rows).
    global_lengths : dict[str, int or None]
        Global dim-0 length per sharded key; ``None`` marks a replicated key.
    device_mesh : DeviceMesh
        1-D device mesh used for the slice; the wrap reuses it.
    """

    tensors: TensorDict
    global_lengths: dict[str, int | None]
    device_mesh: DeviceMesh

    def to(
        self, device: torch.device, non_blocking: bool = False
    ) -> "ShardedProtoTensorDict":
        r"""Return a copy with the local rows moved to *device*.

        Parameters
        ----------
        device : torch.device
            Target device for the local row tensors.
        non_blocking : bool, default=False
            Passed through to ``TensorDict.to`` for async H2D copies.

        Returns
        -------
        ShardedProtoTensorDict
            Payload on *device*.
        """
        return ShardedProtoTensorDict(
            tensors=self.tensors.to(device, non_blocking=non_blocking),
            global_lengths=self.global_lengths,
            device_mesh=self.device_mesh,
        )

    def pin_memory(self) -> "ShardedProtoTensorDict":
        r"""Return a copy with the local rows in pinned host memory.

        Returns
        -------
        ShardedProtoTensorDict
            Payload whose tensors are page-locked.
        """
        return ShardedProtoTensorDict(
            tensors=self.tensors.pin_memory(),
            global_lengths=self.global_lengths,
            device_mesh=self.device_mesh,
        )


def wrap_sharded_tensordict(payload: ShardedProtoTensorDict) -> TensorDict:
    r"""Assemble the sample TensorDict with ShardTensor leaves.

    Sharded keys wrap through the chunk path of ``ShardTensor.from_local``
    (no communication: every rank derives identical shard shapes from the
    global lengths in *payload*); replicated keys pass through as plain
    tensors.

    Parameters
    ----------
    payload : ShardedProtoTensorDict
        This rank's rows, already moved to the target device.

    Returns
    -------
    TensorDict
        Sample with global batch semantics; sharded keys are ``Shard(0)``
        ShardTensors.
    """
    from physicsnemo.domain_parallel import ShardTensor

    out: dict[str, torch.Tensor] = {}
    for key, local in payload.tensors.items():
        global_n = payload.global_lengths.get(key)
        if global_n is None:
            out[key] = local
        else:
            out[key] = ShardTensor.from_local(
                local,
                payload.device_mesh,
                _PLACEMENTS,
                sharding_shapes="chunk",
                global_shape=(global_n, *local.shape[1:]),
            )
    return TensorDict(out, batch_size=[])
