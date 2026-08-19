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

r"""Generic domain-parallel reading: ZarrReader -> Dataset -> ShardTensors.

Rank 0 writes seeded zarr groups to a shared dir; every rank reads through
``ZarrReader(domain_parallel=..., device_mesh=...)`` + ``Dataset`` and must
see per-key ``Shard(0)`` ShardTensors (or plain replicated tensors) whose
gathered values match the on-disk sample -- in manual and auto placement
modes, and composed with coordinated subsampling.
"""

import numpy as np
import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor.placement_types import Shard

from physicsnemo.datapipes import Dataset
from physicsnemo.datapipes.readers.zarr import ZarrReader
from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor

zarr = pytest.importorskip("zarr")

pytestmark = [pytest.mark.multigpu_static, pytest.mark.timeout(300)]

# Uneven on 2/4/8 ranks.
_N_ROWS = 1234
_N_SAMPLES = 2


def _sample_arrays(sample: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(97 + sample)
    return {
        "coords": rng.standard_normal((_N_ROWS, 3), dtype=np.float32),
        "fields": rng.standard_normal((_N_ROWS, 4), dtype=np.float32),
        "params": rng.standard_normal((7,), dtype=np.float32),
    }


@pytest.fixture(scope="module")
def zarr_root(tmp_path_factory, distributed_mesh):
    r"""Shared directory of zarr groups; rank 0 writes, path broadcast."""
    dm = DistributedManager()
    if dm.rank == 0:
        root = tmp_path_factory.mktemp("sharded_zarr")
        for i in range(_N_SAMPLES):
            group = zarr.open_group(str(root / f"sample_{i}.zarr"), mode="w")
            for key, value in _sample_arrays(i).items():
                group[key] = value
        holder = [str(root)]
    else:
        holder = [None]
    dist.broadcast_object_list(holder, src=0)
    return holder[0]


def _make_dataset(zarr_root, distributed_mesh, **reader_kwargs):
    dm = DistributedManager()
    return Dataset(
        ZarrReader(zarr_root, device_mesh=distributed_mesh, **reader_kwargs),
        device=dm.device,
    )


def _assert_sharded_key(td, key, reference, device):
    assert isinstance(td[key], ShardTensor)
    assert td[key]._spec.placements == (Shard(0),)
    torch.testing.assert_close(
        td[key].full_tensor(), torch.from_numpy(reference).to(device)
    )


def _assert_replicated_key(td, key, reference, device):
    assert not isinstance(td[key], ShardTensor)
    torch.testing.assert_close(td[key], torch.from_numpy(reference).to(device))


def test_manual_placements(zarr_root, distributed_mesh):
    r"""Explicit per-key shard/replicate map; absent keys replicate."""
    dm = DistributedManager()
    dataset = _make_dataset(
        zarr_root,
        distributed_mesh,
        domain_parallel={"placements": {"coords": "shard", "fields": "shard"}},
    )
    try:
        for i in range(_N_SAMPLES):
            td, metadata = dataset[i]
            reference = _sample_arrays(i)
            _assert_sharded_key(td, "coords", reference["coords"], dm.device)
            _assert_sharded_key(td, "fields", reference["fields"], dm.device)
            # 'params' is absent from the map -> replicated.
            _assert_replicated_key(td, "params", reference["params"], dm.device)
    finally:
        dataset.close()


def test_auto_placements_default_threshold(zarr_root, distributed_mesh):
    r"""Default auto gate (one row per rank): every key with rows >= world
    size shards; below world size it replicates."""
    dm = DistributedManager()
    world_size = distributed_mesh.size(0)
    dataset = _make_dataset(
        zarr_root, distributed_mesh, domain_parallel={"placements": "auto"}
    )
    try:
        td, _ = dataset[0]
        reference = _sample_arrays(0)
        _assert_sharded_key(td, "coords", reference["coords"], dm.device)
        _assert_sharded_key(td, "fields", reference["fields"], dm.device)
        if world_size <= 7:
            _assert_sharded_key(td, "params", reference["params"], dm.device)
        else:
            _assert_replicated_key(td, "params", reference["params"], dm.device)
    finally:
        dataset.close()


def test_auto_threshold_units(zarr_root, distributed_mesh):
    r"""The configurable auto threshold controls the shard decision."""
    dm = DistributedManager()
    # rows unit: only keys with >= 1000 rows shard -> params (7) replicates,
    # coords/fields (1234) shard.
    dataset = _make_dataset(
        zarr_root,
        distributed_mesh,
        domain_parallel={
            "placements": "auto",
            "auto_threshold": {"value": 1000, "unit": "rows"},
        },
    )
    try:
        td, _ = dataset[0]
        reference = _sample_arrays(0)
        _assert_sharded_key(td, "coords", reference["coords"], dm.device)
        _assert_replicated_key(td, "params", reference["params"], dm.device)
    finally:
        dataset.close()

    # bytes unit with a huge threshold: nothing shards.
    dataset = _make_dataset(
        zarr_root,
        distributed_mesh,
        domain_parallel={
            "placements": "auto",
            "auto_threshold": {"value": 10**12, "unit": "bytes"},
        },
    )
    try:
        td, _ = dataset[0]
        reference = _sample_arrays(0)
        _assert_replicated_key(td, "coords", reference["coords"], dm.device)
        _assert_replicated_key(td, "fields", reference["fields"], dm.device)
    finally:
        dataset.close()


def test_sharded_read_composes_with_subsampling(zarr_root, distributed_mesh):
    r"""The rank chunk is taken OF the coordinated window: the gathered
    sharded read equals a full (unsharded) read with the same seed/epoch."""
    dm = DistributedManager()
    n_window = 600
    subsampling = {"n_points": n_window, "target_keys": ["coords", "fields"]}
    seed = 1234

    def build(domain_parallel=None):
        reader_kwargs = {"coordinated_subsampling": subsampling}
        if domain_parallel is not None:
            reader_kwargs["domain_parallel"] = domain_parallel
            reader_kwargs["device_mesh"] = distributed_mesh
        dataset = Dataset(ZarrReader(zarr_root, **reader_kwargs), device=dm.device)
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataset.set_generator(generator)
        dataset.set_epoch(3)
        return dataset

    reference_ds = build()
    sharded_ds = build(domain_parallel={"placements": "auto"})
    try:
        reference, _ = reference_ds[1]
        td, _ = sharded_ds[1]

        for key in ("coords", "fields"):
            assert isinstance(td[key], ShardTensor)
            assert td[key].shape[0] == n_window
            torch.testing.assert_close(td[key].full_tensor(), reference[key])
        # Replicated keys are identical to the reference read.
        torch.testing.assert_close(td["params"], reference["params"])
    finally:
        reference_ds.close()
        sharded_ds.close()


def test_manual_shard_below_world_size_raises(zarr_root, distributed_mesh):
    r"""Manually sharding a key with fewer rows than ranks is an error."""
    if distributed_mesh.size(0) <= 7:
        pytest.skip("params (7 rows) only violates the floor above 7 ranks")
    dataset = _make_dataset(
        zarr_root,
        distributed_mesh,
        domain_parallel={"placements": {"params": "shard"}},
    )
    try:
        with pytest.raises(ValueError, match="world size"):
            dataset[0]
    finally:
        dataset.close()
