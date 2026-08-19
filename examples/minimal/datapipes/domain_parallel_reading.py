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

r"""Minimal domain-parallel (rank-local) reading example.

Each rank reads only its chunk of the large arrays straight from disk; the
sample arrives as ``Shard(0)`` ShardTensors over a device mesh you construct
in Python and inject into the reader. Small arrays replicate automatically.

Run with::

    torchrun --nproc-per-node 2 examples/minimal/datapipes/domain_parallel_reading.py

The pattern to take away for recipes:

1. The ``domain_parallel`` dict is plain, Hydra-friendly configuration.
2. The ``DeviceMesh`` is NOT configuration -- construct it at runtime from
   ``DistributedManager`` and pass it to the reader alongside the dict.
3. The dataset needs no domain-parallel arguments at all; readers own the
   rank-local read, datasets assemble the ShardTensors on the GPU.
"""

import tempfile
from pathlib import Path

import numpy as np
import torch.distributed as dist
import zarr

from physicsnemo.datapipes import Dataset
from physicsnemo.datapipes.readers.zarr import ZarrReader
from physicsnemo.distributed import DistributedManager


def generate_sample_data(root: Path, n_samples: int = 4, n_points: int = 100_000):
    r"""Write example zarr groups: two large point-wise arrays, one small one.

    The size split is deliberate: ``coords`` and ``fields`` sit above the
    example's ``auto_threshold`` and will be sharded across the domain mesh,
    while ``params`` sits below it and will replicate.

    Parameters
    ----------
    root : Path
        Directory to write the ``sample_<i>.zarr`` groups into.
    n_samples : int, default=4
        Number of zarr groups (samples) to create.
    n_points : int, default=100_000
        Number of rows in the large point-wise arrays.
    """
    rng = np.random.default_rng(0)
    for i in range(n_samples):
        group = zarr.open_group(str(root / f"sample_{i}.zarr"), mode="w")
        group["coords"] = rng.standard_normal((n_points, 3), dtype=np.float32)
        group["fields"] = rng.standard_normal((n_points, 4), dtype=np.float32)
        group["params"] = rng.standard_normal((8,), dtype=np.float32)


def main():
    r"""Run the domain-parallel reading example end to end.

    Initializes distributed, builds a 1-D device mesh, generates example
    data on rank 0, constructs a ``ZarrReader`` with a declarative
    ``domain_parallel`` policy plus the runtime-injected mesh, and prints
    each key's global shape, rank-local shape, and placement to show which
    arrays were sharded versus replicated.
    """
    DistributedManager.initialize()
    dm = DistributedManager()

    # The device mesh is a runtime object: build it here, in Python, and
    # inject it into the reader. It never appears in yaml/Hydra config.
    device_mesh = dm.initialize_mesh([-1], ["domain"])

    # Rank 0 generates example data in a shared location.
    if dm.rank == 0:
        root = Path(tempfile.mkdtemp(prefix="dp_datapipe_example_"))
        generate_sample_data(root)
        holder = [str(root)]
    else:
        holder = [None]
    dist.broadcast_object_list(holder, src=0)
    data_root = holder[0]

    reader = ZarrReader(
        data_root,
        # Optional: coordinated subsampling composes with domain-parallel
        # reading -- each rank reads its chunk OF the subsampled window.
        coordinated_subsampling={
            "n_points": 50_000,
            "target_keys": ["coords", "fields"],
        },
        # Declarative policy (Hydra-friendly): shard large keys, replicate
        # small ones, decided from store metadata before any data is read.
        domain_parallel={
            "placements": "auto",
            "auto_threshold": {"value": 1024, "unit": "rows"},
        },
        # Runtime object, injected in Python.
        device_mesh=device_mesh,
    )

    dataset = Dataset(reader, device=dm.device)

    sample, metadata = dataset[0]
    if dm.rank == 0:
        print(f"sample from {metadata['source_filename']}:")
    for key, value in sample.items():
        local = getattr(value, "_local_tensor", value)
        placement = (
            value._spec.placements if hasattr(value, "_spec") else "(replicated)"
        )
        print(
            f"  rank {dm.rank}: {key}: global {tuple(value.shape)}, "
            f"local {tuple(local.shape)}, {placement}"
        )

    dataset.close()
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
