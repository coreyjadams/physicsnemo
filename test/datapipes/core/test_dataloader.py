# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

"""Tests for DataLoader class."""

import pytest
import torch
from tensordict import TensorDict

import physicsnemo.datapipes.core as dp

# ============================================================================
# Basic DataLoader functionality
# ============================================================================


def test_create_dataloader(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    loader = dp.DataLoader(dataset, batch_size=2)

    # 10 samples / 2 batch_size = 5 batches
    assert len(loader) == 5


def test_iterate_batches(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    loader = dp.DataLoader(dataset, batch_size=2)

    batches = list(loader)
    assert len(batches) == 5

    for batched_data, metadata_list in batches:
        assert isinstance(batched_data, TensorDict)
        assert batched_data["positions"].shape[0] == 2  # batch dim


def test_batch_collation(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    loader = dp.DataLoader(dataset, batch_size=4)

    batched_data, metadata_list = next(iter(loader))

    # Should have batch dimension
    assert batched_data["positions"].shape == (4, 100, 3)
    assert batched_data["features"].shape == (4, 100, 8)


def test_metadata_collation(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    loader = dp.DataLoader(dataset, batch_size=3)

    batched_data, metadata_list = next(iter(loader))

    # Metadata should be lists
    assert isinstance(metadata_list, list)
    assert len(metadata_list) == 3
    assert [m["index"] for m in metadata_list] == [0, 1, 2]


def test_drop_last(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    # Without drop_last: 10 samples / 3 = 4 batches (last has 1)
    loader_keep = dp.DataLoader(dataset, batch_size=3, drop_last=False)
    assert len(loader_keep) == 4

    # With drop_last: 10 samples / 3 = 3 batches
    loader_drop = dp.DataLoader(dataset, batch_size=3, drop_last=True)
    assert len(loader_drop) == 3


def test_last_batch_smaller(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    loader = dp.DataLoader(dataset, batch_size=3, drop_last=False)

    batches = list(loader)
    last_batched_data, last_metadata_list = batches[-1]

    # 10 % 3 = 1, so last batch should have 1 sample
    assert last_batched_data["positions"].shape[0] == 1


# ============================================================================
# DataLoader shuffling
# ============================================================================


def test_shuffle_changes_order(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    # Collect indices from multiple epochs
    torch.manual_seed(42)
    loader = dp.DataLoader(dataset, batch_size=2, shuffle=True)

    indices_epoch1 = []
    for batched_data, metadata_list in loader:
        indices_epoch1.extend([m["index"] for m in metadata_list])

    indices_epoch2 = []
    for batched_data, metadata_list in loader:
        indices_epoch2.extend([m["index"] for m in metadata_list])

    # Different epochs should (likely) have different orders
    # Note: there's a tiny chance they're the same, but very unlikely
    # We mainly check that shuffling doesn't break anything
    assert set(indices_epoch1) == set(range(10))
    assert set(indices_epoch2) == set(range(10))


def test_no_shuffle_preserves_order(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    loader = dp.DataLoader(dataset, batch_size=2, shuffle=False)

    indices = []
    for batched_data, metadata_list in loader:
        indices.extend([m["index"] for m in metadata_list])

    assert indices == list(range(10))


# ============================================================================
# DataLoader prefetching
# ============================================================================


def test_prefetch_disabled(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    loader = dp.DataLoader(
        dataset,
        batch_size=2,
        prefetch_factor=0,  # Disabled
    )

    batches = list(loader)
    assert len(batches) == 5


def test_prefetch_enabled(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    loader = dp.DataLoader(
        dataset,
        batch_size=2,
        prefetch_factor=2,
        use_streams=False,  # CPU mode
    )

    batches = list(loader)
    assert len(batches) == 5

    for batched_data, metadata_list in batches:
        assert batched_data["positions"].shape[0] == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prefetch_with_streams(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
    dataset = dp.Dataset(
        reader,
        transforms=dp.ToDevice("cuda"),
    )
    loader = dp.DataLoader(
        dataset,
        batch_size=2,
        prefetch_factor=2,
        num_streams=4,
        use_streams=True,
    )

    batches = list(loader)
    assert len(batches) == 5

    for batched_data, metadata_list in batches:
        assert batched_data["positions"].device.type == "cuda"


def test_disable_prefetch(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    loader = dp.DataLoader(
        dataset,
        batch_size=2,
        prefetch_factor=2,
    )

    loader.disable_prefetch()

    # Should still work in sync mode
    batches = list(loader)
    assert len(batches) == 5


# ============================================================================
# DataLoader custom collation
# ============================================================================


def test_default_collation(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    loader = dp.DataLoader(dataset, batch_size=3)

    batched_data, metadata_list = next(iter(loader))

    # Default collation stacks tensors
    assert batched_data["positions"].shape == (3, 100, 3)


def test_concat_collation(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    loader = dp.DataLoader(
        dataset,
        batch_size=3,
        collate_fn=dp.ConcatCollator(dim=0, add_batch_idx=True),
    )

    batched_data, metadata_list = next(iter(loader))

    # Concat collation concatenates along dim 0
    assert batched_data["positions"].shape == (300, 3)  # 3 * 100 points
    assert "batch_idx" in batched_data
    assert batched_data["batch_idx"].shape == (300,)


def test_custom_collate_fn(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    def my_collate(samples):
        # Just return first sample
        return samples[0]

    loader = dp.DataLoader(
        dataset,
        batch_size=3,
        collate_fn=my_collate,
    )

    result = next(iter(loader))

    # Should be single sample tuple, not batched
    data, metadata = result
    assert data["positions"].shape == (100, 3)


# ============================================================================
# DataLoader with custom samplers
# ============================================================================


def test_sequential_sampler(numpy_data_dir):
    from torch.utils.data import SequentialSampler

    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    sampler = SequentialSampler(dataset)
    loader = dp.DataLoader(dataset, batch_size=2, sampler=sampler)

    indices = []
    for batched_data, metadata_list in loader:
        indices.extend([m["index"] for m in metadata_list])

    assert indices == list(range(10))


def test_random_sampler(numpy_data_dir):
    from torch.utils.data import RandomSampler

    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    torch.manual_seed(123)
    sampler = RandomSampler(dataset)
    loader = dp.DataLoader(dataset, batch_size=2, sampler=sampler)

    indices = []
    for batched_data, metadata_list in loader:
        indices.extend([m["index"] for m in metadata_list])

    # All indices present, but possibly shuffled
    assert set(indices) == set(range(10))


def test_subset_sampler(numpy_data_dir):
    from torch.utils.data import SubsetRandomSampler

    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    # Only use indices 0, 2, 4, 6, 8
    indices = [0, 2, 4, 6, 8]
    sampler = SubsetRandomSampler(indices)
    loader = dp.DataLoader(dataset, batch_size=2, sampler=sampler)

    seen_indices = []
    for batched_data, metadata_list in loader:
        seen_indices.extend([m["index"] for m in metadata_list])

    assert set(seen_indices) == set(indices)


def test_set_epoch(numpy_data_dir):
    """Test set_epoch for DistributedSampler compatibility."""
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    loader = dp.DataLoader(dataset, batch_size=2)

    # Should not raise even if sampler doesn't have set_epoch
    loader.set_epoch(0)
    loader.set_epoch(1)


# ============================================================================
# End-to-end tests
# ============================================================================


def test_training_loop_simulation(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(
        reader,
        transforms=dp.SubsamplePoints(
            input_keys=["positions", "features"], n_points=50
        ),
    )
    loader = dp.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
    )

    # Simulate 3 epochs
    for epoch in range(3):
        loader.set_epoch(epoch)

        total_samples = 0
        for batched_data, metadata_list in loader:
            batch_size = batched_data["positions"].shape[0]
            total_samples += batch_size

            # Verify transform was applied
            assert batched_data["positions"].shape[1] == 50

        assert total_samples == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_training_loop(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
    dataset = dp.Dataset(
        reader,
        transforms=[
            dp.ToDevice("cuda"),
            dp.Normalize(
                input_keys=["positions"],
                method="mean_std",
                means={"positions": 0.0},
                stds={"positions": 1.0},
            ),
        ],
    )
    loader = dp.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        prefetch_factor=2,
        num_streams=4,
    )

    for batched_data, metadata_list in loader:
        assert batched_data["positions"].device.type == "cuda"

        # Simulate forward pass
        _ = batched_data["positions"].mean()

    torch.cuda.synchronize()


# ============================================================================
# DataLoader errors
# ============================================================================


def test_invalid_batch_size(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    with pytest.raises(ValueError, match="batch_size must be >= 1"):
        dp.DataLoader(dataset, batch_size=0)


# ============================================================================
# DataLoader repr
# ============================================================================


def test_dataloader_repr(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)
    loader = dp.DataLoader(dataset, batch_size=4)

    repr_str = repr(loader)
    assert "DataLoader" in repr_str
    assert "batch_size=4" in repr_str
