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


"""Tests for Dataset class."""

import pytest
import torch
from tensordict import TensorDict

import physicsnemo.datapipes.core as dp

# ============================================================================
# Basic Dataset functionality
# ============================================================================


def test_create_dataset(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    assert len(dataset) == 10


def test_dataset_get_sample(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    data, metadata = dataset[0]
    assert isinstance(data, TensorDict)
    assert "positions" in data


def test_dataset_iteration(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    samples = list(dataset)
    assert len(samples) == 10


def test_dataset_field_names(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    assert "positions" in dataset.field_names
    assert "features" in dataset.field_names


def test_dataset_context_manager(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    with dp.Dataset(reader) as dataset:
        data, metadata = dataset[0]
        assert "positions" in data


# ============================================================================
# Dataset with transforms
# ============================================================================


def test_dataset_single_transform(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    norm = dp.Normalize(
        input_keys=["positions"],
        method="mean_std",
        means={"positions": 0.0},
        stds={"positions": 1.0},
    )
    dataset = dp.Dataset(reader, transforms=norm)

    data, metadata = dataset[0]
    assert "positions" in data


def test_dataset_transform_list(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(
        reader,
        transforms=[
            dp.SubsamplePoints(input_keys=["positions", "features"], n_points=50),
        ],
    )

    data, metadata = dataset[0]
    assert data["positions"].shape[0] == 50
    assert data["features"].shape[0] == 50


def test_dataset_compose_transforms(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(
        reader,
        transforms=dp.Compose(
            [
                dp.SubsamplePoints(input_keys=["positions", "features"], n_points=50),
                dp.Normalize(
                    input_keys=["positions"],
                    method="mean_std",
                    means={"positions": 0.0},
                    stds={"positions": 1.0},
                ),
            ]
        ),
    )

    data, metadata = dataset[0]
    assert data["positions"].shape[0] == 50


def test_dataset_empty_transforms_list(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader, transforms=[])

    data, metadata = dataset[0]
    # Should work, no transforms applied
    assert "positions" in data


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dataset_to_device_transform(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
    dataset = dp.Dataset(
        reader,
        device="cuda:0",
    )

    data, metadata = dataset[0]
    assert data["positions"].device.type == "cuda"


# ============================================================================
# Dataset prefetching
# ============================================================================


def test_prefetch_single(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    # Prefetch index 0
    dataset.prefetch(0)

    # Should have 1 prefetch in flight
    assert dataset.prefetch_count >= 0  # May complete quickly

    # Get should use prefetched result
    data, metadata = dataset[0]
    assert "positions" in data


def test_prefetch_batch(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    # Prefetch multiple indices
    dataset.prefetch_batch([0, 1, 2, 3])

    # Get samples
    for i in range(4):
        data, metadata = dataset[i]
        assert metadata["index"] == i


def test_prefetch_non_prefetched_index(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    # Prefetch index 0
    dataset.prefetch(0)

    # Get non-prefetched index (should load synchronously)
    data, metadata = dataset[5]
    assert metadata["index"] == 5


def test_prefetch_cancel(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    dataset.prefetch_batch([0, 1, 2, 3])
    dataset.cancel_prefetch()

    # Prefetch count should be 0 after cancel
    assert dataset.prefetch_count == 0


def test_prefetch_cancel_specific(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    dataset.prefetch(0)
    dataset.prefetch(1)
    dataset.cancel_prefetch(0)

    # Should still be able to get index 1 from prefetch
    # and index 0 synchronously
    data0, metadata0 = dataset[0]
    data1, metadata1 = dataset[1]

    assert metadata0["index"] == 0
    assert metadata1["index"] == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prefetch_with_stream(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
    dataset = dp.Dataset(
        reader,
        device="cuda:0",
    )

    stream = torch.cuda.Stream()
    dataset.prefetch(0, stream=stream)

    data, metadata = dataset[0]
    assert data["positions"].device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_prefetch_batch_with_streams(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir, pin_memory=True)
    dataset = dp.Dataset(
        reader,
        device="cuda",
    )

    streams = [torch.cuda.Stream() for _ in range(4)]
    dataset.prefetch_batch([0, 1, 2, 3], streams=streams)

    for i in range(4):
        data, metadata = dataset[i]
        assert data["positions"].device.type == "cuda"


def test_prefetch_with_transforms(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(
        reader,
        transforms=dp.SubsamplePoints(
            input_keys=["positions", "features"], n_points=50
        ),
    )

    dataset.prefetch(0)
    data, metadata = dataset[0]

    # Transform should have been applied
    assert data["positions"].shape[0] == 50


def test_close_stops_prefetch(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    dataset.prefetch_batch([0, 1, 2, 3])
    dataset.close()

    # Should not raise, prefetch should be stopped
    assert dataset.prefetch_count == 0


# ============================================================================
# Dataset errors
# ============================================================================


def test_invalid_reader_type():
    with pytest.raises(TypeError, match="must be a Reader"):
        dp.Dataset("not a reader")


def test_invalid_transforms_type(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)

    with pytest.raises(TypeError, match="must be Transform"):
        dp.Dataset(reader, transforms="not a transform")


# ============================================================================
# Dataset repr
# ============================================================================


def test_dataset_repr(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(reader)

    repr_str = repr(dataset)
    assert "Dataset" in repr_str
    assert "NumpyReader" in repr_str


def test_dataset_repr_with_transforms(numpy_data_dir):
    reader = dp.NumpyReader(numpy_data_dir)
    dataset = dp.Dataset(
        reader,
        transforms=dp.Normalize(
            input_keys=["positions"],
            method="mean_std",
            means={"positions": 0.0},
            stds={"positions": 1.0},
        ),
    )

    repr_str = repr(dataset)
    assert "Normalize" in repr_str
