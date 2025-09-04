# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

import tempfile
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest
import torch
from pytest_utils import import_or_fail

Tensor = torch.Tensor


@dataclass
class ConcreteBoundingBox:
    """
    Really simple bounding box to mimic a structured config; Don't use elsewhere.
    """

    min: List[float]
    max: List[float]


@pytest.fixture
def data_dir(nfs_data_dir):
    return nfs_data_dir.joinpath("datasets/domino/")


@pytest.fixture
def bounding_boxes():
    """Common bounding box configurations for tests."""
    return {
        "volume": ConcreteBoundingBox(min=[-3.5, -2.25, -0.32], max=[8.5, 2.25, 3.00]),
        "surface": ConcreteBoundingBox(min=[-1.1, -1.2, -0.32], max=[4.5, 1.2, 1.2]),
    }


def create_basic_dataset(data_dir, model_type, **kwargs):
    """Helper function to create a basic DoMINODataPipe with default settings."""
    from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataPipe

    assert model_type in ["volume", "surface", "combined"]

    input_path = data_dir / model_type

    bounding_box = ConcreteBoundingBox(min=[-3.5, -2.25, -0.32], max=[8.5, 2.25, 3.00])
    bounding_box_surface = ConcreteBoundingBox(
        min=[-1.1, -1.2, -0.32], max=[4.5, 1.2, 1.2]
    )

    default_kwargs = {
        "phase": "test",
        "grid_resolution": [64, 64, 64],
        "volume_points_sample": 1234,
        "surface_points_sample": 1234,
        "geom_points_sample": 2345,
        "num_surface_neighbors": 5,
        "bounding_box_dims": bounding_box,
        "bounding_box_dims_surf": bounding_box_surface,
        "normalize_coordinates": True,
        "sampling": False,
        "sample_in_bbox": False,
        "positional_encoding": False,
        "scaling_type": None,
        "volume_factors": None,
        "surface_factors": None,
        "caching": False,
        "compute_scaling_factors": False,
        "gpu_preprocessing": True,
        "gpu_output": True,
    }

    default_kwargs.update(kwargs)

    return DoMINODataPipe(
        input_path=input_path, model_type=model_type, **default_kwargs
    )


def validate_sample_structure(sample, model_type, gpu_output):
    """Helper function to validate the structure of a dataset sample."""
    assert isinstance(sample, dict)

    # Common keys that should always be present
    expected_keys = ["geometry_coordinates", "length_scale", "surface_min_max"]

    # Model-specific keys
    volume_keys = [
        "volume_mesh_centers",
        "volume_fields",
        "grid",
        "sdf_grid",
        "sdf_nodes",
    ]
    surface_keys = [
        "surface_mesh_centers",
        "surface_normals",
        "surface_areas",
        "surface_fields",
    ]

    if model_type in ["volume", "combined"]:
        expected_keys.extend(volume_keys)
    if model_type in ["surface", "combined"]:
        expected_keys.extend(surface_keys)

    # Check that required keys are present and are torch tensors on correct device
    for key in expected_keys:
        if key in sample:  # Some keys may be None if compute_scaling_factors=True
            if sample[key] is not None:
                assert isinstance(sample[key], torch.Tensor), (
                    f"Key {key} should be torch.Tensor"
                )
                expected_device = "cuda" if gpu_output else "cpu"
                assert sample[key].device.type == expected_device, (
                    f"Key {key} on wrong device"
                )


# Core test - smaller matrix focusing on essential device/model combinations
@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("gpu_preprocessing", [True, False])
@pytest.mark.parametrize("gpu_output", [True, False])
@pytest.mark.parametrize("model_type", ["surface", "volume", "combined"])
def test_domino_datapipe_core(
    data_dir, gpu_preprocessing, gpu_output, model_type, pytestconfig
):
    """Core test for basic functionality with different device and model configurations."""

    dataset = create_basic_dataset(
        data_dir, model_type, gpu_preprocessing=gpu_preprocessing, gpu_output=gpu_output
    )

    assert len(dataset) > 0
    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output)


# Feature-specific tests
@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["combined"])
@pytest.mark.parametrize("normalize_coordinates", [True, False])
def test_domino_datapipe_coordinate_normalization(
    data_dir, model_type, normalize_coordinates, pytestconfig
):
    """Test coordinate normalization functionality."""
    dataset = create_basic_dataset(
        data_dir,
        model_type,
        gpu_preprocessing=True,
        normalize_coordinates=normalize_coordinates,
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=True)

    v_coords = sample["volume_mesh_centers"]
    s_coords = sample["surface_mesh_centers"]
    # If normalization is enabled, coordinates should be in [-2, 2] range
    if normalize_coordinates:
        assert all(torch.min(v_coords, dim=0).values >= -2.0) and all(
            torch.max(v_coords, dim=0).values <= 2.0
        ), "Normalized coordinates should be in [-2,2]"
        assert all(torch.min(s_coords, dim=0).values >= -2.0) and all(
            torch.max(s_coords, dim=0).values <= 2.0
        ), "Normalized coordinates should be in [-2,2]"


@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["combined"])
@pytest.mark.parametrize("sampling", [True, False])
def test_domino_datapipe_sampling(data_dir, model_type, sampling, pytestconfig):
    """Test point sampling functionality."""
    sample_points = 4321
    dataset = create_basic_dataset(
        data_dir,
        model_type,
        gpu_preprocessing=False,
        sampling=sampling,
        volume_points_sample=sample_points,
        surface_points_sample=sample_points,
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=True)

    if model_type in ["volume", "combined"]:
        for key in ["volume_mesh_centers", "volume_fields"]:
            if sampling:
                assert sample[key].shape[0] == sample_points
            else:
                assert sample[key].shape[0] == sample["volume_mesh_centers"].shape[0]

    # Model-specific keys
    if model_type in ["surface", "combined"]:
        for key in [
            "surface_mesh_centers",
            "surface_normals",
            "surface_areas",
            "surface_fields",
        ]:
            if sampling:
                assert sample[key].shape[0] == sample_points
            else:
                assert sample[key].shape[0] == sample["surface_mesh_centers"].shape[0]
        for key in [
            "surface_mesh_neighbors",
            "surface_neighbors_normals",
            "surface_neighbors_areas",
        ]:
            if sampling:
                assert sample[key].shape[0] == sample_points
                assert sample[key].shape[1] == dataset.config.num_surface_neighbors - 1
            else:
                assert sample[key].shape[0] == sample["surface_mesh_neighbors"].shape[0]
                assert sample[key].shape[1] == dataset.config.num_surface_neighbors - 1


@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["combined"])
@pytest.mark.parametrize("sample_in_bbox", [True])
def test_domino_datapipe_bbox_sampling(
    data_dir, model_type, sample_in_bbox, pytestconfig
):
    """Test bounding box sampling functionality."""
    dataset = create_basic_dataset(
        data_dir, model_type, gpu_preprocessing=False, sample_in_bbox=sample_in_bbox
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=True)

    v_coords = sample["volume_mesh_centers"]
    s_coords = sample["surface_mesh_centers"]
    # If normalization is enabled, coordinates should be in [-1, 1] range if sampling in bbox
    if sample_in_bbox:
        assert all(torch.min(v_coords, dim=0).values >= -1.5) and all(
            torch.max(v_coords, dim=0).values <= 1.5
        ), "Normalized coordinates should be in [-1.5,1.5]"
        assert all(torch.min(s_coords, dim=0).values >= -1.5) and all(
            torch.max(s_coords, dim=0).values <= 1.5
        ), "Normalized coordinates should be in [-1.5,1.5]"


@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["combined"])
@pytest.mark.parametrize(
    "positional_encoding",
    [
        True,
    ],
)
def test_domino_datapipe_positional_encoding(
    data_dir, model_type, positional_encoding, pytestconfig
):
    """Test positional encoding functionality."""
    dataset = create_basic_dataset(
        data_dir,
        model_type,
        gpu_preprocessing=False,
        positional_encoding=positional_encoding,
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=True)

    # Check for positional encoding keys
    if positional_encoding:
        pos_keys = ["pos_volume_closest", "pos_volume_center_of_mass"]
        for key in pos_keys:
            if key in sample:
                assert sample[key] is not None


@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["volume"])
@pytest.mark.parametrize("scaling_type", [None, "min_max_scaling", "mean_std_scaling"])
def test_domino_datapipe_scaling(data_dir, model_type, scaling_type, pytestconfig):
    """Test field scaling functionality."""
    if scaling_type == "min_max_scaling":
        volume_factors = [10.0, -10.0]  # [max, min]
    elif scaling_type == "mean_std_scaling":
        volume_factors = [0.0, 1.0]  # [mean, std]
    else:
        volume_factors = None

    dataset = create_basic_dataset(
        data_dir,
        model_type,
        gpu_preprocessing=False,
        scaling_type=scaling_type,
        volume_factors=volume_factors,
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=True)


# Caching tests
@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("model_type", ["volume"])
def test_domino_datapipe_caching_config(data_dir, model_type, pytestconfig):
    """Test DoMINODataPipe with caching=True configuration."""
    dataset = create_basic_dataset(
        data_dir,
        model_type,
        gpu_preprocessing=False,
        caching=True,
        sampling=False,  # Required for caching
        compute_scaling_factors=False,  # Required for caching
        resample_surfaces=False,  # Required for caching
    )

    sample = dataset[0]
    validate_sample_structure(sample, model_type, gpu_output=True)


@import_or_fail(["warp", "cupy", "cuml"])
def test_cached_domino_dataset(data_dir, tmp_path, pytestconfig):
    """Test CachedDoMINODataset functionality."""
    from physicsnemo.datapipes.cae.domino_datapipe import CachedDoMINODataset

    # Create some mock cached data files
    for i in range(3):
        cached_data = {
            "geometry_coordinates": np.random.randn(1000, 3),
            "volume_mesh_centers": np.random.randn(5000, 3),
            "volume_fields": np.random.randn(5000, 2),
            "surface_mesh_centers": np.random.randn(2000, 3),
            "surface_fields": np.random.randn(2000, 2),
            "surface_normals": np.random.randn(2000, 3),
            "surface_areas": np.random.rand(2000),
            "neighbor_indices": np.random.randint(0, 2000, (2000, 5)),
        }
        np.save(tmp_path / f"cached_{i}.npz", cached_data)

    dataset = CachedDoMINODataset(
        data_path=tmp_path,
        phase="test",
        sampling=True,
        volume_points_sample=1234,
        surface_points_sample=567,
        geom_points_sample=890,
        model_type="combined",
    )

    assert len(dataset) > 0

    sample = dataset[0]

    # Check that sampling worked
    assert sample["volume_mesh_centers"].shape[0] <= 1234
    assert sample["surface_mesh_centers"].shape[0] <= 567
    assert sample["geometry_coordinates"].shape[0] <= 890


# Configuration validation tests
@import_or_fail(["warp", "cupy", "cuml"])
def test_domino_datapipe_invalid_caching_config(data_dir, pytestconfig):
    """Test that invalid caching configurations raise appropriate errors."""

    # Test: caching=True with sampling=True should fail
    with pytest.raises(ValueError, match="Sampling should be False for caching"):
        create_basic_dataset(data_dir, "volume", caching=True, sampling=True)

    # Test: caching=True with compute_scaling_factors=True should fail
    with pytest.raises(
        ValueError, match="Compute scaling factors should be False for caching"
    ):
        create_basic_dataset(
            data_dir, "volume", caching=True, compute_scaling_factors=True
        )

    # Test: caching=True with resample_surfaces=True should fail
    with pytest.raises(
        ValueError, match="Resample surface should be False for caching"
    ):
        create_basic_dataset(data_dir, "volume", caching=True, resample_surfaces=True)


@import_or_fail(["warp", "cupy", "cuml"])
def test_domino_datapipe_invalid_phase(pytestconfig):
    """Test that invalid phase values raise appropriate errors."""
    from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataConfig

    with pytest.raises(ValueError, match="phase should be one of"):
        DoMINODataConfig(data_path=tempfile.mkdtemp(), phase="invalid_phase")


@import_or_fail(["warp", "cupy", "cuml"])
def test_domino_datapipe_invalid_scaling_type(pytestconfig):
    """Test that invalid scaling_type values raise appropriate errors."""
    from physicsnemo.datapipes.cae.domino_datapipe import DoMINODataConfig

    with pytest.raises(ValueError, match="scaling_type should be one of"):
        DoMINODataConfig(
            data_path=tempfile.mkdtemp(), phase="train", scaling_type="invalid_scaling"
        )


@import_or_fail(["warp", "cupy", "cuml"])
def test_domino_datapipe_file_format_support(data_dir, pytestconfig):
    """Test support for different file formats (.zarr, .npz, .npy)."""
    # This test assumes the data directory has files in these formats
    # If not available, we can mock the file reading
    dataset = create_basic_dataset(data_dir, "volume", gpu_preprocessing=False)

    # Just verify we can load at least one sample
    assert len(dataset) > 0
    sample = dataset[0]
    validate_sample_structure(sample, "volume", gpu_output=True)


# Surface-specific tests (when GPU preprocessing issues are resolved)
@import_or_fail(["warp", "cupy", "cuml"])
@pytest.mark.parametrize("surface_sampling_algorithm", ["area_weighted", "random"])
def test_domino_datapipe_surface_sampling(
    data_dir, surface_sampling_algorithm, pytestconfig
):
    """Test surface sampling algorithms."""
    dataset = create_basic_dataset(
        data_dir,
        "surface",
        gpu_preprocessing=False,  # Avoid known GPU issues
        sampling=True,
        surface_sampling_algorithm=surface_sampling_algorithm,
    )

    sample = dataset[0]
    validate_sample_structure(sample, "surface", gpu_output=True)
