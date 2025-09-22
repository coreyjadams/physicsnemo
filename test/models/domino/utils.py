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

from dataclasses import dataclass
from typing import Sequence

import pytest
import torch


def generate_test_data(bsize, nx, ny, nz, num_neigh, device):
    """Generate test data for DoMINO"""
    return {
        "pos_volume_closest": torch.randn(bsize, 50, 3).to(device),
        "pos_volume_center_of_mass": torch.randn(bsize, 50, 3).to(device),
        "pos_surface_center_of_mass": torch.randn(bsize, 50, 3).to(device),
        "geometry_coordinates": torch.randn(bsize, 50, 3).to(device),
        "grid": torch.randn(bsize, nx, ny, nz, 3).to(device),
        "surf_grid": torch.randn(bsize, nx, ny, nz, 3).to(device),
        "sdf_grid": torch.randn(bsize, nx, ny, nz).to(device),
        "sdf_surf_grid": torch.randn(bsize, nx, ny, nz).to(device),
        "sdf_nodes": torch.randn(bsize, 50, 1).to(device),
        "surface_mesh_centers": torch.randn(bsize, 50, 3).to(device),
        "surface_mesh_neighbors": torch.randn(bsize, 50, num_neigh, 3).to(device),
        "surface_normals": torch.randn(bsize, 50, 3).to(device),
        "surface_neighbors_normals": torch.randn(bsize, 50, num_neigh, 3).to(device),
        "surface_areas": torch.rand(bsize, 50).to(device) + 1e-6,
        "surface_neighbors_areas": torch.rand(bsize, 50, num_neigh).to(device) + 1e-6,
        "volume_mesh_centers": torch.randn(bsize, 50, 3).to(device),
        "volume_min_max": torch.randn(bsize, 2, 3).to(device),
        "surface_min_max": torch.randn(bsize, 2, 3).to(device),
        "global_params_values": torch.randn(bsize, 2, 1).to(device),
        "global_params_reference": torch.randn(bsize, 2, 1).to(device),
    }


@pytest.fixture
def base_model_params():
    """Base model parameters for testing"""

    @dataclass
    class model_params:
        @dataclass
        class geometry_rep:
            @dataclass
            class geo_conv:
                base_neurons: int = 32
                base_neurons_in: int = 8
                base_neurons_out: int = 8
                surface_hops: int = 1
                volume_hops: int = 1
                volume_radii: Sequence = (0.1, 0.5)
                volume_neighbors_in_radius: Sequence = (10, 10)
                surface_radii: Sequence = (0.05,)
                surface_neighbors_in_radius: Sequence = (10,)
                activation: str = "relu"
                fourier_features: bool = False
                num_modes: int = 5

            @dataclass
            class geo_processor:
                base_filters: int = 8
                activation: str = "relu"
                processor_type: str = "unet"
                self_attention: bool = True
                cross_attention: bool = False

            base_filters: int = 8
            geo_conv = geo_conv
            geo_processor = geo_processor

        @dataclass
        class geometry_local:
            base_layer: int = 512
            volume_neighbors_in_radius: Sequence = (128, 128)
            surface_neighbors_in_radius: Sequence = (128,)
            volume_radii: Sequence = (0.05, 0.1)
            surface_radii: Sequence = (0.05,)

        @dataclass
        class nn_basis_functions:
            base_layer: int = 512
            fourier_features: bool = False
            num_modes: int = 5
            activation: str = "relu"

        @dataclass
        class local_point_conv:
            activation: str = "relu"

        @dataclass
        class aggregation_model:
            base_layer: int = 512
            activation: str = "relu"

        @dataclass
        class position_encoder:
            base_neurons: int = 512
            activation: str = "relu"
            fourier_features: bool = False
            num_modes: int = 5

        @dataclass
        class parameter_model:
            base_layer: int = 512
            fourier_features: bool = True
            num_modes: int = 5
            activation: str = "relu"

        model_type: str = "combined"
        activation: str = "relu"
        interp_res: Sequence = (64, 64, 64)  # Smaller for testing
        use_sdf_in_basis_func: bool = True
        positional_encoding: bool = False
        surface_neighbors: bool = True
        num_neighbors_surface: int = 7
        num_neighbors_volume: int = 7
        use_surface_normals: bool = True
        use_surface_area: bool = True
        encode_parameters: bool = False
        combine_volume_surface: bool = False
        geometry_encoding_type: str = "both"
        solution_calculation_mode: str = "two-loop"
        geometry_rep = geometry_rep
        nn_basis_functions = nn_basis_functions
        aggregation_model = aggregation_model
        position_encoder = position_encoder
        geometry_local = geometry_local

    return model_params


def validate_output_shape_and_values(output, expected_shape, check_finite=True):
    """Validate output tensor shape and values"""
    if output is not None:
        assert output.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {output.shape}"
        )
        if check_finite:
            assert torch.isfinite(output).all(), "Output contains non-finite values"
        assert not torch.isnan(output).any(), "Output contains NaN values"
