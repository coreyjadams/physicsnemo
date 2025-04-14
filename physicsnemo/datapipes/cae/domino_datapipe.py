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

"""
This code provides the datapipe for reading the processed npy files,
generating multi-res grids, calculating signed distance fields, 
positional encodings, sampling random points in the volume and on surface, 
normalizing fields and returning the output tensors as a dictionary.

This datapipe also non-dimensionalizes the fields, so the order in which the variables should 
be fixed: velocity, pressure, turbulent viscosity for volume variables and 
pressure, wall-shear-stress for surface variables. The different parameters such as 
variable names, domain resolution, sampling size etc. are configurable in config.yaml. 
"""

import concurrent.futures
import time
from collections import defaultdict
from pathlib import Path
from typing import (
    Literal,
    Optional,
    Sequence,
    Union,
)

import cuml
import cupy as cp
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from physicsnemo.utils.domino.utils import (
    area_weighted_shuffle_array,
    calculate_center_of_mass,
    calculate_normal_positional_encoding,
    create_grid,
    get_filenames,
    normalize,
    pad,
    shuffle_array,
    standardize,
)
from physicsnemo.utils.profiling import profile
from physicsnemo.utils.sdf import signed_distance_field


def domino_collate_fn(batch):
    """
    This function is a custom collation function to move cupy data to torch tensors on the device.

    For things that aren't cupy arrays, fall back to torch.data.default_convert.  Data, here,
    is a dictionary of numpy arrays or cupy arrays.

    """

    def convert(obj):
        if isinstance(obj, cp.ndarray):
            return torch.utils.dlpack.from_dlpack(obj.toDlpack())
        elif isinstance(obj, list):
            return [convert(x) for x in obj]
        elif isinstance(obj, tuple):
            return tuple(convert(x) for x in obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        else:
            return obj

    batch = [convert(sample) for sample in batch]
    return default_collate(batch)


class DoMINODataPipe(Dataset):
    """
    Datapipe for DoMINO

    """

    def __init__(
        self,
        data_path: Union[str, Path],  # Input data path
        phase: Literal["train", "val", "test"] = "train",  # Train, test or val
        surface_variables: Optional[Sequence] = (
            "pMean",
            "wallShearStress",
        ),  # Names of surface variables
        volume_variables: Optional[Sequence] = (
            "UMean",
            "pMean",
        ),  # Names of volume variables
        sampling: bool = False,  # Sampling True or False
        device: int = 0,  # GPU device id
        grid_resolution: Optional[Sequence] = (
            256,
            96,
            64,
        ),  # Resolution of latent grid
        normalize_coordinates: bool = False,  # Normalize coordinates?
        sample_in_bbox: bool = False,  # Sample points in a specified bounding box
        volume_points_sample: int = 1024,  # Number of volume points sampled per batch
        surface_points_sample: int = 1024,  # Number of surface points sampled per batch
        geom_points_sample: int = 300000,  # Number of STL points sampled per batch
        positional_encoding: bool = False,  # Positional encoding, True or False
        volume_factors=None,  # Non-dimensionalization factors for volume variables
        surface_factors=None,  # Non-dimensionalization factors for surface variables
        scaling_type=None,  # Scaling min_max or mean_std
        model_type=None,  # Model_type, surface, volume or combined
        bounding_box_dims=None,  # Dimensions of bounding box
        bounding_box_dims_surf=None,  # Dimensions of bounding box
        compute_scaling_factors=False,
        num_surface_neighbors=11,  # Surface neighbors to consider
        gpu_preprocessing=True,
        gpu_output=True,
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        data_path = data_path.expanduser()

        self.data_path = data_path

        if phase not in [
            "train",
            "val",
            "test",
        ]:
            raise AssertionError(
                f"phase should be one of ['train', 'val', 'test'], got {phase}"
            )

        if not self.data_path.exists():
            raise AssertionError(f"Path {self.data_path} does not exist")

        if not self.data_path.is_dir():
            raise AssertionError(f"Path {self.data_path} is not a directory")

        self.sampling = sampling
        self.grid_resolution = grid_resolution
        self.normalize_coordinates = normalize_coordinates
        self.model_type = model_type
        self.bounding_box_dims = []
        self.bounding_box_dims.append(
            np.asarray(bounding_box_dims.max).astype(np.float32)
        )
        self.bounding_box_dims.append(
            np.asarray(bounding_box_dims.min).astype(np.float32)
        )

        self.bounding_box_dims_surf = []
        self.bounding_box_dims_surf.append(
            np.asarray(bounding_box_dims_surf.max).astype(np.float32)
        )
        self.bounding_box_dims_surf.append(
            np.asarray(bounding_box_dims_surf.min).astype(np.float32)
        )

        self.filenames = get_filenames(self.data_path)
        total_files = len(self.filenames)

        self.phase = phase
        if phase == "train":
            self.indices = np.array(range(total_files))
        elif phase == "val":
            self.indices = np.array(range(total_files))
        elif phase == "test":
            self.indices = np.array(range(total_files))

        np.random.shuffle(self.indices)
        self.surface_variables = surface_variables
        self.volume_variables = volume_variables
        self.volume_points = volume_points_sample
        self.surface_points = surface_points_sample
        self.geom_points_sample = geom_points_sample
        self.sample_in_bbox = sample_in_bbox
        self.device = device
        self.positional_encoding = positional_encoding
        self.volume_factors = volume_factors
        self.surface_factors = surface_factors
        self.scaling_type = scaling_type
        self.compute_scaling_factors = compute_scaling_factors
        self.num_surface_neighbors = num_surface_neighbors
        self.gpu_preprocessing = gpu_preprocessing
        self.gpu_output = gpu_output

        self.knn = cuml.neighbors.NearestNeighbors(
            n_neighbors=self.num_surface_neighbors, algorithm="rbc"
        )

        self.array_provider = cp if self.gpu_preprocessing else np

        # Add dictionary to store preloaded data
        self.preloaded_data = defaultdict(dict)
        self.max_workers = 8  # Default number of worker threads

        # Define here the keys to read for each __getitem__ call

        # Always read these keys
        self.keys_to_read = ["stl_coordinates", "stl_centers", "stl_faces", "stl_areas"]
        self.keys_to_read_if_available = {
            "stream_velocity": 30.00,
            "air_density": 1.205,
        }
        self.volume_keys = ["volume_mesh_centers", "volume_fields"]
        self.surface_keys = [
            "surface_mesh_centers",
            "surface_normals",
            "surface_areas",
            "surface_fields",
        ]

        if self.model_type == "volume" or self.model_type == "combined":
            self.keys_to_read.extend(self.volume_keys)
        if self.model_type == "surface" or self.model_type == "combined":
            self.keys_to_read.extend(self.surface_keys)

    @profile
    def threaded_data_read(self, filepath, max_workers=None, return_futures=False):

        if max_workers is not None:
            self.max_workers = max_workers

        def load_one(key):
            with np.load(filepath) as data:
                if self.gpu_preprocessing:
                    return key, cp.asarray(data[key])
                else:
                    return key, data[key]  # only load this one

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)

        if return_futures:
            # Return the futures instead of waiting
            futures = {key: executor.submit(load_one, key) for key in self.keys_to_read}

            # Also create a future for checking optional keys
            def check_optional_keys():
                np_file = np.load(filepath)
                optional_results = {}
                for key in self.keys_to_read_if_available:
                    if key in np_file.keys():
                        optional_results[key] = np_file[key]
                return optional_results

            futures["_optional_keys"] = executor.submit(check_optional_keys)
            return futures, executor
        else:
            # Original behavior - wait for all to complete
            results = dict(executor.map(load_one, self.keys_to_read))

            # Check the optional ones:
            np_file = np.load(filepath)

            for key in self.keys_to_read_if_available:
                if key in np_file.keys():
                    results[key] = np_file[key]

            executor.shutdown()
            return results

    def __len__(self):
        return len(self.indices)

    @profile
    def preload_index(self, index, max_workers=None):
        """
        Preload volume data for specified indices using ThreadPoolExecutor.

        Args:
            indices: List of indices to preload. If None, preloads all indices.
            max_workers: Maximum number of worker threads. If None, uses default.
        """
        # Skip if model type is not volume or combined
        if self.model_type not in ["volume", "combined"]:
            print("Preloading skipped: model_type is not 'volume' or 'combined'")
            return

        if max_workers is not None:
            self.max_workers = max_workers

        # Get the target file name:
        actual_index = self.indices[index]
        filepath = self.data_path / self.filenames[actual_index]
        print(
            f"Preloading volume data for {index} files with {self.max_workers} workers..."
        )

        # Load the data with futures
        self.preloaded_data[index] = self.threaded_data_read(
            filepath,
            max_workers=self.max_workers,
            return_futures=True,  # Return futures instead of waiting
        )
        print(f"preloaded_data[index]: {self.preloaded_data[index]}")

        print(f"Preloaded volume data for {len(self.preloaded_data)} files")

    @profile
    def preprocess_combined(self, data_dict):

        # Pull these out:
        STREAM_VELOCITY = data_dict["stream_velocity"].astype(
            self.array_provider.float32
        )
        AIR_DENSITY = data_dict["air_density"].astype(self.array_provider.float32)

        stl_vertices = data_dict["stl_coordinates"]
        stl_centers = data_dict["stl_centers"]
        mesh_indices_flattened = data_dict["stl_faces"]
        stl_sizes = data_dict["stl_areas"]

        xp = self.array_provider

        length_scale = xp.amax(xp.amax(stl_vertices, 0) - xp.amin(stl_vertices, 0))

        # Center of mass calculation
        center_of_mass = calculate_center_of_mass(stl_centers, stl_sizes)

        if self.bounding_box_dims_surf is None:
            s_max = xp.amax(stl_vertices, 0)
            s_min = xp.amin(stl_vertices, 0)
        else:
            s_max = xp.asarray(self.bounding_box_dims_surf[0])
            s_min = xp.asarray(self.bounding_box_dims_surf[1])

        nx, ny, nz = self.grid_resolution
        surf_grid = create_grid(s_max, s_min, [nx, ny, nz])
        surf_grid_reshaped = surf_grid.reshape(nx * ny * nz, 3)

        # SDF calculation on the grid using WARP
        sdf_surf_grid = (
            signed_distance_field(
                stl_vertices,
                mesh_indices_flattened,
                surf_grid_reshaped,
                use_sign_winding_number=True,
            )
            # .numpy()
            .reshape(nx, ny, nz)
        )
        # surf_grid = xp.float32(surf_grid)
        # sdf_surf_grid = xp.float32(sdf_surf_grid)
        # surf_grid_max_min = xp.float32(xp.asarray([s_min, s_max]))

        surf_grid_max_min = xp.concatenate([s_min, s_max], axis=0)

        return_dict = {
            "length_scale": length_scale,
            "surf_grid": surf_grid,
            "sdf_surf_grid": sdf_surf_grid,
            "surface_min_max": surf_grid_max_min,
            "stream_velocity": xp.expand_dims(
                xp.array(STREAM_VELOCITY, dtype=xp.float32), -1
            ),
            "air_density": xp.expand_dims(xp.array(AIR_DENSITY, dtype=xp.float32), -1),
        }

        return (
            return_dict,
            s_min,
            s_max,
            mesh_indices_flattened,
            stl_vertices,
            center_of_mass,
        )

    @profile
    def preprocess_surface(self, data_dict, core_dict, center_of_mass, s_min, s_max):

        nx, ny, nz = self.grid_resolution

        return_dict = {}
        surface_coordinates = data_dict["surface_mesh_centers"]
        surface_normals = data_dict["surface_normals"]
        surface_sizes = data_dict["surface_areas"]
        surface_fields = data_dict["surface_fields"]

        xp = self.array_provider

        if not self.compute_scaling_factors:

            c_max = xp.float32(self.bounding_box_dims[0])
            c_min = xp.float32(self.bounding_box_dims[1])

            ids_in_bbox = xp.where(
                (surface_coordinates[:, 0] > c_min[0])
                & (surface_coordinates[:, 0] < c_max[0])
                & (surface_coordinates[:, 1] > c_min[1])
                & (surface_coordinates[:, 1] < c_max[1])
                & (surface_coordinates[:, 2] > c_min[2])
                & (surface_coordinates[:, 2] < c_max[2])
            )
            surface_coordinates = surface_coordinates[ids_in_bbox]
            surface_normals = surface_normals[ids_in_bbox]
            surface_sizes = surface_sizes[ids_in_bbox]
            surface_fields = surface_fields[ids_in_bbox]

            # Up to here, surface_coordinates is the same.
            # get a seed for random sampling to check numerical exactness:
            seed = int(time.time())

            if self.positional_encoding:
                dx, dy, dz = (
                    (s_max[0] - s_min[0]) / nx,
                    (s_max[1] - s_min[1]) / ny,
                    (s_max[2] - s_min[2]) / nz,
                )
                pos_normals_com_surface_fast = calculate_normal_positional_encoding(
                    surface_coordinates, center_of_mass, cell_length=[dx, dy, dz]
                )
            else:
                pos_normals_com_surface_fast = surface_coordinates - xp.asarray(
                    center_of_mass
                )

            if self.normalize_coordinates:
                s_max = xp.asarray(s_max)
                s_min = xp.asarray(s_min)
                core_dict["surf_grid"] = normalize(
                    cp.asarray(core_dict["surf_grid"]), s_max, s_min
                )
            else:
                surface_coordinates = surface_coordinates.copy()

            # Fit the kNN on ALL points:
            self.knn.fit(surface_coordinates)

            if self.sampling:
                # Perform the sampling:
                (
                    surface_coordinates_sampled_fast,
                    idx_surface_fast,
                ) = area_weighted_shuffle_array(
                    surface_coordinates, self.surface_points, surface_sizes, seed=seed
                )

                # Select out the sampled points for non-neighbor arrays:
                surface_fields_fast = surface_fields[idx_surface_fast]
                pos_normals_com_surface_fast = pos_normals_com_surface_fast[
                    idx_surface_fast
                ]

                # Now, perform the kNN on the sampled points:
                ii = self.knn.kneighbors(
                    surface_coordinates_sampled_fast, return_distance=False
                )

                # Pull out the neighbor elements.  Note that ii is the index into the original
                # points but only for the sampled points
                surface_neighbors_fast = surface_coordinates[ii][:, 1:]
                surface_neighbors_normals_fast = surface_normals[ii][:, 1:]
                surface_neighbors_sizes_fast = surface_sizes[ii][:, 1:]

                # Index into the normals and sizes AFTER getting the neighbors:
                surface_normals_fast = surface_normals[idx_surface_fast]
                surface_sizes_fast = surface_sizes[idx_surface_fast]

                # Update the coordinates to the sampled points:
                surface_coordinates_fast = surface_coordinates_sampled_fast

            else:
                # We are *not* sampling, kNN on ALL points:
                ii = self.knn.kneighbors(surface_coordinates, return_distance=False)

            # Have to normalize neighbors after the kNN and sampling
            if self.normalize_coordinates:
                surface_coordinates_fast = normalize(
                    surface_coordinates_fast, s_max, s_min
                )
                surface_neighbors_fast = normalize(surface_neighbors_fast, s_max, s_min)

            pos_normals_com_surface = pos_normals_com_surface_fast
            surface_coordinates = surface_coordinates_fast
            surface_neighbors = surface_neighbors_fast
            surface_normals = surface_normals_fast
            surface_neighbors_normals = surface_neighbors_normals_fast
            surface_sizes = surface_sizes_fast
            surface_neighbors_sizes = surface_neighbors_sizes_fast
            surface_fields = surface_fields_fast

            if self.scaling_type is not None:
                if self.surface_factors is not None:
                    if self.scaling_type == "mean_std_scaling":
                        surf_mean = self.surface_factors[0]
                        surf_std = self.surface_factors[1]
                        surface_fields = standardize(
                            surface_fields, cp.asarray(surf_mean), cp.asarray(surf_std)
                        )
                    elif self.scaling_type == "min_max_scaling":
                        surf_min = self.surface_factors[1]
                        surf_max = self.surface_factors[0]
                        surface_fields = normalize(
                            surface_fields, cp.asarray(surf_max), cp.asarray(surf_min)
                        )

        else:
            surface_sizes = None
            surface_normals = None
            surface_neighbors = None
            surface_neighbors_normals = None
            surface_neighbors_sizes = None
            pos_normals_com_surface = None

        return_dict.update(
            {
                "pos_surface_center_of_mass": pos_normals_com_surface,
                "surface_mesh_centers": surface_coordinates,
                "surface_mesh_neighbors": surface_neighbors,
                "surface_normals": surface_normals,
                "surface_neighbors_normals": surface_neighbors_normals,
                "surface_areas": surface_sizes,
                "surface_neighbors_areas": surface_neighbors_sizes,
                "surface_fields": surface_fields,
            }
        )

        return return_dict

    @profile
    def preprocess_volume(
        self,
        data_dict,
        core_dict,
        s_min,
        s_max,
        mesh_indices_flattened,
        stl_vertices,
        center_of_mass,
    ):

        return_dict = {}

        nx, ny, nz = self.grid_resolution

        xp = cp

        # Temporary: convert to cupy here:
        volume_coordinates = cp.asarray(data_dict["volume_mesh_centers"])
        volume_fields = cp.asarray(data_dict["volume_fields"])
        center_of_mass = cp.asarray(center_of_mass)

        if not self.compute_scaling_factors:
            if self.bounding_box_dims is None:
                c_max = s_max + (s_max - s_min) / 2
                c_min = s_min - (s_max - s_min) / 2
                c_min[2] = s_min[2]
            else:
                c_max = xp.asarray(self.bounding_box_dims[0])
                c_min = xp.asarray(self.bounding_box_dims[1])

            ids_in_bbox = self.array_provider.where(
                (volume_coordinates[:, 0] > c_min[0])
                & (volume_coordinates[:, 0] < c_max[0])
                & (volume_coordinates[:, 1] > c_min[1])
                & (volume_coordinates[:, 1] < c_max[1])
                & (volume_coordinates[:, 2] > c_min[2])
                & (volume_coordinates[:, 2] < c_max[2])
            )

            if self.sample_in_bbox:
                volume_coordinates = volume_coordinates[ids_in_bbox]
                volume_fields = volume_fields[ids_in_bbox]

            dx, dy, dz = (
                (c_max[0] - c_min[0]) / nx,
                (c_max[1] - c_min[1]) / ny,
                (c_max[2] - c_min[2]) / nz,
            )

            # Generate a grid of specified resolution to map the bounding box
            # The grid is used for capturing structured geometry features and SDF representation of geometry
            grid = create_grid(c_max, c_min, [nx, ny, nz])
            grid_reshaped = grid.reshape(nx * ny * nz, 3)

            # SDF calculation on the grid using WARP
            sdf_grid = (
                signed_distance_field(
                    stl_vertices,
                    mesh_indices_flattened,
                    grid_reshaped,
                    use_sign_winding_number=True,
                )
                # .numpy()
                .reshape(nx, ny, nz)
            )

            if self.sampling:
                volume_coordinates_sampled, idx_volume = shuffle_array(
                    volume_coordinates, self.volume_points
                )
                if volume_coordinates_sampled.shape[0] < self.volume_points:
                    volume_coordinates_sampled = pad(
                        volume_coordinates_sampled,
                        self.volume_points,
                        pad_value=-10.0,
                    )
                volume_fields = volume_fields[idx_volume]
                volume_coordinates = volume_coordinates_sampled

            sdf_nodes, sdf_node_closest_point = signed_distance_field(
                stl_vertices,
                mesh_indices_flattened,
                volume_coordinates,
                include_hit_points=True,
                use_sign_winding_number=True,
            )
            sdf_nodes = xp.asarray(sdf_nodes)
            sdf_node_closest_point = xp.asarray(sdf_node_closest_point)

            sdf_nodes = sdf_nodes.reshape((-1, 1))
            sdf_node_closest_point = sdf_node_closest_point

            if self.positional_encoding:
                pos_normals_closest_vol = calculate_normal_positional_encoding(
                    volume_coordinates,
                    sdf_node_closest_point,
                    cell_length=[dx, dy, dz],
                )
                pos_normals_com_vol = calculate_normal_positional_encoding(
                    volume_coordinates, center_of_mass, cell_length=[dx, dy, dz]
                )
            else:
                pos_normals_closest_vol = volume_coordinates - sdf_node_closest_point
                pos_normals_com_vol = volume_coordinates - center_of_mass

            if self.normalize_coordinates:

                volume_coordinates = normalize(volume_coordinates, c_max, c_min)
                grid = normalize(grid, c_max, c_min)

            if self.scaling_type is not None:
                if self.volume_factors is not None:
                    if self.scaling_type == "mean_std_scaling":
                        vol_mean = self.volume_factors[0]
                        vol_std = self.volume_factors[1]
                        volume_fields = standardize(volume_fields, vol_mean, vol_std)
                    elif self.scaling_type == "min_max_scaling":
                        vol_min = xp.asarray(self.volume_factors[1])
                        vol_max = xp.asarray(self.volume_factors[0])
                        volume_fields = normalize(volume_fields, vol_max, vol_min)

            # volume_fields = np.float32(volume_fields)
            # pos_normals_closest_vol = np.float32(pos_normals_closest_vol)
            # pos_normals_com_vol = np.float32(pos_normals_com_vol)
            # volume_coordinates = np.float32(volume_coordinates)
            # sdf_nodes = np.float32(sdf_nodes)
            # sdf_grid = np.float32(sdf_grid)
            # grid = np.float32(grid)
            # vol_grid_max_min = np.float32(np.asarray([c_min, c_max]))
            vol_grid_max_min = xp.stack([c_min, c_max])

        else:
            pos_normals_closest_vol = None
            pos_normals_com_vol = None
            sdf_nodes = None
            sdf_grid = None
            grid = None
            vol_grid_max_min = None

        return_dict.update(
            {
                "pos_volume_closest": pos_normals_closest_vol,
                "pos_volume_center_of_mass": pos_normals_com_vol,
                "grid": grid,
                "sdf_grid": sdf_grid,
                "sdf_nodes": sdf_nodes,
                "volume_fields": volume_fields,
                "volume_mesh_centers": volume_coordinates,
                "volume_min_max": vol_grid_max_min,
            }
        )

        return return_dict

    @profile
    def preprocess_data(self, data_dict):

        (
            return_dict,
            s_min,
            s_max,
            mesh_indices_flattened,
            stl_vertices,
            center_of_mass,
        ) = self.preprocess_combined(data_dict)

        if self.model_type == "volume" or self.model_type == "combined":
            volume_dict = self.preprocess_volume(
                data_dict,
                return_dict,
                s_min,
                s_max,
                mesh_indices_flattened,
                stl_vertices,
                center_of_mass,
            )

            return_dict.update(volume_dict)

        if self.model_type == "surface" or self.model_type == "combined":
            surface_dict = self.preprocess_surface(
                data_dict, return_dict, center_of_mass, s_min, s_max
            )
            return_dict.update(surface_dict)

        if self.sampling:
            geometry_points = self.geom_points_sample
            geometry_coordinates_sampled, idx_geometry = shuffle_array(
                stl_vertices, geometry_points
            )
            if geometry_coordinates_sampled.shape[0] < geometry_points:
                geometry_coordinates_sampled = pad(
                    geometry_coordinates_sampled, geometry_points, pad_value=-100.0
                )
            geom_centers = geometry_coordinates_sampled
        else:
            geom_centers = stl_vertices

        # geom_centers = self.array_provider.float32(geom_centers)

        return_dict["geometry_coordinates"] = geom_centers

        return return_dict

    @profile
    def __getitem__(self, idx):
        index = self.indices[idx]
        cfd_filename = self.filenames[index]

        # Get all of the data with the threaded data read
        filepath = self.data_path / cfd_filename

        # Check if data was preloaded
        preloaded = False
        if index in self.preloaded_data and self.preloaded_data[index]:
            preloaded = True
            data_or_futures = self.preloaded_data[index]

            # Check if we have futures
            if isinstance(data_or_futures, tuple) and len(data_or_futures) == 2:
                futures, executor = data_or_futures

                # Wait for all futures to complete and gather results
                data_dict = {}
                for key, future in futures.items():
                    if key == "_optional_keys":
                        # Special handling for optional keys
                        optional_results = future.result()
                        data_dict.update(optional_results)
                    else:
                        k, v = future.result()
                        data_dict[k] = v

                # Shutdown the executor
                executor.shutdown()
            else:
                # Already resolved data
                data_dict = data_or_futures
        else:
            # Use return_futures=False for direct loading
            data_dict = self.threaded_data_read(filepath)

        return_dict = self.preprocess_data(data_dict)

        # Before returning, unload the preloaded data if its present:
        if preloaded:
            self.preloaded_data.pop(index)

        if self.gpu_output:
            # move all this to cupy:
            for key, value in return_dict.items():
                if isinstance(value, np.ndarray):
                    return_dict[key] = cp.asarray(value)

        return return_dict


if __name__ == "__main__":
    fm_data = DoMINODataPipe(
        data_path="/code/processed_data/new_models_1/",
        phase="train",
        sampling=False,
        sample_in_bbox=False,
    )
