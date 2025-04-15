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
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Protocol, Sequence, Union

import cuml
import cupy as cp
import numpy as np
import torch
import torch.cuda.nvtx as nvtx
from omegaconf import DictConfig
from scipy.spatial import KDTree
from torch import Tensor
from torch.utils.data import Dataset, default_collate

from physicsnemo.utils.domino.utils import (
    ArrayType,
    area_weighted_shuffle_array,
    calculate_center_of_mass,
    calculate_normal_positional_encoding,
    create_grid,
    get_filenames,
    mean_std_sampling,
    normalize,
    pad,
    # sample_array,
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


class BoundingBox(Protocol):
    """
    Type definition for the required format of bounding box dimensions.
    """

    min: ArrayType
    max: ArrayType


@dataclass
class DoMINOSurfaceData:
    """Surface data configuration for DoMINO dataset.

    Attributes:
        variables: Names of surface variables.
        points_sample: Number of surface points to sample per batch.
        num_neighbors: Number of surface neighbors to consider for nearest neighbors approach.
        resample: Whether to resample the surface before kdtree/knn. Not available if caching.
        resampling_points: Number of points to resample the surface to.
        sampling_algorithm: Algorithm to use for surface sampling ("area_weighted" or "random").
        factors: Non-dimensionalization factors for surface variables.
            If set, and scaling_type is:
            - min_max_scaling -> rescale surface_fields to the min/max set here
            - mean_std_scaling -> rescale surface_fields to the mean and std set here.
        bounding_box_dims: Dimensions of bounding box. Must be an object with min/max
            attributes that are arraylike.
    """

    variables: Optional[Sequence] = ("pMean", "wallShearStress")
    points_sample: int = 1024
    num_neighbors: int = 11
    resample: bool = False
    resampling_points: int = 1_000_000
    sampling_algorithm: str = Literal["area_weighted", "random"]
    factors: Optional[Sequence] = None
    bounding_box_dims: Optional[BoundingBox] = None


@dataclass
class DoMINOVolumeData:
    """Volume data configuration for DoMINO dataset.

    Attributes:
        variables: Names of volume variables.
        points_sample: Number of volume points to sample per batch.
        factors: Non-dimensionalization factors for volume variables scaling.
            If set, and scaling_type is:
            - min_max_scaling -> rescale volume_fields to the min/max set here
            - mean_std_scaling -> rescale volume_fields to the mean and std set here.
        bounding_box_dims: Dimensions of bounding box. Must be an object with min/max
            attributes that are arraylike.
    """

    variables: Optional[Sequence] = ("UMean", "pMean")
    points_sample: int = 1024
    factors: Optional[Sequence] = None
    bounding_box_dims: Optional[BoundingBox] = None


@dataclass
class DoMINODataConfig:
    """Configuration for DoMINO dataset processing pipeline.

    Attributes:
        data_path: Path to the dataset to load.
        phase: Which phase of data to load ("train", "val", or "test").
        surface_variables: (Surface specific) Names of surface variables.
        surface_points_sample: (Surface specific) Number of surface points to sample per batch.
        num__surface_neighbors: (Surface specific) Number of surface neighbors to consider for nearest neighbors approach.
        resample_surfaces: (Surface specific) Whether to resample the surface before kdtree/knn. Not available if caching.
        resampling_points: (Surface specific) Number of points to resample the surface to.
        surface_sampling_algorithm: (Surface specific) Algorithm to use for surface sampling ("area_weighted" or "random").
        surface_factors: (Surface specific) Non-dimensionalization factors for surface variables.
            If set, and scaling_type is:
            - min_max_scaling -> rescale surface_fields to the min/max set here
            - mean_std_scaling -> rescale surface_fields to the mean and std set here.
        bounding_box_dims_surf: (Surface specific) Dimensions of bounding box. Must be an object with min/max
            attributes that are arraylike.
        volume_variables: (Volume specific) Names of volume variables.
        volume_points_sample: (Volume specific) Number of volume points to sample per batch.
        volume_factors: (Volume specific) Non-dimensionalization factors for volume variables scaling.
            If set, and scaling_type is:
            - min_max_scaling -> rescale volume_fields to the min/max set here
            - mean_std_scaling -> rescale volume_fields to the mean and std set here.
        bounding_box_dims: (Volume specific) Dimensions of bounding box. Must be an object with min/max
            attributes that are arraylike.
        grid_resolution: Resolution of the latent grid.
        normalize_coordinates: Whether to normalize coordinates based on min/max values.
            For surfaces: uses s_min/s_max, defined from:
            - Surface bounding box, if defined.
            - Min/max of the stl_vertices
            For volumes: uses c_min/c_max, defined from:
            - Volume bounding_box if defined,
            - 1.5x s_min/max otherwise, except c_min[2] = s_min[2] in this case
        sample_in_bbox: Whether to sample points in a specified bounding box.
            Uses the same min/max points as coordinate normalization.
            Only performed if compute_scaling_factors is false.
        sampling: Whether to downsample the full resolution mesh to fit in GPU memory.
            Surface and volume sampling points are configured separately as:
            - surface.points_sample
            - volume.points_sample
        geom_points_sample: Number of STL points sampled per batch.
            Independent of volume.points_sample and surface.points_sample.
        positional_encoding: Whether to use positional encoding. Affects the calculation of:
            - pos_volume_closest
            - pos_volume_center_of_mass
            - pos_surface_centter_of_mass
        scaling_type: Scaling type for volume variables.
            If used, will rescale the volume_fields and surface fields outputs.
            Requires volume.factor and surface.factor to be set.
        compute_scaling_factors: Whether to compute scaling factors.
            Not available if caching.
            Many preprocessing pieces are disabled if computing scaling factors.
        caching: Whether this is for caching or serving.
        deterministic: Whether to use a deterministic seed for sampling and random numbers.
        gpu_preprocessing: Whether to do preprocessing on the GPU (False for CPU).
        gpu_output: Whether to return output on the GPU as cupy arrays.
            If False, returns numpy arrays.
            You might choose gpu_preprocessing=True and gpu_output=False if caching.
    """

    data_path: Path
    phase: Literal["train", "val", "test"]

    # Surface-specific variables:
    surface_variables: Optional[Sequence] = ("pMean", "wallShearStress")
    surface_points_sample: int = 1024
    num_surface_neighbors: int = 11
    resample_surfaces: bool = False
    resampling_points: int = 1_000_000
    surface_sampling_algorithm: str = Literal["area_weighted", "random"]
    surface_factors: Optional[Sequence] = None
    bounding_box_dims_surf: Optional[BoundingBox] = None

    # Volume specific variables:
    volume_variables: Optional[Sequence] = ("UMean", "pMean")
    volume_points_sample: int = 1024
    volume_factors: Optional[Sequence] = None
    bounding_box_dims: Optional[BoundingBox] = None

    grid_resolution: Union[Sequence, ArrayType] = (256, 96, 64)
    normalize_coordinates: bool = False
    sample_in_bbox: bool = False
    sampling: bool = False
    geom_points_sample: int = 300000
    positional_encoding: bool = False
    scaling_type: Optional[Literal["min_max_scaling", "mean_std_scaling"]] = None
    compute_scaling_factors: bool = False
    caching: bool = False
    deterministic: bool = False
    gpu_preprocessing: bool = True
    gpu_output: bool = True

    def __post_init__(self):
        # Ensure data_path is a Path object:
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)
        self.data_path = self.data_path.expanduser()

        if not self.data_path.exists():
            raise ValueError(f"Path {self.data_path} does not exist")

        if not self.data_path.is_dir():
            raise ValueError(f"Path {self.data_path} is not a directory")

        # Object if caching settings are impossible:
        if self.caching:
            if self.sampling:
                raise ValueError("Sampling should be False for caching")
            if self.compute_scaling_factors:
                raise ValueError("Compute scaling factors should be False for caching")
            if self.resample_surfaces:
                raise ValueError("Resample surface should be False for caching")

        if self.phase not in [
            "train",
            "val",
            "test",
        ]:
            raise ValueError(
                f"phase should be one of ['train', 'val', 'test'], got {self.phase}"
            )
        if self.scaling_type is not None:
            if self.scaling_type not in [
                "min_max_scaling",
                "mean_std_scaling",
            ]:
                raise ValueError(
                    f"scaling_type should be one of ['min_max_scaling', 'mean_std_scaling'], got {self.scaling_type}"
                )


##### TODO
# - put model type in config or leave in __init__
# - check the bounding box protocol works
# - keep the target device in the constructor, not the config.
# - Make sure GPU output always triggers
# - Make sure un-optimized numpy files can load too
# - make sure bounding box is set right in the config
# - Make sure random seeds in utils are set correctly (mostly, not there but here)


class DoMINODataPipe(Dataset):
    """
    Datapipe for DoMINO

    """

    def __init__(
        self,
        input_path,
        model_type: Literal["surface", "volume", "combined"],
        device: int = 0,
        **data_config_overrides,
        # data_path: Union[str, Path],  # Input data path
        # phase: Literal["train", "val", "test"] = "train",  # Train, test or val
        # surface_variables: Optional[Sequence] = (
        #     "pMean",
        #     "wallShearStress",
        # ),  # Names of surface variables
        # volume_variables: Optional[Sequence] = (
        #     "UMean",
        #     "pMean",
        # ),  # Names of volume variables
        # sampling: bool = False,  # Sampling True or False
        # device: int = 0,  # GPU device id
        # grid_resolution: Optional[Sequence] = (
        #     256,
        #     96,
        #     64,
        # ),  # Resolution of latent grid
        # normalize_coordinates: bool = False,  # Normalize coordinates?
        # sample_in_bbox: bool = False,  # Sample points in a specified bounding box
        # volume_points_sample: int = 1024,  # Number of volume points sampled per batch
        # surface_points_sample: int = 1024,  # Number of surface points sampled per batch
        # geom_points_sample: int = 300000,  # Number of STL points sampled per batch
        # positional_encoding: bool = False,  # Positional encoding, True or False
        # volume_factors=None,  # Non-dimensionalization factors for volume variables
        # surface_factors=None,  # Non-dimensionalization factors for surface variables
        # scaling_type=None,  # Scaling min_max or mean_std
        # model_type=None,  # Model_type, surface, volume or combined
        # bounding_box_dims=None,  # Dimensions of bounding box
        # bounding_box_dims_surf=None,  # Dimensions of bounding box
        # compute_scaling_factors=False, # Are you computing scaling factors?
        # num_surface_neighbors=11,  # Surface neighbors to consider
        # gpu_preprocessing=True,
        # gpu_output=True,
        # resample_surfaces=False, # resample surfaces before kdtree
        # resampling_points=1_000_000, # number of points to resample,
        # surface_sampling_algorithm="area_weighted", # area_weighted or random
        # for_caching=False,
        # deterministic_seed=False,
    ):
        # Perform config packaging and validation
        self.config = DoMINODataConfig(data_path=input_path, **data_config_overrides)

        if self.config.deterministic:
            np.random.seed(42)
            cp.random.seed(42)
        else:
            np.random.seed(seed=int(time.time()))
            cp.random.seed(seed=int(time.time()))

        # self.sampling = sampling
        # self.grid_resolution = grid_resolution
        # self.normalize_coordinates = normalize_coordinates
        self.model_type = model_type

        self.filenames = get_filenames(self.config.data_path, exclude_dirs=True)
        total_files = len(self.filenames)

        self.indices = np.array(range(total_files))

        # Why shuffle the indices here if only using random access below?

        np.random.shuffle(self.indices)
        # self.surface_variables = surface_variables
        # self.volume_variables = volume_variables
        # self.volume_points = volume_points_sample
        # self.surface_points = surface_points_sample
        # self.geom_points_sample = geom_points_sample
        # self.sample_in_bbox = sample_in_bbox
        # self.device = device
        # self.positional_encoding = positional_encoding
        # self.volume_factors = volume_factors
        # self.surface_factors = surface_factors
        # self.scaling_type = scaling_type
        # self.compute_scaling_factors = compute_scaling_factors
        # self.num_surface_neighbors = num_surface_neighbors
        # self.gpu_preprocessing = gpu_preprocessing
        # self.gpu_output = gpu_output

        # self.resample_surfaces = resample_surfaces
        # self.resampling_points = resampling_points
        # self.surface_sampling_algorithm = surface_sampling_algorithm
        # self.deterministic_seed = deterministic_seed

        # Determine the array provider based on what device
        # will do preprocessing:
        self.array_provider = cp if self.config.gpu_preprocessing else np
        # Update the arrays for bounding boxes:

        self.config.bounding_box_dims = [
            self.array_provider.asarray(self.config.bounding_box_dims.max).astype(
                "float32"
            ),
            self.array_provider.asarray(self.config.bounding_box_dims.min).astype(
                "float32"
            ),
        ]
        self.config.bounding_box_dims_surf = [
            self.array_provider.asarray(self.config.bounding_box_dims_surf.max).astype(
                "float32"
            ),
            self.array_provider.asarray(self.config.bounding_box_dims_surf.min).astype(
                "float32"
            ),
        ]

        # Initialize a kNN tool based on the preprocessing device:
        if self.config.gpu_preprocessing:
            self.knn = cuml.neighbors.NearestNeighbors(
                n_neighbors=self.config.num_surface_neighbors, algorithm="rbc"
            )
        else:
            self.knn = KDTree

        # Add dictionary to store preloaded data
        self.preloaded_data = defaultdict(dict)

        # Used if threaded data is enabled:
        self.max_workers = 8

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
                # This toggles between cupy and numpy depending on the value of self.config.gpu_preprocessing
                return key, self.array_provider.asarray(data[key])

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
            index: Target index to preload.
            max_workers: Maximum number of worker threads. If None, uses default.
        """

        if max_workers is not None:
            self.max_workers = max_workers

        # Get the target file name:
        actual_index = self.indices[index]
        filepath = self.data_path / self.filenames[actual_index]

        # Load the data with futures
        self.preloaded_data[index] = self.threaded_data_read(
            filepath,
            max_workers=self.max_workers,
            return_futures=True,  # Return futures instead of waiting
        )

    @profile
    def preprocess_combined(self, data_dict):

        # Pull these out and force to fp32:
        STREAM_VELOCITY = data_dict["stream_velocity"].astype(
            self.array_provider.float32
        )
        AIR_DENSITY = data_dict["air_density"].astype(self.array_provider.float32)

        # Pull these pieces out of the data_dict for manipulation
        stl_vertices = data_dict["stl_coordinates"]
        stl_centers = data_dict["stl_centers"]
        mesh_indices_flattened = data_dict["stl_faces"]
        stl_sizes = data_dict["stl_areas"]

        xp = self.array_provider

        length_scale = xp.amax(xp.amax(stl_vertices, 0) - xp.amin(stl_vertices, 0))

        # Center of mass calculation
        center_of_mass = calculate_center_of_mass(stl_centers, stl_sizes)

        if self.config.bounding_box_dims_surf is None:
            s_max = xp.amax(stl_vertices, 0)
            s_min = xp.amin(stl_vertices, 0)
        else:
            s_max = xp.asarray(self.config.bounding_box_dims_surf[0])
            s_min = xp.asarray(self.config.bounding_box_dims_surf[1])

        # SDF calculation on the grid using WARP
        if not self.config.compute_scaling_factors:

            nx, ny, nz = self.config.grid_resolution
            surf_grid = create_grid(s_max, s_min, [nx, ny, nz])
            surf_grid_reshaped = surf_grid.reshape(nx * ny * nz, 3)

            sdf_surf_grid = signed_distance_field(
                stl_vertices,
                mesh_indices_flattened,
                surf_grid_reshaped,
                use_sign_winding_number=True,
            ).reshape(nx, ny, nz)

        else:
            surf_grid = None
            sdf_surf_grid = None

        if self.config.sampling:
            # nvtx.range_push("Geometry Sampling")
            geometry_points = self.config.geom_points_sample
            geometry_coordinates_sampled, idx_geometry = shuffle_array(
                stl_centers, geometry_points
            )
            if geometry_coordinates_sampled.shape[0] < geometry_points:
                geometry_coordinates_sampled = pad(
                    geometry_coordinates_sampled, geometry_points, pad_value=-100.0
                )
            geom_centers = geometry_coordinates_sampled
            # nvtx.range_pop()
        else:
            geom_centers = stl_centers

        # geom_centers = self.array_provider.float32(geom_centers)

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
            "geometry_coordinates": geom_centers,
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

        nx, ny, nz = self.config.grid_resolution

        return_dict = {}
        surface_coordinates = data_dict["surface_mesh_centers"]
        surface_normals = data_dict["surface_normals"]
        surface_sizes = data_dict["surface_areas"]
        surface_fields = data_dict["surface_fields"]

        xp = self.array_provider

        if self.config.resample_surfaces:
            if self.config.resampling_points > surface_coordinates.shape[0]:
                resampling_points = surface_coordinates.shape[0]
            else:
                resampling_points = self.config.resampling_points

            surface_coordinates, idx_s = shuffle_array(
                surface_coordinates, resampling_points
            )
            surface_normals = surface_normals[idx_s]
            surface_sizes = surface_sizes[idx_s]
            surface_fields = surface_fields[idx_s]

        if not self.config.compute_scaling_factors:

            c_max = self.config.bounding_box_dims[0]
            c_min = self.config.bounding_box_dims[1]

            if self.config.sample_in_bbox:
                # TODO - clean this up with vectorization?
                # TODO - the xp.where is likely a useless op.  Need to check.
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

            # Compute the positional encoding before sampling
            if self.config.positional_encoding:
                dx, dy, dz = (
                    (s_max[0] - s_min[0]) / nx,
                    (s_max[1] - s_min[1]) / ny,
                    (s_max[2] - s_min[2]) / nz,
                )
                pos_normals_com_surface = calculate_normal_positional_encoding(
                    surface_coordinates, center_of_mass, cell_length=[dx, dy, dz]
                )
            else:
                pos_normals_com_surface = surface_coordinates - xp.asarray(
                    center_of_mass
                )

            # Fit the kNN (or KDTree, if CPU) on ALL points:
            if self.array_provider == cp:
                self.knn.fit(surface_coordinates)
            else:
                # Under the hood this is instantiating a KDTree.
                # aka here knn is a type, not a class, technically.
                interp_func = self.knn(surface_coordinates)

            if self.config.sampling:
                # Perform the down sampling:
                if self.config.surface_sampling_algorithm == "area_weighted":
                    (
                        surface_coordinates_sampled,
                        idx_surface,
                    ) = area_weighted_shuffle_array(
                        surface_coordinates,
                        self.config.surface_points_sample,
                        surface_sizes,
                    )
                else:
                    surface_coordinates_sampled, idx_surface = shuffle_array(
                        surface_coordinates, self.config.surface_points_sample
                    )

                if (
                    surface_coordinates_sampled.shape[0]
                    < self.config.surface_points_sample
                ):
                    surface_coordinates_sampled = pad(
                        surface_coordinates_sampled,
                        self.config.surface_points_sample,
                        pad_value=-10.0,
                    )

                # Select out the sampled points for non-neighbor arrays:
                surface_fields = surface_fields[idx_surface]
                pos_normals_com_surface = pos_normals_com_surface[idx_surface]

                # Now, perform the kNN on the sampled points:
                if self.array_provider == cp:
                    ii = self.knn.kneighbors(
                        surface_coordinates_sampled, return_distance=False
                    )
                else:
                    _, ii = interp_func.query(
                        surface_coordinates_sampled, k=self.config.num_surface_neighbors
                    )

                # Pull out the neighbor elements.  Note that ii is the index into the original
                # points - but only exists for the sampled points
                # In other words, a point from `surface_coordinates_sampled` has neighbors
                # from the full `surface_coordinates` array.
                surface_neighbors = surface_coordinates[ii][:, 1:]
                surface_neighbors_normals = surface_normals[ii][:, 1:]
                surface_neighbors_sizes = surface_sizes[ii][:, 1:]

                # We could index into these above the knn step too; they aren't dependent on that.
                surface_normals = surface_normals[idx_surface]
                surface_sizes = surface_sizes[idx_surface]

                # Update the coordinates to the sampled points:
                surface_coordinates = surface_coordinates_sampled

            else:
                # We are *not* sampling, kNN on ALL points:
                ii = self.knn.kneighbors(surface_coordinates, return_distance=False)

                # Construct the neighbors arrays:
                surface_neighbors = surface_coordinates[ii][:, 1:]
                surface_neighbors_normals = surface_normals[ii][:, 1:]
                surface_neighbors_sizes = surface_sizes[ii][:, 1:]

            # Have to normalize neighbors after the kNN and sampling
            if self.config.normalize_coordinates:
                core_dict["surf_grid"] = normalize(
                    cp.asarray(core_dict["surf_grid"]), s_max, s_min
                )
                surface_coordinates = normalize(surface_coordinates, s_max, s_min)
                surface_neighbors = normalize(surface_neighbors, s_max, s_min)

            if self.config.scaling_type is not None:
                if self.config.surface_factors is not None:
                    if self.config.scaling_type == "mean_std_scaling":
                        surf_mean = self.config.surface_factors[0]
                        surf_std = self.config.surface_factors[1]
                        # TODO - Are these array calls needed?
                        surface_fields = standardize(
                            surface_fields, cp.asarray(surf_mean), cp.asarray(surf_std)
                        )
                    elif self.config.scaling_type == "min_max_scaling":
                        surf_min = self.config.surface_factors[1]
                        surf_max = self.config.surface_factors[0]
                        # TODO - Are these array calls needed?
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

        nx, ny, nz = self.config.grid_resolution

        xp = self.array_provider

        # # Temporary: convert to cupy here:
        volume_coordinates = data_dict["volume_mesh_centers"]
        volume_fields = data_dict["volume_fields"]

        if not self.config.compute_scaling_factors:
            if self.config.bounding_box_dims is None:
                c_max = s_max + (s_max - s_min) / 2
                c_min = s_min - (s_max - s_min) / 2
                c_min[2] = s_min[2]
            else:
                c_max = xp.asarray(self.config.bounding_box_dims[0])
                c_min = xp.asarray(self.config.bounding_box_dims[1])

            if self.config.sample_in_bbox:
                # TODO - xp.where can probably be removed.
                ids_in_bbox = self.array_provider.where(
                    (volume_coordinates[:, 0] > c_min[0])
                    & (volume_coordinates[:, 0] < c_max[0])
                    & (volume_coordinates[:, 1] > c_min[1])
                    & (volume_coordinates[:, 1] < c_max[1])
                    & (volume_coordinates[:, 2] > c_min[2])
                    & (volume_coordinates[:, 2] < c_max[2])
                )
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
            sdf_grid = signed_distance_field(
                stl_vertices,
                mesh_indices_flattened,
                grid_reshaped,
                use_sign_winding_number=True,
            ).reshape(nx, ny, nz)

            if self.config.sampling:
                volume_coordinates_sampled, idx_volume = shuffle_array(
                    volume_coordinates, self.config.volume_points_sample
                )
                if (
                    volume_coordinates_sampled.shape[0]
                    < self.config.volume_points_sample
                ):
                    volume_coordinates_sampled = pad(
                        volume_coordinates_sampled,
                        self.config.volume_points_sample,
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
            # TODO - is this needed?
            sdf_nodes = xp.asarray(sdf_nodes)
            sdf_node_closest_point = xp.asarray(sdf_node_closest_point)

            sdf_nodes = sdf_nodes.reshape((-1, 1))

            if self.config.positional_encoding:
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

            if self.config.normalize_coordinates:

                volume_coordinates = normalize(volume_coordinates, c_max, c_min)
                grid = normalize(grid, c_max, c_min)

            if self.config.scaling_type is not None:
                if self.config.volume_factors is not None:
                    if self.config.scaling_type == "mean_std_scaling":
                        vol_mean = self.config.volume_factors[0]
                        vol_std = self.config.volume_factors[1]
                        volume_fields = standardize(volume_fields, vol_mean, vol_std)
                    elif self.config.scaling_type == "min_max_scaling":
                        vol_min = xp.asarray(self.config.volume_factors[1])
                        vol_max = xp.asarray(self.config.volume_factors[0])
                        volume_fields = normalize(volume_fields, vol_max, vol_min)

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

        # if self.config.sampling:
        #     # nvtx.range_push("Geometry Sampling")
        #     geometry_points = self.geom_points_sample
        #     geometry_coordinates_sampled, idx_geometry = shuffle_array(
        #         stl_centers, geometry_points
        #     )
        #     if geometry_coordinates_sampled.shape[0] < geometry_points:
        #         geometry_coordinates_sampled = pad(
        #             geometry_coordinates_sampled, geometry_points, pad_value=-100.0
        #         )
        #     geom_centers = geometry_coordinates_sampled
        #     # nvtx.range_pop()
        # else:
        #     geom_centers = stl_centers

        # # geom_centers = self.array_provider.float32(geom_centers)

        # return_dict["geometry_coordinates"] = geom_centers

        return return_dict

    @profile
    def __getitem__(self, idx):

        if self.config.deterministic:
            self.array_provider.seed(idx)

        index = self.indices[idx]
        cfd_filename = self.filenames[index]

        # Get all of the data with the threaded data read
        filepath = self.config.data_path / cfd_filename

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

        if self.config.gpu_output:
            # Make sure this is all on the GPU:
            for key, value in return_dict.items():
                if isinstance(value, np.ndarray):
                    return_dict[key] = cp.asarray(value)
        else:
            for key, value in return_dict.items():
                if isinstance(value, cp.ndarray):
                    return_dict[key] = value.get()

        return return_dict


def compute_scaling_factors(cfg: DictConfig, input_path: str, use_cache: bool) -> None:

    model_type = cfg.model.model_type
    if model_type == "volume" or model_type == "combined":
        vol_save_path = os.path.join(cfg.output, "volume_scaling_factors.npy")
        if not os.path.exists(vol_save_path):
            volume_variable_names = list(cfg.variables.volume.solution.keys())

            fm_dict = DoMINODataPipe(
                input_path,
                phase="train",
                grid_resolution=cfg.model.interp_res,
                volume_variables=volume_variable_names,
                surface_variables=None,
                normalize_coordinates=True,
                sampling=False,
                sample_in_bbox=True,
                volume_points_sample=cfg.model.volume_points_sample,
                geom_points_sample=cfg.model.geom_points_sample,
                positional_encoding=cfg.model.positional_encoding,
                model_type=cfg.model.model_type,
                bounding_box_dims=cfg.data.bounding_box,
                bounding_box_dims_surf=cfg.data.bounding_box_surface,
                compute_scaling_factors=True,
                gpu_preprocessing=True,
                gpu_output=False,
            )

            # Calculate mean
            if cfg.model.normalization == "mean_std_scaling":
                for j in range(len(fm_dict)):
                    d_dict = fm_dict[j]
                    vol_fields = d_dict["volume_fields"]

                    if vol_fields is not None:
                        if j == 0:
                            vol_fields_sum = np.mean(vol_fields, 0)
                        else:
                            vol_fields_sum += np.mean(vol_fields, 0)
                    else:
                        vol_fields_sum = 0.0

                vol_fields_mean = vol_fields_sum / len(fm_dict)

                for j in range(len(fm_dict)):
                    d_dict = fm_dict[j]
                    vol_fields = d_dict["volume_fields"]

                    if vol_fields is not None:
                        if j == 0:
                            vol_fields_sum_square = np.mean(
                                (vol_fields - vol_fields_mean) ** 2.0, 0
                            )
                        else:
                            vol_fields_sum_square += np.mean(
                                (vol_fields - vol_fields_mean) ** 2.0, 0
                            )
                    else:
                        vol_fields_sum_square = 0.0

                vol_fields_std = np.sqrt(vol_fields_sum_square / len(fm_dict))

                vol_scaling_factors = [vol_fields_mean, vol_fields_std]

            if cfg.model.normalization == "min_max_scaling":
                for j in range(len(fm_dict)):
                    d_dict = fm_dict[j]
                    vol_fields = d_dict["volume_fields"]

                    if vol_fields is not None:
                        vol_mean = np.mean(vol_fields, 0)
                        vol_std = np.std(vol_fields, 0)
                        vol_idx = mean_std_sampling(
                            vol_fields, vol_mean, vol_std, tolerance=12.0
                        )
                        vol_fields_sampled = np.delete(vol_fields, vol_idx, axis=0)
                        if j == 0:
                            vol_fields_max = np.amax(vol_fields_sampled, 0)
                            vol_fields_min = np.amin(vol_fields_sampled, 0)
                        else:
                            vol_fields_max1 = np.amax(vol_fields_sampled, 0)
                            vol_fields_min1 = np.amin(vol_fields_sampled, 0)

                            for k in range(vol_fields.shape[-1]):
                                if vol_fields_max1[k] > vol_fields_max[k]:
                                    vol_fields_max[k] = vol_fields_max1[k]

                                if vol_fields_min1[k] < vol_fields_min[k]:
                                    vol_fields_min[k] = vol_fields_min1[k]
                    else:
                        vol_fields_max = 0.0
                        vol_fields_min = 0.0

                    if j > 20:
                        break
                vol_scaling_factors = [vol_fields_max, vol_fields_min]
            np.save(vol_save_path, vol_scaling_factors)

    if model_type == "surface" or model_type == "combined":
        surf_save_path = os.path.join(cfg.output, "surface_scaling_factors.npy")

        if not os.path.exists(surf_save_path):

            volume_variable_names = list(cfg.variables.volume.solution.keys())
            surface_variable_names = list(cfg.variables.surface.solution.keys())

            fm_dict = DoMINODataPipe(
                input_path,
                phase="train",
                grid_resolution=cfg.model.interp_res,
                volume_variables=None,
                surface_variables=surface_variable_names,
                normalize_coordinates=True,
                sampling=False,
                sample_in_bbox=True,
                volume_points_sample=cfg.model.volume_points_sample,
                geom_points_sample=cfg.model.geom_points_sample,
                positional_encoding=cfg.model.positional_encoding,
                model_type=cfg.model.model_type,
                bounding_box_dims=cfg.data.bounding_box,
                bounding_box_dims_surf=cfg.data.bounding_box_surface,
                compute_scaling_factors=True,
            )

            # Calculate mean
            if cfg.model.normalization == "mean_std_scaling":
                for j in range(len(fm_dict)):
                    d_dict = fm_dict[j]
                    surf_fields = d_dict["surface_fields"]

                    if surf_fields is not None:
                        if j == 0:
                            surf_fields_sum = np.mean(surf_fields, 0)
                        else:
                            surf_fields_sum += np.mean(surf_fields, 0)
                    else:
                        surf_fields_sum = 0.0

                surf_fields_mean = surf_fields_sum / len(fm_dict)

                for j in range(len(fm_dict)):
                    d_dict = fm_dict[j]
                    surf_fields = d_dict["surface_fields"]

                    if surf_fields is not None:
                        if j == 0:
                            surf_fields_sum_square = np.mean(
                                (surf_fields - surf_fields_mean) ** 2.0, 0
                            )
                        else:
                            surf_fields_sum_square += np.mean(
                                (surf_fields - surf_fields_mean) ** 2.0, 0
                            )
                    else:
                        surf_fields_sum_square = 0.0

                surf_fields_std = np.sqrt(surf_fields_sum_square / len(fm_dict))

                surf_scaling_factors = [surf_fields_mean, surf_fields_std]

            if cfg.model.normalization == "min_max_scaling":
                for j in range(len(fm_dict)):
                    d_dict = fm_dict[j]
                    surf_fields = d_dict["surface_fields"]

                    if surf_fields is not None:
                        surf_mean = np.mean(surf_fields, 0)
                        surf_std = np.std(surf_fields, 0)
                        surf_idx = mean_std_sampling(
                            surf_fields, surf_mean, surf_std, tolerance=12.0
                        )
                        surf_fields_sampled = np.delete(surf_fields, surf_idx, axis=0)
                        if j == 0:
                            surf_fields_max = np.amax(surf_fields_sampled, 0)
                            surf_fields_min = np.amin(surf_fields_sampled, 0)
                        else:
                            surf_fields_max1 = np.amax(surf_fields_sampled, 0)
                            surf_fields_min1 = np.amin(surf_fields_sampled, 0)

                            for k in range(surf_fields.shape[-1]):
                                if surf_fields_max1[k] > surf_fields_max[k]:
                                    surf_fields_max[k] = surf_fields_max1[k]

                                if surf_fields_min1[k] < surf_fields_min[k]:
                                    surf_fields_min[k] = surf_fields_min1[k]
                    else:
                        surf_fields_max = 0.0
                        surf_fields_min = 0.0

                    if j > 20:
                        break

                surf_scaling_factors = [surf_fields_max, surf_fields_min]
            np.save(surf_save_path, surf_scaling_factors)


class CachedDoMINODataset(Dataset):
    """
    Dataset for reading cached DoMINO data files, with optional resampling.
    Acts as a drop-in replacement for DoMINODataPipe.
    """

    # @nvtx_annotate(message="CachedDoMINODataset __init__")
    def __init__(
        self,
        data_path: Union[str, Path],
        phase: Literal["train", "val", "test"] = "train",
        sampling: bool = False,
        volume_points_sample: Optional[int] = None,
        surface_points_sample: Optional[int] = None,
        geom_points_sample: Optional[int] = None,
        model_type=None,  # Model_type, surface, volume or combined
        deterministic_seed=False,
        surface_sampling_algorithm="area_weighted",
    ):
        super().__init__()

        self.model_type = model_type
        if deterministic_seed:
            np.random.seed(42)

        if isinstance(data_path, str):
            data_path = Path(data_path)
        self.data_path = data_path.expanduser()

        if not self.data_path.exists():
            raise AssertionError(f"Path {self.data_path} does not exist")
        if not self.data_path.is_dir():
            raise AssertionError(f"Path {self.data_path} is not a directory")

        self.deterministic_seed = deterministic_seed
        self.sampling = sampling
        self.volume_points = volume_points_sample
        self.surface_points = surface_points_sample
        self.geom_points = geom_points_sample
        self.surface_sampling_algorithm = surface_sampling_algorithm

        self.filenames = get_filenames(self.data_path, exclude_dirs=True)

        total_files = len(self.filenames)

        self.phase = phase
        self.indices = np.array(range(total_files))

        np.random.shuffle(self.indices)

        if not self.filenames:
            raise AssertionError(f"No cached files found in {self.data_path}")

    def __len__(self):
        return len(self.indices)

    # @nvtx_annotate(message="CachedDoMINODataset __getitem__")
    def __getitem__(self, idx):
        if self.deterministic_seed:
            np.random.seed(idx)
        nvtx.range_push("Load cached file")

        index = self.indices[idx]
        cfd_filename = self.filenames[index]

        filepath = self.data_path / cfd_filename
        result = np.load(filepath, allow_pickle=True).item()
        result = {
            k: v.numpy() if isinstance(v, Tensor) else v for k, v in result.items()
        }

        nvtx.range_pop()
        if not self.sampling:
            return result

        nvtx.range_push("Sample points")

        # Sample volume points if present
        if "volume_mesh_centers" in result and self.volume_points:
            coords_sampled, idx_volume = shuffle_array(
                result["volume_mesh_centers"], self.volume_points
            )
            if coords_sampled.shape[0] < self.volume_points:
                coords_sampled = pad(
                    coords_sampled, self.volume_points, pad_value=-10.0
                )

            result["volume_mesh_centers"] = coords_sampled
            for key in [
                "volume_fields",
                "pos_volume_closest",
                "pos_volume_center_of_mass",
                "sdf_nodes",
            ]:
                if key in result:
                    result[key] = result[key][idx_volume]

        # Sample surface points if present
        if "surface_mesh_centers" in result and self.surface_points:
            if self.surface_sampling_algorithm == "area_weighted":
                coords_sampled, idx_surface = area_weighted_shuffle_array(
                    result["surface_mesh_centers"],
                    self.surface_points,
                    result["surface_areas"],
                )
            else:
                coords_sampled, idx_surface = shuffle_array(
                    result["surface_mesh_centers"], self.surface_points
                )

            if coords_sampled.shape[0] < self.surface_points:
                coords_sampled = pad(
                    coords_sampled, self.surface_points, pad_value=-10.0
                )

            ii = result["neighbor_indices"]
            result["surface_mesh_neighbors"] = result["surface_mesh_centers"][ii]
            result["surface_neighbors_normals"] = result["surface_normals"][ii]
            result["surface_neighbors_areas"] = result["surface_areas"][ii]

            result["surface_mesh_centers"] = coords_sampled

            for key in [
                "surface_fields",
                "surface_areas",
                "surface_normals",
                "pos_surface_center_of_mass",
                "surface_mesh_neighbors",
                "surface_neighbors_normals",
                "surface_neighbors_areas",
            ]:
                if key in result:
                    result[key] = result[key][idx_surface]

            del result["neighbor_indices"]

        # Sample geometry points if present
        if "geometry_coordinates" in result and self.geom_points:
            coords_sampled, _ = shuffle_array(
                result["geometry_coordinates"], self.geom_points
            )
            if coords_sampled.shape[0] < self.geom_points:
                coords_sampled = pad(coords_sampled, self.geom_points, pad_value=-100.0)
            result["geometry_coordinates"] = coords_sampled

        nvtx.range_pop()
        return result


def create_domino_dataset(
    cfg, phase, volume_variable_names, surface_variable_names, vol_factors, surf_factors
):
    if phase == "train":
        input_path = cfg.data.input_dir
    elif phase == "val":
        input_path = cfg.data.input_dir_val
    else:
        raise ValueError(f"Invalid phase {phase}")

    if cfg.data_processor.use_cache:
        return CachedDoMINODataset(
            input_path,
            phase=phase,
            sampling=True,
            volume_points_sample=cfg.model.volume_points_sample,
            surface_points_sample=cfg.model.surface_points_sample,
            geom_points_sample=cfg.model.geom_points_sample,
            model_type=cfg.model.model_type,
            surface_sampling_algorithm=cfg.model.surface_sampling_algorithm,
        )
    else:
        return DoMINODataPipe(
            input_path,
            phase=phase,
            grid_resolution=cfg.model.interp_res,
            volume_variables=volume_variable_names,
            surface_variables=surface_variable_names,
            normalize_coordinates=True,
            sampling=True,
            sample_in_bbox=True,
            volume_points_sample=cfg.model.volume_points_sample,
            surface_points_sample=cfg.model.surface_points_sample,
            geom_points_sample=cfg.model.geom_points_sample,
            positional_encoding=cfg.model.positional_encoding,
            volume_factors=vol_factors,
            surface_factors=surf_factors,
            scaling_type=cfg.model.normalization,
            model_type=cfg.model.model_type,
            bounding_box_dims=cfg.data.bounding_box,
            bounding_box_dims_surf=cfg.data.bounding_box_surface,
            num_surface_neighbors=cfg.model.num_surface_neighbors,
            resample_surfaces=cfg.model.resampling_surface_mesh.resample,
            resampling_points=cfg.model.resampling_surface_mesh.points,
            surface_sampling_algorithm=cfg.model.surface_sampling_algorithm,
        )


if __name__ == "__main__":
    fm_data = DoMINODataPipe(
        data_path="/code/processed_data/new_models_1/",
        phase="train",
        sampling=False,
        sample_in_bbox=False,
    )
