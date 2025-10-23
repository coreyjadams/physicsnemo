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
sampling random points in the volume and on surface,
normalizing fields and returning the output tensors as a dictionary.

This datapipe also non-dimensionalizes the fields, so the order in which the variables should
be fixed: velocity, pressure, turbulent viscosity for volume variables and
pressure, wall-shear-stress for surface variables. The different parameters such as
variable names, domain resolution, sampling size etc. are configurable in config.yaml.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from physicsnemo.datapipes.cae.cae_dataset import (
    CAEDataset,
)
from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.domino.utils import (
    normalize,
    standardize,
    unnormalize,
    unstandardize,
)

# from physicsnemo.utils.sdf import signed_distance_field


@dataclass
class TransolverDataConfig:
    """Base Configuration for Transolver dataset processing pipeline.

    Attributes:
        data_path: Path to the dataset to load.
        phase: Which phase of data to load ("train", "val", or "test").
        resolution: Resolution of the sampled data, per batch.
        volume_sample_from_disk: (Volume specific) If the volume data is in a shuffled state on disk,
            read contiguous chunks of the data rather than the entire volume data.  This greatly
            accelerates IO in bandwidth limited systems or when the volumetric data is very large.
        volume_factors: (Volume specific) Non-dimensionalization factors for volume variables scaling.
            If set, and scaling_type is:
            - min_max_scaling -> rescale volume_fields to the min/max set here
            - mean_std_scaling -> rescale volume_fields to the mean and std set here.
        bounding_box_dims: (Volume specific) Dimensions of bounding box. Must be an object with min/max
            attributes that are arraylike.
        grid_resolution: Resolution of the latent grid.
        normalize_coordinates: Whether to normalize coordinates based on min/max values.
        sample_in_bbox: Whether to sample points in a specified bounding box.
            Uses the same min/max points as coordinate normalization.
            Only performed if compute_scaling_factors is false.
        sampling: Whether to downsample the full resolution mesh to fit in GPU memory.
            Surface and volume sampling points are configured separately as:
            - surface.points_sample
            - volume.points_sample
        geom_points_sample: Number of STL points sampled per batch.
            Independent of volume.points_sample and surface.points_sample.
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

    data_path: Path | None
    model_type: Literal["surface", "volume"] = "surface"
    resolution: int = 200_000
    # Apply some normalization to coordinate values of inputs,
    # and derived features
    # normalize_coordinates: bool = False
    # geom_points_sample: int = 300000

    # Control what features are added to the inputs to the model:
    include_normals: bool = True
    include_sdf: bool = True

    # For controlling the normalization of target values:
    scaling_type: Optional[Literal["min_max_scaling", "mean_std_scaling"]] = None
    normalization_factors: Optional[torch.Tensor] = None

    # Add these invariances?
    rotational_invariance: bool = True
    reference_direction: torch.Tensor = torch.tensor([1.0, 0.0, 0.0])

    translational_invariance: bool = True
    # If none, uses the center of mass:
    reference_origin: torch.Tensor | None = None

    scale_invariance: bool = True
    # Scale factor is aligned with the preferred direction!
    reference_scale_factor: torch.Tensor = torch.tensor([1.0, 1.0, 1.0])

    broadcast_global_features: bool = True

    sample_from_disk: bool = False

    def __post_init__(self):
        if self.data_path is not None:
            # Ensure data_path is a Path object:
            if isinstance(self.data_path, str):
                self.data_path = Path(self.data_path)
            self.data_path = self.data_path.expanduser()

            if not self.data_path.exists():
                raise ValueError(f"Path {self.data_path} does not exist")

            if not self.data_path.is_dir():
                raise ValueError(f"Path {self.data_path} is not a directory")

        if self.scaling_type is not None:
            if self.scaling_type not in [
                # "min_max_scaling",
                "mean_std_scaling",
            ]:
                raise ValueError(
                    f"scaling_type should be one of ['min_max_scaling', 'mean_std_scaling'], got {self.scaling_type}"
                )


class TransolverDataPipe(Dataset):
    """
    Base Datapipe for Transolver

    Leverages a dataset for the actual reading of the data, and this
    object is responsible for preprocessing the data.

    """

    def __init__(
        self,
        input_path,
        model_type: Literal["surface", "volume"],
        pin_memory: bool = False,
        **data_config_overrides,
    ):
        # Perform config packaging and validation
        self.config = TransolverDataConfig(
            data_path=input_path, model_type=model_type, **data_config_overrides
        )

        # Set up the distributed manager:
        if not DistributedManager.is_initialized():
            DistributedManager.initialize()

        self.dataset = None

    # @profile
    # def downsample_geometry(
    #     self,
    #     stl_vertices,
    # ) -> torch.Tensor:
    #     """
    #     Downsample the geometry to the desired number of points.

    #     Args:
    #         stl_vertices: The vertices of the surface.
    #     """

    #     if self.config.sampling:
    #         geometry_points = self.config.geom_points_sample

    #         geometry_coordinates_sampled, idx_geometry = shuffle_array(
    #             stl_vertices, geometry_points
    #         )
    #         if geometry_coordinates_sampled.shape[0] < geometry_points:
    #             raise ValueError(
    #                 "Surface mesh has fewer points than requested sample size"
    #             )
    #         geom_centers = geometry_coordinates_sampled
    #     else:
    #         geom_centers = stl_vertices

    #     return geom_centers

    @torch.no_grad()
    def process_data(self, data_dict):
        """
        Preprocess the data.  We have slight differences between surface and volume data processing,
        mostly revolving around the keys that represent the inputs.

        - For surface data, we use the mesh coordinates and normals as the embeddings.
            - Normals are always normalized to 1.0, and are a relative direction.
            - coordinates can be shifted to the center of mass, and then the whole
              coordinate system can be aligned to the preferred direction.
            - SDF is identically 0 for surface data.
            - Optionally, if the scale invariance is enabled, the coordinates
              are scaled by the (maybe-rotated) scale factor.

        - For Volume data: we still use the volume coordinates
            - normals are approximated as the direction between the volume point
              and closest mesh point.  Normalized to 1.0.
            - SDF is not zero for volume data.


        To make the calculations consistent and logical to follow:
        - First, get the coordinates (volume_mesh_centers or surface_mesh_centers, usually)
          which is a configuration.
        - Second, get the STL information.  We need the "stl_vertices" and "stl_indices"
          to compute an SDF.  We downsample "stl_coordinates" to potentially encode
          a geometry tensor, which is optional.

        Then, start imposing optional symmetries:
        - Impose translation invariance.  For every "position-like" tensor, subtract
          off the reference_origin if translation invariance is enabled.
        - Second, impose scale invariance: for every position-like tensor, multiply
          by the reference scale.
        - Finally, apply rotation invariance.  Normals are rotated, points are rotated.
          Roation requires not just a reference vector (in the config) but a
          vector unique to this example to come from the data - we have to rotate to it.

        After that, the rest is simple:
          - Spatial Encodings are the point locations + normal vectors (optional) + sdf (optional)
            - If the normals aren't provided, we derive them from the center of mass (without SDF) or SDF point (with SDF)
          - Geometry encoding (if using) is the STL coordinates, downsampled.
          - parameter encodings are straight forward vectors / reference values.

        The downstream applications can take the embeddings and the features as needed.

        """

        # Validate that all required keys are present in data_dict
        required_keys = [
            "stl_centers",
        ]

        if self.config.model_type == "volume":
            # We need these for the SDF calculation:
            required_keys.extend(
                [
                    "stl_coordinates",
                    "stl_faces",
                ]
            )

        field_key = f"{self.config.model_type}_fields"
        coords_key = f"{self.config.model_type}_mesh_centers"

        required_keys.extend(
            [
                field_key,
                coords_key,
            ]
        )

        missing_keys = [key for key in required_keys if key not in data_dict]
        if missing_keys:
            raise ValueError(
                f"Missing required keys in data_dict: {missing_keys}. "
                f"Required keys are: {required_keys}"
            )

        # # Start building the preprocessed return dict:
        # return_dict = {
        #     "global_params_values": data_dict["global_params_values"],
        #     "global_params_reference": data_dict["global_params_reference"],
        # }

        ########################################################################
        # Process the core STL information
        ########################################################################

        # This function gets information about the surface scale,
        # and decides what the surface grid will be:

        # stl_coordinates = data_dict["stl_coordinates"]

        # # We always need to calculate the SDF on the surface grid:
        # # This is for the SDF Later:
        # if self.config.normalize_coordinates:
        #     normed_vertices = normalize(data_dict["stl_coordinates"], s_max, s_min)
        # else:
        #     normed_vertices = data_dict["stl_coordinates"]

        # For SDF calculations, make sure the mesh_indices_flattened is an integer array:
        # mesh_indices_flattened = data_dict["stl_faces"].to(torch.int32)

        # This is a center of mass computation for the stl surface,
        # using the size of each mesh point as weight.
        center_of_mass = torch.mean(data_dict["stl_centers"], dim=0)

        fields = data_dict["surface_fields"]
        coords = data_dict["surface_mesh_centers"] - center_of_mass

        return fields, coords

    def scale_model_targets(
        self, fields: torch.Tensor, factors: torch.Tensor
    ) -> torch.Tensor:
        """
        Scale the model targets based on the configured scaling factors.
        """
        if self.config.scaling_type == "mean_std_scaling":
            field_mean = factors[0]
            field_std = factors[1]
            return standardize(fields, field_mean, field_std)
        elif self.config.scaling_type == "min_max_scaling":
            field_min = factors[1]
            field_max = factors[0]
            return normalize(fields, field_max, field_min)

    def unscale_model_outputs(
        self,
        fields: torch.Tensor | None = None,
    ):
        """
        Unscale the model outputs based on the configured scaling factors.

        The unscaling is included here to make it a consistent interface regardless
        of the scaling factors and type used.

        """

        if self.config.scaling_type == "mean_std_scaling":
            field_mean = self.config.volume_factors[0]
            field_std = self.config.volume_factors[1]
            fields = unstandardize(fields, field_mean, field_std)
        elif self.config.scaling_type == "min_max_scaling":
            field_min = self.config.volume_factors[1]
            field_max = self.config.volume_factors[0]
            fields = unnormalize(fields, field_max, field_min)

        return fields

    def set_dataset(self, dataset: Iterable) -> None:
        """
        Pass a dataset to the datapipe to enable iterating over both in one pass.
        """
        self.dataset = dataset

        if self.config.sample_from_disk:
            # We deliberately double the data to read compared to the sampling size:
            self.dataset.set_volume_sampling_size(25 * self.config.volume_points_sample)

    def __len__(self):
        if self.dataset is not None:
            return len(self.dataset)
        else:
            return 0

    def __getitem__(self, idx):
        """
        Function for fetching and processing a single file's data.

        Domino, in general, expects one example per file and the files
        are relatively large due to the mesh size.

        Requires the user to have set a dataset via `set_dataset`.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not present")

        # Get the data from the dataset.
        # Under the hood, this may be fetching preloaded data.
        data_dict = self.dataset[idx]

        return self.__call__(data_dict)

    def __call__(self, data_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Process the incoming data dictionary.
        - Processes the data
        - moves it to GPU
        - adds a batch dimension

        Args:
            data_dict: Dictionary containing the data to process as torch.Tensors.

        Returns:
            Dictionary containing the processed data as torch.Tensors.

        """
        fields, coords = self.process_data(data_dict)

        # Add a batch dimension to the data_dict
        fields = fields.unsqueeze(0)
        coords = coords.unsqueeze(0)

        return fields, coords

    def __iter__(self):
        if self.dataset is None:
            raise ValueError(
                "Dataset is not present, can not use the datapipe as an iterator."
            )

        for i, batch in enumerate(self.dataset):
            yield self.__call__(batch)


def create_transolver_dataset(
    cfg: DictConfig,
    phase: Literal["train", "val", "test"],
    # keys_to_read: list[str],
    # keys_to_read_if_available: dict[str, torch.Tensor],
    scaling_factors: list[float],
    # normalize_coordinates: bool = True,
    device_mesh: torch.distributed.DeviceMesh | None = None,
    placements: dict[str, torch.distributed.tensor.Placement] | None = None,
):
    model_type = cfg.mode
    if phase == "train":
        input_path = cfg.train.data_path
    elif phase == "val":
        input_path = cfg.val.data_path
    # elif phase == "test":
    # input_path = cfg.eval.test_path
    else:
        raise ValueError(f"Invalid phase {phase}")

    # The dataset path works in two pieces:
    # There is a core "dataset" which is loading data and moving to GPU
    # And there is the preprocess step, here.

    # Optionally, and for backwards compatibility, the preprocess
    # object can accept a dataset which will enable it as an iterator.
    # The iteration function will loop over the dataset, preprocess the
    # output, and return it.

    keys_to_read = cfg.data_keys

    overrides = {}

    dm = DistributedManager()

    if torch.cuda.is_available():
        device = dm.device
        consumer_stream = torch.cuda.default_stream()
    else:
        device = torch.device("cpu")
        consumer_stream = None

    if cfg.get("preload_depth", None) is not None:
        preload_depth = cfg.preload_depth
    else:
        preload_depth = 1

    if cfg.get("pin_memory", None) is not None:
        pin_memory = cfg.pin_memory
    else:
        pin_memory = False

    dataset = CAEDataset(
        data_dir=input_path,
        keys_to_read=keys_to_read,
        keys_to_read_if_available={},
        output_device=device,
        preload_depth=preload_depth,
        pin_memory=pin_memory,
        device_mesh=device_mesh,
        placements=placements,
        consumer_stream=consumer_stream,
    )

    datapipe = TransolverDataPipe(
        input_path,
        resolution=cfg.resolution,
        normalization_factors=scaling_factors,
        model_type=model_type,
        scaling_type="mean_std_scaling",
        **overrides,
    )

    datapipe.set_dataset(dataset)

    return datapipe
