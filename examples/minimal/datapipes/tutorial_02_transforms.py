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


import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

# Import core datapipe components
from physicsnemo.datapipes.core import DataLoader, Dataset
from physicsnemo.datapipes.core.readers import ZarrReader

# Import transforms
from physicsnemo.datapipes.core.transforms import (
    Compose,
    Normalize,
    ReScale,
    SubsamplePoints,
    Translate,
)

"""
Tutorial 2: Transforms and Data Preprocessing
==============================================

This tutorial covers the transform system in PhysicsNemo DataPipes.
You'll learn how to:

1. Apply a single transform (Normalize)
2. Compose multiple transforms together
3. Subsample point clouds with SubsamplePoints
4. Use geometric transforms (Translate, ReScale)
5. Save/load normalization statistics from files
6. Denormalize data with the inverse() method

Prerequisites
-------------
Before running this tutorial, generate some synthetic data:

"""
# For regular grid data (Sections 1-2, 5-6):
gen_cmd_regular = """python generate_regular_data.py -n 100 -s "velocity:128,128,128,3 pressure:128,128,128,1 position:128,128,128,3" -b zarr -o output/tutorial_data/"""

# For point cloud data (Sections 3-4, 7):
gen_cmd_cloud = """python generate_variable_points_data.py -n 100 -s "coords:3 features:8" --min-points 50000 --max-points 100000 -b zarr -o output/pointcloud_data/"""

"""
Run this tutorial:
    python tutorial_02_transforms.py

Key Concepts
------------
- **Transform**: An operation that takes a TensorDict and returns a modified TensorDict
- **Compose**: Chains multiple transforms into a pipeline
- **input_keys**: Most transforms specify which fields to operate on
- **state_dict()**: Transforms can be serialized for reproducibility
"""


def check_data_exists(data_path: str, generation_command: str) -> bool:
    """Check if tutorial data exists and provide helpful message if not."""
    path = Path(data_path)
    if not path.exists():
        print(f"ERROR: Data not found at '{data_path}'")
        print()
        print("Please generate tutorial data first:")
        print()
        print(f"    {generation_command}")
        print()
        return False
    return True


# =============================================================================
# Section 1: Single Transform - Normalize
# =============================================================================
def section_1_single_transform():
    """
    Section 1: Applying a Single Transform (Normalize)

    The Normalize transform standardizes tensor values. It supports two methods:
    - mean_std: (x - mean) / std
    - min_max: scales to [-1, 1] range

    Transforms operate on TensorDict objects and return modified TensorDicts.
    """
    print("=" * 70)
    print("Section 1: Single Transform - Normalize")
    print("=" * 70)
    print()

    data_path = "./output/tutorial_data/"
    if not check_data_exists(data_path, gen_cmd_regular):
        return None

    # Create reader and load a sample
    reader = ZarrReader(path=data_path, group_pattern="*.zarr")
    data, metadata = reader[0]

    print("Before normalization:")
    print(
        f"  velocity: mean={data['velocity'].mean():.4f}, std={data['velocity'].std():.4f}"
    )
    print(
        f"  pressure: mean={data['pressure'].mean():.4f}, std={data['pressure'].std():.4f}"
    )
    print()

    # Create a Normalize transform
    # This will subtract mean and divide by std for specified keys
    normalize = Normalize(
        input_keys=["velocity", "pressure"],
        method="mean_std",
        # For real data, you'd compute these from your training set
        means={"velocity": 0.0, "pressure": 0.0},
        stds={"velocity": 0.6, "pressure": 0.6},  # ~std of uniform[-1,1]
    )

    print(f"Transform: {normalize}")
    print()

    # Apply the transform
    normalized_data = normalize(data)

    print("After normalization:")
    print(
        f"  velocity: mean={normalized_data['velocity'].mean():.4f}, std={normalized_data['velocity'].std():.4f}"
    )
    print(
        f"  pressure: mean={normalized_data['pressure'].mean():.4f}, std={normalized_data['pressure'].std():.4f}"
    )
    print()

    # Demonstrate min-max normalization
    print("Min-Max normalization example:")
    normalize_minmax = Normalize(
        input_keys=["velocity"],
        method="min_max",
        mins={"velocity": -1.0},
        maxs={"velocity": 1.0},
    )

    minmax_data = normalize_minmax(data)
    print(
        f"  velocity range: [{minmax_data['velocity'].min():.4f}, {minmax_data['velocity'].max():.4f}]"
    )
    print()

    reader.close()
    return normalize


# =============================================================================
# Section 2: Composing Multiple Transforms
# =============================================================================
def section_2_compose_transforms():
    """
    Section 2: Composing Multiple Transforms

    The Compose class chains multiple transforms together, applying them
    in sequence. This is similar to torchvision.transforms.Compose.

    Transform pipelines are the recommended way to build preprocessing.
    """
    print("=" * 70)
    print("Section 2: Composing Multiple Transforms")
    print("=" * 70)
    print()

    data_path = "./output/tutorial_data/"
    if not check_data_exists(data_path, gen_cmd_regular):
        return None

    reader = ZarrReader(path=data_path, group_pattern="*.zarr")

    # Create multiple transforms
    normalize_velocity = Normalize(
        input_keys=["velocity"],
        method="mean_std",
        means={"velocity": 0.0},
        stds={"velocity": 0.6},
    )

    normalize_pressure = Normalize(
        input_keys=["pressure"],
        method="mean_std",
        means={"pressure": 0.0},
        stds={"pressure": 0.6},
    )

    # Compose them into a pipeline
    transform_pipeline = Compose(
        [
            normalize_velocity,
            normalize_pressure,
        ]
    )

    print(f"Transform pipeline:\n{transform_pipeline}")
    print()

    # Apply pipeline to data
    data, _ = reader[0]

    print("Before pipeline:")
    print(f"  velocity std: {data['velocity'].std():.4f}")
    print(f"  pressure std: {data['pressure'].std():.4f}")

    transformed_data = transform_pipeline(data)

    print("After pipeline:")
    print(f"  velocity std: {transformed_data['velocity'].std():.4f}")
    print(f"  pressure std: {transformed_data['pressure'].std():.4f}")
    print()

    # Better approach: Use transforms directly with Dataset
    print("Using transforms with Dataset (recommended approach):")

    dataset = Dataset(
        reader=reader,
        transforms=[normalize_velocity, normalize_pressure],
    )

    data, _ = dataset[0]
    print(f"  velocity std: {data['velocity'].std():.4f}")
    print(f"  pressure std: {data['pressure'].std():.4f}")
    print()

    dataset.close()
    return transform_pipeline


# =============================================================================
# Section 3: Point Cloud Subsampling
# =============================================================================
def section_3_subsampling():
    """
    Section 3: Point Cloud Subsampling with SubsamplePoints

    Scientific data often involves large point clouds (meshes, particles).
    SubsamplePoints efficiently downsamples while maintaining correspondence
    between related fields (coordinates, features, normals, etc.).

    Supports:
    - Uniform random sampling
    - Poisson disk sampling (for very large datasets)
    - Weighted sampling (e.g., area-weighted for surfaces)
    """
    print("=" * 70)
    print("Section 3: Point Cloud Subsampling")
    print("=" * 70)
    print()

    data_path = "./output/pointcloud_data/"
    if not check_data_exists(data_path, gen_cmd_cloud):
        return None

    reader = ZarrReader(path=data_path, group_pattern="*.zarr")

    # Load a sample to see its original size
    data, metadata = reader[0]

    print("Original point cloud:")
    print(f"  coords shape: {data['coords'].shape}")
    print(f"  features shape: {data['features'].shape}")
    print()

    # Create a SubsamplePoints transform
    # This samples the same indices from both coords and features
    subsample = SubsamplePoints(
        input_keys=["coords", "features"],  # Keys to subsample together
        n_points=10000,  # Target number of points
        algorithm="uniform",  # or "poisson_fixed" for very large data
    )
    # Note: the subsampling will assume a consistent leading dimension for all
    # its input keys: so, it will generate an index of shape [n_points] and slice
    # all input_keys in the same way.

    print(f"Transform: {subsample}")
    print()

    # Apply subsampling
    subsampled_data = subsample(data)

    print("After subsampling:")
    print(f"  coords shape: {subsampled_data['coords'].shape}")
    print(f"  features shape: {subsampled_data['features'].shape}")
    print()

    # Use with Dataset for full pipeline
    print("Using SubsamplePoints in a Dataset:")

    dataset = Dataset(
        reader=reader,
        transforms=[subsample],
    )

    # Iterate over a few samples
    for i in range(3):
        data, _ = dataset[i]
        print(
            f"  Sample {i}: coords {data['coords'].shape}, features {data['features'].shape}"
        )

    print()
    dataset.close()
    return subsample


# =============================================================================
# Section 4: Geometric Transforms
# =============================================================================
def section_4_geometric_transforms():
    """
    Section 4: Geometric Transforms (Translate, ReScale)

    PhysicsNemo provides geometric transforms useful for point clouds and meshes:
    - Translate: Shift coordinates by a fixed offset
    - ReScale: Scale coordinates by a factor

    These are commonly used for data augmentation or centering data.
    """
    print("=" * 70)
    print("Section 4: Geometric Transforms")
    print("=" * 70)
    print()

    data_path = "./output/pointcloud_data/"
    if not check_data_exists(data_path, gen_cmd_cloud):
        return None

    reader = ZarrReader(path=data_path, group_pattern="*.zarr")
    data, _ = reader[0]

    # Original statistics
    print("Original coordinates:")
    coords = data["coords"]
    print(
        f"  Mean: [{coords[:, 0].mean():.4f}, {coords[:, 1].mean():.4f}, {coords[:, 2].mean():.4f}]"
    )
    print(f"  Min:  [{coords.min():.4f}]")
    print(f"  Max:  [{coords.max():.4f}]")
    print()

    # Translate: shift coordinates by subtracting a center point
    # center_key_or_value can be a tensor or a key name referencing a tensor in the data
    translate = Translate(
        input_keys=["coords"],
        center_key_or_value=torch.tensor(
            [-0.5, -0.5, -0.5]
        ),  # Subtract this (shifts by +0.5)
    )

    translated_data = translate(data)
    t_coords = translated_data["coords"]

    print("After Translate([0.5, 0.5, 0.5]):")
    print(
        f"  Mean: [{t_coords[:, 0].mean():.4f}, {t_coords[:, 1].mean():.4f}, {t_coords[:, 2].mean():.4f}]"
    )
    print()

    # ReScale: scale coordinates by dividing by a reference scale
    # To scale UP by 2x, divide by 0.5
    rescale = ReScale(
        input_keys=["coords"],
        reference_scale=torch.tensor([0.5, 0.5, 0.5]),  # Divide by this (scales by 2x)
    )

    rescaled_data = rescale(data)
    r_coords = rescaled_data["coords"]

    print("After ReScale(2.0):")
    print(f"  Min: [{r_coords.min():.4f}]")
    print(f"  Max: [{r_coords.max():.4f}]")
    print()

    # Compose geometric transforms with other transforms
    print("Complete preprocessing pipeline:")

    # First subsample, then center (translate), then scale
    pipeline = Compose(
        [
            SubsamplePoints(input_keys=["coords", "features"], n_points=5000),
            Translate(
                input_keys=["coords"],
                center_key_or_value=torch.tensor(
                    [0.0, 0.0, 0.0]
                ),  # Subtract origin (no-op here)
            ),
            ReScale(
                input_keys=["coords"],
                reference_scale=torch.tensor([0.5, 0.5, 0.5]),  # Scale up by 2x
            ),
        ]
    )

    processed_data = pipeline(data)
    print(f"  Final coords shape: {processed_data['coords'].shape}")
    print(f"  Final features shape: {processed_data['features'].shape}")
    print()

    reader.close()


# =============================================================================
# Section 5: Saving and Loading Normalization Statistics
# =============================================================================
def section_5_stats_serialization():
    """
    Section 5: Saving/Loading Normalization Statistics

    For reproducibility, you can save normalization statistics to files
    and load them later. This is essential for:
    - Using the same normalization at training and inference time
    - Sharing preprocessing configs across experiments
    """
    print("=" * 70)
    print("Section 5: Saving/Loading Normalization Statistics")
    print("=" * 70)
    print()

    data_path = "./output/tutorial_data/"
    if not check_data_exists(data_path, gen_cmd_regular):
        return None

    reader = ZarrReader(path=data_path, group_pattern="*.zarr")

    # In practice, compute statistics from your training data
    print("Step 1: Compute statistics from training data")
    print("  (In practice, iterate over all samples to compute mean/std)")

    # For demo, we'll use known values for uniform[-1,1] data
    velocity_mean = 0.0
    velocity_std = 0.58  # Approximately sqrt(1/3) for uniform[-1,1]
    pressure_mean = 0.0
    pressure_std = 0.58

    print(f"  velocity: mean={velocity_mean}, std={velocity_std}")
    print(f"  pressure: mean={pressure_mean}, std={pressure_std}")
    print()

    # Create a temporary directory for saving stats
    with TemporaryDirectory() as tmpdir:
        stats_file = Path(tmpdir) / "normalization_stats.npz"

        # Step 2: Save statistics to .npz file
        print(f"Step 2: Save statistics to {stats_file.name}")

        # The file format expected by Normalize.load_stats_from_npz:
        # Each field maps to a dict with 'mean', 'std', 'min', 'max' keys
        np.savez(
            stats_file,
            velocity={"mean": np.array(velocity_mean), "std": np.array(velocity_std)},
            pressure={"mean": np.array(pressure_mean), "std": np.array(pressure_std)},
        )
        print("  Stats saved!")
        print()

        # Step 3: Load statistics when creating transform
        print("Step 3: Create Normalize transform from stats file")

        normalize = Normalize(
            input_keys=["velocity", "pressure"],
            method="mean_std",
            stats_file=str(stats_file),
        )

        print(f"  Loaded transform: {normalize}")
        print()

        # Verify it works
        data, _ = reader[0]
        normalized = normalize(data)
        print("Step 4: Verify normalization")
        print(f"  velocity std after normalization: {normalized['velocity'].std():.4f}")
        print(f"  pressure std after normalization: {normalized['pressure'].std():.4f}")
        print()

    # Alternative: Use state_dict() for serialization
    print("Alternative: Using state_dict() for serialization")

    normalize = Normalize(
        input_keys=["velocity"],
        method="mean_std",
        means={"velocity": 0.0},
        stds={"velocity": 0.58},
    )

    state = normalize.state_dict()
    print(f"  state_dict keys: {list(state.keys())}")
    print()

    # Create a new transform and load the state
    new_normalize = Normalize(
        input_keys=["velocity"],
        method="mean_std",
        means={"velocity": 999.0},  # Placeholder
        stds={"velocity": 999.0},
    )
    new_normalize.load_state_dict(state)

    print("  Loaded state into new transform ✓")
    print()

    reader.close()


# =============================================================================
# Section 6: Denormalization with inverse()
# =============================================================================
def section_6_inverse_normalization():
    """
    Section 6: Denormalization with the inverse() Method

    After your model makes predictions, you often need to convert back
    to physical units. The Normalize transform provides an inverse()
    method for this purpose.
    """
    print("=" * 70)
    print("Section 6: Denormalization with inverse()")
    print("=" * 70)
    print()

    data_path = "./output/tutorial_data/"
    if not check_data_exists(data_path, gen_cmd_regular):
        return None

    reader = ZarrReader(path=data_path, group_pattern="*.zarr")
    data, _ = reader[0]

    print("Original data statistics:")
    print(f"  pressure mean: {data['pressure'].mean():.4f}")
    print(f"  pressure std:  {data['pressure'].std():.4f}")
    print(f"  pressure min:  {data['pressure'].min():.4f}")
    print(f"  pressure max:  {data['pressure'].max():.4f}")
    print()

    # Create normalizer
    normalize = Normalize(
        input_keys=["pressure"],
        method="mean_std",
        means={"pressure": 0.0},
        stds={"pressure": 0.58},
    )

    # Forward: normalize
    normalized_data = normalize(data)
    print("After normalization:")
    print(f"  pressure mean: {normalized_data['pressure'].mean():.4f}")
    print(f"  pressure std:  {normalized_data['pressure'].std():.4f}")
    print()

    # Inverse: denormalize
    denormalized_data = normalize.inverse(normalized_data)
    print("After denormalization (inverse):")
    print(f"  pressure mean: {denormalized_data['pressure'].mean():.4f}")
    print(f"  pressure std:  {denormalized_data['pressure'].std():.4f}")
    print()

    # Verify round-trip accuracy
    original_pressure = data["pressure"]
    roundtrip_pressure = denormalized_data["pressure"]
    max_error = (original_pressure - roundtrip_pressure).abs().max()

    print("Round-trip verification:")
    print(f"  Max absolute error: {max_error:.2e}")
    print(f"  Round-trip accurate: {'✓' if max_error < 1e-5 else '✗'}")
    print()

    # Practical example: Model prediction pipeline
    print("Practical example: Model prediction pipeline")
    print("  1. Load data, feed to model")
    print("  2. Model outputs normalized prediction")
    print("  2. Normalize targets → compute model loss")
    print("  4. Denormalize output → get physical values and metrics")
    print()

    reader.close()


# =============================================================================
# Section 7: Complete Pipeline Example
# =============================================================================
def section_7_complete_pipeline():
    """
    Section 7: Complete Preprocessing Pipeline

    This section demonstrates a realistic preprocessing pipeline combining
    multiple transforms in a training-ready configuration.
    """
    print("=" * 70)
    print("Section 7: Complete Preprocessing Pipeline")
    print("=" * 70)
    print()

    data_path = "./output/pointcloud_data/"
    if not check_data_exists(data_path, gen_cmd_cloud):
        return None

    reader = ZarrReader(path=data_path, group_pattern="*.zarr")

    print("Building a complete preprocessing pipeline:")
    print()
    print("  Pipeline steps:")
    print("    1. SubsamplePoints: Reduce to 10,000 points")
    print("    2. Translate: Center at origin")
    print("    3. ReScale: Normalize spatial extent")
    print("    4. Normalize: Standardize feature values")
    print()

    # Define transforms
    transforms = [
        # Step 1: Subsample to manageable size
        SubsamplePoints(
            input_keys=["coords", "features"],
            n_points=10000,
            algorithm="uniform",
        ),
        # Step 2: Translate (center at origin)
        Translate(
            input_keys=["coords"],
            center_key_or_value=torch.tensor([0.0, 0.0, 0.0]),
        ),
        # Step 3: Scale coordinates (divide by 0.5 = multiply by 2)
        ReScale(
            input_keys=["coords"],
            reference_scale=torch.tensor([0.5, 0.5, 0.5]),
        ),
        # Step 4: Normalize features
        Normalize(
            input_keys=["features"],
            method="mean_std",
            means={"features": 0.0},
            stds={"features": 0.58},
        ),
    ]

    # Create dataset with the full pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = Dataset(
        reader=reader,
        transforms=transforms,
        device=device,
    )

    print(f"Dataset created on device: {device}")
    print(f"Number of samples: {len(dataset)}")
    print()

    # Create DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
    )

    print("Sample batch from DataLoader:")
    batch_data = next(iter(dataloader))

    for key in batch_data.keys():
        tensor = batch_data[key]
        print(f"  '{key}': shape={tensor.shape}, device={tensor.device}")
        print(f"          mean={tensor.mean():.4f}, std={tensor.std():.4f}")

    print()

    # Timing comparison
    print("Performance comparison:")

    # Time several iterations
    start = time.time()
    for i, batch_data in enumerate(dataloader):
        if i >= 4:
            break
        # Simulate some computation
        _ = batch_data["features"].sum()
    elapsed = time.time() - start

    print(f"  5 batches loaded and processed in {elapsed:.3f}s")
    print(f"  Average time per batch: {elapsed / 5 * 1000:.1f}ms")
    print()

    dataset.close()
    print("Pipeline complete!")
    print()


# =============================================================================
# Main
# =============================================================================
def main():
    """Run all tutorial sections."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " Tutorial 2: Transforms and Data Preprocessing ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Section 1: Single transform
    section_1_single_transform()

    # Section 2: Compose transforms
    section_2_compose_transforms()

    # Section 3: Point cloud subsampling
    section_3_subsampling()

    # Section 4: Geometric transforms
    section_4_geometric_transforms()

    # Section 5: Stats serialization
    section_5_stats_serialization()

    # Section 6: Inverse normalization
    section_6_inverse_normalization()

    # Section 7: Complete pipeline
    section_7_complete_pipeline()

    print("=" * 70)
    print("Tutorial 2 Complete!")
    print()
    print("Key takeaways:")
    print("  1. Transforms operate on TensorDict and return modified TensorDict")
    print("  2. Compose chains multiple transforms into a pipeline")
    print("  3. SubsamplePoints maintains correspondence between related fields")
    print("  4. Geometric transforms (Translate, ReScale) help with data prep")
    print("  5. Save/load normalization stats for reproducibility")
    print("  6. Use inverse() to convert predictions back to physical units")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
