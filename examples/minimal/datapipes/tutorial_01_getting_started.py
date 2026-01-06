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

import torch

# Import the core datapipe components
from physicsnemo.datapipes.core import (
    DataLoader,
    Dataset,
)
from physicsnemo.datapipes.core.readers import ZarrReader

"""
Tutorial 1: Getting Started with PhysicsNemo DataPipes
======================================================

This tutorial introduces the core concepts of PhysicsNemo's data loading
infrastructure. You'll learn how to:

1. Create a Reader to load data from files
2. Understand the (TensorDict, metadata) return format
3. Wrap a reader in a Dataset
4. Iterate with a DataLoader
5. Access batch data via TensorDict keys

Prerequisites
-------------
Before running this tutorial, generate some synthetic data:
"""

# Generate 100 samples with velocity, pressure, and position fields
gen_cmd = 'python generate_regular_data.py -n 100 -s "velocity:128,128,128,3 pressure:128,128,128,1 position:128,128,128,3" -b zarr -o output/tutorial_data/'

"""
This creates a directory structure like:
    output/tutorial_data/
    ├── sample_000000.zarr/
    ├── sample_000001.zarr/
    ├── ...
    └── metadata.json

Run this tutorial:
    python tutorial_01_getting_started.py

Key Concepts
------------
- **Reader**: Loads raw data from storage (HDF5, Zarr, NumPy, etc.)
- **TensorDict**: A dictionary-like container for named tensors
- **Dataset**: Combines a Reader with transforms and handles device transfer
- **DataLoader**: Batches samples and manages prefetching for efficiency
"""


def check_data_exists(data_path: str) -> bool:
    """Check if tutorial data exists and provide helpful message if not."""
    path = Path(data_path)
    if not path.exists():
        print(f"ERROR: Data not found at '{data_path}'")
        print()
        print("Please generate tutorial data first:")
        print()
        print(gen_cmd)
        print()
        return False
    return True


# =============================================================================
# Section 1: Creating a Reader
# =============================================================================
def section_1_reader_basics():
    """
    Section 1: Creating Your First Reader

    Readers are the foundation of the datapipe system. They handle loading
    data from various file formats and converting it to PyTorch tensors.

    PhysicsNemo provides several built-in readers:
    - ZarrReader: For Zarr arrays (chunked, compressed storage)
    - HDF5Reader: For HDF5 files
    - NumpyReader: For .npy/.npz files
    - VTKReader: For VTK mesh files
    """
    print("=" * 70)
    print("Section 1: Creating Your First Reader")
    print("=" * 70)
    print()

    # Path to our tutorial data
    data_path = "./output/tutorial_data/"

    if not check_data_exists(data_path):
        return None

    # Create a ZarrReader
    # The reader automatically discovers all .zarr files in the directory
    reader = ZarrReader(
        path=data_path,
        group_pattern="*.zarr",  # Match files ending in .zarr
    )

    print(f"Created reader: {reader}")
    print(f"Number of samples: {len(reader)}")
    print(f"Field names: {reader.field_names}")
    print()

    # Let's load a single sample directly from the reader
    print("Loading sample 0 directly from reader...")
    data, metadata = reader[0]

    print(f"Data type: {type(data)}")
    print(f"Metadata: {metadata}")
    print()

    # Examine the TensorDict contents
    print("TensorDict contents:")
    for key in data.keys():
        tensor = data[key]
        print(
            f"  '{key}': shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}"
        )

    print()
    return reader


# =============================================================================
# Section 2: Understanding TensorDict
# =============================================================================
def section_2_tensordict_basics(reader):
    """
    Section 2: Understanding the (TensorDict, metadata) Format

    Every Reader returns a tuple of (TensorDict, metadata):

    - TensorDict: A dictionary-like container holding named tensors
      - Access tensors by key: data["velocity"], data["pressure"]
      - Supports batch operations, device transfers, and more
      - From the tensordict library (PyTorch ecosystem)

    - metadata: A regular Python dict with non-tensor information
      - Source file paths, sample indices, etc.
      - Useful for debugging and tracking data provenance
    """
    print("=" * 70)
    print("Section 2: Understanding TensorDict")
    print("=" * 70)
    print()

    if reader is None:
        print("Skipping - reader not available")
        return

    # Load a sample
    data, metadata = reader[0]

    # TensorDict acts like a dictionary
    print("Accessing data like a dictionary:")
    print(f"  data['velocity'].shape = {data['velocity'].shape}")
    print(f"  data['pressure'].shape = {data['pressure'].shape}")
    print()

    # You can iterate over keys
    print("Iterating over TensorDict:")
    for key, value in data.items():
        print(f"  {key}: {value.shape}")
    print()

    # TensorDict supports device transfers
    print("Device operations:")
    print(f"  Current device: {data.device}")

    if torch.cuda.is_available():
        data_gpu = data.to("cuda")
        print(f"  After .to('cuda'): {data_gpu.device}")
        print(f"  data_gpu['velocity'].device = {data_gpu['velocity'].device}")
    else:
        print("  (CUDA not available - skipping GPU transfer demo)")
    print()

    # Metadata contains non-tensor information
    print("Metadata contents:")
    for key, value in metadata.items():
        print(f"  '{key}': {value}")
    print()


# =============================================================================
# Section 3: Wrapping Reader in Dataset
# =============================================================================
def section_3_dataset_basics():
    """
    Section 3: Wrapping a Reader in a Dataset

    The Dataset class wraps a Reader and adds:
    - Transform pipeline support (covered in Tutorial 2)
    - Automatic device transfer (move data to GPU)
    - Prefetching capabilities for performance

    Dataset is the recommended way to access data for training.
    """
    print("=" * 70)
    print("Section 3: Wrapping Reader in Dataset")
    print("=" * 70)
    print()

    data_path = "./output/tutorial_data/"
    if not check_data_exists(data_path):
        return None

    # Create reader
    reader = ZarrReader(path=data_path, group_pattern="*.zarr")

    # Wrap in Dataset - simplest case, no transforms
    dataset = Dataset(reader=reader)

    print(f"Dataset: {dataset}")
    print(f"Length: {len(dataset)}")
    print()

    # Access samples via indexing (same as reader, but through dataset)
    print("Accessing samples through Dataset:")
    data, metadata = dataset[0]
    print(f"  Sample 0 keys: {list(data.keys())}")
    print()

    # Dataset supports automatic GPU transfer!
    if torch.cuda.is_available():
        print("Creating Dataset with automatic GPU transfer:")
        dataset_gpu = Dataset(reader=reader, device="cuda")

        data_gpu, _ = dataset_gpu[0]
        print(f"  Data device: {data_gpu.device}")
        print(f"  velocity device: {data_gpu['velocity'].device}")

        # Clean up
        dataset_gpu.close()
    else:
        print("(CUDA not available - skipping GPU dataset demo)")
    print()

    return dataset


# =============================================================================
# Section 4: Using the DataLoader
# =============================================================================
def section_4_dataloader_basics(dataset):
    """
    Section 4: Iterating with a DataLoader

    The DataLoader provides batched iteration over a Dataset:
    - Batches multiple samples together
    - Supports shuffling
    - Manages prefetching with CUDA streams for performance
    - Compatible with PyTorch's DistributedSampler

    This is the typical interface for training loops.
    """
    print("=" * 70)
    print("Section 4: Iterating with DataLoader")
    print("=" * 70)
    print()

    if dataset is None:
        print("Skipping - dataset not available")
        return

    # Create a DataLoader with batch_size=4
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,  # Shuffle samples each epoch
    )

    print(f"DataLoader batch_size: {dataloader.batch_size}")
    print(f"Number of batches: {len(dataloader)}")
    print()

    # Iterate over batches
    print("Iterating over batches:")
    for batch_idx, batch_data in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Batch data type: {type(batch_data)}")

        for key in batch_data.keys():
            tensor = batch_data[key]
            # Note: batch dimension is added as first dimension
            print(f"  '{key}': shape={tensor.shape}")

        # Just show first 2 batches for brevity
        if batch_idx >= 1:
            print("\n  ... (showing only first 2 batches)")
            break

    print()


# =============================================================================
# Section 5: Putting It All Together
# =============================================================================
def section_5_training_loop_example():
    """
    Section 5: A Simple Training Loop Example

    This section shows how datapipes fit into a typical training workflow.
    We'll create a mock training loop that demonstrates:
    - Loading batches of data
    - Accessing specific fields for model input/output
    - Basic timing for performance awareness
    """
    print("=" * 70)
    print("Section 5: Training Loop Example")
    print("=" * 70)
    print()

    data_path = "./output/tutorial_data/"
    if not check_data_exists(data_path):
        return

    # Setup: Reader -> Dataset -> DataLoader
    reader = ZarrReader(path=data_path, group_pattern="*.zarr")

    # For GPU training, specify device="cuda"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = Dataset(reader=reader, device=device)

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        drop_last=True,  # Drop incomplete final batch
    )

    print(f"Training on device: {device}")
    print(f"Samples: {len(dataset)}, Batches per epoch: {len(dataloader)}")
    print()

    # Mock training loop
    print("Mock training loop (2 epochs):")
    num_epochs = 2

    for epoch in range(num_epochs):
        epoch_start = time.time()

        for batch_idx, batch_data in enumerate(dataloader):
            # In a real training loop, you would:
            # 1. Extract inputs and targets
            velocity = batch_data["velocity"]  # Input features
            pressure = batch_data["pressure"]  # Target to predict

            # 2. Forward pass through model
            # output = model(velocity)

            # 3. Compute loss
            # loss = criterion(output, pressure)

            # 4. Backward pass and optimize
            # loss.backward()
            # optimizer.step()

            # For demo, just print shapes
            if batch_idx == 0:
                print(
                    f"  Epoch {epoch}: velocity {velocity.shape}, "
                    f"pressure {pressure.shape}, device={velocity.device}"
                )

        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch} completed in {epoch_time:.3f}s")

    print()
    print("Training complete!")
    print()

    # Clean up
    dataset.close()


# =============================================================================
# Main
# =============================================================================
def main():
    """Run all tutorial sections."""
    print()
    print("╔" + "═" * 68 + "╗")
    print(
        "║"
        + " Tutorial 1: Getting Started with PhysicsNemo DataPipes ".center(68)
        + "║"
    )
    print("╚" + "═" * 68 + "╝")
    print()

    # Section 1: Reader basics
    reader = section_1_reader_basics()

    # Section 2: TensorDict format
    section_2_tensordict_basics(reader)

    # Section 3: Dataset wrapper
    dataset = section_3_dataset_basics()

    # Section 4: DataLoader iteration
    section_4_dataloader_basics(dataset)

    # Section 5: Training loop example
    section_5_training_loop_example()

    # Cleanup
    if reader is not None:
        reader.close()
    if dataset is not None:
        dataset.close()

    print("=" * 70)
    print("Tutorial 1 Complete!")
    print()
    print("Key takeaways:")
    print("  1. Readers load raw data and return (TensorDict, metadata) tuples")
    print("  2. TensorDict is a dictionary-like container for named tensors")
    print("  3. Dataset wraps Reader + transforms + automatic device transfer")
    print("  4. DataLoader provides batched iteration for training")
    print()
    print("Next: Tutorial 2 - Transforms and Data Preprocessing")
    print("=" * 70)


if __name__ == "__main__":
    main()
