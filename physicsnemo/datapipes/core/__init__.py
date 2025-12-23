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

"""
datapipe - High-performance GPU-centric data loading for Scientific ML

A modular, composable data pipeline for physics and scientific machine learning.
Designed for clean separation of concerns:

- **Readers**: Load data from sources → TensorDict tuples with CPU tensors
- **Transforms**: Process TensorDict data
- **Dataset**: Reader + transforms pipeline with optional auto device transfer
- **DataLoader**: Batched iteration with optional prefetching

Example:
    >>> import physicsnemo.datapipes.core as dp
    >>> from tensordict import TensorDict
    >>>
    >>> # Create a dataset with transforms and automatic device transfer
    >>> dataset = dp.Dataset(
    ...     reader=dp.HDF5Reader("data.h5", fields=["pressure", "velocity"]),
    ...     transforms=[
    ...         dp.Normalize(input_keys=["pressure"], means={"pressure": 0.0}, stds={"pressure": 1.0}),
    ...         dp.Downsample(input_keys=["pressure", "velocity"], n=10000),
    ...     ],
    ...     device="cuda",  # Automatic GPU transfer!
    ... )
    >>>
    >>> # Create a dataloader
    >>> loader = dp.DataLoader(dataset, batch_size=16, shuffle=True)
    >>>
    >>> # Iterate over batches
    >>> for data, metadata in loader:
    ...     output = model(data["pressure"])
"""

from tensordict import TensorDict

from physicsnemo.datapipes.core.collate import (
    Collator,
    ConcatCollator,
    DefaultCollator,
    FunctionCollator,
    concat_collate,
    default_collate,
    get_collator,
)
from physicsnemo.datapipes.core.dataloader import DataLoader
from physicsnemo.datapipes.core.dataset import Dataset
from physicsnemo.datapipes.core.readers import (
    HDF5Reader,
    NumpyReader,
    Reader,
    ZarrReader,
)
from physicsnemo.datapipes.core.registry import (
    READER_REGISTRY,
    TRANSFORM_REGISTRY,
    ComponentRegistry,
    register_reader,
    register_transform,
)
from physicsnemo.datapipes.core.transforms import (
    Compose,
    Normalize,
    SubsamplePoints,
    Transform,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "TensorDict",  # Re-export from tensordict
    "Dataset",
    "DataLoader",
    # Transforms
    "Transform",
    "Compose",
    "Normalize",
    "SubsamplePoints",
    # Readers
    "Reader",
    "HDF5Reader",
    "ZarrReader",
    "NumpyReader",
    # Collation
    "Collator",
    "DefaultCollator",
    "ConcatCollator",
    "FunctionCollator",
    "default_collate",
    "concat_collate",
    "get_collator",
    # Registry
    "ComponentRegistry",
    "TRANSFORM_REGISTRY",
    "READER_REGISTRY",
    "register_transform",
    "register_reader",
]
