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
Collation utilities - Batch multiple (TensorDict, metadata) tuples.

Collators combine multiple (TensorDict, dict) tuples from Dataset into a single
batched (TensorDict, list[dict]) tuple suitable for model consumption.
The default collator stacks TensorDicts along batch dimension using TensorDict.stack().

Metadata is collated by collecting values into lists.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Sequence, Union

import torch
from tensordict import TensorDict


def _collate_metadata(metadata_list: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Collate metadata from multiple samples.

    Simply returns the list of metadata dicts as-is. Each metadata dict
    corresponds to one sample in the batch.

    Args:
        metadata_list: Sequence of metadata dicts.

    Returns:
        List of metadata dicts.
    """
    return list(metadata_list)


class Collator(ABC):
    """
    Abstract base class for collators.

    Collators take a sequence of (TensorDict, dict) tuples and combine them
    into a single batched (TensorDict, list[dict]) tuple. The base class defines
    the interface; subclasses implement specific batching strategies.

    Metadata is automatically collated into a list of dicts.

    Example:
        >>> class MyCollator(Collator):
        ...     def __call__(
        ...         self,
        ...         samples: Sequence[tuple[TensorDict, dict]]
        ...     ) -> tuple[TensorDict, list[dict]]:
        ...         # Custom batching logic
        ...         ...
    """

    @abstractmethod
    def __call__(
        self, samples: Sequence[tuple[TensorDict, dict[str, Any]]]
    ) -> tuple[TensorDict, list[dict[str, Any]]]:
        """
        Collate a batch of samples.

        Args:
            samples: Sequence of (TensorDict, metadata dict) tuples to batch.

        Returns:
            Tuple of (batched TensorDict, list of metadata dicts).
        """
        raise NotImplementedError


class DefaultCollator(Collator):
    """
    Default collator that stacks TensorDicts along a new batch dimension.

    Uses TensorDict.stack() to efficiently batch all tensors, creating
    shape [batch_size, ...original_shape] for each field.

    All samples must have:
    - The same tensor keys
    - Tensors with matching shapes (per key)
    - Tensors on the same device

    Metadata is collated into a list of dicts.

    Example:
        >>> data1 = TensorDict({"x": torch.randn(10, 3)}, device="cpu")
        >>> data2 = TensorDict({"x": torch.randn(10, 3)}, device="cpu")
        >>> samples = [
        ...     (data1, {"file": "a.h5"}),
        ...     (data2, {"file": "b.h5"}),
        ... ]
        >>> collator = DefaultCollator()
        >>> batched_data, metadata_list = collator(samples)
        >>> batched_data["x"].shape  # torch.Size([2, 10, 3])
        >>> metadata_list  # [{"file": "a.h5"}, {"file": "b.h5"}]
    """

    def __init__(
        self,
        *,
        stack_dim: int = 0,
        keys: Optional[list[str]] = None,
        collate_metadata: bool = True,
    ) -> None:
        """
        Initialize the collator.

        Args:
            stack_dim: Dimension along which to stack tensors (default: 0).
            keys: If provided, only collate these tensor keys. Others are ignored.
            collate_metadata: If True, collate metadata into list (default: True).
        """
        self.stack_dim = stack_dim
        self.keys = keys
        self.collate_metadata = collate_metadata

    def __call__(
        self, samples: Sequence[tuple[TensorDict, dict[str, Any]]]
    ) -> tuple[TensorDict, list[dict[str, Any]]]:
        """
        Collate samples by stacking TensorDicts.

        Args:
            samples: Sequence of (TensorDict, metadata) tuples to batch.

        Returns:
            Tuple of (batched TensorDict, list of metadata dicts).

        Raises:
            ValueError: If samples is empty or samples have mismatched keys/shapes.
        """
        if not samples:
            raise ValueError("Cannot collate empty sequence of samples")

        # Separate data and metadata
        data_list = [data for data, _ in samples]
        metadata_list = [meta for _, meta in samples]

        # Use TensorDict.stack() for efficient batching
        if self.keys is not None:
            # Filter to only requested keys
            data_list = [data.select(*self.keys) for data in data_list]

        batched_data = torch.stack(data_list, dim=self.stack_dim)

        # Collate metadata
        if self.collate_metadata:
            metadata = _collate_metadata(metadata_list)
        else:
            metadata = []

        return batched_data, metadata


class ConcatCollator(Collator):
    """
    Collator that concatenates tensors along an existing dimension.

    Unlike DefaultCollator which creates a new batch dimension, this
    concatenates along an existing dimension. Useful for point clouds
    or other variable-length data where you want to combine all points.

    Optionally adds batch indices to track which points came from which sample.
    Metadata is collated into a list of dicts.

    Example:
        >>> data1 = TensorDict({"points": torch.randn(100, 3)})
        >>> data2 = TensorDict({"points": torch.randn(150, 3)})
        >>> samples = [
        ...     (data1, {"file": "a.h5"}),
        ...     (data2, {"file": "b.h5"}),
        ... ]
        >>> collator = ConcatCollator(dim=0, add_batch_idx=True)
        >>> batched_data, metadata_list = collator(samples)
        >>> batched_data["points"].shape  # torch.Size([250, 3])
        >>> batched_data["batch_idx"].shape  # torch.Size([250])
        >>> metadata_list  # [{"file": "a.h5"}, {"file": "b.h5"}]
    """

    def __init__(
        self,
        *,
        dim: int = 0,
        add_batch_idx: bool = True,
        batch_idx_key: str = "batch_idx",
        keys: Optional[list[str]] = None,
        collate_metadata: bool = True,
    ) -> None:
        """
        Initialize the collator.

        Args:
            dim: Dimension along which to concatenate.
            add_batch_idx: If True, add a tensor of batch indices.
            batch_idx_key: Key for the batch index tensor.
            keys: If provided, only collate these tensor keys.
            collate_metadata: If True, collate metadata into lists (default: True).
        """
        self.dim = dim
        self.add_batch_idx = add_batch_idx
        self.batch_idx_key = batch_idx_key
        self.keys = keys
        self.collate_metadata = collate_metadata

    def __call__(
        self, samples: Sequence[tuple[TensorDict, dict[str, Any]]]
    ) -> tuple[TensorDict, list[dict[str, Any]]]:
        """
        Collate samples by concatenating tensors.

        Args:
            samples: Sequence of (TensorDict, metadata) tuples to batch.

        Returns:
            Tuple of (batched TensorDict, list of metadata dicts).

        Raises:
            ValueError: If samples is empty.
        """
        if not samples:
            raise ValueError("Cannot collate empty sequence of samples")

        # Separate data and metadata
        data_list = [data for data, _ in samples]
        metadata_list = [meta for _, meta in samples]

        first_data = data_list[0]
        keys = self.keys if self.keys else list(first_data.keys())
        device = first_data.device

        batched_tensors = {}
        sizes = []  # Track sizes for batch indices

        for key in keys:
            tensors = []
            for data in data_list:
                if key not in data.keys():
                    raise ValueError(f"Data missing key '{key}'")
                tensor = data[key]
                tensors.append(tensor)
                if key == keys[0]:  # Track sizes from first key
                    sizes.append(tensor.shape[self.dim])

            batched_tensors[key] = torch.cat(tensors, dim=self.dim)

        # Add batch indices
        if self.add_batch_idx:
            batch_indices = []
            for i, size in enumerate(sizes):
                batch_indices.append(
                    torch.full((size,), i, dtype=torch.long, device=device)
                )
            batched_tensors[self.batch_idx_key] = torch.cat(batch_indices, dim=0)

        # Create batched TensorDict
        batched_data = TensorDict(batched_tensors, device=device)

        # Collate metadata
        metadata = _collate_metadata(metadata_list) if self.collate_metadata else []

        return batched_data, metadata


class FunctionCollator(Collator):
    """
    Collator that wraps a user-provided function.

    Allows using any function as a collator without subclassing.

    Example:
        >>> def my_collate(samples):
        ...     # Custom logic
        ...     data_list = [d for d, _ in samples]
        ...     metadata_list = [m for _, m in samples]
        ...     return TensorDict.stack(data_list), metadata_list
        >>> collator = FunctionCollator(my_collate)
    """

    def __init__(
        self,
        fn: Callable[
            [Sequence[tuple[TensorDict, dict[str, Any]]]],
            tuple[TensorDict, list[dict[str, Any]]],
        ],
    ) -> None:
        """
        Initialize with a collation function.

        Args:
            fn: Function that takes a sequence of (TensorDict, dict) tuples
                and returns a (TensorDict, list[dict]) tuple.
        """
        self.fn = fn

    def __call__(
        self, samples: Sequence[tuple[TensorDict, dict[str, Any]]]
    ) -> tuple[TensorDict, list[dict[str, Any]]]:
        """Apply the wrapped function."""
        return self.fn(samples)


# Default collator instance
_default_collator = DefaultCollator()


def default_collate(
    samples: Sequence[tuple[TensorDict, dict[str, Any]]],
) -> tuple[TensorDict, list[dict[str, Any]]]:
    """
    Default collation function using stacking.

    Convenience function that uses DefaultCollator.
    Metadata is collated into a list of dicts.

    Args:
        samples: Sequence of (TensorDict, metadata) tuples to batch.

    Returns:
        Tuple of (batched TensorDict, list of metadata dicts).
    """
    return _default_collator(samples)


def concat_collate(
    samples: Sequence[tuple[TensorDict, dict[str, Any]]],
    dim: int = 0,
    add_batch_idx: bool = True,
) -> tuple[TensorDict, list[dict[str, Any]]]:
    """
    Collation function using concatenation.

    Convenience function that uses ConcatCollator.
    Metadata is collated into a list of dicts.

    Args:
        samples: Sequence of (TensorDict, metadata) tuples to batch.
        dim: Dimension along which to concatenate.
        add_batch_idx: If True, add batch index tensor.

    Returns:
        Tuple of (batched TensorDict, list of metadata dicts).
    """
    collator = ConcatCollator(dim=dim, add_batch_idx=add_batch_idx)
    return collator(samples)


def get_collator(
    collate_fn: Optional[
        Union[
            Collator,
            Callable[
                [Sequence[tuple[TensorDict, dict[str, Any]]]],
                tuple[TensorDict, list[dict[str, Any]]],
            ],
        ]
    ] = None,
) -> Collator:
    """
    Get a Collator instance from various input types.

    Args:
        collate_fn: Collator, callable, or None (uses default).

    Returns:
        Collator instance.
    """
    if collate_fn is None:
        return _default_collator
    elif isinstance(collate_fn, Collator):
        return collate_fn
    elif callable(collate_fn):
        return FunctionCollator(collate_fn)
    else:
        raise TypeError(
            f"collate_fn must be Collator, callable, or None, "
            f"got {type(collate_fn).__name__}"
        )
