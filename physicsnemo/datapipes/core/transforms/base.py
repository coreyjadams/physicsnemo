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
Transform base class - The foundation for all data transformations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from tensordict import TensorDict


class Transform(ABC):
    """
    Abstract base class for all transforms.

    Transforms operate on a TensorDict and return a modified TensorDict.
    They are designed to run on GPU tensors for maximum performance.
    Metadata is not passed to transforms (handled separately by Dataset/DataLoader).

    Subclasses must implement:
        - __call__(data: TensorDict) -> TensorDict

    Optionally override:
        - extra_repr() -> str: For custom repr output
        - state_dict() -> dict: For serialization
        - load_state_dict(state_dict: dict): For deserialization

    Example:
        >>> class MyTransform(Transform):
        ...     def __init__(self, scale: float):
        ...         super().__init__()
        ...         self.scale = scale
        ...
        ...     def __call__(self, data: TensorDict) -> TensorDict:
        ...         # Apply transformation to all tensors
        ...         return data.apply(lambda x: x * self.scale)
    """

    def __init__(self) -> None:
        """Initialize the transform."""
        self._device: Optional[torch.device] = None

    @abstractmethod
    def __call__(self, data: TensorDict) -> TensorDict:
        """
        Apply the transform to a TensorDict.

        Args:
            data: Input TensorDict to transform.

        Returns:
            Transformed TensorDict.
        """
        raise NotImplementedError

    def to(self, device: torch.device | str) -> Transform:
        """
        Move any internal tensors to the specified device.

        Override this method if your transform has internal tensor state.

        Args:
            device: Target device.

        Returns:
            Self for chaining.
        """
        self._device = torch.device(device) if isinstance(device, str) else device
        return self

    @property
    def device(self) -> Optional[torch.device]:
        """The device this transform operates on."""
        return self._device

    def extra_repr(self) -> str:
        """
        Return extra information for repr.

        Override this to add transform-specific info to the repr.
        """
        return ""

    def state_dict(self) -> dict[str, Any]:
        """
        Return a dictionary containing the transform's state.

        Override this for transforms with learnable or configurable state.
        """
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load state from a state dictionary.

        Override this to restore transform state.
        """
        pass

    def __repr__(self) -> str:
        extra = self.extra_repr()
        if extra:
            return f"{self.__class__.__name__}({extra})"
        return f"{self.__class__.__name__}()"
