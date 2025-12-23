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
Registry for datapipe components.

Provides registries for transforms and readers, enabling:
- Short aliases in Hydra configuration
- Component discovery and introspection
- Consistent instantiation patterns

Example usage:
    >>> from physicsnemo.datapipes.core.registry import TRANSFORM_REGISTRY
    >>>
    >>> @TRANSFORM_REGISTRY.register()
    ... class MyTransform(Transform):
    ...     pass
    >>>
    >>> # Get registered component by name
    >>> cls = TRANSFORM_REGISTRY.get("MyTransform")
    >>>
    >>> # List all registered components
    >>> print(TRANSFORM_REGISTRY.list())
"""

from __future__ import annotations

from typing import Callable, Type, TypeVar

T = TypeVar("T")


class ComponentRegistry:
    """
    Registry for datapipe components with short aliases.

    A registry allows components (transforms, readers) to be registered
    with a name and later retrieved by that name. This enables:

    - Hydra configuration with short names instead of full import paths
    - Runtime discovery of available components
    - Validation that a component exists

    Example:
        >>> registry = ComponentRegistry("transforms")
        >>>
        >>> @registry.register()
        ... class Normalize(Transform):
        ...     pass
        >>>
        >>> @registry.register("norm")  # Custom alias
        ... class Normalize(Transform):
        ...     pass
        >>>
        >>> # Retrieve by name
        >>> Normalize = registry.get("Normalize")
        >>> Normalize = registry.get("norm")
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the registry.

        Args:
            name: Human-readable name for this registry (e.g., "transforms").
        """
        self.name = name
        self._registry: dict[str, Type] = {}

    def register(self, name: str | None = None) -> Callable[[Type[T]], Type[T]]:
        """
        Decorator to register a component class.

        Args:
            name: Optional name to register under. If None, uses the class name.

        Returns:
            Decorator function that registers the class.

        Example:
            >>> @registry.register()
            ... class MyTransform(Transform):
            ...     pass
            >>>
            >>> @registry.register("custom_name")
            ... class AnotherTransform(Transform):
            ...     pass
        """

        def decorator(cls: Type[T]) -> Type[T]:
            key = name if name is not None else cls.__name__
            if key in self._registry:
                raise ValueError(
                    f"Component '{key}' is already registered in {self.name} registry. "
                    f"Existing: {self._registry[key]}, New: {cls}"
                )
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> Type:
        """
        Get a registered component by name.

        Args:
            name: The registered name of the component.

        Returns:
            The registered class.

        Raises:
            KeyError: If the name is not registered.
        """
        if name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"Component '{name}' not found in {self.name} registry. "
                f"Available: {available or '(none)'}"
            )
        return self._registry[name]

    def list(self) -> list[str]:
        """
        List all registered component names.

        Returns:
            Sorted list of registered names.
        """
        return sorted(self._registry.keys())

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._registry

    def __len__(self) -> int:
        """Return the number of registered components."""
        return len(self._registry)

    def __repr__(self) -> str:
        return f"ComponentRegistry({self.name!r}, count={len(self)})"


# Global registries for transforms and readers
TRANSFORM_REGISTRY = ComponentRegistry("transforms")
READER_REGISTRY = ComponentRegistry("readers")


def register_transform(name: str | None = None) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a transform class.

    This is a convenience wrapper around TRANSFORM_REGISTRY.register().

    Args:
        name: Optional name to register under. If None, uses the class name.

    Example:
        >>> from physicsnemo.datapipes.core.registry import register_transform
        >>>
        >>> @register_transform()
        ... class MyTransform(Transform):
        ...     pass
    """
    return TRANSFORM_REGISTRY.register(name)


def register_reader(name: str | None = None) -> Callable[[Type[T]], Type[T]]:
    """
    Decorator to register a reader class.

    This is a convenience wrapper around READER_REGISTRY.register().

    Args:
        name: Optional name to register under. If None, uses the class name.

    Example:
        >>> from physicsnemo.datapipes.core.registry import register_reader
        >>>
        >>> @register_reader()
        ... class MyReader(Reader):
        ...     pass
    """
    return READER_REGISTRY.register(name)
