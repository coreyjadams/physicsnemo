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
MLP fusion utility for enhancing neural network performance.

This module provides functionality to fuse sequences of PyTorch modules 
(particularly linear layers and activations) into bandwidth optimized implementations
using various backends like Warp, Triton, or NVMATH. This fusion can 
significantly improve performance by reducing memory bandwidth requirements.
"""

from enum import Enum
from typing import Dict, List, Set, Type

import torch
import torch.fx as fx
import torch.nn as nn
from torch.fx import symbolic_trace


class BackendType(Enum):
    """Enumeration of supported fusion backend types."""

    # WARP = "warp"
    TRITON = "triton"
    # NVMATH = "nvmath"


# Registry of supported operations for extensibility
_SUPPORTED_OPS: Dict[Type[nn.Module], str] = {
    nn.Linear: "linear",
    nn.ReLU: "relu",
}


def is_supported_op(module: nn.Module) -> bool:
    """Check if a module type is supported by the fusion implementation.

    Args:
        module: The PyTorch module to check

    Returns:
        True if the module type is supported, False otherwise
    """
    return type(module) in _SUPPORTED_OPS


def get_supported_ops() -> Set[Type[nn.Module]]:
    """Return the set of currently supported operations.

    Returns:
        A set containing the module classes that are supported for fusion
    """
    return set(_SUPPORTED_OPS.keys())


class FusionBackend:
    """Base class for backend implementations.

    All fusion backends should inherit from this class and implement
    its abstract methods.
    """

    def __init__(self) -> None:
        """Initialize the fusion backend."""
        pass

    def fuse_modules(self, modules: List[nn.Module]) -> nn.Module:
        """Fuse a list of modules into a single optimized module.

        Args:
            modules: List of PyTorch modules to fuse

        Returns:
            A PyTorch module with the fused implementation

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Backend must implement fuse_modules")

    @classmethod
    def is_available(cls) -> bool:
        """Check if this backend is available on the current system.

        Returns:
            True if the backend can be used, False otherwise

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError("Backend must implement is_available")


class FusedMLP(nn.Module):
    """A wrapper module for a fused MLP implementation.

    Helps with managing parameters and ensuring the save/restore utilities work.
    Enforces a strict linear -> activation -> linear -> ... pattern with a maximum
    of 5 linear layers.
    """

    def __init__(self, traced_module: fx.GraphModule) -> None:
        """Initialize a FusedMLP from a traced module.

        Args:
            traced_module: A traced PyTorch module containing the MLP to fuse
        """
        super().__init__()
        self.fused_module = None  # Will be set during initialization

        # Parameters for linear layers
        self.weights = []
        self.biases = []

        # Activation types (0 for identity, 1 for relu)
        self.activations = []

        # Layer information
        self.layers = []  # Will store (layer_type, param_index) tuples

        self.output_shape = None

        self.max_layers = 5

        # Transform the traced module
        self.transform(traced_module)

    def transform(self, traced_module: fx.GraphModule) -> None:
        """Go through the original graph and extract layer information.

        Enforces strict linear -> activation -> linear -> ... pattern with max 5 linear layers.

        Args:
            traced_module: A traced PyTorch module containing the MLP to fuse

        Raises:
            ValueError: If the module pattern doesn't match requirements for fusion
        """
        # Reset counters and storage
        self.weights = []
        self.biases = []
        self.activations = []
        self.layers = []

        # Track expected operation type (0 for linear, 1 for activation)
        expected_op = 0
        linear_count = 0
        activation_count = 0

        for node in traced_module.graph.nodes:
            # Skip placeholders (inputs) and output nodes
            if node.op == "placeholder" or node.op == "output":
                continue

            if node.op == "call_module":
                module = traced_module.get_submodule(node.target)

                # Check if this module type is supported
                if not is_supported_op(module):
                    raise ValueError(
                        f"Unsupported module for fusion: {type(module).__name__}"
                    )

                # Check pattern: must alternate between linear and activation
                if expected_op == 0:  # Expecting linear
                    if not isinstance(module, nn.Linear):
                        raise ValueError(
                            "MLP fusion requires strict linear->activation->linear pattern"
                        )

                    # Check we haven't exceeded max layers
                    if linear_count >= self.max_layers:
                        raise ValueError(
                            f"Maximum of {self.max_layers} linear layers supported"
                        )

                    # Store weights and biases
                    self.weights.append(module.weight)
                    self.biases.append(module.bias)  # Could be None

                    # Add to layer information
                    self.layers.append(("linear", linear_count))
                    linear_count += 1
                    expected_op = 1  # Next should be activation

                else:  # Expecting activation
                    if not isinstance(
                        module, nn.ReLU
                    ):  # Could expand to other activation types
                        raise ValueError(
                            "MLP fusion requires strict linear->activation->linear pattern"
                        )

                    # Store activation type (1 for ReLU)
                    self.activations.append(1)

                    # Add to layer information
                    self.layers.append(("activation", activation_count))
                    activation_count += 1
                    expected_op = 0  # Next should be linear

            # We don't support other operation types
            elif node.op in ["call_function", "call_method"]:
                raise ValueError(
                    f"Unsupported operation type: {node.op}, target: {node.target}"
                )

        # Final validation - we must have at least one linear layer
        if linear_count == 0:
            raise ValueError("No linear layers found in the module")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the fused MLP.

        Args:
            x: Input tensor

        Returns:
            Output tensor

        Raises:
            NotImplementedError: Forward pass implementation is not yet available
        """
        # TODO: Implement the forward pass using the fused kernel
        raise NotImplementedError("Forward pass not implemented")


def validate_modules(modules: List[nn.Module]) -> bool:
    """Validate that all modules in the list are supported for fusion.

    Args:
        modules: List of PyTorch modules to validate

    Returns:
        True if all modules are supported

    Raises:
        ValueError: If any module is not supported for fusion
    """
    for i, module in enumerate(modules):
        if not is_supported_op(module):
            raise ValueError(
                f"Module at index {i} is not supported for fusion: {type(module).__name__}"
            )
    return True


def get_backend_class(backend_type: BackendType) -> Type[FusionBackend]:
    """Get the appropriate backend class based on the backend type.

    This serves as an import guard primarily.
    Args:
        backend_type: The type of backend to use

    Returns:
        The backend class (not an instance)

    Raises:
        ImportError: If the requested backend is not available
        ValueError: If the backend type is unknown
    """
    # if backend_type == BackendType.WARP:
    #     from physicsnemo.utils.fusion.providers.warp import WarpFusedMLP

    #     if not WarpFusedMLP.is_available():
    #         raise ImportError(f"Backend {backend_type.value} is not available")
    #     return WarpFusedMLP
    if backend_type == BackendType.TRITON:
        from physicsnemo.utils.fusion.providers.triton import TritonFusedMLP

        if not TritonFusedMLP.is_available():
            raise ImportError(f"Backend {backend_type.value} is not available")
        return TritonFusedMLP
    # elif backend_type == BackendType.NVMATH:
    #     from physicsnemo.utils.fusion.providers.nvmath import NVMathFusedMLP
    #     if not NVMathFusedMLP.is_available():
    #         raise ImportError(f"Backend {backend_type.value} is not available")
    #     return NVMathFusedMLP()
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")


def fuse_mlp(
    module: nn.Module,
    backend: str = "warp",
) -> nn.Module:
    """
    Fuse a sequence of PyTorch modules into an optimized bandwidth-efficient implementation.

    Args:
        module: A PyTorch module containing the MLP to fuse
        backend: The backend to use for fusion ("warp", "triton", or "nvmath")

    Returns:
        A PyTorch module with the fused implementation

    Raises:
        ValueError: If the backend is unknown or modules are not suitable for fusion
    """
    print(module)
    # Validate modules
    # validate_modules(modules)
    traced_module = symbolic_trace(module)
    print(f"traced_module: {traced_module}")

    # Select backend
    try:
        backend_type = BackendType(backend)
    except ValueError:
        raise ValueError(
            f"Unknown backend: {backend}. Supported backends: {[b.value for b in BackendType]}"
        )

    # Try to get the default backend if specified one is not available
    backend_impl = get_backend_class(backend_type)

    # Create and return the fused module
    return backend_impl(traced_module)
