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

from typing import List

import pytest
import torch
import torch.nn as nn

from physicsnemo.utils.fusion.mlp import fuse_mlp


class SimpleMLP(nn.Module):
    """A simple MLP model with configurable number of Linear+ReLU blocks."""

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()

        layers = []

        # Input layer
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=False))
            # layers.append(nn.ReLU())
            prev_dim = hidden_dim

        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@pytest.mark.parametrize("input_dim", [64, 128])
@pytest.mark.parametrize("hidden_dims", [[32, 64, 32], [128, 64, 32, 16]])
@pytest.mark.parametrize("batch_size", [1, 16])
def test_mlp_fusion_correctness(input_dim, hidden_dims, batch_size):
    """Test that fused MLP produces the same results as the original."""
    # Set a fixed seed for reproducibility
    torch.manual_seed(42)

    # Create a model
    model = SimpleMLP(input_dim, hidden_dims).cuda()
    model.eval()  # Set to evaluation mode

    print(f"model parameters: {list(model.parameters())}")

    # Generate random input data
    x = torch.randn(batch_size, input_dim).cuda()

    # Run the original model
    with torch.no_grad():
        original_output = model(x)

    # Fuse the model
    try:
        fused_model = fuse_mlp(model, backend="triton")

        # Run the fused model with the same input
        with torch.no_grad():
            fused_output = fused_model(x)

        print(f"original_output = {original_output}")
        print(f"fused_output = {fused_output}")
        # print(f"diff = {original_output - fused_output}")
        # Check that outputs match
        assert torch.allclose(
            original_output, fused_output, rtol=1e-4, atol=1e-4
        ), f"Fused model output doesn't match original model output. Max diff: {(original_output - fused_output).abs().max().item()}"

    except ImportError as e:
        pytest.skip(f"Skipping test due to missing backend: {e}")


# @pytest.mark.parametrize("backend", ["warp", "triton", "nvmath"])
# def test_mlp_fusion_backends(backend):
#     """Test that different backends can be selected for fusion."""
#     # Create a small model
#     model = SimpleMLP(32, [16, 8])
#     layers = list(model.layers)

#     # Try to fuse with the specified backend
#     try:
#         fused_model = fuse_mlp(layers, backend=backend)
#         assert fused_model is not None, f"Fusion with {backend} backend failed"
#     except ImportError:
#         pytest.skip(f"Backend {backend} not available, skipping test")


# def test_moduleList_input():
#     """Test that the fusion works with ModuleList input."""
#     model = SimpleMLP(32, [16, 8])
#     module_list = model.layers  # This is already a ModuleList

#     try:
#         fused_model = fuse_mlp(module_list, backend="warp")
#         assert fused_model is not None, "Fusion with ModuleList input failed"
#     except ImportError:
#         pytest.skip("Warp backend not available, skipping test")


# def test_complex_case():
#     """Test a more complex case with multiple Linear+ReLU blocks."""
#     # Create a deeper model
#     model = SimpleMLP(128, [256, 512, 256, 128, 64, 32])

#     # Generate random input data
#     x = torch.randn(32, 128)

#     # Run the original model
#     with torch.no_grad():
#         original_output = model(x)

#     # Fuse the model
#     try:
#         fused_model = fuse_mlp(model.layers, backend="warp")

#         # Run the fused model
#         with torch.no_grad():
#             fused_output = fused_model(x)

#         # Check outputs
#         assert torch.allclose(original_output, fused_output, rtol=1e-4, atol=1e-4), \
#             "Outputs don't match for complex model"
#     except ImportError:
#         pytest.skip("Warp backend not available, skipping test")

if __name__ == "__main__":
    test_mlp_fusion_correctness(
        16,
        [
            16,
        ],
        16,
    )
