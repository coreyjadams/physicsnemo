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

"""Triton-accelerated fused MLP kernels for tall-skinny matmuls.

These kernels fuse a chain of matmuls, biases, and activations into a single
Triton kernel to avoid the global-memory round trips that make small,
un-normalized MLPs bandwidth-bound when the batch dimension is very large.
"""

from physicsnemo.core.version_check import OptionalImport, get_package_hint

# Triton ships with CUDA builds of PyTorch. The kernel modules below import it
# unconditionally, so surface a clean, actionable error here (the package's
# public entry point) instead of a bare NameError/ImportError from deep inside
# the kernel modules when Triton is unavailable.
if not OptionalImport("triton").available:
    raise ImportError(
        "Triton is required to use physicsnemo.nn.functional.fused_mlp.\n"
        + get_package_hint("triton")
    )

from .activations import Activation
from .wrappers import fused_mlp

__all__ = [
    "Activation",
    "fused_mlp",
]
