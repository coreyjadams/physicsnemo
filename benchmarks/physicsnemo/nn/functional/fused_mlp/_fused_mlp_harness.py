# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Shared helpers for the fused-MLP parity and benchmark scripts.

This module holds everything common to ``parity_fused_mlp.py`` and
``benchmark_fused_mlp.py``: matmul-precision setup, the PyTorch reference MLP,
random parameter construction, and small error metrics. Keeping it separate lets
the two entry-point scripts stay focused (and lets you run the benchmark without
paying for parity, or vice versa).

The fused-MM precision is read from the environment when the kernels are first
imported, so :func:`set_precision` must be called *before* :func:`import_fused_mlp`.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

# The fused kernels read this env var at import time to pick the tl.dot precision.
PRECISION_ENV = "PHYSICSNEMO_FUSED_MM_PRECISION"


# ---------------------------------------------------------------------------
# CLI / environment setup.
# ---------------------------------------------------------------------------


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add the arguments shared by the parity and benchmark scripts."""
    parser.add_argument(
        "--precision",
        choices=["tf32", "tf32x3", "ieee"],
        default=None,
        help=(
            "Triton matmul input precision. Sets "
            f"{PRECISION_ENV} before importing the kernels. Use 'ieee' for the "
            "tightest parity tolerance. Defaults to the kernel default (tf32x3)."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Input/weight dtype. Default: float32 (matches the tf32x3 path).",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Number of layers to sweep (1, 2, and/or 3).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")


def set_precision(precision: str | None) -> None:
    """Set the fused-MM precision env var. Call before :func:`import_fused_mlp`."""
    if precision is not None:
        os.environ[PRECISION_ENV] = precision


def current_precision() -> str:
    """Return the active fused-MM precision (kernel default is tf32x3)."""
    return os.environ.get(PRECISION_ENV, "tf32x3")


def require_cuda() -> bool:
    """Return True if CUDA is available, else print guidance and return False."""
    import torch

    if not torch.cuda.is_available():
        print(
            "CUDA is not available. The fused MLP kernels require a CUDA device; "
            "run this script on a GPU (e.g. RTX 3000)."
        )
        return False
    return True


def import_fused_mlp():
    """Import and return ``(fused_mlp, Activation)``.

    Raises on failure; callers should translate to a friendly message/exit code.
    """
    from physicsnemo.nn.functional.fused_mlp import Activation, fused_mlp

    return fused_mlp, Activation


# ---------------------------------------------------------------------------
# Reference implementation.
# ---------------------------------------------------------------------------


def torch_activation(x, activation, Activation):  # noqa: ANN001
    """Apply the torch equivalent of a fused-kernel ``Activation``."""
    import torch.nn.functional as F

    if activation == Activation.RELU:
        return F.relu(x)
    if activation == Activation.LEAKY_RELU:
        # Matches the kernel's leaky_relu slope of 0.01.
        return F.leaky_relu(x, negative_slope=0.01)
    if activation == Activation.SILU:
        return F.silu(x)
    return x


def reference_mlp(x, weights, biases, activation, last_activation, Activation):  # noqa: ANN001
    """Plain PyTorch reference matching the fused kernel's activation schedule.

    The fused kernels apply the activation after every layer except the final
    one, and apply it to the final layer only when ``last_activation`` is True.
    """
    import torch.nn.functional as F

    out = x
    n_layers = len(weights)
    for i, (weight, bias) in enumerate(zip(weights, biases)):
        out = F.linear(out, weight, bias)
        is_last = i == n_layers - 1
        if (not is_last) or last_activation:
            out = torch_activation(out, activation, Activation)
    return out


# ---------------------------------------------------------------------------
# Parameter construction.
# ---------------------------------------------------------------------------


@dataclass
class Params:
    """A set of MLP weights/biases plus a matching input tensor."""

    x: object
    weights: list
    biases: list


def make_params(M, in_features, widths, dtype, device, bias, seed):  # noqa: ANN001
    """Build random input/weights/biases for a tall-skinny MLP.

    ``widths`` lists the output size of each layer; weights follow the PyTorch
    ``nn.Linear`` layout of shape ``[out_features, in_features]``.
    """
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(M, in_features, dtype=dtype, device=device, generator=generator)

    weights = []
    biases = []
    prev = in_features
    for w_out in widths:
        # Xavier-ish scaling keeps activations in a sane range for many layers.
        scale = (2.0 / (prev + w_out)) ** 0.5
        weight = (
            torch.randn(w_out, prev, dtype=dtype, device=device, generator=generator)
            * scale
        )
        weights.append(weight)
        if bias:
            biases.append(
                torch.randn(w_out, dtype=dtype, device=device, generator=generator)
                * 0.1
            )
        else:
            biases.append(None)
        prev = w_out
    return Params(x=x, weights=weights, biases=biases)


def clone_leaf(tensor, requires_grad):  # noqa: ANN001
    """Return an independent leaf clone (optionally requiring grad)."""
    if tensor is None:
        return None
    leaf = tensor.clone().detach()
    leaf.requires_grad_(requires_grad)
    return leaf


# ---------------------------------------------------------------------------
# Error metrics.
# ---------------------------------------------------------------------------


def error_metrics(a, b):  # noqa: ANN001
    """Return (max_abs_err, max_rel_err) between two tensors."""
    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item()
    denom = b.float().abs().max().item()
    max_rel = max_abs / denom if denom > 0 else max_abs
    return max_abs, max_rel


def grad_metrics(a, b, rtol):  # noqa: ANN001
    """Return (max_abs, max_rel, n_bad, n_total) for a gradient comparison.

    ``n_bad`` counts elements whose error exceeds the tolerance band (scaled by
    the global magnitude of this tensor's reference). Returning raw counts lets
    the caller pool the bad fraction across *all* gradient tensors, so a couple
    of legitimately ambiguous ReLU/LeakyReLU kink flips landing in a tiny
    ``grad_bias`` (e.g. 2 of 64 = 3.1%) do not dominate the per-comparison
    verdict, while a systematic error (which corrupts the large ``grad_input``)
    still drives the pooled fraction high.
    """
    diff = (a.float() - b.float()).abs()
    max_abs = diff.max().item()
    denom = b.float().abs().max().item()
    max_rel = max_abs / denom if denom > 0 else max_abs
    threshold = rtol * denom if denom > 0 else rtol
    n_bad = int((diff > threshold).sum().item())
    n_total = diff.numel()
    return max_abs, max_rel, n_bad, n_total
