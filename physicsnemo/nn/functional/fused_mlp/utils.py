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

import math
import os

import triton
import triton.language as tl


def persistent_max_width(device, n_weights: int, itemsize: int = 4) -> int:
    """Largest square width whose weights stay resident in shared memory.

    The weight-stationary persistent kernels keep every weight resident in
    shared memory for the whole grid-stride loop, so the binding constraint is
    ``n_weights * width**2 * itemsize`` (one ``[K, N]`` copy per weight, plus a
    small ``BM=16`` input+hidden staging tile) fitting in the device's per-block
    shared-memory budget. We solve that for the largest ``width`` and route
    wider layers to the tiled kernel instead.

    Computing this from the device (rather than a hard-coded constant) means the
    persistent path automatically covers wider layers on data-center GPUs
    (164-228 KB/block) while staying conservative on consumer parts
    (~100 KB/block), with no per-GPU tuning. A 0.85 budget factor leaves head
    room for the autotuner's larger ``BM``/``num_stages`` configs; any config
    that still overflows is pruned by Triton at autotune time.

    Parameters
    ----------
    device : torch.device | int
        CUDA device whose shared-memory limit is queried.
    n_weights : int
        Number of weight matrices held resident (2 for a two-layer MLP, etc.).
    itemsize : int, optional
        Bytes per weight element (4 for float32/tf32), by default 4.

    Returns
    -------
    int
        Maximum square feature width that stays weight-resident on ``device``.
    """
    import torch

    smem = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    budget = 0.85 * smem
    # a*W^2 + b*W <= budget, with b the BM=16 (input+hidden) staging coefficient.
    a = n_weights * itemsize
    b = 2 * 16 * itemsize
    return int((-b + math.sqrt(b * b + 4 * a * budget)) / (2 * a))

# Upper bound on the bytes used by the per-bucket gradient *partial* buffers in
# the backward pass (see grad_partials). Each weight/bias gradient is normally
# accumulated with tl.atomic_add from every row-block program into a single
# small buffer -- with M/BLOCK_SIZE_M ~ thousands of programs all hitting the
# same [width, width] grad_weight, the atomics serialize and dominate the
# backward. Instead we replicate the gradient across NUM_PARTIALS slots, route
# program ``pid`` to ``pid % NUM_PARTIALS``, and reduce afterwards. More
# partials means less contention but more buffer memory plus a larger reduction
# pass; this caps that trade at a few tens of MB.
_GRAD_PARTIAL_MAX_BYTES = 32 * 1024 * 1024


def grad_partials(num_programs: int, slot_bytes: int) -> int:
    """Number of partial gradient buffers to split atomic accumulation across.

    The backward kernel accumulates weight/bias gradients with atomics. With one
    destination buffer, every one of ``num_programs`` row-block programs contends
    on the same handful of addresses, which serializes badly for the large-``M``
    tall-skinny regime. Splitting into ``P`` partials drops the per-address
    contention by ~``P`` at the cost of a ``[P, *grad_shape]`` buffer (reduced
    with a cheap ``sum(0)`` afterwards).

    ``P`` is clamped so the partial buffers never exceed
    :data:`_GRAD_PARTIAL_MAX_BYTES` and never exceed ``num_programs`` (more
    buckets than programs would only waste memory and reduction work). This makes
    ``P`` shrink automatically for wider layers (bigger per-slot footprint),
    where fewer partials still leave a large contention reduction.

    Parameters
    ----------
    num_programs : int
        Number of row-block programs in the backward launch (``cdiv(M, BM)``).
    slot_bytes : int
        Bytes for a single partial slot -- the summed footprint of every
        replicated gradient buffer (all requested weights and biases).

    Returns
    -------
    int
        Number of partial buckets ``P`` (>= 1).
    """
    if slot_bytes <= 0:
        return 1
    by_mem = _GRAD_PARTIAL_MAX_BYTES // slot_bytes
    return max(1, min(num_programs, by_mem))


PRECISION = tl.constexpr("tf32x3")
if "PHYSICSNEMO_FUSED_MM_PRECISION" in os.environ:
    PRECISION = os.environ["PHYSICSNEMO_FUSED_MM_PRECISION"]

    if PRECISION in ["tf32", "tf32x3", "ieee"]:
        PRECISION = tl.constexpr(PRECISION)
    else:
        raise ValueError(
            f"Invalid precision set to 'PHYSICSNEMO_FUSED_MM_PRECISION': {PRECISION}"
        )


def get_cuda_autotune_config() -> triton.Config:
    """
    Autotuning configuration for triton fused matmuls.

    """

    configs = [
        triton.Config(
            {"BLOCK_SIZE_M": BLOCK_SIZE_M},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for num_stages in [0, 1]
        for num_warps in [4, 8]
        for BLOCK_SIZE_M in [32, 64, 128]
    ]
    return configs


def get_cuda_autotune_config_bwd() -> list:
    """Autotuning configs for the fused-matmul *backward* kernels.

    Exposes ``BLOCK_SIZE_M`` (the row tile each program processes) alongside
    ``num_warps``/``num_stages``. The backward holds several
    ``[BLOCK_SIZE_M, width]`` activation tiles resident per program at once -- the
    recomputed forward blocks plus every requested gradient tensor -- so
    ``BLOCK_SIZE_M`` is the dominant knob on register pressure, and therefore on
    occupancy. In the bandwidth-bound tall-skinny regime a *small*
    ``BLOCK_SIZE_M`` typically wins: it keeps more blocks resident per SM, which
    is what lets the kernel hide memory latency instead of running at a fraction
    of peak bandwidth. The previous fixed ``BLOCK_SIZE_M=128`` left the kernel
    occupancy-starved. Re-tuned per problem shape via the kernel ``key``.
    """

    configs = []
    for num_warps in [4, 8]:
        for num_stages in [1, 2]:
            for BLOCK_SIZE_M in [16, 32, 64, 128]:
                configs.append(
                    triton.Config(
                        {"BLOCK_SIZE_M": BLOCK_SIZE_M},
                        num_stages=num_stages,
                        num_warps=num_warps,
                    )
                )
    return configs


def get_cuda_autotune_config_fwd_tiled() -> list:
    """Autotuning configs for the width-tiled forward kernels.

    Exposes three block-size knobs:

    - ``BLOCK_SIZE_M``  -- rows processed per program (the resident batch tile).
    - ``BLOCK_SIZE_KR`` -- contraction tile streamed in the first-layer reduction
      loop (the "K-reduce" tile).
    - ``BLOCK_SIZE_N``  -- output column tile emitted by the last layer.

    The set spans "no tiling" for narrow widths (``BLOCK_SIZE_N`` >= the feature
    width makes the output loop run once, and ``BLOCK_SIZE_KR`` = 64 covers the
    common <= 64 widths in a single reduction step, reproducing the original
    full-width kernel) through finer tiles that keep wide layers from spilling.
    Re-tuned per problem shape via the kernel ``key``.

    ``BLOCK_SIZE_KR`` is fixed at 64 (a good reduction-tile default). The other
    knobs are chosen to cover the occupancy/register trade-off that dominates
    wider layers: ``num_stages=1`` and small ``BLOCK_SIZE_M`` reduce register
    pressure (the resident hidden tile + accumulator scale with
    ``BLOCK_SIZE_M`` x width), which is what lets a wide layer keep enough blocks
    resident per SM to hide memory latency instead of spilling. Re-tuned per
    problem shape via the kernel ``key``; use a focused sweep (fewer widths /
    batch sizes) to keep cold-start autotuning time down.
    """

    configs = []
    for num_warps in [4, 8]:
        for num_stages in [1, 2]:
            for BLOCK_SIZE_M in [16, 32, 64, 128]:
                for BLOCK_SIZE_N in [64, 128]:
                    configs.append(
                        triton.Config(
                            {
                                "BLOCK_SIZE_M": BLOCK_SIZE_M,
                                "BLOCK_SIZE_KR": 64,
                                "BLOCK_SIZE_N": BLOCK_SIZE_N,
                            },
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
    return configs


def get_cuda_autotune_config_bwd_persistent() -> list:
    """Autotuning configs for the weight-stationary *persistent backward*.

    Like the persistent forward, the grid is ~one program per SM and only
    ``BLOCK_SIZE_M`` (the streamed row tile) is tuned. The backward is much
    heavier on shared memory than the forward: it keeps both weights resident,
    recomputes the forward, and transposes activation tiles for the gradient
    contractions, all on top of the register-resident ``grad_weight``
    accumulators. So this sweep is deliberately conservative -- small row tiles
    and shallow pipelines -- to keep the per-stage staging within the per-block
    budget for every admitted width.

    This must be self-limiting: the Triton in use does **not** reliably prune
    configs that exceed the shared-memory budget (it raises ``OutOfResources`` at
    launch instead of skipping), so every config listed here has to fit. Small
    ``BLOCK_SIZE_M`` is also what the bandwidth-bound backward wants for
    occupancy, so this costs little.

    ``num_stages`` is pinned to 1 and ``BLOCK_SIZE_M`` to 16 for shared-memory
    reasons. On top of the resident weights and the ``grad_weight`` accumulators
    (~2 * width^2 each), the per-iteration working set -- input/grad staging, the
    four activation transposes, the ``tl.dot`` operand staging, and the
    ``output_pre`` recompute -- scales with ``BLOCK_SIZE_M``; at ``BM=32`` that
    overflows a ~100 KB GPU at the widest admitted width, and a deeper pipeline
    double-buffers it. ``BM=16`` keeps the footprint in budget, and the
    grid-stride loop already exposes cross-iteration parallelism, so neither knob
    costs much. The launcher additionally falls back to the tiled backward if a
    config still overflows, but every config here is meant to fit.
    """

    return [
        triton.Config({"BLOCK_SIZE_M": 16}, num_stages=1, num_warps=num_warps)
        for num_warps in [4, 8]
    ]


def get_cuda_autotune_config_fwd_persistent() -> list:
    """Autotuning configs for the weight-stationary *persistent* forward kernels.

    The persistent kernels launch a fixed grid (~one program per SM) and walk
    many row-blocks in a grid-stride loop while keeping the (small) weights
    resident, so each weight is read from DRAM/L2 once per program instead of
    once per row-block. That eliminates the redundant weight streaming that
    dominates the multi-layer tall-skinny regime.

    Only ``BLOCK_SIZE_M`` (the streamed row tile), ``num_warps`` and
    ``num_stages`` are exposed -- the feature widths are held fully resident, so
    there is no ``BLOCK_SIZE_N``/``BLOCK_SIZE_KR`` to tile here. Small
    ``BLOCK_SIZE_M`` keeps the per-iteration activation footprint low (the
    resident weights already dominate the register budget), while a few
    ``num_stages`` options let the autotuner trade pipelining depth against
    register pressure. Re-tuned per problem shape via the kernel ``key``.
    """

    configs = []
    for num_warps in [4, 8]:
        for num_stages in [1, 2, 3]:
            for BLOCK_SIZE_M in [16, 32, 64, 128]:
                configs.append(
                    triton.Config(
                        {"BLOCK_SIZE_M": BLOCK_SIZE_M},
                        num_stages=num_stages,
                        num_warps=num_warps,
                    )
                )
    return configs