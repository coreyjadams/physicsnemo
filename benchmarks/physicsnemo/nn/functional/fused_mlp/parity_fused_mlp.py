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

"""Correctness/parity checks for the Triton fused MLP.

Verifies forward (1/2/3-layer) and backward (1/2-layer) parity of
:func:`physicsnemo.nn.functional.fused_mlp.fused_mlp` against a plain PyTorch
reference. The fused three-layer backward is not implemented yet, so backward
parity is reported for one- and two-layer MLPs only.

This is split out from the benchmark so you can verify correctness without
paying for the (much longer) timing sweep. See ``benchmark_fused_mlp.py`` for
the performance harness.

Examples
--------
Default tolerances (tf32x3)::

    python benchmarks/physicsnemo/nn/functional/fused_mlp/parity_fused_mlp.py

Tightest tolerance (forces IEEE matmul precision)::

    python benchmarks/physicsnemo/nn/functional/fused_mlp/parity_fused_mlp.py \
        --precision ieee
"""

from __future__ import annotations

import argparse

from _fused_mlp_harness import (
    PRECISION_ENV,
    add_common_args,
    clone_leaf,
    current_precision,
    error_metrics,
    grad_metrics,
    import_fused_mlp,
    make_params,
    reference_mlp,
    require_cuda,
    set_precision,
)


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the parity harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    return parser.parse_args()


def run_parity(args, fused_mlp, Activation) -> bool:  # noqa: ANN001
    """Run forward/backward parity checks. Returns True if all pass."""
    import torch

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")

    # Use a true-fp32 reference. The fused kernel sets its matmul precision
    # explicitly via tl.dot(input_precision=...), so it is unaffected by this
    # flag; but the PyTorch reference would otherwise run in tf32 (~1e-3 error),
    # which is *less* accurate than the kernel's tf32x3 and produces a noisy
    # pre-activation that spuriously flips the ReLU/LeakyReLU subgradient at the
    # kink. A true-fp32 reference makes this an honest gold-standard comparison.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Loose tolerances for the default tf32x3 path; tight for ieee.
    precision = current_precision()
    if precision == "ieee":
        fwd_tol, bwd_tol = 1e-4, 1e-3
    else:
        fwd_tol, bwd_tol = 5e-3, 2e-2
    if dtype in (torch.float16, torch.bfloat16):
        fwd_tol, bwd_tol = 5e-2, 1e-1

    print("\n" + "=" * 78)
    print(f"PARITY  (dtype={args.dtype}, {PRECISION_ENV}={precision})")
    print(f"        forward tol={fwd_tol:g}, backward tol={bwd_tol:g}")
    print("=" * 78)

    # Small but non-trivial shapes; M kept modest so autograd reference is cheap.
    M, in_features = 4096, 48
    width_by_layers = {1: [32], 2: [64, 32], 3: [96, 64, 32]}
    activations = [
        Activation.NONE,
        Activation.RELU,
        Activation.LEAKY_RELU,
        Activation.SILU,
    ]

    # A few elements may legitimately flip at a ReLU/LeakyReLU kink; allow a
    # tiny fraction of such elements before failing a backward comparison.
    bad_frac_tol = 2e-3

    all_ok = True
    header = (
        f"{'layers':>6} {'act':>11} {'last':>5} {'bias':>5} {'phase':>8} "
        f"{'max_abs':>11} {'max_rel':>11} {'bad%':>8}  result"
    )
    print(header)
    print("-" * len(header))

    for n_layers in sorted(set(args.layers)):
        widths = width_by_layers[n_layers]
        for activation in activations:
            for last_activation in (False, True):
                for bias in (True, False):
                    base = make_params(
                        M, in_features, widths, dtype, device, bias, args.seed
                    )

                    # Independent leaf tensors for each path.
                    xf = clone_leaf(base.x, True)
                    wf = [clone_leaf(w, True) for w in base.weights]
                    bf = [clone_leaf(b, True) for b in base.biases]

                    xr = clone_leaf(base.x, True)
                    wr = [clone_leaf(w, True) for w in base.weights]
                    br = [clone_leaf(b, True) for b in base.biases]

                    yf = fused_mlp(xf, wf, bf, activation, last_activation)
                    yr = reference_mlp(
                        xr, wr, br, activation, last_activation, Activation
                    )

                    max_abs, max_rel = error_metrics(yf, yr)
                    ok = max_rel <= fwd_tol
                    all_ok = all_ok and ok
                    print(
                        f"{n_layers:>6} {activation.name:>11} {str(last_activation):>5} "
                        f"{str(bias):>5} {'forward':>8} {max_abs:>11.3e} {max_rel:>11.3e} "
                        f"{'-':>8}  {'PASS' if ok else 'FAIL'}"
                    )

                    # Backward parity (1- and 2-layer only; 3-layer bwd deferred).
                    if n_layers >= 3:
                        continue
                    grad_seed = torch.Generator(device=device).manual_seed(
                        args.seed + 1
                    )
                    upstream = torch.randn(
                        *yr.shape, dtype=dtype, device=device, generator=grad_seed
                    )
                    yf.backward(upstream)
                    yr.backward(upstream)

                    worst_abs, worst_rel = 0.0, 0.0
                    total_bad, total_count = 0, 0
                    pairs = [(xf.grad, xr.grad)]
                    pairs += [(w.grad, wr_.grad) for w, wr_ in zip(wf, wr)]
                    pairs += [
                        (b.grad, br_.grad)
                        for b, br_ in zip(bf, br)
                        if b is not None and br_ is not None
                    ]
                    for ga, gb in pairs:
                        if ga is None or gb is None:
                            continue
                        a_abs, a_rel, n_bad, n_total = grad_metrics(ga, gb, bwd_tol)
                        worst_abs = max(worst_abs, a_abs)
                        worst_rel = max(worst_rel, a_rel)
                        total_bad += n_bad
                        total_count += n_total

                    # Pool the bad-element fraction across all gradient tensors.
                    pooled_frac = total_bad / total_count if total_count else 0.0

                    # Pass if the worst-case relative error is within tolerance,
                    # or only a tiny pooled fraction of elements miss (kink flips).
                    ok_b = (worst_rel <= bwd_tol) or (pooled_frac <= bad_frac_tol)
                    all_ok = all_ok and ok_b
                    print(
                        f"{n_layers:>6} {activation.name:>11} {str(last_activation):>5} "
                        f"{str(bias):>5} {'backward':>8} {worst_abs:>11.3e} {worst_rel:>11.3e} "
                        f"{pooled_frac * 100:>7.3f}%  {'PASS' if ok_b else 'FAIL'}"
                    )

    print("-" * len(header))
    print(f"PARITY {'PASSED' if all_ok else 'FAILED'}")
    return all_ok


def main() -> int:
    """Entry point: set precision, import kernels, run parity checks."""
    args = _parse_args()

    # Must set precision before importing the fused kernels.
    set_precision(args.precision)

    import torch

    if not require_cuda():
        return 1

    torch.manual_seed(args.seed)

    try:
        fused_mlp, Activation = import_fused_mlp()
    except ImportError as exc:  # pragma: no cover - environment dependent
        print(f"Failed to import fused_mlp kernels: {exc}")
        return 1

    print(f"Device: {torch.cuda.get_device_name(0)}")

    try:
        parity_ok = run_parity(args, fused_mlp, Activation)
    except Exception as exc:  # noqa: BLE001
        # The most common first-run failure is a Triton autotune config the
        # installed Triton rejects (e.g. num_stages=0). Surface a hint.
        print(f"\nERROR while running: {type(exc).__name__}: {exc}")
        print(
            "\nHint: if this is a Triton compilation error, check the autotune "
            "configs in physicsnemo/nn/functional/fused_mlp/utils.py -- some "
            "Triton versions reject num_stages=0 (use num_stages>=1)."
        )
        raise

    return 0 if parity_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
