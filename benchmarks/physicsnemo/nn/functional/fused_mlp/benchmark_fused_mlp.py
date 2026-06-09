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

"""Performance benchmark for the Triton fused MLP.

Benchmarks forward and forward+backward latency of the fused kernel against
torch eager and ``torch.compile`` across the tall-skinny regime that motivates
the fusion (very large batch dimension ``M``, small feature widths).

Correctness is *not* checked here -- run ``parity_fused_mlp.py`` for that. This
keeps the benchmark fast to iterate on without re-running parity every time.

The fused three-layer backward is not implemented yet, so backward timings are
reported for one- and two-layer MLPs only.

Examples
--------
Full default sweep::

    python benchmarks/physicsnemo/nn/functional/fused_mlp/benchmark_fused_mlp.py

Custom batch/width sweep (drop the out-of-envelope width=256)::

    python benchmarks/physicsnemo/nn/functional/fused_mlp/benchmark_fused_mlp.py \
        --batch-sizes 1000000 4000000 --widths 64 128 --layers 2 3
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from typing import Callable, Optional

from _fused_mlp_harness import (
    PRECISION_ENV,
    add_common_args,
    current_precision,
    import_fused_mlp,
    make_params,
    reference_mlp,
    require_cuda,
    set_precision,
)


def _resource_error_types() -> tuple:
    """Triton out-of-resources error types, tolerant of Triton version layout.

    A wide layer can exceed the device's shared-memory budget for every autotune
    config (e.g. the 3-layer middle weight is loaded whole), in which case Triton
    raises an out-of-resources error. We catch it so the benchmark sweep marks
    that shape ``n/a`` and continues instead of aborting.
    """
    types: list = []
    try:
        from triton.runtime.errors import OutOfResources

        types.append(OutOfResources)
    except Exception:  # noqa: BLE001 - older/newer Triton may move this
        pass
    try:
        from triton.runtime.autotuner import OutOfResources as _AutoOOR

        types.append(_AutoOOR)
    except Exception:  # noqa: BLE001
        pass
    return tuple(types)


_RESOURCE_ERRORS = _resource_error_types()


def _is_cuda_oom(exc: BaseException) -> bool:
    """True if ``exc`` is a CUDA out-of-memory error in any of its disguises.

    A global-memory OOM can surface either as ``torch.OutOfMemoryError`` (from a
    PyTorch allocation) or as a plain ``RuntimeError`` whose message contains
    "out of memory" (Triton wraps the CUDA driver error this way during kernel
    launch/autotune). Both mean "this config doesn't fit", so we skip it rather
    than abort the whole sweep; any other ``RuntimeError`` is a real bug and
    should propagate.
    """
    return "out of memory" in str(exc).lower()


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments for the benchmark harness."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[262_144, 1_048_576, 4_194_304],
        help="Batch (row, M) sizes to sweep in the benchmark.",
    )
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=[64, 128, 256],
        help="Hidden/feature widths to sweep in the benchmark.",
    )
    parser.add_argument(
        "--iters", type=int, default=50, help="Timed iterations per measurement."
    )
    parser.add_argument(
        "--warmup", type=int, default=15, help="Warmup iterations per measurement."
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=["forward", "fwd+bwd"],
        default=["forward", "fwd+bwd"],
        help="Which phases to benchmark. Use '--phases forward' for forward only.",
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help=(
            "Print a full per-implementation distribution block (median/mean/"
            "min/p90/std + TFLOP/s + GB/s) for every config, instead of the "
            "compact one-line-per-config table."
        ),
    )
    return parser.parse_args()


@dataclass
class Timing:
    """Per-call timing distribution in milliseconds."""

    median: float
    mean: float
    min: float
    max: float
    std: float
    p10: float
    p90: float
    n: int

    @property
    def std_pct(self) -> float:
        """Std as a percentage of the median (measurement stability)."""
        return 100.0 * self.std / self.median if self.median > 0 else 0.0


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list (q in [0, 1])."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def _summarize(samples: list[float]) -> "Timing":
    """Build a Timing from raw per-call millisecond samples."""
    ordered = sorted(samples)
    return Timing(
        median=statistics.median(ordered),
        mean=statistics.fmean(ordered),
        min=ordered[0],
        max=ordered[-1],
        std=statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        p10=_percentile(ordered, 0.10),
        p90=_percentile(ordered, 0.90),
        n=len(ordered),
    )


def _time_fns_interleaved(
    fns: "dict[str, Callable[[], None]]", iters: int, warmup: int
) -> "dict[str, Timing]":
    """Time several callables round-robin so they share thermal/clock state.

    Timing each implementation in a separate block (all eager, then all compile,
    then all fused) biases the comparison on a thermally-throttling laptop: the
    GPU is hottest -- and clocked lowest -- by the time the last block runs, so
    whichever impl is measured last is unfairly penalized. Interleaving records
    one timed call of every impl per iteration, so a clock excursion hits all of
    them on (nearly) the same iteration and cancels out of the *ratio* we care
    about. Returns a Timing per key. Callables that are ``None`` are skipped.
    """
    import torch

    active = {name: fn for name, fn in fns.items() if fn is not None}

    # Warmup every impl (absorbs autotune/compile/allocator one-time costs).
    for fn in active.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()

    events = {
        name: (
            [torch.cuda.Event(enable_timing=True) for _ in range(iters)],
            [torch.cuda.Event(enable_timing=True) for _ in range(iters)],
        )
        for name in active
    }

    for i in range(iters):
        for name, fn in active.items():
            starts, ends = events[name]
            starts[i].record()
            fn()
            ends[i].record()
    torch.cuda.synchronize()

    return {
        name: _summarize(
            [s.elapsed_time(e) for s, e in zip(starts, ends)]
        )
        for name, (starts, ends) in events.items()
    }


def _flops_and_bytes(phase, n_layers, M, width, itemsize):  # noqa: ANN001
    """Return (matmul FLOPs, ideal DRAM bytes) for one call of this config.

    Shapes are square at ``width`` (in_features == width). FLOPs count the
    multiply-adds in the matmuls (2*M*K*N per layer). ``bytes`` is an optimistic
    lower bound on DRAM traffic for the *fused* path -- inputs/outputs plus
    weights, with intermediate activations assumed to stay on chip -- so the
    derived GB/s is a roofline reference, not the exact traffic of each impl.
    """
    # Per-layer square matmul: K == N == width.
    fwd_flops = n_layers * 2 * M * width * width
    # Backward computes grad_input (gO @ W) and grad_weight (gO^T @ X), each the
    # same FLOP cost as the forward matmul -> ~2x forward on top of the forward.
    flops = fwd_flops if phase == "forward" else fwd_flops * 3

    weight_elems = n_layers * width * width
    # Forward: read input (M*width), write output (M*width), read weights.
    fwd_bytes = (2 * M * width + weight_elems) * itemsize
    if phase == "forward":
        nbytes = fwd_bytes
    else:
        # Backward additionally streams grad_output and grad_input (M*width each)
        # and writes grad_weight/grad_bias (~weights).
        nbytes = fwd_bytes + (2 * M * width + weight_elems + n_layers * width) * itemsize
    return flops, nbytes


def _throughput(t: Optional[Timing], flops: int, nbytes: int):  # noqa: ANN001
    """Return (TFLOP/s, GB/s) from a median time, or (None, None)."""
    if t is None or t.median <= 0:
        return None, None
    seconds = t.median / 1e3
    return flops / seconds / 1e12, nbytes / seconds / 1e9


def _make_callables(
    params, activation, last_activation, fused_mlp, Activation, compiled_cache, cache_key
):  # noqa: ANN001
    """Build forward callables for fused, eager, and compiled reference.

    The compiled reference is cached in ``compiled_cache`` under ``cache_key``
    (``(n_layers, width)``) and reused across batch sizes and the forward /
    forward+backward phases. Weights and biases are passed as *arguments* to the
    compiled function (not captured in its closure) so the same compiled object
    can be safely reused with both the no-grad tensors of the forward phase and
    the grad-tracked tensors of the backward phase; Dynamo's automatic-dynamic
    handling then reuses the graph across batch sizes instead of recompiling per
    M.
    """
    import torch

    def fused_fwd():
        return fused_mlp(
            params.x, params.weights, params.biases, activation, last_activation
        )

    def eager_fwd():
        return reference_mlp(
            params.x,
            params.weights,
            params.biases,
            activation,
            last_activation,
            Activation,
        )

    compiled = compiled_cache.get(cache_key)
    if compiled is None:
        compiled = torch.compile(
            lambda x, weights, biases: reference_mlp(
                x, weights, biases, activation, last_activation, Activation
            )
        )
        compiled_cache[cache_key] = compiled

    def compiled_fwd():
        return compiled(params.x, params.weights, params.biases)

    return fused_fwd, eager_fwd, compiled_fwd


def _make_bwd_callable(forward_fn, upstream):  # noqa: ANN001
    """Wrap a forward callable so each call also runs backward."""

    def step():
        out = forward_fn()
        out.backward(upstream)
        return out

    return step


def _print_compact_row(
    phase, n_layers, M, width, flops, nbytes, eager_t, compiled_t, fused_t,
    fused_note="not implemented",
):  # noqa: ANN001
    """Print one line per config: baseline medians + fused detail + throughput."""
    tflops, gbs = _throughput(fused_t, flops, nbytes)
    if fused_t is None:
        fused_str = f"{'n/a':>11}"
        f_min = f"{'-':>8}"
        f_p90 = f"{'-':>8}"
        f_std = f"{'-':>7}"
        tflops_str = f"{'-':>8}"
        gbs_str = f"{'-':>8}"
        vs_eager = f"{'-':>7}"
        vs_comp = f"{'-':>7}"
    else:
        fused_str = f"{fused_t.median:>11.3f}"
        f_min = f"{fused_t.min:>8.3f}"
        f_p90 = f"{fused_t.p90:>8.3f}"
        f_std = f"{fused_t.std_pct:>6.1f}%"
        tflops_str = f"{tflops:>8.2f}"
        gbs_str = f"{gbs:>8.1f}"
        vs_eager = f"{eager_t.median / fused_t.median:>6.2f}x"
        vs_comp = f"{compiled_t.median / fused_t.median:>6.2f}x"

    print(
        f"{phase:>9} {n_layers:>6} {M:>10} {width:>6} "
        f"{eager_t.median:>11.3f} {compiled_t.median:>11.3f} {fused_str} "
        f"{f_min} {f_p90} {f_std} "
        f"{tflops_str} {gbs_str} {vs_eager} {vs_comp}"
    )


def _print_detailed_block(
    phase, n_layers, M, width, flops, nbytes, eager_t, compiled_t, fused_t,
    fused_note="not implemented",
):  # noqa: ANN001
    """Print a full per-implementation distribution block for one config."""
    print(
        f"\n[{phase}] layers={n_layers}  M={M:,}  width={width}  "
        f"FLOPs={flops / 1e9:.2f} G  ideal_traffic={nbytes / 1e9:.3f} GB"
    )
    sub = (
        f"   {'impl':>8} {'median':>9} {'mean':>9} {'min':>9} {'p10':>9} "
        f"{'p90':>9} {'max':>9} {'std%':>6} {'TFLOP/s':>8} {'GB/s':>8}"
    )
    print(sub)
    print("   " + "-" * (len(sub) - 3))
    for name, t in (("eager", eager_t), ("compile", compiled_t), ("fused", fused_t)):
        if t is None:
            print(f"   {name:>8} {('n/a (' + fused_note + ')'):>9}")
            continue
        tflops, gbs = _throughput(t, flops, nbytes)
        print(
            f"   {name:>8} {t.median:>9.3f} {t.mean:>9.3f} {t.min:>9.3f} "
            f"{t.p10:>9.3f} {t.p90:>9.3f} {t.max:>9.3f} {t.std_pct:>5.1f}% "
            f"{tflops:>8.2f} {gbs:>8.1f}"
        )
    if fused_t is not None:
        print(
            f"   speedup: fused vs eager "
            f"{eager_t.median / fused_t.median:.2f}x, "
            f"vs compile {compiled_t.median / fused_t.median:.2f}x"
        )


def run_benchmark(args, fused_mlp, Activation) -> None:  # noqa: ANN001
    """Run forward and forward+backward benchmarks across the sweep."""
    import torch

    dtype = getattr(torch, args.dtype)
    device = torch.device("cuda")
    activation = Activation.SILU
    last_activation = False

    # Allow tf32 for the eager/compiled reference so the timing comparison is
    # representative of how these baselines are typically run.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    width_by_layers = {
        1: lambda w: [w],
        2: lambda w: [w, w],
        3: lambda w: [w, w, w],
    }

    precision = current_precision()
    print("\n" + "=" * 78)
    print(
        f"BENCHMARK  (dtype={args.dtype}, {PRECISION_ENV}={precision}, "
        f"act={activation.name}, last_activation={last_activation})"
    )
    print(f"           iters={args.iters}, warmup={args.warmup}")
    print("=" * 78)

    itemsize = torch.empty((), dtype=dtype).element_size()

    header = (
        f"{'phase':>9} {'layers':>6} {'M':>10} {'width':>6} "
        f"{'eager(ms)':>11} {'comp(ms)':>11} {'fused(ms)':>11} "
        f"{'f_min':>8} {'f_p90':>8} {'f_std%':>7} "
        f"{'TFLOP/s':>8} {'GB/s':>8} {'vs_eag':>7} {'vs_cmp':>7}"
    )

    # Compiled references are cached across phases and batch sizes (keyed by
    # (n_layers, width)) so torch.compile is not rebuilt for every row.
    compiled_cache: dict = {}

    for phase in args.phases:
        print(f"\n--- {phase} ---")
        if not args.detailed:
            print(header)
            print("-" * len(header))
        for n_layers in sorted(set(args.layers)):
            for M in args.batch_sizes:
                for width in args.widths:
                    widths = width_by_layers[n_layers](width)
                    # Per-config tensors can be huge at large M (e.g. M=4.19M,
                    # width=256 is a 4 GB output alone). Skip configs that don't
                    # fit instead of aborting the sweep, and free cached blocks
                    # between configs so earlier rows don't starve later ones.
                    try:
                        params = make_params(
                            M, width, widths, dtype, device, True, args.seed
                        )
                        fused_fwd, eager_fwd, compiled_fwd = _make_callables(
                            params,
                            activation,
                            last_activation,
                            fused_mlp,
                            Activation,
                            compiled_cache,
                            (n_layers, width),
                        )

                        if phase == "forward":
                            eager_fn, compiled_fn, fused_fn = (
                                eager_fwd,
                                compiled_fwd,
                                fused_fwd,
                            )
                            fused_supported = True
                        else:
                            # Backward needs requires_grad leaves; rebuild params.
                            params.x.requires_grad_(True)
                            for w in params.weights:
                                w.requires_grad_(True)
                            for b in params.biases:
                                if b is not None:
                                    b.requires_grad_(True)
                            out_shape = (M, widths[-1])
                            upstream = torch.randn(
                                out_shape, dtype=dtype, device=device
                            )
                            eager_fn = _make_bwd_callable(eager_fwd, upstream)
                            compiled_fn = _make_bwd_callable(compiled_fwd, upstream)
                            fused_fn = _make_bwd_callable(fused_fwd, upstream)
                            # Fused 3-layer backward is not implemented.
                            fused_supported = n_layers < 3

                        # Probe fused once before the timed run: triggers
                        # autotune/compile and surfaces NotImplemented / OOM
                        # cleanly, so the interleaved timer only times viable impls.
                        fused_t: Optional[Timing] = None
                        fused_note = "not implemented"
                        include_fused = fused_supported
                        if fused_supported:
                            try:
                                fused_fn()
                                torch.cuda.synchronize()
                            except NotImplementedError:
                                include_fused = False
                            except _RESOURCE_ERRORS:
                                # Every autotune config exceeded shared memory for
                                # this (wide) shape; report n/a and keep going.
                                include_fused = False
                                fused_note = "OOM shared mem"
                                torch.cuda.empty_cache()
                            except RuntimeError as exc:
                                # Global-memory OOM (e.g. partial-grad buffers +
                                # activations at wide M); skip fused, keep going.
                                if not _is_cuda_oom(exc):
                                    raise
                                include_fused = False
                                fused_note = "CUDA OOM"
                                torch.cuda.empty_cache()

                        # Interleave the impls so thermal/clock drift cancels out
                        # of the ratio (eager/compile/fused share each iteration).
                        fns_to_time = {"eager": eager_fn, "compile": compiled_fn}
                        if include_fused:
                            fns_to_time["fused"] = fused_fn
                        timings = _time_fns_interleaved(
                            fns_to_time, args.iters, args.warmup
                        )
                        eager_t = timings["eager"]
                        compiled_t = timings["compile"]
                        fused_t = timings.get("fused")

                        flops, nbytes = _flops_and_bytes(
                            phase, n_layers, M, width, itemsize
                        )

                        if args.detailed:
                            _print_detailed_block(
                                phase, n_layers, M, width, flops, nbytes,
                                eager_t, compiled_t, fused_t, fused_note,
                            )
                        else:
                            _print_compact_row(
                                phase, n_layers, M, width, flops, nbytes,
                                eager_t, compiled_t, fused_t, fused_note,
                            )
                    except RuntimeError as exc:
                        # torch.OutOfMemoryError subclasses RuntimeError; a Triton
                        # CUDA OOM is a bare RuntimeError. Skip either; re-raise
                        # anything that isn't an OOM (those are real failures).
                        if not _is_cuda_oom(exc):
                            raise
                        print(
                            f"\n[{phase}] layers={n_layers} M={M:,} width={width}"
                            "  skipped: CUDA OOM (config too large for this GPU)"
                        )
                    finally:
                        # Drop this config's tensors before the next one so the
                        # allocator cache doesn't accumulate across the sweep.
                        params = None
                        torch.cuda.empty_cache()


def main() -> int:
    """Entry point: set precision, import kernels, run benchmarks."""
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
        run_benchmark(args, fused_mlp, Activation)
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
