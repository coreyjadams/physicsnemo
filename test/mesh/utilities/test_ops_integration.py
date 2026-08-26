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

"""Parity tests: physicsnemo_ops-accelerated paths vs pure-torch fallbacks.

These tests are meaningful only when ``physicsnemo_ops`` is installed; they
skip cleanly otherwise (which is the CI default, where the fallback paths are
exercised by the rest of the suite).
"""

import contextlib

import pytest
import torch

pytest.importorskip("physicsnemo_ops")

from physicsnemo.mesh.utilities import _scatter_ops as so  # noqa: E402
from physicsnemo.utils import _physicsnemo_ops as gate  # noqa: E402
from physicsnemo.utils._index_tuple_ops import unique_index_tuples  # noqa: E402


@contextlib.contextmanager
def ops_disabled(monkeypatch):
    """Temporarily force the pure-torch fallback paths."""
    monkeypatch.setenv(gate._ENV_DISABLE, "1")
    gate._reset_cache()
    try:
        yield
    finally:
        monkeypatch.delenv(gate._ENV_DISABLE, raising=False)
        gate._reset_cache()


@pytest.fixture(autouse=True)
def reset_gate(monkeypatch):
    # Opt CPU tensors into the accelerated kernels so parity is exercised on
    # both devices (the production default is CUDA-only).
    monkeypatch.setenv(gate._ENV_ENABLE_CPU, "1")
    gate._reset_cache()
    yield
    gate._reset_cache()


def _both_paths(monkeypatch, fn, *args, **kwargs):
    """Run ``fn`` with ops enabled and disabled; return (ops_out, fallback_out)."""
    gate._reset_cache()
    if gate.physicsnemo_ops_torch() is None:
        pytest.skip("physicsnemo_ops import failed at runtime")
    with_ops = fn(*args, **kwargs)
    with ops_disabled(monkeypatch):
        without_ops = fn(*args, **kwargs)
    return with_ops, without_ops


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("trailing", [(), (3,), (2, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
def test_scatter_sum_coo_parity(monkeypatch, device, trailing, dtype):
    g = torch.Generator().manual_seed(0)
    n_src, n_dst = 257, 33
    if dtype.is_floating_point:
        src = torch.randn((n_src, *trailing), generator=g).to(
            device=device, dtype=dtype
        )
    else:
        src = torch.randint(-50, 50, (n_src, *trailing), generator=g).to(device)
    index = torch.randint(0, n_dst, (n_src,), generator=g).to(device)

    a, b = _both_paths(monkeypatch, so.scatter_sum_coo, src, index, n_dst)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    assert a.dtype == src.dtype and a.shape == (n_dst, *trailing)


@pytest.mark.parametrize("device", DEVICES)
def test_scatter_sum_coo_init_and_empty(monkeypatch, device):
    g = torch.Generator().manual_seed(1)
    src = torch.randn((64, 3), generator=g).to(device)
    index = torch.randint(0, 10, (64,), generator=g).to(device)
    init = torch.randn((10, 3), generator=g).to(device)

    a, b = _both_paths(monkeypatch, so.scatter_sum_coo, src, index, 10, init=init)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)

    empty_src = torch.empty((0, 3), device=device)
    empty_idx = torch.empty((0,), dtype=torch.long, device=device)
    a, b = _both_paths(monkeypatch, so.scatter_sum_coo, empty_src, empty_idx, 5)
    torch.testing.assert_close(a, b)
    assert a.shape == (5, 3)


@pytest.mark.parametrize("device", DEVICES)
def test_scatter_sum_coo_autograd(monkeypatch, device):
    src = torch.randn((32, 2), device=device, dtype=torch.float64, requires_grad=True)
    index = torch.randint(0, 6, (32,), device=device)

    def run():
        out = so.scatter_sum_coo(src, index, 6)
        (grad,) = torch.autograd.grad(out.sum(), src, create_graph=False)
        return grad

    a, b = _both_paths(monkeypatch, run)
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("op", ["min", "max"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
def test_scatter_min_max_coo_parity(monkeypatch, device, op, dtype):
    g = torch.Generator().manual_seed(2)
    fn = so.scatter_min_coo if op == "min" else so.scatter_max_coo
    n_src, n_dst = 300, 40  # some destinations receive nothing -> keep init
    if dtype.is_floating_point:
        src = torch.randn((n_src,), generator=g).to(device=device, dtype=dtype)
        init_fill = float("inf") if op == "min" else float("-inf")
        init = torch.full((n_dst,), init_fill, dtype=dtype, device=device)
    else:
        src = torch.randint(-100, 100, (n_src,), generator=g).to(device)
        init = torch.full(
            (n_dst,),
            torch.iinfo(dtype).max if op == "min" else torch.iinfo(dtype).min,
            dtype=dtype,
            device=device,
        )
    index = torch.randint(0, n_dst, (n_src,), generator=g).to(device)

    a, b = _both_paths(monkeypatch, fn, src, index, n_dst, init=init)
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
def test_scatter_min_max_coo_multidim(monkeypatch, device):
    g = torch.Generator().manual_seed(3)
    src = torch.randn((128, 3), generator=g).to(device)
    index = torch.randint(0, 12, (128,), generator=g).to(device)
    init = torch.full((12, 3), float("inf"), device=device)
    a, b = _both_paths(monkeypatch, so.scatter_min_coo, src, index, 12, init=init)
    torch.testing.assert_close(a, b)


def _random_offsets(n_segments, n_elements, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    # Random split with some empty segments
    cuts = torch.sort(
        torch.randint(0, n_elements + 1, (n_segments - 1,), generator=g)
    ).values
    offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.long),
            cuts,
            torch.full((1,), n_elements, dtype=torch.long),
        ]
    )
    return offsets.to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["sum", "mean", "min", "max"])
@pytest.mark.parametrize("trailing", [(), (3,)])
def test_segment_csr_parity(monkeypatch, device, kind, trailing):
    fn = {
        "sum": so.segment_sum_csr,
        "mean": so.segment_mean_csr,
        "min": so.segment_min_csr,
        "max": so.segment_max_csr,
    }[kind]
    g = torch.Generator().manual_seed(4)
    n_elements, n_segments = 500, 37
    values = torch.randn((n_elements, *trailing), generator=g).to(device)
    offsets = _random_offsets(n_segments, n_elements, device, seed=5)

    a, b = _both_paths(monkeypatch, fn, values, offsets)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    assert a.shape == (n_segments, *trailing)


@pytest.mark.parametrize("device", DEVICES)
def test_segment_csr_empty_segments_are_zero(monkeypatch, device):
    values = torch.randn((6, 2), device=device)
    # segments: [0:6), [6:6) empty, [6:6) empty
    offsets = torch.tensor([0, 6, 6, 6], device=device)
    for fn in (
        so.segment_sum_csr,
        so.segment_mean_csr,
        so.segment_min_csr,
        so.segment_max_csr,
    ):
        a, b = _both_paths(monkeypatch, fn, values, offsets)
        torch.testing.assert_close(a, b)
        assert (a[1:] == 0).all(), f"{fn.__name__} empty segments must be 0"


@pytest.mark.parametrize("device", DEVICES)
def test_segment_csr_autograd(monkeypatch, device):
    values = torch.randn(
        (50, 3), device=device, dtype=torch.float64, requires_grad=True
    )
    offsets = _random_offsets(8, 50, device, seed=6)

    for fn in (so.segment_sum_csr, so.segment_mean_csr):

        def run(fn=fn):
            out = fn(values, offsets)
            (grad,) = torch.autograd.grad(out.sum(), values)
            return grad

        a, b = _both_paths(monkeypatch, run)
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("n_columns", [1, 2, 3, 4])
def test_unique_index_tuples_parity(monkeypatch, device, dtype, n_columns):
    g = torch.Generator().manual_seed(7)
    bound = 50
    rows = torch.randint(0, bound, (400, n_columns), generator=g, dtype=dtype).to(
        device
    )

    def run():
        return unique_index_tuples(rows, bound, return_inverse=True, return_counts=True)

    (ua, ia, ca), (ub, ib, cb) = _both_paths(monkeypatch, run)
    torch.testing.assert_close(ua, ub)
    torch.testing.assert_close(ia, ib)
    torch.testing.assert_close(ca, cb)
    assert ua.dtype == rows.dtype


@pytest.mark.parametrize("device", DEVICES)
def test_find_edges_in_reference_parity(monkeypatch, device):
    from physicsnemo.mesh.utilities._edge_lookup import find_edges_in_reference

    g = torch.Generator().manual_seed(8)
    n_pts = 200
    ref = torch.randint(0, n_pts, (300, 2), generator=g).to(device)
    ref = ref[ref[:, 0] != ref[:, 1]]
    # Drop duplicate undirected edges so "which reference index" is unique.
    canonical = torch.sort(ref, dim=1).values
    keys = canonical[:, 0] * n_pts + canonical[:, 1]
    _, inverse = torch.unique(keys, return_inverse=True)
    first_occurrence = torch.zeros_like(keys, dtype=torch.bool)
    first_occurrence[
        torch.zeros(inverse.max() + 1, dtype=torch.long, device=device).scatter_reduce_(
            0,
            inverse,
            torch.arange(len(keys), device=device),
            reduce="amin",
            include_self=False,
        )
    ] = True
    ref = ref[first_occurrence]

    queries = torch.cat(
        [ref[:50].flip(1), torch.randint(0, n_pts, (60, 2), generator=g).to(device)]
    )

    def run():
        return find_edges_in_reference(ref, queries, index_bound=n_pts)

    (idx_a, m_a), (idx_b, m_b) = _both_paths(monkeypatch, run)
    torch.testing.assert_close(m_a, m_b)
    # Matched indices must agree exactly (reference has unique undirected rows).
    torch.testing.assert_close(idx_a[m_a], idx_b[m_b])
