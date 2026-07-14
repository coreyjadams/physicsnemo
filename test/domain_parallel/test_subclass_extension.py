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

r"""Tests for the ShardTensor subclass extension points.

A ShardTensor subclass that needs to (a) carry an extra always-present inner
tensor and (b) attach a nested flatten context should be able to do so by
declaring ``_extra_inner_tensors`` and implementing the two subclass hooks —
*without* re-implementing ``__tensor_flatten__`` / ``__tensor_unflatten__`` /
``__metadata_guard__``. These tests exercise a minimal in-tree subclass through
flatten/unflatten round-trips and ``torch.compile`` to prove the base-class
plumbing carries the extra inner tensor and nested context losslessly.
"""

import pytest
import torch

from physicsnemo.domain_parallel import ShardTensor
from test.domain_parallel.test_redistribute import shard_tensor_factory


class _ExtInnerShardTensor(ShardTensor):
    """Minimal subclass that adds one extra inner tensor (``_aux``) and a nested
    flatten context (an opaque ``_payload``), using only the base extension
    points — it does not override the flatten protocol."""

    _extra_inner_tensors = ("_aux",)

    # Backing store for the extra inner tensor + its per-instance sentinel cache.
    _aux_v = None
    _aux_c = None
    _payload = 0

    @property
    def _aux(self) -> torch.Tensor:
        v = self._aux_v
        if v is not None:
            return v
        # Unset: return the stable per-instance sentinel the base helper mints.
        return self._stable_inner_sentinel("_aux_c")

    @_aux.setter
    def _aux(self, value) -> None:
        self._aux_v = value
        self._aux_c = None

    def __subclass_flatten_context__(self):
        return ("marker", self._payload)

    def __subclass_unflatten__(self, subclass_ctx) -> None:
        _marker, self._payload = subclass_ctx


def _as_subclass(mesh, *, uneven=True) -> _ExtInnerShardTensor:
    st = shard_tensor_factory(mesh, uneven=uneven)
    out = _ExtInnerShardTensor.__new__(
        _ExtInnerShardTensor,
        local_tensor=st._local_tensor,
        spec=st._spec,
        requires_grad=False,
    )
    return out


def run_flatten_includes_extra_inner_and_nested_context(mesh):
    st = _as_subclass(mesh)
    st._payload = 7

    inner_names, ctx = st.__tensor_flatten__()

    # The extra inner is declared and appended after ``_local_tensor``.
    assert inner_names == ["_local_tensor", "_aux"]
    # Context is nested: ``(base_ctx, subclass_ctx)``.
    (base_ctx, subclass_ctx) = ctx
    spec, requires_grad = base_ctx
    assert spec is st._spec
    assert subclass_ctx == ("marker", 7)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_flatten_includes_extra_inner_and_nested_context_1d(distributed_mesh):
    run_flatten_includes_extra_inner_and_nested_context(distributed_mesh)


def run_unflatten_roundtrip_is_lossless(mesh):
    st = _as_subclass(mesh)
    st._payload = 13
    st._aux = torch.arange(3, dtype=torch.int64, device=st._local_tensor.device)
    expected_full = st.full_tensor().clone()

    inner_names, ctx = st.__tensor_flatten__()
    inner = {name: getattr(st, name) for name in inner_names}
    rebuilt = type(st).__tensor_unflatten__(inner, ctx, st.shape, st.stride())

    # Reconstructed as the subclass (via ``cls`` — no __class__ reassignment).
    assert isinstance(rebuilt, _ExtInnerShardTensor)
    assert rebuilt.placements == st.placements
    assert rebuilt.shape == st.shape
    # Extra inner tensor and nested context both round-trip.
    torch.testing.assert_close(rebuilt._aux, st._aux)
    assert rebuilt._payload == 13
    torch.testing.assert_close(rebuilt.full_tensor(), expected_full)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_unflatten_roundtrip_is_lossless_1d(distributed_mesh):
    run_unflatten_roundtrip_is_lossless(distributed_mesh)


def run_metadata_guard_ignores_subclass_context(mesh):
    st = _as_subclass(mesh)
    _names, ctx = st.__tensor_flatten__()
    (base_ctx, _subclass_ctx) = ctx

    # Same (spec, requires_grad) but a differing subclass context must still
    # guard as equal — the routing/opaque metadata is not part of the guard.
    same = (base_ctx, ("marker", 999))
    other = (base_ctx, ("different", -1))
    assert _ExtInnerShardTensor.__metadata_guard__(same, other) is True

    # A flat base context (no subclass tail) is accepted too.
    assert _ExtInnerShardTensor.__metadata_guard__(base_ctx, base_ctx) is True


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_metadata_guard_ignores_subclass_context_1d(distributed_mesh):
    run_metadata_guard_ignores_subclass_context(distributed_mesh)


class _RoutedShardTensor(ShardTensor):
    """Subclass that declares routing metadata to ride onto op outputs."""

    _subclass_propagated_attrs = ("_route",)
    _route = None


def run_metadata_propagates_onto_op_results(mesh):
    st = shard_tensor_factory(mesh, uneven=True)
    routed = _RoutedShardTensor.__new__(
        _RoutedShardTensor,
        local_tensor=st._local_tensor,
        spec=st._spec,
        requires_grad=False,
    )
    routed._route = "policyX"

    # A base-typed autowrap result is re-classed to the subclass and the routing
    # attr is copied from the input.
    out = routed * 2.0
    assert isinstance(out, _RoutedShardTensor)
    assert out._route == "policyX"

    # Routing keeps riding along a chained op.
    out2 = out + 1.0
    assert isinstance(out2, _RoutedShardTensor)
    assert out2._route == "policyX"

    # A plain base ShardTensor (no declared propagated attrs) is untouched: its
    # op results stay the base type. This guards the "no overhead / no behavior
    # change for base" property.
    base_out = st * 2.0
    assert type(base_out) is ShardTensor


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_metadata_propagates_onto_op_results_1d(distributed_mesh):
    run_metadata_propagates_onto_op_results(distributed_mesh)


def _sum_squares(x):
    return (x**2).sum()


def run_compile_with_propagated_attrs_does_not_error(mesh):
    # A subclass declaring propagated attrs must still compile (propagation is
    # skipped under compile; metadata would travel via the flatten context).
    st = shard_tensor_factory(mesh, uneven=True)
    routed = _RoutedShardTensor.__new__(
        _RoutedShardTensor,
        local_tensor=st._local_tensor,
        spec=st._spec,
        requires_grad=False,
    )
    routed._route = "policyX"
    x = routed.detach().requires_grad_(True)

    torch._dynamo.reset()
    compiled = torch.compile(_sum_squares, fullgraph=True, backend="aot_eager")
    loss = compiled(x)
    loss.backward()


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_with_propagated_attrs_does_not_error_1d(distributed_mesh):
    run_compile_with_propagated_attrs_does_not_error(distributed_mesh)


def run_compile_subclass_survives(mesh):
    # A subclass carrying an extra inner tensor + nested context must survive
    # Dynamo flatten/unflatten + AOTAutograd with zero graph breaks, forward and
    # backward matching eager. This is the subclass analogue of the base
    # compile-backward smoke test and exercises the extension points end to end.
    x = _as_subclass(mesh).detach().requires_grad_(True)

    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    compiled = torch.compile(_sum_squares, fullgraph=True, backend="aot_eager")

    eager_loss = _sum_squares(x)
    loss = compiled(x)
    torch.testing.assert_close(loss.full_tensor(), eager_loss.full_tensor())

    loss.backward()
    assert not torch._dynamo.utils.counters["graph_break"]


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_subclass_survives_1d(distributed_mesh):
    run_compile_subclass_survives(distributed_mesh)
