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

r"""Tests for ShardTensor integration with ``torch.compile`` / AOTAutograd.

The focus is on the runtime tangent-coercion hook
``ShardTensor.__coerce_same_metadata_as_tangent__``, which AOTAutograd
invokes during the compiled backward when the runtime tangent's spec
doesn't match the recorded one. The tests cover uneven sharding, which
DTensor does not have to handle and which earlier coerce implementations
silently dropped (defaulting back to even chunking).
"""

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate, Shard

from physicsnemo.distributed import DistributedManager
from physicsnemo.domain_parallel import ShardTensor, scatter_tensor
from physicsnemo.domain_parallel._shard_tensor_spec import ShardTensorSpec
from test.domain_parallel.test_redistribute import shard_tensor_factory


def _replicate_placements(mesh):
    return [Replicate()] * mesh.ndim


def run_coerce_replicate_to_uneven_shard(mesh):
    # Round-trip: uneven Shard -> Replicate -> coerce back to recorded uneven Shard.
    st_uneven = shard_tensor_factory(mesh, uneven=True)
    recorded_spec = st_uneven._spec
    expected_local_shape = tuple(st_uneven._local_tensor.shape)
    expected_full = st_uneven.full_tensor().clone()

    st_replicated = st_uneven.redistribute(placements=_replicate_placements(mesh))

    coerced = st_replicated.__coerce_same_metadata_as_tangent__((recorded_spec, False))

    assert isinstance(coerced, ShardTensor)
    assert coerced._spec.placements == recorded_spec.placements
    assert tuple(coerced._local_tensor.shape) == expected_local_shape
    assert torch.allclose(coerced.full_tensor(), expected_full)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_replicate_to_uneven_shard_1d(distributed_mesh):
    run_coerce_replicate_to_uneven_shard(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_replicate_to_uneven_shard_2d(distributed_mesh_2d):
    run_coerce_replicate_to_uneven_shard(distributed_mesh_2d)


def run_coerce_same_placements_unknown_shapes(mesh):
    # Recorded spec carries the same placements but no _sharding_shapes; the
    # hook must accept it without erroring and preserve local data.
    st = shard_tensor_factory(mesh, uneven=True)
    expected_local_shape = tuple(st._local_tensor.shape)
    expected_full = st.full_tensor().clone()

    modified_spec = ShardTensorSpec(
        mesh=st._spec.mesh,
        placements=st._spec.placements,
        tensor_meta=st._spec.tensor_meta,
        _sharding_shapes=None,
    )

    coerced = st.__coerce_same_metadata_as_tangent__((modified_spec, False))

    assert isinstance(coerced, ShardTensor)
    assert coerced._spec.placements == st._spec.placements
    assert coerced._spec._sharding_shapes is None
    assert tuple(coerced._local_tensor.shape) == expected_local_shape
    assert torch.allclose(coerced.full_tensor(), expected_full)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_same_placements_unknown_shapes_1d(distributed_mesh):
    run_coerce_same_placements_unknown_shapes(distributed_mesh)


def run_coerce_expected_type_returns_none(mesh):
    # A *foreign* expected_type (a plain tensor / DTensor, i.e. not a ShardTensor
    # subclass) must short-circuit to None -- the DTensor cross-type convention,
    # which the subclass-friendly path below deliberately preserves.
    st = shard_tensor_factory(mesh, uneven=True)
    out = st.__coerce_same_metadata_as_tangent__(
        (st._spec, False), expected_type=torch.Tensor
    )
    assert out is None


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_expected_type_returns_none_1d(distributed_mesh):
    run_coerce_expected_type_returns_none(distributed_mesh)


class _MarkerShardTensor(ShardTensor):
    """A minimal ``ShardTensor`` subclass used to exercise the subclass-friendly
    cross-type tangent-coercion path. It adds no new inner tensors or metadata;
    its own ``__tensor_unflatten__`` just reclasses a base reconstruction so the
    result carries the subclass type (as a real subclass with per-instance
    routing metadata would)."""

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        st = ShardTensor.__tensor_unflatten__(
            inner_tensors, flatten_spec, outer_size, outer_stride
        )
        st.__class__ = _MarkerShardTensor
        return st


def run_coerce_accepts_nested_flatten_context(mesh):
    # A ShardTensor subclass emits a nested ``(base_ctx, subclass_ctx)`` flatten
    # context. The coerce hook must unwrap it to the base ``(spec, requires_grad)``
    # and behave exactly as it does for the base's flat context.
    st = shard_tensor_factory(mesh, uneven=True)
    recorded_spec = st._spec
    expected_local_shape = tuple(st._local_tensor.shape)
    expected_full = st.full_tensor().clone()

    nested_ctx = ((recorded_spec, False), ("subclass-metadata-placeholder",))
    coerced = st.__coerce_same_metadata_as_tangent__(nested_ctx)

    assert isinstance(coerced, ShardTensor)
    assert coerced._spec.placements == recorded_spec.placements
    assert tuple(coerced._local_tensor.shape) == expected_local_shape
    assert torch.allclose(coerced.full_tensor(), expected_full)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_accepts_nested_flatten_context_1d(distributed_mesh):
    run_coerce_accepts_nested_flatten_context(distributed_mesh)


def run_coerce_empty_and_none_sharding_shapes_equal(mesh):
    # ``{}`` (an op-result re-wrap) and ``None`` (a fresh spec) both mean "no
    # explicit per-rank sharding shapes". With identical placements the hook must
    # treat them as equal and short-circuit to ``self`` -- no redistribute.
    st = shard_tensor_factory(mesh, uneven=True)

    spec_empty = ShardTensorSpec(
        mesh=st._spec.mesh,
        placements=st._spec.placements,
        tensor_meta=st._spec.tensor_meta,
        _sharding_shapes={},
    )
    st_empty = ShardTensor(st._local_tensor, spec_empty, requires_grad=False)

    recorded_spec_none = ShardTensorSpec(
        mesh=st._spec.mesh,
        placements=st._spec.placements,
        tensor_meta=st._spec.tensor_meta,
        _sharding_shapes=None,
    )

    coerced = st_empty.__coerce_same_metadata_as_tangent__((recorded_spec_none, False))
    assert coerced is st_empty


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_empty_and_none_sharding_shapes_equal_1d(distributed_mesh):
    run_coerce_empty_and_none_sharding_shapes_equal(distributed_mesh)


def run_coerce_expected_subclass_rebuilds(mesh):
    # A differing expected_type that *is* a ShardTensor subclass must be rebuilt
    # via that subclass's own ``__tensor_unflatten__`` (reclass + reattach), not
    # dropped to None -- this is what lets a subclass accept a base-typed tangent.
    st = shard_tensor_factory(mesh, uneven=True)
    expected_full = st.full_tensor().clone()

    out = st.__coerce_same_metadata_as_tangent__(
        (st._spec, False), expected_type=_MarkerShardTensor
    )

    assert isinstance(out, _MarkerShardTensor)
    assert out._spec.placements == st._spec.placements
    assert torch.allclose(out.full_tensor(), expected_full)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_expected_subclass_rebuilds_1d(distributed_mesh):
    run_coerce_expected_subclass_rebuilds(distributed_mesh)


def run_unflatten_does_not_force_inner_requires_grad(mesh):
    # ``__tensor_unflatten__`` must not force ``requires_grad`` on the inner: the wrapper
    # carries it (via grad_fn) while a detached op-result local does not, and forcing them to
    # agree trips ``assert_metadata_eq`` on a graph-break re-fake. Matches DTensor.
    st = shard_tensor_factory(mesh, uneven=False)
    local = st._local_tensor.detach()
    assert local.requires_grad is False

    _inner_names, ctx = st.__tensor_flatten__()
    spec, _ = ctx

    rebuilt = ShardTensor.__tensor_unflatten__(
        {"_local_tensor": local}, (spec, True), st.shape, st.stride()
    )

    # Wrapper honors the requested requires_grad; the inner is left untouched
    # (still False) and the passed-in local is not mutated in place.
    assert rebuilt.requires_grad is True
    assert rebuilt._local_tensor.requires_grad is False
    assert local.requires_grad is False


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_unflatten_does_not_force_inner_requires_grad_1d(distributed_mesh):
    run_unflatten_does_not_force_inner_requires_grad(distributed_mesh)


def _sum_squares(x):
    return (x**2).sum()


def run_compile_backward_uneven_shard(mesh):
    # Smoke: compile + backward over an uneven ShardTensor must not raise AOT's
    # "guessed metadata incorrectly"; gradient values are checked in the coerce tests.
    x = shard_tensor_factory(mesh, uneven=True).detach().requires_grad_(True)

    torch._dynamo.reset()
    compiled = torch.compile(_sum_squares, fullgraph=True, backend="aot_eager")

    loss = compiled(x)
    loss.backward()


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_backward_uneven_shard_1d(distributed_mesh):
    run_compile_backward_uneven_shard(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
def test_compile_backward_uneven_shard_2d(distributed_mesh_2d):
    run_compile_backward_uneven_shard(distributed_mesh_2d)


# --- Regression: grads for ShardTensor *inputs* of a compiled region ---------
#
# AOTAutograd's joint trace computes grad_inputs by calling
# ``torch.autograd.grad`` on the wrapped subclass primals, with no
# DisableTorchFunctionSubclass guard. ShardTensor's ``__torch_function__``
# used to route that call through the DTensor fallback, which re-issued the
# graph query on freshly converted tensors (not in the graph); with
# ``allow_unused=True`` this silently produced all-None grads at trace time,
# so ``grad_input_metas`` was stamped plain and every compiled region
# returned plain-tensor gradients for its ShardTensor inputs -- crashing the
# first eager backward upstream that touched ``._local_tensor``.
# ``torch.autograd.grad`` is now in ``_autograd_passthrough_functions``.

_DIM = 64


def run_compiled_grad_input_stays_shard_tensor(mesh, partial_input):
    # Eager producer -> compiled consumer. The gradient the compiled region
    # returns for its ShardTensor input must arrive at the eager producer's
    # backward as a ShardTensor, with the same values as a fully-eager run.
    dm = DistributedManager()
    device = dm.device
    torch.manual_seed(7)

    x_full = torch.randn(1, 32, _DIM, device=device)
    x = scatter_tensor(x_full, 0, mesh, (Shard(1),))
    w0 = torch.randn(_DIM, _DIM, device=device, requires_grad=True)
    consumer = torch.nn.Linear(_DIM, _DIM).to(device)

    def run_once(consumer_fn):
        m = torch.nn.functional.linear(x, w0)
        if partial_input:
            # Mean over the sharded dim: Partial placement from the eager
            # custom reduction op -- the configuration that crashed first.
            m = m.mean(dim=(1,))
        grad_types = []
        m.register_hook(lambda g: grad_types.append(type(g)))
        consumer_fn(m).sum().backward()
        grad, w0.grad = w0.grad, None
        return grad, grad_types

    eager_grad, eager_types = run_once(consumer)
    assert issubclass(eager_types[0], ShardTensor)

    torch._dynamo.reset()
    compiled = torch.compile(
        consumer, fullgraph=True, backend="aot_eager", dynamic=False
    )
    # Second iteration exercises the cached compiled backward path.
    for _ in range(2):
        compiled_grad, compiled_types = run_once(compiled)
        assert compiled_types and issubclass(compiled_types[0], ShardTensor), (
            f"compiled region delivered grad of type {compiled_types} "
            "for its ShardTensor input"
        )
        torch.testing.assert_close(compiled_grad, eager_grad)


@pytest.mark.multigpu_static
@pytest.mark.timeout(180)
@pytest.mark.parametrize("partial_input", [False, True])
def test_compiled_grad_input_stays_shard_tensor_1d(distributed_mesh, partial_input):
    run_compiled_grad_input_stays_shard_tensor(distributed_mesh, partial_input)


def _partial_shard_tensor(mesh, local):
    # Partial local shape == global shape; build through DTensor.from_local
    # (accepts Partial directly) + communication-free spec inference.
    dt = DTensor.from_local(local, mesh, [Partial()] * mesh.ndim, run_check=False)
    return ShardTensor.from_dtensor(dt)


def run_coerce_partial_to_replicate_relabels(mesh):
    # Grad-direction convention: a Partial-labeled tangent is replicate-valued.
    # Coercing it to a recorded Replicate spec must RELABEL, not all-reduce —
    # a real reduction multiplies the tangent by the mesh size. Regression
    # test for the 4x compiled-grad bug with Partial inputs.
    dm = DistributedManager()
    local = torch.full((8, 4), 3.0, device=dm.device)
    st = _partial_shard_tensor(mesh, local)

    recorded = ShardTensorSpec(
        mesh=st._spec.mesh,
        placements=tuple(Replicate() for _ in range(mesh.ndim)),
        tensor_meta=st._spec.tensor_meta,
        _sharding_shapes=None,
    )
    coerced = st.__coerce_same_metadata_as_tangent__((recorded, False))

    assert isinstance(coerced, ShardTensor)
    assert coerced._spec.placements == recorded.placements
    assert torch.allclose(coerced._local_tensor, local), (
        "Partial->Replicate tangent coercion changed values (all-reduced "
        "an already-full tangent?)"
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_partial_to_replicate_relabels_1d(distributed_mesh):
    run_coerce_partial_to_replicate_relabels(distributed_mesh)


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_partial_to_replicate_relabels_2d(distributed_mesh_2d):
    run_coerce_partial_to_replicate_relabels(distributed_mesh_2d)


def run_coerce_replicate_to_partial_relabels(mesh):
    # Reverse direction: recorded spec says Partial, runtime tangent arrived
    # Replicate. Must relabel (values unchanged), not partition (divide by
    # mesh size) — the latent inverse of the 4x bug.
    dm = DistributedManager()
    local = torch.full((8, 4), 3.0, device=dm.device)
    st = ShardTensor.from_local(
        local, mesh, tuple(Replicate() for _ in range(mesh.ndim))
    )

    recorded = ShardTensorSpec(
        mesh=st._spec.mesh,
        placements=tuple(Partial() for _ in range(mesh.ndim)),
        tensor_meta=st._spec.tensor_meta,
        _sharding_shapes=None,
    )
    coerced = st.__coerce_same_metadata_as_tangent__((recorded, False))

    assert isinstance(coerced, ShardTensor)
    assert coerced._spec.placements == recorded.placements
    assert torch.allclose(coerced._local_tensor, local), (
        "Replicate->Partial tangent coercion changed values (partitioned the tangent?)"
    )


@pytest.mark.multigpu_static
@pytest.mark.timeout(120)
def test_coerce_replicate_to_partial_relabels_1d(distributed_mesh):
    run_coerce_replicate_to_partial_relabels(distributed_mesh)


# to_local() differentiability under torch.compile: ShardTensor's eager fallback converts to
# DTensor via autograd.Functions that AOT traces through, dropping the compiled backward; the
# __torch_function__ compile-passthrough restores it. These tests catch a zero/missing gradient.
def _to_local_sq_sum(x):
    return (x.to_local() ** 2).sum()


def _to_local_plain_math(x):
    return (x.to_local() * x.to_local()).sum()


def _op_result_to_local(x):
    # Forces an intermediate op-result ShardTensor before to_local().
    return ((x * 2.0 + x).to_local() ** 2).sum()


# Plain-torch references on the local tensor: an independent gradient oracle.
def _ref_sq_sum(local):
    return (local**2).sum()


def _ref_plain_math(local):
    return (local * local).sum()


def _ref_op_result(local):
    return ((local * 2.0 + local) ** 2).sum()


def run_to_local_grad_under_compile(mesh, fn, ref_fn, backend):
    placements = [Shard(0)] + [Replicate()] * (mesh.ndim - 1)

    base = shard_tensor_factory(mesh, uneven=True).full_tensor().detach()

    # Independent ground truth: plain-torch autograd on the local tensor.
    lr = base.clone().requires_grad_(True)
    grad_ref = torch.autograd.grad(ref_fn(lr), lr)[0]

    # ShardTensor eager path.
    le = base.clone().requires_grad_(True)
    ste = ShardTensor.from_local(le, mesh, placements)
    grad_eager = torch.autograd.grad(fn(ste), le)[0]

    # ShardTensor under torch.compile.
    torch._dynamo.reset()
    lc = base.clone().requires_grad_(True)
    stc = ShardTensor.from_local(lc, mesh, placements)
    compiled = torch.compile(fn, fullgraph=True, backend=backend)
    grad_compiled = torch.autograd.grad(compiled(stc), lc)[0]

    assert grad_compiled is not None, "compiled backward dropped the gradient"
    assert grad_compiled.abs().sum() > 0, "compiled gradient is zero"

    # Eager must match the independent reference (guards the reference itself),
    # and the compiled gradient must match both -- i.e. be numerically correct,
    # not merely nonzero or merely equal to a possibly-wrong eager value.
    torch.testing.assert_close(grad_eager, grad_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_compiled, grad_ref, atol=1e-5, rtol=1e-5)


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize("backend", ["aot_eager", "inductor"])
@pytest.mark.parametrize(
    "fn,ref_fn",
    [
        (_to_local_sq_sum, _ref_sq_sum),
        (_to_local_plain_math, _ref_plain_math),
        (_op_result_to_local, _ref_op_result),
    ],
    ids=["leaf", "plain_math", "op_result"],
)
def test_to_local_grad_under_compile_1d(distributed_mesh, fn, ref_fn, backend):
    run_to_local_grad_under_compile(distributed_mesh, fn, ref_fn, backend)


@pytest.mark.multigpu_static
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "fn,ref_fn",
    [(_to_local_sq_sum, _ref_sq_sum), (_op_result_to_local, _ref_op_result)],
    ids=["leaf", "op_result"],
)
def test_to_local_grad_under_compile_2d(distributed_mesh_2d, fn, ref_fn):
    run_to_local_grad_under_compile(distributed_mesh_2d, fn, ref_fn, "aot_eager")
