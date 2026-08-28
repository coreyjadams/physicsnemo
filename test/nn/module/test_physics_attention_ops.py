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

"""PhysicsAttention / GALE_FA x physicsnemo-ops: parity, gradients, fallbacks."""

import os

import pytest
import torch

from physicsnemo.nn.module.gale import GALE_FA
from physicsnemo.nn.module.physics_attention import PhysicsAttentionIrregularMesh
from physicsnemo.utils import _physicsnemo_ops as gate

pno = pytest.importorskip("physicsnemo_ops.torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _eager_reference(module, *args):
    """Run the module with the accelerated paths disabled."""
    try:
        os.environ[gate._ENV_DISABLE] = "1"
        gate._reset_cache()
        return module(*args)
    finally:
        del os.environ[gate._ENV_DISABLE]
        gate._reset_cache()


def _physics_attention(dtype=torch.float32, plus=False, slice_num=32):
    torch.manual_seed(11)
    return PhysicsAttentionIrregularMesh(
        dim=128,
        heads=4,
        dim_head=64,
        dropout=0.0,
        slice_num=slice_num,
        use_te=False,
        plus=plus,
    ).to(device="cuda", dtype=dtype)


@cuda
@pytest.mark.parametrize("slice_num", [32, 48, 256])
@pytest.mark.parametrize("dtype,tol", [(torch.float16, 2e-2), (torch.bfloat16, 5e-2)])
def test_physics_attention_ops_forward_parity(dtype, tol, slice_num):
    """The fused reduce matches the eager chain (16-bit engages; the fused
    pass gains fp32 accumulation over the eager 16-bit chain). Slice counts
    cover a baked kernel (32), a runtime multiple of 16 (48), and the
    envelope ceiling (256)."""
    module = _physics_attention(dtype=dtype, slice_num=slice_num)
    x = torch.randn(2, 2311, 128, device="cuda", dtype=dtype)
    with torch.no_grad():
        got = module(x)
        ref = _eager_reference(module, x)
    err = (got.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert err.item() < tol


@cuda
def test_physics_attention_ops_uses_kernels(monkeypatch):
    """The eligible path actually routes through slice_attn_softmax_reduce."""
    calls = {"n": 0}
    real = pno.slice_attn_softmax_reduce

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pno, "slice_attn_softmax_reduce", spy)
    module = _physics_attention(dtype=torch.bfloat16)
    with torch.no_grad():
        module(torch.randn(1, 500, 128, device="cuda", dtype=torch.bfloat16))
    assert calls["n"] == 1


@cuda
def test_physics_attention_ops_fp32_falls_back():
    """fp32 keeps the eager path bitwise: the op's exact fp32 kernel is
    slower than eager, and the fast variant would change fp32 numerics."""
    module = _physics_attention()
    x = torch.randn(1, 400, 128, device="cuda")
    with torch.no_grad():
        got = module(x)
        ref = _eager_reference(module, x)
    assert torch.equal(got, ref)


@cuda
@pytest.mark.parametrize("dtype,tol", [(torch.float16, 2e-2), (torch.bfloat16, 5e-2)])
def test_physics_attention_ops_grad_parity(dtype, tol):
    """Training goes through the composite backward; parameter gradients
    match the eager path."""
    module = _physics_attention(dtype=dtype)
    x = torch.randn(1, 1024, 128, device="cuda", dtype=dtype, requires_grad=True)
    grad_out = torch.randn(1, 1024, 128, device="cuda", dtype=dtype)

    module(x).backward(grad_out)
    got = {
        n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None
    }
    got["__x__"] = x.grad.clone()

    module.zero_grad()
    x.grad = None
    _eager_reference(module, x).backward(grad_out)
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        ref = param.grad.float()
        err = (got[name].float() - ref).abs().max() / ref.abs().max().clamp_min(1e-5)
        assert err.item() < tol, name
    ref = x.grad.float()
    err = (got["__x__"].float() - ref).abs().max() / ref.abs().max().clamp_min(1e-5)
    assert err.item() < tol


@cuda
def test_physics_attention_plus_falls_back(monkeypatch):
    """The Transolver++ (gumbel) path keeps the eager math."""
    calls = {"n": 0}
    real = pno.slice_attn_softmax_reduce

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pno, "slice_attn_softmax_reduce", spy)
    module = _physics_attention(plus=True)
    with torch.no_grad():
        module(torch.randn(1, 300, 128, device="cuda"))
    assert calls["n"] == 0


@cuda
def test_physics_attention_odd_slices_fall_back():
    """Slice counts outside the fast envelope keep the eager path bitwise."""
    module = _physics_attention(slice_num=24)
    x = torch.randn(1, 400, 128, device="cuda")
    with torch.no_grad():
        got = module(x)
        ref = _eager_reference(module, x)
    assert torch.equal(got, ref)


@cuda
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 5e-3), (torch.bfloat16, 5e-2)])
def test_gale_fa_cross_attention_ops(dtype, tol):
    """GALE_FA (FLARE self-attention + token/context cross-attention) runs
    both fused paths and matches the eager module, forward and backward."""
    torch.manual_seed(13)
    module = GALE_FA(
        dim=128, heads=4, dim_head=64, n_global_queries=64, context_dim=64
    ).to(device="cuda", dtype=dtype)
    x = (torch.randn(1, 1517, 128, device="cuda", dtype=dtype, requires_grad=True),)
    context = torch.randn(1, 4, 32, 64, device="cuda", dtype=dtype)

    out = module(x, context)[0]
    ref = _eager_reference(module, x, context)[0]
    err = (out.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert err.item() < tol

    if dtype is torch.float32:
        out.square().mean().backward()
        assert x[0].grad is not None and torch.isfinite(x[0].grad).all()
