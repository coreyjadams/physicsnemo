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

"""FLARE x physicsnemo-ops integration: parity, gradients, gate, fallbacks."""

import pytest
import torch

from physicsnemo.nn.module.flare_attention import FLARE
from physicsnemo.utils import _physicsnemo_ops as gate

pno = pytest.importorskip("physicsnemo_ops.torch")

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _flare(dim=128, heads=4, dim_head=64, n_global_queries=64, dtype=torch.float32):
    torch.manual_seed(7)
    return FLARE(
        dim=dim, heads=heads, dim_head=dim_head, n_global_queries=n_global_queries
    ).to(device="cuda", dtype=dtype)


def _eager_reference(module, x):
    """Run the same module with the accelerated paths disabled."""
    gate._reset_cache()
    try:
        import os

        os.environ[gate._ENV_DISABLE] = "1"
        gate._reset_cache()
        return module(x)
    finally:
        del os.environ[gate._ENV_DISABLE]
        gate._reset_cache()


@cuda
@pytest.mark.parametrize("dtype,tol", [(torch.float32, 5e-3), (torch.bfloat16, 5e-2)])
def test_flare_ops_forward_parity(dtype, tol):
    """The ops path matches the eager path within its precision class
    (fp32 runs TF32 tensor cores; bf16 is bf16 either way)."""
    module = _flare(dtype=dtype)
    x = torch.randn(2, 3111, 128, device="cuda", dtype=dtype)
    with torch.no_grad():
        got = module(x)
        ref = _eager_reference(module, x)
    err = (got.float() - ref.float()).abs().max() / ref.float().abs().max()
    assert err.item() < tol


@cuda
def test_flare_ops_uses_kernels(monkeypatch):
    """The eligible fp32 path actually routes through physicsnemo_ops."""
    calls = {"n": 0}
    real = pno.attn_lse_reduce

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pno, "attn_lse_reduce", spy)
    module = _flare()
    x = torch.randn(1, 500, 128, device="cuda")
    with torch.no_grad():
        module(x)
    assert calls["n"] == 1


@cuda
def test_flare_ops_fp32_grad_parity():
    """fp32 training goes through the fused deterministic backward; the
    parameter gradients match the eager path within TF32 tolerance."""
    module = _flare()
    x = torch.randn(1, 2048, 128, device="cuda", requires_grad=True)
    grad_out = torch.randn(1, 2048, 128, device="cuda")

    module(x).backward(grad_out)
    got = {n: p.grad.clone() for n, p in module.named_parameters()}
    got["__x__"] = x.grad.clone()

    module.zero_grad()
    x.grad = None
    _eager_reference(module, x).backward(grad_out)
    for name, param in module.named_parameters():
        ref = param.grad
        scale = ref.abs().max().clamp_min(1e-6)
        err = (got[name] - ref).abs().max() / scale
        assert err.item() < 1e-2, name
    err = (got["__x__"] - x.grad).abs().max() / x.grad.abs().max().clamp_min(1e-6)
    assert err.item() < 1e-2


@cuda
def test_flare_ops_bf16_training_falls_back(monkeypatch):
    """16-bit training keeps the eager (flash) path; inference uses the ops."""
    calls = {"n": 0}
    real = pno.attn_lse_reduce

    def spy(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    monkeypatch.setattr(pno, "attn_lse_reduce", spy)
    module = _flare(dtype=torch.bfloat16)
    x = torch.randn(1, 500, 128, device="cuda", dtype=torch.bfloat16)

    module(x.requires_grad_(True)).sum().backward()
    assert calls["n"] == 0  # training: eager/flash
    with torch.no_grad():
        module(x.detach())
    assert calls["n"] == 1  # inference: ops


@cuda
def test_flare_ops_ineligible_shapes_fall_back():
    """Head dims / query counts outside the fused envelope silently keep the
    eager path (bitwise identical to the disabled gate)."""
    module = _flare(dim=64, heads=4, dim_head=16, n_global_queries=48)
    x = torch.randn(1, 300, 64, device="cuda")
    with torch.no_grad():
        got = module(x)
        ref = _eager_reference(module, x)
    assert torch.equal(got, ref)


@cuda
def test_flare_ops_disable_env(monkeypatch):
    """PHYSICSNEMO_DISABLE_PHYSICSNEMO_OPS=1 restores eager numerics exactly."""
    module = _flare()
    x = torch.randn(1, 400, 128, device="cuda")
    monkeypatch.setenv(gate._ENV_DISABLE, "1")
    gate._reset_cache()
    try:
        with torch.no_grad():
            got = module(x)
            ref = _eager_reference(module, x)
        assert torch.equal(got, ref)
    finally:
        gate._reset_cache()
