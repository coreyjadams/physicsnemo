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

"""Tests for the optional physicsnemo_ops availability gate."""

import sys

import pytest
import torch

from physicsnemo.utils import _physicsnemo_ops as gate


@pytest.fixture()
def clean_gate():
    """Reset the gate cache before and after each test."""
    gate._reset_cache()
    yield
    gate._reset_cache()


def test_gate_returns_module_or_none(clean_gate):
    result = gate.physicsnemo_ops_torch()
    try:
        import physicsnemo_ops.torch as expected
    except Exception:
        expected = None
    assert result is expected


def test_gate_result_is_cached(clean_gate):
    assert gate.physicsnemo_ops_torch() is gate.physicsnemo_ops_torch()


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_env_kill_switch_disables(clean_gate, monkeypatch, value):
    monkeypatch.setenv(gate._ENV_DISABLE, value)
    gate._reset_cache()
    assert gate.physicsnemo_ops_torch() is None


@pytest.mark.parametrize("value", ["", "0", "false", "off"])
def test_env_kill_switch_falsy_values_do_not_disable(clean_gate, monkeypatch, value):
    monkeypatch.setenv(gate._ENV_DISABLE, value)
    gate._reset_cache()
    try:
        import physicsnemo_ops.torch as expected
    except Exception:
        expected = None
    assert gate.physicsnemo_ops_torch() is expected


def test_broken_import_degrades_to_none(clean_gate, monkeypatch):
    # Simulate a broken installation: importing the package raises.
    monkeypatch.delitem(sys.modules, "physicsnemo_ops", raising=False)
    monkeypatch.delitem(sys.modules, "physicsnemo_ops.torch", raising=False)

    import builtins

    real_import = builtins.__import__

    def broken_import(name, *args, **kwargs):
        if name.startswith("physicsnemo_ops"):
            raise RuntimeError("simulated broken extension")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    gate._reset_cache()
    assert gate.physicsnemo_ops_torch() is None


@pytest.mark.parametrize(
    ("dtype", "n_columns", "index_bound", "expected"),
    [
        (torch.int64, 2, 100, True),
        (torch.int32, 8, 100, True),
        (torch.int64, 3, 2**30, False),  # 90 bits > 63
        (torch.int64, 9, 10, False),  # too many columns
        (torch.int64, 0, 10, False),  # zero columns
        (torch.float32, 2, 10, False),  # float rows
        (torch.int64, 2, 0, False),  # invalid bound
        (torch.int64, 2, 1, True),  # degenerate but valid bound
    ],
)
def test_int_rows_ok(dtype, n_columns, index_bound, expected):
    rows = torch.zeros((4, n_columns), dtype=dtype)
    assert gate.int_rows_ok(rows, index_bound) is expected


def test_dtype_predicates_cpu():
    f32 = torch.zeros(2, dtype=torch.float32)
    f16 = torch.zeros(2, dtype=torch.float16)
    i64 = torch.zeros(2, dtype=torch.int64)
    assert gate.segment_sum_dtype_ok(f32)
    assert gate.segment_sum_dtype_ok(i64)
    assert not gate.segment_sum_dtype_ok(f16)  # no f16 on CPU
    assert gate.csr_mean_dtype_ok(f32)
    assert not gate.csr_mean_dtype_ok(i64)
    assert gate.csr_cmp_dtype_ok(i64)
    assert gate.segment_cmp_dtype_ok(i64)


@pytest.mark.parametrize("cpu_opt_in", [False, True])
def test_device_policy_cpu(clean_gate, monkeypatch, cpu_opt_in):
    if cpu_opt_in:
        monkeypatch.setenv(gate._ENV_ENABLE_CPU, "1")
    else:
        monkeypatch.delenv(gate._ENV_ENABLE_CPU, raising=False)
    gate._reset_cache()
    cpu_tensor = torch.zeros(2)
    result = gate.physicsnemo_ops_for(cpu_tensor)
    if gate.physicsnemo_ops_torch() is None:
        assert result is None
    elif cpu_opt_in:
        assert result is gate.physicsnemo_ops_torch()
    else:
        assert result is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_device_policy_cuda(clean_gate, monkeypatch):
    monkeypatch.delenv(gate._ENV_ENABLE_CPU, raising=False)
    gate._reset_cache()
    cuda_tensor = torch.zeros(2, device="cuda")
    assert gate.physicsnemo_ops_for(cuda_tensor) is gate.physicsnemo_ops_torch()
