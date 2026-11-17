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

import pytest
import torch

from physicsnemo.experimental.models.geotransolver_pp import GeoTransolverPP

N_POINTS, N_BOUNDARY = 40, 48


def _model(global_vectors: int) -> GeoTransolverPP:
    torch.manual_seed(0)
    model = GeoTransolverPP(
        out_scalars=2,
        out_vectors=2,
        boundary_scalars=3,
        boundary_vectors=1,
        global_scalars=2,
        global_vectors=global_vectors,
        n_hidden=32,
        n_layers=2,
        n_head=4,
        n_latents=8,
        n_neighbors=4,
    )
    # Double precision keeps round-off far below the test tolerances.
    return model.double().eval()


def _inputs(global_vectors: int, batch: int = 2) -> dict[str, torch.Tensor]:
    torch.manual_seed(1)
    rand = lambda *shape: torch.randn(*shape, dtype=torch.float64)  # noqa: E731
    return {
        "points": rand(batch, N_POINTS, 3),
        "boundary_points": rand(batch, N_BOUNDARY, 3),
        "boundary_normals": torch.nn.functional.normalize(
            rand(batch, N_BOUNDARY, 3), dim=-1
        ),
        "boundary_measure": rand(batch, N_BOUNDARY).abs() + 0.1,
        "boundary_scalars": rand(batch, N_BOUNDARY, 3),
        "boundary_vectors": rand(batch, N_BOUNDARY, 1, 3),
        "global_scalars": rand(batch, 2),
        "global_vectors": rand(batch, global_vectors, 3),
        "point_measure": rand(batch, N_POINTS).abs() + 0.1,
    }


def _random_rotation() -> torch.Tensor:
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    return q * torch.linalg.det(q).sign()


def test_output_shapes():
    scalars, vectors = _model(2)(**_inputs(2))
    assert scalars.shape == (2, N_POINTS, 2)
    assert vectors.shape == (2, N_POINTS, 2, 3)

    # No optional inputs.
    model = GeoTransolverPP(out_scalars=1, out_vectors=0, n_hidden=16, n_head=2)
    inputs = _inputs(0)
    scalars, vectors = model.double()(
        inputs["points"],
        inputs["boundary_points"],
        inputs["boundary_normals"],
        inputs["boundary_measure"],
    )
    assert scalars.shape == (2, N_POINTS, 1)
    assert vectors.shape == (2, N_POINTS, 0, 3)


@pytest.mark.parametrize("global_vectors", [0, 1, 2])
def test_rotation_translation_equivariance(global_vectors):
    r"""Covers frames from global vectors, learned vectors, and a mix."""
    model = _model(global_vectors)
    inputs = _inputs(global_vectors)
    rotation = _random_rotation()
    shift = torch.randn(3, dtype=torch.float64)

    moved = dict(inputs)
    for name in ("points", "boundary_points"):
        moved[name] = inputs[name] @ rotation.T + shift
    for name in ("boundary_normals", "boundary_vectors", "global_vectors"):
        moved[name] = inputs[name] @ rotation.T

    scalars, vectors = model(**inputs)
    moved_scalars, moved_vectors = model(**moved)
    torch.testing.assert_close(moved_scalars, scalars, atol=1e-9, rtol=1e-7)
    torch.testing.assert_close(
        moved_vectors, vectors @ rotation.T, atol=1e-9, rtol=1e-7
    )


def test_scale_invariance():
    model = _model(2)
    inputs = _inputs(2)
    scaled = dict(inputs)
    for name in ("points", "boundary_points"):
        scaled[name] = 7.0 * inputs[name]
    scaled["boundary_measure"] = 49.0 * inputs["boundary_measure"]
    for expected, actual in zip(model(**inputs), model(**scaled)):
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)


def test_point_refinement_and_order():
    r"""Splitting a point's measure across two copies changes nothing."""
    model = _model(2)
    inputs = _inputs(2)
    points, measure = inputs["points"], inputs["point_measure"]
    order = torch.randperm(N_POINTS + 8)
    refined = dict(
        inputs,
        points=torch.cat([points, points[:, :8]], dim=1)[:, order],
        point_measure=torch.cat(
            [measure[:, :8] / 2, measure[:, 8:], measure[:, :8] / 2], dim=1
        )[:, order],
    )
    restore = order.argsort()[:N_POINTS]
    for expected, actual in zip(model(**inputs), model(**refined)):
        torch.testing.assert_close(actual[:, restore], expected, atol=1e-10, rtol=1e-8)


def test_backward():
    model = _model(1)
    scalars, vectors = model(**_inputs(1))
    (scalars.sum() + vectors.sum()).backward()
    assert all(p.grad is not None for p in model.parameters())
