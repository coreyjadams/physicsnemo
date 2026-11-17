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
from torch.nn.functional import normalize

from physicsnemo.datapipes.transforms.mesh.transforms import canonical_frame
from physicsnemo.experimental.models.geotransolver_pp import GeoTransolverPP

N_POINTS = 40
FLOW = torch.tensor([1.0, 0.2, 0.1], dtype=torch.float64)


def _model() -> GeoTransolverPP:
    torch.manual_seed(0)
    model = GeoTransolverPP(
        out_scalars=2,
        out_vectors=2,
        boundary_features={"wall": 1, "inlet": 3},
        global_features=4,
        n_hidden=32,
        n_layers=2,
        n_head=4,
        n_latents=8,
        radius=1.0,
        n_neighbors=64,
    )
    # The condition-dependent norms start at zero; give them values.
    with torch.no_grad():
        for p in model.parameters():
            if not p.any():
                p.normal_(std=0.1)
    # Double precision keeps round-off far below the test tolerances.
    return model.double().eval()


def _problem() -> dict:
    """A lopsided body (wall) and an inlet patch, in the lab frame."""
    g = torch.Generator().manual_seed(1)
    rand = lambda *shape: torch.rand(*shape, dtype=torch.float64, generator=g)  # noqa: E731
    wall = rand(48, 3) ** 2 * torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64)
    inlet = rand(16, 3) * torch.tensor([0.0, 4.0, 4.0], dtype=torch.float64) - 3.0
    return {
        "points": wall[torch.randint(48, (N_POINTS,), generator=g)]
        + 0.3 * rand(N_POINTS, 3),
        "point_measure": rand(N_POINTS) + 0.1,
        "wall": {
            "points": wall,
            "normals": normalize(rand(48, 3) - 0.5, dim=-1),
            "measure": rand(48) + 0.1,
            "scalars": rand(48, 1),
        },
        "inlet": {
            "points": inlet,
            "normals": FLOW.new_tensor([1.0, 0.0, 0.0]).expand(16, 3),
            "measure": rand(16) + 0.1,
            "vectors": FLOW * (1 + rand(16, 1)),
        },
        "flow": FLOW,
    }


def _move(problem: dict, matrix: torch.Tensor, shift: torch.Tensor) -> dict:
    """Apply ``x -> x @ matrix.T + shift``; ``matrix`` may include a scale."""
    scale = torch.linalg.det(matrix).abs() ** (1 / 3)
    out = dict(problem, points=problem["points"] @ matrix.T + shift)
    out["flow"] = problem["flow"] @ matrix.T / scale
    for name in ("wall", "inlet"):
        b = dict(problem[name])
        b["points"] = b["points"] @ matrix.T + shift
        b["normals"] = b["normals"] @ matrix.T / scale
        b["measure"] = b["measure"] * scale**2
        if "vectors" in b:
            b["vectors"] = b["vectors"] @ matrix.T / scale
        out[name] = b
    return out


def _predict(model: GeoTransolverPP, problem: dict):
    """Canonical frame of the wall, then the model; vectors back in the lab frame."""
    wall = problem["wall"]
    origin, axes, source = canonical_frame(
        wall["points"], wall["measure"], [problem["flow"]]
    )
    assert -2 not in source.tolist()
    r = wall["points"] - origin
    length = (
        (wall["measure"] * r.square().sum(-1)).sum() / wall["measure"].sum()
    ).sqrt()

    def boundary(b):
        out = {
            "points": ((b["points"] - origin) @ axes.T / length)[None],
            "normals": (b["normals"] @ axes.T)[None],
            "measure": b["measure"][None],
        }
        features = [
            b[k] @ (axes.T if k == "vectors" else torch.eye(1).double())
            for k in ("scalars", "vectors")
            if k in b
        ]
        return out | {"features": torch.cat(features, dim=-1)[None]}

    scalars, vectors = model(
        ((problem["points"] - origin) @ axes.T / length)[None],
        {name: boundary(problem[name]) for name in ("wall", "inlet")},
        global_features=torch.cat([FLOW.new_tensor([0.3]), problem["flow"] @ axes.T])[
            None
        ],
        point_measure=problem["point_measure"][None],
    )
    return scalars[0], vectors[0] @ axes


def _random_rotation() -> torch.Tensor:
    q, _ = torch.linalg.qr(
        torch.randn(
            3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(2)
        )
    )
    return q * torch.linalg.det(q).sign()


def test_output_shapes():
    scalars, vectors = _predict(_model(), _problem())
    assert scalars.shape == (N_POINTS, 2)
    assert vectors.shape == (N_POINTS, 2, 3)


def test_undeclared_boundary_raises():
    with pytest.raises(ValueError, match="Expected boundaries"):
        _model()(torch.zeros(1, 4, 3, dtype=torch.float64), {"outlet": {}})


@torch.no_grad()
def test_rotation_translation_and_scale():
    model, problem = _model(), _problem()
    matrix = 2.5 * _random_rotation()
    shift = torch.tensor([3.0, -2.0, 5.0], dtype=torch.float64)
    scalars, vectors = _predict(model, problem)
    moved_scalars, moved_vectors = _predict(model, _move(problem, matrix, shift))
    torch.testing.assert_close(moved_scalars, scalars, atol=1e-9, rtol=0)
    torch.testing.assert_close(
        moved_vectors, vectors @ matrix.T / 2.5, atol=1e-9, rtol=0
    )


@torch.no_grad()
def test_boundary_order_and_refinement():
    model, problem = _model(), _problem()
    reference = _predict(model, problem)
    # Shuffle the wall samples and split the first ten in two.
    order = torch.randperm(48, generator=torch.Generator().manual_seed(3))
    wall = {k: v[order] for k, v in problem["wall"].items()}
    wall["measure"] = torch.cat([wall["measure"][:10] / 2, wall["measure"][10:]])
    wall = {k: torch.cat([v, v[:10]]) for k, v in wall.items()}
    for a, b in zip(_predict(model, dict(problem, wall=wall)), reference):
        torch.testing.assert_close(a, b, atol=1e-9, rtol=0)


@torch.no_grad()
def test_point_order_and_refinement():
    model, problem = _model(), _problem()
    reference = _predict(model, problem)
    order = torch.randperm(N_POINTS, generator=torch.Generator().manual_seed(4))
    points, measure = problem["points"][order], problem["point_measure"][order]
    measure = torch.cat([measure[:10] / 2, measure[10:], measure[:10] / 2])
    refined = dict(
        problem, points=torch.cat([points, points[:10]]), point_measure=measure
    )
    for a, b in zip(_predict(model, refined), reference):
        torch.testing.assert_close(a[:N_POINTS], b[order], atol=1e-9, rtol=0)


def test_backward():
    model, problem = _model(), _problem()
    scalars, vectors = _predict(model, problem)
    (scalars.sum() + vectors.sum()).backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
