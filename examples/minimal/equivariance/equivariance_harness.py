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

"""Shared helpers for the equivariance notebooks.

Synthetic bodies with exact normals and area weights, the transformations a
symmetric model must respect, and the error measure the notebooks report.
"""

import torch
from torch.nn.functional import normalize

from physicsnemo.experimental.models.geotransolver_pp import GeoTransolverPP

POSITIONS = ("points", "boundary_points")
VECTORS = ("boundary_normals", "boundary_vectors", "global_vectors")


def _car(p):
    """Wider and taller toward +x, bulged upward: one mirror plane (y)."""
    x, y, z = p.unbind(-1)
    s = 1 + 0.25 * torch.tanh(x)
    return torch.stack([x, y * s, z * s + 0.15 * z**2], dim=-1)


def _blob(p):
    """The car with the mirror plane broken: no symmetry."""
    x, y, z = p.unbind(-1)
    s = 1 + 0.25 * torch.tanh(x)
    return torch.stack(
        [x + 0.1 * y * z, y * s + 0.2 * x**2, z * s + 0.15 * z**2], dim=-1
    )


# name -> (ellipsoid semi-axes, smooth deformation applied to it)
GEOMETRIES = {
    "car": ((2.0, 1.0, 0.6), _car),
    "blob": ((2.0, 1.0, 0.6), _blob),
    "ellipsoid": ((2.0, 1.0, 0.6), None),  # three mirror planes: no sign
    "spheroid": ((2.0, 1.0, 1.0), None),  # two equal axes: no unique frame
}


def sample_surface(name, n, seed=0, refine=1, dtype=torch.float64):
    """Random samples of a body: ``points``, unit ``normals``, area ``measure``.

    ``refine`` > 1 samples the half with (pre-deformation) x > 0 ``refine``
    times as densely and divides those samples' measure to match, as a mesher
    refining one region would.
    """
    axes, deform = GEOMETRIES[name]
    axes = torch.tensor(axes, dtype=dtype)
    g = torch.Generator().manual_seed(seed)
    u = normalize(torch.randn(n, 3, dtype=dtype, generator=g), dim=-1)
    if refine > 1:
        extra = torch.randn((refine - 1) * n, 3, dtype=dtype, generator=g)
        u = torch.cat([u, normalize(extra[extra[:, 0] > 0], dim=-1)])
    weight = torch.where(u[:, 0] > 0, 1.0 / refine, 1.0).to(dtype)

    # Sphere -> ellipsoid: area grows by prod(axes) * |u / axes|.
    points = u * axes
    normals = u / axes
    area = axes.prod() * normals.norm(dim=-1)
    if deform is not None:
        # A smooth map with Jacobian J turns a normal n into J^-T n and grows
        # area by |det J| |J^-T n|.
        jac = torch.func.vmap(torch.func.jacrev(deform))(points)
        normals = torch.linalg.solve(
            jac.transpose(-1, -2), normalize(normals, dim=-1)[..., None]
        )[..., 0]
        area = area * torch.linalg.det(jac).abs() * normals.norm(dim=-1)
        points = deform(points)
    return points, normalize(normals, dim=-1), weight * area


def make_problem(
    geometry="car",
    n_boundary=256,
    n_points=200,
    global_vectors=1,
    seed=0,
    boundary_seed=None,
    refine=1,
    dtype=torch.float64,
):
    """Inputs of :class:`GeoTransolverPP` (batch of 1) for one body.

    Boundary conditions are functions of position on the body, so resampling
    the boundary (``boundary_seed``, ``refine``) describes the same problem.
    The query points depend only on ``seed``.
    """
    seed_b = seed if boundary_seed is None else boundary_seed
    xb, nb, measure = sample_surface(geometry, n_boundary, seed_b, refine, dtype)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    g = torch.Generator().manual_seed(seed + 1000)
    shell, _, _ = sample_surface(geometry, n_points, seed + 1000, dtype=dtype)
    globals_ = torch.tensor(
        [[1.0, 0.1, 0.0], [0.0, 0.0, 1.0], [0.3, 1.0, 0.0]], dtype=dtype
    )
    inputs = {
        "points": shell * (1.05 + torch.rand(n_points, 1, dtype=dtype, generator=g)),
        "boundary_points": xb,
        "boundary_normals": nb,
        "boundary_measure": measure,
        "boundary_scalars": torch.stack([torch.tanh(xb[:, 0]), nb[:, 2]], dim=-1),
        "boundary_vectors": torch.linalg.cross(nb, up.expand_as(nb))[:, None],
        "global_scalars": torch.tensor([0.3], dtype=dtype),
        "global_vectors": globals_[:global_vectors],
        "point_measure": torch.rand(n_points, dtype=dtype, generator=g) + 0.1,
    }
    return {k: v[None] for k, v in inputs.items()}


def randomize_zero_init(module, seed=0):
    """Give zero-initialised parameters random values, as training would.

    The condition-dependent norms start at zero, so an untrained model ignores
    its global conditions there and a symmetry test of them would be empty.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            if not p.any():
                p.copy_(0.1 * torch.randn(p.shape, generator=g))
    return module


def make_model(global_vectors=1, n_hidden=32, seed=0, dtype=torch.float64):
    """Small untrained model matching :func:`make_problem`."""
    torch.manual_seed(seed)
    model = GeoTransolverPP(
        out_scalars=2,
        out_vectors=2,
        boundary_scalars=2,
        boundary_vectors=1,
        global_scalars=1,
        global_vectors=global_vectors,
        n_hidden=n_hidden,
        n_layers=2,
        n_head=4,
        n_latents=8,
        n_neighbors=4,
    )
    return randomize_zero_init(model, seed).to(dtype).eval()


def random_rotation(seed=0, dtype=torch.float64):
    """A random proper rotation matrix."""
    q, _ = torch.linalg.qr(
        torch.randn(3, 3, dtype=dtype, generator=torch.Generator().manual_seed(seed))
    )
    return q * torch.linalg.det(q).sign()


def move(inputs, matrix=None, shift=None, scale=1.0):
    """Apply ``x -> scale * x @ matrix.T + shift`` to a whole problem."""
    out = dict(inputs)
    for name in POSITIONS:
        x = scale * inputs[name]
        x = x if matrix is None else x @ matrix.T
        out[name] = x if shift is None else x + shift
    if matrix is not None:
        for name in VECTORS:
            out[name] = inputs[name] @ matrix.T
    out["boundary_measure"] = scale**2 * inputs["boundary_measure"]
    return out


def _token_keys(which):
    if which == "points":
        return ["points", "point_measure"]
    return [
        f"boundary_{k}" for k in ("points", "normals", "measure", "scalars", "vectors")
    ]


def permute(inputs, which, seed=0):
    """Shuffle the points or the boundary samples. Returns (inputs, restore)."""
    keys = _token_keys(which)
    order = torch.randperm(
        inputs[keys[0]].shape[1], generator=torch.Generator().manual_seed(seed)
    )
    out = dict(inputs, **{k: inputs[k][:, order] for k in keys})
    return out, (order.argsort() if which == "points" else None)


def duplicate(inputs, which, count):
    """Repeat the first ``count`` tokens, splitting their measure in two."""
    keys = _token_keys(which)
    out = dict(inputs)
    for k in keys:
        x = inputs[k]
        if k.endswith("measure"):
            x = torch.cat([x[:, :count] / 2, x[:, count:]], dim=1)
        out[k] = torch.cat([x, x[:, :count]], dim=1)
    n = inputs[keys[0]].shape[1]
    return out, (torch.arange(n) if which == "points" else None)


def rel_err(actual, expected):
    """Largest absolute difference, relative to the largest expected value."""
    scale = expected.abs().max().clamp_min(torch.finfo(expected.dtype).tiny)
    return ((actual - expected).abs().max() / scale).item()


@torch.no_grad()
def compare(model, inputs, moved, matrix=None, restore=None):
    """Errors of (scalars, vectors) on ``moved`` against the expected result.

    Scalars must not change; vectors must rotate by ``matrix``. ``restore``
    indexes the moved outputs back to the original point order.
    """
    scalars, vectors = model(**inputs)
    moved_scalars, moved_vectors = model(**moved)
    if restore is not None:
        moved_scalars, moved_vectors = (
            moved_scalars[:, restore],
            moved_vectors[:, restore],
        )
    if matrix is not None:
        vectors = vectors @ matrix.T
    return rel_err(moved_scalars, scalars), rel_err(moved_vectors, vectors)


def show(mo, title, rows):
    """A titled table; also printed when the notebook runs as a script."""
    if not mo.running_in_notebook():
        print(f"\n{title}")
        for row in rows:
            print("  " + " | ".join(f"{k}: {v}" for k, v in row.items()))
    return mo.vstack([mo.md(f"**{title}**"), mo.ui.table(rows, selection=None)])
