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

Synthetic bodies with exact normals and area weights, the pipeline under
test (canonical frame of the body, then ``GeoTransolverPP``), the
transformations it must respect, and the error measure the notebooks report.
AeroZoo samples are used where a notebook asks for real data.
"""

import os
from pathlib import Path

import torch
from torch.nn.functional import normalize

from physicsnemo.datapipes.transforms.mesh import (
    CanonicalFrameMesh,
    SplitDetachedCells,
)
from physicsnemo.datapipes.transforms.mesh.transforms import canonical_frame
from physicsnemo.experimental.models.geotransolver_pp import GeoTransolverPP

# Inputs that must rotate with the problem, as (boundary, key) or a global key.
VECTORS = (
    ("body", "normals"),
    ("body", "vectors"),
    ("inlet", "normals"),
    ("inlet", "vectors"),
    "global_vectors",
)
AEROZOO = Path(os.environ.get("AEROZOO_DIR", "~/datasets/AeroZoo/samples")).expanduser()
AEROZOO_VELOCITY = ("prescribed", "dirichlet", "velocity")


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
    refining one region would. The measures sum to the body's area.
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
    # Each sample stands for 4 pi / n of the unit sphere it was drawn on.
    return points, normalize(normals, dim=-1), weight * area * (4 * torch.pi / n)


def make_problem(
    geometry="car",
    n_boundary=256,
    n_points=200,
    global_vectors=1,
    seed=0,
    boundary_seed=None,
    refine=1,
    dtype=torch.float64,
    min_component=1e-3,
):
    """One problem in the lab frame: a body, an inlet patch and query points.

    Boundary conditions are functions of position, so resampling the
    boundaries (``boundary_seed``, ``refine``) describes the same problem.
    The query points depend only on ``seed``. ``min_component`` is the
    tie-break threshold of the frame (see ``canonical_frame``).
    """
    seed_b = seed if boundary_seed is None else boundary_seed
    xb, nb, measure = sample_surface(geometry, n_boundary, seed_b, refine, dtype)
    up = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    g = torch.Generator().manual_seed(seed + 1000)
    shell, _, _ = sample_surface(geometry, n_points, seed + 1000, dtype=dtype)
    globals_ = torch.tensor(
        [[1.0, 0.1, 0.0], [0.0, 0.0, 1.0], [0.3, 1.0, 0.0]], dtype=dtype
    )
    # Inlet: a 4 x 4 patch upstream of the body with a sheared velocity.
    n_inlet = max(16, n_boundary // 4)
    gi = torch.Generator().manual_seed(seed_b + 2000)
    yz = 4 * torch.rand(n_inlet, 2, dtype=dtype, generator=gi) - 2
    xi = torch.cat([torch.full((n_inlet, 1), -4.0, dtype=dtype), yz], dim=-1)
    return {
        "points": shell * (1.05 + torch.rand(n_points, 1, dtype=dtype, generator=g)),
        "point_measure": torch.rand(n_points, dtype=dtype, generator=g) + 0.1,
        "boundaries": {
            "body": {
                "points": xb,
                "normals": nb,
                "measure": measure,
                "scalars": torch.stack([torch.tanh(xb[:, 0]), nb[:, 2]], dim=-1),
                "vectors": torch.linalg.cross(nb, up.expand_as(nb))[:, None],
            },
            "inlet": {
                "points": xi,
                "normals": torch.tensor([1.0, 0.0, 0.0], dtype=dtype).expand(
                    n_inlet, 3
                ),
                "measure": torch.full((n_inlet,), 16.0 / n_inlet, dtype=dtype),
                "vectors": (globals_[0] * (1 + 0.2 * yz[:, :1]))[:, None],
            },
        },
        "global_scalars": torch.tensor([0.3], dtype=dtype),
        "global_vectors": globals_[:global_vectors],
        "min_component": min_component,
    }


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


def make_model(
    global_vectors=1, n_hidden=32, seed=0, dtype=torch.float64, n_neighbors=8192
):
    """Small untrained model matching :func:`make_problem`.

    ``n_neighbors`` is large so the neighbor attention sees every boundary
    sample in range; notebook 3 shows what a smaller limit does.
    """
    torch.manual_seed(seed)
    model = GeoTransolverPP(
        out_dim=8,  # two scalars and two vectors
        boundary_features={"body": 2 + 3, "inlet": 3},
        global_features=1 + 3 * global_vectors,
        n_hidden=n_hidden,
        n_layers=2,
        n_head=4,
        n_latents=8,
        radius=1.0,
        n_neighbors=n_neighbors,
    )
    return randomize_zero_init(model, seed).to(dtype).eval()


def random_rotation(seed=0, dtype=torch.float64):
    """A random proper rotation matrix."""
    q, _ = torch.linalg.qr(
        torch.randn(3, 3, dtype=dtype, generator=torch.Generator().manual_seed(seed))
    )
    return q * torch.linalg.det(q).sign()


def body_frame(problem):
    """Origin, axes, length and axis sources of the body's canonical frame.

    Ties in the body's shape go to the global vectors, then to the offset of
    all boundaries' centroid from the body's.
    """
    body = problem["boundaries"]["body"]
    x = torch.cat([b["points"] for b in problem["boundaries"].values()])
    w = torch.cat([b["measure"] for b in problem["boundaries"].values()])
    center = (w[:, None] * x).sum(0) / w.sum()
    size = ((w * (x - center).square().sum(-1)).sum() / w.sum()).sqrt()
    wb = body["measure"] / body["measure"].sum()
    offset = (center - (wb[:, None] * body["points"]).sum(0)) / size
    tie_breakers = [*normalize(problem["global_vectors"], dim=-1), offset]
    origin, axes, source = canonical_frame(
        body["points"],
        body["measure"],
        tie_breakers,
        min_component=problem["min_component"],
    )
    length = (wb * (body["points"] - origin).square().sum(-1)).sum().sqrt()
    return origin, axes, length, source


def canonical(problem):
    """Model inputs (batch of 1) in the body's frame, and the frame's axes."""
    origin, axes, length, _ = body_frame(problem)

    def position(x):
        return ((x - origin) @ axes.T / length)[None]

    boundaries = {}
    for name, b in problem["boundaries"].items():
        features = [b["scalars"]] if "scalars" in b else []
        features.append((b["vectors"] @ axes.T).flatten(-2))
        boundaries[name] = {
            "points": position(b["points"]),
            "normals": (b["normals"] @ axes.T)[None],
            "measure": (b["measure"] / length**2)[None],
            "features": torch.cat(features, dim=-1)[None],
        }
    inputs = {
        "points": position(problem["points"]),
        "boundaries": boundaries,
        "global_features": torch.cat(
            [problem["global_scalars"], (problem["global_vectors"] @ axes.T).flatten()]
        )[None],
        "point_measure": problem["point_measure"][None],
    }
    return inputs, axes


def predict(model, problem):
    """Scalars and lab-frame vectors of the pipeline: frame, then model."""
    inputs, axes = canonical(problem)
    out = model(**inputs)[0]
    return out[:, :2], out[:, 2:].unflatten(-1, (2, 3)) @ axes


def move(problem, matrix=None, shift=None, scale=1.0, skip=None):
    """Apply ``x -> scale * x @ matrix.T + shift`` to a whole problem.

    ``skip`` names one entry of :data:`VECTORS` to leave unrotated.
    """

    def position(x):
        x = scale * x
        x = x if matrix is None else x @ matrix.T
        return x if shift is None else x + shift

    def vector(x, name):
        return x if matrix is None or name == skip else x @ matrix.T

    out = dict(problem, points=position(problem["points"]), boundaries={})
    out["global_vectors"] = vector(problem["global_vectors"], "global_vectors")
    for name, b in problem["boundaries"].items():
        out["boundaries"][name] = dict(
            b,
            points=position(b["points"]),
            normals=vector(b["normals"], (name, "normals")),
            vectors=vector(b["vectors"], (name, "vectors")),
            measure=scale**2 * b["measure"],
        )
    return out


def _tokens(problem, which):
    """The per-token tensors of the query points or of one boundary."""
    if which == "points":
        return {k: problem[k] for k in ("points", "point_measure")}
    return problem["boundaries"][which]


def _with_tokens(problem, which, tokens):
    if which == "points":
        return dict(problem, **tokens)
    return dict(problem, boundaries=dict(problem["boundaries"], **{which: tokens}))


def permute(problem, which, seed=0):
    """Shuffle the points or one boundary's samples. Returns (problem, restore)."""
    tokens = _tokens(problem, which)
    n = len(next(iter(tokens.values())))
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    out = _with_tokens(problem, which, {k: v[order] for k, v in tokens.items()})
    return out, (order.argsort() if which == "points" else None)


def duplicate(problem, which, count):
    """Repeat the first ``count`` tokens, splitting their measure in two."""
    tokens = dict(_tokens(problem, which))
    key = "point_measure" if which == "points" else "measure"
    n = len(tokens[key])
    tokens[key] = torch.cat([tokens[key][:count] / 2, tokens[key][count:]])
    tokens = {k: torch.cat([v, v[:count]]) for k, v in tokens.items()}
    out = _with_tokens(problem, which, tokens)
    return out, (torch.arange(n) if which == "points" else None)


def axes_moved_deg(problem, reference):
    """Largest angle between matching frame axes of two problems, in degrees."""
    cos = (body_frame(problem)[1] * body_frame(reference)[1]).sum(dim=-1)
    return torch.rad2deg(torch.acos(cos.clamp(-1, 1))).max().item()


def rel_err(actual, expected):
    """Largest absolute difference, relative to the largest expected value."""
    scale = expected.abs().max().clamp_min(torch.finfo(expected.dtype).tiny)
    return ((actual - expected).abs().max() / scale).item()


@torch.no_grad()
def compare(model, problem, moved, matrix=None, restore=None):
    """Errors of (scalars, vectors) on ``moved`` against the expected result.

    Scalars must not change; vectors must rotate by ``matrix``. ``restore``
    indexes the moved outputs back to the original point order.
    """
    scalars, vectors = predict(model, problem)
    moved_scalars, moved_vectors = predict(model, moved)
    if restore is not None:
        moved_scalars, moved_vectors = moved_scalars[restore], moved_vectors[restore]
    if matrix is not None:
        vectors = vectors @ matrix.T
    return rel_err(moved_scalars, scalars), rel_err(moved_vectors, vectors)


def aerozoo_cases(limit=None):
    """Downloaded AeroZoo samples, smallest first (empty if there are none)."""
    cases = sorted(AEROZOO.glob("*/domain.pdmsh"), key=lambda p: p.stat().st_size)
    return cases[:limit]


def aerozoo_transform():
    """Body frame of an AeroZoo sample, from benchmark-legal inputs only.

    The body's cells are first moved out of the wall boundaries into
    ``body_no_slip`` and ``body_slip``.
    """
    return _Chain(SplitDetachedCells(["no_slip", "slip"]), _aerozoo_frame())


class _Chain:
    """Apply domain transforms in order."""

    def __init__(self, *transforms):
        self.transforms = transforms

    def apply_to_domain(self, domain):
        for transform in self.transforms:
            domain = transform.apply_to_domain(domain)
        return domain


def _aerozoo_frame():
    return CanonicalFrameMesh(
        boundary_name=["body_no_slip", "body_slip"],
        direction_fields={"inlet": AEROZOO_VELOCITY},
        rescale=True,
        store_frame_as="frame",
        transform_point_data=True,
        transform_cell_data=True,
    )


def load_aerozoo(path, stride=1000):
    """An AeroZoo sample with every ``stride``-th interior point."""
    from physicsnemo.mesh import DomainMesh

    domain = DomainMesh.load(path)
    keep = torch.arange(0, domain.interior.n_points, stride)
    return DomainMesh(
        interior=domain.interior.slice_points(keep), boundaries=domain.boundaries
    )


def aerozoo_inputs(
    domain, n_boundary=2048, n_points=512, seed=0, boundary_seed=None, systematic=True
):
    """``GeoTransolverPP`` inputs from a canonical-frame AeroZoo sample.

    The boundary tokens are ``n_boundary`` cells in total, each weighted to
    stand for the cells left out. ``systematic`` draws them as ISLA does
    (:func:`physicsnemo.experimental.models.isla._sampling.sample_boundary_cells`):
    equal shares per boundary, probability proportional to area, spread
    along a space-filling curve. Otherwise each boundary gets an equal share
    of uniformly random cells. ``boundary_seed`` changes the draw only. The
    inlet velocity is divided by its area-weighted RMS speed, a scale taken
    from the sample itself.
    """
    from physicsnemo.experimental.models.isla._sampling import sample_boundary_cells

    g = torch.Generator().manual_seed(seed if boundary_seed is None else boundary_seed)
    meshes = {n: m for n, m in domain.boundaries.items() if m.n_cells}
    areas = {n: _faces_float64(m)[1] for n, m in meshes.items()}
    if systematic:
        start = torch.rand((), dtype=torch.float64, generator=g)
        centroids = [_faces_float64(m)[0] for m in meshes.values()]
        kept, weight = sample_boundary_cells(
            list(areas.values()), centroids, n_boundary, start
        )
        offsets = torch.tensor([0, *(len(a) for a in areas.values())]).cumsum(0)
    boundaries = {}
    for i, (name, mesh) in enumerate(meshes.items()):
        area = areas[name]
        if systematic:
            here = (kept >= offsets[i]) & (kept < offsets[i + 1])
            keep, measure = kept[here] - offsets[i], weight[here]
        else:
            keep = torch.randperm(mesh.n_cells, generator=g)[
                : n_boundary // len(meshes)
            ]
            measure = area[keep] * area.sum() / area[keep].sum()
        boundaries[name] = {
            "points": mesh.cell_centroids[keep][None],
            "normals": mesh.cell_normals[keep][None],
            "measure": measure.float()[None],
        }
        if name == "inlet":
            v = mesh.cell_data[AEROZOO_VELOCITY]
            speed = ((area * v.square().sum(-1)).sum() / area.sum()).sqrt().float()
            boundaries[name]["features"] = (v[keep] / speed)[None]
    g = torch.Generator().manual_seed(seed)
    keep = torch.randperm(domain.interior.n_points, generator=g)[:n_points]
    volume = domain.interior.point_data["_target_quadrature_measure"]
    return {
        "points": domain.interior.points[keep][None],
        "boundaries": boundaries,
        "point_measure": volume[keep].reshape(1, -1),
    }


def _faces_float64(mesh):
    """Cell centroids and areas in float64 (float32 areas vanish on thin cells)."""
    p = mesh.points.double()[mesh.cells]
    area = torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]).norm(dim=-1) / 2
    return p.mean(dim=1), area


def aerozoo_moved(domain, rotation):
    """The sample rotated, shifted by about half its size, and three times as large.

    A larger shift costs float32 digits of the positions near the body."""
    size = domain.interior.points.abs().max()
    return (
        domain.translate(0.1 * size * torch.tensor([3.0, -2.0, 5.0]))
        .transform(rotation, transform_point_data=True, transform_cell_data=True)
        .scale(3.0)
    )


def aerozoo_isla(n_hidden=32, seed=0):
    """Small untrained ISLA interior model on the same boundaries, for reference."""
    from physicsnemo.experimental.models.isla import ISLA

    torch.manual_seed(seed)
    return ISLA(
        output_schema={"pressure": {"rank": 0}, "velocity": {"rank": 1}},
        boundary_schemas={
            "inlet": {".".join(AEROZOO_VELOCITY): {"rank": 1}},
            "outlet": {},
            "no_slip": {},
            "slip": {},
        },
        interior=True,
        hidden=n_hidden,
        n_layers=2,
        n_slices=8,
        n_boundary_tokens=2048,
    ).eval()


def isla_predict(model, domain, n_points=512):
    """ISLA's (pressure, velocity) at the first interior points, lab frame.

    ISLA takes the sample as it is: it needs no canonical frame.
    """
    from physicsnemo.mesh import Mesh

    meshes = {}
    for name, mesh in domain.boundaries.items():
        if mesh.n_cells == 0:
            continue
        data = {}
        if name == "inlet":
            data = {
                "prescribed": {
                    "dirichlet": {"velocity": mesh.cell_data[AEROZOO_VELOCITY]}
                }
            }
        meshes[name] = Mesh(points=mesh.points, cells=mesh.cells, cell_data=data)
    out = model(
        boundary_meshes=meshes, prediction_points=domain.interior.points[:n_points]
    )
    return out.point_data["pressure"], out.point_data["velocity"]


def aerozoo_model(n_hidden=32, seed=0):
    """Small untrained model for :func:`aerozoo_inputs`."""
    torch.manual_seed(seed)
    model = GeoTransolverPP(
        out_dim=4,  # a scalar and a vector
        boundary_features={
            "inlet": 3,
            "outlet": 0,
            "no_slip": 0,
            "slip": 0,
            "body_no_slip": 0,
            "body_slip": 0,
        },
        n_hidden=n_hidden,
        n_layers=2,
        n_head=4,
        n_latents=8,
        radius=1.0,
        n_neighbors=64,
    )
    return randomize_zero_init(model, seed).eval()


def show(mo, title, rows):
    """A titled table; also printed when the notebook runs as a script."""
    if not mo.running_in_notebook():
        print(f"\n{title}")
        for row in rows:
            print("  " + " | ".join(f"{k}: {v}" for k, v in row.items()))
    return mo.vstack([mo.md(f"**{title}**"), mo.ui.table(rows, selection=None)])
