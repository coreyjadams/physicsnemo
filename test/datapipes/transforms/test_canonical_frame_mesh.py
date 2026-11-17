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

"""Tests for the geometry-derived canonical frame."""

import pytest
import torch

from physicsnemo.datapipes.transforms.mesh import CanonicalFrameMesh
from physicsnemo.datapipes.transforms.mesh.transforms import canonical_frame
from physicsnemo.mesh import DomainMesh, Mesh


def _rotation(seed: int) -> torch.Tensor:
    q, _ = torch.linalg.qr(
        torch.randn(
            3, 3, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)
        )
    )
    return q * torch.linalg.det(q).sign()


def _body(n: int = 400, seed: int = 0) -> torch.Tensor:
    """Asymmetric cloud: distinct extents, skewed along every axis."""
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(n, 3, dtype=torch.float64, generator=g)
    return (u**2) * torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64)


def test_canonical_coordinates_survive_rotation_and_translation():
    x = _body()
    R, t = _rotation(1), torch.tensor([3.0, -2.0, 5.0], dtype=torch.float64)
    o1, a1, source = canonical_frame(x)
    o2, a2, _ = canonical_frame(x @ R.T + t)
    assert sorted(source.tolist()) == [-1, 0, 0]
    torch.testing.assert_close((x - o1) @ a1.T, (x @ R.T + t - o2) @ a2.T)
    torch.testing.assert_close(
        torch.linalg.det(a1), torch.tensor(1.0, dtype=torch.float64)
    )


def test_measure_weights_cancel_duplicated_points():
    x = _body()
    w = torch.ones(len(x), dtype=torch.float64)
    # Refine one region: duplicate 100 points, splitting their measure.
    x_ref = torch.cat([x, x[:100]])
    w_ref = torch.cat([w, w[:100]])
    w_ref[:100] = w_ref[-100:] = 0.5
    for a, b in zip(canonical_frame(x, w), canonical_frame(x_ref, w_ref)):
        torch.testing.assert_close(a, b)


def test_mirror_symmetric_body_gets_right_handed_frame():
    x = _body()
    x = torch.cat([x, x * torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)])
    R = _rotation(2)
    o1, a1, source = canonical_frame(x)
    o2, a2, _ = canonical_frame(x @ R.T)
    assert sorted(source.tolist()) == [-1, 0, 0]
    torch.testing.assert_close((x - o1) @ a1.T, (x @ R.T - o2) @ a2.T)


@pytest.mark.parametrize("scale", [(4.0, 1.0, 1.0), (1.0, 1.0, 1.0)])
def test_tie_breakers_settle_equal_variances(scale):
    # Square rod / cube: equal variances and no third moment.
    side = torch.linspace(-0.5, 0.5, 8, dtype=torch.float64)
    x = torch.cartesian_prod(side, side, side) * torch.tensor(
        scale, dtype=torch.float64
    )
    flow = torch.tensor([0.3, 0.8, 0.5], dtype=torch.float64)
    up = torch.tensor([0.1, -0.2, 0.9], dtype=torch.float64)
    R = _rotation(4)

    assert -2 in canonical_frame(x)[2].tolist()  # nothing to go on: arbitrary

    o1, a1, source = canonical_frame(x, tie_breakers=[flow, up])
    o2, a2, _ = canonical_frame(x @ R.T, tie_breakers=[R @ flow, R @ up])
    assert -2 not in source.tolist()
    torch.testing.assert_close((x - o1) @ a1.T, (x @ R.T - o2) @ a2.T)
    torch.testing.assert_close(
        torch.linalg.det(a2), torch.tensor(1.0, dtype=torch.float64)
    )


def test_domain_uses_named_boundary_and_rotates_fields():
    body = _body().float()
    R = _rotation(3).float()
    t = torch.tensor([1.0, 2.0, 3.0])
    query = torch.randn(50, 3)
    velocity = torch.randn(50, 3)
    u_inf = torch.tensor([1.0, 0.2, 0.0])

    def run(rot, shift):
        domain = DomainMesh(
            interior=Mesh(
                points=query @ rot.T + shift,
                point_data={"velocity": velocity @ rot.T},
            ),
            boundaries={"vehicle": Mesh(points=body @ rot.T + shift)},
            global_data={"U_inf": u_inf @ rot.T},
        )
        return CanonicalFrameMesh(
            boundary_name="vehicle",
            store_frame_as="frame",
            transform_point_data=True,
            transform_global_data=True,
        ).apply_to_domain(domain)

    a, b = run(torch.eye(3), torch.zeros(3)), run(R, t)
    kw = {"atol": 1e-4, "rtol": 1e-4}
    torch.testing.assert_close(a.interior.points, b.interior.points, **kw)
    torch.testing.assert_close(
        a.interior.point_data["velocity"], b.interior.point_data["velocity"], **kw
    )
    torch.testing.assert_close(a.global_data["U_inf"], b.global_data["U_inf"], **kw)
    # The stored frame maps canonical coordinates back to the input.
    torch.testing.assert_close(
        b.interior.points @ b.global_data["frame_axes"] + b.global_data["frame_origin"],
        query @ R.T + t,
        **kw,
    )


def test_domain_finds_detached_body_among_walls():
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    shell = sphere_icosahedral.load(subdivisions=2)
    shell = Mesh(points=4.0 * shell.points, cells=shell.cells)
    front = shell.cell_centroids[:, 0] < 0
    body = sphere_icosahedral.load(subdivisions=1)
    # A rod along z, off-center: equal variances in x-y, left to the tie-breakers.
    body_points = body.points * torch.tensor([0.3, 0.3, 1.0]) + torch.tensor(
        [0.5, 0.2, 0.0]
    )
    wall_points = torch.cat([shell.points, body_points])
    wall_cells = torch.cat([shell.cells[~front], body.cells + shell.n_points])
    velocity = torch.tensor([1.0, 0.3, 0.2]).expand(int(front.sum()), 3)
    query = torch.randn(50, 3)

    def run(rot, shift):
        domain = DomainMesh(
            interior=Mesh(points=query @ rot.T + shift),
            boundaries={
                "inlet": Mesh(
                    points=shell.points @ rot.T + shift,
                    cells=shell.cells[front],
                    cell_data={"velocity": velocity @ rot.T},
                ),
                "no_slip": Mesh(points=wall_points @ rot.T + shift, cells=wall_cells),
            },
        )
        return CanonicalFrameMesh(
            wall_names=["no_slip"],
            direction_fields={"inlet": "velocity"},
            rescale=True,
            store_frame_as="frame",
        ).apply_to_domain(domain)

    a = run(torch.eye(3), torch.zeros(3))
    # Moved, rotated and three times the size: same canonical coordinates.
    b = run(3 * _rotation(5).float(), torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(
        b.global_data["frame_length"], 3 * a.global_data["frame_length"]
    )
    # The origin is the body's centroid, not the enclosure's.
    torch.testing.assert_close(
        a.global_data["frame_origin"], torch.tensor([0.5, 0.2, 0.0]), atol=1e-5, rtol=0
    )
    assert -2 not in a.global_data["frame_source"].tolist()
    torch.testing.assert_close(a.interior.points, b.interior.points, atol=1e-4, rtol=0)
