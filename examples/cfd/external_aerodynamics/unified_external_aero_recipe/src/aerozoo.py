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


"""AeroZoo-specific datapipe transforms.

AeroZoo samples are in SI units and the benchmark withholds the reference
scales, so the scales used here come from one sample's boundary inputs.
Recipe-local module registered into the global datapipe component registry;
import it before Hydra instantiation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from physicsnemo.datapipes.keys import as_nested_key
from physicsnemo.datapipes.registry import register
from physicsnemo.datapipes.transforms.mesh import SubsampleMesh
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import EFFECTIVE_MEASURE_KEY, set_point_measures


def _with(domain: DomainMesh, **parts) -> DomainMesh:
    """A copy of ``domain`` with some of interior / boundaries / global_data replaced."""
    return DomainMesh(
        interior=parts.get("interior", domain.interior),
        boundaries=parts.get("boundaries", domain.boundaries),
        global_data=parts.get("global_data", domain.global_data),
    )


@register()
class SubsampleInteriorPoints(MeshTransform):
    r"""Keep ``n_points`` random interior points; boundaries are left whole.

    The reader's own point subsample reads a contiguous block, which is one
    patch of the domain when the file stores points in solver order.

    ``volume_field`` names a ``point_data`` field with each point's cell
    volume. It becomes the points' measure, which weights the loss and the
    metrics, and is removed as a field.
    """

    def __init__(self, n_points: int, volume_field: str | None = None) -> None:
        super().__init__()
        self._sampler = SubsampleMesh(n_points=n_points)
        self.volume_field = volume_field
        self._generator: torch.Generator | None = None

    def __call__(self, mesh: Mesh) -> Mesh:
        raise TypeError("SubsampleInteriorPoints needs a DomainMesh, not a Mesh")

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Subsample the interior's points."""
        self._sampler._generator = self._generator
        interior = domain.interior
        if self.volume_field is not None:
            point_data = interior.point_data.clone()
            volume = point_data.pop(self.volume_field)
            interior = interior.with_data(point_data=point_data)
            set_point_measures(interior, volume.reshape(-1), dimension=3)
        return _with(domain, interior=self._sampler(interior))


@register()
class NonDimensionalizeByInlet(MeshTransform):
    r"""Scale fields by a speed taken from the sample's own inlet.

    The speed :math:`U` is the area-weighted RMS magnitude of the inlet's
    prescribed velocity. Velocities are divided by :math:`U` and gauge
    pressures by :math:`\tfrac12 \rho U^2`. ``global_data`` gets

    - ``speed``: :math:`U`, to map predictions back;
    - ``conditions``: :math:`\log_{10}(U L / \nu)` and the area-weighted
      mean inlet velocity over :math:`U` (four numbers), as model inputs.

    :math:`L` is ``global_data[length]``, the length the coordinates were
    divided by (``CanonicalFrameMesh(rescale=True, store_frame_as=...)``).

    Parameters
    ----------
    interior_fields : Mapping[str, str]
        ``interior.point_data`` field to ``"velocity"`` or ``"pressure"``.
    boundary_velocities : Mapping[str, str]
        Boundary name to a ``cell_data`` velocity field to scale.
    inlet : str
        Boundary whose velocity field (from ``boundary_velocities``)
        defines the speed.
    length, density, viscosity : str
        ``global_data`` keys.
    """

    def __init__(
        self,
        interior_fields: Mapping[str, str],
        boundary_velocities: Mapping[str, str],
        inlet: str = "inlet",
        length: str = "frame_length",
        density: str = "physics.density",
        viscosity: str = "physics.kinematic_viscosity",
    ) -> None:
        super().__init__()
        for name, kind in interior_fields.items():
            if kind not in ("velocity", "pressure"):
                raise ValueError(f"{name}: expected velocity or pressure, got {kind!r}")
        self.interior_fields = {as_nested_key(k): v for k, v in interior_fields.items()}
        self.boundary_velocities = {
            k: as_nested_key(v) for k, v in boundary_velocities.items()
        }
        self.inlet = inlet
        self.length, self.density, self.viscosity = (
            as_nested_key(k) for k in (length, density, viscosity)
        )

    def __call__(self, mesh: Mesh) -> Mesh:
        raise TypeError("NonDimensionalizeByInlet needs a DomainMesh, not a Mesh")

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Scale the fields and add ``speed`` and ``conditions``."""
        inlet = domain.boundaries[self.inlet]
        velocity = inlet.cell_data[self.boundary_velocities[self.inlet]].double()
        # float64 areas: float32 areas vanish on thin cells
        p = inlet.points.double()[inlet.cells]
        area = torch.linalg.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]).norm(dim=-1)
        speed = ((area * velocity.square().sum(-1)).sum() / area.sum()).sqrt()
        mean = (area[:, None] * velocity).sum(0) / area.sum() / speed
        g = domain.global_data
        reynolds = speed * g[self.length].double() / g[self.viscosity].double()
        dynamic_pressure = 0.5 * g[self.density].double() * speed**2
        scale = {"velocity": speed.float(), "pressure": dynamic_pressure.float()}

        point_data = domain.interior.point_data.clone()
        for key, kind in self.interior_fields.items():
            point_data[key] = point_data[key] / scale[kind]
        boundaries = dict(domain.boundaries.items())
        for name, key in self.boundary_velocities.items():
            cell_data = boundaries[name].cell_data.clone()
            cell_data[key] = cell_data[key] / scale["velocity"]
            boundaries[name] = boundaries[name].with_data(cell_data=cell_data)
        global_data = g.clone()
        global_data["speed"] = speed.float()
        global_data["conditions"] = torch.cat([reynolds.log10()[None], mean]).float()
        return _with(
            domain,
            interior=domain.interior.with_data(point_data=point_data),
            boundaries=boundaries,
            global_data=global_data,
        )


@register()
class MergeBoundaryPoints(MeshTransform):
    r"""Add one point-cloud boundary holding the points of several others.

    For models that read the boundary as one set of tokens. The new
    boundary ``name`` has the points of ``sources`` in order, their measures,
    and one ``point_data`` field ``features``: each of ``fields`` (zeros
    where a source lacks it) followed by a one-hot code of the source.

    Parameters
    ----------
    sources : Sequence[str]
        Point-cloud boundaries to merge.
    fields : Mapping[str, int]
        ``point_data`` field to its number of channels.
    name : str
        Name of the new boundary.
    """

    def __init__(
        self, sources: Sequence[str], fields: Mapping[str, int], name: str = "tokens"
    ) -> None:
        super().__init__()
        self.sources = list(sources)
        self.fields = {as_nested_key(k): int(v) for k, v in fields.items()}
        self.name = name

    def __call__(self, mesh: Mesh) -> Mesh:
        raise TypeError("MergeBoundaryPoints needs a DomainMesh, not a Mesh")

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Add the merged boundary."""
        features, measures = [], []
        for i, source in enumerate(self.sources):
            mesh = domain.boundaries[source]
            columns = [
                mesh.point_data[key].reshape(mesh.n_points, width)
                if key in mesh.point_data.keys(True, True)
                else mesh.points.new_zeros(mesh.n_points, width)
                for key, width in self.fields.items()
            ]
            code = mesh.points.new_zeros(mesh.n_points, len(self.sources))
            code[:, i] = 1
            features.append(torch.cat([*columns, code], dim=-1))
            measures.append(mesh.point_data[EFFECTIVE_MEASURE_KEY])
        merged = Mesh(
            points=torch.cat([domain.boundaries[s].points for s in self.sources]),
            point_data={
                "features": torch.cat(features),
                EFFECTIVE_MEASURE_KEY: torch.cat(measures),
            },
        )
        boundaries = dict(domain.boundaries.items())
        boundaries[self.name] = merged
        return _with(domain, boundaries=boundaries)
