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

"""AeroZoo configs end to end on a synthetic sample in the AeroZoo layout.

Runs the real ``datasets/aerozoo_volume.yaml`` transforms (no reader), the
recipe collate, and each AeroZoo model config at a small size.
"""

from __future__ import annotations

import aerozoo  # noqa: F401  (registers the AeroZoo transforms)
import hydra
import pytest
import torch
from collate import build_collate_fn
from datasets import apply_model_config, load_dataset_config
from omegaconf import OmegaConf
from test_synthetic_configs import _RECIPE_ROOT, _compose_train_cfg

from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

MODELS = {
    "geotransolver_pp_volume_aerozoo": {"n_hidden": 32, "n_layers": 2, "n_head": 4},
    "geotransolver_volume_aerozoo": {
        "n_hidden": 32,
        "n_layers": 2,
        "n_head": 4,
        "slice_num": 8,
        "n_hidden_local": 8,
    },
    "geotransolver_bc_volume_aerozoo": {
        "n_hidden": 32,
        "n_layers": 2,
        "n_head": 4,
        "slice_num": 8,
    },
}


def _sample(rotation: torch.Tensor | None = None, n_points: int = 300) -> DomainMesh:
    """A lopsided body inside a sphere: inlet and outlet halves, no enclosure wall."""
    shell = sphere_icosahedral.load(subdivisions=2)
    shell_points = 5.0 * shell.points
    front = shell.cell_centroids[:, 0] < 0
    body = sphere_icosahedral.load(subdivisions=1)
    body_points = body.points * torch.tensor([1.0, 0.6, 0.3]) + 0.1 * body.points**2
    g = torch.Generator().manual_seed(0)
    interior = 4.0 * torch.rand(n_points, 3, generator=g) - 2.0
    flow = torch.tensor([10.0, 1.0, 0.0])
    velocity = flow + torch.randn(n_points, 3, generator=g)
    if rotation is not None:
        shell_points, body_points = shell_points @ rotation.T, body_points @ rotation.T
        interior, velocity = interior @ rotation.T, velocity @ rotation.T
        flow = flow @ rotation.T

    def wall(points, cells):
        zeros = torch.zeros(len(cells), 3)
        return Mesh(
            points=points,
            cells=cells,
            cell_data={"prescribed": {"dirichlet": {"velocity": zeros}}},
        )

    return DomainMesh(
        interior=Mesh(
            points=interior,
            point_data={
                "solution": {
                    "velocity": velocity,
                    "gauge_pressure": torch.randn(n_points, generator=g),
                },
                "_target_quadrature_measure": torch.rand(n_points, generator=g) + 0.1,
            },
        ),
        boundaries={
            "inlet": Mesh(
                points=shell_points,
                cells=shell.cells[front],
                cell_data={
                    "prescribed": {
                        "dirichlet": {"velocity": flow.expand(int(front.sum()), 3)}
                    }
                },
            ),
            "outlet": Mesh(points=shell_points, cells=shell.cells[~front]),
            "no_slip": wall(body_points, body.cells),
            # No slip cells at all, as in many AeroZoo samples.
            "slip": Mesh(points=body_points, cells=body.cells[:0]),
        },
        global_data={
            "physics": {
                "density": torch.tensor(1.225),
                "kinematic_viscosity": torch.tensor(1.5e-5),
            }
        },
    )


def _pipeline(model: str, overrides: list[str] | None = None):
    """The dataset's transforms and the collate for one model config."""
    cfg = _compose_train_cfg(
        model,
        "aerozoo_volume",
        overrides=["sampling_resolution=200", *(overrides or [])],
    )
    dataset_cfg = load_dataset_config(_RECIPE_ROOT / "datasets" / "aerozoo_volume.yaml")
    dataset_cfg = OmegaConf.merge(
        {"sampling_resolution": cfg.sampling_resolution, "boundary_tokens": 64},
        apply_model_config(dataset_cfg, cfg),
        {"boundary_tokens": 64, "dataset_paths": {"aerozoo": "unused"}},
    )
    transforms = [hydra.utils.instantiate(t) for t in dataset_cfg.pipeline.transforms]
    targets = OmegaConf.to_container(dataset_cfg.targets)
    collate = build_collate_fn(
        cfg.input_type,
        OmegaConf.to_container(cfg.forward_kwargs, resolve=True),
        targets,
    )
    return cfg, transforms, collate


def _run(transforms, domain: DomainMesh) -> DomainMesh:
    for transform in transforms:
        transform.set_generator(torch.Generator().manual_seed(0))
        domain = transform.apply_to_domain(domain)
    return domain


@pytest.mark.parametrize("model", list(MODELS))
def test_aerozoo_config_forward(model: str) -> None:
    """Pipeline, collate and a small model give finite outputs of the target shape."""
    cfg, transforms, collate = _pipeline(model)
    batch = collate([(_run(transforms, _sample()), None)])
    net = hydra.utils.instantiate(OmegaConf.merge(cfg.model, MODELS[model])).eval()
    with torch.no_grad():
        out = net(**batch["forward_kwargs"])
    assert out.shape == (1, 200, 4)
    assert torch.isfinite(out).all()
    assert batch["targets"]["velocity"].shape == (1, 200, 3)
    assert batch["target_measure"].shape == (1, 200)


def test_aerozoo_pipeline_scales_and_frame() -> None:
    """A rotated sample gives the same pipeline output; the scales are the sample's."""
    _, transforms, _ = _pipeline("geotransolver_pp_volume_aerozoo")
    q, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=torch.Generator().manual_seed(1))
    )
    q = q * torch.linalg.det(q).sign()
    a, b = _run(transforms, _sample()), _run(transforms, _sample(q))
    # The body is found, and everything is in its frame: a rotated sample
    # gives the same inputs and targets.
    assert a.boundaries["body_no_slip_tokens"].n_points == 64
    assert a.boundaries["no_slip_tokens"].n_points == 0
    assert -2 not in a.global_data["frame_source"].tolist()
    for x, y in (
        (a.interior.points, b.interior.points),
        (a.interior.point_data["velocity"], b.interior.point_data["velocity"]),
        (a.interior.point_data["sdf"], b.interior.point_data["sdf"]),
        (a.global_data["conditions"], b.global_data["conditions"]),
        (a.boundaries["tokens"].points, b.boundaries["tokens"].points),
    ):
        torch.testing.assert_close(x, y, atol=2e-4, rtol=0)
    # The inlet speed is the unit of velocity.
    inlet_speed = a.boundaries["inlet_tokens"].point_data["velocity"].norm(dim=-1)
    torch.testing.assert_close(inlet_speed, torch.ones(64), atol=1e-5, rtol=0)
    assert (a.interior.point_data["sdf"] >= 0).all()
    # Inlet and outlet are equal halves of a sphere crossed by a uniform flow.
    assert a.global_data["conditions"].shape == (5,)
    assert 0.3 < a.global_data["conditions"][4] < 0.7


def test_aerozoo_lab_axes_override() -> None:
    """``canonical_axes=false`` keeps the stored axes."""
    _, transforms, _ = _pipeline(
        "geotransolver_volume_aerozoo", ["+dataset_overrides.canonical_axes=false"]
    )
    out = _run(transforms, _sample())
    torch.testing.assert_close(out.global_data["frame_axes"], torch.eye(3))


def test_rms_metric_is_the_error_vectors_rms_length() -> None:
    """``rms`` weights by measure and sums a vector's components."""
    from metrics import MetricCalculator
    from tensordict import TensorDict

    target = TensorDict({"velocity": torch.zeros(1, 2, 3)}, batch_size=[1, 2])
    pred = TensorDict(
        {"velocity": torch.tensor([[[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]]])},
        batch_size=[1, 2],
    )
    calc = MetricCalculator({"velocity": "vector"}, metrics=["rms"])
    # |error| = 5 on the first point, which holds 1/4 of the volume.
    out = calc(pred, target, torch.tensor([[1.0, 3.0]]))
    torch.testing.assert_close(out["velocity_rms"], torch.tensor(2.5))
