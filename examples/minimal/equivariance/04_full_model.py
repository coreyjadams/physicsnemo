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

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys

    import marimo as mo
    import torch

    sys.path.insert(0, str(mo.notebook_dir()))
    import equivariance_harness as h

    return h, mo, torch


@app.cell
def _(mo):
    mo.md("""
    # 4. Full model

    `GeoTransolverPP`, untrained, on one synthetic body. Each row changes
    the problem in a way that should leave the answer alone (scalars
    unchanged, vectors rotated with the problem) and reports the largest
    deviation relative to the largest output.

    - **round-off**: holds by construction.
    - **sampling noise**: the boundary was sampled differently; the error
      should shrink as samples are added (second table).
    - **not enforced**: reflections are deliberately not a symmetry.
    """)
    return


@app.cell
def _(h, mo):
    geometry = mo.ui.dropdown(list(h.GEOMETRIES), value="car", label="Body")
    global_vectors = mo.ui.dropdown(["0", "1", "2"], value="1", label="Global vectors")
    n_boundary = mo.ui.slider(
        steps=[64, 256, 1024, 4096], value=256, label="Boundary samples"
    )
    dtype = mo.ui.dropdown(["float64", "float32"], value="float64", label="Precision")
    seed = mo.ui.number(0, 100, value=0, label="Seed")
    mo.hstack([geometry, global_vectors, n_boundary, dtype, seed], justify="start")
    return dtype, geometry, global_vectors, n_boundary, seed


@app.cell
def _(dtype, geometry, global_vectors, h, seed, torch):
    precision = getattr(torch, dtype.value)
    model = h.make_model(int(global_vectors.value), seed=seed.value, dtype=precision)

    def problem(n_boundary, **kwargs):
        return h.make_problem(
            geometry.value,
            n_boundary,
            global_vectors=int(global_vectors.value),
            seed=seed.value,
            dtype=precision,
            **kwargs,
        )

    return model, precision, problem


@app.cell
def _(h, mo, model, n_boundary, precision, problem, seed, torch):
    def symmetry_rows():
        inputs = problem(n_boundary.value)
        rotation = h.random_rotation(seed.value + 1, precision)
        mirror = rotation @ torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=precision))
        shift = torch.tensor([3.0, -2.0, 5.0], dtype=precision)
        m = n_boundary.value
        # (moved inputs, no rotation, index restoring the original point order)
        moved, restore = h.permute(inputs, "points")
        shuffled = (moved, None, restore)
        moved, restore = h.duplicate(inputs, "points", 50)
        split = (moved, None, restore)
        # (name, expected, moved inputs, matrix the vectors rotate by, restore)
        cases = [
            ("rotation", "round-off", h.move(inputs, rotation), rotation, None),
            ("translation", "round-off", h.move(inputs, shift=shift), None, None),
            (
                "rotation + translation",
                "round-off",
                h.move(inputs, rotation, shift),
                rotation,
                None,
            ),
            ("all lengths x 7", "round-off", h.move(inputs, scale=7.0), None, None),
            ("points shuffled", "round-off", *shuffled),
            (
                "boundary shuffled",
                "round-off",
                h.permute(inputs, "boundary")[0],
                None,
                None,
            ),
            ("points split, measure shared", "round-off", *split),
            (
                "boundary split, measure shared",
                "round-off",
                h.duplicate(inputs, "boundary", m // 4)[0],
                None,
                None,
            ),
            (
                "boundary resampled",
                "sampling noise",
                problem(m, boundary_seed=seed.value + 7),
                None,
                None,
            ),
            (
                "boundary refined x 4 on one half",
                "sampling noise",
                problem(m, refine=4),
                None,
                None,
            ),
            ("reflection", "not enforced", h.move(inputs, mirror), mirror, None),
        ]
        rows = []
        for name, expected, moved, matrix, restore in cases:
            scalars, vectors = h.compare(model, inputs, moved, matrix, restore)
            rows.append(
                {
                    "change": name,
                    "expected": expected,
                    "scalars": f"{scalars:.1e}",
                    "vectors": f"{vectors:.1e}",
                }
            )
        return rows

    h.show(mo, "Symmetry errors", symmetry_rows())
    return


@app.cell
def _(h, mo, model, problem):
    def resampling_rows():
        reference = problem(16384, boundary_seed=12345)
        rows = []
        for m in (64, 256, 1024, 4096):
            scalars, vectors = h.compare(model, reference, problem(m))
            rows.append(
                {
                    "boundary samples": m,
                    "scalars": f"{scalars:.1e}",
                    "vectors": f"{vectors:.1e}",
                }
            )
        return rows

    h.show(
        mo,
        "Boundary resampling: difference from a 16384-sample boundary",
        resampling_rows(),
    )
    return


if __name__ == "__main__":
    app.run()
