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
    import math
    import sys

    import marimo as mo
    import torch

    sys.path.insert(0, str(mo.notebook_dir()))
    import equivariance_harness as h

    return h, math, mo, torch


@app.cell
def _(mo):
    mo.md("""
    # 2. Boundary conditions

    The model takes boundary conditions as inputs: scalars and vectors on
    each boundary sample, plus global scalars and vectors. Global vectors
    also define the frame. This notebook checks:

    - **Symmetry** with 0, 1 and 2 global vectors (learned frame, mixed,
      fully given).
    - **Controls.** Forgetting to rotate one input must break the result.
      If it does not, the model is ignoring that input.
    - **Sensitivity.** Changing each boundary condition must change the
      output.
    - **Frame conditioning.** What happens as the two global vectors that
      define the frame become parallel.
    """)
    return


@app.cell
def _(h, mo):
    geometry = mo.ui.dropdown(list(h.GEOMETRIES), value="car", label="Body")
    dtype = mo.ui.dropdown(["float64", "float32"], value="float64", label="Precision")
    seed = mo.ui.number(0, 100, value=0, label="Seed")
    mo.hstack([geometry, dtype, seed], justify="start")
    return dtype, geometry, seed


@app.cell
def _(dtype, geometry, h, seed, torch):
    precision = getattr(torch, dtype.value)
    rotation = h.random_rotation(seed.value + 1, precision)
    shift = torch.tensor([3.0, -2.0, 5.0], dtype=precision)

    def setup(global_vectors):
        model = h.make_model(global_vectors, seed=seed.value, dtype=precision)
        inputs = h.make_problem(
            geometry.value,
            global_vectors=global_vectors,
            seed=seed.value,
            dtype=precision,
        )
        return model, inputs

    def fmt(errors):
        return {"scalars": f"{errors[0]:.1e}", "vectors": f"{errors[1]:.1e}"}

    return fmt, precision, rotation, setup, shift


@app.cell
def _(fmt, h, mo, rotation, setup, shift):
    def symmetry_rows():
        rows = []
        for k, frame in ((0, "learned"), (1, "one given, one learned"), (2, "given")):
            model, inputs = setup(k)
            moved = h.move(inputs, rotation, shift)
            rows.append(
                {"global vectors": k, "frame": frame}
                | fmt(h.compare(model, inputs, moved, rotation))
            )
        return rows

    h.show(mo, "Rotation + translation error (should be round-off)", symmetry_rows())
    return


@app.cell
def _(fmt, h, mo, rotation, setup, shift):
    def control_rows():
        model, inputs = setup(2)
        rows = []
        for name in h.VECTORS:
            moved = h.move(inputs, rotation, shift)
            moved[name] = inputs[name]  # left behind
            rows.append(
                {"input not rotated": name}
                | fmt(h.compare(model, inputs, moved, rotation))
            )
        return rows

    h.show(mo, "Controls: one input left unrotated (should be large)", control_rows())
    return


@app.cell
def _(fmt, h, mo, setup, torch):
    @torch.no_grad()
    def sensitivity_rows():
        model, inputs = setup(2)
        reference = model(**inputs)
        rows = []
        for name in (
            "boundary_scalars",
            "boundary_vectors",
            "global_scalars",
            "global_vectors",
        ):
            changed = dict(inputs, **{name: 1.1 * inputs[name]})
            out = model(**changed)
            rows.append(
                {"input scaled by 1.1": name}
                | fmt([h.rel_err(a, b) for a, b in zip(out, reference)])
            )
        return rows

    h.show(
        mo,
        "Sensitivity: relative output change (should not be zero)",
        sensitivity_rows(),
    )
    return


@app.cell
def _(mo):
    mo.md("""
    `global_vectors` scaled by 1.1 keeps the directions, so the frame is
    unchanged; any output change comes from the magnitude reaching the
    network.

    ## Frame conditioning

    With two global vectors the frame is built from them by Gram-Schmidt.
    Below, the second vector is tilted toward the first. The error should
    stay at round-off until the vectors are nearly parallel, then grow.
    """)
    return


@app.cell
def _(fmt, h, math, mo, precision, rotation, setup, shift, torch):
    def conditioning_rows():
        model, inputs = setup(2)
        first = torch.tensor([1.0, 0.0, 0.0], dtype=precision)
        second = torch.tensor([0.0, 0.0, 1.0], dtype=precision)
        rows = []
        for degrees in (90, 10, 1, 1e-2, 1e-4, 1e-6, 0):
            angle = math.radians(degrees)
            vectors = torch.stack(
                [first, math.cos(angle) * first + math.sin(angle) * second]
            )[None]
            case = dict(inputs, global_vectors=vectors)
            rows.append(
                {"angle between vectors (deg)": degrees}
                | fmt(h.compare(model, case, h.move(case, rotation, shift), rotation))
            )
        return rows

    h.show(
        mo, "Rotation + translation error vs frame conditioning", conditioning_rows()
    )
    return


if __name__ == "__main__":
    app.run()
