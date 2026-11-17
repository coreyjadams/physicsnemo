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
    # 2. Boundary conditions

    Each boundary condition is a named set of samples with its own fields:
    here a `body` (two scalars and a vector per sample) and an `inlet` (a
    velocity vector per sample), plus global scalars and vectors. The
    pipeline puts everything in the body's canonical frame, where a vector
    is three ordinary feature channels, and then runs the model. This
    notebook checks:

    - **Symmetry** with 0, 1 and 2 global vectors.
    - **Controls.** Forgetting to rotate one input must break the result.
      If it does not, the model is ignoring that input.
    - **Sensitivity.** Changing each boundary condition must change the
      output.
    - **Tie-breakers.** What fixes each axis of the frame, per body, as
      boundary information is taken away.
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
        problem = h.make_problem(
            geometry.value,
            global_vectors=global_vectors,
            seed=seed.value,
            dtype=precision,
        )
        return model, problem

    def fmt(errors):
        return {"scalars": f"{errors[0]:.1e}", "vectors": f"{errors[1]:.1e}"}

    return fmt, precision, rotation, setup, shift


@app.cell
def _(fmt, h, mo, rotation, setup, shift):
    def symmetry_rows():
        rows = []
        for k in (0, 1, 2):
            model, problem = setup(k)
            moved = h.move(problem, rotation, shift)
            rows.append(
                {"global vectors": k} | fmt(h.compare(model, problem, moved, rotation))
            )
        return rows

    h.show(mo, "Rotation + translation error (should be round-off)", symmetry_rows())
    return


@app.cell
def _(fmt, h, mo, rotation, setup, shift):
    def control_rows():
        model, problem = setup(2)
        rows = []
        for name in h.VECTORS:
            moved = h.move(problem, rotation, shift, skip=name)
            label = name if isinstance(name, str) else " ".join(name)
            rows.append(
                {"input not rotated": label}
                | fmt(h.compare(model, problem, moved, rotation))
            )
        return rows

    h.show(mo, "Controls: one input left unrotated (should be large)", control_rows())
    return


@app.cell
def _(fmt, h, mo, setup, torch):
    @torch.no_grad()
    def sensitivity_rows():
        model, problem = setup(2)
        reference = h.predict(model, problem)

        def scaled(boundary, key):
            fields = dict(problem["boundaries"][boundary])
            fields[key] = 1.1 * fields[key]
            return dict(problem, boundaries=problem["boundaries"] | {boundary: fields})

        cases = {
            "body scalars": scaled("body", "scalars"),
            "body vectors": scaled("body", "vectors"),
            "inlet vectors": scaled("inlet", "vectors"),
            "global_scalars": dict(
                problem, global_scalars=1.1 * problem["global_scalars"]
            ),
            "global_vectors": dict(
                problem, global_vectors=1.1 * problem["global_vectors"]
            ),
            "inlet removed": dict(
                problem, boundaries={"body": problem["boundaries"]["body"]}
            ),
        }
        return [
            {"change": name}
            | fmt([h.rel_err(a, b) for a, b in zip(h.predict(model, case), reference)])
            for name, case in cases.items()
        ]

    h.show(
        mo,
        "Sensitivity: relative output change (should not be zero)",
        sensitivity_rows(),
    )
    return


@app.cell
def _(mo):
    mo.md("""
    The first five rows scale an input by 1.1. Scaling `global_vectors`
    keeps the directions, so the frame is unchanged; the output change comes
    from the magnitude reaching the network.

    ## Tie-breakers

    The frame is the body's principal axes. What the shape leaves open is
    settled by a fixed list: the body's third moment (0), then each global
    vector (1, 2), then the offset of all boundaries' centroid from the
    body's (the last number). `-1` is the right-hand rule and `-2` means
    nothing fixed the axis, so it is arbitrary and the error is not
    round-off. The bodies are random samples, so even the ellipsoid and the
    spheroid are slightly lopsided; with few samples that noise is what the
    third moment picks up.
    """)
    return


@app.cell
def _(fmt, h, mo, precision, rotation, seed, shift):
    def tie_break_rows():
        rows = []
        for name in h.GEOMETRIES:
            for k, inlet in ((1, True), (0, True), (0, False)):
                model = h.make_model(k, seed=seed.value, dtype=precision)
                problem = h.make_problem(
                    name, global_vectors=k, seed=seed.value, dtype=precision
                )
                if not inlet:
                    problem["boundaries"].pop("inlet")
                moved = h.move(problem, rotation, shift)
                rows.append(
                    {
                        "body": name,
                        "global vectors": k,
                        "inlet": inlet,
                        "axes fixed by": str(h.body_frame(problem)[3].tolist()),
                    }
                    | fmt(h.compare(model, problem, moved, rotation))
                )
        return rows

    h.show(mo, "What fixes the frame, and the rotation error", tie_break_rows())
    return


if __name__ == "__main__":
    app.run()
