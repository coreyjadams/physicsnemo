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

    The whole pipeline, untrained, on one synthetic body with an inlet: the
    body's canonical frame, then `GeoTransolverPP`, then vectors rotated
    back. Each row changes the problem in a way that should leave the answer
    alone (scalars unchanged, vectors rotated with the problem) and reports
    the largest deviation relative to the largest output.

    - **round-off**: holds by construction.
    - **sampling noise**: the boundary was sampled differently; the error
      should shrink as samples are added (second table). It does so only
      while the frame stays put. The bodies here are random samples, so a
      third moment that is zero on the true shape (across the car's mirror
      plane) is noise of about 1/sqrt(samples). If that noise is above the
      tie-break threshold it sets the axis, and a resample can flip it:
      the `frame moved` column then reads about 180 degrees. On a real mesh
      the frame is computed once from every cell, where this noise is far
      smaller.
    - **not enforced**: reflections are deliberately not a symmetry.

    The last section runs the same check on AeroZoo samples.
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
    threshold = mo.ui.dropdown(
        ["0.001", "0.01", "0.1", "0.3"], value="0.001", label="Tie-break threshold"
    )
    mo.hstack(
        [geometry, global_vectors, n_boundary, dtype, seed, threshold], justify="start"
    )
    return dtype, geometry, global_vectors, n_boundary, seed, threshold


@app.cell
def _(dtype, geometry, global_vectors, h, seed, threshold, torch):
    precision = getattr(torch, dtype.value)
    model = h.make_model(int(global_vectors.value), seed=seed.value, dtype=precision)

    def problem(n_boundary, **kwargs):
        return h.make_problem(
            geometry.value,
            n_boundary,
            global_vectors=int(global_vectors.value),
            seed=seed.value,
            dtype=precision,
            min_component=float(threshold.value),
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
                "body samples shuffled",
                "round-off",
                h.permute(inputs, "body")[0],
                None,
                None,
            ),
            ("points split, measure shared", "round-off", *split),
            (
                "body samples split, measure shared",
                "round-off",
                h.duplicate(inputs, "body", m // 4)[0],
                None,
                None,
            ),
            (
                "boundaries resampled",
                "sampling noise",
                problem(m, boundary_seed=seed.value + 7),
                None,
                None,
            ),
            (
                "body refined x 4 on one half",
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
                    "frame moved (deg)": f"{h.axes_moved_deg(moved, inputs):.1f}"
                    if matrix is None
                    else "",
                }
            )
        return rows

    h.show(mo, "Symmetry errors", symmetry_rows())
    return


@app.cell
def _(h, mo, model, problem):
    def resampling_rows():
        reference = problem(4096, boundary_seed=12345)
        rows = []
        for m in (64, 256, 1024):
            scalars, vectors = h.compare(model, reference, problem(m))
            rows.append(
                {
                    "boundary samples": m,
                    "scalars": f"{scalars:.1e}",
                    "vectors": f"{vectors:.1e}",
                    "frame moved (deg)": f"{h.axes_moved_deg(problem(m), reference):.1f}",
                }
            )
        return rows

    h.show(
        mo,
        "Boundary resampling: difference from a 4096-sample boundary",
        resampling_rows(),
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## AeroZoo samples

    Real flow problems, in float32: `CanonicalFrameMesh` finds the body
    among the wall cells, builds its frame (ties go to the mean inlet
    velocity, then the enclosure offset) and rescales by the body's RMS
    radius. The same cells and points are then given to the model. The
    whole sample is rotated, shifted by about half its size and made three
    times as large before the transform; the outputs should agree to
    float32 round-off (about 1e-5).

    The ISLA columns run an untrained ISLA interior model on the same two
    samples as a reference. ISLA uses no frame, so it gets the samples as
    they are, and its velocity is compared after rotating it back.
    Both models are small and untrained: the columns compare symmetry
    error only, not accuracy.
    """)
    return


@app.cell
def _(mo):
    n_cases = mo.ui.slider(1, 16, value=3, label="AeroZoo samples (smallest first)")
    n_cases
    return (n_cases,)


@app.cell
def _(h, mo, n_cases, seed, torch):
    @torch.no_grad()
    def aerozoo_rows():
        transform, model = h.aerozoo_transform(), h.aerozoo_model(seed=seed.value)
        isla = h.aerozoo_isla(seed=seed.value)
        rotation = h.random_rotation(seed.value + 1, torch.float32)
        rows = []
        for path in h.aerozoo_cases(n_cases.value):
            domain = h.load_aerozoo(path)
            moved = h.aerozoo_moved(domain, rotation)
            a, b = transform.apply_to_domain(domain), transform.apply_to_domain(moved)
            isla_a, isla_b = h.isla_predict(isla, domain), h.isla_predict(isla, moved)
            out_a = model(**h.aerozoo_inputs(a, seed=seed.value))
            out_b = model(**h.aerozoo_inputs(b, seed=seed.value))
            rows.append(
                {
                    "sample": path.parent.name[-12:],
                    "axes fixed by": str(a.global_data["frame_source"].tolist()),
                    "scalars": f"{h.rel_err(out_b[0], out_a[0]):.1e}",
                    "vectors (canonical)": f"{h.rel_err(out_b[1], out_a[1]):.1e}",
                    "ISLA scalars": f"{h.rel_err(isla_b[0], isla_a[0]):.1e}",
                    "ISLA vectors": f"{h.rel_err(isla_b[1], isla_a[1] @ rotation.T):.1e}",
                }
            )
        return rows

    _rows = aerozoo_rows()
    (
        h.show(mo, "AeroZoo: rotation + translation + scale error", _rows)
        if _rows
        else mo.md(f"No AeroZoo samples under `{h.AEROZOO}`; set `AEROZOO_DIR`.")
    )
    return


if __name__ == "__main__":
    app.run()
