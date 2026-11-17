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

    from physicsnemo.datapipes.transforms.mesh.transforms import canonical_frame

    return canonical_frame, h, mo, torch


@app.cell
def _(mo):
    mo.md("""
    # 1. Geometries and the canonical frame

    `canonical_frame` takes a body's surface samples and area weights and
    returns an origin and three axes. This notebook checks, for each
    synthetic body:

    - **Uniqueness.** Is the frame well defined, or does the function
      refuse (equal variances, too many mirror planes)?
    - **Rotation and translation.** Coordinates in the frame must not
      change when the body is moved.
    - **Sampling.** How far does the frame move when the surface is
      sampled again, and when one region is refined, with and without
      area weights?
    """)
    return


@app.cell
def _(h, mo):
    geometry = mo.ui.dropdown(list(h.GEOMETRIES), value="car", label="Body")
    n_samples = mo.ui.slider(
        steps=[256, 1024, 4096, 16384], value=4096, label="Surface samples"
    )
    refine = mo.ui.slider(2, 8, value=4, label="Refinement factor (x > 0 half)")
    seed = mo.ui.number(0, 100, value=0, label="Seed")
    mo.hstack([geometry, n_samples, refine, seed], justify="start")
    return geometry, n_samples, refine, seed


@app.cell
def _(canonical_frame, h, torch):
    def frame(points, weights=None):
        """Origin and axes only."""
        return canonical_frame(points, weights)[:2]

    def axis_drift_deg(axes, reference):
        """Largest angle between matching axes of two frames, in degrees."""
        cos = (axes * reference).sum(dim=-1).clamp(-1, 1)
        return torch.rad2deg(torch.acos(cos)).max().item()

    def diagnostics(name, n, seed, refine):
        x, _, w = h.sample_surface(name, n, seed)
        origin, axes, source = canonical_frame(x, w)
        n_arbitrary = (source == -2).sum().item()
        status = f"{n_arbitrary} arbitrary axes" if n_arbitrary else "unique"

        # Moving the body must not change coordinates in its own frame.
        rotation = h.random_rotation(seed + 1)
        shift = torch.tensor([3.0, -2.0, 5.0], dtype=x.dtype)
        moved = x @ rotation.T + shift
        origin_m, axes_m = frame(moved, w)
        move_err = h.rel_err((moved - origin_m) @ axes_m.T, (x - origin) @ axes.T)

        # A second sample of the same surface, and a locally refined one.
        x2, _, w2 = h.sample_surface(name, n, seed + 7)
        origin2, axes2 = frame(x2, w2)
        xr, _, wr = h.sample_surface(name, n, seed, refine)
        origin_r, axes_r = frame(xr, wr)
        origin_u, axes_u = frame(xr)  # refined, weights ignored
        return {
            "body": name,
            "frame": status,
            "moved: coordinate error": f"{move_err:.1e}",
            "resampled: axes (deg)": f"{axis_drift_deg(axes2, axes):.2f}",
            "resampled: origin": f"{(origin2 - origin).norm().item():.1e}",
            "refined, weighted: axes (deg)": f"{axis_drift_deg(axes_r, axes):.2f}",
            "refined, weighted: origin": f"{(origin_r - origin).norm().item():.1e}",
            "refined, unweighted: axes (deg)": f"{axis_drift_deg(axes_u, axes):.2f}",
            "refined, unweighted: origin": f"{(origin_u - origin).norm().item():.1e}",
        }

    return axis_drift_deg, diagnostics, frame


@app.cell
def _(diagnostics, h, mo, n_samples, refine, seed):
    h.show(
        mo,
        "All bodies",
        [
            diagnostics(name, n_samples.value, seed.value, refine.value)
            for name in h.GEOMETRIES
        ],
    )
    return


@app.cell
def _(mo):
    mo.md("""
    How to read it:

    - `moved` should be at round-off for every body with a unique frame.
      With arbitrary axes the number is meaningless along those axes.
    - `resampled` is the noise floor of the sampling.
    - `refined, weighted` should sit at that noise floor. `refined,
      unweighted` shows what the frame would do if it followed point
      density.
    - Origins are in body units (the bodies are about 4 long).
    """)
    return


@app.cell
def _(axis_drift_deg, frame, geometry, h, mo, seed, torch):
    def convergence(name, seed):
        x, _, w = h.sample_surface(name, 65536, seed + 99)
        origin, axes = frame(x, w)
        evals = torch.linalg.eigvalsh(
            ((w / w.sum())[:, None] * (x - origin)).T @ (x - origin)
        )
        gaps = (evals[1:] - evals[:-1]) / evals[-1]
        rows = []
        for n in (256, 1024, 4096, 16384):
            xn, _, wn = h.sample_surface(name, n, seed)
            origin_n, axes_n = frame(xn, wn)
            rows.append(
                {
                    "samples": n,
                    "axes vs 65536 samples (deg)": f"{axis_drift_deg(axes_n, axes):.2f}",
                    "origin vs 65536 samples": f"{(origin_n - origin).norm().item():.1e}",
                }
            )
        return rows, gaps.min().item()

    _rows, _gap = convergence(geometry.value, seed.value)
    mo.vstack(
        [
            h.show(mo, f"Convergence with sample count: {geometry.value}", _rows),
            mo.md(
                f"Smallest gap between covariance eigenvalues, relative to the "
                f"largest: **{_gap:.3f}** (below 0.01 the two axes count as equal and need a tie-breaker)."
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
