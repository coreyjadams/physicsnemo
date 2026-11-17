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

    from physicsnemo.experimental.models.geotransolver_pp.geotransolver_pp import (
        _Block as Block,
    )
    from physicsnemo.models.geotransolver.context_projector import (
        GeometricFeatureProcessor,
    )

    return Block, GeometricFeatureProcessor, h, mo, torch


@app.cell
def _(mo):
    mo.md("""
    # 3. Attention block

    One measure-weighted FLARE++ block, on random tokens, outside the
    model. Its inputs are already rotation-invariant, so what matters
    here is the dependence on the point set:

    - **Order.** Shuffling tokens must shuffle the output the same way.
      Shuffling context tokens must change nothing.
    - **Refinement.** Splitting a token into two copies that share its
      measure must change nothing. With the measure ignored (control),
      it must change something.

    The last table runs the same order test on GeoTransolver's current
    ball-query feature layer for comparison.
    """)
    return


@app.cell
def _(mo):
    dim = mo.ui.slider(steps=[16, 32, 64, 128], value=32, label="Hidden size")
    n_tokens = mo.ui.slider(steps=[32, 128, 512, 2048], value=128, label="Tokens")
    n_context = mo.ui.slider(steps=[16, 64, 256], value=64, label="Context tokens")
    dtype = mo.ui.dropdown(["float64", "float32"], value="float64", label="Precision")
    seed = mo.ui.number(0, 100, value=0, label="Seed")
    mo.hstack([dim, n_tokens, n_context, dtype, seed], justify="start")
    return dim, dtype, n_context, n_tokens, seed


@app.cell
def _(Block, dim, dtype, h, n_context, n_tokens, seed, torch):
    precision = getattr(torch, dtype.value)
    torch.manual_seed(seed.value)
    block = (
        h.randomize_zero_init(
            Block(dim.value, 4, 8, 2, dim.value, "gelu", use_context=True), seed.value
        )
        .to(precision)
        .eval()
    )

    def rand(*shape):
        return torch.randn(*shape, dtype=precision)

    x, context, cond = (
        rand(1, n_tokens.value, dim.value),
        rand(1, n_context.value, dim.value),
        rand(1, dim.value),
    )
    measure = torch.rand(1, n_tokens.value, dtype=precision) + 0.1
    context_measure = torch.rand(1, n_context.value, dtype=precision) + 0.1

    @torch.no_grad()
    def run(x, measure, context, context_measure, use_measure=True):
        """The block, with measures normalised as the model does."""

        def log(m):
            m = m if use_measure else torch.ones_like(m)
            return (m / m.sum(dim=-1, keepdim=True)).log()

        return block(x, log(measure), cond, context, log(context_measure))

    def split(tokens, measure, count):
        """Append copies of the first ``count`` tokens, halving their measure."""
        return (
            torch.cat([tokens, tokens[:, :count]], dim=1),
            torch.cat(
                [measure[:, :count] / 2, measure[:, count:], measure[:, :count] / 2],
                dim=1,
            ),
        )

    return context, context_measure, measure, run, split, x


@app.cell
def _(context, context_measure, h, measure, mo, run, split, torch, x):
    def block_rows():
        reference = run(x, measure, context, context_measure)
        n, m = x.shape[1], context.shape[1]
        order, context_order = torch.randperm(n), torch.randperm(m)
        x2, measure2 = split(x, measure, n // 4)
        context2, context_measure2 = split(context, context_measure, m // 4)
        uniform = run(x, measure, context, context_measure, use_measure=False)
        cases = [
            (
                "tokens shuffled",
                "round-off",
                run(x[:, order], measure[:, order], context, context_measure),
                reference[:, order],
            ),
            (
                "context shuffled",
                "round-off",
                run(
                    x,
                    measure,
                    context[:, context_order],
                    context_measure[:, context_order],
                ),
                reference,
            ),
            (
                "tokens split, measure shared",
                "round-off",
                run(x2, measure2, context, context_measure)[:, :n],
                reference,
            ),
            (
                "context split, measure shared",
                "round-off",
                run(x, measure, context2, context_measure2),
                reference,
            ),
            (
                "control: tokens split, measure ignored",
                "large",
                run(x2, measure2, context, context_measure, use_measure=False)[:, :n],
                uniform,
            ),
            (
                "control: context split, measure ignored",
                "large",
                run(x, measure, context2, context_measure2, use_measure=False),
                uniform,
            ),
        ]
        return [
            {"change": name, "expected": expected, "error": f"{h.rel_err(a, b):.1e}"}
            for name, expected, a, b in cases
        ]

    h.show(mo, "Measure-weighted FLARE++ block", block_rows())
    return


@app.cell
def _(GeometricFeatureProcessor, dim, h, mo, n_tokens, seed, torch):
    @torch.no_grad()
    def baseline_rows():
        torch.manual_seed(seed.value)
        layer = GeometricFeatureProcessor(
            radius=0.5, neighbors_in_radius=8, feature_dim=3, hidden_dim=dim.value
        ).eval()
        points = torch.rand(1, n_tokens.value, 3)
        order = torch.randperm(n_tokens.value)
        reference = layer(points, points)
        shuffled = layer(points[:, order], points[:, order])
        return [
            {
                "change": "points shuffled",
                "expected": "round-off if order-invariant",
                "error": f"{h.rel_err(shuffled, reference[:, order]):.1e}",
            }
        ]

    h.show(
        mo,
        "Baseline: GeoTransolver ball query + MLP (GeometricFeatureProcessor)",
        baseline_rows(),
    )
    return


if __name__ == "__main__":
    app.run()
