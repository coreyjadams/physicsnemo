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
    from physicsnemo.experimental.models.geotransolver_pp.geotransolver_pp import (
        _NeighborAttention as NeighborAttention,
    )
    from physicsnemo.models.geotransolver.context_projector import (
        GeometricFeatureProcessor,
    )

    return Block, GeometricFeatureProcessor, NeighborAttention, h, mo, torch


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

    The second table does the same for the neighbor attention that
    replaces GeoTransolver's ball query + MLP: each point attends over the
    boundary tokens a radius search finds within a radius of it. The update
    is that of DeFormer's local attention block, with the tokens' measures
    added to the scores. While every token in range fits in the neighbor
    limit, neither the order nor the refinement of the tokens matters.
    Above the limit the search keeps the first tokens it finds, so the
    result changes with the order of the tokens and with their refinement
    (the copies of a split token use up slots). Set the limit above the
    number of tokens expected in range.

    The attention also has one extra slot with score 0 and no message, and a
    window that falls to zero at the radius. A token entering the range
    then changes the result gradually (last rows). With a plain softmax the
    first token in range would get weight 1 however far away it is.

    The last table runs the order test on GeoTransolver's current
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

    return context, context_measure, measure, precision, run, split, x


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
def _(
    NeighborAttention, dim, h, mo, n_context, n_tokens, precision, seed, split, torch
):
    @torch.no_grad()
    def neighbor_rows():
        n, m = n_tokens.value, n_context.value
        g = torch.Generator().manual_seed(seed.value)

        def rand(*shape):
            return torch.rand(*shape, dtype=precision, generator=g)

        x, tokens = rand(1, n, dim.value), rand(1, m, dim.value)
        points, token_points = rand(1, n, 3), rand(1, m, 3)
        points[:, 0] += 10  # one point with no boundary in range
        measure = rand(1, m) + 0.1

        def layer(limit):
            torch.manual_seed(seed.value)
            return (
                NeighborAttention(dim.value, 4, 0.5, limit, "gelu").to(precision).eval()
            )

        def run(
            layer, tokens, token_points, measure, use_measure=True, points=points, x=x
        ):
            measure = measure if use_measure else torch.ones_like(measure)
            log_measure = (measure / measure.sum(dim=-1, keepdim=True)).log()
            return layer(x, points, tokens, token_points, log_measure)

        full, limited = layer(2 * m), layer(8)

        def one_token(distance):
            """Change of point 0's state from one token ``distance`` away."""
            at = points[:, :1] + torch.tensor([distance, 0.0, 0.0], dtype=precision)
            out = run(full, tokens[:, :1], at, measure[:, :1])
            return out[:, 0] - x[:, 0]

        far = one_token(10.0)
        order, point_order = (
            torch.randperm(m, generator=g),
            torch.randperm(n, generator=g),
        )
        shuffled = (tokens[:, order], token_points[:, order], measure[:, order])
        tokens2, measure2 = split(tokens, measure, m // 4)
        doubled = (
            tokens2,
            torch.cat([token_points, token_points[:, : m // 4]], 1),
            measure2,
        )
        reference = run(full, tokens, token_points, measure)
        cases = [
            ("tokens shuffled", "round-off", run(full, *shuffled), reference),
            (
                "tokens split, measure shared",
                "round-off",
                run(full, *doubled),
                reference,
            ),
            (
                "points shuffled",
                "round-off",
                run(
                    full,
                    tokens,
                    token_points,
                    measure,
                    points=points[:, point_order],
                    x=x[:, point_order],
                ),
                reference[:, point_order],
            ),
            (
                "control: tokens split, measure ignored",
                "large",
                run(full, *doubled, use_measure=False),
                run(full, tokens, token_points, measure, use_measure=False),
            ),
            (
                "limit of 8 neighbors: tokens shuffled",
                "not zero",
                run(limited, *shuffled),
                run(limited, tokens, token_points, measure),
            ),
            (
                "limit of 8 neighbors: tokens split, measure shared",
                "not zero",
                run(limited, *doubled),
                run(limited, tokens, token_points, measure),
            ),
        ]
        cases += [
            (
                f"one token at {d} x radius, vs none in range",
                "0 beyond the radius, growing inside",
                one_token(d * 0.5),
                far,
            )
            for d in (1.01, 0.99, 0.9, 0.5)
        ]
        return [
            {"change": name, "expected": expected, "error": f"{h.rel_err(a, b):.1e}"}
            for name, expected, a, b in cases
        ]

    h.show(
        mo, "Neighbor attention over boundary tokens within a radius", neighbor_rows()
    )
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
