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

r"""GeoTransolverPP: GeoTransolver in the canonical frame of the geometry.

``GeoTransolverPP`` is a working name. Like GeoTransolver, the model routes
mesh points through latent tokens and conditions them on an encoded context.
Here the context is the boundary: every boundary condition is a named set of
samples with its own fields. The model expects positions and vectors in the
canonical, non-dimensional frame of the geometry
(:class:`~physicsnemo.datapipes.transforms.mesh.CanonicalFrameMesh`), so it
reads plain coordinates and components and is itself an ordinary network.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float

from physicsnemo.core.meta import ModelMetaData
from physicsnemo.core.module import Module
from physicsnemo.nn import Mlp
from physicsnemo.nn.functional import knn


@dataclass
class GeoTransolverPPMetaData(ModelMetaData):
    r"""Meta data for :class:`GeoTransolverPP`."""

    amp: bool = True


class _AdaLayerNorm(nn.Module):
    r"""Layer norm whose scale and shift are set by the global condition."""

    def __init__(self, dim: int, cond_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(
        self, x: Float[torch.Tensor, "B N C"], cond: Float[torch.Tensor, "B D"]
    ) -> Float[torch.Tensor, "B N C"]:
        scale, shift = self.to_scale_shift(cond)[:, None].chunk(2, dim=-1)
        return self.norm(x) * (1 + scale) + shift


class _Block(nn.Module):
    r"""Measure-weighted FLARE++ block with optional context for the latents.

    Tokens are gathered into ``n_latents`` latent tokens with a
    measure-weighted softmax, so the latents estimate integrals over the
    domain and not sums over mesh points. With ``use_context``, the latents
    then cross-attend to context tokens and the result is added residually.
    Finally the latents are scattered back to the tokens.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        n_latents: int,
        mlp_ratio: int,
        cond_dim: int,
        act: str,
        use_context: bool,
    ) -> None:
        super().__init__()
        self.heads = heads
        dim_head = dim // heads
        self.norm_attn = _AdaLayerNorm(dim, cond_dim)
        self.norm_mlp = _AdaLayerNorm(dim, cond_dim)
        self.seeds = nn.Parameter(torch.randn(1, heads, n_latents, dim_head))
        # Fused projection: token keys, seed keys, seed values, token values.
        self.to_qkv = nn.Linear(dim, 4 * dim)
        if use_context:
            self.cross_q = nn.Linear(dim_head, dim_head)
            self.cross_kv = nn.Linear(dim, 2 * dim)
        self.out = nn.Linear(dim, dim)
        self.mlp = Mlp(dim, dim * mlp_ratio, dim, act_layer=act)

    def _split_heads(self, x: torch.Tensor, chunks: int) -> tuple[torch.Tensor, ...]:
        return tuple(
            rearrange(t, "b n (h d) -> b h n d", h=self.heads)
            for t in x.chunk(chunks, dim=-1)
        )

    def forward(
        self,
        x: Float[torch.Tensor, "B N C"],
        log_measure: Float[torch.Tensor, "B N"],
        cond: Float[torch.Tensor, "B D"],
        context: Float[torch.Tensor, "B M C"] | None = None,
        context_log_measure: Float[torch.Tensor, "B M"] | None = None,
    ) -> Float[torch.Tensor, "B N C"]:
        key, seed_k, seed_v, value = self._split_heads(
            self.to_qkv(self.norm_attn(x, cond)), 4
        )
        # Adding log(measure) to the scores turns each softmax over tokens
        # into a quadrature-weighted average.
        bias = log_measure[:, None, None, :].to(key.dtype)
        seeds = self.seeds.to(key.dtype).expand(x.shape[0], -1, -1, -1)
        latent_q = F.scaled_dot_product_attention(seeds, seed_k, seed_v, attn_mask=bias)
        latent_v = F.scaled_dot_product_attention(latent_q, key, value, attn_mask=bias)

        if context is not None:
            context_k, context_v = self._split_heads(self.cross_kv(context), 2)
            latent_v = latent_v + F.scaled_dot_product_attention(
                self.cross_q(latent_v),
                context_k,
                context_v,
                attn_mask=context_log_measure[:, None, None, :].to(key.dtype),
            )

        y = F.scaled_dot_product_attention(key, latent_q, latent_v)
        x = x + self.out(rearrange(y, "b h n d -> b n (h d)"))
        return x + self.mlp(self.norm_mlp(x, cond))


class _NeighborAttention(nn.Module):
    r"""Each point attends over its nearest tokens within ``radius``.

    The update follows DeFormer's ``KNNAttentionBlock``: the encoded offset
    from token to point enters the attention scores as a per-head bias and
    the message as an attention-weighted edge aggregate, and one MLP updates
    the point from its normalized state, the message, the edge aggregate and
    the log-sum-exp of the scores. Two things differ. The tokens are a
    second set (cross-attention; pass the same set twice for
    self-attention), and the attention is a quadrature over the boundary in
    range, not an average over whichever tokens are there:

    - the scores carry the log of each token's measure, in units of
      ``radius`` squared, and of a window that falls smoothly to zero at
      ``radius``;
    - the softmax has one extra slot with score 0 and an empty message.

    So the message fades in as boundary enters the range. Without the extra
    slot a single token at the edge of the range would count as much as a
    full neighborhood.

    The neighbors are the ``n_neighbors`` nearest tokens, dropped beyond
    ``radius``, so they do not depend on the order of the tokens. While
    every token in range fits in ``n_neighbors`` the result does not depend
    on the token density either; above that the nearest ones stand for the
    rest. The log-sum-exp is the log of one plus the integral in range.
    """

    def __init__(
        self,
        dim: int,
        heads: int,
        radius: float,
        n_neighbors: int,
        act: str,
        edge_dim: int = 32,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.radius = radius
        self.n_neighbors = n_neighbors
        self.norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim, bias=False)
        # Offset and distance, in units of the radius.
        self.edge = nn.Sequential(
            Mlp(4, edge_dim, edge_dim, act_layer=act), nn.LayerNorm(edge_dim)
        )
        self.edge_bias = nn.Linear(edge_dim, heads, bias=False)
        self.update = nn.Sequential(
            Mlp(2 * dim + heads * edge_dim + heads, 2 * dim, dim, act_layer=act),
            nn.LayerNorm(dim),
        )

    def forward(
        self,
        x: Float[torch.Tensor, "B N C"],
        points: Float[torch.Tensor, "B N 3"],
        tokens: Float[torch.Tensor, "B M C"],
        token_points: Float[torch.Tensor, "B M 3"],
        token_log_measure: Float[torch.Tensor, "B M"],
    ) -> Float[torch.Tensor, "B N C"]:
        k = min(self.n_neighbors, token_points.shape[1])
        index = torch.stack(
            [knn(t, p, k)[0] for t, p in zip(token_points, points)]
        ).long()  # (B, N, k)
        batch = torch.arange(points.shape[0], device=points.device)[:, None, None]
        offsets = (points[:, :, None] - token_points[batch, index]) / self.radius
        distance = offsets.norm(dim=-1, keepdim=True)
        # Smooth window: 1 at the point, 0 from the radius on.
        window = (1 - distance[..., 0].square()).clamp_min(0).square()
        edge = self.edge(torch.cat([offsets, distance], dim=-1))  # (B, N, k, E)

        normalized = self.norm(x)
        query = rearrange(self.to_q(normalized), "b n (h d) -> b n h d", h=self.heads)
        key = rearrange(
            self.to_k(self.token_norm(tokens))[batch, index],
            "b n k (h d) -> b n k h d",
            h=self.heads,
        )
        value = rearrange(
            self.to_v(tokens)[batch, index], "b n k (h d) -> b n k h d", h=self.heads
        )
        logits = torch.einsum("bnhd,bnkhd->bnhk", query, key) * key.shape[-1] ** -0.5
        logits = logits + self.edge_bias(edge).transpose(-1, -2)
        bias = (
            token_log_measure[batch, index] - 2 * math.log(self.radius) + window.log()
        )
        logits = logits + bias[:, :, None].to(logits.dtype)
        # The extra slot: score 0, no message.
        logits = torch.cat([logits, torch.zeros_like(logits[..., :1])], dim=-1)
        lse = torch.logsumexp(logits, dim=-1).detach()
        weights = torch.softmax(logits, dim=-1)[..., :-1]
        message = torch.einsum("bnhk,bnkhd->bnhd", weights, value).flatten(-2)
        edge_message = torch.einsum("bnhk,bnke->bnhe", weights, edge).flatten(-2)
        return x + self.update(
            torch.cat([normalized, message, edge_message, lse], dim=-1)
        )


class GeoTransolverPP(Module):
    r"""GeoTransolver for boundary-driven PDEs, in the geometry's canonical frame.

    The boundary is a model input: each boundary condition is a named set of
    samples with a position, a normal, a measure (area) and that condition's
    own feature channels. Each condition has its own input layer, so
    conditions may carry different fields. A boundary encoder turns all
    samples into context tokens: each sample first attends over the samples
    near it on the boundary, then FLARE++ blocks mix all of them. Each mesh
    point first attends over the boundary tokens near it, and is then
    processed by FLARE++ blocks whose latent tokens cross-attend to the
    whole context.

    Positions, normals and vector features are expected in the canonical,
    non-dimensional frame of the geometry, as
    :class:`~physicsnemo.datapipes.transforms.mesh.CanonicalFrameMesh` with
    ``rescale=True`` produces. A vector is then three ordinary feature
    channels. With such inputs:

    - Rotating, translating or uniformly rescaling the whole problem leaves
      the inputs, and so the outputs, unchanged. Vector outputs are
      components in the canonical frame: ``v = v_canonical @ axes`` maps
      them back.
    - Points and boundary samples enter the attention only through
      measure-weighted averages, so refining or resampling the mesh changes
      predictions only by quadrature error, and their order does not matter.

    Parameters
    ----------
    out_scalars : int
        Number of scalar output fields.
    out_vectors : int
        Number of vector output fields.
    boundary_features : Mapping[str, int]
        Number of feature channels per sample for each boundary condition,
        by name. ``0`` for a condition with no fields, such as a no-slip
        wall.
    global_features : int, optional
        Number of global condition channels, such as a Reynolds number or
        the components of a freestream direction. Default is 0.
    n_hidden : int, optional
        Hidden dimension. Default is 256.
    n_layers : int, optional
        Number of blocks that process the mesh points. Default is 8.
    n_encoder_layers : int, optional
        Number of blocks in the boundary encoder. Default is 2.
    n_head : int, optional
        Number of attention heads. Default is 8.
    n_latents : int, optional
        Number of latent tokens per head. Default is 64.
    mlp_ratio : int, optional
        Ratio of MLP hidden dimension to ``n_hidden``. Default is 4.
    radius : float, optional
        Range of the two neighbor attentions, in canonical units (the
        geometry's RMS radius). Default is 0.5.
    n_neighbors : int, optional
        Each point or boundary sample attends to its nearest ``n_neighbors``
        boundary samples within ``radius``. Default is 32.
    act : str, optional
        Activation function name. Default is ``"gelu"``.

    Forward
    -------
    points : torch.Tensor
        Mesh points to predict at, of shape :math:`(B, N, 3)`.
    boundaries : Mapping[str, Mapping[str, torch.Tensor]]
        For each condition given (any non-empty subset of the declared
        names): ``"points"`` of shape :math:`(B, M, 3)`, unit ``"normals"``
        of shape :math:`(B, M, 3)`, positive ``"measure"`` of shape
        :math:`(B, M)`, and ``"features"`` of shape :math:`(B, M, C)` if the
        condition declares any. Measures are areas in canonical units
        (after the rescaling of positions).
    global_features : torch.Tensor | None, optional
        Shape :math:`(B, C_g)`. Required if ``global_features > 0``.
    point_measure : torch.Tensor | None, optional
        Positive cell volume (or area) per point, of shape :math:`(B, N)`.
        Only relative values matter. Default is uniform.

    Outputs
    -------
    tuple[torch.Tensor, torch.Tensor]
        Scalars of shape :math:`(B, N, \text{out\_scalars})` and vectors of
        shape :math:`(B, N, \text{out\_vectors}, 3)`.

    Notes
    -----
    Known limits of this first version: the canonical frame jumps between
    nearly symmetric geometries; a point farther than ``radius`` from the
    boundary sees it only through the latent tokens; there is a single
    point stream; Transformer Engine and token-sharded inputs are not
    supported.

    Examples
    --------
    >>> import torch
    >>> model = GeoTransolverPP(
    ...     out_scalars=1, out_vectors=1, boundary_features={"wall": 0, "inlet": 3},
    ...     n_hidden=32, n_layers=2, n_head=4, n_latents=8,
    ... )
    >>> def samples(m, channels):
    ...     out = {
    ...         "points": torch.randn(1, m, 3),
    ...         "normals": torch.nn.functional.normalize(torch.randn(1, m, 3), dim=-1),
    ...         "measure": torch.rand(1, m) + 0.1,
    ...     }
    ...     return out | ({"features": torch.randn(1, m, channels)} if channels else {})
    >>> scalars, vectors = model(
    ...     torch.randn(1, 100, 3), {"wall": samples(64, 0), "inlet": samples(16, 3)}
    ... )
    >>> scalars.shape, vectors.shape
    (torch.Size([1, 100, 1]), torch.Size([1, 100, 1, 3]))
    """

    def __init__(
        self,
        out_scalars: int,
        out_vectors: int,
        boundary_features: Mapping[str, int],
        global_features: int = 0,
        n_hidden: int = 256,
        n_layers: int = 8,
        n_encoder_layers: int = 2,
        n_head: int = 8,
        n_latents: int = 64,
        mlp_ratio: int = 4,
        radius: float = 0.5,
        n_neighbors: int = 32,
        act: str = "gelu",
    ) -> None:
        super().__init__(meta=GeoTransolverPPMetaData())
        if n_hidden % n_head != 0:
            raise ValueError(
                f"n_hidden must be divisible by n_head, got {n_hidden} and {n_head}"
            )
        self.out_scalars = out_scalars
        self.out_vectors = out_vectors
        self.n_global_features = global_features

        # One input layer per condition: position, normal, its own fields.
        # The layer's bias tells the conditions apart.
        self.boundary_embed = nn.ModuleDict(
            {
                name: Mlp(6 + channels, 2 * n_hidden, n_hidden, act_layer=act)
                for name, channels in sorted(boundary_features.items())
            }
        )
        self.point_embed = Mlp(3, 2 * n_hidden, n_hidden, act_layer=act)
        # The constant 1 keeps the condition defined when there are no globals.
        self.cond_embed = Mlp(1 + global_features, n_hidden, n_hidden, act_layer=act)

        def block(use_context: bool) -> _Block:
            return _Block(
                n_hidden, n_head, n_latents, mlp_ratio, n_hidden, act, use_context
            )

        self.encoder = nn.ModuleList([block(False) for _ in range(n_encoder_layers)])
        self.surface = _NeighborAttention(n_hidden, n_head, radius, n_neighbors, act)
        self.neighbors = _NeighborAttention(n_hidden, n_head, radius, n_neighbors, act)
        self.blocks = nn.ModuleList([block(True) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(n_hidden)
        self.out = nn.Linear(n_hidden, out_scalars + 3 * out_vectors)

    def forward(
        self,
        points: Float[torch.Tensor, "B N 3"],
        boundaries: Mapping[str, Mapping[str, torch.Tensor]],
        global_features: Float[torch.Tensor, "B Cg"] | None = None,
        point_measure: Float[torch.Tensor, "B N"] | None = None,
    ) -> tuple[Float[torch.Tensor, "B N S"], Float[torch.Tensor, "B N V 3"]]:
        r"""Predict scalar and vector fields at ``points``."""
        unknown = set(boundaries) - set(self.boundary_embed)
        if unknown or not boundaries:
            raise ValueError(
                f"Expected boundaries from {sorted(self.boundary_embed)}, "
                f"got {sorted(boundaries)}"
            )
        if global_features is None:
            global_features = points.new_zeros(points.shape[0], 0)
        if global_features.shape[-1] != self.n_global_features:
            raise ValueError(
                f"Expected {self.n_global_features} global_features, "
                f"got {global_features.shape[-1]}"
            )
        if point_measure is None:
            point_measure = points.new_ones(points.shape[:2])

        # Boundary encoder: geometry and boundary conditions become context.
        names = sorted(boundaries)
        context = torch.cat(
            [
                self.boundary_embed[name](
                    torch.cat(
                        [
                            boundaries[name][key]
                            for key in ("points", "normals", "features")
                            if key in boundaries[name]
                        ],
                        dim=-1,
                    )
                )
                for name in names
            ],
            dim=1,
        )
        boundary_points = torch.cat(
            [boundaries[name]["points"] for name in names], dim=1
        )
        measure = torch.cat([boundaries[name]["measure"] for name in names], dim=1)
        boundary_log_measure = (measure / measure.sum(dim=-1, keepdim=True)).log()
        log_area = measure.log()
        cond = self.cond_embed(
            torch.cat(
                [global_features.new_ones(points.shape[0], 1), global_features], dim=-1
            )
        )
        context = self.surface(
            context, boundary_points, context, boundary_points, log_area
        )
        for block in self.encoder:
            context = block(context, boundary_log_measure, cond)

        x = self.neighbors(
            self.point_embed(points), points, context, boundary_points, log_area
        )
        log_measure = (point_measure / point_measure.sum(dim=-1, keepdim=True)).log()
        for block in self.blocks:
            x = block(x, log_measure, cond, context, boundary_log_measure)

        out = self.out(self.out_norm(x))
        scalars = out[..., : self.out_scalars]
        vectors = out[..., self.out_scalars :].unflatten(-1, (self.out_vectors, 3))
        return scalars, vectors
