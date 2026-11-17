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

r"""GeoTransolverPP: a rotation- and translation-equivariant FLARE++ model.

``GeoTransolverPP`` is a working name. Like GeoTransolver, the model routes
mesh points through latent tokens and conditions them on an encoded context.
Here the context is the boundary (geometry plus boundary conditions), and
every feature the network sees is invariant to rotation and translation.
"""

from __future__ import annotations

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

# Floor on the nearest-boundary distance, in units of the reference length,
# so the neighbor weights stay finite for points that lie on the boundary.
_MIN_DISTANCE = 1e-3


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


class GeoTransolverPP(Module):
    r"""Rotation- and translation-equivariant FLARE++ model for boundary-driven PDEs.

    The boundary is a model input: sampled points carrying a normal, a
    measure (area), and boundary-condition scalars and vectors. The model
    builds an origin, a reference length and an orthonormal frame from the
    boundary and the global vectors, and expresses every position and vector
    in that frame. A boundary encoder turns the boundary samples into context
    tokens. The mesh points are then processed by FLARE++ blocks whose latent
    tokens cross-attend to that context. Vector outputs are predicted as
    components in the frame and rotated back.

    Symmetries that hold by construction:

    - Rotation and translation of the whole problem: scalar outputs are
      unchanged and vector outputs rotate with the problem.
    - Uniform rescaling of all positions (the reference length is derived
      from the boundary).
    - Points and boundary samples enter the attention only through
      measure-weighted averages, so refining or resampling the mesh changes
      predictions only by quadrature error.

    Parity (reflection) is deliberately not built in.

    Parameters
    ----------
    out_scalars : int
        Number of scalar output fields.
    out_vectors : int
        Number of vector output fields.
    boundary_scalars : int, optional
        Number of scalar features per boundary sample, such as
        boundary-condition type and values. Default is 0.
    boundary_vectors : int, optional
        Number of vector features per boundary sample, such as wall
        velocity. Default is 0.
    global_scalars : int, optional
        Number of global scalar conditions, such as Reynolds number.
        Default is 0.
    global_vectors : int, optional
        Number of global vector conditions, such as the freestream velocity.
        The first two define the frame. If fewer than two are given, the
        missing frame vectors are learned from the boundary. Default is 0.
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
    n_neighbors : int, optional
        Number of nearest boundary samples that describe the boundary near
        each point. Default is 8.
    act : str, optional
        Activation function name. Default is ``"gelu"``.

    Forward
    -------
    points : torch.Tensor
        Mesh points to predict at, of shape :math:`(B, N, 3)`.
    boundary_points : torch.Tensor
        Boundary sample positions of shape :math:`(B, M, 3)`.
    boundary_normals : torch.Tensor
        Unit normals of shape :math:`(B, M, 3)`.
    boundary_measure : torch.Tensor
        Positive area per boundary sample, of shape :math:`(B, M)`. Only
        relative values matter.
    boundary_scalars : torch.Tensor | None, optional
        Shape :math:`(B, M, C_s)`. Required if ``boundary_scalars > 0``.
    boundary_vectors : torch.Tensor | None, optional
        Shape :math:`(B, M, K_v, 3)`. Required if ``boundary_vectors > 0``.
    global_scalars : torch.Tensor | None, optional
        Shape :math:`(B, C_g)`. Required if ``global_scalars > 0``.
    global_vectors : torch.Tensor | None, optional
        Shape :math:`(B, K_g, 3)`. Required if ``global_vectors > 0``.
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
    Known limits of this first version: the frame is undefined when its two
    defining vectors are parallel or, for learned vectors, when the boundary
    is too symmetric to define a direction; positions enter the embedding
    MLPs as raw frame coordinates; there is a single point stream;
    Transformer Engine and token-sharded inputs are not supported.

    Examples
    --------
    >>> import torch
    >>> model = GeoTransolverPP(
    ...     out_scalars=1, out_vectors=1, global_vectors=1, n_hidden=32,
    ...     n_layers=2, n_head=4, n_latents=8,
    ... )
    >>> boundary = torch.randn(1, 64, 3)
    >>> normals = torch.nn.functional.normalize(torch.randn(1, 64, 3), dim=-1)
    >>> scalars, vectors = model(
    ...     torch.randn(1, 100, 3), boundary, normals, torch.rand(1, 64) + 0.1,
    ...     global_vectors=torch.tensor([[[1.0, 0.0, 0.0]]]),
    ... )
    >>> scalars.shape, vectors.shape
    (torch.Size([1, 100, 1]), torch.Size([1, 100, 1, 3]))
    """

    def __init__(
        self,
        out_scalars: int,
        out_vectors: int,
        boundary_scalars: int = 0,
        boundary_vectors: int = 0,
        global_scalars: int = 0,
        global_vectors: int = 0,
        n_hidden: int = 256,
        n_layers: int = 8,
        n_encoder_layers: int = 2,
        n_head: int = 8,
        n_latents: int = 64,
        mlp_ratio: int = 4,
        n_neighbors: int = 8,
        act: str = "gelu",
    ) -> None:
        super().__init__(meta=GeoTransolverPPMetaData())
        if n_hidden % n_head != 0:
            raise ValueError(
                f"n_hidden must be divisible by n_head, got {n_hidden} and {n_head}"
            )
        self.out_scalars = out_scalars
        self.out_vectors = out_vectors
        self.n_boundary_scalars = boundary_scalars
        self.n_boundary_vectors = boundary_vectors
        self.n_global_scalars = global_scalars
        self.n_global_vectors = global_vectors
        self.n_neighbors = n_neighbors

        # Frame vectors the global conditions do not supply are learned as
        # invariant-weighted sums of boundary offsets and normals.
        self.n_learned_frame_vectors = max(0, 2 - global_vectors)
        if self.n_learned_frame_vectors > 0:
            self.frame_net = Mlp(
                2 + 2 * global_vectors + boundary_scalars,
                n_hidden,
                2 * self.n_learned_frame_vectors,
                act_layer=act,
            )

        # Normal, scalars and vectors of a boundary sample, in the frame.
        boundary_feature_dim = 3 + boundary_scalars + 3 * boundary_vectors
        self.boundary_embed = Mlp(
            3 + boundary_feature_dim, 2 * n_hidden, n_hidden, act_layer=act
        )
        # Position, offset to the boundary, distance, nearby boundary features.
        self.point_embed = Mlp(
            3 + 3 + 1 + boundary_feature_dim, 2 * n_hidden, n_hidden, act_layer=act
        )
        # The constant 1 keeps the condition defined when there are no globals.
        self.cond_embed = Mlp(
            1 + global_scalars + 3 * global_vectors, n_hidden, n_hidden, act_layer=act
        )

        def block(use_context: bool) -> _Block:
            return _Block(
                n_hidden, n_head, n_latents, mlp_ratio, n_hidden, act, use_context
            )

        self.encoder = nn.ModuleList([block(False) for _ in range(n_encoder_layers)])
        self.blocks = nn.ModuleList([block(True) for _ in range(n_layers)])
        self.out_norm = nn.LayerNorm(n_hidden)
        self.out = nn.Linear(n_hidden, out_scalars + 3 * out_vectors)

    def _frame(
        self,
        offsets: Float[torch.Tensor, "B M 3"],
        normals: Float[torch.Tensor, "B M 3"],
        weights: Float[torch.Tensor, "B M"],
        boundary_scalars: Float[torch.Tensor, "B M Cs"],
        global_vectors: Float[torch.Tensor, "B Kg 3"],
    ) -> Float[torch.Tensor, "B 3 3"]:
        r"""Right-handed orthonormal frame whose rows rotate with the problem."""
        candidates = global_vectors
        if self.n_learned_frame_vectors > 0:
            offset_dir = F.normalize(offsets, dim=-1)
            global_dir = F.normalize(global_vectors, dim=-1)
            invariants = torch.cat(
                [
                    offsets.norm(dim=-1, keepdim=True),
                    (normals * offset_dir).sum(dim=-1, keepdim=True),
                    torch.einsum("bmd,bkd->bmk", offset_dir, global_dir),
                    torch.einsum("bmd,bkd->bmk", normals, global_dir),
                    boundary_scalars,
                ],
                dim=-1,
            )
            on_offsets, on_normals = self.frame_net(invariants).chunk(2, dim=-1)
            learned = torch.einsum(
                "bm,bmk,bmd->bkd", weights, on_offsets, offsets
            ) + torch.einsum("bm,bmk,bmd->bkd", weights, on_normals, normals)
            candidates = torch.cat([global_vectors, learned], dim=1)

        e1 = F.normalize(candidates[:, 0], dim=-1)
        e2 = candidates[:, 1]
        e2 = F.normalize(e2 - (e2 * e1).sum(dim=-1, keepdim=True) * e1, dim=-1)
        e3 = torch.linalg.cross(e1, e2)
        return torch.stack([e1, e2, e3], dim=1)

    def _embed_points(
        self,
        points: Float[torch.Tensor, "B N 3"],
        boundary_points: Float[torch.Tensor, "B M 3"],
        boundary_features: Float[torch.Tensor, "B M F"],
        boundary_weights: Float[torch.Tensor, "B M"],
    ) -> Float[torch.Tensor, "B N C"]:
        r"""Embed frame-space points with a smooth summary of the nearby boundary."""
        k = min(self.n_neighbors, boundary_points.shape[1])
        index = torch.stack(
            [knn(b, p, k)[0] for b, p in zip(boundary_points, points)]
        )  # (B, N, k)
        batch = torch.arange(points.shape[0], device=points.device)[:, None, None]
        offsets = points[:, :, None] - boundary_points[batch, index]  # (B, N, k, 3)
        distance = offsets.norm(dim=-1)
        nearest = distance.amin(dim=-1, keepdim=True)
        # Close to the boundary only the nearest samples count; far away the
        # neighbors are averaged by measure.
        weights = boundary_weights[batch, index] * torch.exp(
            -(((distance - nearest) / (nearest + _MIN_DISTANCE)) ** 2)
        )
        weights = weights / weights.sum(dim=-1, keepdim=True)
        offset = (weights[..., None] * offsets).sum(dim=2)
        features = (weights[..., None] * boundary_features[batch, index]).sum(dim=2)
        return self.point_embed(
            torch.cat(
                [points, offset, offset.norm(dim=-1, keepdim=True), features], dim=-1
            )
        )

    def forward(
        self,
        points: Float[torch.Tensor, "B N 3"],
        boundary_points: Float[torch.Tensor, "B M 3"],
        boundary_normals: Float[torch.Tensor, "B M 3"],
        boundary_measure: Float[torch.Tensor, "B M"],
        boundary_scalars: Float[torch.Tensor, "B M Cs"] | None = None,
        boundary_vectors: Float[torch.Tensor, "B M Kv 3"] | None = None,
        global_scalars: Float[torch.Tensor, "B Cg"] | None = None,
        global_vectors: Float[torch.Tensor, "B Kg 3"] | None = None,
        point_measure: Float[torch.Tensor, "B N"] | None = None,
    ) -> tuple[Float[torch.Tensor, "B N S"], Float[torch.Tensor, "B N V 3"]]:
        r"""Predict scalar and vector fields at ``points``."""
        n_batch, n_boundary = boundary_points.shape[:2]
        if boundary_scalars is None:
            boundary_scalars = boundary_points.new_zeros(n_batch, n_boundary, 0)
        if boundary_vectors is None:
            boundary_vectors = boundary_points.new_zeros(n_batch, n_boundary, 0, 3)
        if global_scalars is None:
            global_scalars = boundary_points.new_zeros(n_batch, 0)
        if global_vectors is None:
            global_vectors = boundary_points.new_zeros(n_batch, 0, 3)
        if point_measure is None:
            point_measure = points.new_ones(points.shape[:2])
        if not torch.compiler.is_compiling():
            for name, tensor, expected in (
                ("boundary_scalars", boundary_scalars, self.n_boundary_scalars),
                ("boundary_vectors", boundary_vectors[..., 0], self.n_boundary_vectors),
                ("global_scalars", global_scalars, self.n_global_scalars),
                ("global_vectors", global_vectors[..., 0], self.n_global_vectors),
            ):
                if tensor.shape[-1] != expected:
                    raise ValueError(
                        f"Expected {expected} {name}, got {tensor.shape[-1]}"
                    )

        # Origin, reference length and frame: all derived from the boundary,
        # all measure-weighted so they do not depend on the boundary mesh.
        weights = boundary_measure / boundary_measure.sum(dim=-1, keepdim=True)
        origin = torch.einsum("bm,bmd->bd", weights, boundary_points)[:, None]
        offsets = boundary_points - origin
        length = torch.einsum("bm,bm->b", weights, offsets.square().sum(dim=-1)).sqrt()
        length = length[:, None, None]
        frame = self._frame(
            offsets / length,
            boundary_normals,
            weights,
            boundary_scalars,
            global_vectors,
        )

        def to_frame(vectors: torch.Tensor) -> torch.Tensor:
            return torch.einsum("b...d,bkd->b...k", vectors, frame)

        boundary_xyz = to_frame(offsets / length)
        boundary_features = torch.cat(
            [
                to_frame(boundary_normals),
                boundary_scalars,
                to_frame(boundary_vectors).flatten(-2),
            ],
            dim=-1,
        )
        cond = self.cond_embed(
            torch.cat(
                [
                    global_scalars.new_ones(n_batch, 1),
                    global_scalars,
                    to_frame(global_vectors).flatten(-2),
                ],
                dim=-1,
            )
        )

        # Boundary encoder: geometry and boundary conditions become context.
        boundary_log_measure = weights.log()
        context = self.boundary_embed(
            torch.cat([boundary_xyz, boundary_features], dim=-1)
        )
        for block in self.encoder:
            context = block(context, boundary_log_measure, cond)

        x = self._embed_points(
            to_frame((points - origin) / length),
            boundary_xyz,
            boundary_features,
            weights,
        )
        log_measure = (point_measure / point_measure.sum(dim=-1, keepdim=True)).log()
        for block in self.blocks:
            x = block(x, log_measure, cond, context, boundary_log_measure)

        out = self.out(self.out_norm(x))
        scalars = out[..., : self.out_scalars]
        components = out[..., self.out_scalars :].unflatten(-1, (self.out_vectors, 3))
        # Vector outputs are components in the frame; rotate them back.
        vectors = torch.einsum("bnvk,bkd->bnvd", components, frame)
        return scalars, vectors
