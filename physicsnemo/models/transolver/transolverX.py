# ignore_header_test
# ruff: noqa: E402
""""""

"""
Transolver model. This code was modified from, https://github.com/thuml/Transolver

The following license is provided from their source,

MIT License

Copyright (c) 2024 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
from einops import rearrange

try:
    import transformer_engine.pytorch as te

    TE_AVAILABLE = True
except (ImportError, FileNotFoundError):
    TE_AVAILABLE = False

import physicsnemo  # noqa: F401 for docs

from ..meta import ModelMetaData
from ..module import Module
from .Physics_Attention import (
    PhysicsAttentionIrregularMesh,
    gumbel_softmax,
)
from .transolver import MLP

ACTIVATION = {
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU(0.1),
    "softplus": nn.Softplus,
    "ELU": nn.ELU,
    "silu": nn.SiLU,
}


class PhysicsAttentionX(PhysicsAttentionIrregularMesh):
    """
    This is an extension of the Physics Attention Mechanism to support
    cross-attention with a context vector.
    """

    def __init__(
        self,
        dim,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = True,
        plus: bool = False,
        context_dim: int = 0,
    ):
        super().__init__(dim, heads, dim_head, dropout, slice_num, use_te, plus)

        linear_layer = te.Linear if self.use_te else nn.Linear

        # We have additional parameters, here:
        self.cross_q = linear_layer(dim_head, dim_head)
        self.cross_k = linear_layer(context_dim, dim_head)
        self.cross_v = linear_layer(context_dim, dim_head)

        # This is the learnable mixing weight between self and cross attention.
        self.state_mixing = nn.Parameter(torch.tensor(0.0))

    def compute_slice_attention_cross(
        self, slice_tokens: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the cross-attention between the slice tokens and the context.
        """

        # Project the slice and context tokens:
        q = self.cross_q(slice_tokens)
        k = self.cross_k(context)
        v = self.cross_v(context)

        # Compute the attention:
        if self.use_te:
            cross_attention = self.attn_fn(q, k, v)
        else:
            cross_attention = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=False
            )

        return cross_attention

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass of the Physics AttentionX module.

        Input x should have shape of [Batch, N_tokens, N_Channels] ([B, N, C])
        """

        # Project the inputs onto learned spaces:
        if self.plus:
            x_mid = self.project_input_onto_slices(x)
            # In transolver ++, fx_mid is gone.
            # x_mid is used to compute the projections instead:
            fx_mid = x_mid
        else:
            x_mid, fx_mid = self.project_input_onto_slices(x)

        # Perform the linear projection of learned latent space onto slices:
        slice_projections = self.in_project_slice(x_mid)

        # Slice projections has shape [B, N_head, N_tokens, Head_dim], but head_dim may have changed!

        # Use the slice projections and learned spaces to compute the slices, and their weights:
        slice_weights, slice_tokens = self.compute_slices_from_projections(
            slice_projections, fx_mid
        )
        # slice_weights has shape [Batch, N_heads, N_tokens, Slice_num]
        # slice_tokens has shape  [Batch, N_heads, N_tokens, head_dim]

        # Apply attention to the slice tokens
        if self.use_te:
            self_slice_token = self.compute_slice_attention_te(slice_tokens)
        else:
            self_slice_token = self.compute_slice_attention_sdpa(slice_tokens)

        # HERE, we are differing: apply cross-attention with physical states:
        if context is not None:
            cross_slice_token = self.compute_slice_attention_cross(
                slice_tokens, context
            )

            # Apply learnable mixing:
            mixing_weight = torch.sigmoid(self.state_mixing)
            out_slice_token = (
                mixing_weight * self_slice_token
                + (1 - mixing_weight) * cross_slice_token
            )

        else:
            # Just keep self attention:
            out_slice_token = self_slice_token

        # Shape unchanged

        # Deslice:
        outputs = self.project_attention_outputs(out_slice_token, slice_weights)

        # Outputs now has the same shape as the original input x

        return outputs


class TransolverX_block(nn.Module):
    """Transformer encoder block, replacing standard attention with physics attention."""

    def __init__(
        self,
        num_heads: int,
        hidden_dim: int,
        dropout: float,
        act="gelu",
        mlp_ratio=4,
        last_layer=False,
        out_dim=1,
        slice_num=32,
        use_te=True,
        plus: bool = False,
        context_dim: int = 0,
    ):
        super().__init__()

        if use_te and not TE_AVAILABLE:
            raise ImportError(
                "Transformer Engine is not installed. Please install it with `pip install transformer-engine`."
            )

        self.last_layer = last_layer
        if use_te:
            self.ln_1 = te.LayerNorm(hidden_dim)
        else:
            self.ln_1 = nn.LayerNorm(hidden_dim)

        self.Attn = PhysicsAttentionX(
            hidden_dim,
            heads=num_heads,
            dim_head=hidden_dim // num_heads,
            dropout=dropout,
            slice_num=slice_num,
            use_te=use_te,
            plus=plus,
            context_dim=context_dim,
        )

        if use_te:
            self.ln_mlp1 = te.LayerNormMLP(
                hidden_size=hidden_dim,
                ffn_hidden_size=hidden_dim * mlp_ratio,
            )
        else:
            self.ln_mlp1 = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                MLP(
                    hidden_dim,
                    hidden_dim * mlp_ratio,
                    hidden_dim,
                    n_layers=0,
                    res=False,
                    act=act,
                    use_te=False,
                ),
            )

    def forward(self, fx: torch.Tensor, global_context: torch.Tensor):
        fx = self.Attn(self.ln_1(fx), global_context) + fx
        fx = self.ln_mlp1(fx) + fx

        return fx


@dataclass
class MetaData(ModelMetaData):
    name: str = "Transolver"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class ContextProjector(nn.Module):
    """
    This context projector is like half of a learnable PhysicsAttention layer.
    It projects the context values onto a physical state space, but never back.
    It is used to construct the global context vectors in TransolverX.

    The global features are used in all the blocks of the model.
    """

    def __init__(
        self,
        dim,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        slice_num: int = 64,
        use_te: bool = True,
        plus: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.plus = plus
        self.scale = dim_head**-0.5
        self.use_te = use_te

        # Keep below here:
        if use_te:
            self.in_project_x = te.Linear(dim, inner_dim)
            if not plus:
                self.in_project_fx = te.Linear(dim, inner_dim)
        else:
            self.in_project_x = nn.Linear(dim, inner_dim)
            if not plus:
                self.in_project_fx = nn.Linear(dim, inner_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        if plus:
            linear_layer = te.Linear if self.use_te else nn.Linear
            self.proj_temperature = torch.nn.Sequential(
                linear_layer(self.dim_head, slice_num),
                nn.GELU(),
                linear_layer(slice_num, 1),
                nn.GELU(),
            )

        if self.use_te:
            self.in_project_slice = te.Linear(dim_head, slice_num)
        else:
            self.in_project_slice = nn.Linear(dim_head, slice_num)

    def project_input_onto_slices(
        self, x
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Project the input onto the slice space.
        """
        x_mid = rearrange(
            self.in_project_x(x), "B N (h d) -> B h N d", h=self.heads, d=self.dim_head
        )
        if self.plus:
            return x_mid
        else:
            fx_mid = rearrange(
                self.in_project_fx(x),
                "B N (h d) -> B h N d",
                h=self.heads,
                d=self.dim_head,
            )

            return x_mid, fx_mid

    def compute_slices_from_projections(
        self, slice_projections: torch.Tensor, fx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute slice weights and slice tokens from input projections and latent features.

        Args:
            slice_projections (torch.Tensor):
                The projected input tensor of shape [Batch, N_heads, N_tokens, Slice_num],
                representing the projection of each token onto each slice for each attention head.
            fx (torch.Tensor):
                The latent feature tensor of shape [Batch, N_heads, N_tokens, Head_dim],
                representing the learned states to be aggregated by the slice weights.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - slice_weights: Tensor of shape [Batch, N_heads, N_tokens, Slice_num],
                representing the normalized weights for each slice per token and head.
                - slice_token: Tensor of shape [Batch, N_heads, Slice_num, Head_dim],
                representing the aggregated latent features for each slice, head, and batch.

        Notes:
            - The function first computes a temperature-scaled softmax over the slice projections to obtain slice weights.
            - It then aggregates the latent features (fx) for each slice using these weights.
            - The aggregated features are normalized by the sum of weights for numerical stability.
        """

        # Project the latent space vectors on to the weight computation space,
        # and compute a temperature adjusted softmax.

        if self.plus:
            temperature = self.temperature + self.proj_temperature(fx)
            clamped_temp = torch.clamp(temperature, min=0.01).to(
                slice_projections.dtype
            )
            slice_weights = gumbel_softmax(
                slice_projections, clamped_temp
            )  # [Batch, N_heads, N_tokens, Slice_num]

        else:
            clamped_temp = torch.clamp(self.temperature, min=0.5, max=5).to(
                slice_projections.dtype
            )
            slice_weights = nn.functional.softmax(
                slice_projections / clamped_temp, dim=-1
            )  # [Batch, N_heads, N_tokens, Slice_num]

        # Cast to the computation type (since the parameter is probably fp32)
        slice_weights = slice_weights.to(slice_projections.dtype)

        # This does the projection of the latent space fx by the weights:

        # Computing the slice tokens is a matmul followed by a normalization.
        # It can, unfortunately, overflow in reduced precision, so normalize first:
        slice_norm = slice_weights.sum(2)  # [Batch, N_heads, Slice_num]
        normed_weights = slice_weights / (slice_norm[:, :, None, :] + 1e-2)
        slice_token = torch.matmul(normed_weights.transpose(2, 3), fx)

        # Return the original weights, not the normed weights:
        return slice_weights, slice_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        *Reduced* forward pass of the Physics Attention module.

        Input x should have shape of [Batch, N_tokens, N_Channels] ([B, N, C])
        """

        # All of this is derived from the PhysicsAttention Layer

        # Project the inputs onto learned spaces:
        if self.plus:
            x_mid = self.project_input_onto_slices(x)
            # In transolver ++, fx_mid is gone.
            # x_mid is used to compute the projections instead:
            fx_mid = x_mid
        else:
            x_mid, fx_mid = self.project_input_onto_slices(x)

        # Perform the linear projection of learned latent space onto slices:
        slice_projections = self.in_project_slice(x_mid)

        # Slice projections has shape [B, N_head, N_tokens, Head_dim], but head_dim may have changed!

        # Use the slice projections and learned spaces to compute the slices, and their weights:
        _, slice_tokens = self.compute_slices_from_projections(
            slice_projections, fx_mid
        )
        # _ has shape [Batch, N_heads, N_tokens, Slice_num]
        # slice_tokens has shape  [Batch, N_heads, N_tokens, head_dim]

        return slice_tokens


class TransolverX(Module):
    """
    Transolver model, adapted from original transolver code.

    Transolver is an adaptation of the transformer architecture, with a physics-attention
    mechanism replacing the standard attention mechanism.

    For more architecture details, see: https://arxiv.org/pdf/2402.02366 and https://arxiv.org/pdf/2502.02414

    Transolver can work on structured or unstructured data points as a model construction choice:
    - unstructured data (like a mesh) should provide some sort of positional encoding to accompany inputs
    - structured data (2D and 3D grids) can provide positional encodings optionally

    When constructing Transolver, you can choose to use "unified position" or not.  If you select "unified
    position" (`unified_pos=True`), then

    If using structured data, pass the structured shape as a tuple in the model constructor.
    Length 2 tuples are assumed to be image-like, length 3 tuples are assumed to be 3D voxel like.
    Other structured shape sizes are not supported.  Passing a structured_shape of None assumes irregular data.

    Output shape will have the same spatial shape as the input shape, with potentially more features

    Also can support Transolver++ implementation.  When using the distributed algorithm
    of Transolver++, use PhysicsNeMo's ShardTensor implementation to support automatic
    domain parallelism and 2D parallelization (data parallel + domain parallel, for example).

    Note
    ----


    Parameters
    ----------
    functional_dim : int
        The dimension of the input values, not including any embeddings.  No Default.
        Input will be concatenated with embeddings or unified position before processing
        with PhysicsAttention blocks.  Originally known as "fun_dim"
    out_dim : int
        The dimension of the output of the model.  This is a mandatory parameter.
    embedding_dim : int | None
        The spatial dimension of the input data embeddings.  Should include not just
        position but all computed embedding features.  Default is None, but if
        `unified_pos=False` this is a mandatory parameter.  Originally named "space_dim"
    n_layers : int
        The number of transformer PhysicsAttention layers in the model.  Default of 4.
    n_hidden : int
        The hidden dimension of the transformer.  Default of 256.  Projection is made
        from the input data + embeddings in the early preprocessing, before the
        PhysicsAttention layers.
    dropout : float
        The dropout rate, applied across the PhysicsAttention Layers.  Default is 0.0
    n_head : int
        The number of attention heads in each PhysicsAttention Layer.  Default is 8.  Note
        that the number of heads must evenly divide the `n_hidden` parameter to yield an
        integer head dimension.
    act : str
        The activation function, default is gelu.
    mlp_ratio : int
        The ratio of hidden dimension in the MLP, default is 4.  Used in the MLPs in the
        PhysicsAttention Layers.
    slice_num : int
        The number of slices in the PhysicsAttention layers.  Default is 32.  Represents the
        number of learned states each layer should project inputs onto.
    unified_pos : bool
        Whether to use unified positional embeddings.  Unified positions are only available for
        structured data (2D grids, 3D grids).  They are computed once initially, and reused through
        training in place of embeddings.
    ref : int
        The reference dimension size when using unified positions.  Default is 8.  Will be
        used to create a linear grid in spatial dimensions to serve as spatial embeddings.
        If `unified_pos=False`, this value is unused.
    structured_shape : None | tuple(int)
        The shape of the latent space.  If None, assumes irregular latent space.  If not
        `None`, this parameter can only be a length-2 or length-3 tuple of ints.
    use_te: bool
        Whether to use transformer engine backend when possible.
    time_input : bool
        Whether to include time embeddings. Default is false
    plus: bool
        Use Transolver++ implementation in the Physics Attention layers.

    """

    def __init__(
        self,
        functional_dim: int,
        out_dim: int,
        geometry_dim: int | None = None,
        global_dim: int | None = None,
        n_layers: int = 4,
        n_hidden: int = 256,
        dropout: float = 0.0,
        n_head: int = 8,
        act: str = "gelu",
        mlp_ratio: int = 4,
        slice_num: int = 32,
        use_te: bool = True,
        time_input: bool = False,
        plus: bool = False,
    ) -> None:
        super().__init__(meta=MetaData())
        self.__name__ = "Transolver"

        self.use_te = use_te
        # Check that the hidden dimension and head dimensions are compatible:
        if not n_hidden % n_head == 0:
            raise ValueError(
                f"Transolver requires n_hidden % n_head == 0, but instead got {n_hidden % n_head}"
            )

        # These are to project geometry embeddings and global embeddings onto
        # a physical state space:
        context_dim = 0
        if geometry_dim is not None:
            self.geometry_tokenizer = ContextProjector(
                geometry_dim,
                n_head,
                n_hidden // n_head,
                dropout,
                slice_num,
                use_te,
                plus,
            )
            context_dim += n_hidden // n_head
        if global_dim is not None:
            self.global_tokenizer = ContextProjector(
                global_dim, n_head, n_hidden // n_head, dropout, slice_num, use_te, plus
            )
            context_dim += n_hidden // n_head

        # This MLP is the initial projection onto the hidden space
        self.preprocess = MLP(
            functional_dim,
            n_hidden * 2,
            n_hidden,
            n_layers=0,
            res=False,
            act=act,
            use_te=use_te,
        )

        self.n_hidden = n_hidden

        self.blocks = nn.ModuleList(
            [
                TransolverX_block(
                    num_heads=n_head,
                    hidden_dim=n_hidden,
                    dropout=dropout,
                    act=act,
                    mlp_ratio=mlp_ratio,
                    slice_num=slice_num,
                    last_layer=(_ == n_layers - 1),
                    use_te=use_te,
                    plus=plus,
                    context_dim=context_dim,
                )
                for _ in range(n_layers)
            ]
        )

        if use_te:
            self.ln_mlp_out = te.LayerNormLinear(
                in_features=n_hidden, out_features=out_dim
            )
        else:
            self.ln_mlp_out = nn.Sequential(
                nn.LayerNorm(n_hidden),
                nn.Linear(n_hidden, out_dim),
            )

        self.time_input = time_input
        if time_input:
            self.time_fc = nn.Sequential(
                nn.Linear(n_hidden, n_hidden), nn.SiLU(), nn.Linear(n_hidden, n_hidden)
            )

    def project_geometry_to_states(self, geometry: torch.Tensor) -> torch.Tensor:
        geometry_features = self.geometry_project(geometry)
        geometry_features = geometry_features / self.geometry_temperature

    def forward(
        self,
        local_embedding: torch.Tensor,
        global_embedding: torch.Tensor | None = None,
        geometry: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """

        Forward pass of the TransolverX model.

        This model is an extension of Transolver.  Physical States
        are still used, and learned, with an addition: we initially
        construct a global embedding of "context" that gets used in every
        block.  This is a projection of global embedding and geometry information
        into physical states.  It is, in many ways, "half" of a PhysicsAttention
        layer: it projects onto states but never back.

        Args:
            local_embedding: torch.Tensor | None
                The local embedding of the input data.  Output will be similar shape.
            global_embedding: torch.Tensor | None
                The global embedding of the input data.  If None, it is not used.
            geometry: torch.Tensor | None
                The geometry of the input data.  If None, it is not used.
        """

        # First, construct the global context vectors:
        global_context_input = []

        if geometry is not None:
            geometry_states = self.geometry_tokenizer(geometry)
            global_context_input.append(geometry_states)

        if global_embedding is not None:
            global_states = self.global_tokenizer(global_embedding)
            global_context_input.append(global_states)

        # Construct the embedding states:
        if len(global_context_input) > 0:
            embedding_states = torch.cat(global_context_input, dim=-1)

        # Project the inputs to the hidden dimension:
        x = self.preprocess(local_embedding)

        for block in self.blocks:
            x = block(x, embedding_states)

        # Now, pass the data through the model:
        x = self.ln_mlp_out(x)

        return x
