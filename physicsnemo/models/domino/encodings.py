# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

"""
This code contains the DoMINO model architecture.
The DoMINO class contains an architecture to model both surface and
volume quantities together as well as separately (controlled using
the config.yaml file)
"""

import math

import torch
import torch.nn as nn

from .mlps import MLP


def fourier_encode_vectorized(
    coords: torch.Tensor, freqs: torch.Tensor
) -> torch.Tensor:
    """Vectorized Fourier feature encoding

    Args:
        coords: Tensor containing coordinates, of shape (batch_size, D)
        freqs: Tensor containing frequencies, of shape (F,) (num frequencies)

    Returns:
        Tensor containing Fourier features, of shape (batch_size, D * 2 * F)
    """

    D = coords.shape[-1]
    F = freqs.shape[0]

    freqs = freqs[None, None, :, None]  # reshape to [*, F, 1] for broadcasting

    coords = coords.unsqueeze(-2)  # [*, 1, D]
    scaled = (coords * freqs).reshape(*coords.shape[:-2], D * F)  # [*, D, F]
    features = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)  # [*, D, 2F]

    return features.reshape(*coords.shape[:-2], D * 2 * F)  # [*, D * 2F]


class EncodingMLP(nn.Module):
    """
    This is an MLP that will, optionally, fourier encode the input features.

    The encoded features are concatenated to the original inputs, and then
    processed with an MLP.

    Args:
        input_features: The number of input features to the MLP.
        base_layer: The number of neurons in the hidden layer of the MLP.
        fourier_features: Whether to fourier encode the input features.
        num_modes: The number of modes to use for the fourier encoding.
        activation: The activation function to use in the MLP.

    """

    def __init__(
        self,
        input_features: int,
        base_layer: int,
        fourier_features: bool,
        num_modes: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.fourier_features = fourier_features

        # self.num_modes = model_parameters.num_modes

        if self.fourier_features:
            input_features_calculated = input_features + input_features * num_modes * 2
            self.register_buffer(
                "freqs", torch.exp(torch.linspace(0, math.pi, self.num_modes))
            )
        else:
            input_features_calculated = input_features

        self.mlp = MLP(
            input_features=input_features_calculated,
            base_layer=base_layer,
            output_features=base_layer,
            activation=activation,
            n_layers=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fourier_features:
            x = torch.cat((x, fourier_encode_vectorized(x, self.freqs)), dim=-1)

        return self.mlp(x)
