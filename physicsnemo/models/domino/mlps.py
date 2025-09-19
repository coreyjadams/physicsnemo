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

import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    FlexibleMulti-layer perceptron (MLP) module.

    This is reused in various domino layers to simplify and unify
    the MLP implementations.
    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        base_layer: int,
        activation: nn.Module,
        n_layers: int,
    ):
        super(MLP, self).__init__()
        self.input_features = input_features

        modules = []

        if n_layers == 1:
            # Single layer: input_features -> output_features
            modules.append(nn.Linear(input_features, output_features))
        else:
            # First layer: input_features -> base_layer
            modules.append(nn.Linear(input_features, base_layer))
            modules.append(activation)

            # Hidden layers: base_layer -> base_layer
            for _ in range(n_layers - 2):
                modules.append(nn.Linear(base_layer, base_layer))
                modules.append(activation)

            # Final layer: base_layer -> output_features (no activation)
            modules.append(nn.Linear(base_layer, output_features))

        self.mlp_modules = torch.nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_modules(x)


class AggregationModel(MLP):
    """
    Neural network module to aggregate local geometry encoding with basis functions.

    This module combines basis function representations with geometry encodings
    to predict the final output quantities. It serves as the final prediction layer
    that integrates all available information sources.

    """

    def __init__(
        self,
        input_features: int,
        output_features: int,
        base_layer: int,
        activation: nn.Module,
    ):
        super().__init__(
            input_features=input_features,
            output_features=output_features,
            base_layer=base_layer,
            activation=activation,
            n_layers=5,
        )


class LocalPointConv(MLP):
    """Layer for local geometry point kernel"""

    def __init__(
        self,
        input_features: int,
        base_layer: int,
        output_features: int,
        activation: nn.Module,
    ):
        super().__init__(
            input_features=input_features,
            base_layer=base_layer,
            output_features=output_features,
            activation=activation,
            n_layers=2,
        )
