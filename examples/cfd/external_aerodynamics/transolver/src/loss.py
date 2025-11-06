# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
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

import torch
from typing import Literal

from physicsnemo.distributed import DistributedManager


def loss_fn(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: Literal["surface", "volume"],
) -> torch.Tensor:
    """
    Compute the main loss function for the model.

    Args:
        pred: Predicted tensor from the model.
        target: Ground truth tensor.
        others: Dictionary of additional tensors (e.g., surface_areas, surface_normals, stream_velocity).

    Returns:
        Loss value as a scalar tensor.
    """
    if mode == "surface":
        loss = loss_fn_surface(pred, target, "mse")
    elif mode == "volume":
        loss = loss_fn_volume(pred, target, "mse")
    return loss


def loss_fn_volume(
    output: torch.Tensor, target: torch.Tensor, loss_type: Literal["mse", "smooth_l1"]
) -> torch.Tensor:
    """Calculate loss for surface data by handling scalar and vector components separately.

    Args:
        output: Predicted surface values from the model.
        target: Ground truth surface values.
        loss_type: Type of loss to calculate ("mse" or "rmse").

    Returns:
        Combined scalar and vector loss as a scalar tensor.
    """

    # Separate the scalar and vector components:
    output_pressure, output_vel, output_nut = torch.split(output, [1, 3, 1], dim=2)
    target_pressure, target_vel, target_nut = torch.split(target, [1, 3, 1], dim=2)

    if loss_type == "mse":
        loss_pressure = torch.nn.functional.mse_loss(output_pressure, target_pressure)
        loss_wall_vel = torch.nn.functional.mse_loss(output_vel, target_vel)
        loss_nut = torch.nn.functional.mse_loss(output_nut, target_nut)

        loss = (loss_pressure + loss_wall_vel + loss_nut) / 4.0

    elif loss_type == "smooth_l1":
        loss_pressure = torch.nn.functional.smooth_l1_loss(
            output_pressure, target_pressure
        )
        loss_wall_vel = torch.nn.functional.smooth_l1_loss(output_vel, target_vel)
        loss_nut = torch.nn.functional.smooth_l1_loss(output_nut, target_nut)

        loss = loss_pressure + loss_wall_vel + loss_nut

    return loss


def loss_fn_surface(
    output: torch.Tensor, target: torch.Tensor, loss_type: Literal["mse", "rmse"]
) -> torch.Tensor:
    """Calculate loss for surface data by handling scalar and vector components separately.

    Args:
        output: Predicted surface values from the model.
        target: Ground truth surface values.
        loss_type: Type of loss to calculate ("mse" or "rmse").

    Returns:
        Combined scalar and vector loss as a scalar tensor.
    """
    # Separate the scalar and vector components:
    output_pressure, output_sheer = torch.split(output, [1, 3], dim=2)
    target_pressure, target_sheer = torch.split(target, [1, 3], dim=2)

    if loss_type == "mse":
        loss_pressure = torch.nn.functional.mse_loss(output_pressure, target_pressure)
        loss_wall_sheer = torch.nn.functional.mse_loss(output_sheer, target_sheer)

        loss = loss_pressure + loss_wall_sheer

    elif loss_type == "smooth_l1":
        loss_pressure = torch.nn.functional.smooth_l1_loss(
            output_pressure, target_pressure
        )
        loss_wall_sheer = torch.nn.functional.smooth_l1_loss(output_sheer, target_sheer)

        loss = loss_pressure + loss_wall_sheer

    return loss
