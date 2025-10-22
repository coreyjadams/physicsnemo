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
This is a standalone script for benchmarking and testing the Transolver
datapipe in surface or volume mode.
"""

import time
import os
import re
import torch
import torchinfo

from typing import Literal, Any


import hydra
from omegaconf import DictConfig, OmegaConf


import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler


from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.datapipes.cae.transolver_datapipe import (
    TransolverDataPipe,
    create_transolver_dataset,
)


# This is included for GPU memory tracking:
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo


from physicsnemo.utils.profiling import profile, Profiler


@profile
def main(cfg: DictConfig):
    """Main training function

    Args:
        cfg: Hydra configuration object
    """

    DistributedManager.initialize()

    # Set up distributed training
    dist_manager = DistributedManager()

    # Set up logging
    logger = RankZeroLoggingWrapper(PythonLogger(name="training"), dist_manager)

    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    norm_data = np.load(norm_file)
    norm_factors = {
        "mean": torch.from_numpy(norm_data["mean"]).to(dist_manager.device),
        "std": torch.from_numpy(norm_data["std"]).to(dist_manager.device),
    }
    # Training dataset

    train_dataset = create_transolver_dataset(
        cfg.data,
        phase="train",
        scaling_factors=norm_factors,
    )

    # Validation dataset

    val_dataset = TransolverDataPipe(
        data_path=cfg.data.val.data_path,  # Assuming validation data path is configured
        max_workers=cfg.data.max_workers,
        pin_memory=cfg.data.pin_memory,
        keys_to_read=cfg.data.data_keys,
        large_keys=cfg.data.large_keys,
    )

    num_replicas = dist_manager.world_size
    data_rank = dist_manager.rank

    # Set up distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=num_replicas,
        rank=data_rank,
        shuffle=True,
        drop_last=True,
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=num_replicas,
        rank=data_rank,
        shuffle=False,  # No shuffling for validation
        drop_last=True,
    )

    # Load the normalization file:
    if cfg.data.mode == "surface":
        norm_file = "surface_fields_normalization.npz"
    elif cfg.data.mode == "volume":
        raise Exception("Volume training not yet supported.")

    # Set up optimizer and scheduler
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

    # Set up learning rate scheduler based on config
    scheduler_cfg = cfg.scheduler
    scheduler_name = scheduler_cfg.name
    scheduler_params = dict(scheduler_cfg.params)

    if scheduler_name == "OneCycleLR":
        scheduler_params.setdefault("max_lr", cfg.optimizer.lr)
        # Dynamically compute total_steps
        total_steps = len(list(train_sampler)) * cfg.training.num_epochs
        scheduler_params["total_steps"] = total_steps
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)
    elif scheduler_name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, **scheduler_params
        )
    elif scheduler_name == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    precision = getattr(cfg.training, "precision", "float32")
    scaler = GradScaler() if precision == "float16" else None

    ckpt_args = {
        "path": f"{cfg.output_dir}/{cfg.run_id}/checkpoints",
        "optimizer": optimizer,
        "scheduler": scheduler,
        "models": model,
    }

    loaded_epoch = load_checkpoint(device=dist_manager.device, **ckpt_args)

    if cfg.training.compile:
        model = torch.compile(model)

    # Training loop
    logger.info("Starting training...")
    for epoch in range(loaded_epoch, cfg.training.num_epochs):
        # Set the epoch in the samplers
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        start_time = time.time()
        # Training phase
        train_loss = train_epoch(
            train_dataset,
            train_sampler,
            model,
            output_pad_size,
            optimizer,
            scheduler,
            logger,
            writer,
            epoch,
            cfg,
            dist_manager,
            norm_factors,
            scaler,
        )
        end_time = time.time()
        train_duration = end_time - start_time

        start_time = time.time()
        # Validation phase
        val_loss = val_epoch(
            val_dataset,
            val_sampler,
            model,
            output_pad_size,
            logger,
            val_writer,
            epoch,
            cfg,
            dist_manager,
            norm_factors,
        )
        end_time = time.time()
        val_duration = end_time - start_time

        # Log epoch results
        logger.info(
            f"Epoch [{epoch}/{cfg.training.num_epochs}] Train Loss: {train_loss:.6f} [duration: {train_duration:.2f}s] Val Loss: {val_loss:.6f} [duration: {val_duration:.2f}s]"
        )

        # save checkpoint
        if epoch % cfg.training.save_interval == 0 and dist_manager.rank == 0:
            save_checkpoint(**ckpt_args, epoch=epoch)

        if scheduler_name == "StepLR":
            scheduler.step()

    logger.info("Training completed!")


@hydra.main(version_base=None, config_path="conf", config_name="train_surface")
def launch(cfg: DictConfig):
    """Launch training with hydra configuration

    Args:
        cfg: Hydra configuration object
    """

    # If you want to use `line_profiler` or PyTorch's profiler, enable them here.

    profiler = Profiler()
    # profiler.enable("torch")
    # profiler.enable("line_profiler")
    profiler.initialize()
    main(cfg)
    profiler.finalize()


if __name__ == "__main__":
    launch()
