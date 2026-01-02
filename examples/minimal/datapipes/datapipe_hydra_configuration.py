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

"""
Example: Datapipe Configuration via Hydra

This script demonstrates the same functionality as datapipe_python_configuration.py
but configured entirely via Hydra YAML configuration files.

Configuration files:
  - conf/datapipe_demo.yaml (main config)
  - conf/reader/tensorstore_zarr.yaml (reader config)
  - conf/transforms/normalize.yaml (transforms config)

All components including coordinated_subsampling are configured declaratively
in YAML and instantiated via hydra.utils.instantiate().
"""

import time

import hydra
from omegaconf import DictConfig, OmegaConf

import physicsnemo  # noqa: F401
from physicsnemo.datapipes.core import Dataset, DataLoader


@hydra.main(version_base=None, config_path="./conf", config_name="datapipe_demo")
def main(cfg: DictConfig):
    """Main function demonstrating Hydra-based datapipe configuration."""

    # Print the resolved configuration
    print("=" * 80)
    print("Resolved Configuration:")
    print("=" * 80)
    print(OmegaConf.to_yaml(cfg, sort_keys=False))
    print("=" * 80)
    print()

    # Instantiate reader, transforms, and dataset from config
    # Coordinated subsampling is now configured directly in the reader config
    reader = hydra.utils.instantiate(cfg.reader)
    transforms = hydra.utils.instantiate(cfg.transforms)

    # Create dataset with reader and transforms
    dataset = Dataset(
        reader=reader,
        transforms=[transforms],
    )

    print(f"Reader: {reader}")
    print(f"Length of reader: {len(reader)}")
    print()

    # Loop over individual entries in the dataset
    print("=" * 80)
    print("Iterating over Dataset (individual samples)")
    print("=" * 80)
    start = time.time()
    for i, data in enumerate(dataset):
        end = time.time()
        print(f"Dataset Iteration {i} time taken: {end - start:.2f} seconds")
        for key, value in data.items():
            print(
                f" - key: {key}, shape: {value.shape}, type: {type(value)}, "
                f"mean of {value.mean():.2f}, std of {value.std():.2f}"
            )
        start = time.time()
    print()

    # Create DataLoader
    print("=" * 80)
    print("Iterating over DataLoader (batched samples)")
    print("=" * 80)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.dataloader.batch_size,
    )

    start = time.time()
    for i, batch in enumerate(dataloader):
        end = time.time()
        print(f"DataLoader Iteration {i} time taken: {end - start:.2f} seconds")
        for key, value in batch.items():
            print(
                f" - key: {key}, shape: {value.shape}, type: {type(value)}, "
                f"mean of {value.mean():.2f}, std of {value.std():.2f}"
            )
        start = time.time()


if __name__ == "__main__":
    main()
