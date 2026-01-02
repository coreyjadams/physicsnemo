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

import time
import torch
import physicsnemo

import hydra
from omegaconf import OmegaConf, DictConfig


@hydra.main(config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, sort_keys=True))

    reader = hydra.utils.instantiate(cfg.reader)

    print(f"reader: {reader}")

    print(f"length of reader: {len(reader)}")

    for i in range(len(reader)):
        start = time.time()
        data = reader._load_sample(i)
        end = time.time()
        print(f"Iteration {i} time taken: {end - start:.2f} seconds")
        for key, value in data.items():
            print(f" - key: {key}, shape: {value.shape}, type: {type(value)}")


if __name__ == "__main__":
    main()
