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


from physicsnemo.datapipes.core import Dataset, DataLoader

from physicsnemo.datapipes.core.readers import TensorStoreZarrReader

from physicsnemo.datapipes.core.transforms import Normalize


def main():
    reader = TensorStoreZarrReader(
        # All the data here:
        path="./variable_size_data/zarr/",
        # Match to files / folders ending in .zarr
        group_pattern="*.zarr",
        # Enable this to set a coordinated subsampling from disk.  Meaning,
        # the a random - but consistent - slice of these tensors gets loaded and nothing else:
        coordinated_subsampling={
            "n_points": 10000,
            "target_keys": ["features", "coords"],
        },
    )

    # Add a transform to rescale features to between 0 and 1?
    normalize = Normalize(
        input_keys=["features"],
        means=1.0,  # Shift the mean by this amount, which presumably is measured on the data in advance
        stds=0.5,  # scale the std by this amount
    )

    dataset = Dataset(
        reader=reader,
        transforms=[normalize],
    )

    print(f"reader: {reader}")

    print(f"length of reader: {len(reader)}")

    # This loops over individual entries in the dataset:
    start = time.time()
    for i, data in enumerate(dataset):
        end = time.time()
        print(f"Dataset Iteration {i} time taken: {end - start:.2f} seconds")
        for key, value in data.items():
            print(
                f" - key: {key}, shape: {value.shape}, type: {type(value)}, mean of {value.mean():.2f}, std of {value.std():.2f}"
            )
        start = time.time()

    # Or, do it in a datapipe:
    datapipe = DataLoader(
        dataset=dataset,
        batch_size=5,
    )

    start = time.time()
    for i, batch in enumerate(datapipe):
        end = time.time()
        print(f"DataLoader Iteration {i} time taken: {end - start:.2f} seconds")
        for key, value in batch.items():
            print(
                f" - key: {key}, shape: {value.shape}, type: {type(value)}, mean of {value.mean():.2f}, std of {value.std():.2f}"
            )
        start = time.time()


if __name__ == "__main__":
    main()
