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

"""Write the split manifest the recipe reads for AeroZoo.

The recipe selects samples from ``<root>/manifest.json``, a mapping of split
part (``full_train``, ``bank_cars_test``, ...) to case IDs. This writes it
from the dataset's own ``splits/splits.json`` and ``release-v1.3.json``,
following the benchmark rules: only ``mean_stable`` cases, and no body-free
ducts in the test sets of the ``bank_*`` and ``layout`` splits. Pass
``--all`` to keep every released case instead.

    python datasets/aerozoo_manifest.py /path/to/AeroZoo
"""

import argparse
import json
from pathlib import Path


def build_manifest(root: Path, canonical: bool = True) -> dict[str, list[str]]:
    """Split part -> case IDs, filtered by the benchmark rules if ``canonical``."""
    splits = json.loads((root / "splits" / "splits.json").read_text())
    if not canonical:
        return splits
    release = json.loads((root / "release-v1.3.json").read_text())
    stable = {
        case
        for case, record in release["cases"].items()
        if record.get("solution_quality") == "mean_stable"
    }
    body_free = set(
        json.loads((root / "splits" / "body-free-ducts.json").read_text())["case_ids"]
    )
    manifest = {}
    for part, cases in splits.items():
        split, kind = part.rsplit("_", 1)
        body_only = kind == "test" and (split.startswith("bank_") or split == "layout")
        manifest[part] = [
            case
            for case in cases
            if case in stable and not (body_only and case in body_free)
        ]
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="AeroZoo root directory")
    parser.add_argument("--all", action="store_true", help="keep unstable cases too")
    args = parser.parse_args()
    manifest = build_manifest(args.root, canonical=not args.all)
    (args.root / "manifest.json").write_text(json.dumps(manifest))
    print(f"Wrote {args.root / 'manifest.json'}: {len(manifest)} split parts")
