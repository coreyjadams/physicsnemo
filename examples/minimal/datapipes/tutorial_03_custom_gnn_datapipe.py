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
Tutorial 3: Custom Collation for Graph Neural Networks
=======================================================

This tutorial demonstrates how to build a GNN-ready data pipeline using
PhysicsNeMo DataPipes. You'll learn how to:

1. Build a transform that computes KNN graph edges
2. Use PyTorch Geometric's built-in batching via `Batch.from_data_list`
3. Put it all together in a GNN-ready pipeline

Prerequisites
-------------
Before running this tutorial, generate point cloud data:

    python generate_variable_points_data.py -n 100 -s "coords:3 features:8" --min-points 50000 --max-points 100000 -b zarr -o output/pointcloud_data/

Run this tutorial:
    python tutorial_03_custom_gnn_datapipe.py

Key Concepts
------------
- **Custom Transform**: Subclass `Transform` and implement `__call__()`
- **PyG Collator**: Use `torch_geometric.data.Batch.from_data_list()` for easy batching
- **PyG Batching**: Automatic edge index offsetting, feature concatenation, and batch tensor

GNN Batching Background
-----------------------
Graph Neural Networks require special batching because graphs have variable
numbers of nodes and edges. PyTorch Geometric uses a "disjoint graph" approach:
- Concatenate all node features into one large tensor
- Offset edge indices so each graph's edges point to the correct nodes
- Add a `batch` tensor indicating which graph each node belongs to
"""

import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data as PyGData

# Import core datapipe components
from physicsnemo.datapipes.core import DataLoader, Dataset
from physicsnemo.datapipes.core.collate import Collator
from physicsnemo.datapipes.core.readers import ZarrReader
from physicsnemo.datapipes.core.transforms import (
    KNNNeighbors,
    SubsamplePoints,
)


def check_data_exists(data_path, gen_cmd):
    """Check if tutorial data exists and provide helpful message if not."""
    path = Path(data_path)
    if not path.exists():
        print(f"ERROR: Data not found at '{data_path}'")
        print()
        print("Please generate tutorial data first:")
        print()
        print(f"    {gen_cmd}")
        print()
        return False
    return True


# =============================================================================
# Section 2: PyG-Style Graph Collator
# =============================================================================


class PyGCollator(Collator):
    """
    Collator that batches graphs using PyTorch Geometric's built-in batching.

    This collator converts each sample to a PyG Data object, then uses
    `Batch.from_data_list()` to handle all the complexity of graph batching:
    - Node features are concatenated: (N1 + N2 + ... + Nb, F)
    - Edge indices are automatically offset and concatenated
    - A `batch` tensor tracks which nodes belong to which graph

    Example:
        Graph 0: 100 nodes, edges [[0,1,2], [1,2,0]]
        Graph 1: 150 nodes, edges [[0,1], [1,0]]

        Batched (handled automatically by PyG):
        - nodes: (250, F)
        - edge_index: [[0,1,2,100,101], [1,2,0,101,100]]  # Graph 1 offset by 100
        - batch: [0]*100 + [1]*150
    """

    def __init__(
        self,
        edge_index_key: str = "edge_index",
        collate_metadata: bool = False,
    ) -> None:
        """
        Initialize the PyG-style collator.

        Args:
            edge_index_key: Key for edge indices in the input data.
                Expected shape is [num_nodes, k] from KNN, which will be
                converted to PyG's [2, num_edges] format.
        """
        self.collate_metadata = collate_metadata
        self.edge_index_key = edge_index_key

    @staticmethod
    def knn_to_edge_index(knn_indices: torch.Tensor) -> torch.Tensor:
        """
        Convert KNN indices to PyG edge_index format.

        Args:
            knn_indices: Tensor of shape [num_nodes, k] where each row contains
                the k nearest neighbor indices for that node.

        Returns:
            edge_index: Tensor of shape [2, num_nodes * k] in PyG COO format,
                where edge_index[0] is source nodes and edge_index[1] is target nodes.
        """
        num_nodes, k = knn_indices.shape
        # Source nodes: each node index repeated k times
        source = torch.arange(num_nodes, device=knn_indices.device).repeat_interleave(k)
        # Target nodes: flatten the KNN indices
        target = knn_indices.reshape(-1)
        return torch.stack([source, target], dim=0)

    def __call__(
        self, samples: Sequence[tuple[dict, dict[str, Any]]]
    ) -> tuple[PyGBatch, list[dict[str, Any]]]:
        """
        Collate graphs into a batched PyG Batch object.

        Args:
            samples: Sequence of (TensorDict/dict, metadata) tuples.

        Returns:
            Tuple of (PyG Batch, list of metadata dicts).
        """
        if not samples:
            raise ValueError("Cannot collate empty sequence of samples")

        # Separate data and metadata
        data_list = [data for data, _ in samples]

        # Convert each sample to a PyG Data object
        pyg_data_list = []
        for data in data_list:
            # Build kwargs for PyG Data, renaming edge_index_key to 'edge_index'
            data_kwargs = {}
            for key in data.keys():
                tensor = data[key]
                if key == self.edge_index_key:
                    # Convert from KNN format [num_nodes, k] to PyG format [2, num_edges]
                    data_kwargs["edge_index"] = self.knn_to_edge_index(tensor)
                else:
                    data_kwargs[key] = tensor

            pyg_data_list.append(PyGData(**data_kwargs))

        # Use PyG's built-in batching - handles edge index offsetting automatically
        batched_data = PyGBatch.from_data_list(pyg_data_list)

        if self.collate_metadata:
            metadata_list = [meta for _, meta in samples]
            return batched_data, list(metadata_list)
        else:
            return batched_data

    def __repr__(self) -> str:
        return f"PyGCollator(edge_index_key={self.edge_index_key})"


data_path = "./output/pointcloud_data/"
gen_cmd = 'python generate_variable_points_data.py -n 100 -s "coords:3 features:8" --min-points 50000 --max-points 100000 -b zarr -o output/pointcloud_data/'
# =============================================================================
# Section 3: Demonstration
# =============================================================================


def section_1_knn_transform():
    """
    Section 1: Computing KNN Graph Edges

    Shows how to use the ComputeKNNEdges transform to build
    graph structure from point cloud positions.
    """
    print("=" * 70)
    print("Section 1: Computing KNN Graph Edges")
    print("=" * 70)
    print()

    if not check_data_exists(data_path, gen_cmd):
        return None

    # Load a sample using ZarrReader
    reader = ZarrReader(path=data_path, group_pattern="*.zarr")
    data, metadata = reader[0]

    print(f"Loaded sample with {data['coords'].shape[0]} points")
    print(f"Fields: {list(data.keys())}")
    print()

    # Create and apply the KNN edge transform
    knn_transform = KNNNeighbors(
        points_key="coords",
        queries_key="coords",  # Apply the kNN to itself.
        k=8,
        extract_keys=["features"],
    )
    print(f"Transform: {knn_transform}")
    print()

    data_with_edges = knn_transform(data)

    edge_index = "neighbors_indices"

    print("After transform:")
    print(f"  Fields: {list(data_with_edges.keys())}")
    print(f"  edge_index shape: {data_with_edges[edge_index].shape}")
    print()

    # Verify graph structure
    n_nodes = data_with_edges["coords"].shape[0]
    n_edges = data_with_edges[edge_index].shape[1]

    print(f"Graph structure:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Edges / node: {n_edges}")
    print()

    reader.close()
    return knn_transform


def section_2_pyg_collator():
    """
    Section 2: PyG-Style Graph Batching

    Demonstrates how the PyGCollator uses PyG's Batch.from_data_list()
    to combine multiple graphs into a single batched graph.
    """
    print("=" * 70)
    print("Section 2: PyG-Style Graph Collator")
    print("=" * 70)
    print()

    if not check_data_exists(data_path, gen_cmd):
        return None

    reader = ZarrReader(path=data_path, group_pattern="*.zarr")
    knn_transform = KNNNeighbors(
        points_key="coords",
        queries_key="coords",  # Apply the kNN to itself.
        k=8,
        extract_keys=["features"],
    )

    # Load and transform a few samples
    print("Loading 3 individual graphs:")
    samples = []
    for i in range(3):
        data, meta = reader[i]
        data = knn_transform(data)
        samples.append((data, meta))
        n_nodes = data["coords"].shape[0]
        n_edges = data["neighbors_indices"].shape[1]
        print(f"  Graph {i}: {n_nodes} nodes, {n_edges} edges")

    print()

    # Apply collator - uses PyG's Batch.from_data_list() internally
    collator = PyGCollator(edge_index_key="neighbors_indices", collate_metadata=True)
    print(f"Collator: {collator}")
    print()

    batched_data, batch_metadata = collator(samples)

    print(f"Batched graph (type: {type(batched_data).__name__}):")
    for key in batched_data.keys():
        tensor = batched_data[key]
        print(f"  {key}: shape={tensor.shape}")
    print(f"Batch metadata: {batch_metadata}")

    print()
    print("Batch tensor distribution (nodes per graph):")
    batch = batched_data.batch
    for i in range(3):
        count = (batch == i).sum().item()
        print(f"  Graph {i}: {count} nodes")
    print()

    reader.close()


def section_3_complete_pipeline():
    """
    Section 3: Complete GNN Data Pipeline

    Puts everything together: reader, transforms, collator,
    and DataLoader for a complete GNN training pipeline.
    """
    print("=" * 70)
    print("Section 3: Complete GNN Data Pipeline")
    print("=" * 70)
    print()

    if not check_data_exists(data_path, gen_cmd):
        return

    # 1. Create reader
    print("Step 1: Create reader")
    reader = ZarrReader(path=data_path, group_pattern="*.zarr")
    print(f"  Reader: {len(reader)} samples")
    print()

    # 2. Define transforms
    print("Step 2: Define transforms")
    transforms = [
        # Subsample to fixed size for consistent batching
        SubsamplePoints(
            input_keys=["coords", "features"],
            n_points=500,
            algorithm="uniform",
        ),
        # Compute graph edges
        KNNNeighbors(
            points_key="coords",
            queries_key="coords",
            k=8,
            extract_keys=["features"],
        ),
    ]
    print(f"  Transforms: {[type(t).__name__ for t in transforms]}")
    print()

    # 3. Create dataset
    print("Step 3: Create dataset")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = Dataset(
        reader=reader,
        transforms=transforms,
        device=device,
    )
    print(f"  Dataset: {len(dataset)} samples on {device}")
    print()

    # 4. Create dataloader with PyG collator
    print("Step 4: Create DataLoader with PyG collator")
    collator = PyGCollator(edge_index_key="neighbors_indices")
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collator,
        collate_metadata=False,
    )
    print(f"  DataLoader: batch_size=4, {len(dataloader)} batches")
    print()

    # 5. Iterate over batches
    print("Step 5: Iterate over batches")
    print("-" * 50)

    start = time.time()
    for batch_idx, batch_data in enumerate(dataloader):
        elapsed = time.time() - start

        print(f"\nBatch {batch_idx} (loaded in {elapsed:.3f}s):")
        print(f"  Batch type: {type(batch_data)}")
        print(f"  Total nodes: {batch_data.coords.shape[0]}")
        print(f"  Total edges: {batch_data.edge_index.shape[1]}")

        # Show per-graph breakdown
        batch = batch_data.batch
        n_graphs = batch.max().item() + 1
        print(f"  Graphs in batch: {n_graphs}")
        for i in range(n_graphs):
            n_nodes = (batch == i).sum().item()
            print(f"    Graph {i}: {n_nodes} nodes")

        # Show data shapes for GNN input
        print(f"  Data shapes:")
        print(f"    coords: {batch_data.coords.shape}")
        print(f"    features: {batch_data.features.shape}")
        print(f"    edge_index: {batch_data.edge_index.shape}")
        print(f"    batch: {batch_data.batch.shape}")

        if batch_idx >= 1:
            print("\n  ... (showing first 2 batches)")
            break

        start = time.time()

    print()
    print("-" * 50)
    print()

    # 6. Show PyG integration
    print("Step 6: PyTorch Geometric Integration")
    print("-" * 50)
    print()
    print("The batch is already a PyG Batch object! Use it directly with PyG models:")
    print()
    print("    for pyg_batch, _ in dataloader:")
    print("        # pyg_batch is already a torch_geometric.data.Batch")
    print("        # Access attributes directly:")
    print("        #   pyg_batch.coords, pyg_batch.features, pyg_batch.edge_index")
    print("        #   pyg_batch.batch (node-to-graph assignment)")
    print("        output = model(pyg_batch.features, pyg_batch.edge_index)")
    print()

    dataset.close()


# =============================================================================
# Main
# =============================================================================


def main():
    """Run all tutorial sections."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " Tutorial 3: Custom Collation for GNNs ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    # Section 1: KNN transform
    section_1_knn_transform()

    # Section 2: PyG collator
    section_2_pyg_collator()

    # Section 3: Complete pipeline
    section_3_complete_pipeline()

    print("=" * 70)
    print("Tutorial 3 Complete!")
    print()
    print("Key takeaways:")
    print("  1. KNNNeighbors transform: Computes graph structure from point clouds")
    print("  2. PyGCollator: Uses Batch.from_data_list() for simple, correct batching")
    print("  3. Returns a PyG Batch object that works directly with PyG models")
    print("  4. The batch tensor tracks which nodes belong to which graph")
    print()
    print("Next: Tutorial 4 - Configuration with Hydra")
    print("=" * 70)


if __name__ == "__main__":
    main()
