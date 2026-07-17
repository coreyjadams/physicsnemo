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

import torch
import torch.distributed as dist
import torch.nn as nn

from model import HybridViT

from utils import (
    parse_args,
    print_and_save_results,
    get_csv_filename,
    save_result_incremental,
    end_to_end_benchmark,
)

from physicsnemo.distributed import DistributedManager

# Data parallelism: plain DDP works directly with ShardTensor activations.
from torch.nn.parallel import DistributedDataParallel as DDP

# FSDP2 is only needed when the model itself holds distributed (DTensor)
# parameters, which DDP cannot manage.
from torch.distributed.fsdp import fully_shard

# Imports for Domain Parallelism
from physicsnemo.domain_parallel import scatter_tensor
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.distributed.tensor.placement_types import (
    Replicate,
    Shard,
)


def shard_pos_embed(model: nn.Module, domain_mesh) -> None:
    """Shard the positional embedding across the domain mesh, in place.

    Only used on the FSDP2 path (``--fsdp``), where the parameters become
    distributed tensors anyway.  The positional embedding is laid out
    ``(1, num_patches, embed_dim)`` and the activations it is added to are
    sequence-sharded, so here it is *split* -- not replicated -- across the
    domain ranks.  (Without ``--fsdp`` it stays a plain replicated parameter
    and ShardTensor auto-promotes it in the forward pass.)

    Note carefully: this uses plain ``DTensor`` (``distribute_tensor``), not
    ``ShardTensor``.  Parameters are statically shaped, so DTensor's even-chunk
    sharding is exactly right; ``ShardTensor`` exists for the *activations*,
    whose sharding may be uneven and data-dependent.  ShardTensor interoperates
    with DTensor arguments directly, so the two mix freely in the forward pass.

    ``distribute_tensor`` broadcasts from rank 0 of the mesh by default, so this
    also synchronizes the embedding across the domain group.
    """
    model.register_parameter(
        "pos_embed",
        nn.Parameter(
            distribute_tensor(model.pos_embed.data, domain_mesh, [Shard(1)]),
            requires_grad=model.pos_embed.requires_grad,
        ),
    )


def broadcast_plain_params(model: nn.Module, domain_mesh) -> None:
    """Broadcast all plain (non-DTensor) parameters over the domain group.

    Every rank in a domain group holds a full replica of the non-spatial
    weights, so they must start identical.  DDP and FSDP2 are not involved on
    this axis (they operate on the ``ddp`` axis, or not at all), so we sync
    explicitly from rank 0 of the domain group.
    """
    group = domain_mesh.get_group()
    src = dist.get_global_rank(group, 0)
    with torch.no_grad():
        for param in model.parameters():
            if isinstance(param, DTensor):
                continue
            dist.broadcast(param.data, src=src, group=group)


def compile_non_attention_blocks(model: nn.Module) -> None:
    """Regionally compile the model, leaving sharded attention eager.

    With ``domain_size > 1`` the sequence-sharded SDPA dispatches to ring
    attention, which cannot live inside a compiled region (see
    ``physicsnemo/domain_parallel/shard_utils/attention_patches.py``).  So we
    compile everything around it: the patch embedding, each block's MLP and
    norms, and the classification head.

    ``dynamic=False``: every image size in the sweep runs fixed shapes, so
    symbolic tracing buys nothing here.  It also matters for correctness
    under FSDP2: all FSDP-managed submodules share one dynamo wrapper frame,
    so the norm/MLP/head compilations look like one function called with
    changing shapes, and automatic-dynamic would promote sizes to SymInts.
    """
    model.patch_embed = torch.compile(model.patch_embed, dynamic=False)
    for block in model.stages:
        block.norm1 = torch.compile(block.norm1, dynamic=False)
        block.norm2 = torch.compile(block.norm2, dynamic=False)
        block.mlp = torch.compile(block.mlp, dynamic=False)
    model.head = torch.compile(model.head, dynamic=False)


def main():
    """Main benchmarking script."""
    # Configuration

    args = parse_args()

    image_sizes = list(
        range(args.image_size_start, args.image_size_stop + 1, args.image_size_step)
    )
    device = torch.device("cuda")

    # Generate image sizes based on start, stop, and step
    if args.dimension == 2:
        image_sizes = list(
            range(args.image_size_start, args.image_size_stop + 1, args.image_size_step)
        )
    elif args.dimension == 3:
        image_sizes = list(
            range(
                args.image_size_start,
                args.image_size_stop + 1,
                args.image_size_step,
            )
        )

    # Initialize distributed manager first
    DistributedManager.initialize()
    dm = DistributedManager()

    # Set device based on local rank
    device = dm.device
    torch.cuda.set_device(device)

    # Resolve and validate the parallelism layout against the actual world
    # size.  The mesh must tile the whole job: ddp_size * domain_size ==
    # world_size.  This also guarantees the FSDP2 mesh (the "ddp" axis) is
    # consistent with the domain axis, since FSDP2 below shards over exactly
    # that ddp mesh dimension.
    if dm.world_size % args.domain_size != 0:
        raise ValueError(
            f"World size {dm.world_size} is not divisible by domain size "
            f"{args.domain_size}"
        )
    if args.ddp_size == -1:
        args.ddp_size = dm.world_size // args.domain_size
    if args.ddp_size * args.domain_size != dm.world_size:
        raise ValueError(
            f"ddp_size ({args.ddp_size}) x domain_size ({args.domain_size}) = "
            f"{args.ddp_size * args.domain_size} must equal the world size "
            f"({dm.world_size}). Pass --ddp_size -1 to infer it."
        )
    if args.fsdp and dm.world_size == 1:
        raise ValueError(
            "--fsdp requires a distributed run (world size > 1): no device "
            "mesh exists in a single-process job."
        )

    # Build the mesh whenever the job is distributed at all, so both axes are
    # explicit: DDP is handed the "ddp" mesh group directly (never the default
    # world group), and FSDP2 shards over that same axis.
    ddp_mesh = None
    domain_mesh = None
    if dm.world_size > 1:
        mesh = dm.initialize_mesh(
            mesh_shape=(
                args.ddp_size,
                args.domain_size,
            ),
            mesh_dim_names=["ddp", "domain"],
        )
        ddp_mesh = mesh["ddp"]
        domain_mesh = mesh["domain"]

    num_classes = 1000
    precision_mode = (
        "FP16" if args.use_mixed_precision and torch.cuda.is_available() else "FP32"
    )

    if dm.rank == 0:
        print(f"Device: {device}")
        print(f"Batch size: {args.batch_size}")
        print(f"Domain size: {args.domain_size}")
        print(f"DDP size: {args.ddp_size}")
        print(f"Number of classes: {num_classes}")
        print(f"Precision: {precision_mode}")
        print(f"torch.compile: {args.compile}")
        print("-" * 80)

    results = []

    ddp_size = args.ddp_size
    domain_size = args.domain_size

    # Set up incremental CSV output
    csv_filename = get_csv_filename(args, precision_mode)

    for img_size in image_sizes:
        if dm.rank == 0:
            print(f"\nTesting image size: {img_size}x{img_size}")

        # Each image size traces to different shapes; drop stale compiled
        # graphs so we never hit the recompilation limit across the sweep.
        if args.compile:
            torch._dynamo.reset()

        if args.dimension == 2:
            full_img_size = (img_size, img_size)
        elif args.dimension == 3:
            full_img_size = (img_size, img_size, img_size)

        if args.batch_size % ddp_size != 0 or args.batch_size // ddp_size == 0:
            raise ValueError(
                f"Batch size {args.batch_size} is not divisible by DDP size {ddp_size}"
            )

        # Create synthetic data - scale the batch size down by DDP size.
        x = torch.randn(args.batch_size // ddp_size, 3, *full_img_size, device=device)
        target = torch.randint(
            0, num_classes, (args.batch_size // ddp_size,), device=device
        )

        # Domain Parallel NOTE: we're generating data once per GPU but only keeping the data once per domain.
        # In a real application, you'd do this properly - each GPU would read it's own shard of the data.

        if args.domain_size > 1:
            # When scattering the data, we need to know the global rank of the source
            # But by definition, we use the domain_rank == 0 as the source.  Convert:
            global_rank_of_source = torch.distributed.get_global_rank(
                domain_mesh.get_group(), 0
            )

            # Scatter the input data across the domain:
            x = scatter_tensor(
                x,
                global_rank_of_source,
                domain_mesh,
                placements=(
                    Shard(2),
                ),  # Shard along the 2nd dimension (B C **H** W) which is the Height
                global_shape=x.shape,  # This will be inferred if not provided!
                dtype=x.dtype,  # This will be inferred if not provided!
            )

            target = scatter_tensor(
                target,
                global_rank_of_source,
                domain_mesh,
                placements=(
                    Replicate(),
                ),  # REPLICATE the target - gradients will still be scattered properly.
                global_shape=target.shape,  # This will be inferred if not provided!
                dtype=target.dtype,  # This will be inferred if not provided!
            )

        # The model is a completely vanilla nn.Module in every configuration.
        # ShardTensor activations auto-promote plain weights when they meet in
        # the forward pass, so no distribute_module / model surgery is needed.
        model = HybridViT(
            img_size=full_img_size, in_channels=3, num_classes=num_classes
        )
        model = model.to(device)

        if args.domain_size > 1:
            if args.fsdp:
                # On the FSDP2 path the parameters become distributed tensors
                # anyway, so split the positional embedding across the domain
                # to match the sequence-sharded activations it is added to.
                # Without --fsdp it stays plain and replicated: ShardTensor
                # auto-promotes it in the forward pass, which keeps every
                # parameter a plain tensor and lets ordinary DDP manage the
                # model.
                shard_pos_embed(model, domain_mesh)

            # Sync the replicated weights across the domain group.  Neither
            # DDP nor FSDP2 will do this for us on the domain axis.
            broadcast_plain_params(model, domain_mesh)

        if args.compile and args.domain_size > 1:
            # Sharded (ring) attention can't live inside a compiled region;
            # compile the rest of the model around it.  Do this on the
            # underlying module, before any FSDP2 wrapping.
            compile_non_attention_blocks(model)

        if args.fsdp:
            # Opt-in parameter sharding: shard the weights over the ddp axis
            # with FSDP2.  Gradients of the replicated weights are already
            # reduced over the domain axis by ShardTensor's gradient boundary,
            # so FSDP2 only needs the ddp mesh.  With ddp_size == 1 this is a
            # degenerate (size-1) shard - no communication, but the parameters
            # are uniformly DTensors on every --fsdp configuration.
            fully_shard(model, mesh=ddp_mesh)
        elif args.ddp_size > 1:
            # Replicated data parallelism: all parameters are plain
            # tensors (pos_embed is only distributed on the FSDP2 path),
            # so standard DDP just works - it broadcasts weights at
            # construction, all-reduces gradients in backward, and accepts
            # ShardTensor activations directly.  Pass the ddp mesh group
            # explicitly rather than relying on the default (world) group,
            # so DDP's size always matches the mesh axis.
            model = DDP(
                model,
                device_ids=[dm.local_rank],
                output_device=dm.local_rank,
                process_group=ddp_mesh.get_group(),
            )

        if args.compile and args.domain_size == 1:
            # No sharded attention in the graph: compile the whole model
            # (DDP-wrapped models compile fine via DDPOptimizer).  Fixed
            # shapes per sweep step, so no dynamic tracing.
            model = torch.compile(model, dynamic=False)

        result = end_to_end_benchmark(
            args, model, (x, target), full_img_size, device, num_classes
        )
        results.append(result)

        if dm.rank == 0:
            save_result_incremental(csv_filename, result, args, dm.world_size)
            print(f"Completed image size: {img_size}x{img_size}")

    if dm.rank == 0:
        print_and_save_results(results, args, precision_mode, dm.world_size)


if __name__ == "__main__":
    main()
