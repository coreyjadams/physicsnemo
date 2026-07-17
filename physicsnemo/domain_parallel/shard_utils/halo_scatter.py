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

r"""``torch.compile``-safe halo scatter-correction for ShardTensor.

A ``Shard(0)`` ShardTensor with a ``[owned | borrowed-ghost]`` row layout must,
after an in-place row scatter that writes into ghost rows, fold those
contributions back into their owners and refresh the ghost rows from the corrected
owners. This module exposes that correction as an AOT-traceable primitive:

* :func:`halo_reverse_exchange` / :func:`halo_forward_exchange` -- the
  fold-to-owner and refresh-ghost halves.
* :func:`halo_scatter_correct` -- the fused ``forward(reverse(padded))`` as a single
  ``torch.library.custom_op`` (opaque to fake mode, a packed-tensor routing arg, a
  self-adjoint backward) that survives both ``aot_eager`` and inductor.

Routing is passed as a packed tensor (from :func:`pack_halo_routing`), so the
primitive makes no partitioner or spatial assumptions and the values may change
across steps without recompiling. The data movement is a pluggable backend
(:class:`_HaloBackend`) chosen by :func:`select_halo_backend`
(``PHYSICSNEMO_HALO_BACKEND`` override), and all entry points accept a neighbour
sub-``group`` to bound the coordination span. :func:`register_halo_scatter_handlers`
wires the correction onto ``ShardTensor.scatter_add`` / ``index_add``.
"""

from __future__ import annotations

import os
from typing import Protocol

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol
from torch.distributed.device_mesh import DeviceMesh

__all__ = [
    "funcol_all_to_all_v_rows",
    "halo_forward_exchange",
    "halo_reverse_exchange",
    "halo_scatter_correct",
    "pack_halo_routing",
    "register_halo_scatter_handlers",
    "select_halo_backend",
]


def _funcol_group_arg(group: object) -> object:
    r"""Return *group* in the form functional collectives accept: ``(DeviceMesh,
    0)``, a ``ProcessGroup`` / group-name ``str`` unchanged, or the default world
    group for ``None`` (funcol rejects ``None``)."""
    if isinstance(group, DeviceMesh):
        return (group, 0)
    if group is None:
        return dist.distributed_c10d._get_default_group()
    return group


def _halo_group_name(group: object) -> str:
    r"""Resolve *group* to its c10d group-name string (``""`` for the default world
    group) -- the traceable token a ``custom_op`` can carry, since a
    ``ProcessGroup`` is not a valid op argument."""
    if group is None:
        return ""
    if isinstance(group, str):
        return group
    if isinstance(group, DeviceMesh):
        return group._dim_group_names[0]
    return group.group_name


def funcol_all_to_all_v_rows(
    send_rows: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: object = None,
) -> torch.Tensor:
    r"""AOT-traceable variable-sized ``all_to_all`` over the rows (dim 0) of a
    tensor, via ``funcol.all_to_all_single``.

    Parameters
    ----------
    send_rows : torch.Tensor
        ``(sum(send_counts), *F)`` send buffer, rows ordered by destination rank.
    send_counts : list[int]
        Rows sent to each rank (plain ``int`` -- graph constants under compile).
    recv_counts : list[int]
        Rows received from each rank.
    group : ProcessGroup or DeviceMesh or str or None, optional, default=None
        Collective group; ``None`` resolves to the default world group.

    Returns
    -------
    torch.Tensor
        ``(sum(recv_counts), *F)`` received rows, ordered by source rank.
    """
    trailing = tuple(send_rows.shape[1:])
    row_size = 1
    for d in trailing:
        row_size *= d
    flat_send = send_rows.contiguous().reshape(-1)
    send_flat = [c * row_size for c in send_counts]
    recv_flat = [c * row_size for c in recv_counts]
    total_recv = sum(recv_counts)
    flat_recv = funcol.wait_tensor(
        funcol.all_to_all_single(
            flat_send, recv_flat, send_flat, _funcol_group_arg(group)
        )
    )
    return flat_recv.reshape((total_recv,) + trailing)


# Backend seam: a transport owns the whole reverse/forward exchange, because the
# data-movement structure -- not just the collective call -- is transport-specific.
# The funcol backend builds dense destination-ordered buffers and loops over the
# whole group (what ``all_to_all_single`` needs); the symmetric-memory backend stages
# into a symmetric workspace and pulls each neighbour block with ``get_buffer``.


class _HaloBackend(Protocol):
    name: str

    def reverse(
        self,
        padded: torch.Tensor,
        n_owned: int,
        send_indices: list[torch.Tensor],
        send_sizes: list[list[int]],
        rank: int,
        world_size: int,
        group: object,
    ) -> torch.Tensor: ...

    def forward(
        self,
        owned: torch.Tensor,
        send_indices: list[torch.Tensor],
        send_sizes: list[list[int]],
        rank: int,
        world_size: int,
        group: object,
    ) -> torch.Tensor: ...


class _FuncolHaloBackend:
    r"""Functional-collective transport: dense ``all_to_all_single`` over the whole
    group. Portable everywhere (incl. gloo/CPU) and the default fallback."""

    name = "funcol"

    def reverse(
        self, padded, n_owned, send_indices, send_sizes, rank, world_size, group
    ):
        r"""Fold ghost rows back into owners via a reverse row all-to-all-v."""
        ghost = padded[n_owned:].contiguous()

        # Send each ghost block back to the owner it was borrowed from.
        rev_indices: list[torch.Tensor] = []
        offset = 0
        for r in range(world_size):
            n = int(send_sizes[r][rank])
            rev_indices.append(
                torch.arange(
                    offset, offset + n, device=padded.device, dtype=torch.int64
                )
            )
            offset += n
        send_rows = torch.cat(
            [ghost.index_select(0, rev_indices[j]) for j in range(world_size)], dim=0
        )
        send_counts = [int(send_sizes[r][rank]) for r in range(world_size)]
        recv_counts = [int(send_sizes[rank][j]) for j in range(world_size)]
        received_back = funcol_all_to_all_v_rows(
            send_rows, send_counts, recv_counts, group
        )

        # Fold the returned contributions into the lent owned rows (float64
        # accumulator keeps the sum well-conditioned for float32 inputs).
        acc_dtype = torch.float64 if padded.dtype == torch.float32 else padded.dtype
        owned = padded[:n_owned].to(acc_dtype)
        offset = 0
        for j in range(world_size):
            n = int(send_sizes[rank][j])
            if n == 0:
                continue
            owned = owned.index_add(
                0, send_indices[j], received_back[offset : offset + n].to(acc_dtype)
            )
            offset += n
        return owned.to(padded.dtype)

    def forward(self, owned, send_indices, send_sizes, rank, world_size, group):
        r"""Refresh ghost rows from owners via a forward row all-to-all-v."""
        send_rows = torch.cat(
            [owned.index_select(0, send_indices[j]) for j in range(world_size)], dim=0
        )
        send_counts = [int(send_sizes[rank][j]) for j in range(world_size)]
        recv_counts = [int(send_sizes[i][rank]) for i in range(world_size)]
        ghost_new = funcol_all_to_all_v_rows(send_rows, send_counts, recv_counts, group)
        return torch.cat([owned, ghost_new], dim=0)


def _symm_group_name(group: object) -> str:
    r"""Resolve *group* to a c10d group-name string usable with
    ``get_symm_mem_workspace`` (unlike :func:`_halo_group_name`, ``None`` resolves to
    the *named* default world group, not ``""``)."""
    if group is None:
        return dist.distributed_c10d._get_default_group().group_name
    if isinstance(group, str):
        return group
    if isinstance(group, DeviceMesh):
        return group._dim_group_names[0]
    return group.group_name


def _global_max_staged_rows(send_sizes: list[list[int]], world_size: int) -> int:
    r"""Rows the symmetric workspace must hold on every rank: the group-wide max over
    ranks of ``max(ghost rows, lent rows)``. Identical on all ranks (all hold the full
    ``send_sizes``), so the symmetric allocation stays uniform."""
    m = 0
    for r in range(world_size):
        ghost = sum(int(send_sizes[i][r]) for i in range(world_size))
        lent = sum(int(send_sizes[r][j]) for j in range(world_size))
        m = max(m, ghost, lent)
    return m


def _require_symm_mem(tensor: torch.Tensor):
    r"""Return the ``_symmetric_memory`` module, or raise a clear error when the
    symmetric-memory backend cannot serve *tensor* (CPU, or torch without it)."""
    if not tensor.is_cuda:
        raise RuntimeError(
            "the symmetric-memory halo backend requires CUDA tensors; "
            "set PHYSICSNEMO_HALO_BACKEND=funcol for CPU/gloo."
        )
    try:
        import torch.distributed._symmetric_memory as symm_mem
    except Exception as exc:  # pragma: no cover - torch build without symm-mem
        raise RuntimeError(
            "torch.distributed._symmetric_memory is unavailable; "
            "set PHYSICSNEMO_HALO_BACKEND=funcol."
        ) from exc
    return symm_mem


# Signal channels for the per-neighbour readiness / completion fences. Reverse and
# forward use disjoint channels so a straggler's reverse signal is never mistaken for a
# forward one.
_REV_READY, _REV_DONE, _FWD_READY, _FWD_DONE = 0, 1, 2, 3


class _SymmMemHaloBackend:
    r"""Symmetric-memory one-sided transport (NVSHMEM device-initiated across nodes,
    CUDA-IPC ``get_buffer`` within a node).

    Each rank stages its exchange block into a symmetric workspace, then *pulls* each
    neighbour's block with ``get_buffer``. Only real ghost/lent data moves, and the
    coordination is neighbour-local: ``put_signal`` / ``wait_signal`` fence each
    exchange peer-to-peer (readiness before a pull, completion before a buffer is
    reused) instead of a group-wide ``barrier``, so a rank synchronizes with
    O(neighbours) peers rather than O(world). Numerically identical to
    :class:`_FuncolHaloBackend` (the correctness oracle); the region offsets mirror that
    backend's dense destination-ordered layout.
    """

    name = "symm_mem"

    @staticmethod
    def _row_numel(feat_shape: tuple[int, ...]) -> int:
        n = 1
        for d in feat_shape:
            n *= int(d)
        return n

    def reverse(
        self, padded, n_owned, send_indices, send_sizes, rank, world_size, group
    ):
        r"""Fold ghost rows back into owners via a one-sided reverse exchange."""
        symm_mem = _require_symm_mem(padded)
        feat = tuple(padded.shape[1:])
        row_numel = self._row_numel(feat)
        dtype = padded.dtype
        group_name = _symm_group_name(group)
        max_rows = _global_max_staged_rows(send_sizes, world_size)

        # Readers pull FROM my buffer (peers I borrowed from); sources are the peers I
        # pull from (peers I lent to). Reverse sends ghost contributions back to owners.
        readers = [s for s in range(world_size) if int(send_sizes[s][rank]) > 0]
        sources = [j for j in range(world_size) if int(send_sizes[rank][j]) > 0]

        acc_dtype = torch.float64 if dtype == torch.float32 else dtype
        owned = padded[:n_owned].to(acc_dtype)
        with torch.cuda.device(padded.device):
            handle = symm_mem.get_symm_mem_workspace(
                group_name, max(1, max_rows * row_numel * padded.element_size())
            )
            # Stage this rank's whole ghost region ([from_0 | from_1 | ...]); the block
            # borrowed from owner o already sits at o's pull offset. Signal each reader
            # its data is staged.
            ghost = padded[n_owned:].contiguous()
            if ghost.shape[0]:
                handle.get_buffer(rank, tuple(ghost.shape), dtype).copy_(ghost)
            for r in readers:
                handle.put_signal(r, channel=_REV_READY)
            # Pull each lent-to peer's staged contributions and fold them into the rows
            # this rank lent; signal that peer its buffer is free once the read is done.
            for j in sources:
                n = int(send_sizes[rank][j])
                off = sum(int(send_sizes[d][j]) for d in range(rank)) * row_numel
                handle.wait_signal(j, channel=_REV_READY)
                recv = handle.get_buffer(j, (n, *feat), dtype, storage_offset=off)
                owned = owned.index_add(0, send_indices[j], recv.to(acc_dtype))
                handle.put_signal(j, channel=_REV_DONE)
            # Hold until every reader has finished pulling, so the next phase's staging
            # cannot overwrite this buffer mid-read.
            for r in readers:
                handle.wait_signal(r, channel=_REV_DONE)
        return owned.to(dtype)

    def forward(self, owned, send_indices, send_sizes, rank, world_size, group):
        r"""Refresh ghost rows from the corrected owners via a one-sided exchange."""
        symm_mem = _require_symm_mem(owned)
        feat = tuple(owned.shape[1:])
        row_numel = self._row_numel(feat)
        dtype = owned.dtype
        group_name = _symm_group_name(group)
        max_rows = _global_max_staged_rows(send_sizes, world_size)

        # Forward broadcasts owners to ghosts, so the roles swap: readers are the peers
        # I lent to; sources are the peers I borrowed from.
        readers = [j for j in range(world_size) if int(send_sizes[rank][j]) > 0]
        sources = [i for i in range(world_size) if int(send_sizes[i][rank]) > 0]

        with torch.cuda.device(owned.device):
            handle = symm_mem.get_symm_mem_workspace(
                group_name, max(1, max_rows * row_numel * owned.element_size())
            )
            # Stage the rows lent to each peer, destination-ordered; signal each reader.
            send_rows = torch.cat(
                [owned.index_select(0, send_indices[j]) for j in range(world_size)],
                dim=0,
            )
            if send_rows.shape[0]:
                handle.get_buffer(rank, tuple(send_rows.shape), dtype).copy_(send_rows)
            for r in readers:
                handle.put_signal(r, channel=_FWD_READY)
            # Pull each refreshed ghost block from its owner (source-rank order); signal
            # that owner its buffer is free once the block is copied out.
            ghost_blocks = {}
            for i in sources:
                n = int(send_sizes[i][rank])
                off = sum(int(send_sizes[i][d]) for d in range(rank)) * row_numel
                handle.wait_signal(i, channel=_FWD_READY)
                gb = handle.get_buffer(i, (n, *feat), dtype, storage_offset=off)
                ghost_blocks[i] = gb.clone()
                handle.put_signal(i, channel=_FWD_DONE)
            for r in readers:
                handle.wait_signal(r, channel=_FWD_DONE)
        if not ghost_blocks:
            return owned
        ghost_new = torch.cat([ghost_blocks[i] for i in sources], dim=0)
        return torch.cat([owned, ghost_new], dim=0)


_FUNCOL_BACKEND = _FuncolHaloBackend()
_SYMM_MEM_BACKEND = _SymmMemHaloBackend()


_symm_capability_cache: dict[str, bool] = {}


def _resolve_pg(group: object):
    r"""Resolve *group* to a ``ProcessGroup`` (or ``None`` if it cannot be), for the
    backend check in :func:`_symm_mem_usable`."""
    if group is None:
        return dist.distributed_c10d._get_default_group()
    if isinstance(group, DeviceMesh):
        try:
            return group.get_group() if group.ndim == 1 else group.get_group(0)
        except Exception:
            return None
    if isinstance(group, str):
        try:
            return dist.distributed_c10d._resolve_process_group(group)
        except Exception:
            return None
    return group


def _probe_symm_mem_ipc(symm_mem, group_name: str) -> bool:
    r"""One-time collective check that a symmetric workspace can be rendezvoused for
    *group_name* (i.e. CUDA-IPC / P2P is available). Every rank must call this together;
    :func:`_symm_mem_usable` caches the verdict so it happens at most once per group."""
    try:
        with torch.cuda.device(torch.cuda.current_device()):
            symm_mem.get_symm_mem_workspace(group_name, 1024)
        return True
    except Exception:  # pragma: no cover - probed only on real multi-GPU hardware
        return False


def _symm_mem_usable(group: object) -> bool:
    r"""Whether the symmetric-memory transport is auto-selectable for *group*.

    Selects symm-mem when NVSHMEM is available (a clean, non-collective probe -- the
    device-initiated multi-node path) or when a symmetric-workspace rendezvous succeeds
    for *group* (the intra-node CUDA-IPC path). The rendezvous is collective and cached
    per group; it is safe because the only distributed caller is the halo exchange
    itself, which every rank enters together. ``funcol`` (no requirement) always remains
    a working fallback, so a ``False`` here still lands on a correct path.
    """
    if not torch.cuda.is_available():
        return False
    try:
        import torch.distributed as _dist
        import torch.distributed._symmetric_memory as symm_mem
    except Exception:  # pragma: no cover - torch build without symm-mem
        return False
    if not (_dist.is_available() and _dist.is_initialized()):
        return False
    # symm-mem needs a CUDA/NCCL group; skip gloo/CPU so the collective probe below is
    # never issued on a transport that cannot serve it (which could hang, not raise).
    pg = _resolve_pg(group)
    if pg is None:
        return False
    try:
        if "nccl" not in str(_dist.get_backend(pg)).lower():
            return False
    except Exception:
        return False
    try:
        nvshmem = bool(symm_mem.is_nvshmem_available())
    except Exception:  # pragma: no cover
        nvshmem = False
    if nvshmem:
        return True
    group_name = _symm_group_name(group)
    cached = _symm_capability_cache.get(group_name)
    if cached is None:
        cached = _probe_symm_mem_ipc(symm_mem, group_name)
        _symm_capability_cache[group_name] = cached
    return cached


def select_halo_backend(group: object = None) -> _HaloBackend:
    r"""Return the halo transport backend for *group*.

    Honours ``PHYSICSNEMO_HALO_BACKEND`` (``"funcol"`` | ``"symm_mem"``); otherwise
    picks the symmetric-memory backend when usable and ``funcol`` (the fallback)
    otherwise.

    Parameters
    ----------
    group : ProcessGroup or DeviceMesh or str or None, optional, default=None
        Collective group used for the capability check.

    Returns
    -------
    _HaloBackend
        The selected transport backend.
    """
    forced = os.environ.get("PHYSICSNEMO_HALO_BACKEND")
    if forced == "funcol":
        return _FUNCOL_BACKEND
    if forced == "symm_mem":
        return _SYMM_MEM_BACKEND
    if forced:
        raise ValueError(
            f"PHYSICSNEMO_HALO_BACKEND={forced!r} is not a known halo backend "
            "(expected 'funcol' or 'symm_mem')."
        )
    if _symm_mem_usable(group):
        return _SYMM_MEM_BACKEND
    return _FUNCOL_BACKEND


def halo_reverse_exchange(
    padded: torch.Tensor,
    n_owned: int,
    send_indices: list[torch.Tensor],
    send_sizes: list[list[int]],
    rank: int,
    world_size: int,
    group: object = None,
) -> torch.Tensor:
    r"""Fold borrowed ghost rows back into their owners (transpose of the forward
    halo gather), using the transport backend selected for *group*.

    ``padded`` is ``[owned (n_owned) | ghost]`` with ghost rows grouped by source
    rank; each ghost row is summed back into its owning row.

    Parameters
    ----------
    padded : torch.Tensor
        ``(n_owned + n_ghost, *F)`` local tensor, ``[owned | ghost]``.
    n_owned : int
        Number of owned rows (length of the returned block).
    send_indices : list[torch.Tensor]
        ``send_indices[j]`` = owned-row indices this rank lent to rank ``j``.
    send_sizes : list[list[int]]
        ``send_sizes[i][j]`` = rows rank ``i`` lent to rank ``j``.
    rank : int
        This rank (sub-group-relative when *group* is a sub-group).
    world_size : int
        Group size (sub-group size when *group* is a sub-group).
    group : ProcessGroup or DeviceMesh or str or None, optional, default=None
        Collective group; ``None`` = default world group.

    Returns
    -------
    torch.Tensor
        ``(n_owned, *F)`` owned block with every borrowed contribution summed in.
    """
    return select_halo_backend(group).reverse(
        padded, n_owned, send_indices, send_sizes, rank, world_size, group
    )


def halo_forward_exchange(
    owned: torch.Tensor,
    send_indices: list[torch.Tensor],
    send_sizes: list[list[int]],
    rank: int,
    world_size: int,
    group: object = None,
) -> torch.Tensor:
    r"""Refresh ghost rows from the owners: gather each peer's lent rows and append
    them, returning the ``[owned | ghost]`` layout (inverse of
    :func:`halo_reverse_exchange`), using the backend selected for *group*.

    Parameters
    ----------
    owned : torch.Tensor
        ``(n_owned, *F)`` owned block.
    send_indices : list[torch.Tensor]
        ``send_indices[j]`` = owned-row indices this rank lent to rank ``j``.
    send_sizes : list[list[int]]
        ``send_sizes[i][j]`` = rows rank ``i`` lent to rank ``j``.
    rank : int
        This rank (sub-group-relative when *group* is a sub-group).
    world_size : int
        Group size (sub-group size when *group* is a sub-group).
    group : ProcessGroup or DeviceMesh or str or None, optional, default=None
        Collective group; ``None`` = default world group.

    Returns
    -------
    torch.Tensor
        ``(n_owned + n_ghost, *F)`` padded tensor with ghost rows refreshed.
    """
    return select_halo_backend(group).forward(
        owned, send_indices, send_sizes, rank, world_size, group
    )


def _scatter_correct_dense(
    padded: torch.Tensor,
    send_indices: list[torch.Tensor],
    send_sizes: list[list[int]],
    n_owned: int,
    rank: int,
    world_size: int,
    group: object,
) -> torch.Tensor:
    r"""``forward(reverse(padded))`` over *group* using a single selected backend."""
    backend = select_halo_backend(group)
    owned = backend.reverse(
        padded, n_owned, send_indices, send_sizes, rank, world_size, group
    )
    return backend.forward(owned, send_indices, send_sizes, rank, world_size, group)


def pack_halo_routing(
    send_indices: list[list[int]] | list[torch.Tensor],
    send_sizes: list[list[int]],
    n_owned: int,
    rank: int,
    world_size: int,
    device: object = None,
) -> torch.Tensor:
    r"""Pack halo routing into a 1-D int64 tensor for :func:`halo_scatter_correct`.

    The packed tensor is meant to ride as a graph input (e.g. a ShardTensor extra
    inner tensor), so its values may change across steps and survive Dynamo graph
    breaks without recompiling -- unlike routing baked in as ``int[]`` constants,
    which are guarded and force a recompile on any change.

    Parameters
    ----------
    send_indices : list[list[int]] or list[torch.Tensor]
        ``send_indices[j]`` = owned-row indices this rank lent to rank ``j``.
    send_sizes : list[list[int]]
        ``send_sizes[i][j]`` = rows rank ``i`` lent to rank ``j``.
    n_owned : int
        Number of owned rows.
    rank : int
        This rank (sub-group-relative when a sub-group is used).
    world_size : int
        Group size (sub-group size when a sub-group is used).
    device : torch.device or str or None, optional, default=None
        Device for the packed tensor.

    Returns
    -------
    torch.Tensor
        1-D int64 routing tensor consumed by :func:`halo_scatter_correct`, laid out
        as ``[world_size, n_owned, rank, n_flat, *send_sizes, *send_idx_lens,
        *send_idx_flat]``.

    Notes
    -----
    Index arrays are concatenated as tensors (their lengths come from shapes), so a
    device-resident ``send_indices`` is never moved to host -- the pack is free of a
    value-dependent device sync.
    """
    idx_tensors = [
        idx.reshape(-1).to(torch.int64)
        if isinstance(idx, torch.Tensor)
        else torch.tensor(idx, dtype=torch.int64)
        for idx in send_indices
    ]
    lens = [int(t.numel()) for t in idx_tensors]  # shapes only -- no value sync
    ss = [int(send_sizes[i][j]) for i in range(world_size) for j in range(world_size)]
    if device is None and idx_tensors:
        device = idx_tensors[0].device
    header = torch.tensor(
        [world_size, n_owned, rank, sum(lens), *ss, *lens],
        dtype=torch.int64,
        device=device,
    )
    if not idx_tensors:
        return header
    return torch.cat([header, torch.cat([t.to(device) for t in idx_tensors])])


def _unpack_halo_routing(routing: torch.Tensor):
    r"""Inverse of :func:`pack_halo_routing` (runs eagerly inside the op).

    Materializes only the small fixed header (``world_size``, ``n_owned``, ``rank``,
    and the ``world_size^2`` counts + ``world_size`` lengths) to host -- the counts
    are needed as ``int[]`` split sizes for the variable ``all_to_all``. The
    (potentially large) index arrays stay on-device as views, never round-tripped.
    """
    world_size, n_owned, rank, _n_flat = routing[:4].tolist()
    body_len = world_size * world_size + world_size
    body = routing[4 : 4 + body_len].tolist()
    ss, lens = body[: world_size * world_size], body[world_size * world_size :]
    send_sizes = [
        [ss[i * world_size + j] for j in range(world_size)] for i in range(world_size)
    ]
    send_indices, o = [], 4 + body_len
    for length in lens:
        send_indices.append(routing[o : o + length])  # device view -- no sync
        o += length
    return send_indices, send_sizes, n_owned, rank, world_size


@torch.library.custom_op("physicsnemo::halo_scatter_correct", mutates_args=())
def _halo_scatter_correct_op(
    padded: torch.Tensor, routing: torch.Tensor, group_name: str
) -> torch.Tensor:
    r"""Dispatcher-visible ``forward(reverse(padded))``, opaque to fake mode.

    ``routing`` is a 1-D int64 tensor (a graph input, not baked constants), so its
    values may change across steps and graph breaks without recompiling. The op body
    runs only at runtime on real tensors -- unpacking ``routing`` there -- while the
    trace sees only :func:`_halo_scatter_correct_fake`. Runs over the group named
    ``group_name`` (``""`` = default world group).
    """
    group = group_name or None
    send_indices, send_sizes, n_owned, rank, world_size = _unpack_halo_routing(routing)
    return _scatter_correct_dense(
        padded, send_indices, send_sizes, n_owned, rank, world_size, group
    )


@_halo_scatter_correct_op.register_fake
def _halo_scatter_correct_fake(padded, routing, group_name):
    return torch.empty_like(padded)


def _halo_correct_setup_context(ctx, inputs, output):
    _padded, routing, group_name = inputs
    ctx.routing = routing
    ctx.group_name = group_name


def _halo_correct_backward(ctx, grad):
    # forward(reverse(.)) is self-adjoint, so the VJP is the op applied to grad.
    grad_in = _halo_scatter_correct_op(grad.contiguous(), ctx.routing, ctx.group_name)
    return grad_in, None, None


_halo_scatter_correct_op.register_autograd(
    _halo_correct_backward, setup_context=_halo_correct_setup_context
)


def halo_scatter_correct(
    padded: torch.Tensor,
    routing: torch.Tensor,
    group: object = None,
) -> torch.Tensor:
    r"""``torch.compile``-safe halo scatter-correction on a ``[owned | ghost]``
    tensor.

    Folds borrowed-ghost contributions back into their owners and refreshes the
    ghost rows (``forward(reverse(padded))``) as a single AOT-traceable, inductor-
    lowerable, differentiable graph node. ``routing`` (from :func:`pack_halo_routing`)
    rides as a graph-input tensor, so it survives Dynamo graph breaks and per-step
    value changes without recompiling.

    Parameters
    ----------
    padded : torch.Tensor
        ``(n_owned + n_ghost, *F)`` local tensor, ``[owned | ghost]``.
    routing : torch.Tensor
        1-D int64 routing tensor from :func:`pack_halo_routing`.
    group : ProcessGroup or DeviceMesh or str or None, optional, default=None
        Collective group; a neighbour sub-group bounds the coordination span.

    Returns
    -------
    torch.Tensor
        ``(n_owned + n_ghost, *F)`` corrected padded tensor.
    """
    return _halo_scatter_correct_op(padded, routing, _halo_group_name(group))


# ======================================================================
# ShardTensor scatter/index-add integration.
#
# A ShardTensor whose local tensor is a ``[owned | ghost]`` halo and which carries
# the packed routing as an extra inner tensor named ``_halo_meta_packed`` (declared
# via ``_extra_inner_tensors``) gets its ``scatter_add`` / ``index_add`` corrected
# automatically. Correction runs as a ``__torch_function__`` handler (invoked for
# both eager and ``torch.compile`` tracing) that scatters on the plain local, emits
# :func:`halo_scatter_correct`, and re-wraps. It must run there rather than in
# ``__torch_dispatch__``: a traceable wrapper subclass connects its autograd only
# for the intercepted op itself, so a halo correction applied to inner tensors at
# the dispatch level is dropped from the compiled backward. The routing rides as a
# graph-input inner tensor (survives graph breaks / value changes without
# recompiling). Tensors without routing fall through unchanged -- registering is a
# safe opt-in.
# ======================================================================


def register_halo_scatter_handlers() -> None:
    r"""Register ``scatter_add`` / ``index_add`` halo-correction handlers on
    ``ShardTensor`` (idempotent, opt-in).

    Handlers apply :func:`halo_scatter_correct` only to tensors carrying a non-empty
    ``_halo_meta_packed`` routing inner tensor; all other ShardTensors fall through
    to the default behavior, so registering does not change base behavior.
    """
    from physicsnemo.domain_parallel.shard_tensor import (
        ShardTensor,
        _torch_function_fallback_via_dtensor,
    )

    def _local(x):
        return x._local_tensor if isinstance(x, ShardTensor) else x

    def _routing(self):
        r = getattr(self, "_halo_meta_packed", None)
        return r if (r is not None and r.numel() > 0) else None

    def _needs_grad(*tensors):
        return torch.is_grad_enabled() and any(
            bool(getattr(t, "requires_grad", False))
            or getattr(t, "grad_fn", None) is not None
            for t in tensors
        )

    def _build(src_type, local, spec, routing, requires_grad):
        out = src_type.__new__(
            src_type, local_tensor=local, spec=spec, requires_grad=requires_grad
        )
        out._halo_meta_packed = routing
        return out

    class _WrapLocalAsShard(torch.autograd.Function):
        # Attach a grad_fn to the op-result wrapper so the tangent flows
        # wrapper -> local -> the halo/scatter graph. A wrapper built by a bare
        # ``__new__`` is an autograd leaf, so the halo correction's backward is
        # dropped. Mirrors ``_FromTorchTensor``.
        @staticmethod
        def forward(ctx, local, src_type, spec, routing):
            return _build(src_type, local, spec, routing, local.requires_grad)

        @staticmethod
        def backward(ctx, grad_out):
            g = (
                grad_out._local_tensor
                if isinstance(grad_out, ShardTensor)
                else grad_out
            )
            return g, None, None, None

    def _wrap_like(src, local, routing, requires_grad):
        if requires_grad:
            return _WrapLocalAsShard.apply(local, type(src), src._spec, routing)
        return _build(type(src), local, src._spec, routing, False)

    def _scatter_handler(f, types, args, kwargs):
        self = args[0]
        routing = _routing(self)
        if routing is None:
            return _torch_function_fallback_via_dtensor(f, args, kwargs)
        dim, index, src = args[1], args[2], args[3]
        local_result = f(_local(self), dim, _local(index), _local(src))
        corrected = halo_scatter_correct(local_result, routing, group=self._spec.mesh)
        return _wrap_like(self, corrected, routing, _needs_grad(self, index, src))

    for func in (torch.Tensor.scatter_add, torch.Tensor.index_add):
        ShardTensor.register_function_handler(func, _scatter_handler)
