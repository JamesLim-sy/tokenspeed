# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Distributed sparse selection over sharded Index-K storage."""

from dataclasses import dataclass

import torch
from tokenspeed_kernel.ops.attention.dsa import DSAIndexMetadata, dsa_index_candidates

from tokenspeed.runtime.distributed.comm_ops import all_gather
from tokenspeed.runtime.layers.attention.dcp.placement import (
    CachePlacement,
    resolve_cache_slots,
)


def _same_view(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.shape == right.shape
        and left.stride() == right.stride()
        and left.dtype == right.dtype
        and left.device == right.device
        and left.data_ptr() == right.data_ptr()
    )


def _query_tile_rows(columns: int, page_size: int, max_logits_bytes: int) -> int:
    """Size query tiles by logical FP32 logits, allowing at least one row.

    This is a tiling target, not a peak-memory cap: kernel row alignment,
    selection scratch, candidate exchange and retained metadata are additional.
    """
    if columns <= 0 or page_size <= 0:
        raise ValueError("Index tiling requires positive page geometry")
    if max_logits_bytes <= 0:
        raise ValueError("Index tiling requires a positive logits budget")
    row_bytes = columns * page_size * torch.float32.itemsize
    return max(1, max_logits_bytes // row_bytes)


@dataclass(frozen=True)
class DCPIndexPlan:
    """Layer-invariant query and page mappings for one decode forward."""

    page_table: torch.Tensor
    seq_lens: torch.Tensor
    placement: CachePlacement
    page_size: int
    q_len_per_req: int
    max_logits_bytes: int
    tile_rows: int
    local_page_table: torch.Tensor
    query_requests: torch.Tensor
    causal_lens: torch.Tensor
    tiles: tuple[DSAIndexMetadata, ...]


@dataclass
class DCPIndexerMetadata:
    """Per-forward memoization owned by one cache group's DSA backend.

    Replace this object on every metadata refresh. Input values must remain
    unchanged across layers of that forward, even when their storage is reused
    on the next step. Preparation is recorded once during graph capture and
    executes again on replay from the refreshed input buffers.
    """

    plan: DCPIndexPlan | None = None

    def prepare(
        self,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        *,
        placement: CachePlacement,
        page_size: int,
        q_len_per_req: int,
        max_logits_bytes: int,
    ) -> DCPIndexPlan:
        """Prepare query rows, ownership translation and per-tile cache holders."""
        if self.plan is not None:
            plan = self.plan
            if (
                not _same_view(plan.page_table, page_table)
                or not _same_view(plan.seq_lens, seq_lens)
                or plan.placement != placement
                or plan.page_size != page_size
                or plan.q_len_per_req != q_len_per_req
                or plan.max_logits_bytes != max_logits_bytes
            ):
                raise ValueError("DCP index inputs changed without a metadata refresh")
            return plan
        if (
            page_table.ndim != 2
            or seq_lens.ndim != 1
            or page_table.shape[0] != seq_lens.numel()
            or page_table.shape[1] == 0
            or page_size <= 0
            or q_len_per_req <= 0
        ):
            raise ValueError("Invalid DCP decode index metadata geometry")
        rows = seq_lens.numel() * q_len_per_req
        tile_rows = _query_tile_rows(page_table.shape[1], page_size, max_logits_bytes)
        row_ids = torch.arange(rows, device=seq_lens.device, dtype=torch.int32)
        requests = row_ids // q_len_per_req
        causal_lens = (
            seq_lens[requests.long()] - (q_len_per_req - 1) + row_ids % q_len_per_req
        ).contiguous()
        slots, owned = resolve_cache_slots(page_table.long() * page_size, placement)
        local_table = torch.where(owned, slots // page_size, -1).to(torch.int32)
        tiles = tuple(
            DSAIndexMetadata(
                local_table,
                requests[start : start + tile_rows],
                causal_lens[start : start + tile_rows],
                page_size,
            )
            for start in range(0, rows, tile_rows)
        )
        self.plan = DCPIndexPlan(
            page_table=page_table,
            seq_lens=seq_lens,
            placement=placement,
            page_size=page_size,
            q_len_per_req=q_len_per_req,
            max_logits_bytes=max_logits_bytes,
            tile_rows=tile_rows,
            local_page_table=local_table,
            query_requests=requests,
            causal_lens=causal_lens,
            tiles=tiles,
        )
        return self.plan


def merge_index_candidates(
    offsets: torch.Tensor,
    scores: torch.Tensor,
    *,
    topk: int,
    group: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge disjoint local candidates with one global Top-K selection.

    Communicates only K candidate positions and scores per query and rank.
    Int32 positions and FP32 scores share one all-gather as interleaved int32
    records; scores are reinterpreted bitwise, including mandatory +inf.
    Returns global logical positions and valid counts, identical on every rank.
    Positive infinity marks mandatory candidates; negative infinity is padding.
    Equal-score candidates have no prescribed logical-position order.
    """
    if offsets.shape != scores.shape or offsets.ndim != 2 or topk <= 0:
        raise ValueError("Candidate offsets/scores must be matching matrices")
    if len(group) > 1:
        if offsets.dtype == torch.int32 and scores.dtype == torch.float32:
            # Numeric conversion would lose large positions or score bits.
            # Stack also packs strided producer views into contiguous records.
            packed = torch.stack((offsets, scores.view(torch.int32)), dim=-1)
            gathered = all_gather(packed.flatten(1), group, dim=-1)
            candidates = gathered.reshape(
                offsets.shape[0], offsets.shape[1] * len(group), packed.shape[-1]
            )
            offsets, score_bits = candidates.unbind(dim=-1)
            scores = score_bits.view(torch.float32)
        else:
            # Keep the existing input domain for callers using other dtypes.
            offsets = all_gather(offsets.contiguous(), group, dim=-1)
            scores = all_gather(scores.contiguous(), group, dim=-1)
    valid = (offsets >= 0) & ~torch.isnan(scores) & (scores > -float("inf"))
    scores = torch.where(valid, scores, -float("inf"))
    selected_scores, order = torch.topk(scores, topk, dim=-1)
    result = offsets.gather(1, order)
    valid = selected_scores > -float("inf")
    result = torch.where(valid, result, -1)
    return result, valid.sum(dim=-1, dtype=torch.int32)


def select_dsa_topk(
    q: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_requests: torch.Tensor,
    causal_lens: torch.Tensor,
    *,
    placement: CachePlacement,
    page_size: int,
    topk: int,
    softmax_scale: float,
    initial_tokens: int,
    local_tokens: int,
    max_logits_bytes: int,
    metadata: DCPIndexPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return request-relative global Top-K from local Index-K pages.

    Metadata has global positions throughout. Only cache addresses are local;
    global Top-K precedes the attention-side ownership mask and LSE reduction.
    ``max_logits_bytes`` is a positive target for logical FP32 logits per
    query tile, with at least one row per tile. Kernel padding and other
    temporary allocations can exceed this target; it is not a peak-memory cap.
    ``metadata`` optionally supplies the decode forward's prepared mappings;
    the input views, ownership and tiling must match the prepared plan.
    """
    if max_logits_bytes <= 0:
        raise ValueError("Index tiling requires a positive logits budget")
    if q.shape[0] == 0:
        return (
            torch.empty((0, topk), device=q.device, dtype=torch.int32),
            torch.empty(0, device=q.device, dtype=torch.int32),
        )
    if metadata is None:
        tile = _query_tile_rows(page_table.shape[1], page_size, max_logits_bytes)
        slots, owned = resolve_cache_slots(page_table.long() * page_size, placement)
        local_table = torch.where(owned, slots // page_size, -1).to(torch.int32)
    else:
        if (
            not _same_view(metadata.page_table, page_table)
            or not _same_view(metadata.query_requests, query_requests)
            or not _same_view(metadata.causal_lens, causal_lens)
            or q.shape[0] != metadata.query_requests.numel()
            or metadata.placement != placement
            or metadata.page_size != page_size
            or metadata.max_logits_bytes != max_logits_bytes
        ):
            raise ValueError("DCP selection inputs do not match the prepared plan")
        local_table = metadata.local_page_table
        tile = metadata.tile_rows
    outputs, lengths = [], []
    for start in range(0, q.shape[0], tile):
        end = min(q.shape[0], start + tile)
        offsets, scores = dsa_index_candidates(
            q[start:end],
            weights[start:end],
            index_k_cache,
            local_table,
            query_requests[start:end],
            causal_lens[start:end],
            page_size=page_size,
            topk=topk,
            softmax_scale=softmax_scale,
            initial_tokens=initial_tokens,
            local_tokens=local_tokens,
            solution=None,
            metadata=None if metadata is None else metadata.tiles[start // tile],
        )
        indices, lens = merge_index_candidates(
            offsets, scores, topk=topk, group=placement.group
        )
        outputs.append(indices)
        lengths.append(lens)
    return torch.cat(outputs), torch.cat(lengths)
