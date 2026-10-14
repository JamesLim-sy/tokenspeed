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

import os
import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tokenspeed.runtime.layers.attention.dcp import indexer
from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.dcp.metadata import (
    CompactDCPLayout,
    CompactDCPMetadata,
    DCPPageTableMetadata,
    PositionPreservingDCPLayout,
    PositionPreservingDCPMetadata,
    refresh_dcp_page_table_metadata,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("block_size", [64, 128])
def test_layouts_share_placement_and_preserve_partial_tail(device, degree, block_size):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Nonconsecutive ownership IDs; second row includes null/out-of-capacity IDs.
    blocks = torch.tensor([[2, 5, 8], [0, 10, 3]], dtype=torch.int32, device=device)
    lengths = torch.tensor([2 * block_size + 17, 0], dtype=torch.int32, device=device)
    subpages = block_size // 64
    pages = (
        (blocks[..., None] * subpages + torch.arange(subpages, device=device))
        .reshape(2, -1)
        .int()
    )
    token_totals = torch.zeros_like(lengths)
    for rank in range(degree):
        common = dict(virtual_block_count=10, degree=degree, rank=rank, previous=None)
        preserved = refresh_dcp_page_table_metadata(
            page_table=blocks, layout=PositionPreservingDCPLayout(), **common
        )
        compact = refresh_dcp_page_table_metadata(
            page_table=pages,
            layout=CompactDCPLayout(lengths, 64, block_size),
            **common,
        )
        assert isinstance(preserved, PositionPreservingDCPMetadata)
        assert isinstance(compact, CompactDCPMetadata)
        assert isinstance(compact, DCPPageTableMetadata)
        expected_pages = []
        expected_tokens = 0
        for col, block in enumerate([2, 5, 8]):
            owned = (block - 1) % degree == rank
            local_block = (block - 1) // degree + 1
            assert preserved.local_page_table[0, col].item() == (
                local_block if owned else -1
            )
            if owned:
                count = min(block_size, int(lengths[0]) - col * block_size)
                expected_tokens += count
                expected_pages.extend(
                    local_block * subpages + i for i in range((count + 63) // 64)
                )
        assert (
            compact.local_page_table[0, : len(expected_pages)].tolist()
            == expected_pages
        )
        assert not compact.local_page_table[0, len(expected_pages) :].any()
        assert compact.local_seq_lens.tolist() == [expected_tokens, 0]
        assert not compact.local_page_table[1].any()
        assert preserved.local_page_table[1, :2].tolist() == [-1, -1]
        token_totals += compact.local_seq_lens
    torch.testing.assert_close(token_totals, lengths)


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_refresh_row_views_and_graph_replay(compact, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    table = torch.tensor(
        [[2, 3, 4], [1, 2, 3], [2, 3, 4]], dtype=torch.int32, device=device
    )
    lengths = torch.tensor([145, 129, 128], dtype=torch.int32, device=device)

    def layout(n):
        return (
            CompactDCPLayout(lengths[:n], 64, 64)
            if compact
            else PositionPreservingDCPLayout()
        )

    common = dict(virtual_block_count=9, degree=2, rank=1)
    metadata = refresh_dcp_page_table_metadata(
        page_table=table, layout=layout(3), previous=None, **common
    )
    pointers = (
        metadata.local_page_table.data_ptr(),
        (metadata.local_seq_lens if compact else metadata.owner_mask).data_ptr(),
    )

    def refresh(n):
        updated = refresh_dcp_page_table_metadata(
            page_table=table[:n],
            layout=layout(n),
            previous=metadata.slice_requests(0, n),
            **common,
        )
        assert updated.local_page_table.data_ptr() == pointers[0]
        assert (
            updated.local_seq_lens if compact else updated.owner_mask
        ).data_ptr() == pointers[1]

    refresh(1)
    refresh(3)
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            refresh(1)
    table[0].copy_(torch.tensor([3, 2, 4], device=device))
    lengths[0] = 65
    if device == "cuda":
        graph.replay()
    else:
        refresh(1)
    if compact:
        assert metadata.local_page_table[0].tolist() == [1, 0, 0]
        assert metadata.local_seq_lens[0].item() == 1
    else:
        assert metadata.local_page_table[0].tolist() == [-1, 1, 2]
        assert metadata.owner_mask[0].tolist() == [False, True, True]


def test_refresh_rejects_layout_topology_and_page_geometry_changes():
    table = torch.tensor([[2, 3]], dtype=torch.int32)
    lengths = torch.tensor([65], dtype=torch.int32)
    layout = CompactDCPLayout(lengths, 64, 128)
    kwargs = dict(
        page_table=table, virtual_block_count=10, degree=2, rank=0, layout=layout
    )
    metadata = refresh_dcp_page_table_metadata(previous=None, **kwargs)
    for changes, message in [
        ({"layout": PositionPreservingDCPLayout()}, "layout changed"),
        ({"rank": 1}, "topology changed"),
        ({"virtual_block_count": 12}, "capacity changed"),
        ({"layout": replace(layout, block_granularity=64)}, "geometry changed"),
        ({"layout": replace(layout, page_size=0)}, "whole kernel pages"),
        ({"layout": replace(layout, seq_lens=torch.ones(2))}, "lengths must match"),
        (
            # A captured graph would keep replaying against the old buffers.
            {"page_table": torch.tensor([[2, 3, 4]], dtype=torch.int32)},
            "buffers do not match",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            refresh_dcp_page_table_metadata(previous=metadata, **(kwargs | changes))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("block_size", [64, 128])
def test_query_visibility_reserve_pages_advance_and_reanchor(
    device, degree, block_size
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Ownership follows physical virtual IDs, not logical page positions.
    blocks = [[5, 2, 8, 1], [2, 5, 1, 8], [0, 0, 0, 0]]
    subpages = block_size // 64
    table = torch.tensor(
        [[b * subpages + s for b in row for s in range(subpages)] for row in blocks],
        dtype=torch.int32,
        device=device,
    )
    visible = torch.tensor([[62, 63, 64, 65]] * 2 + [[0] * 4], device=device).int()
    lengths = visible[:, -1].contiguous()
    for rank in range(degree):
        common = dict(virtual_block_count=10, degree=degree, rank=rank)
        layout = CompactDCPLayout(lengths, 64, block_size, visible)
        metadata = refresh_dcp_page_table_metadata(
            page_table=table, layout=layout, previous=None, **common
        )
        buffers = (
            metadata.local_page_table,
            metadata.page_prefix,
            metadata.local_visible_lens,
            metadata.local_seq_lens,
        )
        pointers = [t.data_ptr() for t in buffers]
        compact_before = metadata.local_page_table.clone()

        def check(endpoints):
            expected = []
            for row, row_blocks in enumerate(blocks):
                # Enumerate visible global tokens independently of prefix arithmetic.
                owned = [
                    (b > 0 and (b - 1) % degree == rank)
                    for b in row_blocks
                    for _ in range(block_size)
                ]
                expected.append([sum(owned[:end]) for end in endpoints[row]])
                pages = [
                    ((b - 1) // degree + 1) * subpages + s
                    for b in row_blocks
                    if b > 0 and (b - 1) % degree == rank
                    for s in range(subpages)
                ]
                assert metadata.local_page_table[row].tolist() == pages + [0] * (
                    table.shape[1] - len(pages)
                )
            assert metadata.local_visible_lens.tolist() == expected
            assert metadata.local_seq_lens.tolist() == [row[-1] for row in expected]
            assert [t.data_ptr() for t in buffers] == pointers

        metadata.refresh_visible_lengths(visible)
        if device == "cuda":
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                metadata.refresh_visible_lengths(visible)
        # Advance across kernel and ownership pages; rollback; noncausal draft;
        # exact full-table endpoint; idle; then restore a live sequence.
        for endpoints in (
            [62, 63, 64, 65],
            [126, 127, 128, 129],
            [190, 191, 192, 193],
            [61, 62, 63, 64],
            [129] * 4,
            list(range(4 * block_size - 3, 4 * block_size + 1)),
            [0] * 4,
            [62, 63, 64, 65],
        ):
            expected_endpoints = [endpoints, endpoints, [0] * 4]
            visible.copy_(torch.tensor(expected_endpoints, device=device))
            metadata.local_visible_lens.fill_(-1)
            metadata.local_seq_lens.fill_(-1)
            if device == "cuda":
                graph.replay()
            else:
                metadata.refresh_visible_lengths(visible)
            check(expected_endpoints)
            torch.testing.assert_close(metadata.local_page_table, compact_before)

        # A row view must update the parent's buffers, including the query bounds.
        subset = metadata.slice_requests(1, 2)
        visible[1].zero_()
        subset.refresh_visible_lengths(visible[1:2])
        assert metadata.local_visible_lens[1].tolist() == [0] * 4
        assert metadata.local_seq_lens[1].item() == 0


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("queries", [1, 3])
def test_query_visibility_full_refresh_reuses_storage(device, queries):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    table = torch.tensor([[5, 2, 8, 1]], dtype=torch.int32, device=device)
    visible = torch.tensor(
        [list(range(66 - queries, 66))], dtype=torch.int32, device=device
    )
    lengths = torch.tensor([65], dtype=torch.int32, device=device)
    layout = CompactDCPLayout(lengths, 64, 64, visible)
    common = dict(
        page_table=table, layout=layout, virtual_block_count=10, degree=2, rank=0
    )
    metadata = refresh_dcp_page_table_metadata(previous=None, **common)

    def refresh():
        updated = refresh_dcp_page_table_metadata(previous=metadata, **common)
        assert updated.page_prefix.data_ptr() == metadata.page_prefix.data_ptr()
        assert (
            updated.local_visible_lens.data_ptr()
            == metadata.local_visible_lens.data_ptr()
        )

    refresh()
    if device == "cuda":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            refresh()
    table.copy_(torch.tensor([[2, 5, 1, 8]], device=device))
    if device == "cuda":
        graph.replay()
    else:
        refresh()
    assert metadata.local_page_table.tolist() == [[3, 1, 0, 0]]
    assert metadata.page_prefix.tolist() == [[0, 0, 1, 2, 2]]
    assert metadata.local_visible_lens.tolist() == [[0] * (queries - 1) + [1]]
    assert metadata.local_seq_lens.tolist() == [1]
    with pytest.raises(ValueError, match="visibility buffers changed"):
        refresh_dcp_page_table_metadata(
            previous=metadata,
            **(common | {"layout": replace(layout, visible_lens=None)}),
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("block_size", [64, 128])
def test_verify_query_lengths_cross_ownership_boundaries(device, block_size):
    """A verify window's per-query endpoints, fed through the layout's
    ``visible_lens``, map to the owned-token counts a brute-force scan of the
    cyclic placement gives, across kernel-page and ownership-block edges."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    blocks = torch.tensor([[2, 5, 8], [0, 0, 0]], dtype=torch.int32, device=device)
    subpages = block_size // 64
    pages = (
        blocks[..., None] * subpages + torch.arange(subpages, device=device)
    ).reshape(2, -1)
    offsets = torch.arange(-5, 1, dtype=torch.int32, device=device)
    for length in (1, block_size - 1, block_size + 2, 2 * block_size + 1):
        lens = torch.tensor([length, 0], dtype=torch.int32, device=device)
        causal = (lens[:, None] + offsets).clamp_min(0)
        noncausal = lens[:, None].expand(-1, 6).contiguous()
        for rank in range(4):
            expected = [
                sum(
                    (int(blocks[0, t // block_size]) - 1) % 4 == rank
                    for t in range(max(0, length - 5 + i))
                )
                for i in range(6)
            ]
            for visible, rows in ((causal, expected), (noncausal, [expected[-1]] * 6)):
                metadata = refresh_dcp_page_table_metadata(
                    page_table=pages,
                    virtual_block_count=10,
                    degree=4,
                    rank=rank,
                    layout=CompactDCPLayout(lens, 64, block_size, visible),
                    previous=None,
                )
                assert metadata.local_visible_lens.tolist() == [rows, [0] * 6]
                assert metadata.local_seq_lens.tolist() == [expected[-1], 0]


# Index planning shares page mappings across layers and resets on each refresh.


def _metadata_backend(
    monkeypatch, *, degree, spec_num_tokens, index_topk, index_kpool, dense_backend
):
    from tokenspeed.runtime.layers.attention.backends.paged import dsa
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.dsa import DSAConfig

    spec = DSAConfig(
        num_attention_heads=16,
        num_kv_heads=1,
        head_dim=192,
        attn_tp_size=degree,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        scaling=192**-0.5,
        kv_cache_dim=576,
        index_topk=index_topk,
        index_head_dim=128,
        index_n_heads=16,
        index_k_format="fp8_scaled",
        index_kpool=index_kpool,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=64,
        context_len=128,
        max_bs=2,
        speculative_num_draft_tokens=spec_num_tokens,
        kvp_size=degree,
        kvp_group=tuple(range(degree)),
        components=(spec,),
    )
    monkeypatch.setitem(dsa.global_server_args_dict, "numerics", "auto")
    monkeypatch.setitem(dsa.global_server_args_dict, "dsa_slot_order", "selection")
    monkeypatch.setattr(dsa, "_make_dense_leaf", lambda *args: dense_backend)
    # Keep the real constructor so tests use the backend's declared group fields.
    return dsa.DSABackend(config, spec, kernel_page_size=64)


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("topk", [0, 3, 512])
def test_backend_validates_candidate_count_at_initialization(monkeypatch, degree, topk):
    def create_backend():
        return _metadata_backend(
            monkeypatch,
            degree=degree,
            spec_num_tokens=1,
            index_topk=topk,
            index_kpool=None,
            dense_backend=SimpleNamespace(),
        )

    if degree > 1 and topk in (0, 3):
        with pytest.raises(ValueError, match="positive power of two"):
            create_backend()
    else:
        # Replicated storage retains non-power-of-two and disabled-index choices.
        assert create_backend().index_topk == topk


@pytest.mark.parametrize("use_metadata", [False, True])
@pytest.mark.parametrize(
    "budget,tile_sizes",
    [(1, [1, 1, 1]), (511, [1, 1, 1]), (1024, [2, 1]), (1536, [3])],
)
def test_query_tiling_budget_matches_planning_and_selection(
    budget, tile_sizes, use_metadata, monkeypatch
):
    table = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
    lengths = torch.full((3,), 128, dtype=torch.int32)
    placement = CachePlacement(64, 7, (0,), 0)
    plan = (
        indexer.DCPIndexerMetadata().prepare(
            table,
            lengths,
            placement=placement,
            page_size=64,
            q_len_per_req=1,
            max_logits_bytes=budget,
        )
        if use_metadata
        else None
    )
    requests = (
        torch.arange(3, dtype=torch.int32) if plan is None else plan.query_requests
    )
    causal = lengths if plan is None else plan.causal_lens
    seen = []

    def candidates(q, weights, cache, local_table, reqs, ends, **kwargs):
        seen.append(reqs.tolist())
        if plan is not None:
            assert kwargs["metadata"] is plan.tiles[len(seen) - 1]
        return (
            torch.tensor([[0, 1]], dtype=torch.int32).expand(q.shape[0], -1),
            torch.tensor([[2.0, 1.0]]).expand(q.shape[0], -1),
        )

    monkeypatch.setattr(indexer, "dsa_index_candidates", candidates)
    positions, counts = indexer.select_dsa_topk(
        torch.zeros(3, 16, 128),
        torch.zeros(3, 16),
        torch.empty(7 * 64, 132, dtype=torch.uint8),
        table,
        requests,
        causal,
        placement=placement,
        page_size=64,
        topk=2,
        softmax_scale=0.1,
        initial_tokens=0,
        local_tokens=0,
        max_logits_bytes=budget,
        metadata=plan,
    )
    assert [len(rows) for rows in seen] == tile_sizes
    assert [request for rows in seen for request in rows] == [0, 1, 2]
    assert positions.tolist() == [[0, 1]] * 3
    assert counts.tolist() == [2] * 3


@pytest.mark.parametrize("rows", [0, 1])
@pytest.mark.parametrize("budget", [0, -1])
def test_nonpositive_logits_budget_is_rejected(budget, rows):
    table = torch.tensor([[1]], dtype=torch.int32)
    lengths = torch.tensor([1], dtype=torch.int32)
    placement = CachePlacement(64, 2, (0,), 0)
    with pytest.raises(ValueError, match="positive.*logits budget"):
        indexer.DCPIndexerMetadata().prepare(
            table,
            lengths,
            placement=placement,
            page_size=64,
            q_len_per_req=1,
            max_logits_bytes=budget,
        )
    with pytest.raises(ValueError, match="positive.*logits budget"):
        indexer.select_dsa_topk(
            torch.empty(rows, 16, 128),
            torch.empty(rows, 16),
            torch.empty(0, 132, dtype=torch.uint8),
            table,
            torch.zeros(rows, dtype=torch.int32),
            torch.ones(rows, dtype=torch.int32),
            placement=placement,
            page_size=64,
            topk=2,
            softmax_scale=0.1,
            initial_tokens=0,
            local_tokens=0,
            max_logits_bytes=budget,
            metadata=None,
        )


@pytest.mark.parametrize("degree", [2, 4])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("query_width", [1, 3])
def test_decode_plan_reuses_translation_and_preserves_geometry(
    degree, block_size, query_width
):
    page_size = 64
    subpages = block_size // page_size
    blocks = torch.tensor([[3, 2, 5], [0, 10, 1]], dtype=torch.int32)
    table = (blocks[..., None] * subpages + torch.arange(subpages)).reshape(2, -1)
    seq_lens = torch.tensor([137, 3], dtype=torch.int32)
    kwargs = dict(
        placement=CachePlacement(block_size, 10, tuple(range(degree)), 0),
        page_size=page_size,
        q_len_per_req=query_width,
        max_logits_bytes=2 * table.shape[1] * page_size * torch.float32.itemsize,
    )
    metadata = indexer.DCPIndexerMetadata()
    with mock.patch.object(
        indexer, "resolve_cache_slots", wraps=indexer.resolve_cache_slots
    ) as translate:
        plan = metadata.prepare(table, seq_lens, **kwargs)
        assert metadata.prepare(table[:], seq_lens[:], **kwargs) is plan
        assert translate.call_count == 1
    expected = []
    for row in table.tolist():
        expected.append(
            [
                (
                    ((page // subpages - 1) // degree + 1) * subpages + page % subpages
                    if 0 < page // subpages < 10
                    and (page // subpages - 1) % degree == 0
                    else -1
                )
                for page in row
            ]
        )
    assert plan.local_page_table.tolist() == expected
    assert plan.query_requests.tolist() == [0] * query_width + [1] * query_width
    assert plan.causal_lens.tolist() == [
        length - query_width + 1 + token
        for length in seq_lens.tolist()
        for token in range(query_width)
    ]
    assert [tile.query_requests.numel() for tile in plan.tiles] == [2] * query_width
    assert all(tile.local_page_table is plan.local_page_table for tile in plan.tiles)
    # An independent cache group must not inherit another group's plan.
    assert indexer.DCPIndexerMetadata().prepare(table, seq_lens, **kwargs) is not plan
    with pytest.raises(ValueError, match="without a metadata refresh"):
        metadata.prepare(table.clone(), seq_lens, **kwargs)
    # Slot reuse changes values in the same persistent buffers on the next step.
    table.fill_(2 * subpages)
    seq_lens.fill_(7)
    refreshed = indexer.DCPIndexerMetadata().prepare(table, seq_lens, **kwargs)
    assert (refreshed.local_page_table == -1).all()
    assert refreshed.causal_lens[-1] == 7


@pytest.mark.parametrize("degree,index_kpool", [(1, None), (1, 128), (2, None)])
@pytest.mark.parametrize("spec_num_tokens", [1, 4])
def test_backend_refresh_and_draft_updates_invalidate_index_plan(
    degree, index_kpool, spec_num_tokens, monkeypatch
):
    from tokenspeed.runtime.layers.attention.backends.paged import dsa
    from tokenspeed.runtime.layers.attention.backends.paged.trtllm_mla import (
        TRTLLMMLADecodeMetadata,
    )

    metadata = TRTLLMMLADecodeMetadata(seq_lens_k=torch.tensor([4], dtype=torch.int32))
    backend = _metadata_backend(
        monkeypatch,
        degree=degree,
        spec_num_tokens=spec_num_tokens,
        index_topk=2048,
        index_kpool=index_kpool,
        dense_backend=SimpleNamespace(
            forward_decode_metadata=metadata,
            refresh_decode_metadata=lambda *args, **kwargs: None,
            child_backends=lambda: (),
        ),
    )
    monkeypatch.setattr(dsa, "dsa_plan", lambda **kwargs: None)
    table = torch.tensor([[1]], dtype=torch.int32)
    lengths = torch.tensor([4], dtype=torch.int32)
    from tokenspeed.runtime.execution.graph_ptr_guard import (
        snapshot_graph_metadata,
        verify_graph_metadata,
    )

    snapshot = None
    previous = None
    for update in (
        lambda: backend.init_forward_metadata_capture_cuda_graph(1, lengths, table),
        lambda: backend.refresh_decode_metadata(1, 1, lengths, table),
        lambda: backend.refresh_decode_metadata(
            1, 0, lengths, table, for_graph_replay=True
        ),
        lambda: backend.advance_draft_forward_metadata(lengths),
        lambda: backend.update_draft_forward_metadata(lengths),
    ):
        update()
        current = backend._dcp_indexer
        if degree == 1:
            assert current is None
        else:
            assert current is not previous
            assert current.plan is None
            current.prepare(
                table,
                lengths,
                placement=CachePlacement(64, 4, (0, 1), 0),
                page_size=64,
                q_len_per_req=1,
                max_logits_bytes=1024,
            )
        previous = current
        if snapshot is None:
            snapshot = snapshot_graph_metadata(backend)
        else:
            verify_graph_metadata(backend, snapshot, context="DCP metadata refresh")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_decode_plan_graph_refreshes_ownership_and_lengths():
    page_size = 64
    table = torch.tensor([[1, 2, 3], [2, 4, 6]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([145, 129], dtype=torch.int32, device="cuda")
    kwargs = dict(
        placement=CachePlacement(page_size, 9, (0, 1), 0),
        page_size=page_size,
        q_len_per_req=2,
        max_logits_bytes=table.shape[1] * page_size * torch.float32.itemsize,
    )

    def forward():
        # Each runner refresh replaces the holder before the first layer.
        metadata = indexer.DCPIndexerMetadata()
        plan = metadata.prepare(table, lengths, **kwargs)
        assert metadata.prepare(table[:], lengths[:], **kwargs) is plan
        return plan.local_page_table, plan.query_requests, plan.causal_lens

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    for pages, ends in (
        ([[1, 2, 3], [2, 4, 6]], [145, 129]),
        ([[0, 0, 0], [0, 0, 0]], [0, 0]),
        ([[2, 5, 1], [7, 1, 3]], [65, 171]),
    ):
        table.copy_(torch.tensor(pages, device="cuda", dtype=torch.int32))
        lengths.copy_(torch.tensor(ends, device="cuda", dtype=torch.int32))
        graph.replay()
        for actual, expected in zip(captured, forward()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("degree", [1, 2])
@pytest.mark.parametrize("mixed", [False, True])
def test_extend_metadata_clears_old_decode_memo(degree, mixed, monkeypatch):
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.paged import dsa
    from tokenspeed.runtime.layers.attention.backends.paged.trtllm_mla import (
        TRTLLMMLADecodeMetadata,
    )

    backend = _metadata_backend(
        monkeypatch,
        degree=degree,
        spec_num_tokens=1,
        index_topk=2048,
        index_kpool=None,
        dense_backend=SimpleNamespace(
            forward_decode_metadata=TRTLLMMLADecodeMetadata(),
            chunked_prefill_metadata=None,
            init_forward_metadata=lambda *args, **kwargs: None,
        ),
    )
    backend._dcp_indexer = indexer.DCPIndexerMetadata()
    previous = backend._dcp_indexer
    monkeypatch.setattr(dsa, "dsa_plan", lambda **kwargs: None)
    lengths = torch.tensor([4, 9], dtype=torch.int32)
    backend.init_forward_metadata(
        2,
        1,
        lengths,
        torch.tensor([[1], [2]], dtype=torch.int32),
        ForwardMode.MIXED if mixed else ForwardMode.EXTEND,
        extend_seq_lens=lengths[:1],
        extend_seq_lens_cpu=lengths[:1],
        extend_prefix_lens=torch.zeros(1, dtype=torch.int32),
        extend_prefix_lens_cpu=torch.zeros(1, dtype=torch.int32),
        extend_with_prefix=False,
        query_shard=None,
        page_table_cpu=None,
    )
    if mixed and degree > 1:
        assert backend.require_dcp_indexer_metadata() is not previous
        assert backend.require_dcp_indexer_metadata().plan is None
    else:
        assert backend._dcp_indexer is None


@pytest.mark.parametrize(
    "batch,columns,tile_sizes", [(2, 2, [2]), (64, 4096, [64]), (65, 4096, [64, 1])]
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA slot conversion")
def test_model_reuses_plan_but_maps_each_layers_candidates(
    batch, columns, tile_sizes, monkeypatch
):
    from tokenspeed.runtime.models import glm5

    memo = indexer.DCPIndexerMetadata()
    placement = CachePlacement(64, 9, (0,), 0)
    # Cover the 64 MiB decode tiling boundary without allocating large logits.
    table = torch.zeros(batch, columns, dtype=torch.int32, device="cuda")
    table[:, :2] = torch.tensor([7, 1], dtype=torch.int32)
    table[0, :2] = torch.tensor([5, 2], dtype=torch.int32)
    lengths = torch.full((batch,), 81, dtype=torch.int32, device="cuda")
    lengths[0] = 70
    cache = torch.empty((8 * 64, 132), dtype=torch.uint8, device="cuda")
    ctx = SimpleNamespace(
        num_extends=1,
        attn_backend=SimpleNamespace(
            forward_decode_metadata=SimpleNamespace(),
            cache_placement=lambda layer: placement,
            require_dcp_indexer_metadata=lambda: memo,
        ),
        token_to_kv_pool=SimpleNamespace(
            get_index_k_buffer=lambda layer: cache,
            arena=SimpleNamespace(kv_page_size=64),
        ),
    )
    layer = glm5.GlmMoeDsaAttention.__new__(glm5.GlmMoeDsaAttention)
    torch.nn.Module.__init__(layer)
    layer.attn_mqa = SimpleNamespace(layer_id=0)
    layer.indexer = SimpleNamespace(weights_softmax_scale=0.1)
    layer._decode_topk_indices_buffer = None
    layer._decode_topk_lens_buffer = None
    layer._retired_decode_workspaces = []
    seen = []

    def candidates(q, weights, index_k, local_table, requests, causal, **kwargs):
        tile = kwargs["metadata"]
        tile.validate(local_table, requests, causal, 64)
        assert index_k is cache
        assert q.shape[0] == requests.numel()
        assert causal.tolist() == lengths[requests.long()].tolist()
        seen.append(tile)
        # Different layers choose different positions; only the mappings reuse.
        offsets = torch.tensor([[2, -1]], dtype=torch.int32, device=q.device).repeat(
            q.shape[0], 1
        )
        scores = torch.tensor([[1.0, -torch.inf]], device=q.device).repeat(
            q.shape[0], 1
        )
        offsets[requests == 0] = torch.tensor(
            [iteration, 64], dtype=torch.int32, device=q.device
        )
        scores[requests == 0] = torch.tensor([3.0, 2.0], device=q.device)
        return offsets, scores

    monkeypatch.setattr(indexer, "dsa_index_candidates", candidates)
    with mock.patch.object(
        indexer, "resolve_cache_slots", wraps=indexer.resolve_cache_slots
    ) as translate:
        for iteration in range(2):
            result = layer._compute_decode_topk_indices_portable(
                indexer_output=glm5.GlmDsaIndexerOutput(
                    torch.zeros(
                        batch + 1, 16, 128, dtype=torch.bfloat16, device="cuda"
                    ),
                    torch.empty(0),
                    torch.zeros(batch + 1, 16, device="cuda"),
                ),
                ctx=ctx,
                seq_lens=lengths[:],
                page_table=table[:],
                q_len_per_req=1,
                decode_start=1,
                num_tokens=batch + 1,
                num_decode_tokens=batch,
                topk=2,
            )
            assert result.topk_indices.tolist() == [
                [-1, -1],
                [320 + iteration, 128],
            ] + [[450, -1]] * (batch - 1)
            assert result.topk_lens.tolist() == [0, 2] + [1] * (batch - 1)
        assert translate.call_count == 1
    assert memo.plan.max_logits_bytes == glm5._DCP_DECODE_LOGITS_BUDGET_BYTES
    assert [tile.query_requests.numel() for tile in seen] == tile_sizes * 2
    assert all(
        first is second
        for first, second in zip(seen[: len(tile_sizes)], seen[len(tile_sizes) :])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("budget,num_tiles", [(2048, 1), (512, 4)])
def test_tiled_selection_reuses_metadata_through_runtime_and_graph(budget, num_tiles):
    from tokenspeed_kernel.ops.kvcache.triton import index_k_block_split_scatter
    from tokenspeed_kernel.ops.quantization import quantize_fp8

    torch.manual_seed(943)
    q = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(4, 16, device="cuda")
    keys = torch.randn(256, 128, device="cuda", dtype=torch.bfloat16)
    values, scales = quantize_fp8(
        keys, granularity="token_group", group_size=128, scale_encoding="float32"
    )
    cache = torch.zeros(256, 132, device="cuda", dtype=torch.uint8)
    index_k_block_split_scatter(
        cache,
        values,
        scales,
        torch.arange(256, device="cuda"),
        page_size=64,
        head_dim=128,
        group_size=128,
        write_mask=None,
    )
    table = torch.tensor([[3, 1], [2, 3]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([127, 65], device="cuda", dtype=torch.int32)
    placement = CachePlacement(64, 4, (0,), 0)

    def forward():
        memo = indexer.DCPIndexerMetadata()
        outputs = []
        for _ in range(2):
            plan = memo.prepare(
                table,
                lengths,
                placement=placement,
                page_size=64,
                q_len_per_req=2,
                max_logits_bytes=budget,
            )
            assert len(plan.tiles) == num_tiles
            outputs.append(
                indexer.select_dsa_topk(
                    q,
                    weights,
                    cache,
                    table,
                    plan.query_requests,
                    plan.causal_lens,
                    placement=placement,
                    page_size=64,
                    topk=512,
                    softmax_scale=0.1,
                    initial_tokens=0,
                    local_tokens=0,
                    max_logits_bytes=budget,
                    metadata=plan,
                )
            )
        return outputs

    forward()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = forward()
    for pages, ends in (
        ([[3, 1], [2, 3]], [127, 65]),
        ([[0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 1]], [17, 91]),
    ):
        table.copy_(torch.tensor(pages, dtype=torch.int32, device="cuda"))
        lengths.copy_(torch.tensor(ends, dtype=torch.int32, device="cuda"))
        graph.replay()
        expected_counts = [
            max(0, end - 1 + token) for end in ends for token in range(2)
        ]
        for offsets, counts in captured:
            assert counts.tolist() == expected_counts
            for row, count in enumerate(expected_counts):
                assert set(offsets[row, :count].tolist()) == set(range(count))
                assert (offsets[row, count:] == -1).all()
