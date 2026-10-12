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

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tokenspeed.runtime.layers.attention.dcp import indexer
from tokenspeed.runtime.layers.attention.dcp.placement import CachePlacement


@pytest.mark.parametrize("degree", [2, 4])
@pytest.mark.parametrize("block_size", [64, 128])
@pytest.mark.parametrize("query_width", [1, 3])
def test_decode_plan_reuses_translation_and_preserves_geometry(
    degree, block_size, query_width
):
    subpages = block_size // 64
    blocks = torch.tensor([[3, 2, 5], [0, 10, 1]], dtype=torch.int32)
    table = (blocks[..., None] * subpages + torch.arange(subpages)).reshape(2, -1)
    seq_lens = torch.tensor([137, 3], dtype=torch.int32)
    kwargs = dict(
        placement=CachePlacement(block_size, 10, tuple(range(degree)), 0),
        page_size=64,
        q_len_per_req=query_width,
        max_logits_bytes=2 * table.shape[1] * 64 * 4,
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


@pytest.mark.parametrize("degree", [1, 2])
def test_backend_refresh_and_draft_updates_invalidate_index_plan(degree, monkeypatch):
    from tokenspeed.runtime.layers.attention.backends.paged import dsa
    from tokenspeed.runtime.layers.attention.backends.paged.trtllm_mla import (
        TRTLLMMLADecodeMetadata,
    )

    backend = dsa.DSABackend.__new__(dsa.DSABackend)
    backend.dcp_group = tuple(range(degree))
    backend.spec_num_tokens = 1
    backend.kernel_page_size = 64
    metadata = TRTLLMMLADecodeMetadata(seq_lens_k=torch.tensor([4], dtype=torch.int32))
    backend._dense_backend = SimpleNamespace(
        forward_decode_metadata=metadata,
        refresh_decode_metadata=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(dsa, "dsa_plan", lambda **kwargs: None)
    table = torch.tensor([[1]], dtype=torch.int32)
    lengths = torch.tensor([4], dtype=torch.int32)
    from tokenspeed.runtime.execution.graph_ptr_guard import (
        snapshot_graph_metadata,
        verify_graph_metadata,
    )

    backend._dense_backend.child_backends = lambda: ()
    snapshot = None
    previous = None
    for update in (
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
    table = torch.tensor([[1, 2, 3], [2, 4, 6]], dtype=torch.int32, device="cuda")
    lengths = torch.tensor([145, 129], dtype=torch.int32, device="cuda")
    kwargs = dict(
        placement=CachePlacement(64, 9, (0, 1), 0),
        page_size=64,
        q_len_per_req=2,
        max_logits_bytes=3 * 64 * 4,
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


@pytest.mark.parametrize("mixed", [False, True])
def test_extend_metadata_clears_old_decode_memo(mixed, monkeypatch):
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.layers.attention.backends.paged import dsa
    from tokenspeed.runtime.layers.attention.backends.paged.trtllm_mla import (
        TRTLLMMLADecodeMetadata,
    )

    backend = dsa.DSABackend.__new__(dsa.DSABackend)
    backend.dcp_group = (0, 1)
    backend.spec_num_tokens = 1
    backend.kernel_page_size = 64
    backend.is_draft = False
    backend._dcp_indexer = indexer.DCPIndexerMetadata()
    previous = backend._dcp_indexer
    backend._dense_backend = SimpleNamespace(
        forward_decode_metadata=TRTLLMMLADecodeMetadata(),
        chunked_prefill_metadata=None,
        init_forward_metadata=lambda *args, **kwargs: None,
    )
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
    if mixed:
        assert backend.require_dcp_indexer_metadata() is not previous
        assert backend.require_dcp_indexer_metadata().plan is None
    else:
        assert backend._dcp_indexer is None


def test_model_reuses_plan_but_maps_each_layers_candidates(monkeypatch):
    from tokenspeed.runtime.models import glm5

    memo = indexer.DCPIndexerMetadata()
    placement = CachePlacement(64, 9, (0,), 0)
    table = torch.tensor([[5, 2], [7, 1]], dtype=torch.int32)
    lengths = torch.tensor([70, 81], dtype=torch.int32)
    cache = torch.empty((8 * 64, 132), dtype=torch.uint8)
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
        assert causal.tolist() == [70, 81]
        seen.append(tile)
        # Different layers choose different positions; only the mappings reuse.
        return (
            torch.tensor([[len(seen) - 1, 64], [2, -1]], dtype=torch.int32),
            torch.tensor([[3.0, 2.0], [1.0, -torch.inf]]),
        )

    monkeypatch.setattr(indexer, "dsa_index_candidates", candidates)
    with mock.patch.object(
        indexer, "resolve_cache_slots", wraps=indexer.resolve_cache_slots
    ) as translate:
        for iteration in range(2):
            result = layer._compute_decode_topk_indices_portable(
                indexer_output=glm5.GlmDsaIndexerOutput(
                    torch.zeros(3, 16, 128, dtype=torch.bfloat16),
                    torch.empty(0),
                    torch.zeros(3, 16),
                ),
                ctx=ctx,
                seq_lens=lengths[:],
                page_table=table[:],
                q_len_per_req=1,
                decode_start=1,
                num_tokens=3,
                num_decode_tokens=2,
                topk=2,
            )
            assert result.topk_indices.tolist() == [
                [-1, -1],
                [320 + iteration, 128],
                [450, -1],
            ]
            assert result.topk_lens.tolist() == [0, 2, 1]
        assert translate.call_count == 1
    assert seen[0] is seen[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tiled_selection_reuses_metadata_through_runtime_and_graph():
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
                max_logits_bytes=512,
            )
            assert len(plan.tiles) == 4
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
                    max_logits_bytes=512,
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
