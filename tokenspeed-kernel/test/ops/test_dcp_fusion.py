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

"""DCP fusion preserves candidate order, partial numerics and graph refresh."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa._triton import candidate_exchange as exchange
from tokenspeed_kernel.ops.attention.dsa._triton import index_candidates as candidates
from tokenspeed_kernel.ops.attention.dsv4._triton import dcp
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


@pytest.mark.parametrize("rows", [0, 1, 7])
@pytest.mark.parametrize("degree", [1, 3, 8])
@pytest.mark.parametrize("width,topk", [(16, 8), (2053, 2049)])
def test_rank_major_merge_preserves_order_and_bits(rows, degree, width, topk):
    ids = torch.arange(degree * rows * width, device="cuda", dtype=torch.int32)
    ids = ids.reshape(degree, rows, width) + 2**24 + 1
    scores = torch.rand((degree, rows, width), device="cuda")
    if rows:
        ids[:, 0, -1] = -1
        scores[:, 0, :4] = torch.tensor(
            [torch.inf, -torch.inf, torch.nan, -0.0], device="cuda"
        )
        scores[:, -1, 4:] = -torch.inf
    records = torch.stack((ids, scores.view(torch.int32)), dim=-1)
    for source in range(degree):
        packed = exchange.pack_index_candidates(
            records[source, ..., 0], records[source, ..., 1].view(torch.float32)
        )
        assert packed.data_ptr() == records[source].data_ptr()
    actual, counts = exchange.merge_packed_index_candidates(records, topk=topk)
    row_ids = ids.permute(1, 0, 2).reshape(rows, degree * width)
    row_scores = scores.permute(1, 0, 2).reshape(rows, degree * width)
    valid = (row_ids >= 0) & ~torch.isnan(row_scores) & (row_scores > -torch.inf)
    values, order = torch.topk(torch.where(valid, row_scores, -torch.inf), topk, dim=-1)
    expected = torch.where(values > -torch.inf, row_ids.gather(1, order), -1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        counts, (values > -torch.inf).sum(1).int(), atol=0, rtol=0
    )


def test_candidate_gather_bounds_reads_with_uninitialized_tail():
    logits = torch.full((2, 128), torch.nan, device="cuda")
    values = torch.tensor(
        [[3.0, 4.0, torch.inf, torch.nan, -torch.inf]] * 2, device="cuda"
    )
    offsets = torch.tensor(
        [[0, 63, 64, 1 << 29, -1]] * 2, device="cuda", dtype=torch.int32
    )
    positions = torch.tensor([[9, 7], [5, 3]], device="cuda", dtype=torch.int32)
    lengths = torch.tensor([[64], [0]], device="cuda", dtype=torch.int32)
    ids, scores = candidates.gather_index_candidates(
        offsets,
        logits,
        positions,
        64,
        lengths=lengths,
        selected_scores=values,
    )
    torch.testing.assert_close(
        ids,
        torch.tensor(
            [[576, 639, -1, -1, -1], [-1] * 5], device="cuda", dtype=torch.int32
        ),
    )
    torch.testing.assert_close(
        scores,
        torch.tensor(
            [[3.0, 4.0, -torch.inf, -torch.inf, -torch.inf], [-torch.inf] * 5],
            device="cuda",
        ),
    )
    packed = exchange.pack_index_candidates(ids, scores)
    assert packed.data_ptr() == ids.data_ptr()


def test_length_aware_helpers_do_not_specialize_on_batch_geometry():
    def run(rows, width):
        logits = torch.randn(rows, width, device="cuda")
        positions = torch.arange(
            (width + 63) // 64, device="cuda", dtype=torch.int32
        ).repeat(rows, 1)
        lengths = torch.full((rows, 1), width, dtype=torch.int32, device="cuda")
        candidates.mask_index_scores(
            logits, positions, lengths, lengths.flatten(), 64, 0, 0, mask_tail=False
        )
        offsets = torch.arange(8, device="cuda", dtype=torch.int32).repeat(rows, 1)
        return candidates.gather_index_candidates(
            offsets,
            logits,
            positions,
            64,
            lengths=lengths,
            selected_scores=logits[:, :8],
        )

    run(4, 256)
    with assert_no_triton_compile(
        candidates._mask_scores, candidates._gather_candidates
    ):
        for rows, width in ((1, 64), (3, 128), (16, 516)):
            run(rows, width)


def _partials(degree, tokens, dim, dtype, strided):
    heads = degree * 2
    outputs = torch.randn(
        degree, tokens, heads, dim * (2 if strided else 1), device="cuda", dtype=dtype
    )
    if strided:
        outputs = outputs[..., ::2]
    lses = torch.randn(degree, tokens, heads * 2, device="cuda")[..., ::2] * 3
    if tokens >= 5:
        lses[0, 0] = torch.inf
        outputs[0, 0] = torch.nan
        lses[:, 1] = -torch.inf
        outputs[:, 1] = torch.nan
        lses[0, 2, 0] = torch.nan
        if degree > 1:
            lses[0, 3] = -1000
            lses[1:, 3] = 1000
            outputs[0, 3] = torch.nan
    return outputs, lses


def _reference(outputs, lses, sink):
    lses = torch.where(torch.isposinf(lses), -torch.inf, lses)
    maximum = lses.max(0).values
    safe = torch.where(torch.isneginf(maximum), 0, maximum)
    mass = (lses - safe).exp()
    denominator = mass.sum(0)
    weights = mass / denominator.clamp_min(torch.finfo(torch.float32).tiny)
    values = torch.where(torch.isneginf(lses)[..., None], 0, outputs.float())
    result = (values * weights[..., None]).sum(0)
    if sink is not None:
        normalizer = torch.where(
            denominator == 0, -torch.inf, maximum + denominator.log()
        )
        factor = torch.where(
            torch.isneginf(normalizer), 0, 1 / (1 + (sink - normalizer).exp())
        )
        result *= factor[..., None]
    return result


@pytest.mark.parametrize("degree", [1, 3, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "dim,strided,use_sink", [(15, True, True), (512, False, False)]
)
def test_partial_transport_and_merge(degree, dtype, dim, strided, use_sink):
    torch.manual_seed(819)
    outputs, lses = _partials(degree, 5, dim, dtype, strided)
    sink = torch.randn(degree * 2, device="cuda") if use_sink else None
    sent = [
        dcp.dispatch_dcp_attention_partials(outputs[r], lses[r], degree=degree)
        for r in range(degree)
    ]
    expected = _reference(outputs, lses, sink)
    for owner in range(degree):
        received = torch.stack([payload[owner] for payload in sent])
        local_sink = None if sink is None else sink[owner * 2 : (owner + 1) * 2]
        kwargs = dict(head_dim=dim, partial_dtype=dtype, sink=local_sink)
        actual = dcp.combine_dcp_attention_partials(
            received, output_dtype=torch.float32, **kwargs
        )
        torch.testing.assert_close(
            actual,
            expected[:, owner * 2 : (owner + 1) * 2],
            rtol=3e-6,
            atol=2e-6,
            equal_nan=True,
        )
        rounded = dcp.combine_dcp_attention_partials(
            received, output_dtype=dtype, **kwargs
        )
        torch.testing.assert_close(
            rounded, actual.to(dtype), rtol=0, atol=0, equal_nan=True
        )
        assert rounded.is_contiguous()


def test_fusions_graph_refresh_and_batch_sizes_do_not_recompile():
    degree, width, topk, dim = 2, 16, 8, 16

    def run(rows):
        records = torch.zeros(
            (degree, rows, width, 2), device="cuda", dtype=torch.int32
        )
        records[..., 0] = torch.arange(width, device="cuda", dtype=torch.int32)
        records[..., 1] = torch.arange(width, device="cuda").float().view(torch.int32)
        selected, counts = exchange.merge_packed_index_candidates(records, topk=topk)
        outputs, lses = _partials(degree, rows, dim, torch.bfloat16, False)
        packed = dcp.dispatch_dcp_attention_partials(outputs[0], lses[0], degree=degree)
        merged = dcp.combine_dcp_attention_partials(
            packed,
            head_dim=dim,
            partial_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            sink=None,
        )
        return records, packed, selected, counts, merged

    run(4)
    with assert_no_triton_compile(
        exchange._extract_rank_major_scores,
        exchange._gather_selected_candidates,
        dcp._dispatch_dcp_attention_partials,
        dcp._combine_dcp_attention_partials,
    ):
        for rows in (1, 3, 16, 33):
            run(rows)
    records, packed, selected, counts, merged = run(3)
    saved_records, saved_packed = records.clone(), packed.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected, counts = exchange.merge_packed_index_candidates(records, topk=topk)
        merged = dcp.combine_dcp_attention_partials(
            packed,
            head_dim=dim,
            partial_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            sink=None,
        )
    graph.replay()
    expected = selected.clone(), counts.clone(), merged.clone()
    records[..., 0] = -1
    packed[..., -1] = torch.tensor(-torch.inf, device="cuda").view(torch.int32)
    graph.replay()
    assert (selected == -1).all() and (counts == 0).all() and (merged == 0).all()
    records.copy_(saved_records)
    packed.copy_(saved_packed)
    graph.replay()
    for got, want in zip((selected, counts, merged), expected):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
