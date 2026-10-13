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

"""DCP mask, length-aware selection and logical candidate gathering on Hopper."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import deep_gemm, dsa_index_candidates
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.hip is not None
    or not current_platform().is_hopper,
    reason="requires an NVIDIA Hopper GPU",
)

deep_select = pytest.importorskip(
    "tokenspeed_kernel.ops.attention.dsa._cute_dsl.deep_select"
)

PAGE_SIZE = 64
INITIAL_TOKENS = 4
LOCAL_TOKENS = 8


def _scores(rows, width, *, ties):
    # Keep a padded, non-contiguous row view, as DeepGEMM can return one.
    stride = (width + 3) // 4 * 4 + 4
    storage = torch.full((rows, stride), torch.nan, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(421)
    values = torch.randperm(width, generator=generator, device="cuda").float()
    if ties:
        values = values.remainder(7)
    scores = storage[:, :width]
    scores.copy_(values.expand(rows, -1))
    scores[:, 7] = torch.nan
    scores[:, 8] = torch.inf
    scores[:, 9] = -torch.inf
    return scores


def _metadata(width, lengths):
    rows = len(lengths)
    # Missing foreign pages leave gaps in global logical positions.
    positions = (
        torch.arange((width + PAGE_SIZE - 1) // PAGE_SIZE, device="cuda") * 2
    ).to(torch.int32)
    positions = positions.repeat(rows, 1)
    causal = []
    for length in lengths:
        last = max(0, length - 1)
        causal.append(
            (last // PAGE_SIZE * 2) * PAGE_SIZE + last % PAGE_SIZE + 1 if length else 0
        )
    return (
        positions,
        torch.tensor(lengths, dtype=torch.int32, device="cuda").reshape(-1, 1),
        torch.tensor(causal, dtype=torch.int32, device="cuda"),
    )


def _select_candidates(scores, positions, lengths, causal, topk):
    return deep_gemm._select_dcp_index_candidates(
        scores,
        positions,
        lengths,
        causal,
        page_size=PAGE_SIZE,
        topk=topk,
        initial_tokens=INITIAL_TOKENS,
        local_tokens=LOCAL_TOKENS,
    )


def _check_candidates(raw, positions, lengths, causal, topk, logical, values):
    # Build the oracle from the unmasked input and global positions, rather
    # than reusing either the production mask or its selected offsets.
    raw, positions = raw.cpu(), positions.cpu()
    logical, values = logical.cpu(), values.cpu()
    for row, length in enumerate(lengths.flatten().cpu().tolist()):
        columns = torch.arange(length)
        global_positions = (
            positions[row, columns // PAGE_SIZE] * PAGE_SIZE + columns % PAGE_SIZE
        )
        expected = raw[row, :length].clone()
        expected[~torch.isfinite(expected)] = -torch.inf
        forced = (global_positions < INITIAL_TOKENS) | (
            global_positions >= int(causal[row]) - LOCAL_TOKENS
        )
        expected[forced] = torch.inf
        expected_count = min(topk, int((expected > -torch.inf).sum()))
        valid = logical[row] >= 0
        assert int(valid.sum()) == expected_count
        assert (logical[row, ~valid] == -1).all()
        assert torch.isneginf(values[row, ~valid]).all()
        chosen_positions = logical[row, valid].tolist()
        assert len(set(chosen_positions)) == expected_count
        expected_by_position = dict(zip(global_positions.tolist(), expected.tolist()))
        assert set(chosen_positions).issubset(expected_by_position)
        assert set(global_positions[forced].tolist()).issubset(chosen_positions)
        chosen_values = values[row, valid]
        torch.testing.assert_close(
            chosen_values,
            torch.tensor([expected_by_position[p] for p in chosen_positions]),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            chosen_values.sort(descending=True).values,
            torch.topk(expected, expected_count).values,
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize(
    "width,topk",
    [(128, 512), (512, 512), (8193, 512), (8193, 1024), (65536, 2048)],
)
@pytest.mark.parametrize("ties", [False, True])
def test_dcp_mask_select_gather(width, topk, ties, monkeypatch):
    lengths = [min(width, n) for n in (0, 1, topk - 1, topk, topk + 1, width)]
    scores = _scores(len(lengths), width, ties=ties)
    raw = scores.clone()
    positions, lengths, causal = _metadata(width, lengths)
    calls = []
    select = deep_select.deepselect_topk

    def checked_select(scores, ends, *args, **kwargs):
        torch.testing.assert_close(ends, lengths.flatten(), atol=0, rtol=0)
        calls.append(True)
        return select(scores, ends, *args, **kwargs)

    monkeypatch.setattr(deep_select, "deepselect_topk", checked_select)
    logical, values = _select_candidates(scores, positions, lengths, causal, topk)
    assert calls == [True]
    _check_candidates(raw, positions, lengths, causal, topk, logical, values)


def test_dcp_selector_graph_refresh_and_precompiled_clusters():
    # An eager short-row call must also prepare the larger cluster variants
    # that a subsequent batch can choose, without specializing on row geometry.
    small = torch.ones((1, 128), device="cuda")
    small_positions, small_lengths, small_causal = _metadata(128, [128])
    _select_candidates(small, small_positions, small_lengths, small_causal, 512)
    compiled = deep_select.compiled_selector.cache_info().misses

    width, topk = 65536, 512
    raw = _scores(3, width, ties=False)
    scores = torch.empty_strided(raw.shape, raw.stride(), device="cuda")
    positions, lengths, causal = _metadata(width, [width, topk + 1, 0])

    def run():
        scores.copy_(raw)
        return _select_candidates(scores, positions, lengths, causal, topk)

    run()
    assert deep_select.compiled_selector.cache_info().misses == compiled
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logical, values = run()
    for new_lengths in ([width, topk + 1, 0], [0, 0, 0], [1, 87, topk], [width, 0, 1]):
        _, updated_lengths, updated_causal = _metadata(width, new_lengths)
        lengths.copy_(updated_lengths)
        causal.copy_(updated_causal)
        graph.replay()
        _check_candidates(raw, positions, lengths, causal, topk, logical, values)
    assert deep_select.compiled_selector.cache_info().misses == compiled


@pytest.mark.parametrize(
    "unsupported",
    ["topk", "row_stride", "base_alignment", "column_stride", "storage", "dependency"],
)
def test_dcp_selector_retains_torch_fallback(unsupported, monkeypatch):
    topk = 512
    if unsupported == "topk":
        scores = torch.randn((2, 128), device="cuda")
        topk = 4096
    elif unsupported == "row_stride":
        scores = torch.randn((2, 514), device="cuda")[:, :513]
    elif unsupported == "base_alignment":
        scores = torch.randn((2, 520), device="cuda")[:, 1:514]
    elif unsupported == "column_stride":
        scores = torch.randn((2, 1024), device="cuda")[:, ::2]
    elif unsupported == "storage":
        # The logical row fits, but the final aligned vector load would not.
        scores = torch.randn(517, device="cuda").as_strided((1, 513), (516, 1), 4)
    else:
        scores = torch.randn((2, 1024), device="cuda")
        monkeypatch.setattr(deep_gemm, "_dcp_deep_select", None)
    lengths = torch.full(
        (scores.shape[0], 1), scores.shape[1], dtype=torch.int32, device="cuda"
    )

    def unexpected_warmup(*args):
        pytest.fail("unsupported inputs must not compile or launch DeepSelect")

    monkeypatch.setattr(deep_gemm, "_warmup_dcp_topk", unexpected_warmup)
    assert not deep_gemm._dcp_topk_supports_lengths(scores, topk)
    raw = scores.clone()
    positions, lengths, causal = _metadata(scores.shape[1], lengths.flatten().tolist())
    logical, values = _select_candidates(scores, positions, lengths, causal, topk)
    _check_candidates(raw, positions, lengths, causal, topk, logical, values)


def test_deep_gemm_facade_selects_with_original_empty_lengths(monkeypatch):
    table = torch.tensor(
        [[3, -1, 1, 2], [-1, -1, -1, -1]], dtype=torch.int32, device="cuda"
    )
    requests = torch.tensor([0, 1, 0], dtype=torch.int32, device="cuda")
    causal = torch.tensor([193, 80, 0], dtype=torch.int32, device="cuda")
    q = torch.ones((3, 16, 128), dtype=torch.bfloat16, device="cuda")
    weights = torch.ones((3, 16), device="cuda")
    cache = torch.zeros((256, 132), dtype=torch.uint8, device="cuda")
    select = deep_select.deepselect_topk
    calls = []

    def checked_select(scores, ends, *args, **kwargs):
        assert ends.tolist() == [129, 0, 0]
        calls.append(True)
        return select(scores, ends, *args, **kwargs)

    monkeypatch.setattr(deep_select, "deepselect_topk", checked_select)
    logical, values = dsa_index_candidates(
        q,
        weights,
        cache,
        table,
        requests,
        causal,
        page_size=PAGE_SIZE,
        topk=512,
        softmax_scale=0.1,
        initial_tokens=INITIAL_TOKENS,
        local_tokens=LOCAL_TOKENS,
        solution="deep_gemm",
    )
    assert calls == [True]
    assert set(logical[0, logical[0] >= 0].tolist()) == set(range(64)) | set(
        range(128, 193)
    )
    assert (logical[1:] == -1).all()
    assert torch.isneginf(values[1:]).all()
