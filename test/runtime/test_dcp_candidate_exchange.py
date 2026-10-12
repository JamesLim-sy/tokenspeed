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

"""DCP candidate transport preserves indices and score bits with one gather."""

import pytest
import torch

from tokenspeed.runtime.layers.attention.dcp import indexer


def _candidates(rank, rows):
    offsets = torch.arange(24, dtype=torch.int32).reshape(3, 8)
    offsets = offsets + 2**24 + 1 + rank * 32
    offsets[0, -1] = 2**31 - 1 - rank
    bits = torch.tensor(
        [
            0,
            -(2**31),
            2139095040,
            -8388608,
            2143289345,
            -4194302,
            1065353216,
            1073741824,
        ],
        dtype=torch.int32,
    ).repeat(3, 1)
    offsets[1:, :] = -1
    if rank == 0:
        offsets[1, :2] = torch.tensor([7, 11])
    return offsets[:rows], bits[:rows].view(torch.float32)


def _reference(parts, topk):
    offsets = torch.cat([part[0] for part in parts], dim=1)
    scores = torch.cat([part[1] for part in parts], dim=1)
    valid = (offsets >= 0) & ~torch.isnan(scores) & (scores > -torch.inf)
    values, order = torch.topk(torch.where(valid, scores, -torch.inf), topk, dim=1)
    valid = values > -torch.inf
    return torch.where(valid, offsets.gather(1, order), -1), valid.sum(1).int()


@pytest.mark.parametrize("degree", [1, 2, 4, 8])
@pytest.mark.parametrize("rows", [0, 1, 3])
@pytest.mark.parametrize("strided", [False, True])
def test_candidate_exchange_preserves_bits_and_selection(
    monkeypatch, degree, rows, strided
):
    parts = [_candidates(rank, rows) for rank in range(degree)]
    offsets, scores = parts[0]
    if strided:
        offsets = offsets.repeat_interleave(2, dim=1)[:, ::2]
        scores = scores.repeat_interleave(2, dim=1)[:, ::2]
    calls = []

    def gather(payload, group, dim):
        calls.append(group)
        assert group == tuple(range(degree))
        assert dim == -1
        assert payload.dtype == torch.int32
        assert payload.is_contiguous()
        records = payload.reshape(rows, 8, 2)
        torch.testing.assert_close(records[..., 0], offsets, atol=0, rtol=0)
        torch.testing.assert_close(
            records[..., 1], scores.view(torch.int32), atol=0, rtol=0
        )
        return torch.cat(
            [
                torch.stack((ids, values.view(torch.int32)), -1).flatten(1)
                for ids, values in parts
            ],
            dim=dim,
        )

    monkeypatch.setattr(indexer, "all_gather", gather)
    actual = indexer.merge_index_candidates(
        offsets, scores, topk=4, group=tuple(range(degree))
    )
    expected = _reference(parts, 4)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert len(calls) == int(degree > 1)


@pytest.mark.parametrize(
    "offset_dtype,score_dtype",
    [(torch.int64, torch.float32), (torch.int32, torch.float64)],
)
def test_candidate_exchange_retains_other_dtypes(
    monkeypatch, offset_dtype, score_dtype
):
    parts = [
        tuple(
            t.to(dtype)
            for t, dtype in zip(_candidates(rank, 3), (offset_dtype, score_dtype))
        )
        for rank in range(2)
    ]
    calls = []

    def gather(tensor, group, dim):
        position = len(calls)
        calls.append(tensor.dtype)
        return torch.cat([part[position] for part in parts], dim=dim)

    monkeypatch.setattr(indexer, "all_gather", gather)
    actual = indexer.merge_index_candidates(*parts[0], topk=4, group=(0, 1))
    expected = _reference(parts, 4)
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
    assert calls == [offset_dtype, score_dtype]
