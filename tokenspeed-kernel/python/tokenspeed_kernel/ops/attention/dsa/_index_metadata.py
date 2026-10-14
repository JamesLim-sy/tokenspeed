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

"""Caller-owned, per-forward planning state for a DSA index query tile."""

from dataclasses import dataclass

import torch

INDEX_METADATA_FEATURE = "index_candidate_metadata"


@dataclass
class DSAIndexMetadata:
    """Reuse a tile's derived metadata across layers of one forward.

    The bound page table, query requests, causal lengths and page size must
    stay unchanged during that forward. Create a new instance after a metadata
    refresh, even when the inputs reuse the same storage. Queries, weights and
    layer-specific Index-K values are never cached.

    A leaf prepares ``state`` on first use. During graph capture that records
    the preparation before the first consuming layer; replay executes it again
    from the captured inputs, once per forward. The caller owns the lifetime.
    """

    local_page_table: torch.Tensor
    query_requests: torch.Tensor
    causal_lens: torch.Tensor
    page_size: int
    kernel_name: str | None = None
    state: object | None = None

    def validate(
        self,
        local_page_table: torch.Tensor,
        query_requests: torch.Tensor,
        causal_lens: torch.Tensor,
        page_size: int,
    ) -> None:
        """Reject reusing a tile with different input views or page geometry."""
        if page_size != self.page_size:
            raise ValueError("DSA index metadata page size changed")
        for expected, actual in (
            (self.local_page_table, local_page_table),
            (self.query_requests, query_requests),
            (self.causal_lens, causal_lens),
        ):
            if (
                expected.shape != actual.shape
                or expected.stride() != actual.stride()
                or expected.dtype != actual.dtype
                or expected.device != actual.device
                or expected.data_ptr() != actual.data_ptr()
            ):
                raise ValueError(
                    "DSA index metadata inputs changed; prepare a new tile"
                )
