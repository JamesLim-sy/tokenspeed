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

"""Portable single-launch logical-to-physical expert dispatch."""

import torch
from tokenspeed_kernel._triton import tl, triton
from tokenspeed_kernel.platform import CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures


@triton.jit(
    do_not_specialize=["total", "top_k", "table_width", "row_stride", "col_stride"],
)
def _dispatch_topk_ids_kernel(
    ids,
    replicas,
    counts,
    out,
    total,
    top_k,
    table_width,
    row_stride,
    col_stride,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = index < total
    row = index // top_k
    rank = index % top_k
    logical = tl.load(
        ids + row.to(tl.int64) * row_stride + rank.to(tl.int64) * col_stride,
        mask=valid,
        other=0,
    ).to(tl.int64)
    count = tl.load(counts + logical, mask=valid, other=1)
    replica = (row + rank) % count
    physical = tl.load(
        replicas + logical * table_width + replica,
        mask=valid,
        other=0,
    )
    tl.store(out + index, physical, mask=valid)


@register_kernel(
    "moe",
    "dispatch",
    name="triton_dispatch_topk_ids",
    solution="triton",
    capability=CapabilityRequirement(vendors=frozenset({"nvidia", "amd"})),
    signatures=format_signatures("topk_ids", "dense", {torch.int32, torch.int64}),
    priority=Priority.PORTABLE + 1,
)
def triton_dispatch_topk_ids(
    topk_ids: torch.Tensor,
    replicas: torch.Tensor,
    num_replicas: torch.Tensor,
) -> torch.Tensor:
    """Map logical ids onto physical replicas in one launch.

    Args:
        topk_ids: Int32/int64 logical ids shaped [tokens, top_k].
        replicas: Contiguous int32 [num_logical, max_replicas] physical ids.
        num_replicas: Contiguous int32 [num_logical] valid replica counts.

    Returns:
        Contiguous physical ids with the input shape and dtype. Table contents
        and input strides are read at runtime, including during graph replay.
    """
    out = torch.empty(topk_ids.shape, dtype=topk_ids.dtype, device=topk_ids.device)
    if topk_ids.numel():
        _dispatch_topk_ids_kernel[(triton.cdiv(topk_ids.numel(), 256),)](
            topk_ids,
            replicas,
            num_replicas,
            out,
            topk_ids.numel(),
            topk_ids.shape[1],
            replicas.shape[1],
            topk_ids.stride(0),
            topk_ids.stride(1),
            BLOCK=256,
        )
    return out
