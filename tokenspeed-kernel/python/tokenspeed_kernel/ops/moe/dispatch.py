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

"""Expert placement dispatch: logical top-k ids onto physical replicas.

With redundant experts a logical expert has several physical replicas, each
on some rank. Routing then emits physical ids: every rank runs the same
routing over the same tokens, so the replica must be a pure function of the
token row and route rank -- ``replicas[logical, (row + rank) % count]`` --
for exactly one rank to own each (token, expert) pair. ``dispatch_topk_ids``
is the production implementation of that choice, fused into one GPU launch.
See ``README.md`` for backend selection. Load counting is the runtime's
(``record_expert_load``), applied to the mapped ids after routing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import (
    dense_tensor_format,
    format_signature,
    format_signatures,
)


@dataclass(frozen=True)
class ExpertDispatch:
    """One MoE layer's logical-to-physical routing tables.

    Attributes:
        replicas: ``[num_logical, X]`` int32 physical ids of every logical
            expert's replicas, ``-1`` padded; contiguous.
        num_replicas: ``[num_logical]`` int32 valid entries per row, all >= 1.
    """

    replicas: torch.Tensor
    num_replicas: torch.Tensor

    def __post_init__(self) -> None:
        if self.replicas.ndim != 2 or self.replicas.dtype != torch.int32:
            raise ValueError("replicas must be an int32 [num_logical, X] table")
        if (
            self.num_replicas.shape != (self.replicas.shape[0],)
            or self.num_replicas.dtype != torch.int32
        ):
            raise ValueError("num_replicas must be int32 [num_logical]")
        if not (self.replicas.is_contiguous() and self.num_replicas.is_contiguous()):
            raise ValueError("dispatch tables must be contiguous")
        if self.replicas.device != self.num_replicas.device:
            raise ValueError("dispatch tables must be on the same device")

    @property
    def max_replicas(self) -> int:
        return self.replicas.shape[1]


def dispatch_topk_ids(topk_ids: torch.Tensor, dispatch: ExpertDispatch) -> torch.Tensor:
    """Map logical top-k ids onto physical replicas (the serving path).

    Replica ``(row + route rank) % count`` of each logical expert, so every
    rank running the same routing over the same rows picks the same replica
    and the replicas of a hot expert share its routes evenly. The GPU kernels
    are graph-capturable; the ids must all be real experts in ``[0, num_logical)``
    (the runtime masks zero experts and padding around this call).

    Args:
        topk_ids: ``[tokens, top_k]`` logical ids, every entry in
            ``[0, num_logical)``.
        dispatch: The layer's replica tables, on the same device as the ids.

    Returns:
        ``[tokens, top_k]`` physical ids in ``topk_ids``' dtype.
    """
    if topk_ids.ndim != 2:
        raise ValueError("topk_ids must have shape [tokens, top_k]")
    if topk_ids.device != dispatch.replicas.device:
        raise ValueError("topk_ids and dispatch tables must be on the same device")
    if not topk_ids.is_cuda or topk_ids.dtype not in (torch.int32, torch.int64):
        return torch_dispatch_topk_ids(
            topk_ids, dispatch.replicas, dispatch.num_replicas
        )
    kernel = select_kernel(
        "moe",
        "dispatch",
        format_signature(topk_ids=dense_tensor_format(topk_ids.dtype)),
    )
    return kernel(topk_ids, dispatch.replicas, dispatch.num_replicas)


@register_kernel(
    "moe",
    "dispatch",
    name="torch_dispatch_topk_ids",
    solution="torch",
    signatures=format_signatures("topk_ids", "dense", {torch.int32, torch.int64}),
    priority=Priority.REFERENCE,
)
def torch_dispatch_topk_ids(
    topk_ids: torch.Tensor,
    replicas: torch.Tensor,
    num_replicas: torch.Tensor,
) -> torch.Tensor:
    """Reference implementation of the replica mapping.

    Args:
        topk_ids: Logical ids shaped [tokens, top_k].
        replicas: Contiguous int32 [num_logical, max_replicas] physical ids.
        num_replicas: Contiguous int32 [num_logical] valid replica counts.

    Returns:
        Physical ids with the input shape and dtype.
    """
    rows = torch.arange(topk_ids.shape[0], device=topk_ids.device)[:, None]
    ranks = torch.arange(topk_ids.shape[1], device=topk_ids.device)[None, :]
    logical = topk_ids.long()
    count = num_replicas.long()[logical]
    physical = replicas.long()[logical, (rows + ranks) % count]
    return physical.to(topk_ids.dtype)
