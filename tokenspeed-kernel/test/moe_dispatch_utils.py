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

"""Shared dispatch inputs and graph checks for portable and NVIDIA tests."""

import torch
from tokenspeed_kernel.ops.moe.dispatch import ExpertDispatch, torch_dispatch_topk_ids

DISPATCH_SHAPES = [
    (0, 8, 4),
    (3, 0, 4),
    (1, 1, 1),
    (5, 3, 3),
    (17, 7, 7),
    (257, 8, 4),
    (4096, 8, 4),
    (513, 32, 16),
]


def make_dispatch_inputs(tokens, top_k, width, dtype, layout, device):
    generator = torch.Generator().manual_seed(17)
    experts = 37
    replicas = torch.randperm(experts * width, generator=generator).reshape(
        experts, width
    )
    counts = torch.randint(
        1, width + 1, (experts,), generator=generator, dtype=torch.int32
    )
    counts[0], counts[-1] = 1, width
    replicas[torch.arange(width)[None, :] >= counts[:, None]] = -1
    dispatch = ExpertDispatch(
        replicas.to(device=device, dtype=torch.int32), counts.to(device)
    )

    if layout == "contiguous":
        shape = (tokens, top_k)
    elif layout == "sliced":
        shape = (tokens * 2 + 1, top_k * 2 + 1)
    elif layout == "transposed":
        shape = (top_k, tokens)
    elif layout == "expanded":
        shape = (tokens, 1)
    else:
        raise ValueError(f"unknown layout {layout!r}")
    ids = torch.randint(experts, shape, dtype=dtype, generator=generator).to(device)
    if layout == "sliced":
        ids = ids[1::2, 1::2]
    elif layout == "transposed":
        ids = ids.t()
    elif layout == "expanded":
        ids = ids.expand(tokens, top_k)
    return ids, dispatch


def reference_dispatch(ids, dispatch):
    # The oracle stays on CPU, independent of either GPU implementation.
    return torch_dispatch_topk_ids(
        ids.cpu(), dispatch.replicas.cpu(), dispatch.num_replicas.cpu()
    )


def check_dispatch(implementation, ids, dispatch):
    actual = implementation(ids, dispatch)
    torch.testing.assert_close(
        actual.cpu(), reference_dispatch(ids, dispatch), rtol=0, atol=0
    )
    assert actual.dtype == ids.dtype
    assert actual.device == ids.device
    assert actual.shape == ids.shape


def check_graph_updates(implementation, dtype):
    ids, dispatch = make_dispatch_inputs(129, 7, 5, dtype, "sliced", "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        implementation(ids, dispatch)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = implementation(ids, dispatch)

    for iteration in range(3):
        # Mimic rebalancing: both table contents and replica counts change
        # without replacing their storage. Routing ids change on replay too.
        if iteration:
            ids.copy_((ids + 1) % dispatch.replicas.shape[0])
            dispatch.replicas.copy_(
                torch.arange(
                    dispatch.replicas.numel(), device="cuda", dtype=torch.int32
                ).reshape_as(dispatch.replicas)
                + iteration * 1000
            )
            dispatch.num_replicas.copy_(
                (dispatch.num_replicas % dispatch.max_replicas) + 1
            )
        graph.replay()
        torch.testing.assert_close(
            output.cpu(), reference_dispatch(ids, dispatch), rtol=0, atol=0
        )
