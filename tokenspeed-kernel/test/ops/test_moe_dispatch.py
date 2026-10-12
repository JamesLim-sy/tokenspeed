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

"""Portable fused dispatch correctness and runtime scalar specialization."""

import pytest
import torch
from moe_dispatch_utils import (
    DISPATCH_SHAPES,
    check_dispatch,
    check_graph_updates,
    make_dispatch_inputs,
)
from tokenspeed_kernel.ops.moe.dispatch import dispatch_topk_ids
from tokenspeed_kernel.ops.moe.triton.dispatch import _dispatch_topk_ids_kernel
from tokenspeed_kernel.selection import kernel_override
from utils import assert_no_triton_compile, kernel_supported

pytestmark = pytest.mark.skipif(
    not kernel_supported("triton_dispatch_topk_ids"),
    reason="requires Triton GPU dispatch",
)


@pytest.mark.parametrize("kernel_name", ["triton_dispatch_topk_ids", None])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layout", ["contiguous", "sliced", "transposed", "expanded"])
@pytest.mark.parametrize("tokens,top_k,width", DISPATCH_SHAPES)
def test_dispatch_matches_reference(kernel_name, dtype, layout, tokens, top_k, width):
    ids, dispatch = make_dispatch_inputs(tokens, top_k, width, dtype, layout, "cuda")
    if kernel_name is None:
        check_dispatch(dispatch_topk_ids, ids, dispatch)
    else:
        with kernel_override("moe", "dispatch", kernel_name):
            check_dispatch(dispatch_topk_ids, ids, dispatch)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_dispatch_graph_reads_updated_tables(dtype):
    with kernel_override("moe", "dispatch", "triton_dispatch_topk_ids"):
        check_graph_updates(dispatch_topk_ids, dtype)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_dispatch_batch_shapes_do_not_recompile(dtype):
    with kernel_override("moe", "dispatch", "triton_dispatch_topk_ids"):
        ids, dispatch = make_dispatch_inputs(17, 7, 3, dtype, "contiguous", "cuda")
        dispatch_topk_ids(ids, dispatch)
        with assert_no_triton_compile(_dispatch_topk_ids_kernel):
            for tokens, top_k, width in [
                (1, 1, 1),
                (16, 16, 16),
                (31, 3, 7),
                (257, 8, 4),
            ]:
                for layout in ["contiguous", "sliced", "transposed", "expanded"]:
                    ids, dispatch = make_dispatch_inputs(
                        tokens, top_k, width, dtype, layout, "cuda"
                    )
                    check_dispatch(dispatch_topk_ids, ids, dispatch)
