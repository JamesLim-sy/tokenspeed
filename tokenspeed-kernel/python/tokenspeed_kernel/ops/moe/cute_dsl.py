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

"""Optional NVIDIA single-launch expert dispatch using CuTe DSL."""

from functools import cache

import torch
from tokenspeed_kernel.platform import ArchVersion, CapabilityRequirement
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import format_signatures

try:
    import cutlass
    import cutlass.cute as cute
    from cuda.bindings.driver import CUstream
    from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


if _AVAILABLE:

    @cute.kernel
    def _dispatch_topk_ids_kernel(
        ids: cute.Tensor,
        replicas: cute.Tensor,
        counts: cute.Tensor,
        out: cute.Tensor,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        index = block * 256 + thread
        if index < cute.size(out):
            row = index // ids.shape[1]
            rank = index % ids.shape[1]
            logical = cutlass.Int64(ids[row, rank])
            count = counts[logical]
            replica = (row + rank) % count
            out[row, rank] = replicas[logical, replica]

    @cute.jit
    def _launch_dispatch_topk_ids(
        ids: cute.Tensor,
        replicas: cute.Tensor,
        counts: cute.Tensor,
        out: cute.Tensor,
        stream: CUstream,
    ):
        _dispatch_topk_ids_kernel(ids, replicas, counts, out).launch(
            grid=((cute.size(out) + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    @cache
    def _compile_dispatch(dtype: torch.dtype, device_index: int):
        # All extents and input strides are symbolic, including strides 0/1.
        # Only dtype and device specialize the binary; rebalancing updates
        # table contents in place and never changes the compiled launch.
        id_type = cutlass.Int32 if dtype == torch.int32 else cutlass.Int64
        rows, top_k, experts, width = (cute.sym_int() for _ in range(4))
        ids = make_fake_tensor(
            id_type,
            (rows, top_k),
            (cute.sym_int64(), cute.sym_int64()),
            assumed_align=id_type.width // 8,
        )
        replicas = make_fake_tensor(
            cutlass.Int32,
            (experts, width),
            (width, 1),
            assumed_align=4,
        )
        counts = make_fake_tensor(cutlass.Int32, (experts,), (1,), assumed_align=4)
        out = make_fake_tensor(
            id_type,
            (rows, top_k),
            (top_k, 1),
            assumed_align=id_type.width // 8,
        )
        with torch.cuda.device(device_index):
            return cute.compile(
                _launch_dispatch_topk_ids,
                ids,
                replicas,
                counts,
                out,
                make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )

    @register_kernel(
        "moe",
        "dispatch",
        name="cute_dsl_dispatch_topk_ids",
        solution="cute_dsl",
        capability=CapabilityRequirement(
            vendors=frozenset({"nvidia"}),
            min_arch_version=ArchVersion(9, 0),
        ),
        signatures=format_signatures("topk_ids", "dense", {torch.int32, torch.int64}),
        priority=Priority.PERFORMANT,
    )
    def cute_dsl_dispatch_topk_ids(
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
            Contiguous physical ids with the input shape and dtype. The FFI
            accepts strided tensors directly and uses their current stream,
            without Python DLPack conversions during CUDA graph capture.
        """
        out = torch.empty(topk_ids.shape, dtype=topk_ids.dtype, device=topk_ids.device)
        if topk_ids.numel():
            compiled = _compile_dispatch(topk_ids.dtype, topk_ids.device.index)
            compiled(topk_ids, replicas, num_replicas, out)
        return out
