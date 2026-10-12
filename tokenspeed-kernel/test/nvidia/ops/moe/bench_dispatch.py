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

"""Compare Torch, Triton and CuTe DSL expert dispatch, including allocations.

Run from the repository root with PYTHONPATH=tokenspeed-kernel/python::

    python tokenspeed-kernel/test/nvidia/ops/moe/bench_dispatch.py \
        --tokens 1 16 128 512 4096 16384 --top-k 8 --experts 256 \
        --replicas 4 --calls 64 --repeats 15 --output /tmp/moe_dispatch.json

Graph timing excludes CPU dispatch overhead. Eager wall time includes Python,
allocations, launches and one synchronization per batch of calls. Neither
measurement includes expert GEMMs or communication.
"""

import argparse
import json
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import torch
from tokenspeed_kernel.ops.moe.dispatch import ExpertDispatch
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_KERNELS = (
    "torch_dispatch_topk_ids",
    "triton_dispatch_topk_ids",
    "cute_dsl_dispatch_topk_ids",
)


def _measure(ids, dispatch, calls, repeats):
    # Resolve each implementation once, excluding selector overhead from every
    # backend. The Torch leaf is the original unfused serving implementation.
    implementations = {
        name: select_kernel(
            "moe",
            "dispatch",
            format_signature(topk_ids=dense_tensor_format(ids.dtype)),
            override=name,
        )
        for name in _KERNELS
    }
    for name, kernel in implementations.items():
        if kernel.name != name:
            raise ValueError(
                "Unset TOKENSPEED_KERNEL_OVERRIDE_MOE_DISPATCH before benchmarking"
            )
    graphs = {}
    outputs = {}
    for name in _KERNELS:
        kernel = implementations[name]
        for _ in range(3):
            output = kernel(ids, dispatch.replicas, dispatch.num_replicas)
        outputs[name] = output
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls):
                kernel(ids, dispatch.replicas, dispatch.num_replicas)
        graphs[name] = graph
        graph.replay()
    torch.cuda.synchronize()
    for output in outputs.values():
        torch.testing.assert_close(output, outputs[_KERNELS[0]], atol=0, rtol=0)

    samples = {name: [] for name in _KERNELS}
    eager_samples = {name: [] for name in _KERNELS}
    for iteration in range(repeats):
        # Rotate measurement order to reduce clock/thermal ordering bias.
        order = _KERNELS[iteration % 3 :] + _KERNELS[: iteration % 3]
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / calls)
        for name in order:
            kernel = implementations[name]
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            for _ in range(calls):
                kernel(ids, dispatch.replicas, dispatch.num_replicas)
            torch.cuda.synchronize()
            eager_samples[name].append((time.perf_counter() - start_time) * 1e6 / calls)
    return {
        name: {
            "graph_median_us": statistics.median(samples[name]),
            "graph_min_us": min(samples[name]),
            "graph_max_us": max(samples[name]),
            "graph_samples_us": samples[name],
            "eager_wall_median_us": statistics.median(eager_samples[name]),
        }
        for name in _KERNELS
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--experts", type=int, required=True)
    parser.add_argument("--replicas", type=int, required=True)
    parser.add_argument("--calls", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("top_k", "experts", "replicas", "calls", "repeats"):
        if vars(args)[name] < 1:
            parser.error(f"{name} must be positive")
    if any(tokens < 1 for tokens in args.tokens):
        parser.error("tokens must be positive")

    torch.cuda.set_device(args.device)
    generator = torch.Generator().manual_seed(17)
    replicas = (
        torch.randperm(args.experts * args.replicas, generator=generator)
        .reshape(args.experts, args.replicas)
        .to(device="cuda", dtype=torch.int32)
    )
    counts = torch.randint(
        1, args.replicas + 1, (args.experts,), generator=generator, dtype=torch.int32
    ).to("cuda")
    replicas[torch.arange(args.replicas, device="cuda")[None, :] >= counts[:, None]] = (
        -1
    )
    dispatch = ExpertDispatch(replicas, counts)
    report = {
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tokenspeed_triton": version("tokenspeed-triton"),
        "cutlass_dsl": version("nvidia-cutlass-dsl"),
        "calls_per_graph": args.calls,
        "repeats": args.repeats,
        "experts": args.experts,
        "max_replicas": args.replicas,
        "results": [],
    }
    for dtype in (torch.int32, torch.int64):
        for tokens in args.tokens:
            ids = torch.randint(
                args.experts, (tokens, args.top_k), dtype=dtype, generator=generator
            ).to("cuda")
            timings = _measure(ids, dispatch, args.calls, args.repeats)
            result = {
                "dtype": str(dtype),
                "tokens": tokens,
                "top_k": args.top_k,
                "timings": timings,
            }
            report["results"].append(result)
            medians = " ".join(
                f"{name}: graph={timings[name]['graph_median_us']:.3f}us "
                f"eager={timings[name]['eager_wall_median_us']:.3f}us"
                for name in _KERNELS
            )
            print(f"{dtype} tokens={tokens} top_k={args.top_k} {medians}", flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
