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

"""Collect real CUDA work and native MFU/Estimated MBU ranges with one command."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch
from tokenspeed_kernel.op_profile import (
    CostRegistry,
    DenseMatmulCost,
    OpProfile,
    ProfileSession,
    profiled,
)
from tokenspeed_kernel.profile_nvtx import NvtxTimeline, build_nvtx_bridge
from tokenspeed_kernel.profile_utilization import HardwareProfile
from tokenspeed_kernel.registry import register_kernel
from tokenspeed_kernel.signature import format_signatures


@register_kernel(
    "gemm",
    "mm",
    name="example_cuda_profile_mm",
    solution="reference",
    signatures=format_signatures(("a", "b"), "dense", {torch.bfloat16}),
    profile=OpProfile(cost_rule="example_bf16_mm"),
)
def cuda_mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.mm(a, b)


class ExampleLayer(torch.nn.Module):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.register_buffer(
            "up", torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
        )
        self.register_buffer("down", torch.randn_like(self.up))

    @profiled(level="layer")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cuda_mm(cuda_mm(x, self.up), self.down)


def collect(args: argparse.Namespace) -> None:
    hardware = HardwareProfile.from_json(args.hardware)
    layer = ExampleLayer(args.size)
    x = torch.randn_like(layer.up)
    for _ in range(5):
        reference = layer(x)
    torch.cuda.synchronize()
    session = ProfileSession(
        run_id="native-example", model_id="two-gemm-layer", rank=0, backend="nvtx"
    )
    session.bind_layer(layer, layer_id=7)
    costs = CostRegistry()
    dense = DenseMatmulCost(lhs="a", rhs="b", output="return")

    def cost(record):
        tensors = [*record.inputs.values(), *record.outputs.values()]
        if any(t.dtype != "torch.bfloat16" for t in tensors):
            raise ValueError("Example rule requires BF16 GEMM")
        result = dense(record)
        estimated = None
        assumptions = result.assumptions
        if args.hbm_policy == "logical_io_once":
            estimated = result.logical_read_bytes + result.logical_write_bytes
            assumptions += (
                "HBM estimate assumes one operand read and one output write; "
                "cache reuse, reloads and workspace are excluded",
            )
        return replace(
            result,
            compute_type="bf16_tensor",
            estimated_hbm_bytes=estimated,
            assumptions=assumptions,
        )

    costs.register("example_bf16_mm", cost)
    with NvtxTimeline(
        session,
        costs=costs,
        hardware=hardware,
        stream=torch.cuda.current_stream(),
        max_scopes=args.iterations * 3,
        library=args.library,
        output=args.output_dir / "metrics.json",
    ):
        for iteration in range(args.iterations):
            with session.iteration(iter_id=iteration):
                result = layer(x)
    torch.testing.assert_close(result, reference, rtol=0, atol=0)
    print(
        f"Validated {args.iterations} layer calls and {len(session.ops)} real CUDA GEMMs"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nsys", required=True, help="Nsight Systems 2026.5+ executable"
    )
    parser.add_argument("--hardware", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hbm-policy", choices=["unavailable", "logical_io_once"], required=True
    )
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--compiler", default="cc")
    parser.add_argument("--library", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.size < 1 or args.iterations < 1:
        parser.error("size and iterations must be positive")
    if args.library is not None:
        collect(args)
        return

    HardwareProfile.from_json(args.hardware)
    resolved = shutil.which(args.nsys)
    if resolved is None:
        parser.error(f"Nsight executable not found: {args.nsys}")
    nsys = Path(resolved).resolve()
    version = subprocess.check_output([str(nsys), "--version"], text=True)
    match = re.search(r"version\s+(\d+)\.(\d+)", version)
    if match is None or tuple(map(int, match.groups())) < (2026, 5):
        parser.error(
            "Native deferred ranges require the validated Nsight Systems 2026.5+ path"
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bridge = build_nvtx_bridge(
        include_dir=nsys.parent / "nvtx/include",
        output=args.output_dir / "nvtx_profile.so",
        compiler=args.compiler,
    )
    subprocess.run(
        [
            str(nsys),
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            f"--output={args.output_dir / 'timeline'}",
            sys.executable,
            str(Path(__file__).resolve()),
            "--nsys",
            str(nsys),
            "--hardware",
            str(args.hardware.resolve()),
            "--output-dir",
            str(args.output_dir.resolve()),
            "--hbm-policy",
            args.hbm_policy,
            "--size",
            str(args.size),
            "--iterations",
            str(args.iterations),
            "--library",
            str(bridge),
        ],
        check=True,
    )
    report = args.output_dir / "timeline.nsys-rep"
    if not report.is_file() or not (args.output_dir / "metrics.json").is_file():
        raise RuntimeError(
            "Capture did not produce both the native timeline and its metrics"
        )
    print(f"Open {report} in Nsight Systems 2026.5+: Threads > NVTX > TokenSpeed")


if __name__ == "__main__":
    main()
