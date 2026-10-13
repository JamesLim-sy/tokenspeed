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

"""Exercise head-owner DCP merging through AutoBackend and captured NCCL."""

import sys
from datetime import timedelta
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, suite="runtime-2gpu")


def _inputs(rank, degree, dtype, phase):
    generator = torch.Generator().manual_seed(710 + rank)
    output = torch.randn(3, degree * 2, 63, generator=generator).to(dtype)
    lse = torch.randn(3, degree * 2, generator=generator) * 3
    lse[1] = -torch.inf
    output[1] = torch.nan
    if rank == 0:
        lse[0, 0] = torch.inf
        output[0, 0] = torch.nan
    if phase == "empty" or (phase == "concentrated" and rank != 0):
        lse.fill_(-torch.inf)
        output.fill_(torch.nan)
    return output, lse


def _check(actual, rank, degree, dtype, phase, sink, keep_all_heads):
    parts = [_inputs(source, degree, dtype, phase) for source in range(degree)]
    lses = torch.stack([part[1] for part in parts]).double()
    lses = torch.where(torch.isposinf(lses), -torch.inf, lses)
    maximum = lses.max(0).values
    mass = (lses - torch.where(torch.isneginf(maximum), 0, maximum)).exp()
    denominator = mass.sum(0)
    weights = mass / denominator.clamp_min(torch.finfo(torch.float64).tiny)
    outputs = torch.stack([part[0] for part in parts]).double()
    outputs = torch.where(torch.isneginf(lses)[..., None], 0, outputs)
    expected = (outputs * weights[..., None]).sum(0)
    if not keep_all_heads:
        expected = expected[:, rank * 2 : (rank + 1) * 2]
    if sink is not None:
        lse = torch.where(denominator == 0, -torch.inf, maximum + denominator.log())
        lse = lse[:, rank * 2 : (rank + 1) * 2]
        factor = torch.where(torch.isneginf(lse), 0, 1 / (1 + (sink.cpu() - lse).exp()))
        expected *= factor[..., None]
    actual = actual.cpu()
    # Validate the rounded interval implied by strict FP32 arithmetic error.
    # NCCL reductions and owner folds may straddle a 16-bit rounding midpoint.
    tolerance = 2e-6 + 3e-6 * expected.abs()
    lower, upper = (expected - tolerance).to(dtype), (expected + tolerance).to(dtype)
    assert ((actual >= lower) & (actual <= upper)).all()
    assert actual.is_contiguous()


def _worker(rank, degree, rendezvous):
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.comm_backend.registry import set_global_backend
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )
    from tokenspeed.runtime.layers.attention.dcp.comm import combine_attention_partials

    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=degree,
        device_id=device,
        timeout=timedelta(seconds=90),
    )
    graph = None
    try:
        group = tuple(range(degree))
        coordination = dist.new_group(backend="gloo", timeout=timedelta(seconds=90))
        process_group_manager.register_process_group("nccl", group, dist.group.WORLD)
        process_group_manager.register_process_group("gloo", group, coordination)
        backend = AutoBackend()
        backend.configure(use_pynccl=False)
        set_global_backend(backend)
        cases = [
            (torch.bfloat16, False, False),
            (torch.float16, False, False),
            (torch.float32, False, False),
            (torch.bfloat16, True, False),
            (torch.bfloat16, False, True),
        ]
        for dtype, use_sink, keep_all_heads in cases:
            output, lse = [x.to(device) for x in _inputs(rank, degree, dtype, "normal")]
            sink = torch.tensor([-1.0, 0.5], device=device) if use_sink else None

            def combine():
                return combine_attention_partials(
                    output,
                    lse,
                    group=group,
                    rank=rank,
                    sink=sink,
                    keep_all_heads=keep_all_heads,
                )

            for _ in range(2):
                result = combine()
            _check(result, rank, degree, dtype, "normal", sink, keep_all_heads)
            torch.cuda.synchronize()
            dist.barrier(group=coordination)
            graph = torch.cuda.CUDAGraph()
            with mock.patch.object(
                backend.nccl, "all_to_all_single", wraps=backend.nccl.all_to_all_single
            ) as exchange:
                with torch.cuda.graph(graph):
                    result = combine()
            assert exchange.call_count == int(not keep_all_heads)
            for phase in ("empty", "concentrated", "normal"):
                values, masses = _inputs(rank, degree, dtype, phase)
                output.copy_(values)
                lse.copy_(masses)
                torch.cuda.synchronize()
                dist.barrier(group=coordination)
                graph.replay()
                _check(result, rank, degree, dtype, phase, sink, keep_all_heads)
            graph.reset()
            graph = None
        torch.cuda.synchronize()
        dist.barrier(group=coordination)
    finally:
        if graph is not None:
            graph.reset()
        dist.destroy_process_group()


def test_partial_exchange_with_real_collectives_and_graph(tmp_path):
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs")
    mp.spawn(
        _worker, args=(2, f"file://{tmp_path / 'rendezvous'}"), nprocs=2, join=True
    )
