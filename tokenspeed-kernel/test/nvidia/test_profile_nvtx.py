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

import csv
import json
import os
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.profile_nvtx import build_nvtx_bridge


def test_native_ranges_preserve_thread_nesting_and_gpu_projection(tmp_path):
    nsys = os.environ.get("TOKENSPEED_TEST_NSYS")
    if not nsys or not torch.cuda.is_available():
        pytest.skip("Requires CUDA and TOKENSPEED_TEST_NSYS pointing to Nsight 2026.5+")
    kernel_root = Path(__file__).resolve().parents[2]
    hardware = tmp_path / "hardware.json"
    hardware.write_text(
        json.dumps(
            {
                "name": "analytical test peaks",
                "compute_flops_per_second": {"bf16_tensor": 1e12},
                "hbm_bytes_per_second": 1e12,
                "source": "synthetic capacities for export consistency test",
            }
        )
    )
    output = tmp_path / "capture"
    env = dict(os.environ, PYTHONPATH=str(kernel_root / "python"))
    command = [
        sys.executable,
        str(kernel_root / "test/nvidia/profile_layers_cuda.py"),
        "--nsys",
        nsys,
        "--hardware",
        str(hardware),
        "--output-dir",
        str(output),
        "--hbm-policy",
        "logical_io_once",
        "--size",
        "256",
        "--iterations",
        "2",
    ]
    subprocess.run(
        command, env=env, check=True, capture_output=True, text=True, timeout=180
    )
    report = json.loads((output / "metrics.json").read_text())
    rows = report["device_analysis"]["scopes"]
    assert len(rows) == 6
    assert all(r["mfu"] > 0 and r["estimated_mbu"] > 0 for r in rows)
    projection = output / "projection"
    subprocess.run(
        [
            nsys,
            "stats",
            "--report=nvtx_gpu_proj_trace",
            "--format=csv",
            f"--output={projection}",
            str(output / "timeline.nsys-rep"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with (output / "projection_nvtx_gpu_proj_trace.csv").open() as source:
        projected = {r["Name"].split(":", 1)[-1]: r for r in csv.DictReader(source)}
    with sqlite3.connect(f"file:{output / 'timeline.sqlite'}?mode=ro", uri=True) as db:
        events = db.execute(
            "SELECT start,end,eventType,globalTid,text FROM NVTX_EVENTS WHERE text LIKE '%MFU=%'"
        ).fetchall()
    assert len(events) == 6  # finish must not duplicate ranges.
    assert {e[2] for e in events} == {59}  # Actual thread NVTX push/pop ranges.
    assert len({e[3] for e in events}) == 1
    assert events[0][3] % (1 << 24) == report["device_analysis"]["thread_id"]
    for row in rows:
        projection_row = projected[row["nvtx_label"]]
        assert int(projection_row["NumGPUOps"]) == (1 if row["kind"] == "op" else 2)
        assert int(projection_row["Lvl"]) == (1 if row["kind"] == "op" else 0)
        assert int(projection_row["TID"]) == report["device_analysis"]["thread_id"]
        cost = row["useful_flops"]
        assert row["mfu"] == pytest.approx(
            cost / (1e12 * row["gpu_duration_ns"] * 1e-9)
        )
        assert f"MFU={100 * row['mfu']:.2f}%" in projection_row["Name"]


def test_finalize_once_and_recover_after_workload_exception(tmp_path):
    nsys = os.environ.get("TOKENSPEED_TEST_NSYS")
    if not nsys or not torch.cuda.is_available():
        pytest.skip("Requires CUDA and TOKENSPEED_TEST_NSYS pointing to Nsight 2026.5+")
    bridge = build_nvtx_bridge(
        include_dir=Path(nsys).resolve().parent / "nvtx/include",
        output=tmp_path / "bridge.so",
        compiler="cc",
    )
    script = tmp_path / "workload.py"
    script.write_text(textwrap.dedent("""\
        import json
        import sys
        from dataclasses import replace
        from pathlib import Path
        import torch
        from tokenspeed_kernel.op_profile import (
            ProfileSession, CostRegistry, DenseMatmulCost, OpProfile, wrap_registered_op,
        )
        from tokenspeed_kernel.profile_nvtx import NvtxTimeline
        from tokenspeed_kernel.profile_utilization import HardwareProfile

        root = Path(sys.argv[1])
        def work(a, b):
            if b is None:
                raise ValueError("intentional workload failure")
            return a @ b
        mm = wrap_registered_op(work, family="gemm", mode="mm", name="test",
                                profile=OpProfile(cost_rule="mm"))
        dense = DenseMatmulCost(lhs="a", rhs="b", output="return")
        costs = CostRegistry()
        costs.register("mm", lambda op: replace(dense(op), compute_type="test",
                                                estimated_hbm_bytes=100))
        hardware = HardwareProfile("synthetic", {"test": 1e12}, 1e12, "test")
        a = torch.ones(16, 16, device="cuda")
        reference = mm(a, a)
        for case in ("failed", "valid", "cpu"):
            session = ProfileSession(run_id=case, model_id=case, rank=0, backend="nvtx")
            timeline = NvtxTimeline(session, costs=costs, hardware=hardware,
                stream=torch.cuda.current_stream(), max_scopes=2,
                library=root / "bridge.so", output=root / f"{case}.json")
            try:
                with timeline:
                    with session.iteration(iter_id=0):
                        if case == "failed":
                            mm(a, None)
                        elif case == "cpu":
                            mm(torch.ones(16, 16), torch.ones(16, 16))
                        else:
                            result = mm(a, a)
                    timeline.finish()
                    before = (root / f"{case}.json").stat().st_mtime_ns
                    timeline.finish()
                    assert (root / f"{case}.json").stat().st_mtime_ns == before
            except ValueError as error:
                assert case == "failed" and str(error) == "intentional workload failure"
            else:
                assert case != "failed"
            report = json.loads((root / f"{case}.json").read_text())
            row = report["device_analysis"]["scopes"][0]
            assert (row["mfu"] is not None) == (case == "valid")
            if case == "cpu":
                assert "declared CUDA device" in row["mfu_missing_reason"]
        torch.testing.assert_close(result, reference)
        # A call outside the context still has its original behavior.
        torch.testing.assert_close(mm(a, a), reference)
    """))
    kernel_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(kernel_root / "python"))
    subprocess.run(
        [
            nsys,
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            f"--output={tmp_path / 'recovery'}",
            sys.executable,
            str(script),
            str(tmp_path),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    subprocess.run(
        [
            nsys,
            "export",
            "--type=sqlite",
            f"--output={tmp_path / 'recovery.sqlite'}",
            str(tmp_path / "recovery.nsys-rep"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with sqlite3.connect(
        f"file:{tmp_path / 'recovery.sqlite'}?mode=ro", uri=True
    ) as db:
        labels = [
            r[0]
            for r in db.execute("SELECT text FROM NVTX_EVENTS WHERE text LIKE '%MFU=%'")
        ]
    assert len(labels) == 3
    assert sum("MFU=N/A" in label for label in labels) == 2


def test_real_collective_busbw_in_native_timeline(tmp_path):
    nsys = os.environ.get("TOKENSPEED_TEST_NSYS")
    if not nsys or torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA devices and TOKENSPEED_TEST_NSYS")
    bridge = build_nvtx_bridge(
        include_dir=Path(nsys).resolve().parent / "nvtx/include",
        output=tmp_path / "bridge.so",
        compiler="cc",
    )
    script = tmp_path / "collectives.py"
    script.write_text(textwrap.dedent("""\
        import os
        import sys
        from datetime import timedelta
        from pathlib import Path
        import torch
        import torch.distributed as dist
        from tokenspeed_kernel.op_profile import CostRegistry, OpProfile, ProfileSession, profiled, wrap_registered_op
        from tokenspeed_kernel.profile_costs import CollectiveCost
        from tokenspeed_kernel.profile_nvtx import NvtxTimeline
        from tokenspeed_kernel.profile_utilization import HardwareProfile

        root = Path(sys.argv[1])
        rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", timeout=timedelta(seconds=90), device_id=torch.device("cuda", rank))
        size = dist.get_world_size()
        def reduce(x):
            dist.all_reduce(x, op=dist.ReduceOp.SUM, async_op=True).wait()
            return x
        def scatter(x, out):
            dist.reduce_scatter_tensor(out, x, op=dist.ReduceOp.SUM, async_op=True).wait()
            return out
        def gather(x, out):
            dist.all_gather_into_tensor(out, x, async_op=True).wait()
            return out
        functions = {"all_reduce_sum": reduce, "reduce_scatter_sum": scatter, "all_gather": gather}
        wrapped = {name: wrap_registered_op(fn, family="collective", mode=name, name=name, profile=OpProfile(cost_rule=name)) for name, fn in functions.items()}
        costs = CostRegistry()
        for name in functions:
            costs.register(name, CollectiveCost(name, "return" if name == "all_gather" else "x", "output" if name == "all_gather" else "input", size, "world:0-1"))
        full = torch.full((16384,), rank + 1., device="cuda")
        shard = torch.full((16384 // size,), rank + 1., device="cuda")
        scattered = torch.empty_like(shard)
        gathered = torch.empty_like(full)
        class Layer:
            @profiled(level="layer")
            def forward(self):
                reduced = wrapped["all_reduce_sum"](full.clone())
                wrapped["reduce_scatter_sum"](full, scattered)
                wrapped["all_gather"](shard, gathered)
                return reduced
        layer = Layer()
        for _ in range(3):
            layer.forward()
        torch.cuda.synchronize()
        session = ProfileSession(run_id="collectives", model_id="collective-validation", rank=rank, backend="nvtx")
        session.bind_layer(layer, layer_id=0)
        with NvtxTimeline(session, costs=costs, hardware=HardwareProfile("test", {}, None, "bandwidth needs no peak"), stream=torch.cuda.current_stream(), max_scopes=4, library=root / "bridge.so", output=root / f"metrics.rank{rank}.json"):
            with session.iteration(iter_id=0):
                reduced = layer.forward()
        torch.testing.assert_close(reduced, torch.full_like(full, 3.), rtol=0, atol=0)
        torch.testing.assert_close(scattered, torch.full_like(shard, 3.), rtol=0, atol=0)
        expected = torch.cat([torch.full_like(shard, float(i+1)) for i in range(size)])
        torch.testing.assert_close(gathered, expected, rtol=0, atol=0)
        dist.destroy_process_group()
    """))
    kernel_root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, PYTHONPATH=str(kernel_root / "python"), OMP_NUM_THREADS="1")
    subprocess.run(
        [
            nsys,
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            f"--output={tmp_path / 'collectives'}",
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(script),
            str(tmp_path),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    subprocess.run(
        [
            nsys,
            "stats",
            "--report=nvtx_gpu_proj_trace",
            "--format=csv",
            f"--output={tmp_path / 'projection'}",
            str(tmp_path / "collectives.nsys-rep"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    with (tmp_path / "projection_nvtx_gpu_proj_trace.csv").open() as source:
        projected = {r["Name"].split(":", 1)[-1]: r for r in csv.DictReader(source)}
    for rank in range(2):
        report = json.loads((tmp_path / f"metrics.rank{rank}.json").read_text())
        scopes = report["device_analysis"]["scopes"]
        assert len(scopes) == 4
        for row, expected_bytes in zip(
            scopes, (65536, 32768, 32768, 131072), strict=True
        ):
            assert row["bus_bytes"] == expected_bytes
            assert row["busbw_bytes_per_second"] == pytest.approx(
                expected_bytes / (row["gpu_duration_ns"] * 1e-9)
            )
            assert row["mfu"] is None and row["estimated_mbu"] is None
            label = row["nvtx_label"]
            assert f"busbw={row['busbw_bytes_per_second']/1e9:.2f} GB/s" in label
            assert (
                int(projected[label]["TID"]) == report["device_analysis"]["thread_id"]
            )
            assert int(projected[label]["NumGPUOps"]) >= 1
    with sqlite3.connect(
        f"file:{tmp_path / 'collectives.sqlite'}?mode=ro", uri=True
    ) as db:
        events = db.execute(
            "SELECT eventType, globalTid FROM NVTX_EVENTS WHERE text LIKE '%busbw=%'"
        ).fetchall()
    assert len(events) == 8
    assert {event[0] for event in events} == {59}
    assert len({event[1] for event in events}) == 2
