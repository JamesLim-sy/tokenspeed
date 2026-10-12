# MoE ops

Per-op notes for `ops/moe/`. Backend adapters keep their own READMEs
(`flashinfer/README.md`).

## Expert placement dispatch (`dispatch.py`)

With redundant experts (`--ep-num-redundant-experts R`) a logical expert has
several physical replicas across the EP ranks, and routing must emit physical
ids. Under replicated-input EP every rank runs the same routing over the same
all-gathered tokens, so the replica choice has to be a pure function of the
token row and the route rank for exactly one rank to own each (token, expert)
pair: `replicas[logical, (row + rank) % count]`.

`ExpertDispatch` holds one layer's tables (`replicas [num_logical, X]` int32,
`-1` padded to a fixed width; `num_replicas [num_logical]` int32). The runtime
keeps them as views of the placement's device tables so an in-place
rebalance reaches captured graphs without reallocating.

`dispatch_topk_ids(topk_ids, dispatch)` fuses both table lookups and the
modulo into one GPU launch over `[tokens, top_k]`. Every id must be a real expert in
`[0, num_logical)`; the runtime masks zero experts and padded slots around
the call and counts the mapped ids into its load counters afterwards.

The `moe.dispatch` registry provides:

| Kernel | Use |
| --- | --- |
| `cute_dsl_dispatch_topk_ids` | Preferred on supported NVIDIA GPUs when CuTe DSL is installed; TVM-FFI avoids Python tensor conversion on each launch. |
| `triton_dispatch_topk_ids` | Portable NVIDIA/AMD implementation and fallback when CuTe DSL is unavailable. |
| `torch_dispatch_topk_ids` | Original tensor implementation for reference; also used on CPU and for other input dtypes/devices. |

Both fused kernels support int32/int64 ids, non-contiguous and broadcast
views, and empty inputs. They return a contiguous tensor with the original
shape and dtype. Tables and ids must share a device, and each replica count
must be in `[1, X]`. No table contents are copied or cached on the host.
Warm up before CUDA graph capture as for other JIT kernels. Subsequent graph
replays read the current ids, replica entries and counts at the captured
addresses.

Token count, top-k, table width and input strides are runtime values.
Triton disables scalar specialization for these arguments; CuTe uses symbolic
dimensions and strides, with a compilation cache keyed only by dtype and
device. Changing the batch shape does not require another compilation.

For an explicit comparison, set
`TOKENSPEED_KERNEL_OVERRIDE_MOE_DISPATCH=triton_dispatch_topk_ids` or
`TOKENSPEED_KERNEL_OVERRIDE_MOE_DISPATCH=cute_dsl_dispatch_topk_ids`.
The same selection is available in Python without changing the serving API:

```python
from tokenspeed_kernel.selection import kernel_override

with kernel_override("moe", "dispatch", "triton_dispatch_topk_ids"):
    physical_ids = dispatch_topk_ids(logical_ids, dispatch)
```

### Validation and timing

The GPU tests compare exact integer results for both dtypes, partial blocks,
different top-k/table widths and strided views. They also replay captured
graphs after in-place updates and guard against per-batch recompilation.
From the repository root:

```bash
python -m pytest -q tokenspeed-kernel/test/test_moe_dispatch.py \
  tokenspeed-kernel/test/ops/test_moe_dispatch.py \
  tokenspeed-kernel/test/nvidia/ops/moe/test_dispatch_cute_dsl.py

PYTHONPATH=tokenspeed-kernel/python \
  python tokenspeed-kernel/test/nvidia/ops/moe/bench_dispatch.py \
  --tokens 1 16 128 512 4096 16384 --top-k 8 --experts 256 \
  --replicas 4 --calls 64 --repeats 15 --output /tmp/moe_dispatch.json
```

The benchmark compares the original Torch implementation with both fused
implementations, including output allocation. It reports CUDA graph timing
and eager wall time separately, rotating their measurement order. Kernel
selection occurs before timing. These are dispatch-only measurements:
the optimization applies to replica dispatch and does not measure expert
GEMMs, communication, or end-to-end serving throughput.
