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

"""Explicit collective traffic and attention matrix-product cost rules."""

from __future__ import annotations

from dataclasses import dataclass

from tokenspeed_kernel.op_profile import CollectiveTraffic, OpCost, OpRecord


@dataclass(frozen=True)
class CollectiveCost:
    """Describe one rank's synchronous collective using its full-array tensor.

    collective selects all_reduce_sum, reduce_scatter_sum or all_gather.
    tensor and tensor_source (input/output) identify the full input for reduction
    or full output for gathering; in-place output buffers can be input arguments.
    world_size is the participating group size, not the machine's GPU count.
    group must uniquely identify that group within the profile session.
    """

    collective: str
    tensor: str
    tensor_source: str
    world_size: int
    group: str

    def __post_init__(self) -> None:
        CollectiveTraffic(self.collective, 0, self.world_size, self.group)
        if self.tensor_source not in {"input", "output"}:
            raise ValueError("tensor_source must be input or output")

    def __call__(self, op: OpRecord) -> OpCost:
        tensors = op.inputs if self.tensor_source == "input" else op.outputs
        if self.tensor not in tensors:
            raise ValueError(f"Missing collective full-array tensor: {self.tensor}")
        tensor = tensors[self.tensor]
        count = 1
        for dim in tensor.shape:
            count *= dim
        if self.collective != "all_reduce_sum" and count % self.world_size:
            raise ValueError("Full collective array must split evenly across ranks")
        total = tensor.logical_bytes
        read = total // self.world_size if self.collective == "all_gather" else total
        write = (
            total // self.world_size
            if self.collective == "reduce_scatter_sum"
            else total
        )
        return OpCost(
            useful_flops=0,
            executed_flops=None,
            logical_read_bytes=read,
            logical_write_bytes=write,
            estimated_hbm_bytes=None,
            rule="nccl_collective.v1",
            assumptions=(
                "total bytes already include dtype element size",
                "NCCL-tests per-rank normalized bus bytes, not physical link traffic",
                "equal partitions for ReduceScatter and AllGather",
                "reduction arithmetic excluded from model useful FLOPs",
                "collective work must join the timed stream before scope exit",
            ),
            communication=CollectiveTraffic(
                self.collective, total, self.world_size, self.group
            ),
        )


def attention_matmul_flops(
    *,
    batch_size: int,
    query_heads: int,
    query_length: int,
    kv_length: int,
    qk_head_dim: int,
    value_head_dim: int,
    mask: str,
) -> int:
    """Return useful forward QK^T plus PV FLOPs, counting an FMA as two.

    Dimensions describe unpadded, fixed-length sequences. mask explicitly selects
    full, causal_upper_left or causal_lower_right alignment. Scaling, softmax,
    projections, dropout, padding and implementation-specific recompute are excluded.
    """
    dims = (
        batch_size,
        query_heads,
        query_length,
        kv_length,
        qk_head_dim,
        value_head_dim,
    )
    if any(type(dim) is not int or dim < 1 for dim in dims):
        raise ValueError("Attention dimensions must be positive integers")
    q, k = query_length, kv_length
    triangle = min(q, k)
    if mask == "full":
        pairs = q * k
    elif mask == "causal_upper_left":
        pairs = triangle * (triangle + 1) // 2 + (q - triangle) * k
    elif mask == "causal_lower_right":
        pairs = triangle * (2 * k - triangle + 1) // 2
    else:
        raise ValueError(f"Unsupported attention mask: {mask}")
    return 2 * batch_size * query_heads * pairs * (qk_head_dim + value_head_dim)


@dataclass(frozen=True)
class AttentionMatmulCost:
    """Fixed-length attention cost for explicit [batch, heads, sequence, dim] roles.

    q/k/v/output name recorded tensor arguments and return paths. mask has the
    alignment defined by attention_matmul_flops. compute_type is an explicit
    precision/engine key, or None when unavailable. GQA uses the query head count.
    """

    q: str
    k: str
    v: str
    output: str
    mask: str
    compute_type: str | None

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            q, k, v, out = (
                op.inputs[self.q],
                op.inputs[self.k],
                op.inputs[self.v],
                op.outputs[self.output],
            )
        except KeyError as exc:
            raise ValueError(f"Missing attention tensor role: {exc}") from exc
        if any(len(t.shape) != 4 for t in (q, k, v, out)):
            raise ValueError(
                "AttentionMatmulCost requires [batch, heads, sequence, dim] tensors"
            )
        batch, heads, qlen, qdim = q.shape
        kb, kvheads, klen, kdim = k.shape
        vb, vheads, vlen, vdim = v.shape
        if not (
            batch == kb == vb
            and qdim == kdim
            and kvheads == vheads
            and kvheads > 0
            and heads % kvheads == 0
            and klen == vlen
            and out.shape == (batch, heads, qlen, vdim)
        ):
            raise ValueError("Attention tensor shapes do not match")
        flops = attention_matmul_flops(
            batch_size=batch,
            query_heads=heads,
            query_length=qlen,
            kv_length=klen,
            qk_head_dim=qdim,
            value_head_dim=vdim,
            mask=self.mask,
        )
        return OpCost(
            useful_flops=flops,
            executed_flops=None,
            logical_read_bytes=q.logical_bytes + k.logical_bytes + v.logical_bytes,
            logical_write_bytes=out.logical_bytes,
            estimated_hbm_bytes=None,
            rule="attention_matmuls.v1",
            assumptions=(
                f"useful QK^T and PV only; mask={self.mask}",
                "FMA counts as two; GQA uses query heads; QK and V dimensions are separate",
                "softmax, scale, projections, padding and recompute excluded",
                "logical tensor IO does not establish HBM traffic or executed FLOPs",
            ),
            compute_type=self.compute_type,
        )
