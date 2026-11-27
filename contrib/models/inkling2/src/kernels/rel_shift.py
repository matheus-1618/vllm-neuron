# SPDX-License-Identifier: Apache-2.0
"""The relative-position bias band of a prefill chunk as one NKI kernel: each output row is a contiguous slice.

``ops.relative_bias_shift`` places, for every (head, query) row, ``kv_len`` consecutive columns of the shifted logits
``f [rows, E + S]`` (its last column zero) starting at column ``col0 = kv_start - q_start + E - 1``; columns outside
``[0, E + S - 1)`` read the zero column. In torch that is an ``index_select`` along the last dim, which Neuron runs as
an indirect copy per element (software DMA descriptors). Here ``f`` is padded with ``kv_len`` zeros on both sides, so
row ``r`` of the output is the slice ``fp[r, kv_len + col0 : kv_len + col0 + kv_len]``: one indirect DMA per tile of
128 rows, the start of each row's slice given as a flat element offset (one per partition).

Inputs (HBM): ``fp [rows * Wp]`` (flat, bf16, ``Wp = E + S + 2 kv_len``), ``start [rows // 128, 128]`` int32 (flat
element offset of each row's slice, tile-major). Output ``[rows, kv_len]``. Experimental (INKLING2_NKI_SHIFT=1).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

P = 128
CHUNK = 4096          # columns a DMA (8 KB of bf16 a partition)

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def inkling2_rel_shift(fp, start, kv_len: int):
        tiles, _ = start.shape
        total = fp.shape[0]
        out = nl.ndarray((tiles * P, kv_len), dtype=fp.dtype, buffer=nl.shared_hbm)
        view = fp.reshape((total, 1))
        idx = nl.ndarray((P, tiles), dtype=start.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=idx, src=start.ap(pattern=[[1, P], [P, tiles]], offset=0))
        for t in range(tiles):
            for c0 in range(0, kv_len, CHUNK):
                n = min(CHUNK, kv_len - c0)
                at = nl.ndarray((P, 1), dtype=start.dtype, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=at, data=idx[0:P, t:t + 1], op0=nl.add, operand0=c0)
                buf = nl.ndarray((P, CHUNK), dtype=fp.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=buf[:, 0:n],
                              src=view.ap(pattern=[[1, P], [1, n]], offset=0, vector_offset=at[0:P, 0:1],
                                          indirect_dim=0))
                nisa.dma_copy(dst=out[t * P:(t + 1) * P, c0:c0 + n], src=buf[:, 0:n])
        return out
except ImportError:  # pragma: no cover
    inkling2_rel_shift = None


def prepare(f: torch.Tensor, col0: torch.Tensor, kv_len: int):
    """``f [rows, W]`` (W = E + S, last column zero), ``col0 [1]``: the padded flat ``fp`` and the tile-major
    ``start``. Columns of the slice that fall outside ``[0, W - 1)`` must read zeros: the padding of ``kv_len`` on both
    sides holds zeros, and columns ``>= W - 1`` of ``f`` past the band are the zero column or padding."""
    rows, W = f.shape
    fp = F.pad(f, (kv_len, kv_len))                                     # [rows, W + 2 kv_len]
    Wp = W + 2 * kv_len
    # a start left of column 0 reads the left padding; clamp so no slice leaves its own row's padding
    c = (col0.view(1) + kv_len).clamp(0, Wp - kv_len)
    start = (torch.arange(rows, device=f.device, dtype=torch.int64) * Wp + c).to(torch.int32)
    return fp.reshape(-1).contiguous(), start.view(rows // P, P).contiguous()
