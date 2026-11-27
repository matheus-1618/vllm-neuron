# SPDX-License-Identifier: Apache-2.0
"""The prefill MoE over tiles of tokens grouped by expert, as one NKI kernel with one dynamic loop.

The torch prefill MoE (``ops.moe_dense``) multiplies every token of a chunk by all experts (16 times the routed work
at top 8 of 128). The first grouped kernel (``moe_prefill_grouped``) loops over every expert statically and gathers
and scatter-adds each expert's tokens with software DMA descriptors. Here the grouping is done once, in torch:

    xg[r] = x[src[r]]        rows grouped by expert, each expert's rows padded to a multiple of TT (a tile)

and the kernel runs one dynamic loop (``nl.fori_loop``) over the active tiles. Tile ``i`` belongs to expert
``tile_e[i]``; it reads its ``TT`` rows contiguously, the expert's weights with hardware descriptor generation (the
expert id in a register), and writes its ``TT`` output rows (already weighted by the router) contiguously:

    yg[tile] = (r * 4 tanh(g / 4) sigmoid(g) * 25 tanh(u / 25)) @ down[e],   [g | u] = xg[tile] @ gate_up[e]

Each token's output is then the sum of its ``k`` rows of ``yg`` (a gather, in torch). Rows past an expert's count
hold zeros (``x`` row ``T``) and weight 0; tiles past the active count are not run and never read.

Inputs (HBM): ``xg [NT * TT, H]`` bf16, ``wt [NT * TT, 1]`` fp32, ``tile_e [NT, 1]`` int32, ``ntile [1, 1]`` int32,
``gate_up [E, H, 2 I_r]``, ``down [E, I_r, H]`` bf16. Output ``yg [NT * TT, H]`` bf16 (this rank's partial sums).
Experimental, opt-in (``INKLING2_NKI_MOE_PREFILL=tiles``).
"""

from __future__ import annotations

import torch

GATE_CAP, UP_CAP = 4.0, 25.0
P = 128
TT = 32               # rows a tile
N_TILE = 512

try:
    import nki
    import nki.isa as nisa
    import nki.language as nl

    @nki.jit
    def inkling2_moe_prefill_tiles(xg, wt, tile_e, ntile, gate_up, down):
        R, H = xg.shape
        NT = R // TT
        E, _, I2 = gate_up.shape
        I = I2 // 2
        H1 = H // P
        out = nl.ndarray((R, H), dtype=xg.dtype, buffer=nl.shared_hbm)

        nt_sb = nl.ndarray((1, 1), dtype=ntile.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=nt_sb, src=ntile)
        n_reg = nisa.register_alloc()
        nisa.register_load(n_reg, nt_sb[0:1, 0:1])

        gu_view = gate_up.reshape((E, P, H1 * I2))          # partition p: hidden rows p * H1 .. p * H1 + H1 - 1
        xg_v = xg.reshape((NT, TT, H))
        wt_v = wt.reshape((NT, TT, 1))
        te_v = tile_e.reshape((NT, 1, 1))
        out_v = out.reshape((NT, TT, H))
        n_tiles = [(c * N_TILE, min(N_TILE, H - c * N_TILE)) for c in range((H + N_TILE - 1) // N_TILE)]
        i_parts = [(s, min(P, I - s)) for s in range(0, I, P)]

        def tile(ti):
            e_sb = nl.ndarray((1, 1), dtype=tile_e.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=e_sb, src=te_v.select(0, ti))
            e_reg = nisa.register_alloc()
            nisa.register_load(e_reg, e_sb[0:1, 0:1])
            wgu = nl.ndarray((P, H1 * I2), dtype=gate_up.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=wgu, src=gu_view.select(0, e_reg), dge_mode=nisa.dge_mode.hwdge)
            d_e = down.select(0, e_reg)
            wd = []
            for s, sz in i_parts:
                t_ = nl.ndarray((P, H), dtype=down.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=t_[0:sz, :], src=d_e[s:s + sz, :], dge_mode=nisa.dge_mode.hwdge)
                wd.append(t_)
            xs = nl.ndarray((TT, H), dtype=xg.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=xs, src=xg_v.select(0, ti))
            rw = nl.ndarray((TT, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=rw, src=wt_v.select(0, ti))

            # x^T blocks (hidden h = p * H1 + j on partition p, the weights' layout), gate / up in PSUM
            gu_ps = nl.ndarray((TT, I2), dtype=nl.float32, buffer=nl.psum)
            for j in range(H1):
                tp = nl.ndarray((P, TT), dtype=xg.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=xs[:, j:H:H1])
                xT = nl.ndarray((P, TT), dtype=xg.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=xT, src=tp)
                nisa.nc_matmul(dst=gu_ps, stationary=xT, moving=wgu[:, j * I2:(j + 1) * I2], accumulate=(j > 0))
            sig = nl.ndarray((TT, I), dtype=nl.float32, buffer=nl.sbuf)
            g = nl.ndarray((TT, I), dtype=nl.float32, buffer=nl.sbuf)
            u = nl.ndarray((TT, I), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sig, op=nl.sigmoid, data=gu_ps[:, 0:I])
            nisa.activation(dst=g, op=nl.tanh, data=gu_ps[:, 0:I], scale=1.0 / GATE_CAP)
            nisa.tensor_tensor(dst=g, data1=g, data2=sig, op=nl.multiply)
            nisa.activation(dst=u, op=nl.tanh, data=gu_ps[:, I:I2], scale=1.0 / UP_CAP)
            nisa.tensor_scalar(dst=u, data=u, op0=nl.multiply, operand0=GATE_CAP * UP_CAP,
                               op1=nl.multiply, operand1=rw)
            a = nl.ndarray((TT, I), dtype=down.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=a, data1=g, data2=u, op=nl.multiply)
            aT = []
            for s, sz in i_parts:
                tp = nl.ndarray((sz, TT), dtype=down.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=tp, data=a[:, s:s + sz])
                sb = nl.ndarray((sz, TT), dtype=down.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=sb, src=tp)
                aT.append(sb)
            y = nl.ndarray((TT, H), dtype=xg.dtype, buffer=nl.sbuf)
            for n0, nsz in n_tiles:
                ps = nl.ndarray((TT, N_TILE), dtype=nl.float32, buffer=nl.psum)
                for pi, (s, sz) in enumerate(i_parts):
                    nisa.nc_matmul(dst=ps[:, 0:nsz], stationary=aT[pi][0:sz, :], moving=wd[pi][0:sz, n0:n0 + nsz],
                                   accumulate=(pi > 0))
                nisa.tensor_copy(dst=y[:, n0:n0 + nsz], src=ps[:, 0:nsz])
            nisa.dma_copy(dst=out_v.select(0, ti), src=y)

        nl.fori_loop(0, n_reg, tile)
        return out
except ImportError:  # pragma: no cover
    inkling2_moe_prefill_tiles = None


def tables(idx: torch.Tensor, w: torch.Tensor, E: int):
    """From ``idx``, ``w [T, k]``: ``src [NT * TT]`` (token of each grouped row, ``T`` for padding), ``wt [NT * TT, 1]``,
    ``tile_e [NT, 1]`` (``E`` past the active tiles), ``ntile [1, 1]`` and ``pos [T * k]`` (the grouped row of each
    token's slot). ``NT = T k / TT + E`` (each expert wastes at most one partial tile)."""
    T, k = idx.shape
    NT = T * k // TT + E
    flat_e = idx.reshape(-1)
    flat_t = torch.arange(T, device=idx.device).repeat_interleave(k)
    onehot = torch.nn.functional.one_hot(flat_e, E).to(torch.int32)                      # [T k, E]
    rank = (torch.cumsum(onehot, 0) - 1).gather(1, flat_e.view(-1, 1)).view(-1)          # slot within its expert
    tiles_e = (onehot.sum(0) + TT - 1) // TT                                              # [E]
    cum = torch.cumsum(tiles_e, 0)
    pos = (cum - tiles_e)[flat_e] * TT + rank                                             # [T k]
    j = torch.arange(NT, device=idx.device).view(-1, 1)
    tile_e = (cum.view(1, -1) <= j).sum(1).to(torch.int32).view(NT, 1)
    src = torch.full((NT * TT,), T, dtype=torch.int64, device=idx.device).scatter(0, pos, flat_t)
    wt = torch.zeros(NT * TT, dtype=torch.float32, device=idx.device).scatter(0, pos, w.reshape(-1).float())
    ntile = cum[-1:].to(torch.int32).view(1, 1)
    return src, wt.view(-1, 1), tile_e, ntile, pos


def moe_prefill(wrap, x, idx, w, gate_up, down):
    """``x [T, H]`` bf16, ``idx``, ``w [T, k]``; returns this rank's partial ``[T, H]`` fp32."""
    T, k = idx.shape
    src, wt, tile_e, ntile, pos = tables(idx, w, gate_up.shape[0])
    xg = torch.cat([x, x.new_zeros(1, x.shape[1])]).index_select(0, src)
    yg = wrap(inkling2_moe_prefill_tiles)(xg.contiguous(), wt, tile_e, ntile, gate_up, down)
    return yg.index_select(0, pos).view(T, k, -1).float().sum(1)
