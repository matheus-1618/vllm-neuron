# SPDX-License-Identifier: Apache-2.0
"""``model.PagedIO`` (the paged caches as vLLM's hybrid KV cache manager hands them out) against contiguous caches,
on a small random configuration in fp32 on the CPU.

The paging is simulated as the manager and the Neuron runner do it: every cache layer has its own block table (as if
in its own cache group); a sliding-window layer's blocks that fall out of the window are freed and refilled with
another request's data; in decode a sliding layer gets its table cut to the window (``swa_kv_pos_offset`` = position of
its first key, the runner's ``_compute_swa_decode_tensors``); a decode step pads its second row with slot 0 (the null
block). The conv-state record lives in sliding-window layers too. Needs vLLM and vllm-neuron (model.py's imports).
"""

import types

import pytest
import torch

pytest.importorskip("vllm_neuron")

from inkling2_neuron import ops  # noqa: E402
from inkling2_neuron.config import Inkling2Config  # noqa: E402
from inkling2_neuron import model as M  # noqa: E402

BS = 32            # block size of the test (the server uses 128)
NBLK = 2048        # blocks in the pool


def tiny():
    # one KV head a rank, as on the server (3 query heads)
    return Inkling2Config(hidden_size=256, num_layers=18, vocab_size=1024, unpadded_vocab_size=1000, num_heads=3,
                          num_kv_heads=1, head_dim=64, d_rel=8, rel_extent=64, window=64, log_floor=32,
                          num_experts=8, top_k=2, moe_inter=128, loop_start=6, loop_end=12, passes=3, gate_rank=16,
                          full_layers=(5, 11, 17), tp_size=1)


def params(cfg, g):
    def R(*s, scale=0.05):
        return torch.randn(*s, generator=g) * scale

    H, d, hq, hkv, E, I = cfg.hidden_size, cfg.head_dim, cfg.q_heads_per_rank, cfg.kv_heads_per_rank, \
        cfg.num_experts, cfg.inter_per_rank
    layers = []
    for L in range(cfg.num_layers):
        full = cfg.is_full(L)
        p = dict(wq=R(H, hq * d), q_scale=R(hq, d) + 1, wk=R(H, hkv * d), wv=R(H, hkv * d), wo=R(hq * d, H),
                 wgate=R(H, hq * d), wsink=R(H, hq), wr=R(H, hq * cfg.d_rel), r_scale=R(hq, cfg.d_rel) + 1,
                 r_proj=R(cfg.d_rel, cfg.rel_extent), post_norm=R(H) + 1, mlp_query=R(1, H),
                 post_attn_conv=R(H, 4), post_mlp_conv=R(H, 4), router=R(H, E), router_bias=R(E),
                 gate_up=R(E, H, 2 * I), down=R(E, I, H))
        if L > 0:
            p["attn_query"] = R(3 if full else 1, H)
        if full:
            p.update(k_static=R(4, hkv * d), v_static=R(4, hkv * d), k_wproj=R(H, hkv * 4), v_wproj=R(H, hkv * 4),
                     q_norm=R(H) + 1, k_norm=R(H) + 1, v_norm=R(H) + 1)
        else:
            p.update(k_conv=R(hkv * d, 4), v_conv=R(hkv * d, 4), in_norm=R(H) + 1)
        if cfg.loop_start <= L < cfg.loop_end:
            for n in ("q", "k", "v", "mlp"):
                p[f"gate_{n}"] = (R(2 * H, cfg.gate_rank), R(cfg.gate_rank, H), R(cfg.gate_rank, H))
        layers.append(p)
    return {"layers": layers, "message_query": R(cfg.loop_layers, 4, H), "final_query": R(1, H),
            "norm": R(H) + 1, "embed": R(cfg.vocab_size, H, scale=1.0), "embed_norm": R(H) + 1}


class ContiguousIO(ops.LayerIO):
    def __init__(self, cfg):
        self.cfg, self.k, self.v, self.win = cfg, {}, {}, {}

    def kv(self, c, k, v, window=None):
        if c in self.k:
            k, v = torch.cat([self.k[c], k], 2), torch.cat([self.v[c], v], 2)
        self.k[c], self.v[c] = k, v
        return k, v, torch.arange(k.shape[2]).view(1, -1).expand(k.shape[0], -1)

    def window(self, c, slot):
        if (c, slot) in self.win:
            return self.win[(c, slot)]
        w = self.cfg.kv_heads_per_rank * self.cfg.head_dim if slot < 2 else self.cfg.hidden_size
        return torch.zeros(self.positions.shape[0], ops.CONV_WINDOW, w)

    def put_window(self, c, slot, seq):
        self.win[(c, slot)] = ops.next_window(seq, seq.shape[1] - ops.CONV_WINDOW)


class Pager:
    """Block tables per cache layer for one request, with sliding layers' out-of-window blocks freed."""

    def __init__(self, cfg, names, sliding, g):
        self.cfg, self.names, self.sliding, self.g = cfg, names, sliding, g
        self.tables = {n: [] for n in names}
        self.orig = {}              # freed blocks' ids (still in the prefix cache unless reused)
        self.free = list(torch.randperm(NBLK - 1, generator=g).add(1).tolist())   # block 0 is the null block

    def grow(self, n_tokens):
        for n in self.names:
            while len(self.tables[n]) * BS < n_tokens:
                self.tables[n].append(self.free.pop())

    def release(self, n_computed, caches, reuse=True):
        """vLLM frees a sliding layer's blocks whose tokens all precede ``n_computed - window + 1``; with ``reuse``
        another request then writes them (here: noise), else they stay in the prefix cache as they are."""
        keep_from = max(0, n_computed - self.cfg.window + 1)
        for n in self.names:
            if not self.sliding[n]:
                continue
            for i in range(keep_from // BS):
                b = self.tables[n][i]
                if b:
                    if reuse:
                        for t in caches[n]:
                            t[b].normal_(generator=self.g)
                        self.free.append(b)       # reused by the next allocation (any group)
                    self.orig.setdefault(n, {})[i] = b
                    self.tables[n][i] = 0


def metadata(pager, rows, positions, S, decode, n=None):
    """attn_metadata for ``rows`` (a list of (pager or None for a padding row)) at ``positions [B]``; tokens past
    ``n`` of a prefill chunk are padding (slot 0)."""
    n = S if n is None else n
    md = {}
    for name_ in pager.names:
        width = max(len(p.tables[name_]) for p in rows if p is not None)
        if decode and pager.sliding[name_]:
            nb = pager.cfg.window // BS + 1
        table, slots, offs = [], [], []
        for b, p in enumerate(rows):
            q = int(positions[b])
            if p is None:
                row, sl, off = [-1] * (nb if decode and pager.sliding[name_] else width), [0] * S, 0
            else:
                full = p.tables[name_] + [-1] * (width - len(p.tables[name_]))
                sl = [p.tables[name_][(q + s) // BS] * BS + (q + s) % BS if s < n else 0 for s in range(S)]
                if decode and pager.sliding[name_]:
                    start = max(0, q // BS - nb + 1)
                    row = [full[start + i] if start + i < len(full) else -1 for i in range(nb)]
                    off = start * BS
                else:
                    row, off = full, 0
            table.append(row)
            slots += sl
            offs.append(off)
        md[name_] = {"block_table_tensor": torch.tensor(table), "slot_mapping": torch.tensor(slots)}
        if decode and pager.sliding[name_]:
            md[name_]["swa_kv_pos_offset"] = torch.tensor(offs, dtype=torch.int32)
    return md


def model_stub(cfg, caches, names):
    m = types.SimpleNamespace(config=cfg, moe_kernel=False, attn_kernel=False, moe_prefill_kernel=False,
                              block_end_records=True,
                              tp_group=types.SimpleNamespace(world_size=1))
    m.state_layout = M.StateLayout(cfg, BS)
    shared = {}

    def flat(n):
        k, v = caches[n]
        return M.Inkling2ForCausalLM._flat_kv(k, v, shared), tuple(k.shape)

    m._kv_flat = [flat(M.cache_name(c)) for c in range(cfg.cache_layers)]
    m._state_flat = [flat(M.state_name(j)) for j in range(M.STATE_LAYERS)]
    return m


def logits(prm, cfg, ids, io, dense):
    h = ops.rms_norm(prm["embed"][ids], prm["embed_norm"], cfg.eps, dtype=torch.float32)
    return ops.text_model(prm, cfg, h, io, dense)


def hma_layout(cfg, names, sliding, g):
    """As vLLM's hybrid manager lays the caches out: groups of ``n_full`` layers (the full layers one group, the
    sliding ones in order), one block table a group, and layers at one position of different groups sharing a
    buffer. Returns (caches, pager) with each name's table the group's (one list object)."""
    full = [n for n in names if not sliding[n]]
    slid = [n for n in names if sliding[n]]
    gs = len(full)
    groups = [full] + [slid[i:i + gs] for i in range(0, len(slid), gs)]
    d = cfg.head_dim
    bufs = [torch.zeros(2, NBLK, 1, BS, d) for _ in range(gs)]
    caches = {n: tuple(bufs[i].unbind(0)) for grp in groups for i, n in enumerate(grp)}
    pager = Pager(cfg, names, sliding, g)
    for grp in groups:
        t = []
        for n in grp:
            pager.tables[n] = t
    return caches, pager


@pytest.mark.parametrize("pad_row,hma", [(False, False), (True, False), (False, True)])
def test_paged_io_matches_contiguous(pad_row, hma, monkeypatch):
    monkeypatch.setattr(M, "DTYPE", torch.float32)     # the state record in fp32, as the contiguous reference keeps it
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    cfg = tiny()
    prm = params(cfg, g)
    names = [M.cache_name(c) for c in range(cfg.cache_layers)] + [M.state_name(j) for j in range(M.STATE_LAYERS)]
    sliding = {M.cache_name(c): not cfg.is_full(cfg.cache_layer_model(c)) for c in range(cfg.cache_layers)}
    sliding.update({M.state_name(j): True for j in range(M.STATE_LAYERS)})
    d = cfg.head_dim
    if hma:
        caches, pager = hma_layout(cfg, names, sliding, g)
    else:
        caches = {n: tuple(torch.zeros(2, NBLK, 1, BS, d).unbind(0)) for n in names}
    stub = model_stub(cfg, caches, names)
    ref = ContiguousIO(cfg)
    T, chunk, n_dec = 160, 32, 24          # 5 prefill chunks (2.5 windows), then decode steps
    ids = torch.randint(0, 1000, (1, T + n_dec), generator=g)
    if not hma:
        pager = Pager(cfg, names, sliding, g)
    out_p, out_r = [], []
    with torch.no_grad():
        for t0 in range(0, T, chunk):
            pager.grow(t0 + chunk)
            pos = torch.tensor([t0])
            io = M.PagedIO(stub, metadata(pager, [pager], pos, chunk, False), (pos.view(1, 1) + torch.arange(chunk)),
                           torch.tensor([chunk]), (torch.tensor([[t0]]) > 0), 1, chunk)
            out_p.append(logits(prm, cfg, ids[:, t0:t0 + chunk], io, True)); io.publish()
            ref.positions = pos.view(1, 1) + torch.arange(chunk)
            out_r.append(logits(prm, cfg, ids[:, t0:t0 + chunk], ref, True))
            pager.release(t0 + chunk, caches)
        for t in range(T, T + n_dec):
            pager.grow(t + 1)
            rows = [pager, None] if pad_row else [pager]
            pos = torch.tensor([t] + ([0] if pad_row else []))
            B = len(rows)
            tok = ids[:, t:t + 1].expand(B, 1)
            io = M.PagedIO(stub, metadata(pager, rows, pos, 1, True), pos.view(B, 1), 1, None, B, 1)
            out_p.append(logits(prm, cfg, tok, io, False)[:1]); io.publish()
            ref.positions = torch.tensor([[t]])
            out_r.append(logits(prm, cfg, ids[:, t:t + 1], ref, False))
            pager.release(t + 1, caches)
    a, b = torch.cat(out_p, 1), torch.cat(out_r, 1)
    dmax = (a - b).abs().max().item()
    assert torch.isfinite(a).all() and dmax < 1e-3 * max(1.0, b.abs().max().item()), dmax


def test_prefix_hit_matches_contiguous(monkeypatch):
    """Request B shares request A's first 3 blocks (a prefix-cache hit; A's sliding blocks out of the window already
    freed) while A went on to 128, and continues from position 96: its logits match a contiguous run of B's whole sequence. This needs the
    state record at the end of every block a prefill chunk fills."""
    monkeypatch.setattr(M, "DTYPE", torch.float32)
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(2)
    cfg = tiny()
    prm = params(cfg, g)
    names = [M.cache_name(c) for c in range(cfg.cache_layers)] + [M.state_name(j) for j in range(M.STATE_LAYERS)]
    sliding = {M.cache_name(c): not cfg.is_full(cfg.cache_layer_model(c)) for c in range(cfg.cache_layers)}
    sliding.update({M.state_name(j): True for j in range(M.STATE_LAYERS)})
    d = cfg.head_dim
    caches = {n: tuple(torch.zeros(2, NBLK, 1, BS, d).unbind(0)) for n in names}
    stub = model_stub(cfg, caches, names)
    P, chunk, T, n_dec = 96, 64, 160, 8
    ids = torch.randint(0, 1000, (1, T + n_dec), generator=g)
    a = Pager(cfg, names, sliding, g)

    def prefill(pager, t0, n, reuse=True):
        pager.grow(t0 + n)
        pos = torch.tensor([t0])
        S = chunk
        io = M.PagedIO(stub, metadata(pager, [pager], pos, S, False, n), pos.view(1, 1) + torch.arange(S),
                       torch.tensor([n]), (torch.tensor([[t0]]) > 0), 1, S)
        tok = ids[:, t0:t0 + S] if t0 + S <= ids.shape[1] else None
        out = logits(prm, cfg, tok, io, True)
        io.publish()
        pager.release(t0 + n, caches, reuse)
        return out[:, :n]

    with torch.no_grad():
        prefill(a, 0, 64, reuse=False)
        prefill(a, 64, 64, reuse=False)   # block 2 ends inside this chunk (position 95): only its block-end record has it
        b = Pager(cfg, names, sliding, g)
        b.free = a.free
        first_needed = max(0, P - cfg.window + 1) // BS      # B's sliding layers need the blocks of its window only
        for n in names:
            ids_ = [a.orig.get(n, {}).get(i, a.tables[n][i]) for i in range(P // BS)]
            b.tables[n] = [0 if sliding[n] and i < first_needed else x for i, x in enumerate(ids_)]
        out_b = [prefill(b, P, 64)]
        for t in range(P + 64, T + n_dec):
            b.grow(t + 1)
            pos = torch.tensor([t])
            io = M.PagedIO(stub, metadata(b, [b], pos, 1, True), pos.view(1, 1), 1, None, 1, 1)
            out_b.append(logits(prm, cfg, ids[:, t:t + 1], io, False)); io.publish()
            b.release(t + 1, caches)
        ref = ContiguousIO(cfg)
        ref.positions = torch.arange(T + n_dec).view(1, -1)
        want = logits(prm, cfg, ids, ref, True)[:, P:]
    got = torch.cat(out_b, 1)
    dmax = (got - want).abs().max().item()
    assert torch.isfinite(got).all() and dmax < 1e-3 * max(1.0, want.abs().max().item()), dmax


def test_layers_sharing_a_buffer_get_one_flat_tensor():
    """With the hybrid KV cache manager, layers of different groups share a buffer: they must get the same flat
    tensor object (distinct overlapping views would become a synthetic base written back in full on every write)."""
    buf = torch.zeros(2 * 4 * 1 * 8 * 16)
    k, v = buf[:512].view(4, 1, 8, 16), buf[512:].view(4, 1, 8, 16)
    k2, v2 = buf[:512].view(4, 1, 8, 16), buf[512:].view(4, 1, 8, 16)     # another layer's views of the same buffer
    shared = {}
    a = M.Inkling2ForCausalLM._flat_kv(k, v, shared)
    b = M.Inkling2ForCausalLM._flat_kv(k2, v2, shared)
    assert a is b and a.numel() == buf.numel()
    a.view(2, 4, 1, 8, 16)[0, 1, 0, 2, 3] = 5.0
    assert k[1, 0, 2, 3] == 5.0
