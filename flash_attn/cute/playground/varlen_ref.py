from functools import partial
from typing import Optional
import torch
import torch.nn.functional as F

import math
from typing import Optional, Tuple, Literal
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Simple for loop over batch dim implementation
# Constructs bwd graph as well
def torch_flash_ref(
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        cu_seqlens_q: torch.Tensor = None, 
        cu_seqlens_k: torch.Tensor = None, 
        seqused_q: torch.Tensor = None,
        seqused_k: torch.Tensor = None,
        total_q: int = 0,
        total_k: int = 0,
        softmax_scale: Optional[float] = None, 
        causal: bool = False, 
        **kwargs
    ):

    """
    q: (total_q, H, d) if cu_seqlens_q is not None, otherwise (B, L, H, d)
    k: (total_k, H_kv, d) if cu_seqlens_k is not None, otherwise (B, L, H_kv, d)
    v: (total_k, H_kv, d_v) if cu_seqlens_k is not None, otherwise (B, L, H_kv, d_v)
    cu_seqlens_q: (B+1,) int32, cumulative
    cu_seqlens_k: (B+1,) int32, cumulative

    seqused_q: (B+1,) int32
    seqused_k: (B+1,) int32
    Returns:
        out packed like q: (total_q, H, d_v)
    """

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.dim() == 1
        assert total_q == q.shape[0]
        assert q.dim() == 3
        H = q.shape[1]
        B = cu_seqlens_q.shape[0] - 1
    else:
        assert q.dim() == 4
        H = q.shape[2]
        B = q.shape[0]

    if cu_seqlens_k is not None:
        assert cu_seqlens_k.dim() == 1
        assert total_k == k.shape[0] == v.shape[0]
        assert k.dim() == v.dim() == 3
        H_kv = k.shape[1]
        B_kv = cu_seqlens_k.shape[0] - 1
    else:
        assert k.dim() == v.dim() == 4
        assert k.shape[0] == v.shape[0] # batch dims match for k and v
        H_kv = k.shape[2]
        B_kv = k.shape[0]

    d = q.shape[-1]
    d_v = v.shape[-1]

    assert H_kv == v.shape[-2] # H_kv matches for k and v
    assert d == k.shape[-1] # d_head matches for q and k
    assert B == B_kv

    assert q.device == k.device == v.device
    assert q.is_floating_point() and k.is_floating_point() and v.is_floating_point()

    device = q.device
    dtype = q.dtype

    # Asserts to maybe remove at some point (as we add features)
    assert seqused_q is None
    assert seqused_k is None
    assert d == d_v

    hcseq_q = cu_seqlens_q.to(device='cpu')
    hcseq_k = cu_seqlens_k.to(device='cpu')

    outs = []
    for b in range(B):
        if hcseq_q is not None:
            q_start, q_end = int(hcseq_q[b]), int(hcseq_q[b+1])
            qb = q[q_start:q_end]        
        else:
            qb = q[b]

        if hcseq_k is not None:
            k_start, k_end = int(hcseq_k[b]), int(hcseq_k[b+1])
            kb = k[k_start:k_end]
            vb = v[k_start:k_end]
        else:
            kb = k[b]
            vb = v[b]
            
        # Now qb,kb,vb are (roughly) (L, H, d)

        # Reformat to (B=1, H, L, d)
        qb = qb.permute(1, 0, 2).unsqueeze(0)  # (1, H, Lq, d_q)
        kb = kb.permute(1, 0, 2).unsqueeze(0)  # (1, H, Lk, d_k)
        vb = vb.permute(1, 0, 2).unsqueeze(0)  # (1, H, Lk, d_v)

        # PyTorch SDPA: scale overrides default 1/sqrt(d) if provided
        ob = F.scaled_dot_product_attention(
            qb, kb, vb,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
            scale=softmax_scale,
            enable_gqa=H_kv!=H
        )  # (1, H, Lq, d_v)

        # Back to (Lq, H, d_v)
        ob = ob.squeeze(0).permute(1, 0, 2).contiguous()
        outs.append(ob)

    if cu_seqlens_q is not None:
        out = torch.cat(outs, dim=0).to(device=device, dtype=dtype)
    else:
        out = torch.stack(outs, dim=0).to(device=device, dtype=dtype)
    return out

@torch.no_grad()
def _stats(name, a, b, atol, rtol):
    diff = (a - b).float()
    mean_abs = diff.abs().mean().item()
    mean_rel = (diff.abs().mean() / b.abs().clamp_min(1e-6).mean().item())
    print(f"{name}: mean_abs={mean_abs:.4e}, mean_rel={mean_rel:.4e}, sum_fa={a.sum()}, sum_ref={b.sum()}")
    return mean_abs < atol and mean_rel < rtol


def score_mod(sc, score, b, h, q_idx, kv_idx):
    # score is the pre-softmax dot product (already scaled by `scale` inside flex)
    # We apply a smooth cap: sc * tanh(score / sc)
    return torch.tanh(score / sc) * sc



MHAType = Literal["mha", "gqa", "mqa"]

def flex_attn_wrapper(
    q: torch.Tensor, # (B, Hq, Lq, Dh)
    k: torch.Tensor, # (B, Hkv, Lk, Dh)
    v: torch.Tensor, # (B, Hkv, Lk, Dv)
    *,
    causal: bool,
    softmax_scale: Optional[float] = None,
    mha_type: MHAType = "mha", # "mha" | "gqa" | "mqa"
    softcap: Optional[float] = None,
    window: Tuple[int, int] = (-1, -1), # (left, right); (-1,-1) = no window
) -> torch.Tensor:
    """
    Shapes:
      q: (B, Hq, Lq, Dh)
      k: (B, Hkv, Lk, Dh)
      v: (B, Hkv, Lk, Dv)
    Returns:
      out: (B, Hq, Lq, Dv)
    """
    assert q.ndim == k.ndim == v.ndim == 4, "q/k/v must be 4D (B, H, L, D)."
    B, Hq, Lq, Dh = q.shape
    Bk, Hkv, Lk, Dh_k = k.shape
    Bv, Hkv_v, Lk_v, Dv = v.shape
    assert B == Bk == Bv, "Batch size mismatch."
    assert Hkv == Hkv_v and Lk == Lk_v, "K/V head or length mismatch."
    assert Dh == Dh_k, "q and k must share the same head dim (Dh)."
    assert q.device == k.device == v.device, "q/k/v must be on same device."

    if mha_type == "mha":
        enable_gqa = False
        assert Hq == Hkv, "MHA requires Hq == Hkv."
    elif mha_type == "mqa":
        enable_gqa = True
        assert Hkv == 1, "MQA requires Hkv == 1 (KV shared across all query heads)."
    elif mha_type == "gqa":
        enable_gqa = True
        assert Hq % Hkv == 0, "GQA requires Hq to be a multiple of Hkv."
    else:
        raise ValueError("mha_type must be one of {'mha','gqa','mqa'}")

    scale = softmax_scale if softmax_scale is not None else (1.0 / math.sqrt(Dh))

    if softcap == 0.0: softcap = None

    if softcap is not None:
        sc = float(softcap)
        scoremod = partial(score_mod, sc)
    else:
        scoremod = None

    left, right = window
    use_window = not (left == -1 and right == -1)

    if causal or use_window:
        def mask_fn(b, h, q_idx, kv_idx):
            ok = True
            if causal:
                ok = ok and (kv_idx <= q_idx)
            if use_window:
                # If causal is True, we still allow a right window, but causal already restricts kv_idx<=q_idx.
                # Window is relative to q position: [q_idx - left, q_idx + right]
                if left >= 0:
                    ok = ok and (kv_idx >= q_idx - left)
                if right >= 0:
                    ok = ok and (kv_idx <= q_idx + right)
            return ok

        block_mask = create_block_mask(mask_fn, B, Hq, Lq, Lk, device=q.device)
    else:
        block_mask = None # dense

    out = flex_attention(
        q, k, v,
        block_mask=block_mask,
        score_mod=scoremod,
        scale=scale,
        enable_gqa=enable_gqa,
    )
    return out




# Simple for loop over batch dim implementation
# Constructs bwd graph as well
def torch_flex_ref(
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        cu_seqlens_q: torch.Tensor = None, 
        cu_seqlens_k: torch.Tensor = None, 
        seqused_q: torch.Tensor = None,
        seqused_k: torch.Tensor = None,
        total_q: int = 0,
        total_k: int = 0,
        softmax_scale: Optional[float] = None, 
        causal: bool = False, 
        mha_type: str = "mha",
        softcap: Optional[float] = None,
        window: tuple[int, int] = (-1, -1)
    ):

    """
    q: (total_q, H, d) if cu_seqlens_q is not None, otherwise (B, L, H, d)
    k: (total_k, H_kv, d) if cu_seqlens_k is not None, otherwise (B, L, H_kv, d)
    v: (total_k, H_kv, d_v) if cu_seqlens_k is not None, otherwise (B, L, H_kv, d_v)
    cu_seqlens_q: (B+1,) int32, cumulative
    cu_seqlens_k: (B+1,) int32, cumulative

    seqused_q: (B+1,) int32
    seqused_k: (B+1,) int32
    Returns:
        out packed like q: (total_q, H, d_v)
    """

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.dim() == 1
        assert total_q == q.shape[0]
        assert q.dim() == 3
        H = q.shape[1]
        B = cu_seqlens_q.shape[0] - 1
    else:
        assert q.dim() == 4
        H = q.shape[2]
        B = q.shape[0]

    if cu_seqlens_k is not None:
        assert cu_seqlens_k.dim() == 1
        assert total_k == k.shape[0] == v.shape[0]
        assert k.dim() == v.dim() == 3
        H_kv = k.shape[1]
        B_kv = cu_seqlens_k.shape[0] - 1
    else:
        assert k.dim() == v.dim() == 4
        assert k.shape[0] == v.shape[0] # batch dims match for k and v
        H_kv = k.shape[2]
        B_kv = k.shape[0]

    d = q.shape[-1]
    d_v = v.shape[-1]

    assert H_kv == v.shape[-2] # H_kv matches for k and v
    assert d == k.shape[-1] # d_head matches for q and k
    assert B == B_kv

    assert q.device == k.device == v.device
    assert q.is_floating_point() and k.is_floating_point() and v.is_floating_point()

    device = q.device
    dtype = q.dtype

    # Asserts to maybe remove at some point (as we add features)
    assert seqused_q is None
    assert seqused_k is None
    assert d == d_v

    hcseq_q = cu_seqlens_q.to(device='cpu')
    hcseq_k = cu_seqlens_k.to(device='cpu')

    outs = []
    for b in range(B):
        if hcseq_q is not None:
            q_start, q_end = int(hcseq_q[b]), int(hcseq_q[b+1])
            qb = q[q_start:q_end]
        else:
            qb = q[b]

        if hcseq_k is not None:
            k_start, k_end = int(hcseq_k[b]), int(hcseq_k[b+1])
            kb = k[k_start:k_end]
            vb = v[k_start:k_end]
        else:
            kb = k[b]
            vb = v[b]
            
        # Now qb,kb,vb are (roughly) (L, H, d)

        # Reformat to (B=1, H, L, d)
        qb = qb.permute(1, 0, 2).unsqueeze(0)  # (1, H, Lq, d_q)
        kb = kb.permute(1, 0, 2).unsqueeze(0)  # (1, H, Lk, d_k)
        vb = vb.permute(1, 0, 2).unsqueeze(0)  # (1, H, Lk, d_v)

        ob = flex_attn_wrapper(
            qb,              # (B, Hq, Lq, Dh)
            kb,              # (B, Hkv, Lk, Dh)
            vb,              # (B, Hkv, Lk, Dv)
            causal=causal,
            softmax_scale=softmax_scale,
            mha_type=mha_type,
            softcap=softcap,
            window=window,
        )

        # Back to (Lq, H, d_v)
        ob = ob.squeeze(0).permute(1, 0, 2).contiguous()
        outs.append(ob)

    if cu_seqlens_q is not None:
        out = torch.cat(outs, dim=0).to(device=device, dtype=dtype)
    else:
        out = torch.stack(outs, dim=0).to(device=device, dtype=dtype)
    return out

def generate_varlen_args(
    batch_size=8,
    n_heads=16,
    d_head=128,
    min_len=32,
    max_len=64,
    mha_type="mha",
    seqlen_q_eq_kv=True,
    dtype = torch.bfloat16,
): # Need Q, K, V, dO, dPsum, lse_log2, dq_accum, dK, dV, softmax_scale, cu_seqlen_q, cu_seqlen_k

    torch.manual_seed(0)
    device = "cuda"

    assert seqlen_q_eq_kv # For now...

    assert mha_type in ["mha", "mqa", "gqa"]

    lens_q = torch.randint(low=min_len, high=max_len + 1, size=(batch_size,))
    if seqlen_q_eq_kv:
        lens_k = lens_q.clone()
    else:
        lens_k = torch.randint(low=min_len, high=max_len + 1, size=(batch_size,))

    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32), lens_q.cumsum(0)])
    cu_seqlens_k = torch.cat([torch.zeros(1, dtype=torch.int32), lens_k.cumsum(0)])

    total_q = cu_seqlens_q[-1]
    total_k = cu_seqlens_k[-1]
    hcseqk = cu_seqlens_k.clone()
    
    cu_seqlens_q = cu_seqlens_q.contiguous().to(dtype=torch.int32, device=device)
    cu_seqlens_k = cu_seqlens_k.contiguous().to(dtype=torch.int32, device=device)

    if mha_type == "gqa":
        H = 3 * n_heads
        H_kv = n_heads
    elif mha_type == "mha":
        H = H_kv = n_heads
    else: # MQA
        H = n_heads
        H_kv = 1

    d_head_v = d_head

    q = torch.randn(total_q, H, d_head, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_k, H_kv, d_head, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_k, H_kv, d_head_v, device=device, dtype=dtype, requires_grad=True)

    return q, k, v, cu_seqlens_q, cu_seqlens_k, total_q, total_k