import torch
from math import sqrt
import flash_attn.cute
import torch.nn.functional as F
from flash_attn.cute import flash_attn_varlen_func, flash_attn_func

def torch_flash_ref(q, k, v, softmax_scale=None, causal=True):
    """
    q: [B, Sq, H, D]
    k: [B, Sk, H_kv, D]
    v: [B, Sk, H_kv, Dv]  (Dv can differ from D)
    returns out_ref [B, Sq, H, Dv] computed by PyTorch FlashAttention backend
    """
    B, Sq, H, D = q.shape
    _, Sk, H_kv, Dk = k.shape
    _, Sk2, H_kv2, Dv = v.shape
    assert Sk == Sk2 and H_kv == H_kv2 and Dk == D, "Incompatible K/V shapes"

    # Expand K/V heads for GQA to match Q heads
    g = H // H_kv
    assert H_kv * g == H, "H must be multiple of H_kv for GQA"
    k_rep = k.repeat_interleave(g, dim=2)        # [B, Sk, H, D]
    v_rep = v.repeat_interleave(g, dim=2)        # [B, Sk, H, Dv]

    # import pdb; pdb.set_trace()

    # SDPA expects [..., L, E]; flatten B and H
    q_ = q.permute(0,2,1,3)# .reshape(B*H, Sq, D)          # [B*H, Sq, D]
    k_ = k_rep.permute(0,2,1,3)# .reshape(B*H, Sk, D)      # [B*H, Sk, D]
    v_ = v_rep.permute(0,2,1,3)# .reshape(B*H, Sk, Dv)     # [B*H, Sk, Dv]

    scale = (1.0 / sqrt(D)) if softmax_scale is None else float(softmax_scale)

    # Force the FlashAttention backend
    with torch.nn.attention.sdpa_kernel(backends=[torch.nn.attention.SDPBackend.FLASH_ATTENTION]):
        out = F.scaled_dot_product_attention(
            q_, k_, v_,
            is_causal=causal,
            scale=scale,
        )  # [B*H, Sq, Dv]

    out = out.reshape(B, H, Sq, Dv).permute(0, 2, 1, 3).contiguous()  # [B, Sq, H, Dv]
    return out.to(q.dtype)


def check_flash_vs_ref(q, k, v, softmax_scale=None, causal=True):
    print("Running flash_attn_func...")
    out, lse = flash_attn_func(
        q.clone(), k.clone(), v.clone(),
        softmax_scale,
        causal,
        (None, None),    # window_size
        None,            # learnable_sink
        0.0,             # softcap
        None             # pack_gqa
    )

    print("Running torch_flash_ref...")
    out_ref = torch_flash_ref(q.clone(), k.clone(), v.clone(), softmax_scale=softmax_scale, causal=causal)

    # import pdb; pdb.set_trace()

    # Compare
    def _stats(a, b, name):
        diff = (a - b).float()
        max_abs = diff.abs().max().item()
        denom = b.abs().clamp_min(1e-6)
        max_rel = (diff.abs() / denom).max().item()
        print(f"{name}: max_abs={max_abs:.4e}, max_rel={max_rel:.4e}")

    _stats(out, out_ref, "OUT  (flash vs ref)")
    # _stats(lse, lse_ref, "LSE  (flash vs ref)")

    # Reasonable tolerances for bf16
    atol_out = 2e-2
    rtol_out = 2e-2
    atol_lse = 2e-2
    rtol_lse = 2e-2

    out_ok = torch.allclose(out.float(), out_ref.float(), atol=atol_out, rtol=rtol_out)
    # lse_ok = torch.allclose(lse.float(), lse_ref.float(), atol=atol_lse, rtol=rtol_lse)
    print(f"Outputs close? {out_ok}")
    # print(f"LSE close? {lse_ok}")
    # return out_ok and lse_ok
    return out_ok


# --- Example usage with your shapes ---
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    # Example dims (can plug in your exact params)
    B = 2
    H = H_kv = 8
    Sq = Sk = 64
    D = Dv = 128

    q = torch.randn(B, Sq, H,   D,  device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, Sk, H_kv, D,  device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, Sk, H_kv, Dv, device=device, dtype=dtype, requires_grad=True)

    ok = check_flash_vs_ref(q, k, v, softmax_scale=1.0 / (D ** 0.5), causal=True)
    print("Match within tolerance:", ok)