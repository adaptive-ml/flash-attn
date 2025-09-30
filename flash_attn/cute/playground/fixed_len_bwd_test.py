import torch
import torch.nn.functional as F
from math import sqrt
from torch.nn.attention import sdpa_kernel, SDPBackend
from flash_attn.cute import flash_attn_func

# ---------- Torch SDPA (FlashAttention backend) ----------
def _sdpa_flash(q, k, v, causal: bool, scale: float):
    """
    Expects 4D tensors [B, H, L, D] and will force the FLASH_ATTENTION backend.
    Raises RuntimeError if FLASH is not available for the given dtype/hardware/shapes.
    """
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(
            q, k, v, is_causal=causal, scale=scale
        )

def torch_flash_fwd(q, k, v, softmax_scale=None, causal=True):
    """
    q: [B, Sq, H, D]
    k: [B, Sk, H_kv, D]
    v: [B, Sk, H_kv, Dv]
    return: out_ref [B, Sq, H, Dv]
    """
    B, Sq, H, D = q.shape
    _, Sk, H_kv, Dk = k.shape
    _, Sk2, H_kv2, Dv = v.shape
    assert Sk == Sk2 and H_kv == H_kv2 and Dk == D, "Incompatible K/V shapes for SDPA"

    # Expand K/V heads for GQA
    g = H // H_kv
    assert g * H_kv == H, "H must be an integer multiple of H_kv for GQA"
    k_rep = k.repeat_interleave(g, dim=2)   # [B, Sk, H, D]
    v_rep = v.repeat_interleave(g, dim=2)   # [B, Sk, H, Dv]

    # Make 4D [B, H, L, D]
    q4 = q.permute(0, 2, 1, 3).contiguous()     # [B,H,Sq,D]
    k4 = k_rep.permute(0, 2, 1, 3).contiguous() # [B,H,Sk,D]
    v4 = v_rep.permute(0, 2, 1, 3).contiguous() # [B,H,Sk,Dv]

    scale = (1.0 / sqrt(D)) if softmax_scale is None else float(softmax_scale)
    out4 = _sdpa_flash(q4, k4, v4, causal=causal, scale=scale)    # [B,H,Sq,Dv]
    return out4.permute(0, 2, 1, 3).contiguous()                  # [B,Sq,H,Dv]


# ---------- Backward checker ----------
@torch.no_grad()
def _stats(name, a, b):
    diff = (a - b).float()
    mean_abs = diff.abs().mean().item()
    mean_rel = (diff.abs().mean() / b.abs().clamp_min(1e-6).mean().item())
    print(f"{name}: mean_abs={mean_abs:.4e}, mean_rel={mean_rel:.4e}, sum_fa={a.sum()}, sum_ref={b.sum()}")

def check_backward_vs_torch_flash(q, k, v, softmax_scale=None, causal=True,
                                  atol=3e-2, rtol=3e-2):
    """
    Compares dQ/dK/dV from your flash_attn_func against PyTorch SDPA (Flash backend).
    Inputs:
      q,k,v: [B, S{q|k}, H{ or H_kv}, D{ or Dv}] with requires_grad=True
    Returns True if all grads are close within tolerances.
    """
    assert q.requires_grad and k.requires_grad and v.requires_grad, "Set requires_grad=True on inputs"

    # Copy inputs for the two runs (so grads don't accumulate on the same Tensor)
    def clone_like(t):
        c = t.clone().detach().requires_grad_(True)
        return c

    q_fa, k_fa, v_fa = map(clone_like, (q, k, v))
    q_t,  k_t,  v_t  = map(clone_like, (q, k, v))

    # --- Forward (your FlashAttention op) ---
    out_fa, lse_fa = flash_attn_func(
        q_fa, k_fa, v_fa,
        (1.0 / q.shape[-1]**0.5) if softmax_scale is None else softmax_scale,
        causal,
        (None, None),    # window_size
        None,            # learnable_sink
        0.0,             # softcap
        None             # pack_gqa
    )

    # --- Forward (Torch SDPA Flash) ---
    out_t = torch_flash_fwd(q_t, k_t, v_t, softmax_scale=softmax_scale, causal=causal)

    # Use the same upstream gradient to compare backward paths
    grad_out = torch.randn_like(out_fa)

    grad_fa = clone_like(grad_out)
    grad_t = clone_like(grad_out)

    _stats("dO", grad_fa, grad_t)
    _stats("O", out_fa, out_t)

    # print(q_fa.grad, k_fa.grad, v_fa.grad, q_t.grad, k_t.grad, v_t.grad)

    # --- Backward through your op ---
    out_fa.backward(grad_fa, retain_graph=False)
    dq_fa, dk_fa, dv_fa = q_fa.grad, k_fa.grad, v_fa.grad

    # --- Backward through Torch SDPA Flash ---
    out_t.backward(grad_t, retain_graph=False)
    dq_t, dk_t, dv_t = q_t.grad, k_t.grad, v_t.grad

    # --- Compare grads ---
    _stats("dQ", dq_fa, dq_t)
    _stats("dK", dk_fa, dk_t)
    _stats("dV", dv_fa, dv_t)

    ok_q = torch.allclose(dq_fa.float(), dq_t.float(), atol=atol, rtol=rtol)
    ok_k = torch.allclose(dk_fa.float(), dk_t.float(), atol=atol, rtol=rtol)
    ok_v = torch.allclose(dv_fa.float(), dv_t.float(), atol=atol, rtol=rtol)
    print(f"Close? dQ={ok_q}, dK={ok_k}, dV={ok_v}")
    return ok_q and ok_k and ok_v


# For testing full bwd pipeline
if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda"

    dtype = torch.bfloat16

    B = 2
    H = H_kv = 8
    Sq = Sk = 512
    D = Dv = 128

    q = torch.randn(B, Sq, H,   D,  device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, Sk, H_kv, D,  device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, Sk, H_kv, Dv, device=device, dtype=dtype, requires_grad=True)

    # doesnt change anything...
    # q_orig = torch.randn(B, H, Sq, D, device=device, dtype=dtype, requires_grad=True)
    # k_orig = torch.randn(B, H_kv, Sk, D, device=device, dtype=dtype, requires_grad=True)
    # v_orig = torch.randn(B, H_kv, Sk, Dv, device=device, dtype=dtype, requires_grad=True)

    # q = q_orig.permute([0, 2, 1, 3])
    # k = k_orig.permute([0, 2, 1, 3])
    # v = v_orig.permute([0, 2, 1, 3])

    causal = True
    ok = check_backward_vs_torch_flash(q, k, v, softmax_scale=1.0/sqrt(D), causal=causal)
    print("Backward match within tolerance:", ok)