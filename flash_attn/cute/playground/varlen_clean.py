import torch
import torch.nn.functional as F
from math import sqrt
from torch.nn.attention import sdpa_kernel, SDPBackend
from flash_attn.cute import flash_attn_varlen_func

from flash_attn.cute.playground.varlen_ref import (
    torch_flash_ref, 
    _stats, 
    generate_varlen_args,
)

def check_backward_vs_torch_flash(
    q, k, v, 
    cu_seqlens_q=None, 
    cu_seqlens_k=None, 
    seqused_q=None, 
    seqused_k=None, 
    total_q=None, # Only need if varlen
    total_k=None,
    softmax_scale=None, 
    causal=True,
    atol=3e-2, 
    rtol=3e-2,
):
    assert q.requires_grad and k.requires_grad and v.requires_grad, "Set requires_grad=True on inputs"

    # Copy inputs for the two runs (so grads don't accumulate on the same Tensor)
    def clone_like(t):
        c = t.clone().detach().requires_grad_(True)
        return c

    q_fa, k_fa, v_fa = map(clone_like, (q, k, v))
    q_t,  k_t,  v_t  = map(clone_like, (q, k, v))

    if cu_seqlens_q is not None:
        cu_seqlens_q_fa = cu_seqlens_q.clone()
        cu_seqlens_q_t = cu_seqlens_q.clone()
    else:
        cu_seqlens_q_fa = None
        cu_seqlens_q_t = None

    if cu_seqlens_k is not None:
        cu_seqlens_k_fa = cu_seqlens_k.clone()
        cu_seqlens_k_t = cu_seqlens_k.clone()
    else:
        cu_seqlens_k_fa = None
        cu_seqlens_k_t = None

    out_fa, lse_fa = flash_attn_varlen_func(
        q_fa, k_fa, v_fa,
        cu_seqlens_q=cu_seqlens_q_fa,
        cu_seqlens_k=cu_seqlens_k_fa,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        softmax_scale=(1.0 / q.shape[-1]**0.5) if softmax_scale is None else softmax_scale,
        causal=causal,
        window_size=(None, None),
        learnable_sink=None,
        softcap=0.0,
        pack_gqa=None,
    )

    out_t = torch_flash_ref(
        q_t, k_t, v_t, 
        cu_seqlens_q=cu_seqlens_q_t, 
        cu_seqlens_k=cu_seqlens_k_t, 
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        total_q=total_q,
        total_k=total_k,
        softmax_scale=softmax_scale, 
        causal=causal
    )

    # Use the same upstream gradient to compare backward paths
    grad_out = torch.randn_like(out_fa)

    grad_fa = clone_like(grad_out)
    grad_t = clone_like(grad_out)

    _stats("dO", grad_fa, grad_t)
    _stats("O", out_fa, out_t)

    # Cute bwd
    out_fa.backward(grad_fa, retain_graph=False)
    dq_fa, dk_fa, dv_fa = q_fa.grad, k_fa.grad, v_fa.grad

    # Ref bwd
    out_t.backward(grad_t, retain_graph=False)
    dq_t, dk_t, dv_t = q_t.grad, k_t.grad, v_t.grad

    import pdb; pdb.set_trace()

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
    B = 5
    H = 7
    D = 64
    min_len = 256
    max_len = 1024
    causal=True
    softmax_scale=None
    mha_type="mqa"
    # Tests to look at...
    # B     H       D       min_len     max_len     causal      softmax_scale       dtype
    # 7     10      64      1           64          True        2.0                 bfloat16    <--- Look into... probably some to float error?

    q, k, v, cu_seqlens_q, cu_seqlens_k, total_q, total_k = generate_varlen_args(
        batch_size=B,
        n_heads=H,
        d_head=D,
        min_len=min_len,
        max_len=max_len,
        mha_type=mha_type,
        seqlen_q_eq_kv=True
    )

    print(f"{cu_seqlens_k=}")

    ok = check_backward_vs_torch_flash(
        q, k, v, 
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_k=cu_seqlens_k, 
        total_q=total_q, total_k=total_k, 
        softmax_scale=softmax_scale, 
        causal=causal
    )
    print("Backward match within tolerance:", ok)