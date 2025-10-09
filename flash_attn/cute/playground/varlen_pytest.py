import pytest

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


# @pytest.mark.parametrize("x", [0.5, 1.0, 1.01, 1.5])
# def test_x(x: float):
#     assert x == 1.0



# # @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn])
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
# # @pytest.mark.parametrize("mha_type", ["mha"])
# @pytest.mark.parametrize("has_learnable_sink", [False, True])
# # @pytest.mark.parametrize("has_learnable_sink", [False])
# # @pytest.mark.parametrize("has_qv", [False, True])
# @pytest.mark.parametrize("has_qv", [False])
# # @pytest.mark.parametrize("deterministic", [False, True])
# @pytest.mark.parametrize("deterministic", [False])
# # @pytest.mark.parametrize("softcap", [0.0, 15.0])
# @pytest.mark.parametrize("softcap", [0.0])
# @pytest.mark.parametrize("local", [False, True])
# # @pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("causal", [False, True])
# # @pytest.mark.parametrize("causal", [True])
# # @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# # @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192, 256])
# # @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# # @pytest.mark.parametrize('d', [56, 80])
# # @pytest.mark.parametrize("d", [64, 128, 256])
# # @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128])
# # @pytest.mark.parametrize("d", [64, 96, 128, 192])
# # @pytest.mark.parametrize("d", [64, 128])
# @pytest.mark.parametrize("d", [128, 192])
# # @pytest.mark.parametrize("d", [128])
# @pytest.mark.parametrize(
#     "seqlen_q,seqlen_k",
#     [
#         (1, 1),
#         (64, 128),
#         (128, 192),
#         (256, 256),
#         (239, 1),
#         (799, 3),
#         (113, 203),
#         (113, 128),
#         (128, 217),
#         (113, 211),
#         (108, 256),
#         (256, 512),
#         (384, 256),
#         (640, 128),
#         (512, 256),
#         (1024, 1024),
#         (1023, 1024),
#         (1024, 1023),
#         (4096, 4096),
#         (4224, 4224),
#     ],
# )



@pytest.mark.parametrize("B", [1, 7, 50, 200])
@pytest.mark.parametrize("H", [1, 4, 10])
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("min_seq_len", [1, 32, 256])
@pytest.mark.parametrize("max_seq_len", [8, 64, 1024, 16384])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("softmax_scale", [None, 1.0, 2.0])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("mha_type", ["mha"])
# @pytest.mark.parametrize("softcap", [0.0, 15.0])
def test_varlen(
    B,
    H,
    D,
    min_seq_len,
    max_seq_len,
    causal,
    softmax_scale,
    dtype,
    mha_type,
    # softcap,
    # local,
    # deterministic,
    # has_qv,
    # has_learnable_sink,
):
    sum_seqlen = ((min_seq_len + max_seq_len + 1) // 2) * B * H
    if min_seq_len > max_seq_len:
        pytest.skip("Skipping min_seq_len > max_seq_len")
    
    if sum_seqlen >= 20_000:
        pytest.skip("Skipping seq_len >= 20_000")

    if softmax_scale is not None and (max_seq_len > 64 or D > 64 or H < 5 or B > 7):
        pytest.skip("Pruning softmax_scale tests for numerical stability")

    # if (causal or local) and seqlen_k < seqlen_q:
        # pytest.skip("Causal attention requires seqlen_k >= seqlen_q")

    q, k, v, cu_seqlens_q, cu_seqlens_k, total_q, total_k = generate_varlen_args(
        batch_size=B,
        n_heads=H,
        d_head=D,
        min_len=min_seq_len,
        max_len=max_seq_len,
        mha_type=mha_type,
        seqlen_q_eq_kv=True,
        dtype=dtype
    )

    ok = check_backward_vs_torch_flash(
        q, k, v, 
        cu_seqlens_q, cu_seqlens_k, 
        total_q=total_q, total_k=total_k, 
        softmax_scale=softmax_scale, 
        causal=causal
    )
    assert ok

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
    softcap=0.0,
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
        softcap=softcap,
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

    # _stats("dO", grad_fa, grad_t)
    # _stats("O", out_fa, out_t)

    # Cute bwd
    out_fa.backward(grad_fa, retain_graph=False)
    dq_fa, dk_fa, dv_fa = q_fa.grad, k_fa.grad, v_fa.grad

    # Ref bwd
    out_t.backward(grad_t, retain_graph=False)
    dq_t, dk_t, dv_t = q_t.grad, k_t.grad, v_t.grad

    _stats("dQ", dq_fa, dq_t)
    _stats("dK", dk_fa, dk_t)
    _stats("dV", dv_fa, dv_t)

    ok_q = torch.allclose(dq_fa.float(), dq_t.float(), atol=atol, rtol=rtol)
    ok_k = torch.allclose(dk_fa.float(), dk_t.float(), atol=atol, rtol=rtol)
    ok_v = torch.allclose(dv_fa.float(), dv_t.float(), atol=atol, rtol=rtol)
    # print(f"Close? dQ={ok_q}, dK={ok_k}, dV={ok_v}")
    return ok_q and ok_k and ok_v
