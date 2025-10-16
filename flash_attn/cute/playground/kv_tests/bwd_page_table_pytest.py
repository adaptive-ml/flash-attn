import pytest

import torch
from flash_attn.cute import flash_attn_varlen_func

from kv_utils import (
    clone_like,
    _stats,
    reconstruct_packed_from_paged,
    generate_batched_args,
)

# @pytest.mark.parametrize("batch_size", [1])
# @pytest.mark.parametrize("n_heads", [1])
# @pytest.mark.parametrize("d_head", [64, 128])
# @pytest.mark.parametrize("max_seq_len", [128 * 16])
@pytest.mark.parametrize("batch_size", [1, 7, 16, 53])
@pytest.mark.parametrize("n_heads", [1, 4, 7])
@pytest.mark.parametrize("d_head", [64, 128])
# @pytest.mark.parametrize("d_head", [192]) # Makes launch params illegal?
@pytest.mark.parametrize("max_seq_len", [256, 128 * 16, 128 * 32, 128 * 80, 192 * 80])
@pytest.mark.parametrize("causal", [True, False])
def test_fwd_page_table(
    batch_size: int,
    n_heads: int,
    d_head: int,
    max_seq_len: int,
    causal: bool,

):
    page_size = 128
    if d_head == 128 and not causal:
        page_size = 192

    # Maybe fix at some point...
    if page_size != 128:
        pytest.skip('Page size differs between fwd and bwd')

    (
        q0, k0, v0, 
        qc, kc, vc, 
        page_table, 
        seqused_k, 
        seqused_q, 
        cu_seqlens_q, 
        cu_seqlens_k 
    ) = generate_batched_args(
        batch_size=batch_size, 
        n_heads=n_heads, 
        d_head=d_head, 
        max_seq_len=max_seq_len, 
        page_size=page_size
    )

    out_varlen, lse_varlen = flash_attn_varlen_func(
        q=q0, 
        k=k0, 
        v=v0, 
        cu_seqlens_q=cu_seqlens_q.clone(), 
        cu_seqlens_k=cu_seqlens_k.clone(), 
        causal=causal,
    )

    out_paged, lse_paged = flash_attn_varlen_func(
        q=qc, 
        k=kc, 
        v=vc, 
        cu_seqlens_q=cu_seqlens_q.clone(), 
        seqused_k=seqused_k.clone(),
        page_table=page_table,
        causal=causal,
    )

    # Use the same upstream gradient to compare backward paths
    grad_out = torch.randn_like(out_varlen)

    grad_varlen = clone_like(grad_out)
    grad_paged = clone_like(grad_out)

    # Should be exactly the same...
    fwd_atol=3e-8 
    fwd_rtol=3e-8
    mean_ok_out = _stats("out", out_varlen, out_paged, atol=fwd_atol, rtol=fwd_rtol)
    mean_ok_lse = _stats("lse", lse_varlen, lse_paged, atol=fwd_atol, rtol=fwd_rtol)
    assert mean_ok_out
    assert mean_ok_lse

    # paged bwd
    out_paged.backward(grad_paged, retain_graph=False)
    dq_paged, dk_paged, dv_paged = qc.grad, kc.grad, vc.grad

    # varlen bwd
    out_varlen.backward(grad_varlen, retain_graph=False)
    dq_varlen, dk_varlen, dv_varlen = q0.grad, k0.grad, v0.grad

    # dQ may differ slightly since atomic adds
    atol, rtol = 7e-3, 2e-4 # 1 machine epsilon atol....

    dk_paged_reshaped, dv_paged_reshaped = reconstruct_packed_from_paged(dk_paged, dv_paged, page_table, seqused_k, cu_seqlens_k, page_size)

    mean_ok_dq = _stats("dQ", dq_varlen, dq_paged, atol=atol, rtol=rtol)
    mean_ok_dk = _stats("dK", dk_varlen, dk_paged_reshaped, atol=atol, rtol=rtol)
    mean_ok_dv = _stats("dV", dv_varlen, dv_paged_reshaped, atol=atol, rtol=rtol)
    assert mean_ok_dq
    assert mean_ok_dk
    assert mean_ok_dv

    ok_q = torch.allclose(dq_varlen.float(), dq_paged.float(), atol=atol, rtol=rtol)
    ok_k = torch.allclose(dk_varlen.float(), dk_paged_reshaped.float(), atol=atol, rtol=rtol)
    ok_v = torch.allclose(dv_varlen.float(), dv_paged_reshaped.float(), atol=atol, rtol=rtol)
    print(f"Close? dQ={ok_q}, dK={ok_k}, dV={ok_v}")