# Tests that we can read from KV paged and write to dK/dV paged in bwd all with the same page table

import torch
from flash_attn.cute import flash_attn_varlen_func

from kv_utils import (
    clone_like,
    _stats,
    reconstruct_packed_from_paged,
    generate_batched_args,
)

if __name__ == "__main__":
    causal = True
    # page_size = 128 if causal else 192
    page_size = 128
    (
        q0, k0, v0, 
        qc, kc, vc, 
        page_table, 
        seqused_k, 
        seqused_q, 
        cu_seqlens_q, 
        cu_seqlens_k 
    ) = generate_batched_args(
        batch_size=7, 
        n_heads=13, 
        d_head=128, 
        max_seq_len=15 * 192, 
        page_size=page_size,
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
    _stats("out", out_varlen, out_paged, atol=fwd_atol, rtol=fwd_rtol)
    _stats("lse", lse_varlen, lse_paged, atol=fwd_atol, rtol=fwd_rtol)

    # paged bwd
    out_paged.backward(grad_paged, retain_graph=False)
    dq_paged, dk_paged, dv_paged = qc.grad, kc.grad, vc.grad

    # varlen bwd
    out_varlen.backward(grad_varlen, retain_graph=False)
    dq_varlen, dk_varlen, dv_varlen = q0.grad, k0.grad, v0.grad

    # dQ may differ slightly since atomic adds
    atol, rtol = 7e-3, 2e-4 # 1 machine epsilon....

    dk_paged_reshaped, dv_paged_reshaped = reconstruct_packed_from_paged(dk_paged, dv_paged, page_table, seqused_k, cu_seqlens_k, page_size)

    _stats("dQ", dq_varlen, dq_paged, atol=atol, rtol=rtol)
    _stats("dK", dk_varlen, dk_paged_reshaped, atol=atol, rtol=rtol)
    _stats("dV", dv_varlen, dv_paged_reshaped, atol=atol, rtol=rtol)

    ok_q = torch.allclose(dq_varlen.float(), dq_paged.float(), atol=atol, rtol=rtol)
    ok_k = torch.allclose(dk_varlen.float(), dk_paged_reshaped.float(), atol=atol, rtol=rtol)
    ok_v = torch.allclose(dv_varlen.float(), dv_paged_reshaped.float(), atol=atol, rtol=rtol)
    print(f"Close? dQ={ok_q}, dK={ok_k}, dV={ok_v}")

    # import pdb; pdb.set_trace()