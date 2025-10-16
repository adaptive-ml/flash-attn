import torch
from typing import Tuple
from flash_attn.cute import flash_attn_varlen_func, _flash_attn_fwd, _flash_attn_bwd

from kv_utils import (
    ceil_div, 
    clone_like,
    _stats,
    reconstruct_packed_from_paged,
    chunk,
    generate_args,
)

if __name__ == "__main__":
    # Only testing causal for now, don't think causal=False should work
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
    ) = generate_args(
        n_heads=4, 
        d_head=128, 
        seq_len=256,
        dtype=torch.float16,
        page_size=page_size,
        mha_type='gqa',
    )

    n_chunks = 2 # min(3, ceil_div(qc.shape[0], page_size))

    # Use the same upstream gradient to compare backward paths
    # good enough for now since assuming headdim = headdim_v
    grad_out = torch.randn_like(q0)

    grad_varlen = clone_like(grad_out)
    grad_varlen_two = clone_like(grad_out)
    grad_paged = clone_like(grad_out)

    # Varlen Computation
    out_varlen, lse_varlen = flash_attn_varlen_func(
        q=q0, 
        k=k0, 
        v=v0, 
        cu_seqlens_q=cu_seqlens_q.clone(), 
        cu_seqlens_k=cu_seqlens_k.clone(), 
        causal=causal,
    )

    out_varlen.backward(grad_varlen, retain_graph=False)
    dq_varlen, dk_varlen, dv_varlen = q0.grad, k0.grad, v0.grad

    # (Naive) Paged Computation
    out_paged, lse_paged = _flash_attn_fwd(
        q=qc,
        k=kc,
        v=vc,
        cu_seqlens_q=cu_seqlens_q.clone(),
        cu_seqlens_k=None,
        seqused_q=None,
        seqused_k=seqused_k.clone(),
        page_table=page_table,
        softmax_scale=None,
        causal=causal,
        window_size_left=None, 
        window_size_right=None,
        learnable_sink=None,
        softcap=0.0,
        pack_gqa=None,
    )

    dq_paged = torch.empty_like(qc)
    # Views of chunks
    (
        q_chunked, 
        out_chunked, 
        lse_chunked, 
        grad_chunked, 
        dq_chunked
    ) = chunk(
        qc, 
        out_paged, 
        lse_paged, 
        grad_paged, 
        dq_paged, 
        n_chunks,
        page_size,
    )

    print(f"Seq Len = {qc.shape[0]}, n_chunks = {n_chunks}")
    chunk_sizes = [x.shape[0] for x in q_chunked]
    print(f"chunk_sizes: {chunk_sizes}")

    dk_paged = torch.zeros_like(kc)
    dv_paged = torch.zeros_like(vc)
    n_pages = kc.shape[0]
    offset = 0
    total_seq_len = qc.shape[0] # for now...
    seq_len_remaining = total_seq_len
    for chunk_idx in reversed(range(n_chunks)):
        q_cur = q_chunked[chunk_idx]
        out_cur = out_chunked[chunk_idx]
        lse_cur = lse_chunked[chunk_idx]
        grad_cur = grad_chunked[chunk_idx]
        dq_cur = dq_chunked[chunk_idx]
        # Need to restrict seq len but not what we pass into 
        # k_cur, v_cur, dk_cur, dv_cur since pages may be out of order (at least with my arg generation...)
        k_cur = kc 
        v_cur = vc
        dk_cur = dk_paged
        dv_cur = dv_paged
        offset += ceil_div(q_cur.shape[0], page_size)

        cu_seqlens = torch.tensor([0, q_cur.shape[0]], device=q_cur.device, dtype=torch.int32)
        seqused = torch.tensor([seq_len_remaining], device=k_cur.device, dtype=torch.int32)

        # Things to check:
        #   - seqused looks reasonable (remaining seq len)
        #   - cu_seqlens looks reasonable (0, cur seq len q)
        #   - dk_cur, dv_cur

        for name, t in [
            # ("q_cur", q_cur),
            # ("out_cur", out_cur),
            # ("lse_cur", lse_cur),
            # ("grad_cur", grad_cur),
            ("dq_cur", dq_cur),
            # ("k_cur", k_cur),
            # ("v_cur", v_cur),
            ("dk_cur", dk_cur),
            ("dv_cur", dv_cur),
        ]:
            assert t.is_contiguous(), f"{name} is not contiguous..."


        _flash_attn_bwd(
            q=q_cur,
            k=k_cur,    # In later iterations, we refer to pages that may be past num pages that would 
                        # be implied by visible seqlen ks... make sure that is handled correctly
            v=v_cur,
            out=out_cur,
            dout=grad_cur,
            lse=lse_cur,
            softmax_scale=None,
            causal=causal,
            softcap=0.0,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=None,
            seqused_q=None,
            seqused_k=seqused,
            page_table=page_table,
            dq=dq_cur,
            dk=dk_cur, 
            dv=dv_cur,
        )

        seq_len_remaining -= q_cur.shape[0]

    # Should be exactly the same...
    fwd_atol=3e-8 
    fwd_rtol=3e-8
    _stats("out", out_varlen, out_paged, atol=fwd_atol, rtol=fwd_rtol)
    _stats("lse", lse_varlen, lse_paged, atol=fwd_atol, rtol=fwd_rtol)

    # dQ may differ slightly since atomic adds
    atol, rtol = 3e-2, 3e-2

    dk_paged_reshaped, dv_paged_reshaped = reconstruct_packed_from_paged(
        dk_paged, 
        dv_paged, 
        page_table, 
        seqused_k, 
        cu_seqlens_k, 
        page_size
    )

    _stats("dQ", dq_varlen, dq_paged, atol=atol, rtol=rtol)
    _stats("dK", dk_varlen, dk_paged_reshaped, atol=atol, rtol=rtol)
    _stats("dV", dv_varlen, dv_paged_reshaped, atol=atol, rtol=rtol)

    ok_q = torch.allclose(dq_varlen.float(), dq_paged.float(), atol=atol, rtol=rtol)
    ok_k = torch.allclose(dk_varlen.float(), dk_paged_reshaped.float(), atol=atol, rtol=rtol)
    ok_v = torch.allclose(dv_varlen.float(), dv_paged_reshaped.float(), atol=atol, rtol=rtol)
    print(f"Close? dQ={ok_q}, dK={ok_k}, dV={ok_v}")

    import pdb; pdb.set_trace()