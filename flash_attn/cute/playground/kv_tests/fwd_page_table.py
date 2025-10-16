import torch
from typing import Tuple
from flash_attn.cute import flash_attn_varlen_func

from kv_utils import (
    _stats,
    generate_batched_args,
)

if __name__ == "__main__":
    causal = False
    page_size = 128 if causal else 192
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
        n_heads=40, 
        d_head=128, 
        max_seq_len=512, 
        page_size=page_size
    )


    out_varlen, lse_varlen = flash_attn_varlen_func(
        q=q0, 
        k=k0, 
        v=v0, 
        cu_seqlens_q=cu_seqlens_q, 
        cu_seqlens_k=cu_seqlens_k, 
        causal=causal,
    )

    out_paged, lse_paged = flash_attn_varlen_func(
        q=qc, 
        k=kc, 
        v=vc, 
        cu_seqlens_q=cu_seqlens_q, 
        seqused_k=seqused_k,
        page_table=page_table,
        causal=causal,
    )

    # Should be exactly the same...
    atol=3e-8 
    rtol=3e-8
    _stats("out", out_varlen, out_paged, atol=atol, rtol=rtol)
    _stats("lse", lse_varlen, lse_paged, atol=atol, rtol=rtol)

    # import pdb; pdb.set_trace()