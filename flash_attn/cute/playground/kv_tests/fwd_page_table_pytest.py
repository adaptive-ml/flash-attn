import pytest

import torch
from typing import Tuple
from flash_attn.cute import flash_attn_varlen_func

from kv_utils import (
    _stats,
    generate_batched_args,
)

# For some reason, batch_size = 1, n_heads = 1 runs extremely slow?
# @pytest.mark.parametrize("batch_size", [1])
# @pytest.mark.parametrize("n_heads", [1])
# @pytest.mark.parametrize("d_head", [64, 128])
# @pytest.mark.parametrize("max_seq_len", [128 * 16])
@pytest.mark.parametrize("batch_size", [1, 7, 16, 53])
@pytest.mark.parametrize("n_heads", [1, 4, 7])
@pytest.mark.parametrize("d_head", [64, 128])
# @pytest.mark.parametrize("d_head", [192]) # Makes launch params illegal?
@pytest.mark.parametrize("max_seq_len", [128 * 16, 128 * 32, 128 * 80, 192 * 80])
@pytest.mark.parametrize("causal", [True, False])
def test_fwd_page_table(
    batch_size: int,
    n_heads: int,
    d_head: int,
    max_seq_len: int,
    causal: bool,

):
    # Mirroring current n_block_size computation in interface
    page_size = 128
    if d_head == 128 and not causal:
        page_size = 192

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
        page_size=page_size,
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
    mean_ok_out = _stats("out", out_varlen, out_paged, atol=atol, rtol=rtol)
    mean_ok_lse = _stats("lse", lse_varlen, lse_paged, atol=atol, rtol=rtol)
    assert mean_ok_out
    assert mean_ok_lse
