import math
import torch
from typing import Optional, Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flash_attn.cute import flash_attn_varlen_func, _flash_attn_fwd
from flash_attn.cute.flash_bwd import FlashAttentionBackwardSm80
from flash_attn.cute.flash_bwd_postprocess import FlashAttentionBackwardPostprocess
from flash_attn.cute.flash_bwd_preprocess import FlashAttentionBackwardPreprocess

from kv_utils import (
    ceil_div, 
    clone_like,
    _stats,
    reconstruct_packed_from_paged,
    chunk,
    generate_args,
)


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
}

def diff_kv_setup(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seqused_k: torch.Tensor,
    page_table: torch.Tensor, # page table for k, v, dk, dv
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    # Maybe remove default args...
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    n_block_size: int = 128,
    pack_gqa: bool = False,
):

    (
        q, k, v, 
        out, dout, lse, 
        cu_seqlens_q, seqused_k,
        page_table, 
        dq, dk, dv,
    ) = [
        maybe_contiguous(t) 
        for t in (
            q, k, v,
            out, dout, lse,
            cu_seqlens_q, seqused_k,
            page_table,
            dq, dk, dv,
        )
    ]

    num_head, head_dim = q.shape[-2:]

    batch_size = cu_seqlens_q.shape[0] - 1
    seqlen_q = None
    total_q = q.shape[0]

    assert page_table.dtype == torch.int32, "page_table must be int32"
    assert page_table.stride(-1) == 1, "page_table must be contiguous in the last dimension"
    max_num_pages_per_seq = page_table.shape[1]
    assert page_table.shape == (batch_size, max_num_pages_per_seq)
    num_pages, page_size = k.shape[:2]
    seqlen_k = total_k = num_pages * page_size

    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]

    assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
    assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)

    assert cu_seqlens_q.shape == (batch_size + 1,), "cu_seqlens_q must have shape (batch_size + 1,)"
    assert out.shape == (total_q, num_head, head_dim_v)
    assert dout.shape == (total_q, num_head, head_dim_v)
    assert lse.shape == (num_head, total_q), "lse must have shape (num_head, total_q)"

    assert q.dtype in [torch.float16, torch.bfloat16], "inputs must be float16 or bfloat16"
    assert q.dtype == k.dtype == v.dtype == out.dtype == dout.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, seqused_k]:
        assert t.dtype == torch.int32, "cu_seqlens_q, cu_seqlens_k must be int32"
    assert lse.dtype == torch.float32, "lse must be float32"
    assert all(
        t is None or t.is_cuda for t in 
        (
            q, k, v, 
            out, dout, lse, 
            cu_seqlens_q, seqused_k, 
            page_table,
            dq, dk, dv,
        )
    ), "inputs must be on CUDA device"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    assert head_dim <= 256, "head_dim must be less than or equal to 256"
    alignment = 16 // q.element_size()
    assert head_dim % alignment == 0, f"head_dim must be divisible by {alignment}"
    assert head_dim_v % alignment == 0, f"head_dim_v must be divisible by {alignment}"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    if dq is None:
        dq = torch.empty_like(q)
    if dk is None:
        dk = torch.zeros_like(k)
    if dv is None:
        dv = torch.zeros_like(v)

    assert causal == True, "Diff KV only works for causal"
    assert pack_gqa == False, "Pack gqa not supported (yet)"
    assert page_size % n_block_size == 0, "Page size {page_size} must be multiple of n_block_size {n_block_size}"

    head_dim_rounded = (head_dim + 32 - 1) // 32 * 32
    head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32

    device = q.device

    if qhead_per_kvhead > 1:
        if page_table is not None:
            dk_accum = torch.zeros(num_pages, num_head_kv, page_size * head_dim_rounded, dtype=torch.float32, device=device)
            dv_accum = torch.zeros(num_pages, num_head_kv, page_size * head_dim_v_rounded, dtype=torch.float32, device=device)
    else:
        dk_accum = dv_accum = None

    dtype = torch2cute_dtype_map[q.dtype]
    
    return (
        dtype, batch_size,
        num_head, num_head_kv,
        seqlen_q, total_q,
        seqlen_k, total_k,
        head_dim, head_dim_v,
        head_dim_rounded, head_dim_v_rounded,
        qhead_per_kvhead,
        num_pages, page_size,
        dq, dk, dv, # sending back since creating if not given
        dk_accum, dv_accum, # used in runner internally, none if qhead_per_kvhead == 1
    )

def get_dlpack_tensors(
    k: torch.Tensor,
    v: torch.Tensor,
    page_table: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    seqused_k: torch.Tensor,
    dk_accum: Optional[torch.Tensor] = None,
    dv_accum: Optional[torch.Tensor] = None,
):

    k_tensor, v_tensor, dk_tensor, dv_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (k, v, dk, dv)
    ]
    dk_accum_tensor, dv_accum_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1) if t is not None else None
        for t in (dk_accum, dv_accum)
    ]
    page_table_tensor = from_dlpack(page_table.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1) if page_table is not None else None
    seqused_k_tensor = from_dlpack(seqused_k.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=seqused_k.ndim-1)

    return (
        k_tensor, v_tensor,
        page_table_tensor,
        dk_tensor, dv_tensor,
        seqused_k_tensor,
        dk_accum_tensor, dv_accum_tensor,
    )

# only accept dl_packed tensors
def flash_diff_kv_bwd_chunk(
    q_chunk_tensor: cute.Tensor,
    k_tensor: cute.Tensor, # Full paged K
    v_tensor: cute.Tensor, # Full paged V
    page_table_tensor: cute.Tensor, # Page table for k, v, dk, dv
    o_tensor: cute.Tensor,
    do_tensor: cute.Tensor,
    lse_tensor: cute.Tensor,
    dq_chunk_tensor: cute.Tensor,
    # Only passed in if mha
    dk_tensor: Optional[cute.Tensor], # Full paged dK
    dv_tensor: Optional[cute.Tensor], # Full paged dV
    # Only passed in if gqa/mqa
    dk_accum_tensor: Optional[cute.Tensor], # Full paged dK
    dv_accum_tensor: Optional[cute.Tensor], # Full paged dV
    dtype, device,
    num_threads,
    num_head, 
    head_dim, head_dim_v,
    head_dim_rounded, head_dim_v_rounded,
    total_q,
    n_block_size,
    m_block_size,
    qhead_per_kvhead,
    seq_len_remaining,
    current_stream: cuda.CUstream,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: float = 0.0,
    pack_gqa: bool = False,
):

    # TODO: dedup
    num_stages_Q: int = 2
    num_stages_dO: int = 2
    SdP_swapAB: bool = False
    dKV_swapAB: bool = False
    dQ_swapAB: bool = False
    AtomLayoutMSdP: int = 2
    AtomLayoutNdKV: int = 2
    AtomLayoutMdQ: int = 2
    V_in_regs: bool = False

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    # For q, varlen format
    cu_seqlens = torch.tensor([0, total_q], device=device, dtype=torch.int32)
    # For k, needed for page table
    seqused = torch.tensor([seq_len_remaining], device=device, dtype=torch.int32)
    cu_seqlens_tensor, seqused_tensor = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=t.ndim-1) if t is not None else None
        for t in (cu_seqlens, seqused)
    ]

    # Not exposed outside --> create here
    total_q_rounded_padded = (total_q + cu_seqlens_q.shape[0] * m_block_size - 1) // m_block_size * m_block_size
    dq_accum = torch.empty(num_head, total_q_rounded_padded * head_dim_rounded, dtype=torch.float32, device=device)
    dpsum = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)
    lse_log2 = torch.empty(num_head, total_q_rounded_padded, dtype=torch.float32, device=device)

    dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
        from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
        for t in (dq_accum, dpsum, lse_log2)
    ]

    seqused_q_tensor = None
    cu_seqlens_k_tensor = None

    # Preprocess kernel: compute (o * dout).sum(dim=-1), lse * log2_e, and zero out dq_accum.
    compile_key_pre = (dtype, head_dim_v, m_block_size, num_threads)
    if compile_key_pre not in flash_diff_kv_bwd_chunk.compile_cache_pre:
        fa_bwd_pre = FlashAttentionBackwardPreprocess(
            dtype, head_dim_v, m_block_size, num_threads=num_threads,
        )
        # TODO: check @can_implement
        flash_diff_kv_bwd_chunk.compile_cache_pre[compile_key_pre] = cute.compile(
            fa_bwd_pre, o_tensor, do_tensor, dpsum_tensor, lse_tensor, lse_log2_tensor,
            dq_accum_tensor, cu_seqlens_tensor, seqused_q_tensor, current_stream
        )
    flash_diff_kv_bwd_chunk.compile_cache_pre[compile_key_pre](
        o_tensor, do_tensor, dpsum_tensor, lse_tensor, lse_log2_tensor, dq_accum_tensor, 
        cu_seqlens_tensor, seqused_q_tensor, current_stream
    )

    # Backward kernel: compute dk, dv, dq_accum.
    # Only need to compile once, but for now just keep in chunk
    compile_key = (
        dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, softcap != 0.0, m_block_size,
        n_block_size, num_threads, pack_gqa, num_stages_Q, num_stages_dO, SdP_swapAB, dKV_swapAB, dQ_swapAB,
        AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs, page_table is None
    )
    if compile_key not in flash_diff_kv_bwd_chunk.compile_cache:
        assert page_size == None or page_size % n_block_size == 0, f"Only page_size values that are multiples of {n_block_size} are supported for paged KV on SM 8.0"
        fa_bwd_sm80 = FlashAttentionBackwardSm80(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead,
            m_block_size,
            n_block_size,
            num_stages_Q,
            num_stages_dO,
            num_threads,
            pack_gqa,
            causal,
            SdP_swapAB,
            dKV_swapAB,
            dQ_swapAB,
            AtomLayoutMSdP,
            AtomLayoutNdKV,
            AtomLayoutMdQ,
            V_in_regs=V_in_regs,
        )
        # TODO: check @can_implement
        flash_diff_kv_bwd_chunk.compile_cache[compile_key] = cute.compile(
            fa_bwd_sm80,
            q_chunk_tensor, k_tensor, v_tensor, do_tensor, lse_log2_tensor, dpsum_tensor,
            dq_accum_tensor,
            dk_tensor if qhead_per_kvhead == 1 else dk_accum_tensor,
            dv_tensor if qhead_per_kvhead == 1 else dv_accum_tensor,
            softmax_scale,
            current_stream,
            cu_seqlens_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_tensor,
            page_table_tensor,
        )

    flash_diff_kv_bwd_chunk.compile_cache[compile_key](
        q_chunk_tensor, k_tensor, v_tensor, do_tensor, lse_log2_tensor, dpsum_tensor,
        dq_accum_tensor,
        dk_tensor if qhead_per_kvhead == 1 else dk_accum_tensor,
        dv_tensor if qhead_per_kvhead == 1 else dv_accum_tensor,
        softmax_scale,
        current_stream,
        cu_seqlens_tensor,
        cu_seqlens_k_tensor,
        seqused_q_tensor,
        seqused_tensor,
        page_table_tensor,
    )

    # Postprocess kernel: convert dq_accum from float32 to dq in bf16/fp16
    compile_key_post = (dtype, head_dim, m_block_size, num_threads, AtomLayoutMdQ, dQ_swapAB)
    if compile_key_post not in flash_diff_kv_bwd_chunk.compile_cache_post:
        fa_bwd_post = FlashAttentionBackwardPostprocess(
            dtype, head_dim, m_block_size, num_threads, AtomLayoutMdQ, dQ_swapAB
        )
        # TODO: check @can_implement
        flash_diff_kv_bwd_chunk.compile_cache_post[compile_key_post] = cute.compile(
            fa_bwd_post, dq_accum_tensor, dq_chunk_tensor, softmax_scale, cu_seqlens_tensor,
            seqused_q_tensor, current_stream
        )
    flash_diff_kv_bwd_chunk.compile_cache_post[compile_key_post](
        dq_accum_tensor, dq_chunk_tensor, softmax_scale, cu_seqlens_tensor, seqused_q_tensor, current_stream
    )

flash_diff_kv_bwd_chunk.compile_cache_pre = {}
flash_diff_kv_bwd_chunk.compile_cache = {}
flash_diff_kv_bwd_chunk.compile_cache_post = {}



# TODO: Can probably be optimized + cleaned up quite a bit
def diff_kv_runner(
    q: torch.Tensor,
    k: torch.Tensor, # Full paged K
    v: torch.Tensor, # Full paged V
    cu_seqlens_q: Optional[torch.Tensor], # Using varlen q format...
    seqused_k: Optional[torch.Tensor], # Needed for page table...
    page_table: torch.Tensor, # Page table for k, v, dk, dv
    out: torch.Tensor,
    dout: torch.Tensor,
    lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor], # Full paged dK
    dv: Optional[torch.Tensor], # Full paged dV
    n_chunks: int,
    softmax_scale: Optional[float] = None,
    causal: bool = True, # Only works with causal=True...
    softcap: float = 0.0, # I don't think this works yet
    pack_gqa: bool = False,
):
    m_block_size: int = 64
    n_block_size: int = 128
    num_threads: int = 256
    dKV_swapAB: bool = False
    AtomLayoutNdKV: int = 2

    device=q.device

    # Get implied params + create necessary torch tensors if they don't exist
    (
        dtype, batch_size,
        num_head, num_head_kv,
        seqlen_q, total_q,
        seqlen_k, total_k,
        head_dim, head_dim_v,
        head_dim_rounded, head_dim_v_rounded,
        qhead_per_kvhead,
        num_pages, page_size,
        dq, dk, dv,
        dk_accum, dv_accum,
    ) = diff_kv_setup(
        q=q, k=k, v=v,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        page_table=page_table,
        out=out, dout=dout,
        lse=lse,
        dq=dq, dk=dk, dv=dv,
        softmax_scale=softmax_scale,
        causal=causal,
        n_block_size=n_block_size,
        pack_gqa=pack_gqa,
    )
        
    # Views of chunks
    (
        q_chunked, 
        out_chunked, 
        lse_chunked, 
        grad_chunked, 
        dq_chunked,
    ) = chunk(
        q, 
        out, 
        lse, 
        dout, 
        dq, 
        n_chunks,
        page_size,
    )

    # Convert everything that needs to be created outside of loop to dlpack
    (
        k_tensor, v_tensor,
        page_table_tensor,
        dk_tensor, dv_tensor,
        seqused_k_tensor,
        dk_accum_tensor, dv_accum_tensor,
    ) = get_dlpack_tensors(
        k=k, v=v,
        page_table=page_table,
        dk=dk, dv=dv,
        seqused_k=seqused_k,
        dk_accum=dk_accum, dv_accum=dv_accum,
    )

    offset = 0
    total_seq_len = q.shape[0] # for now
    seq_len_remaining = total_seq_len

    print(f"Seq Len = {q.shape[0]}, n_chunks = {n_chunks}")
    chunk_sizes = [x.shape[0] for x in q_chunked]
    print(f"chunk_sizes: {chunk_sizes}")

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    for chunk_idx in reversed(range(n_chunks)):
        q_cur = q_chunked[chunk_idx]
        out_cur = out_chunked[chunk_idx]
        lse_cur = lse_chunked[chunk_idx]
        grad_cur = grad_chunked[chunk_idx]
        dq_cur = dq_chunked[chunk_idx]
        offset += ceil_div(q_cur.shape[0], page_size)

        dtype = torch2cute_dtype_map[q.dtype]
        q_chunk_tensor, o_chunk_tensor, do_chunk_tensor, dq_chunk_tensor = [
            from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=t.ndim - 1)
            for t in (q_cur, out_cur, grad_cur, dq_cur)
        ]
        lse_chunk_tensor = from_dlpack(lse_cur.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=lse_cur.ndim - 1)

        # TODO: make some structs to clean this up...
        flash_diff_kv_bwd_chunk(
            q_chunk_tensor=q_chunk_tensor,
            k_tensor=k_tensor, # Full paged K
            v_tensor=v_tensor, # Full paged V
            page_table_tensor=page_table_tensor, # Page table for k, v, dk, dv
            o_tensor=o_chunk_tensor,
            do_tensor=do_chunk_tensor,
            lse_tensor=lse_chunk_tensor,
            dq_chunk_tensor=dq_chunk_tensor,
            # Only passed in if mha
            dk_tensor=dk_tensor, # Full paged dK
            dv_tensor=dv_tensor, # Full paged dV
            # Only passed in if gqa/mqa
            dk_accum_tensor=dk_accum_tensor, # Full paged dK
            dv_accum_tensor=dv_accum_tensor, # Full paged dV
            dtype=dtype, device=device,
            num_threads=num_threads,
            num_head=num_head, 
            head_dim=head_dim, head_dim_v=head_dim_v,
            head_dim_rounded=head_dim_rounded, head_dim_v_rounded=head_dim_v_rounded,
            total_q=q_cur.shape[0],
            n_block_size=n_block_size,
            m_block_size=m_block_size,
            qhead_per_kvhead=qhead_per_kvhead,
            seq_len_remaining=seq_len_remaining,
            current_stream=current_stream,
            softmax_scale=softmax_scale,
            causal=causal,
            softcap=softcap,
            pack_gqa=pack_gqa,
        )

        seq_len_remaining -= q_cur.shape[0]

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    seqused_k_tensor = None
    cu_seqlens_k_tensor = None
    # dk_accum is (n_pages, head_idx_kv, page_size * d_head)
    # can maybe just pretend this is fixed_len with batch_size=n_pages and seq_len = page_size
    # ^seems to work, but note that this touches parts of the last page that are supposed to be unfilled 
    # (don't think it matters though unless we expect those things to be zeroed or something across runs?)
    if qhead_per_kvhead > 1:
        # Postprocess kernel: convert dk_accum & dv_accum from float32 to bf16/fp16
        compile_key_post = (dtype, head_dim, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB)
        if compile_key_post not in diff_kv_runner.compile_cache_post:
            fa_bwd_post = FlashAttentionBackwardPostprocess(
                dtype, head_dim, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB
            )
            # TODO: check @can_implement
            diff_kv_runner.compile_cache_post[compile_key_post] = cute.compile(
                fa_bwd_post, dk_accum_tensor, dk_tensor, softmax_scale, cu_seqlens_k_tensor, seqused_k_tensor, current_stream
            )
        diff_kv_runner.compile_cache_post[compile_key_post](
            dk_accum_tensor, dk_tensor, softmax_scale, cu_seqlens_k_tensor, seqused_k_tensor, current_stream
        )
        compile_key_post = (dtype, head_dim_v, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB)
        if compile_key_post not in diff_kv_runner.compile_cache_post:
            fa_bwd_post = FlashAttentionBackwardPostprocess(
                dtype, head_dim_v, n_block_size, num_threads, AtomLayoutNdKV, dKV_swapAB
            )
            # TODO: check @can_implement
            diff_kv_runner.compile_cache_post[compile_key_post] = cute.compile(
                fa_bwd_post, dv_accum_tensor, dv_tensor, cutlass.Float32(1.0), cu_seqlens_k_tensor, seqused_k_tensor, current_stream
            )
        diff_kv_runner.compile_cache_post[compile_key_post](
            dv_accum_tensor, dv_tensor, cutlass.Float32(1.0), cu_seqlens_k_tensor, seqused_k_tensor, current_stream
        )

diff_kv_runner.compile_cache_post = {}




if __name__ == "__main__":
    # Only testing causal for now, don't think causal=False should work
    causal = True
    mha_type = 'gqa'
    # page_size = 128 if causal else 192
    page_size = 256
    (
        q0, k0, v0, 
        qc, kc, vc, 
        page_table, 
        seqused_k, 
        seqused_q, 
        cu_seqlens_q, 
        cu_seqlens_k 
    ) = generate_args(
        n_heads=8, 
        d_head=128, 
        seq_len=10301,
        dtype=torch.float16,
        page_size=page_size,
        mha_type=mha_type,
    )

    n_chunks = 15 # min(3, ceil_div(qc.shape[0], page_size))

    # Use the same upstream gradient to compare backward paths
    # good enough for now since assuming headdim = headdim_v
    grad_out = torch.randn_like(q0)

    grad_varlen = clone_like(grad_out)
    grad_paged = clone_like(grad_out)

    softmax_scale = None

    # Varlen Computation
    out_varlen, lse_varlen = flash_attn_varlen_func(
        q=q0, 
        k=k0, 
        v=v0, 
        cu_seqlens_q=cu_seqlens_q.clone(), 
        cu_seqlens_k=cu_seqlens_k.clone(), 
        softmax_scale=softmax_scale,
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
        softmax_scale=softmax_scale,
        causal=causal,
        window_size_left=None, 
        window_size_right=None,
        learnable_sink=None,
        softcap=0.0,
        pack_gqa=None,
    )

    dq_paged = torch.empty_like(qc)
    dk_paged = torch.zeros_like(kc)
    dv_paged = torch.zeros_like(vc)

    diff_kv_runner(
        q=qc,
        k=kc, # Full paged K
        v=vc, # Full paged V
        cu_seqlens_q=cu_seqlens_q, # Using varlen q format...
        seqused_k=seqused_k, # Needed for page table...
        page_table=page_table, # Page table for k, v, dk, dv
        out=out_paged,
        dout=grad_paged,
        lse=lse_paged,
        dq=dq_paged,
        dk=dk_paged, # Full paged dK
        dv=dv_paged, # Full paged dV
        n_chunks=n_chunks,
        softmax_scale=softmax_scale,
        causal=causal, 
        softcap = 0.0, # I don't think this works yet
        pack_gqa = False,
    )

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