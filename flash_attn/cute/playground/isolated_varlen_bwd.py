import torch
import torch.nn.functional as F
import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from flash_attn.cute import _flash_attn_bwd, FlashAttentionBackwardPreprocess, FlashAttentionBackwardSm80, utils

def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

# Generate lengths
def generate_varlen_main_bwd_args(
    m_block_size,
    batch_size=8,
    n_heads=16,
    d_head=128,
    min_len=32,
    max_len=64,
    seqlen_q_eq_kv=True,
    softmax_scale=1.0 # TODO: Test changing this at some point
): # Need Q, K, V, dO, dPsum, lse_log2, dq_accum, dK, dV, softmax_scale, cu_seqlen_q, cu_seqlen_k

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    assert seqlen_q_eq_kv # For now...

    # Figure out which parts of the kernel/computations need to be changed with cuseqlens

    lens_q = torch.randint(low=min_len, high=max_len + 1, size=(batch_size,))
    if seqlen_q_eq_kv:
        lens_k = lens_q.clone()
    else:
        lens_k = torch.randint(low=min_len, high=max_len + 1, size=(batch_size,))
    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32), lens_q.cumsum(0)])
    cu_seqlens_k = torch.cat([torch.zeros(1, dtype=torch.int32), lens_k.cumsum(0)])

    total_q = cu_seqlens_q[-1]
    total_k = cu_seqlens_k[-1]

    # Now cu_seqlens_q and cu_seqlens_k exist

    H = H_kv = n_heads
    d_head_v = d_head

    q = torch.randn(total_q, H, d_head, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(total_k, H_kv, d_head, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(total_k, H_kv, d_head_v, device=device, dtype=dtype, requires_grad=True)

    total_q_rounded = (total_q + m_block_size - 1) // m_block_size * m_block_size

    dout = torch.randn(total_q, H, d_head_v, device=device, dtype=dtype, requires_grad=True)
    dpsum = torch.randn(H, total_q_rounded, device=device, dtype=torch.float32, requires_grad=True)

    lse_log2 = torch.ones(H, total_q_rounded, device=device, dtype=torch.float32, requires_grad=True)
    dq_accum = torch.zeros(H, total_q_rounded * d_head, device=device, dtype=torch.float32, requires_grad=True)
    dk = torch.empty(total_k, H_kv, d_head, device=device, dtype=dtype, requires_grad=True)
    dv = torch.empty(total_k, H_kv, d_head, device=device, dtype=dtype, requires_grad=True)

    return q, k, v, dout, dpsum, lse_log2, dq_accum, dk, dv, softmax_scale, cu_seqlens_k, cu_seqlens_q


# Generate lengths
def generate_varlen_bwd_preprocess_args(
    m_block_size,
    batch_size=8,
    n_heads=16,
    d_head=128,
    min_len=32,
    max_len=64,
    seqlen_q_eq_kv=True 
): # Only need O, dO, dpsum, lse, lse_log2, dq_accum, cu_seqlens_q
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    assert seqlen_q_eq_kv # For now...

    lens_q = torch.randint(low=min_len, high=max_len + 1, size=(batch_size,))
    if seqlen_q_eq_kv:
        lens_k = lens_q.clone()
    else:
        lens_k = torch.randint(low=min_len, high=max_len + 1, size=(batch_size,))
    cu_seqlens_q = torch.cat([torch.zeros(1, dtype=torch.int32), lens_q.cumsum(0)]).contiguous().to(dtype=torch.int32, device=device)
    cu_seqlens_k = torch.cat([torch.zeros(1, dtype=torch.int32), lens_k.cumsum(0)]).contiguous().to(dtype=torch.int32, device=device)

    total_q = cu_seqlens_q[-1]
    total_k = cu_seqlens_k[-1]

    H = H_kv = n_heads
    d_head_v = d_head

    lse = torch.ones(H, total_q, device=device, dtype=torch.float32, requires_grad=True)
    out = torch.randn(total_q, H, d_head_v, device=device, dtype=dtype, requires_grad=True)
    dout = torch.randn(total_q, H, d_head_v, device=device, dtype=dtype, requires_grad=True)

    total_q_rounded = (total_q + m_block_size - 1) // m_block_size * m_block_size
    lse_log2 = torch.empty(H, total_q_rounded, device=device, dtype=torch.float32, requires_grad=True)
    dq_accum = torch.empty(H, total_q_rounded * d_head, device=device, dtype=torch.float32, requires_grad=True)
    dpsum = torch.empty(H, total_q_rounded, device=device, dtype=torch.float32, requires_grad=True)

    return total_q, out, dout, dpsum, lse, lse_log2, dq_accum


# Removes dependence on interface.py
def preprocess_caller():
    # 1. Torch setup (maybe split out)

    m_block_size = 64
    num_threads = 256

    torch2cute_dtype_map = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }

    batch_size = 8
    num_heads = 16
    head_dim = 128

    min_seq_len = 32
    max_seq_len = 64

    seqlen_q_eq_kv = True



    total_q, out, dout, dpsum, lse, lse_log2, dq_accum = generate_varlen_bwd_preprocess_args(m_block_size, batch_size, num_heads, head_dim, min_seq_len, max_seq_len, seqlen_q_eq_kv)

    softmax_scale = 1.0 / math.sqrt(head_dim)

    # 4. torch to dlpack
    dtype = torch2cute_dtype_map[out.dtype]


    o_tensor, do_tensor = [
        utils.convert_from_dlpack(t.detach(), leading_dim=t.ndim - 1, alignment=16, divisibility=8)
        for t in (out, dout)
    ]

    lse_tensor = from_dlpack(lse.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=lse.ndim - 1)
    dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
        utils.convert_from_dlpack(t.detach(), leading_dim=t.ndim - 1, alignment=16, divisibility=4)
        for t in (dq_accum, dpsum, lse_log2)
    ]

    # 5. Kernel compilation
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    is_varlen_preprocess = True

    # Preprocess kernel: compute (o * dout).sum(dim=-1), lse * log2_e, and zero out dq_accum.
    compile_key_pre = (dtype, head_dim, m_block_size, num_threads)
    if compile_key_pre not in _flash_attn_bwd.compile_cache_pre:
        fa_bwd_pre = FlashAttentionBackwardPreprocess(
            dtype, head_dim, m_block_size, num_threads=num_threads
        )
        # TODO: check @can_implement
        _flash_attn_bwd.compile_cache_pre[compile_key_pre] = cute.compile(
            fa_bwd_pre, o_tensor, do_tensor, dpsum_tensor, lse_tensor, lse_log2_tensor,
            dq_accum_tensor, current_stream
        )
    # 6. Kernel Call
    _flash_attn_bwd.compile_cache_pre[compile_key_pre](
        o_tensor, do_tensor, dpsum_tensor, lse_tensor, lse_log2_tensor, dq_accum_tensor, current_stream
    )

    cuda.cuStreamSynchronize(current_stream)

    # 7. Verification

    dpsum_ref = out * dout
    dpsum_ref = dpsum_ref.sum(dim=-1)
    dpsum_ref = dpsum_ref.permute([1, 0])

    LOG2_E = math.log2(math.e)
    lse_log2_ref = lse * LOG2_E
    dq_accum_ref = torch.zeros_like(dq_accum)


    # _stats('dq_accum', dq_accum, dq_accum_ref)

    _stats('lse_log2', lse_log2[:, : total_q], lse_log2_ref)
    _stats('dpsum', dpsum[:, : total_q], dpsum_ref)
    print(f"dq_accum sum = {dq_accum.sum()}")

    # import pdb; pdb.set_trace()


def main_bwd_caller():
    # 1. Torch setup (maybe split out)
    torch.manual_seed(0)
    device = "cuda"

    dtype = torch.bfloat16
    batch_size = 2
    seqlen_q = seqlen_k = 128
    num_heads = num_heads_kv = 8
    head_dim = head_dim_v = 128
    min_len = 32
    max_len = 64
    seqlen_q_eq_kv = True # TODO: Try to get this working later

    causal = False # TODO: Try to get this working later...
    softcap = 0.0

    m_block_size: int = 64
    n_block_size: int = 128
    num_threads: int = 256
    num_stages_Q: int = 2
    num_stages_dO: int = 2
    SdP_swapAB: bool = False
    dKV_swapAB: bool = False
    dQ_swapAB: bool = False
    AtomLayoutMSdP: int = 2
    AtomLayoutNdKV: int = 2
    AtomLayoutMdQ: int = 2
    V_in_regs: bool = False

    torch2cute_dtype_map = {
        torch.float16: cutlass.Float16,
        torch.bfloat16: cutlass.BFloat16,
        torch.float32: cutlass.Float32,
    }

    # Need q, k, v, do, lse_log2, dpsum, dq_accum, dk, dv, softmax_scale

    # TODO: See what rounding stuff i need to do here
    # seqlen_q_rounded = (seqlen_q + m_block_size - 1) // m_block_size * m_block_size
    # head_dim_rounded = (head_dim + 32 - 1) // 32 * 32

    qhead_per_kvhead = num_heads // num_heads_kv

    (
        q, 
        k, 
        v, 
        dout, 
        dpsum, 
        lse_log2, 
        dq_accum, 
        dk, 
        dv, 
        softmax_scale, 
        cu_seqlens_k, 
        cu_seqlens_q
    ) = generate_varlen_main_bwd_args(
        m_block_size, 
        batch_size, 
        num_heads, 
        head_dim, 
        min_len, 
        max_len, 
        seqlen_q_eq_kv, 
        softmax_scale
    )

    q, k, v, dout, lse_log2 = [maybe_contiguous(t) for t in (q, k, v, dout, lse_log2)]

    # 3. Not strictly needed, but for simplicity
    assert qhead_per_kvhead == 1

    # for now ignoring MQA/GQA
    # if qhead_per_kvhead > 1:
    #     seqlen_k_rounded = (seqlen_k + n_block_size - 1) // n_block_size * n_block_size
    #     head_dim_v_rounded = (head_dim_v + 32 - 1) // 32 * 32
    #     dk_accum = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded * head_dim_rounded, dtype=torch.float32, device=device)
    #     dv_accum = torch.zeros(batch_size, num_head_kv, seqlen_k_rounded * head_dim_v_rounded, dtype=torch.float32, device=device)
    # 4. torch to dlpack

    q_tensor, k_tensor, v_tensor, do_tensor, dk_tensor, dv_tensor = [
        utils.convert_from_dlpack(t.detach(), leading_dim=t.ndim - 1, alignment=16, divisibility=8)
        for t in (q, k, v, dout, dk, dv)
    ]
    dq_accum_tensor, dpsum_tensor, lse_log2_tensor = [
        utils.convert_from_dlpack(t.detach(), leading_dim=t.ndim - 1, alignment=16, divisibility=4)
        for t in (dq_accum, dpsum, lse_log2)
    ]


    # for now ignoring MQA/GQA
    # if qhead_per_kvhead > 1:
    #     dk_accum_tensor, dv_accum_tensor = [
    #         from_dlpack(t.detach(), assumed_align=16).mark_layout_dynamic(leading_dim=2)
    #         for t in (dk_accum, dv_accum)
    #     ]

    cu_seqlens_q_tensor, cu_seqlens_k_tensor = [
        from_dlpack(t.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=0) if t is not None else None
        for t in (cu_seqlens_q, cu_seqlens_k)
    ]

    # 5. Kernel compilation
    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)


    # Backward kernel: compute dk, dv, dq_accum.
    compile_key = (
        dtype, head_dim, head_dim_v, qhead_per_kvhead, causal, softcap != 0.0, m_block_size,
        n_block_size, num_threads, num_stages_Q, num_stages_dO, SdP_swapAB, dKV_swapAB, dQ_swapAB,
        AtomLayoutMSdP, AtomLayoutNdKV, AtomLayoutMdQ, V_in_regs
    )
    if compile_key not in _flash_attn_bwd.compile_cache:
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
        _flash_attn_bwd.compile_cache[compile_key] = cute.compile(
            fa_bwd_sm80, q_tensor, k_tensor, v_tensor, do_tensor, lse_log2_tensor, dpsum_tensor,
            dq_accum_tensor,
            dk_tensor, # if qhead_per_kvhead == 1 else dk_accum_tensor,
            dv_tensor, # if qhead_per_kvhead == 1 else dv_accum_tensor,
            softmax_scale, 
            current_stream,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
        )

    # 6. Kernel Call
    _flash_attn_bwd.compile_cache[compile_key](
        q_tensor, k_tensor, v_tensor, do_tensor, lse_log2_tensor, dpsum_tensor,
        dq_accum_tensor,
        dk_tensor, # if qhead_per_kvhead == 1 else dk_accum_tensor,
        dv_tensor, # if qhead_per_kvhead == 1 else dv_accum_tensor,
        softmax_scale, 
        current_stream,
        cu_seqlens_q_tensor,
        cu_seqlens_k_tensor,
    )

    # 7. Verification
    # Need to verify dq_accum (dq in f32), dk, dv
    _stats_single('dq_accum', dq_accum)
    _stats_single('dk', dk)
    _stats_single('dv', dv)


@torch.no_grad()
def _stats(name, a, b):
    diff = (a - b).float()
    mean_abs = diff.abs().mean().item()
    mean_rel = (diff.abs().mean() / b.abs().clamp_min(1e-6).mean().item())
    print(f"{name}: mean_abs={mean_abs:.4e}, mean_rel={mean_rel:.4e}, sum_fa={a.sum()}, sum_ref={b.sum()}")

@torch.no_grad()
def _stats_single(name, a):
    print(f"{name}: sum={a.sum():.4e}")
    

if __name__ == "__main__":
    # preprocess_caller()
    main_bwd_caller()