# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import math
import itertools

import torch

from einops import rearrange, repeat

from testing import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
    pad_input,
    unpad_input,
)
from flash_attn.cute import flash_attn_varlen_func

def test_flash_attn_varlen_output(
    seqlen_q, seqlen_k, d, add_unused_qkv, causal, local, softcap, deterministic, has_qv, has_learnable_sink, mha_type, dtype
):
    if causal:
        assert seqlen_k == seqlen_q

    # See actual testing fcn for this
    assert add_unused_qkv == False
    assert local == False 
    assert deterministic == False 
    assert has_qv == False
    assert has_learnable_sink == False
    assert softcap == 0.0
    assert dtype == torch.bfloat16
    assert mha_type == 'mha'

    torch.manual_seed(0)
    device = "cuda"

    batch_size = 49 if seqlen_q <= 1024 else 7
    nheads = nheads_kv = 8
    dtype_ref = torch.bfloat16

    dv = d
    attention_chunk = 0 # What is this...

    q_ref = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype_ref)

    q_ref = q_ref.to(dtype).to(dtype_ref).requires_grad_()
    k_ref = torch.randn(batch_size, seqlen_k, nheads_kv, d, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()
    v_ref = torch.randn(batch_size, seqlen_k, nheads_kv, dv, device=device, dtype=dtype_ref).to(dtype).to(dtype_ref).requires_grad_()

    window_size = (None, None)
    learnable_sink = None
    q_descale, k_descale, v_descale = None, None, None

    q, k, v = [x.detach().requires_grad_() for x in (q_ref, k_ref, v_ref)]
    qv = None

    query_padding_mask = generate_random_padding_mask(
        seqlen_q, batch_size, device, mode="random", zero_lengths=False
    )

    # TODO: test zero_lengths
    key_padding_mask = generate_random_padding_mask(
        # seqlen_k, batch_size, device, mode="random", zero_lengths=True
        seqlen_k, batch_size, device, mode="random", zero_lengths=False
    )

    def _gen_unused_masks(padding_mask, add_unused, max_seq_len, bs, device):
        if add_unused:
            another_mask = generate_random_padding_mask(max_seq_len, bs, device)
            attn_mask = torch.logical_and(padding_mask, another_mask)
            unused_mask = torch.logical_xor(
                torch.logical_or(padding_mask, another_mask), attn_mask
            )
        else:
            attn_mask = padding_mask
            unused_mask = None
        return attn_mask, unused_mask

    query_padding_mask, query_unused_mask = _gen_unused_masks(
        query_padding_mask, add_unused_qkv, seqlen_q, batch_size, q.device
    )
    # query_padding_mask[:] = True
    # query_unused_mask = None
    key_padding_mask, key_unused_mask = _gen_unused_masks(
        key_padding_mask, add_unused_qkv, seqlen_k, batch_size, k.device
    )

    if causal or local:
        key_padding_mask = query_padding_mask

    (
        q_unpad,
        k_unpad,
        v_unpad,
        qv_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        qv,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, qv=qv, kvpacked=False,
                    query_unused_mask=query_unused_mask, key_unused_mask=key_unused_mask)
    q_unpad, k_unpad, v_unpad = [x.detach().to(dtype).requires_grad_() for x in (q_unpad, k_unpad, v_unpad)]
    out_ref, attn_ref = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        qv=None,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        window_size=window_size,
        attention_chunk=attention_chunk,
        learnable_sink=learnable_sink,
        softcap=softcap
    )
    out_pt, attn_pt = attention_ref(
        q_ref,
        k_ref,
        v_ref,
        query_padding_mask,
        key_padding_mask,
        causal=causal,
        qv=None,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
        window_size=window_size,
        attention_chunk=attention_chunk,
        learnable_sink=learnable_sink,
        softcap=softcap,
        upcast=False,
        reorder_ops=True,
        intermediate_dtype=dtype if dtype == torch.float8_e4m3fn else None,
    )

    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    if query_unused_mask is not None:
        q_zero_masking = rearrange(query_unused_mask, "b s -> b s 1 1")

    # Numerical error if we just do any arithmetic on out_ref
    fwd_atol = 2 * (out_ref + 0.3 - 0.3 - out_ref).abs().max().item()
    rtol = 2 if softcap == 0.0 else 3

    pack_gqa = None
    num_splits = 1

    out_unpad, lse = flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        # max_seqlen_k,
        # seqused_q=seqused_q,
        # seqused_k=seqused_k,
        causal=causal,
        # qv=qv_unpad,
        # q_descale=q_descale,
        # k_descale=k_descale, v_descale=v_descale,
        window_size=window_size,
        # attention_chunk=attention_chunk,
        learnable_sink=learnable_sink,
        softcap=softcap,
        pack_gqa=pack_gqa,
    )
    out = output_pad_fn(out_unpad)
    if query_unused_mask is not None:
        out.masked_fill_(q_zero_masking, 0.0)
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    # if not causal:
    #     print(f"LSE max diff: {(lse - lse_ref).abs().max().item()}")
    # breakpoint()

    # Check that FlashAttention's numerical error is at most 3x the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= rtol * (out_pt - out_ref).abs().max().item() + fwd_atol


    # Bwd stuff?
    if (
        dtype != torch.float8_e4m3fn
        and not has_qv
        and not dv > 256
        and not attention_chunk != 0
        and dv == d
        and not has_learnable_sink
        and False
    ):
        g_unpad = torch.randn_like(out_unpad)
        do_o = ((g_unpad.float() * out_unpad.float()).sum(-1)).transpose(-1, -2)
        # import flash_attn_3_cuda
        # dq_unpad, dk_unpad, dv_unpad, softmax_d, dq_accum, lse_log2 = flash_attn_3_cuda.bwd_varlen(
        #     g_unpad,
        #     q_unpad,
        #     k_unpad,
        #     v_unpad,
        #     out_unpad,
        #     lse,
        #     None,
        #     None,
        #     None,
        #     cu_seqlens_q,
        #     cu_seqlens_k,
        #     None, None,
        #     max_seqlen_q,
        #     max_seqlen_k,
        #     d ** (-0.5),
        #     causal,
        #     window_size[0], window_size[1],
        #     softcap,
        #     deterministic,
        #     0,  # sm_margin
        # )
        dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(out_unpad, (q_unpad, k_unpad, v_unpad), g_unpad)
        dq = dq_pad_fn(dq_unpad)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        if key_unused_mask is not None:
            k_zero_masking = rearrange(key_unused_mask, "b s -> b s 1 1")
            dk.masked_fill_(k_zero_masking, 0.0)
            dv.masked_fill_(k_zero_masking, 0.0)
        if query_unused_mask is not None:
            dq.masked_fill_(q_zero_masking, 0.0)
        # print(f"dO_O max diff: {(softmax_d - do_o).abs().max().item()}")
        # assert (softmax_d - do_o).abs().max().item() <= 1e-5
        # assert dq_accum.abs().max().item() == 0.0
        g = output_pad_fn(g_unpad)

        # qk = torch.einsum('bthd,bshd->bhts', q / (d ** 0.5), k).float()
        # qk = torch.masked_fill(qk, rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
        # dS = torch.einsum('bthd,bshd->bhts', g.float(), v.float())
        # P = torch.softmax(qk, -1)
        # dP = P * (dS - (g.float() * out.float()).sum(-1).transpose(1, 2).unsqueeze(-1))
        # dQ = torch.einsum('bhts,bshd->bthd', dP, k.float())
        # dV = torch.einsum('bhts,bthd->bshd', P, g.float())
        # dK = torch.einsum('bhts,bthd->bshd', dP, q.float())


        # dq, dk, dv = torch.autograd.grad(out, (q, k, v), g)
        dq_ref, dk_ref, dv_ref = torch.autograd.grad(out_ref, (q_ref, k_ref, v_ref), g)
        dq_pt, dk_pt, dv_pt = torch.autograd.grad(out_pt, (q_ref, k_ref, v_ref), g)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")
        # breakpoint()
        dq_atol = 2 * (dq_ref + 0.3 - 0.3 - dq_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
        assert (dq - dq_ref).abs().max().item() <= rtol * (dq_pt - dq_ref).abs().max().item() + dq_atol
        dk_atol = 2 * (dk_ref + 0.3 - 0.3 - dk_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
        assert (dk - dk_ref).abs().max().item() <= rtol * (dk_pt - dk_ref).abs().max().item() + dk_atol
        dv_atol = 2 * (dv_ref + 0.3 - 0.3 - dv_ref).abs().max().item() + (0 if softcap == 0 else 3e-4)
        assert (dv - dv_ref).abs().max().item() <= rtol * (dv_pt - dv_ref).abs().max().item() + dv_atol


if __name__ == "__main__":
    seqlen_q, seqlen_k = 256, 256
    d_head = 128

    # Things to not touch probably?
    add_unused_qkv = local = deterministic = has_qv = has_learnable_sink = False
    softcap = 0.0
    dtype = torch.bfloat16
    mha_type = 'mha'

    causal = True
    test_flash_attn_varlen_output(seqlen_q, seqlen_k, d_head, add_unused_qkv, causal, local, softcap, deterministic, has_qv, has_learnable_sink, mha_type, dtype)