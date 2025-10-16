import pytest

import torch
from typing import Tuple
from flash_attn.cute import(
    flash_attn_varlen_func, 
    _flash_attn_fwd, 
    _flash_attn_bwd,
)

def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b

@torch.no_grad()
def _make_prefix_sums(lengths: torch.Tensor) -> torch.Tensor:
    # lengths: (B,), int32 (CPU or CUDA ok)
    cu = torch.zeros(lengths.numel() + 1, dtype=torch.int32, device=lengths.device)
    cu[1:] = torch.cumsum(lengths.to(torch.int32), dim=0)
    return cu

def generate_args(
    batch_size: int = 8,
    n_heads: int = 16,
    d_head: int = 128,
    max_seq_len: int = 128 * 32,
    dtype: torch.dtype = torch.bfloat16,
    page_size: int = 192,
    device: str = "cuda",
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor,   # Q0, K0, V0  (packed)
    torch.Tensor, torch.Tensor, torch.Tensor,   # Qc, Kc, Vc  (paged; Qc==Q0)
    torch.Tensor,                                # page_table (B, max_num_pages)
    torch.Tensor, torch.Tensor,                  # seqused_k, seqused_q (token counts)
    torch.Tensor, torch.Tensor                   # cu_seqlens_q, cu_seqlens_k
]:
    """
    Returns:
      Q0: (total_q,   H, d)
      K0: (total_k,   H, d)
      V0: (total_k,   H, d)   # d_v == d_head here
      Qc: (total_q,   H, d)   # identical to Q0 (no paging for Q)
      Kc: (total_pages, page_size, H, d)   # pages globally shuffled
      Vc: (total_pages, page_size, H, d)
      page_table: (B, max_num_pages) int32, -1 padded; maps (seq, local_page_idx) -> global page id in Kc/Vc
      seqused_k: (B,) int32 token counts for K/V
      seqused_q: (B,) int32 token counts for Q
      cu_seqlens_q: (B+1,) int32 prefix sums of seqused_q
      cu_seqlens_k: (B+1,) int32 prefix sums of seqused_k
    """
    torch.manual_seed(0)
    assert max_seq_len >= 1
    H = H_kv = n_heads
    d = d_head
    d_v = d_head
    max_num_pages = ceil_div(max_seq_len, page_size)

    # --- sample per-seq token lengths (varlen); ensure >=1 token
    # seqused_q = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32)
    # seqused_k = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32)
    seqused_q = torch.tensor([max_seq_len], dtype=torch.int32)
    seqused_k = torch.tensor([max_seq_len], dtype=torch.int32)


    # What would it mean to have more query than kv...?
    with torch.no_grad():
        seqused_q = torch.minimum(seqused_q, seqused_k)

    # prefix sums (packed indexing)
    cu_seqlens_q = _make_prefix_sums(seqused_q)
    cu_seqlens_k = _make_prefix_sums(seqused_k)
    total_q = int(cu_seqlens_q[-1].item())
    total_k = int(cu_seqlens_k[-1].item())

    # --- create packed Q/K/V (these are the "vanilla varlen" tensors)
    Q0 = torch.randn(total_q, H, d, device=device, dtype=dtype, requires_grad=True)
    K0 = torch.randn(total_k, H_kv, d, device=device, dtype=dtype, requires_grad=True)
    V0 = torch.randn(total_k, H_kv, d_v, device=device, dtype=dtype, requires_grad=True)

    # with torch.no_grad():
    #     for i in range(total_q):
    #         Q0[i, :, :] = i

    #     for i in range(total_k):
    #         K0[i, :, :] = i
    #         V0[i, :, :] = i


    # --- build paged K/V with shuffled global page order, and page_table mapping
    # First: slice K0/V0 per sequence, segment into pages (pad last page with zeros),
    # collect all pages in a list (unshuffled), then shuffle globally and remap page_table.
    page_table = torch.full((batch_size, max_num_pages), -1, dtype=torch.int32)  # CPU for now
    pages_K = []
    pages_V = []
    page_meta = []  # (seq_id, local_page_idx, unshuffled_global_idx)

    # CPU offsets for slicing in packed view
    cu_k_cpu = cu_seqlens_k.cpu()
    for i in range(batch_size):
        L = int(seqused_k[i].item())
        start = int(cu_k_cpu[i].item())
        end = int(cu_k_cpu[i + 1].item())
        assert end - start == L

        # slice packed K/V for this sequence
        Ki = K0[start:end]   # (L, H, d)
        Vi = V0[start:end]   # (L, H, d_v)

        n_pages_i = ceil_div(L, page_size)
        assert n_pages_i <= max_num_pages, "Increase max_seq_len or max_num_pages"

        # write local page mapping (-1 padded)
        # (we'll fill with remapped (shuffled) ids later)
        # page_table[i, :n_pages_i] will be set after shuffling
        for lp in range(n_pages_i):
            off = lp * page_size
            last = min(off + page_size, L)
            n_tok = last - off

            # make a full page buffer and copy tokens
            pk = torch.zeros(page_size, H_kv, d, device=device, dtype=dtype)
            pv = torch.zeros(page_size, H_kv, d_v, device=device, dtype=dtype)
            if n_tok > 0:
                pk[:n_tok].copy_(Ki[off:last])
                pv[:n_tok].copy_(Vi[off:last])

            pages_K.append(pk)
            pages_V.append(pv)
            page_meta.append((i, lp))  # (seq, local_page_idx)

    total_pages = len(pages_K)
    if total_pages == 0:
        # degenerate (shouldn't happen because lengths >=1)
        Kc = torch.empty(0, page_size, H_kv, d, device=device, dtype=dtype)
        Vc = torch.empty(0, page_size, H_kv, d_v, device=device, dtype=dtype)
    else:
        # stack unshuffled, then shuffle
        Kc_unshuf = torch.stack(pages_K, dim=0)  # (P, page_size, H, d)
        Vc_unshuf = torch.stack(pages_V, dim=0)  # (P, page_size, H, d_v)

        perm = torch.randperm(total_pages)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(total_pages, device=perm.device)

        Kc = Kc_unshuf[perm].contiguous().detach()  # detach: these are storage views of K0; keep graph on K0
        Vc = Vc_unshuf[perm].contiguous().detach()

        # fill page_table with *shuffled* global ids
        # unshuffled index u -> shuffled index s = inv_perm[u]
        # unshuffled order corresponds to page_meta order
        for u, (seq_id, local_page_idx) in enumerate(page_meta):
            s = int(inv_perm[u].item())
            page_table[seq_id, local_page_idx] = s

    # safety: cap any sequences that would exceed max_num_pages (shouldn't happen given sampling)
    for i in range(batch_size):
        n_pages_i = ceil_div(int(seqused_k[i].item()), page_size)
        assert n_pages_i <= max_num_pages, "Increase max_seq_len or page_size to cover lengths."


    # ship ints to device
    seqused_q = seqused_q.to(device=device, dtype=torch.int32)
    seqused_k = seqused_k.to(device=device, dtype=torch.int32)
    cu_seqlens_q = cu_seqlens_q.to(device=device, dtype=torch.int32)
    cu_seqlens_k = cu_seqlens_k.to(device=device, dtype=torch.int32)
    page_table = page_table.to(device=device, dtype=torch.int32)

    Kc = Kc.detach().requires_grad_()
    Vc = Vc.detach().requires_grad_()
    Qc = Q0.detach().clone().requires_grad_()

    return (
        Q0, K0, V0,     # packed varlen
        Qc, Kc, Vc,     # paged (Qc == Q0)
        page_table,     # (B, max_num_pages) with -1 padding
        seqused_k, seqused_q,
        cu_seqlens_q, cu_seqlens_k,
    )

def clone_like(t):
    c = t.clone().detach().requires_grad_(True)
    return c

@torch.no_grad()
def _stats(name, a, b, atol, rtol):
    diff = (a - b).float()
    mean_abs = diff.abs().mean().item()
    mean_rel = (diff.abs().mean() / (b.abs().mean().item() + 1e-10))
    print(f"{name}: mean_abs={mean_abs:.4e}, mean_rel={mean_rel:.4e}, sum_fa={a.sum()}, sum_ref={b.sum()}")
    return mean_abs < atol and mean_rel < rtol

@torch.no_grad()
def reconstruct_packed_from_paged(
    dK_pages: torch.Tensor,         # (P, page_size, H_kv, d_k)
    dV_pages: torch.Tensor,         # (P, page_size, H_kv, d_v)
    page_table: torch.Tensor,       # (B, max_num_pages) int32; (seq, local_page) -> global page id, -1 padded
    seqused_k: torch.Tensor,        # (B,) int32 token counts for K/V
    cu_seqlens_k: torch.Tensor,     # (B+1,) int32 prefix sums of seqused_k
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = dK_pages.device
    total_k = int(cu_seqlens_k[-1].item())
    _, _, H_kv, d_k = dK_pages.shape
    d_v = dV_pages.shape[-1]

    dK0 = torch.zeros(total_k, H_kv, d_k, device=device, dtype=dK_pages.dtype)
    dV0 = torch.zeros(total_k, H_kv, d_v, device=device, dtype=dV_pages.dtype)

    cu_k_cpu = cu_seqlens_k.detach().cpu()
    seqk_cpu = seqused_k.detach().cpu()
    pt_cpu   = page_table.detach().cpu()

    B = seqused_k.numel()
    for b in range(B):
        Lk = int(seqk_cpu[b].item())
        if Lk <= 0:
            continue
        start_tok = int(cu_k_cpu[b].item())
        n_pages_b = ceil_div(Lk, page_size)

        for lp in range(n_pages_b):
            gpid = int(pt_cpu[b, lp].item())
            if gpid < 0:
                continue
            off  = lp * page_size
            n_tok = min(page_size, Lk - off)
            if n_tok <= 0:
                continue

            dK0[start_tok + off : start_tok + off + n_tok].copy_(dK_pages[gpid, :n_tok])
            dV0[start_tok + off : start_tok + off + n_tok].copy_(dV_pages[gpid, :n_tok])

    return dK0, dV0

# Unused for now...
@torch.no_grad()
def reconstruct_paged(
    dKc: torch.Tensor,              # (total_k, H, d)
    dVc: torch.Tensor,              # (total_k, H, d_v)
    page_table: torch.Tensor,       # (B, max_num_pages) int32; maps (seq, local_page)
    seqused_k: torch.Tensor,        # (B,) int32 token counts for K/V
    cu_seqlens_k: torch.Tensor,     # (B+1,) int32 prefix sums of seqused_k
    page_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reconstruct paged varlen-style grads (dK0, dV0) from packed grads (dKc, dVc).
    Assumes dKc/dVc are already the grads on the packed input (no flow back to K0/V0).
    """
    device = dKc.device

    # sizes for K/V
    _, H_kv, d_k = dKc.shape
    _, _, d_v = dVc.shape

    # Work with small CPU scalars for indices; tensors can stay on CUDA.
    cu_k_cpu = cu_seqlens_k.detach().cpu()
    seqk_cpu = seqused_k.detach().cpu()
    pt_cpu = page_table.detach().cpu()

    # total_pages = sum([ceil_div(slen, page_size) for slen in seqk_cpu])
    total_pages = int(page_table.max().item()) + 1

    # Need pages
    dK0 = torch.zeros(total_pages, page_size, H_kv, d_k, device=device, dtype=dKc.dtype)
    dV0 = torch.zeros(total_pages, page_size, H_kv, d_v, device=device, dtype=dVc.dtype)

    for batch_id in range(page_table.shape[0]):
        for idx in range(page_table.shape[1]):
            page_id = pt_cpu[batch_id, idx].item()
            if page_id == -1: continue
            start = int(cu_k_cpu[batch_id].item())
            total_len = int(seqk_cpu[batch_id].item())
            off = idx * page_size
            page_len = min(total_len - off, page_size)
            dK0[page_id, :page_len, :, :].copy_(dKc[start + off:start + off + page_len])
            dV0[page_id, :page_len, :, :].copy_(dVc[start + off:start + off + page_len])
    return dK0, dV0

# Assuming standard varlen format and batch size = 1
def chunk(
        qc: torch.Tensor,
        out_paged: torch.Tensor,
        lse_paged: torch.Tensor,
        grad_paged: torch.Tensor,
        dq_paged: torch.Tensor,
        n_chunks: int,
        page_size: int,
    ) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
]:
    # Make chunks multiples of page table size (might even be fine without, though bad perf?)

    assert qc.shape[0] == out_paged.shape[0] == lse_paged.shape[1] == grad_paged.shape[0] == dq_paged.shape[0]
    seqlen = qc.shape[0]
    total_pages = ceil_div(seqlen, page_size)
    pages_per_chunk = total_pages // n_chunks
    assert pages_per_chunk > 0, "Can't do less than 1 page per chunk" # Maybe this is actually fine?
    offset = 0
    extra = total_pages % n_chunks
    q_chunked, out_chunked, lse_chunked, grad_chunked, dq_chunked = [], [], [], [], []
    for chunk_idx in range(n_chunks):
        # TODO: reverse later since last page is partially filled --> not as even as it could be
        pages_in_chunk = pages_per_chunk + 1 if chunk_idx < extra else pages_per_chunk
        q_chunked.append(qc[offset:offset + (pages_in_chunk) * page_size])
        out_chunked.append(out_paged[offset:offset + (pages_in_chunk) * page_size])
        lse_chunked.append(lse_paged[:, offset:offset + (pages_in_chunk) * page_size])
        grad_chunked.append(grad_paged[offset:offset + (pages_in_chunk) * page_size])
        dq_chunked.append(dq_paged[offset:offset + (pages_in_chunk) * page_size])
        offset += pages_in_chunk * page_size

    return q_chunked, out_chunked, lse_chunked, grad_chunked, dq_chunked

@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("n_heads", [1, 4, 10])
@pytest.mark.parametrize("d_head", [64, 128])
@pytest.mark.parametrize("seq_len", [1, 32, 1024, 2048, 3000, 4096, 5000, 8192, 10000, 100000])
@pytest.mark.parametrize("n_chunks", [1, 2, 8, 24])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True])
# @pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
def test_diff_kv(
    batch_size: int,
    n_heads: int,
    d_head: int,
    seq_len: int,
    n_chunks: int,
    dtype,
    causal: bool,
):
    assert batch_size == 1
    assert causal == True

    page_size = 128
    if n_chunks > ceil_div(seq_len, page_size):
        pytest.skip('n_chunks has to be <= num pages')

    (
        q0, k0, v0, 
        qc, kc, vc, 
        page_table, 
        seqused_k, 
        seqused_q, 
        cu_seqlens_q, 
        cu_seqlens_k 
    ) = generate_args(
        batch_size=batch_size, 
        n_heads=n_heads, 
        d_head=d_head, 
        max_seq_len=seq_len, 
        dtype=dtype,
        page_size=page_size,
    )

    # Use the same upstream gradient to compare backward paths
    # good enough for now since assuming headdim = headdim_v
    grad_out = torch.randn_like(q0)

    grad_varlen = clone_like(grad_out)
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

    n_chunks = min(3, ceil_div(qc.shape[0], page_size))
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

    # Old
    # dq_paged, dk_paged, dv_paged = _flash_attn_bwd(
    #     q=qc,
    #     k=kc,
    #     v=vc,
    #     out=out_paged,
    #     dout=grad_paged,
    #     lse=lse_paged,
    #     softmax_scale=None,
    #     causal=causal,
    #     softcap=0.0,
    #     cu_seqlens_q=cu_seqlens_q.clone(),
    #     cu_seqlens_k=None,
    #     seqused_q=None,
    #     seqused_k=seqused_k.clone(),
    #     page_table=page_table,
    # )

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

    mean_dq_ok = _stats("dQ", dq_varlen, dq_paged, atol=atol, rtol=rtol)
    mean_dk_ok = _stats("dK", dk_varlen, dk_paged_reshaped, atol=atol, rtol=rtol)
    mean_dv_ok = _stats("dV", dv_varlen, dv_paged_reshaped, atol=atol, rtol=rtol)

    ok_q = torch.allclose(dq_varlen.float(), dq_paged.float(), atol=atol, rtol=rtol)
    ok_k = torch.allclose(dk_varlen.float(), dk_paged_reshaped.float(), atol=atol, rtol=rtol)
    ok_v = torch.allclose(dv_varlen.float(), dv_paged_reshaped.float(), atol=atol, rtol=rtol)
    print(f"Close? dQ={ok_q}, dK={ok_k}, dV={ok_v}")

    assert dq_paged.isnan().sum() == 0
    assert dk_paged_reshaped.isnan().sum() == 0
    assert dv_paged_reshaped.isnan().sum() == 0

    assert mean_dq_ok
    assert mean_dk_ok
    assert mean_dv_ok

    assert ok_q
    assert ok_k
    assert ok_v