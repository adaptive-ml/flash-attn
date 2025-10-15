import torch
from typing import Tuple
from flash_attn.cute import flash_attn_varlen_func
import pytest

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
    seqused_q = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32)
    seqused_k = torch.randint(1, max_seq_len + 1, (batch_size,), dtype=torch.int32)
    # seqused_q = seqused_k = torch.ones(batch_size) * 256

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

    # Q is not paged here; provide Qc identical to Q0
    Qc = Q0

    # ship ints to device
    seqused_q = seqused_q.to(device=device, dtype=torch.int32)
    seqused_k = seqused_k.to(device=device, dtype=torch.int32)
    cu_seqlens_q = cu_seqlens_q.to(device=device, dtype=torch.int32)
    cu_seqlens_k = cu_seqlens_k.to(device=device, dtype=torch.int32)
    page_table = page_table.to(device=device, dtype=torch.int32)

    return (
        Q0, K0, V0,     # packed varlen
        Qc, Kc, Vc,     # paged (Qc == Q0)
        page_table,     # (B, max_num_pages) with -1 padding
        seqused_k, seqused_q,
        cu_seqlens_q, cu_seqlens_k,
    )

torch.no_grad()
def _stats(name, a, b, atol, rtol):
    diff = (a - b).float()
    mean_abs = diff.abs().mean().item()
    mean_rel = (diff.abs().mean() / b.abs().clamp_min(1e-6).mean().item())
    print(f"{name}: mean_abs={mean_abs:.4e}, mean_rel={mean_rel:.4e}, sum_fa={a.sum()}, sum_ref={b.sum()}")
    return mean_abs < atol and mean_rel < rtol

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

    q0, k0, v0, qc, kc, vc, page_table, seqused_k, seqused_q, cu_seqlens_q, cu_seqlens_k = generate_args(
        batch_size, 
        n_heads, 
        d_head, 
        max_seq_len, 
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
    mean_ok_out = _stats("out", out_varlen, out_paged, atol=atol, rtol=rtol)
    mean_ok_lse = _stats("lse", lse_varlen, lse_paged, atol=atol, rtol=rtol)
    assert mean_ok_out
    assert mean_ok_lse
