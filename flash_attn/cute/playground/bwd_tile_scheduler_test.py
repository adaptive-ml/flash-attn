# bwd_probe.py
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import cuda.bindings.driver as cuda

from flash_attn.cute.tile_scheduler import ParamsBase, SingleTileVarlenScheduler, TileSchedulerArgumentsBwd

class SchedulerTest:

    arch = 90

    def __init__(self, m_block_size, n_block_size):
        self.qhead_per_kvhead_packgqa = 1
        self.element_size = 2
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size

    @cute.jit
    def __call__(
        self,
        *,
        num_block: cutlass.Int32,
        num_head: cutlass.Int32,
        num_batch: cutlass.Int32,
        head_dim: cutlass.Int32,
        head_dim_v: cutlass.Int32,
        total_k: cutlass.Int32,
        cu_seqlens_k: cute.Tensor,
        stream: cuda.CUstream,
    ):
        args = TileSchedulerArgumentsBwd(
            num_block=num_block,
            num_head=num_head,
            num_batch=num_batch,
            headdim=head_dim,
            headdim_v=head_dim_v,
            total_k=total_k,
            tile_shape_mn=(self.m_block_size, self.n_block_size),
            mCuSeqlensK=cu_seqlens_k,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead_packgqa,
            element_size=self.element_size,
        )

        params = SingleTileVarlenScheduler.to_underlying_arguments(args) # passed in directly to kernel

        grid_dim = SingleTileVarlenScheduler.get_grid_shape(params)

        cute.printf("Grid Dim is {}", grid_dim)

        # call kernel here
        self.kernel(
            params,
        ).launch(
            grid=grid_dim,
            block=(32, 1, 1),
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        params: ParamsBase, 
        # out_triplets, 
        # out_valid
    ):
        bid = cute.arch.block_idx()[0]
        sched = SingleTileVarlenScheduler.create(params)
        work_tile = sched.initial_work_tile_info()
        if cute.arch.lane_idx() == 0:
            n_block, head_idx, batch_idx = work_tile.tile_idx
            cute.printf(
                "bid: {} n_block: {} head: {} batch: {} is_valid: {}",
                bid,
                n_block,
                head_idx,
                batch_idx,
                work_tile.is_valid_tile,
            )
            # out_triplets[bid, 0] = n_block   # block
            # out_triplets[bid, 1] = head_idx   # head
            # out_triplets[bid, 2] = batch_idx   # batch
            # out_valid[bid] = cutlass.Int32(1) if work_tile.is_valid_tile else cutlass.Int32(0)


def setup_caller(
    *,
    num_block: int,
    num_head: int,
    num_batch: int,
    head_dim: int,
    head_dim_v: int,
    total_k: int,
    m_block_size: int,
    n_block_size: int,
    cu_seqlens_k: torch.Tensor,
    qhead_per_kvhead_packgqa: int = 1,
    element_size: int = 2
):
    scheduler_test = SchedulerTest(m_block_size, n_block_size)

    dlpack_cuseqk = from_dlpack(
        cu_seqlens_k.detach(), assumed_align=4,

    ).mark_layout_dynamic(leading_dim=0)

    compute_capability = torch.cuda.get_device_capability()[0]

    current_stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    # Compile and run
    compiled_func = cute.compile(scheduler_test,
        num_block=num_block,
        num_head=num_head,
        num_batch=num_batch,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        total_k=total_k,
        cu_seqlens_k=dlpack_cuseqk,
        stream=current_stream
    )

    print("Compiled Scheduler Test")

    compiled_func(
        num_block=num_block,
        num_head=num_head,
        num_batch=num_batch,
        head_dim=head_dim,
        head_dim_v=head_dim_v,
        total_k=total_k,
        cu_seqlens_k=dlpack_cuseqk,
        stream=current_stream,
    )

    cuda.cuStreamSynchronize(current_stream)



# Generate Lengths
def generate_varlen_bwd_seqlens(
    batch_size=8,
    min_len=32,
    max_len=64,
):
    torch.manual_seed(0)
    device = "cuda"

    lens_k = torch.randint(low=min_len, high=max_len + 1, size=(batch_size,))
    cu_seqlens_k = torch.cat([torch.zeros(1, dtype=torch.int32), lens_k.cumsum(0)]).contiguous()

    return cu_seqlens_k.to(dtype=torch.int32, device=device)


@cute.jit
def thread_0():
    tidx, _, _ = cute.arch.thread_idx()
    m_block, num_head, batch_size = cute.arch.block_idx()
    return tidx == 0 and m_block == 0 and num_head == 0 and batch_size == 0


@cute.jit
def dbg(label: cutlass.Constexpr, x, compile_only: cutlass.Constexpr = False):
    tidx, _, _ = cute.arch.thread_idx()
    m_block, num_head, batch_size = cute.arch.block_idx()

    print("Compile-Time", label, ":", x)

    if cutlass.const_expr(not compile_only):
        if thread_0():
            cute.printf("Runtime " + label + ": {}\n", x)

@cute.jit
def dbg_val(label: cutlass.Constexpr, x):
    tidx, _, _ = cute.arch.thread_idx()
    m_block, num_head, batch_size = cute.arch.block_idx()

    if (tidx == 0) & (m_block == 0) & (num_head == 0) & (batch_size == 0):
        cute.printf("Runtime " + label + ": {}\n", x)

@cute.jit
def dbg_val_32(label: cutlass.Constexpr, x):
    tidx, _, _ = cute.arch.thread_idx()
    m_block, num_head, batch_size = cute.arch.block_idx()
    zz = cute.composition(x, cute.make_layout((2, 16)))
    if (tidx == 0) & (m_block == 0) & (num_head == 0) & (batch_size == 0):
        cute.printf("Runtime " + label + "[0]: {}\n", zz[0, None])
        cute.printf("Runtime " + label + "[1]: {}\n", zz[1, None])


if __name__ == "__main__":
    batch_size = 2
    min_len=32
    max_len=64

    # TODO: Remove, I don't think this is used anywhere
    num_block = 10000

    num_head = 4
    d_head = 128

    m_block_size = 128
    n_block_size = 32

    cu_seqlens_k = generate_varlen_bwd_seqlens(
        batch_size=batch_size,
        min_len=min_len,
        max_len=max_len,
    )

    print(f'{n_block_size=}')
    print(f'{num_head=}')
    print(f'{cu_seqlens_k=}')

    setup_caller(
        num_block=num_block,
        num_head=num_head,
        num_batch=batch_size,
        head_dim=d_head,
        head_dim_v=d_head,
        total_k=cu_seqlens_k[-1].item(),
        m_block_size=m_block_size,
        n_block_size=n_block_size,
        cu_seqlens_k=cu_seqlens_k
    )