import cutlass
import cutlass.cute as cute

# from flash_attn.cute.debug import dbg, thread_0, dbg_val, dbg_val_32

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