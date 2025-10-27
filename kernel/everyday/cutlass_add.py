import cutlass
import cutlass.cute as cute

@cute.kernel
def tensor_add_kernel(
    gA: cute.Tensor,  # Input tensor A
    gB: cute.Tensor,  # Input tensor B
    gC: cute.Tensor,  # Output tensor C = A + B
):

    # thread, block indices and threads per block
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()

    thread_idx = bidx * bdim + tidx  # Global thread ID

    m, n = gA.shape  # Get tensor dimensions (M rows × N columns)

    ni = thread_idx % n  # col index
    mi = thread_idx // n  # row index

    a = gA[mi, ni]  # component
    b = gB[mi, ni]  # component

    gC[mi, ni] = a + b

@cute.jit
def tensor_add(
    mA: cute.Tensor,  # Input tensor A
    mB: cute.Tensor,  # Input tensor B
    mC: cute.Tensor,  # Output tensor C
):

    num_threads_per_block = 256

    m, n = mA.shape

    kernel = tensor_add_kernel(mA, mB, mC)

    kernel.launch(
        grid=((m * n) // num_threads_per_block, 1, 1),
        block=(num_threads_per_block, 1, 1),
    )

if __name__ == '__main__':

    import torch
    from functools import partial
    from typing import List

    from cutlass.cute.runtime import from_dlpack


    M, N = 16384, 8192

    a = torch.randn(M, N, device="cuda", dtype=torch.float16)
    b = torch.randn(M, N, device="cuda", dtype=torch.float16)
    c = torch.zeros(M, N, device="cuda", dtype=torch.float16)

    a_ = from_dlpack(a, assumed_align=16)  # CuTe tensor A
    b_ = from_dlpack(b, assumed_align=16)  # CuTe tensor B
    c_ = from_dlpack(c, assumed_align=16)  # CuTe tensor C


    tensor_add_ = cute.compile(tensor_add, a_, b_, c_)

    tensor_add_(a_, b_, c_)

    torch.testing.assert_close(c, a + b)
    # print(c)