import triton
import triton.langauge as tl


@triton.jit
def mat_vect_kernel(
    ptr_batched_matrices, ptr_vector, ptr_answer,
    bat_mat_stride_element, # how many indices to traverse in memory to get from Aij to Ai(j+1)
    bat_mat_stride_row, # how many idx to advance from Aij to A(i+1)j
    vect_stride_element,  # we assume vector is not batched
    ans_stride_row, # Ax is a column vector, so we move 1 element to get to the next row
    M, N, # M x N dim matrices
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr  # a block matrix of dim BLOCK_M x BLOCK_N 
):
  tile_id = tl.program_id(axis=0) # we will use 1D thread block

  # Whereas a pointer refers to a single element in memory, 
  # a block pointer is an abstract pointer that refers to a block of memory.
  # To specify a block ptr is to specific a region of memory
  block_ptr_batched_matrices = tl.make_block_ptr(
      base=ptr_batched_matrices,
      shape=(M, N),
      strides=(bat_mat_stride_row, bat_mat_stride_element),
      offsets=(tile_id * BLOCK_M, 0),
      block_shape=(BLOCK_M, BLOCK_N),
      order=(1, 0)  # order of dimensions from major to minor axes. Here row major, iterate dim 1 then dim 0.
  )

  block_ptr_vector = tl.make_block_ptr(
      base=ptr_vector,
      shape=(N,),
      strides=(vect_stride_element, ),
      offsets=(0,),
      block_shape=(BLOCK_N, ),
      order=(0,)
  )

  block_ptr_answer = tl.make_block_ptr(
      base=ptr_answer,
      shape=(M,),
      strides=(ans_stride_row,),
      offsets=(tile_id * BLOCK_M,),
      block_shape=(BLOCK_M,),
      order=(0,)
  )

  # a subset of the output vector's components
  ans = tl.zeros((BLOCK_M,), dtype=tl.float32)

  for i in range(tl.cdiv(N, BLOCK_N)):
    matrix_row_block = tl.load(block_ptr_batched_matrices, boundary_check=(0, 1), padding_option='zero')
    vector_col_block = tl.load(block_ptr_vector, boundary_check=(0,), padding_option='zero')

    ans += tl.sum(matrix_row_block * vector_col_block[None, :], axis=1)

    block_ptr_batched_matrices.advance((0, BLOCK_N))
    block_ptr_vector.advance((BLOCK_N,))

  tl.store(block_ptr_answer, ans, boundary_check=(0,))