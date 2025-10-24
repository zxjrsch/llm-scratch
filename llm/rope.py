import torch

"""
Let dim be an even positive integer representing the dimension of a head in the transformer block.
Let x be a vector of length dim. 
Let k be a non-negative integer (k represents the position of the token that gave rise to x).
Let b be a real number, which, together with k, and dim, parametrizes the rotation matrix.

TODO Use einops to simplied RoPE implementation.
"""


def single_vector_rope(vect: torch.Tensor, base: float, pos: int) -> torch.Tensor:
    assert len(vect.shape) == 1
    dim = vect.shape[0]

    assert dim % 2 == 0

    angles = torch.logspace(base=base, start=-2 / dim, end=-1, steps=dim // 2)
    angles *= pos

    # # test case 1
    # exponents = torch.tensor([-2*d / dim for d in range(1, dim//2+1)])
    # angles2 = pos * torch.pow(base, exponents)
    # print(angles, angles2, angles - angles2)

    # # test case 2
    # matrices = []
    # cos = angles.cos()
    # sin = angles.sin()
    # for i in range(dim//2):
    #     rot = torch.tensor([
    #         [cos[i], -sin[i]],
    #         [sin[i], cos[i]]
    #     ])
    #     matrices.append(rot)
    # R = torch.block_diag(*matrices)
    # y_test = R @ vect
    # print(y_test)

    cos = angles.cos()
    sin = angles.sin()

    # batch of 2 x 2 rotation matrices
    rot = torch.stack([cos, -sin, sin, cos], dim=1).reshape(-1, 2, 2)

    # matmul a batch of 2D vectors
    y = rot @ vect.reshape(-1, 2, 1)
    y = y.flatten()

    # # test case 3
    # print(y - y_test)

    return y


def matrix_rope(row_major_matrix: torch.Tensor, base: float) -> torch.Tensor:
    """Batched operation of single_vector_rope() to
    the rows of the row_major_matrix
    """
    assert len(row_major_matrix.shape) == 2
    seq_len, dim = row_major_matrix.shape

    angles = torch.logspace(start=-2 / dim, end=-1, steps=dim // 2, base=base)
    pos = torch.arange(seq_len, dtype=torch.float)
    # batch of angles
    angles = pos.reshape(-1, 1) @ angles.reshape(1, -1)
    assert angles.shape == (seq_len, dim // 2)

    cos = angles.cos()
    sin = angles.sin()

    # (seq_len) x (dim // 2) x (2D rotation matrices)
    rot = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(seq_len, dim // 2, 2, 2)

    # (seq_len) x (dim // 2) x (2D column vector)
    y = rot @ row_major_matrix.reshape(seq_len, dim // 2, 2, 1)
    y = y.flatten(start_dim=1, end_dim=-1)

    return y


def batched_matrix_rope(matrix_batch: torch.Tensor, base: float) -> torch.Tensor:
    """
    matrix_batch has shape (batch, seq, dim)
    """
    assert len(matrix_batch.shape) == 3
    seq_len, dim = matrix_batch.shape[-2:]

    angles = torch.logspace(start=-2 / dim, end=-1, steps=dim // 2, base=base)
    pos = torch.arange(seq_len, dtype=torch.float)
    # batch of angles
    angles = pos.reshape(-1, 1) @ angles.reshape(1, -1)
    assert angles.shape == (seq_len, dim // 2)

    cos = angles.cos()
    sin = angles.sin()

    # (batch) x (seq_len) x (dim // 2) x (2D rotation matrices)
    rot = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(1, seq_len, dim // 2, 2, 2)

    # (batch) x (seq_len) x (dim // 2) x (2D column vector)
    y = rot @ matrix_batch.reshape(-1, seq_len, dim // 2, 2, 1)  # -1 is batch size
    y = y.flatten(start_dim=2, end_dim=-1)

    return y


if __name__ == "__main__":
    dim = 10
    seq_len = 7
    b = 10

    pos = 10
    vect = torch.linspace(1, dim, dim)
    result_vector = single_vector_rope(vect, b, pos)

    row_major_matrix = torch.rand(seq_len, dim)
    result_matrix = matrix_rope(row_major_matrix, b)
    # # test case 4
    # test_case_matrix = torch.empty(seq_len, dim)
    # for i in range(seq_len):
    #     test_case_matrix[i] = single_vector_rope(row_major_matrix[i], b, i)
    # print(result_matrix - test_case_matrix)

    batch_size = 11
    batched_matrices = torch.rand(batch_size, seq_len, dim)
    result_batched_matrix = batched_matrix_rope(batched_matrices, b)
    # # test case 5
    # test_result_batch = torch.empty(batch_size, seq_len, dim)
    # for i in range(batch_size):
    #     test_result_batch[i] = matrix_rope(batched_matrices[i], b)
    # print(test_result_batch - result_batched_matrix)
