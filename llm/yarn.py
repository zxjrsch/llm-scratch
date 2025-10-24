from math import log

import einops
import torch


def compute_yarn_theta(initial_context_len: int, extended_context_len: int, alpha: float, beta: float, base: float, dim: int) -> torch.Tensor:
    """Notation as in YaRN paper."""

    assert dim % 2 == 0

    s = extended_context_len / initial_context_len
    exponent = dim / (dim - 2)
    b_prime = base * s**exponent

    wavelength = 2 * torch.pi * torch.logspace(base=b_prime, start=2/dim, end=1, steps=dim//2)
    ratio = initial_context_len / wavelength

    gamma = torch.clamp((ratio-alpha)/(beta - alpha), min=0, max=1)

    rope_theta = torch.logspace(base=base, start=-2/dim, end=-1, steps=dim//2)

    yarn_theta = (1 - gamma) / s * rope_theta + gamma * rope_theta

    return yarn_theta


def compute_yarn_qk_scalar(initial_context_len: int, extended_context_len: int) -> float:
    sqrt = 1 + log(extended_context_len/initial_context_len) / 10
    return sqrt


def batched_matrix_yarn(matrix_batch: torch.Tensor, 
                        initial_context_len: int, 
                        extended_context_len: int, 
                        alpha: float, 
                        beta: float, 
                        base: float, 
                        dim: int) -> torch.Tensor:
    """
    matrix_batch has shape (batch, seq, dim)
    """
    assert len(matrix_batch.shape) == 3
    seq_len, dim = matrix_batch.shape[-2:]

    angles = compute_yarn_theta(initial_context_len, extended_context_len, alpha, beta, base, dim)
    pos = torch.arange(seq_len, dtype=torch.float)
    # batch of angles
    angles = pos.reshape(-1, 1) @ angles.reshape(1, -1)
    assert angles.shape == (seq_len, dim // 2)

    cos = angles.cos()
    sin = angles.sin()

    # (batch) x (seq_len) x (dim // 2) x (2D rotation matrices)
    rot = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(1, seq_len, dim//2, 2, 2)

    # (batch) x (seq_len) x (dim // 2) x (2D column vector)
    y = rot @ matrix_batch.reshape(-1, seq_len, dim//2, 2, 1)   # -1 is batch size
    y = y.flatten(start_dim=2, end_dim=-1)

    qk_scalar = compute_yarn_qk_scalar(initial_context_len, extended_context_len)
    y *= qk_scalar

    return y


class YaRN(torch.nn.Module):
    def __init__(
        self,    
        initial_context_len: int, 
        extended_context_len: int, 
        alpha: float, 
        beta: float, 
        base: float, 
        qk_head_dim: int
    ):
        super().__init__()
        angles = compute_yarn_theta(initial_context_len, extended_context_len, alpha, beta, base, qk_head_dim)
        pos = torch.arange(extended_context_len, dtype=torch.float)
        angles = pos.reshape(-1, 1) @ angles.reshape(1, -1)
        cos = angles.cos()
        sin = angles.sin()
        rotation_matrix = torch.stack([cos, -sin, sin, cos], dim=-1).reshape(extended_context_len, qk_head_dim//2, 2, 2)
        self.register_buffer('rotation_matrix', rotation_matrix, persistent=False)

        self.qk_scalar = compute_yarn_qk_scalar(initial_context_len, extended_context_len)

    
    def forward(self, x: torch.Tensor):
        y = self.rotation_matrix @ einops.rearrange(x, '... (pairs two) -> ... pairs two 1', two = 2)
        y *= self.qk_scalar # could be fused with normalization by 1/sqrt(d) in QK computation
        return einops.rearrange(y, '... pairs two 1-> ... (pairs two)', two=2)

def get_yarn_module(
    initial_context_len, 
    extended_context_len,
    dim_qk_head,
    alpha, 
    beta, 
    base,
    **kwargs
):
    rope_module=YaRN(
        initial_context_len=initial_context_len, 
        extended_context_len=extended_context_len,
        alpha=alpha,
        beta=beta,
        base=base,
        qk_head_dim=dim_qk_head
    )
    
    return rope_module
    
if __name__ == '__main__':
    initial_context_len = 1024
    extended_context_len = 5000
    alpha = 1
    beta = 32
    base = 10_000
    dim = 3*2**2
    angles = compute_yarn_theta(initial_context_len, extended_context_len, alpha, beta, base, dim)
    yarn_qk_scalar = compute_yarn_qk_scalar(initial_context_len, extended_context_len)

    seq_len = 7
    batch_size = 11
    batched_matrices = torch.rand(batch_size, seq_len, dim)
    result_batched_matrix = batched_matrix_yarn(batched_matrices, initial_context_len, extended_context_len, alpha, beta, base, dim)

    groups = 3
    group_size = 2
    x = torch.rand(batch_size, groups, group_size, extended_context_len, dim)
    yarn = YaRN(initial_context_len, extended_context_len, alpha, beta, base, dim)
    y = yarn(x)