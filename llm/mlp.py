import math

import einops
import torch

from llm.norm import RMSNorm


def create_parameter(*tensor_shape):
    p = torch.nn.Parameter(torch.empty(tensor_shape))
    torch.nn.init.kaiming_uniform_(p, a=math.sqrt(5))
    return p


class MoE_MLP(torch.nn.Module):
    def __init__(self, num_experts, experts_per_token, dim_model, dim_expert):
        super().__init__()
        self.expert_gate = torch.nn.Linear(in_features=dim_model, out_features=num_experts, bias=False)

        # create_parameter(N, m, n) should be thought of a stack of N linear mappings Rn -> Rm
        self.SwiGLU_in_proj = create_parameter(num_experts, dim_expert, dim_model)
        self.SwiGLU_in_proj_gating = create_parameter(num_experts, dim_expert, dim_model)
        self.SwiGLU_out_proj = create_parameter(num_experts, dim_model, dim_expert)

        self.silu = torch.nn.SiLU()

        self.norm = RMSNorm(dim_model)
        self.experts_per_token = experts_per_token

    def forward(self, x: torch.Tensor):
        y = self.norm(x)

        experts = torch.topk(
            input=self.expert_gate(y),
            k=self.experts_per_token,
            dim=-1,
            largest=True,
            sorted=True
        )

        y = einops.rearrange(y, 'b s (1 d) -> b s 1 d')

        # selection of experts for each batch and every token
        W1 = self.SwiGLU_in_proj[experts.indices, ...]

        # abbrev batch, sequence, active_experts, m model_dim, n expert_dim, column
        W1x = einops.einsum(W1, y, 'b s a n m, b s a m -> b s a n')
        W1x = self.silu(W1x)
        del W1

        W2 = self.SwiGLU_in_proj_gating[experts.indices, ...]
        W2x = einops.einsum(W2, y, 'b s a n m, b s a m -> b s a n')   # abbrev as in W1x

        gate = W1x * W2x
        del W1x, W2x

        W3 = self.SwiGLU_out_proj[experts.indices, ...]

        SwiGLU: torch.Tensor = einops.einsum(W3, gate, 'b s a m n, b s a n -> b s a m')
        del W3

        # Shape: (batch, sequence, active_experts)
        expert_weights = torch.softmax(experts.values, -1)
        
        # linear combination of experts by softmax probabilities
        y = torch.einsum('bsam, bsa -> bsm', SwiGLU, expert_weights)

        # skip connection
        x += y

        return x


if __name__ == '__main__':
    dim_model = 64
    dim_expert = 5
    x = torch.linspace(1, dim_model**2, dim_model**2).reshape(dim_model ,dim_model)
    num_experts = 16
    experts_per_token = 8
    index = torch.randint(0, dim_model-1, size=(1,2,3, dim_model))
    subset = x[index, ...]
    # print(subset, index.shape, subset.shape)

    y = create_parameter(1,2,3)
    
    moe_mlp = MoE_MLP(num_experts=num_experts, experts_per_token=experts_per_token, dim_model=dim_model, dim_expert=dim_expert)
    batch_size = 16
    seq_len = 5
    x = torch.rand(batch_size, seq_len, dim_model)
    y = moe_mlp(x)