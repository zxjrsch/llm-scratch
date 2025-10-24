from dataclasses import asdict, dataclass

import torch

from llm.attention import GroupedQueryAttention
from llm.mlp import MoE_MLP
from llm.yarn import get_yarn_module


class TransformerBlock(torch.nn.Module):
    def __init__(
            self, 
            dim_model, 
            dim_qk_head, 
            dim_v_head, 
            num_heads, 
            num_groups, 
            num_experts,
            experts_per_token,
            dim_expert,
            shared_rope_module,
            **kwargs,
    ):
        super().__init__()

        self.GQA = GroupedQueryAttention(
            dim_model=dim_model, 
            dim_qk_head=dim_qk_head,
            dim_v_head=dim_v_head,
            num_heads=num_heads,
            num_groups=num_groups,
            rope_module=shared_rope_module
        )

        self.moe_mlp = MoE_MLP(
            num_experts=num_experts,
            experts_per_token=experts_per_token,
            dim_model=dim_model,
            dim_expert=dim_expert
        )

    def forward(self, x):

        x = self.GQA(x)
        x = self.moe_mlp(x)
        
        return x
    
if __name__ == '__main__':

    @dataclass
    class ModelConfig:
        dim_model: int = 64
        dim_qk_head: int = 8
        dim_v_head: int = 32
        num_heads: int = 8
        num_groups: int = 4
        num_experts: int = 8
        experts_per_token: int = 4
        dim_expert: int = 32
        initial_context_len: int = 128
        extended_context_len: int = 256
        alpha: float = 1
        beta: float = 32
        base: float = 10_000

    config = ModelConfig()

    shared_rope_module = get_yarn_module(**asdict(config))
    transformer_block = TransformerBlock(**asdict(config), shared_rope_module=shared_rope_module)

    batch_size = 2
    x = torch.rand(batch_size, config.extended_context_len, config.dim_model)
    y = transformer_block(x)
    print(y.shape)