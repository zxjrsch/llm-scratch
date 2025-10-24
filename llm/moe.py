from dataclasses import asdict, dataclass

import torch

from llm.block import TransformerBlock
from llm.norm import RMSNorm
from llm.yarn import YaRN


class MoE(torch.nn.Module):
    def __init__(
        self, 
        vocab_size, 
        dim_model,
        dim_qk_head, 
        dim_v_head, 
        num_heads, 
        num_groups, 
        num_experts,
        experts_per_token,
        dim_expert,
        num_transformer_layers,
        initial_context_len,
        extended_context_len,
        yarn_alpha,
        yarn_beta,
        yarn_base,
        **kwargs
    ):
        super().__init__()

        self.embedding = torch.nn.Embedding(vocab_size, dim_model)

        self.shared_rope_module = YaRN(
            initial_context_len=initial_context_len, 
            extended_context_len=extended_context_len,
            alpha=yarn_alpha,
            beta=yarn_beta,
            base=yarn_base,
            qk_head_dim=dim_qk_head
        )
        
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerBlock(
                dim_model, 
                dim_qk_head, 
                dim_v_head, 
                num_heads, 
                num_groups, 
                num_experts,
                experts_per_token,
                dim_expert,
                self.shared_rope_module,
            ) for _ in range(num_transformer_layers)
        ])

        self.norm = RMSNorm(dim_model)

        self.unembedding = torch.nn.Linear(dim_model, vocab_size, bias=False)


    def forward(self, x):
        x = self.embedding(x)

        for layer in self.transformer_blocks:
            x = layer(x)

        x = self.norm(x)
        x = self.unembedding(x)

        return x

if __name__ == '__main__':

    @dataclass
    class ModelConfig:
        vocab_size: int = 1024
        num_transformer_layers: int = 4
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
        yarn_alpha: float = 1
        yarn_beta: float = 32
        yarn_base: float = 10_000

    config = ModelConfig()

    moe = MoE(**asdict(config))

    batch_size = 2
    size = (batch_size, config.extended_context_len)

    tokens = torch.randint(low=0, high=config.vocab_size-1, size=size)
    logits = moe(tokens)
    prob = torch.softmax(logits, dim=-1)

    print(prob.shape)
    