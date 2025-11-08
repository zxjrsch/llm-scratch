from dataclasses import asdict, dataclass

import torch

from llm.moe import MoE

# python -m llm.autocast_example

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

    # example 1
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        logits = moe(tokens)
        prob = torch.softmax(logits, dim=-1)
        print(prob.dtype)

    # example 2: auotcast, gradient scaling, and grad clipping

    class ExampleModel(torch.nn.Module):
        def __init__(self, n=10):
            super().__init__()
            self.layer = torch.nn.Linear(n, n)

        @torch.autocast(device_type='cuda')
        def forward(self, x):
            return self.layer(x)

    m = ExampleModel()
    # scales gradient to prevent underflow in mixed precison
    scaler = torch.amp.GradScaler()
    optimizer = torch.optim.AdamW(m.parameters())
    
    for i in range(10):
        optimizer.zero_grad()
        x = torch.rand(100, 10)
        with torch.autocast(device_type='cuda'):
            
            out = m(x)
            loss = out.mean()

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1)
        scaler.step(optimizer)
        scaler.update()

    # more examples https://docs.pytorch.org/docs/stable/notes/amp_examples.html