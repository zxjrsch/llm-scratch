import einops
import torch

from llm.norm import RMSNorm
from llm.yarn import YaRN


class GroupedQueryAttention(torch.nn.Module):
    def __init__(self, dim_model, dim_qk_head, dim_v_head, num_heads, num_groups, rope_module):
        super().__init__()

        assert num_heads % num_groups == 0

        # query + key + values
        concat_qkv_dim = num_heads * dim_qk_head + num_groups * (dim_qk_head + dim_v_head)
        self.in_projection = torch.nn.Linear(
            in_features=dim_model,
            out_features=concat_qkv_dim,
            bias=False
        )

        self.out_projection = torch.nn.Linear(
            in_features=num_heads*dim_v_head,
            out_features=dim_model,
            bias=False
        )

        self.rms_norm = RMSNorm(dim_model)
        self.rope: torch.nn.Module = rope_module
        
        self.dim_qk_head = dim_qk_head
        self.dim_v_head = dim_v_head
        self.num_heads = num_heads
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor):
        y: torch.Tensor = self.rms_norm(x)
        y = self.in_projection(y)   # shape [batch, sequence, concat_qkv_dim]

        concat_q_dim = self.num_heads * self.dim_qk_head
        concat_k_dim = self.num_groups * self.dim_qk_head
        concat_v_dim = self.num_groups * self.dim_v_head

        q, k, v = y.split(split_size=[concat_q_dim, concat_k_dim, concat_v_dim], dim=-1)
        
        # Abbrev.
        # b batch, s sequence, g groups, h heads_per_group, d dim_head (q or k or v, depending on case)
    
        q = einops.rearrange(
            q.contiguous(), 
            'b s (g h d) -> b g h s d',
            g = self.num_groups,
            h = self.num_heads // self.num_groups,  # redundant but clear
            d = self.dim_qk_head
        )
        q = self.rope(q)

        k = einops.rearrange(
            k.contiguous(), 
            'b s (g d) -> b g 1 s d',
            g=self.num_groups,
            d=self.dim_qk_head
        )
        k = self.rope(k)
        
        v = einops.rearrange(
            v.contiguous(), 
            'b s (g d) -> b g 1 s d',
            g=self.num_groups,
            d=self.dim_v_head
        )

        qk = einops.einsum(q, k, 'b g h s1 d, b g h s2 d -> b g h s1 s2')
        del y, q, k

        s = qk.shape[-1]
        mask = torch.full((s, s), -float('inf'))
        mask = torch.triu(mask, diagonal=1)

        qk /= self.dim_qk_head ** 0.5
        qk += mask

        attn_weights = torch.softmax(qk, dim=-1)
        del qk

        # Abbrev p probability
        qkv = einops.einsum(attn_weights, v, 'b g h s p, b g h p d -> b g h s d')
        qkv = einops.rearrange(qkv, 'b g h s d -> b s (g h d)')
        y = self.out_projection(qkv)    # attention output

        # skip connection 
        x += y
        return x


if __name__ == '__main__':
    (initial_context_len, 
     extended_context_len, 
     alpha, beta, base, 
     qk_head_dim, 
     dim_v_head, 
     num_heads, 
     num_groups,
     batch,
     dim_model) = (8, 16, 1, 32, 10_000, 32, 10, 16, 4, 16, 64)

    rope = YaRN(initial_context_len, extended_context_len, alpha, beta, base, qk_head_dim)

    gqa = GroupedQueryAttention(
        dim_model=dim_model, 
        dim_qk_head=qk_head_dim, 
        dim_v_head=dim_v_head, 
        num_heads=num_heads, 
        num_groups=num_groups, 
        rope_module=rope
    )

    x = torch.rand(batch, extended_context_len, dim_model)

    h = gqa(x)
