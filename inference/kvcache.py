from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor


class KVCache:
    # Dimensions don't handle MHA/GQA yet
    def __init__(self, prefill_keys: Float[Tensor, 'batch sequence qk_dim'], prefill_values: Float[Tensor, 'batch sequence v_dim']):
        self.keys = prefill_keys
        self.values = prefill_values

    def cache_kv_pair(self, k: Float[Tensor, 'batch qk_dim'], v: Float[Tensor, 'batch v_dim']):
        """If RoPE is used, then this method assumes k has been rotated."""
        k = rearrange(k, 'batch qk_dim -> batch 1 qk_dim')
        v = rearrange(v, 'batch v_dim -> batch 1 v_dim')
        self.keys = torch.cat([self.keys, k], dim=1).contiguous()
        self.values = torch.cat([self.values, v], dim=1).contiguous()

    def compute_attention(self, q: Float[Tensor, 'batch qk_dim']):
        """After [q,k,v] are computed, first cache kv, then call this method."""
        qk = einsum(q, self.keys, 'batch qk_dim, batch seq qk_dim -> batch seq')
        qk_dim = q.shape[-1]
        qk *= qk_dim ** -0.5
        qk = qk - qk.max(dim=-1, keepdim=True).values
        qk = torch.softmax(qk, dim=-1)
        qkv = einsum(qk, self.values, 'batch seq, batch seq v_dim -> batch v_dim')
        return qkv
        

if __name__ == '__main__':
    import torch

    (batch, sequence) = (32, 1024)
    qk_dim = 64
    v_dim = 128
    prefill_keys = torch.rand(batch, sequence, qk_dim)
    prefill_values = torch.rand(batch, sequence, v_dim)
    q = torch.rand(batch, qk_dim)
    k = torch.rand(batch, qk_dim)
    v = torch.rand(batch, v_dim)

    cache = KVCache(prefill_keys=prefill_keys, prefill_values=prefill_values)
    cache.cache_kv_pair(k=k, v=v)
    qkv = cache.compute_attention(q=q)
    print(qkv.shape)