# llm-scratch

Controlling every detail of LLM training, by building from the ground up.


## Current Features 

1. Mixture of experts architecture defined in `llm/moe.py`. No optimizations yet.
2. Loss functions (CE, DPO, GRPO/GSPO). TODO double check.
3. Optimizer (non distributed) and learning rate scheduler (warmup, cosine annealing, post-annealing)

## For Kernels 
```bash 
uv pip install nvidia-cutlass-dsl triton
```

## Tokenizers 

```bash
uv run maturin develop --release --manifest-path tokenizer/Cargo.toml 
```