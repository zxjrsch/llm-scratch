# llm-scratch

Controlling every detail of LLM training, by building from the ground up.


## Current Features 

1. Mixture of experts architecture defined in `llm/moe.py`. No optimizations yet.
2. Loss functions (CE, DPO, GRPO/GSPO). TODO double check.

## For Kernels 
```bash 
uv pip install nvidia-cutlass-dsl triton
```