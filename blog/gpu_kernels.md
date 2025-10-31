## Note

The best reference remains to be the 3 guides for CUDA, PTX and CUTLASS. However these manuals are dense, and the following 
references make concepts more digestible.

## GPU Architecture

1. [Hopper](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

2. [Blackwell](https://developer.nvidia.com/blog/inside-nvidia-blackwell-ultra-the-chip-powering-the-ai-factory-era/)

3. [GPU Glossary](https://modal.com/gpu-glossary)

4. [Semi Analysis: Tensor Core from Volta to Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell)

5. [Tensor core Gen 5 matrix multiply and accumulate instruction (Blackwell tcgen05.mma)](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)

## Profiler

1. [Video walkthrough](https://www.youtube.com/watch?v=F_BazucyCMw)

2. [Nsight](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#memory-chart)


## Performance Optimization

1. [Deep learning performance docs](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html)

2. [Modular blog on how to optimize GEMM on Blackwell](https://www.modular.com/matrix-multiplication-on-blackwell)

3. [Programming Blackwell tensor cores with Cutlass](https://www.nvidia.com/en-us/on-demand/session/gtc25-s72720/)

4. [cuBLAS is a baseline for the Mojo blogs](https://docs.nvidia.com/cuda/cublas/)

5. [Matmul on Hopper](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog)

6. [A nice writeup](https://siboehm.com/articles/22/CUDA-MMM)

## DSL

1. We have already mentioned Cutlass and Mojo, see [tile-lang](https://tilelang.com/tutorials/debug_tools_for_tilelang.html) and the corresponding [paper](https://arxiv.org/pdf/2504.17577)

## Code Samples 

1. [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)
2. [MXFP8 kernel for MoE training](https://cursor.com/blog/kernels)
3. [Expert communication kernels](https://research.perplexity.ai/articles/efficient-and-portable-mixture-of-experts-communication), also see Github for NVSHMEM kernel sample.
4. [FlashInfer](https://docs.flashinfer.ai/api/gemm.html)
