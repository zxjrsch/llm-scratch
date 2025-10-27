## NVSHMEM

- NVSHMEM proessing element: a unit of distributed work. Each element has a process id. 
- Synchronization: each NVSHMEM processing element calls initialization, then performs NVSHMEM operation, and ends by each calling a termination function.
- NVSHMEM jobs has a global memory accessible by all the processing elements, and forms the mechanism for collective communication among participating processing elements within a NVSHMEM job. This GPU memory is allocated by the cpu with a specific function, and memory allocated by other means are private to each processing element. This memory can be accessed by the CUDA kernel and the cpu. 
This shared memory is called partitioned global address space. 

```cu
#include <stdio.h>
#include <cuda.h>
#include <nvshmem.h>
#include <nvshmemx.h>

__global__ void simple_shift(int *destination) {
    int mype = nvshmem_my_pe();     // local
    int npes = nvshmem_n_pes();     // world
    int peer = (mype + 1) % npes;   // destination

    nvshmem_int_p(destination, mype, peer);     // communicate
}

int main(void) {

    // rough idea 
    // see more at https://docs.nvidia.com/nvshmem/api/api/overview.html

    nvshmem_init();

    nvshmem_malloc();

    simple_shift<<<>>>();
    
    nvshmem_free();

    nvshmem_finalize();
}
```

### Resources
To use NVSHMEM in Python see [documentation](https://docs.nvidia.com/nvshmem/api/api/language_bindings/python/index.html)
and [examples](https://docs.nvidia.com/nvshmem/api/examples/language_bindings/python/index.html#torch-distributed-processgroup-initialization-example). 

See [GEMM](https://docs.nvidia.com/nvshmem/api/examples.html#gemm-allreduce-fused-kernel-example) example.

[Collective communication for 100k+ GPUs](https://arxiv.org/pdf/2510.20171v1)

[DeepEP](https://github.com/deepseek-ai/DeepEP/tree/main)

[Efficient and Portable Mixture-of-Experts Communication](https://www.perplexity.ai/hub/blog/efficient-and-portable-mixture-of-experts-communication)
