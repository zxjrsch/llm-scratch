#include <cstdlib>
#include <cstdio>
#include <cassert>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"


// kernel 
template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static 
__launch_bounds__(decltype(size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
        TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
        TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
        TC const* C, CStride dC, CSmemLayout sC_layout, CThreadLayout tC,
        Alpha alpha, Beta beta)
{
    using namespace cute;

    // full tensors
    Tensor mA = make_tensor(make_gmem_ptr(A), select<0, 2>(shapp_MNK), dA); // (M, K)
    Tensor mB = make_tensor(make_gmem_ptr(B), select<1, 2>(shapp_MNK), dB); // (N, K)
    Tensor mC = make_tensor(make_gmem_ptr(C), select<0, 1>(shapp_MNK), dC); // (M, N)

    // local tile for thread block 
    auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _); // (m, n, k)
    Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X, _1>{});
    Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1, _1, X>{});

    // shared memory tensor location
    __shared__ TA smemA[cosize_v<ASmemLayout>];
    __shared__ TB smemB[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);
    Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);

    // partitioning tile access across threads 
    Tensor tAgA = local_partition(gA, tA, threadIdx.x);
    Tensor tAsA = local_partition(sA, tA, threadIdx.x);

    Tensor tBgB = local_paritition(gB, tB, threadIdx.x);
    Tensor tBsB = local_parititon(sB, tB, threadIdx.x);

    Tensor tCsA = local_partition(sA, tC, threadIdx.x, Step<_1, X>{});
    Tensor tCsB = local_partition(sB, tC, threadIdx.x, Step<X, _1>{});
    Tensor tCgC = local_partition(gC, tC, threadIdx.x, Step<_1, _1>{});

    // allocate accumulator
    Tensor tCrC = make_tensor_like(tCgC);
    clear(tCrC);

    // main loop 

    auto K_TILE_MAX = size<2>(tAgA);

    for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
    {
        // copy from global to shared memory 
        copy(tAgA(_, _, k_tile), tAsA);
        copy(tBgB(_, _, k_tile), tBsB);

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        gemm(tCsA, tCsB, tCrC);

        __syncthreads();
    }

    // Epilogue
    axpby(alpha, tCrC, beta, tCgC);

}

template<class TA, class TB, class TC,
         class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC *C, int ldC,
        cudaStream_t stream = 0)
{
    using namespace cute;

    // dynamic shapes
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K)

    // NT strides
    auto dA = make_stride(Int<1>{}, ldA);
    auto dB = make_stride(Int<1>{}, ldB);
    auto dC = make_stride(Int<1>{}, ldC);

    // CTA tile size
    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    // smem layout 
    auto sA = make_layout(make_shape(bM, bK));
    auto sB = make_layout(make_shape(bN, bK));
    auto sC = make_layout(make_shape(bM, bN));

    dim3 dimBlock(size(tC));
    dim3 dimGrid(size(ceil_div(M, bM)), 
                 size(ceil_div(N, bN)));

    gemm_device<<<dimGrid, dimBlock, 0, stream>>>(
        prob_shape, cta_tiler,
        A, dA, sA, tA,
        B, dB, sB, tB,
        C, dC, sC, tC,
        alpha, beta
    );

}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void 
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA, 
        TB const* B, int ldB, 
        Beta beta,
        TC* C, int ldC,
    cudaStream_t stream = 0)
{
    using namespace cute;

    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    auto prob_shape = make_shape(M, N, K);

    auto dA = make_stride(ldA, Int<1>{});
    auto dB = make_stride(ldB, Int<1>{});
    auto dC = make_stride(Int<1>{}, ldC);

    auto bM = Int<128>{};
    auto bN = Int<128>{};
    auto bK = Int<8>{};
    auto cta_tiler = make_shape(bM, bN, bK);

    auto sA = make_layout(make_shape(bM, bK), LayoutRight{});
    auto sB = make_layout(make_shape(bN, bK), LayoutRight{});
    auto sC = make_layout(make_shape(bM, bN));

    auto tA = make_layout(make_shape(Int<32>, Int<8>{}), LayoutRight{});
    auto tB = make_layout(make_shape(Int<32>{}, Int<8>{}, LayoutRight{}));
    auto tC = make_layout(make_shape(Int<16>{}, Int<16>{}));

    dim3 dimBlock(size(tC));
    dim3 dimGrid(size(ceil_div(M, bM)), size(ceil_div(N, bM)));

    gemm_device<<<dimGrid, dimBlock, 0, stream>>>(
        prob_shape, cta_tiler,
        A, dA, sA, tA,
        B, dB, sB, tB,
        C, dC, sC, tC,
        alpha, beta
    );
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void 
gemm(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     TA const* A, int ldA,
     TB const* B, int ldB,
     Beta beta, 
     TC * C, int ldC,
     cudaStream_t stream = 0)
{
    if (transA == 'N' && transB == 'T') {
        return gemm_nt(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    } else
    if (transA == 'T' && transB == 'N') {
        return gemm_tn(m, n, k, alpha, A, ldA, B, ldB, beta, C, ldC, stream);
    }
    assert(false && "Not Implemented");
}

int main(int argc, char** argv)
{
    int m = 5120;
    int n = 5120;
    int k = 4096;
    char transA = 'N';
    char transB = 'T';

    using TA = float;
    using TB = float;
    using TC = float;
    using TI = float;

    TI alpha = 1.0;
    TI beta = 0.0;

    cute::device_int(0);

    thrust::host_vector<TA> h_A(m*k);
    thrust::host_vector<TB> h_B(n*k);
    thrust::host_vector<TC> h_C(m*n);

    for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < n*k; ++j) h_A[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
    for (int j = 0; j < m*n; ++j) h_A[j] = static_cast<TC>(-1);

    thrust::device_vector<TA> d_A = h_A;
    thrust::device_vector<TB> d_B = h_B;
    thrust::device_vector<TC> d_C = h_C;

    double gflops = (2.0*m*n*k) * 1e-9;
    const int timing_iterations = 100;
    GPU_Clock timer;

    int ldA = 0, ldB = 0, ldC = m;

    if (transA == 'N') {
        ldA = m;
    } else if (transA == 'T') {
        ldA = k;
    } else {
        assert(false);
    }

    if (transB == 'N') {
        ldB = k;
    } else if (transB == 'T') {
        ldB = n;
    } else {
        assert(false);
    }

    d_C = h_C;
    gemm(transA, transB, m, n, k, alpha,
        d_A.data().get(), ldA.
        d_B.data().get(), ldB,
        beta,
        d_C.data().get(), ldC);

    thurst::host_vecot<TC> cute_result = d_C;

    timer.start();
    for (int i = 0; i < timing_iterations; ++i) {
        gemm(transA, transB, m, n, k, alpha,
            d_A.data().get(), ldA.
            d_B.data().get(), ldB,
            beta,
            d_C.data().get(), ldC);
    }
    double cute_time = timer.seconds() / timing_iterations;

    printf("Cute GEMM [%6.1f] GFlops/s (%6.4f) ms\n", gflops / cute_time, cute_time*1000)

}
