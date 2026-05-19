// BGRADA kernel microbenchmark: V1 (reduce_sum_columns_kernel) vs V5
// (bgrada_v5_kernel + bgrad_finalize_v5_kernel).
//
// Both kernels do the same math: dbias[i] = sum_{j=0..k-1} A[i + j*m]
// where A is __half col-major (m x k). V5 also pre-divides by 256 and clamps.
//
// We benchmark across the actual M/N shapes the production DLRM-DCNv2 model
// uses, copied directly from the per-GEMM events in the recent
// HCTR_NATIVE_TRACE_DETAIL=2 trace:
//   bot_mlp:  m={128, 256, 512},  k=6912 (per-GPU batch at bs=1x, 8GPU)
//   top_mlp:  m={256, 512, 1024}, k=6912
//   single-GPU baseline (Phase 6 README context): k=55296 (full bs=1x)
//
// Build (inside ROCm container):
//   /opt/rocm/lib/llvm/bin/clang++ -O3 -std=c++17 \
//       --offload-arch=gfx950 -x hip /rps_out/bgrada_microbench.cu \
//       -o /rps_out/bgrada_microbench
// Run:
//   /rps_out/bgrada_microbench

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>

#define HIP_CHECK(x) do { hipError_t e = (x); if (e) { \
    fprintf(stderr,"HIP error %d at %s:%d: %s\n",e,__FILE__,__LINE__,hipGetErrorString(e)); \
    exit(1);}} while(0)

// ============================================================
// V1: legacy reduce_sum_columns_kernel (lifted verbatim).
// One thread per output row. Each thread scans k columns of A.
// 1 block per ceil(m/256). Block dim 256.
// ============================================================
__global__ void v1_reduce_sum_columns_kernel(const __half* __restrict__ A,
                                              __half* __restrict__ dbias,
                                              int m, int k) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  float sum = 0.0f;
  for (int j = 0; j < k; ++j) {
    sum += __half2float(A[i + static_cast<size_t>(j) * m]);
  }
  constexpr float kPreDivide = 256.0f;
  sum /= kPreDivide;
  if (!isfinite(sum)) sum = 0.0f;
  constexpr float kFp16Max = 65504.0f;
  if (sum > kFp16Max) sum = kFp16Max;
  else if (sum < -kFp16Max) sum = -kFp16Max;
  dbias[i] = __float2half(sum);
}

// ============================================================
// V5: 2D tile + per-block partial sum + atomicAdd to FP32 scratch +
// separate finalize kernel. Lifted verbatim from
// fused_gemm_functors.cu:bgrada_v5_kernel / bgrad_finalize_v5_kernel.
// BLOCK_M = 64 rows per block, WAVES = 4 waves per block, N_TILE = 1024.
// ============================================================
template <int BLOCK_M, int WAVES_PER_BLOCK>
__global__ void v5_bgrada_kernel(const __half* __restrict__ A,
                                  float* __restrict__ scratch_fp32,
                                  int m, int k, int n_tile) {
  int wave_id = threadIdx.x / BLOCK_M;
  int lane    = threadIdx.x % BLOCK_M;
  int i = blockIdx.x * BLOCK_M + lane;
  if (i >= m) return;

  int j_block_start = blockIdx.y * n_tile;
  int j_block_end   = j_block_start + n_tile;
  if (j_block_end > k) j_block_end = k;
  int span = (j_block_end - j_block_start + WAVES_PER_BLOCK - 1) / WAVES_PER_BLOCK;
  int j_start = j_block_start + wave_id * span;
  int j_end   = j_start + span;
  if (j_end > j_block_end) j_end = j_block_end;

  float sum = 0.0f;
  for (int j = j_start; j < j_end; ++j) {
    sum += __half2float(A[static_cast<size_t>(i) + static_cast<size_t>(j) * m]);
  }
  __shared__ float wave_sums[WAVES_PER_BLOCK][BLOCK_M];
  wave_sums[wave_id][lane] = sum;
  __syncthreads();
  if (wave_id == 0) {
    float total = 0.0f;
    #pragma unroll
    for (int w = 0; w < WAVES_PER_BLOCK; ++w) total += wave_sums[w][lane];
    atomicAdd(&scratch_fp32[i], total);
  }
}

__global__ void v5_finalize_kernel(float* __restrict__ scratch,
                                    __half* __restrict__ dbias,
                                    int m, float bgrad_div) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  float total = scratch[i] / bgrad_div;
  if (!isfinite(total)) total = 0.0f;
  constexpr float kFp16Max = 65504.0f;
  if (total >  kFp16Max) total =  kFp16Max;
  else if (total < -kFp16Max) total = -kFp16Max;
  dbias[i] = __float2half(total);
  scratch[i] = 0.0f;  // self-reset for next iter
}

// ============================================================
// Driver
// ============================================================
struct Shape {
  int m, k;
  const char* label;
};

static double bench_v1(const __half* d_A, __half* d_dbias, int m, int k,
                       hipStream_t stream, int iters) {
  dim3 block(256, 1, 1);
  dim3 grid((m + 255) / 256, 1, 1);
  // warmup
  for (int w = 0; w < 5; ++w) {
    v1_reduce_sum_columns_kernel<<<grid, block, 0, stream>>>(d_A, d_dbias, m, k);
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventRecord(e0, stream));
  for (int i = 0; i < iters; ++i) {
    v1_reduce_sum_columns_kernel<<<grid, block, 0, stream>>>(d_A, d_dbias, m, k);
  }
  HIP_CHECK(hipEventRecord(e1, stream));
  HIP_CHECK(hipEventSynchronize(e1));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
  HIP_CHECK(hipEventDestroy(e0)); HIP_CHECK(hipEventDestroy(e1));
  return double(ms) / iters * 1000.0;  // -> microseconds
}

static double bench_v5(const __half* d_A, __half* d_dbias, float* d_scratch,
                       int m, int k, hipStream_t stream, int iters) {
  constexpr int BLOCK_M = 64;
  constexpr int WAVES   = 4;
  constexpr int kThreads = BLOCK_M * WAVES;
  constexpr int N_TILE  = 1024;
  int grid_x = (m + BLOCK_M - 1) / BLOCK_M;
  int grid_y = (k + N_TILE - 1)  / N_TILE;
  dim3 grid(grid_x, grid_y, 1);
  // warmup (5x)
  for (int w = 0; w < 5; ++w) {
    v5_bgrada_kernel<BLOCK_M, WAVES><<<grid, kThreads, 0, stream>>>(
        d_A, d_scratch, m, k, N_TILE);
    int fin_grid = (m + 255) / 256;
    v5_finalize_kernel<<<fin_grid, 256, 0, stream>>>(d_scratch, d_dbias, m, 256.0f);
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0)); HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventRecord(e0, stream));
  for (int i = 0; i < iters; ++i) {
    v5_bgrada_kernel<BLOCK_M, WAVES><<<grid, kThreads, 0, stream>>>(
        d_A, d_scratch, m, k, N_TILE);
    int fin_grid = (m + 255) / 256;
    v5_finalize_kernel<<<fin_grid, 256, 0, stream>>>(d_scratch, d_dbias, m, 256.0f);
  }
  HIP_CHECK(hipEventRecord(e1, stream));
  HIP_CHECK(hipEventSynchronize(e1));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
  HIP_CHECK(hipEventDestroy(e0)); HIP_CHECK(hipEventDestroy(e1));
  return double(ms) / iters * 1000.0;  // -> microseconds
}

int main(int argc, char** argv) {
  int iters = 100;
  if (argc > 1) iters = std::atoi(argv[1]);

  // DLRM-DCNv2 BGRADA shapes covered:
  //   bot_mlp m ∈ {128, 256, 512}
  //   top_mlp m ∈ {256, 512, 1024}
  //   k (BGRADA contracted dim) = per-GPU batch
  // We bench both per-GPU batches (k=6912 at 8-GPU bs=1x) AND the
  // single-GPU bs=1x (k=55296) that Phase 6's README claim was made on.
  std::vector<Shape> shapes;
  for (int m : {128, 256, 512, 1024}) {
    shapes.push_back({m, 6912,  "8-GPU bs=1x (k=6912)"});
    shapes.push_back({m, 55296, "1-GPU bs=1x (k=55296)"});
  }

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Max alloc size: 1024 * 55296 = 57M elements * 2 B = 113 MB. Fine.
  size_t max_elem = 1024ull * 55296;
  __half* d_A;
  HIP_CHECK(hipMalloc(&d_A, max_elem * sizeof(__half)));
  // Fill with small random-ish values via a single kernel to avoid host->device.
  std::vector<__half> host_init(max_elem);
  for (size_t i = 0; i < max_elem; ++i)
    host_init[i] = __float2half(((i % 17) - 8) * 0.001f);
  HIP_CHECK(hipMemcpy(d_A, host_init.data(),
                      max_elem * sizeof(__half), hipMemcpyHostToDevice));
  __half* d_dbias_v1;
  __half* d_dbias_v5;
  float*  d_scratch;
  HIP_CHECK(hipMalloc(&d_dbias_v1, 4096 * sizeof(__half)));
  HIP_CHECK(hipMalloc(&d_dbias_v5, 4096 * sizeof(__half)));
  HIP_CHECK(hipMalloc(&d_scratch,  4096 * sizeof(float)));
  HIP_CHECK(hipMemset(d_scratch, 0, 4096 * sizeof(float)));

  printf("=== BGRADA kernel microbenchmark on AMD MI355X ===\n");
  printf("    iters per measurement: %d\n", iters);
  printf("    data dtype: __half (FP16) col-major (m x k)\n\n");
  printf("%4s %6s   %18s   %12s  %12s   %6s   %10s   %10s\n",
         "M", "K", "context", "V1 µs/call", "V5 µs/call",
         "V5/V1", "V1 GB/s", "V5 GB/s");
  printf("---- ------   ------------------   ------------  ------------   ------   ----------   ----------\n");

  for (const auto& s : shapes) {
    int m = s.m, k = s.k;
    double bytes_per_call = double(m) * double(k) * 2.0;  // single read of A
    double v1_us = bench_v1(d_A, d_dbias_v1, m, k, stream, iters);
    double v5_us = bench_v5(d_A, d_dbias_v5, d_scratch, m, k, stream, iters);
    double v1_gbs = bytes_per_call / (v1_us * 1e-6) / 1e9;
    double v5_gbs = bytes_per_call / (v5_us * 1e-6) / 1e9;
    printf("%4d %6d   %18s   %12.2f  %12.2f   %6.2fx %10.1f   %10.1f\n",
           m, k, s.label, v1_us, v5_us, v1_us / v5_us, v1_gbs, v5_gbs);
  }

  printf("\n(MI350X HBM3 theoretical peak: ~5.3 TB/s = 5300 GB/s)\n");

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_dbias_v1));
  HIP_CHECK(hipFree(d_dbias_v5));
  HIP_CHECK(hipFree(d_scratch));
  HIP_CHECK(hipStreamDestroy(stream));
  return 0;
}
