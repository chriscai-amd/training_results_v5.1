#include <prims/mlcommon_linalg_hip.cuh>
#include "hip/hip_runtime.h"
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <hip/hip_fp16.h>

#include <algorithm>
#include <array>  // ROCm port: cuda/std/array -> std array
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <include/utils.cuh>
#include <layers/relu_layer.hpp>
#include <string>
// HugeCTR ROCm port: cuml linalg replaced by hand-written HIP shim
// HugeCTR ROCm port: cuml linalg replaced by hand-written HIP shim
#include <utils.hpp>

namespace HugeCTR {

namespace {

struct alignas(8) half4 : public std::array<__half2, 2> {};

// ROCm port (May 2026): 8x __half = 16-byte int4-load version.
// At bs8x the existing half4 (= int2 = 8-byte) ReluLayer runs 14 calls/iter
// at ~78 us/call = 1.09 ms/iter (4.1 % of bs8x iter). Switching from int2
// to int4 loads/stores doubles the per-kernel HBM throughput and roughly
// halves wall time for these memory-bound element-wise kernels.
struct alignas(16) half8 : public std::array<__half2, 4> {};

template <typename MainOp, typename Fallback>
__global__ void half4_relu_kernel(__half* __restrict__ out, const __half* __restrict__ in, int size,
                                  MainOp main_op, Fallback fallback) {
  const __half2 zero2 = TypeFunc<__half2>::zero();
  const half4 zero4 = half4{zero2, zero2};
  half4* out4 = reinterpret_cast<half4*>(out);
  const half4* in4 = reinterpret_cast<const half4*>(in);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int size4 = size / 4;

  for (int i = tid; i < size4; i += stride) {
    main_op(in4, zero2, i, out4);
  }

  const __half zero = TypeFunc<__half>::zero();
  int rmdr_base = size4 * 4;

  for (int i = rmdr_base + tid; i < size; i += stride) {
    fallback(in, zero, i, out);
  }
}

template <typename MainOp, typename Fallback>
__global__ void half8_relu_kernel(__half* __restrict__ out, const __half* __restrict__ in, int size,
                                  MainOp main_op, Fallback fallback) {
  const __half2 zero2 = TypeFunc<__half2>::zero();
  half8* out8 = reinterpret_cast<half8*>(out);
  const half8* in8 = reinterpret_cast<const half8*>(in);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int size8 = size / 8;

  for (int i = tid; i < size8; i += stride) {
    main_op(in8, zero2, i, out8);
  }

  const __half zero = TypeFunc<__half>::zero();
  int rmdr_base = size8 * 8;

  for (int i = rmdr_base + tid; i < size; i += stride) {
    fallback(in, zero, i, out);
  }
}

}  // namespace

template <typename T>
ReluLayer<T>::ReluLayer(const core23::Tensor& input_tensor, const core23::Tensor& output_tensor,
                        const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer({input_tensor}, {output_tensor}, gpu_resource) {}

template <typename T>
void ReluLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  int len = input_tensors_[0].num_elements();

  auto fop = [] __device__(T in) { return (in > T(0)) ? in : T(0); };

  MLCommon::LinAlg::unaryOp(output_tensors_[0].template data<T>(), input_tensors_[0].template data<T>(), len, fop,
                            get_gpu().get_stream());
}

template <typename T>
void ReluLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());

  int len = input_tensors_[0].num_elements();
  auto bop = [] __device__(T d_out, T d_in) { return (d_in > T(0)) ? d_out : T(0); };
  MLCommon::LinAlg::binaryOp(input_tensors_[0].template data<T>(), output_tensors_[0].template data<T>(),
                             input_tensors_[0].template data<T>(), len, bop, get_gpu().get_stream());
}

ReluLayer<__half>::ReluLayer(const core23::Tensor& input_tensor,
                             const core23::Tensor& output_tensor,
                             const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer({input_tensor}, {output_tensor}, gpu_resource) {}

// ROCm port: half8 (int4-vectorized) ReLU was added but measures FLAT
// at bs8x when controlled for node-effect noise (vec8 = v1 within ±0.1 %
// 4-trial-each). Conclusion: half4 already saturates the kernel's HBM
// bandwidth on these element counts, so vec8 doesn't help. half8 path
// kept for opt-in via `HCTR_RELU_KERNEL=vec8`; default = v1 (= half4).
namespace {
inline bool relu_use_vec8(const __half* a, const __half* b, size_t size) {
  if ((reinterpret_cast<uintptr_t>(a) % 16) != 0) return false;
  if ((reinterpret_cast<uintptr_t>(b) % 16) != 0) return false;
  static const bool opt_in = []() {
    const char* env = getenv("HCTR_RELU_KERNEL");
    return env && std::string(env) == "vec8";
  }();
  return opt_in;
}
}  // namespace

void ReluLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const size_t BLOCK_DIM = 1024;

  const auto size = input_tensors_[0].num_elements();
  const auto grid_dim = get_gpu().get_sm_count() * 4;
  __half* outp = output_tensors_[0].template data<__half>();
  const __half* inp = input_tensors_[0].template data<__half>();

  if (relu_use_vec8(outp, inp, size)) {
    half8_relu_kernel<<<grid_dim, BLOCK_DIM, 0, get_gpu().get_stream()>>>(
        outp, inp, size,
        [] __device__(const half8* in8, const __half2 zero2, int i, half8* out8) {
          const int4 hack = reinterpret_cast<const int4*>(in8)[i];
          half8 t = *reinterpret_cast<const half8*>(&hack);
          const half8 mask = {__hgt2(t[0], zero2), __hgt2(t[1], zero2),
                              __hgt2(t[2], zero2), __hgt2(t[3], zero2)};
          const half8 res  = {__hmul2(t[0], mask[0]), __hmul2(t[1], mask[1]),
                              __hmul2(t[2], mask[2]), __hmul2(t[3], mask[3])};
          reinterpret_cast<int4*>(out8)[i] = *reinterpret_cast<const int4*>(&res);
        },
        [] __device__(const __half* in, const __half zero, int i, __half* out) {
          __half t = __ldg(in + i);
          __half mask = __hgt(t, zero);
          out[i] = __hmul(t, mask);
        });
  } else {
    half4_relu_kernel<<<grid_dim, BLOCK_DIM, 0, get_gpu().get_stream()>>>(
        outp, inp, size,
        [] __device__(const half4* in4, const __half2 zero2, int i, half4* out4) {
          const int2 hack = reinterpret_cast<const int2*>(in4)[i];
          half4 t = *reinterpret_cast<const half4*>(&hack);

          const half4 mask = {__hgt2(t[0], zero2), __hgt2(t[1], zero2)};
          const half4 res = {__hmul2(t[0], mask[0]), __hmul2(t[1], mask[1])};

          reinterpret_cast<int2*>(out4)[i] = *reinterpret_cast<const int2*>(&res);
        },
        [] __device__(const __half* in, const __half zero, int i, __half* out) {
          __half t = __ldg(in + i);
          __half mask = __hgt(t, zero);
          out[i] = __hmul(t, mask);
        });
  }
}

void ReluLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());

  const size_t BLOCK_DIM = 1024;

  const size_t size = input_tensors_[0].num_elements();
  const size_t grid_dim = get_gpu().get_sm_count() * 4;
  __half* outp = input_tensors_[0].template data<__half>();      // d_in (overwrite slot)
  const __half* inp = output_tensors_[0].template data<__half>(); // d_out

  if (relu_use_vec8(outp, inp, size)) {
    half8_relu_kernel<<<grid_dim, BLOCK_DIM, 0, get_gpu().get_stream()>>>(
        outp, inp, size,
        [] __device__(const half8* in8, const __half2 zero2, int i, half8* out8) {
          // out8 -> d_in (= __ in[i] = ReLU(d_in[i] > 0) * d_out[i] in fwd terms,
          //               but here in HCTR's bprop terminology the kernel's "out8"
          //               is the d_in tensor that already holds the FWD activations;
          //               we read its sign via mask, then multiply d_out.
          const int4 t_hack  = reinterpret_cast<const int4*>(out8)[i];
          const half8 t      = *reinterpret_cast<const half8*>(&t_hack);
          const half8 mask   = {__hgt2(t[0], zero2), __hgt2(t[1], zero2),
                                __hgt2(t[2], zero2), __hgt2(t[3], zero2)};
          const int4 t2_hack = reinterpret_cast<const int4*>(in8)[i];
          const half8 t2     = *reinterpret_cast<const half8*>(&t2_hack);
          const half8 res    = {__hmul2(t2[0], mask[0]), __hmul2(t2[1], mask[1]),
                                __hmul2(t2[2], mask[2]), __hmul2(t2[3], mask[3])};
          reinterpret_cast<int4*>(out8)[i] = *reinterpret_cast<const int4*>(&res);
        },
        [] __device__(const __half* in, const __half zero, int i, __half* out) {
          __half t = out[i];
          __half mask = __hgt(t, zero);
          out[i] = __hmul(__ldg(in + i), mask);
        });
  } else {
    half4_relu_kernel<<<grid_dim, BLOCK_DIM, 0, get_gpu().get_stream()>>>(
        outp, inp, size,
        [] __device__(const half4* in4, const __half2 zero2, int i, half4* out4) {
          const int2 t_hack = reinterpret_cast<const int2*>(out4)[i];
          const half4 t = *reinterpret_cast<const half4*>(&t_hack);

          const half4 mask = {__hgt2(t[0], zero2), __hgt2(t[1], zero2)};

          const int2 t2_hack = reinterpret_cast<const int2*>(in4)[i];
          const half4 t2 = *reinterpret_cast<const half4*>(&t2_hack);

          const half4 res = {__hmul2(t2[0], mask[0]), __hmul2(t2[1], mask[1])};

          reinterpret_cast<int2*>(out4)[i] = *reinterpret_cast<const int2*>(&res);
        },
        [] __device__(const __half* in, const __half zero, int i, __half* out) {
          __half t = out[i];
          __half mask = __hgt(t, zero);
          out[i] = __hmul(__ldg(in + i), mask);
        });
  }
}

template class ReluLayer<float>;
template class ReluLayer<__half>;

}  // namespace HugeCTR
