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

#include <common.hpp>
#include <layers/concat_layer.hpp>
#include <network_buffer_channels.hpp>
#include <utils.hpp>

#include <cstdlib>
#include <string>

namespace HugeCTR {

namespace {

template <typename T>
__global__ void concat_fwd_kernel(T* out, const int2 out_dim, T* in, const int2 in_dim,
                                  int offset) {
  for (int mi = blockIdx.x; mi < in_dim.x; mi += gridDim.x) {
    for (int ni = threadIdx.x; ni < in_dim.y; ni += blockDim.x) {
      out[mi * out_dim.y + offset + ni] = in[mi * in_dim.y + ni];
    }
  }
}

template <typename T>
__global__ void concat_bwd_kernel(T* out, const int2 out_dim, T* in, const int2 in_dim,
                                  int offset) {
  for (int mi = blockIdx.x; mi < in_dim.x; mi += gridDim.x) {
    for (int ni = threadIdx.x; ni < in_dim.y; ni += blockDim.x) {
      in[mi * in_dim.y + ni] = out[mi * out_dim.y + offset + ni];
    }
  }
}

// ROCm port (May 2026): vectorized concat fwd/bwd using 16-byte loads.
// Profile (bs8x rocprofv3 trace) shows the scalar concat_*_kernel above
// taking 4.9 % of iter (4 calls/iter x 327 us/call = 1.31 ms/iter).
// At per-GPU batch 55296 with concat widths [128, 3328, 3456], each row
// is 256/6656/6912 bytes of __half = 16/416/432 dwords, all multiples
// of 8 = 16 bytes. We use int4 loads (= 8x __half) for the aligned
// fast-path; falls back to scalar for unaligned widths.
template <typename T>
__global__ void concat_fwd_kernel_vec8(T* out, const int2 out_dim, const T* in,
                                       const int2 in_dim, int offset) {
  // Each thread copies 8 half (= 16 bytes = 1 int4) per inner iteration.
  // This is ~4-8x faster on AMD CDNA3 vs single-half copies because:
  //   - one global-load instead of 8 (saturates HBM bandwidth)
  //   - one global-store instead of 8 (same)
  //   - 8x fewer instructions per element
  // For sizeof(T)==4 (float) the dispatch in concat_use_vec8 returns false
  // so this kernel is never launched; compile-time guard keeps the int4
  // load/store path out of the float instantiation.
  if constexpr (sizeof(T) == 2) {
    constexpr int VEC = 16 / sizeof(T);   // 8 elements per vec
    const int in_w_vec = in_dim.y / VEC;
    for (int mi = blockIdx.x; mi < in_dim.x; mi += gridDim.x) {
      const int4* in_row  = reinterpret_cast<const int4*>(in  + mi * in_dim.y);
      int4*       out_row = reinterpret_cast<int4*>      (out + mi * out_dim.y + offset);
      for (int vi = threadIdx.x; vi < in_w_vec; vi += blockDim.x) {
        out_row[vi] = in_row[vi];
      }
    }
  }
}

template <typename T>
__global__ void concat_bwd_kernel_vec8(T* out, const int2 out_dim, T* in, const int2 in_dim,
                                       int offset) {
  if constexpr (sizeof(T) == 2) {
    constexpr int VEC = 16 / sizeof(T);
    const int in_w_vec = in_dim.y / VEC;
    for (int mi = blockIdx.x; mi < in_dim.x; mi += gridDim.x) {
      int4*       in_row  = reinterpret_cast<int4*>      (in  + mi * in_dim.y);
      const int4* out_row = reinterpret_cast<const int4*>(out + mi * out_dim.y + offset);
      for (int vi = threadIdx.x; vi < in_w_vec; vi += blockDim.x) {
        in_row[vi] = out_row[vi];
      }
    }
  }
}

}  // namespace

template <typename T>
ConcatLayer<T>::ConcatLayer(const std::vector<core23::Tensor>& input_tensors,
                            core23::Tensor& output_tensor,
                            const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer(input_tensors, {}, gpu_resource) {
  try {
    if (input_tensors_.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Empty input tensors");
    }
    size_t n_input_tensors = input_tensors_.size();
    int64_t height = 0;
    int64_t new_width = 0;
    for (size_t i = 0; i < n_input_tensors; i++) {
      auto cur_in_shape = input_tensors_[i].shape();
      if (i != 0) {
        auto first_in_shape = input_tensors_[0].shape();
        if (cur_in_shape.size(0) != first_in_shape.size(0)) {
          HCTR_OWN_THROW(Error_t::WrongInput, "All the input tensors must have the same height");
        }
      }
      if (cur_in_shape.dims() != 2) {
        HCTR_OWN_THROW(Error_t::WrongInput, "Only 2D tensors can be concatenated");
      }
      if (i == 0) {
        height = cur_in_shape.size(0);
      }
      new_width += cur_in_shape.size(1);
    }
    core23::BufferParams buf_p{.channel = GetBlobsBufferChannel()};

    output_tensor = core23::Tensor(
        input_tensors_[0].my_params().shape({height, new_width}).buffer_params(buf_p));
    output_tensors_.push_back(output_tensor);
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

// ROCm port: dispatch to the int4-vectorized fwd/bwd kernels when the row
// width (in_dim.y) and the offset within the output row are both multiples
// of (16 / sizeof(T)) elements, i.e. naturally 16-byte-aligned. For DLRM-DCNv2
// at FP16 the concat slabs are widths {128, 3328, 3456} which are all
// multiples of 8 -- the fast path always engages. Env knob HCTR_CONCAT_KERNEL=v1
// reverts to the original scalar kernel (kept for diagnostic A/B testing).
template <typename T>
static inline bool concat_use_vec8(const int row_width, const int row_offset) {
  if (sizeof(T) != 2) return false;  // vec8 path is __half-only
  const int VEC = 16 / static_cast<int>(sizeof(T));
  if ((row_width  % VEC) != 0) return false;
  if ((row_offset % VEC) != 0) return false;
  static const bool force_v1 = []() {
    const char* env = getenv("HCTR_CONCAT_KERNEL");
    return env && std::string(env) == "v1";
  }();
  return !force_v1;
}

template <typename T>
void ConcatLayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());
  auto stream = get_gpu().get_stream();

  int n_input_tensors = input_tensors_.size();
  int block_size = 256;
  int n_blocks = get_gpu().get_sm_count() * 8;
  auto& output_tensor = output_tensors_[0];
  T* out = output_tensor.template data<T>();
  const int2 out_dim = {static_cast<int>(output_tensor.shape().size(0)),
                        static_cast<int>(output_tensor.shape().size(1))};
  int offset = 0;
  for (auto& input_tensor : input_tensors_) {
    T* in = input_tensor.template data<T>();
    const int2 in_dim = {static_cast<int>(input_tensor.shape().size(0)),
                         static_cast<int>(input_tensor.shape().size(1))};

    if (concat_use_vec8<T>(in_dim.y, offset)) {
      concat_fwd_kernel_vec8<<<n_blocks, block_size, 0, stream>>>(out, out_dim, in, in_dim, offset);
    } else {
      concat_fwd_kernel<<<n_blocks, block_size, 0, stream>>>(out, out_dim, in, in_dim, offset);
    }
    offset += in_dim.y;
  }
}

template <typename T>
void ConcatLayer<T>::bprop() {
  CudaDeviceContext context(get_device_id());
  auto stream = get_gpu().get_stream();

  int block_size = 256;
  int n_blocks = get_gpu().get_sm_count() * 8;
  auto& output_tensor = output_tensors_[0];
  T* out = output_tensor.template data<T>();
  const int2 out_dim = {static_cast<int>(output_tensor.shape().size(0)),
                        static_cast<int>(output_tensor.shape().size(1))};
  int grid_size = std::min(out_dim.x, n_blocks);
  int offset = 0;
  for (std::size_t i = 0; i < input_tensors_.size(); i++) {
    auto& input_tensor = input_tensors_[i];
    T* in = input_tensor.template data<T>();
    const int2 in_dim = {static_cast<int>(input_tensor.shape().size(0)),
                         static_cast<int>(input_tensor.shape().size(1))};

    if (concat_use_vec8<T>(in_dim.y, offset)) {
      concat_bwd_kernel_vec8<<<grid_size, block_size, 0, stream>>>(out, out_dim, in, in_dim, offset);
    } else {
      concat_bwd_kernel<<<grid_size, block_size, 0, stream>>>(out, out_dim, in, in_dim, offset);
    }
    offset += in_dim.y;
  }
}

template class ConcatLayer<float>;
template class ConcatLayer<__half>;

}  // namespace HugeCTR
