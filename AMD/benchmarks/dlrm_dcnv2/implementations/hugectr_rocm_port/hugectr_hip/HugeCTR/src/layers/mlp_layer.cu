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

#include <hctr_tracing.hpp>
#include <layers/mlp_layer.hpp>
#include <perfetto_emitter.hpp>
#include <type_traits>
#include <unordered_map>
#include <vector>
// ROCm port Phase 14i.5 (2026-05-13): CK-Tile fused MLP wrapper.
// Public API in cktile_mlp_kernel.hpp (extern "C", no CK-Tile templates exposed).
// Implementation in HugeCTR/src/layers/cktile_mlp_kernel.cu compiled SEPARATELY
// with isolated flags (no -fopenmp, no warp-compat shim) so CK-Tile templates
// instantiate cleanly. Routed when HCTR_USE_CK_TILE_MLP=1 env var is set.
#include <cktile_mlp_kernel.hpp>

namespace HugeCTR {

// HugeCTR ROCm port: explicit instantiations moved to end-of-file (was here at
// the top, BEFORE the definitions, which is invalid C++ that amdclang rejects
// silently by emitting an empty .o).

template <typename T>
MLPLayer<T>::MLPLayer(const std::vector<core23::Tensor>& bottom_tensors,
                      const std::vector<core23::Tensor>& top_tensors,
                      const std::vector<int64_t>& num_outputs,
                      const std::shared_ptr<GPUResource>& gpu_resource,
                      const std::vector<Activation_t>& acts, const std::vector<bool>& use_bias,
                      std::vector<Initializer_t> initializer_types, bool skip_head_dgrad,
                      bool async_wgrad, bool fuse_wb, bool enable_tf32_compute)
    : TrainableLayer<T>(bottom_tensors, top_tensors, gpu_resource, initializer_types),
      num_outputs_(num_outputs),
      acts_(acts),
      use_bias_(use_bias),
      skip_head_dgrad_(skip_head_dgrad),
      async_wgrad_(async_wgrad),
      fuse_wb_(fuse_wb),
      enable_tf32_compute_(enable_tf32_compute),
      event_overlap_created_(false) {
  int num_layers = num_outputs.size();
  train_tensors_.resize(num_layers);
  mask_tensors_.resize(num_layers);
  output_mask_.resize(num_layers);
  dact_tensors_.resize(num_layers);
  layer_desc_.resize(num_layers);
  layer_algo_.resize(num_layers);

  // Phase 14n.2: per-layer scratch buffers for CK-Tile weight pre-transpose.
  // Lazy-allocated on first use; freed in destructor.
  cktile_kernel_T_buffers_.assign(num_layers, nullptr);
  cktile_kernel_T_allocated_.assign(num_layers, false);

  for (int i = 0; i < num_layers; i++) {
    const auto& bottom_tensor_dim =
        i == 0 ? this->input_tensors_[0].shape() : train_tensors_[i - 1].shape();
    int64_t batch_size = bottom_tensor_dim.size(0);
    int64_t input_size = bottom_tensor_dim.size(1);
    int64_t output_size = num_outputs[i];

    core23::Shape kernel_dim = {input_size, output_size};
    core23::Shape bias_dim = {1, output_size};

    this->set_weight(i * 2, kernel_dim);
    kernels_.push_back(this->get_weight(i * 2));
    this->set_weight(i * 2 + 1, bias_dim);
    biases_.push_back(this->get_weight(i * 2 + 1));
    this->set_wgrad(i * 2, kernel_dim);
    kernels_grad_.push_back(this->get_wgrad(i * 2));
    this->set_wgrad(i * 2 + 1, bias_dim);
    db_tensors_.push_back(this->get_wgrad(i * 2 + 1));

    const auto& train_in_tensor = i == 0 ? this->input_tensors_[0] : train_tensors_[i - 1];
    int64_t num_output = num_outputs[i];

    core23::BufferParams buffer_params = {};
    buffer_params.channel = GetBlobsBufferChannel();
    core23::Device device(core23::DeviceType::GPU, gpu_resource->get_device_id());

    if (i != num_layers - 1) {
      core23::Shape shape({train_in_tensor.shape().size(0), num_output});
      auto data_type = core23::ToScalarType<T>::value;
      train_tensors_[i] = core23::Tensor(
          core23::TensorParams().data_type(data_type).shape(shape).device(device).buffer_params(
              buffer_params));
      if (acts_[i] == Activation_t::Relu) {
        mask_tensors_[i] = core23::Tensor(
            core23::TensorParams().data_type(data_type).shape(shape).device(device).buffer_params(
                buffer_params));
        dact_tensors_[i] = core23::Tensor(
            core23::TensorParams().data_type(data_type).shape(shape).device(device).buffer_params(
                buffer_params));
      }
    } else {
      core23::Shape shape({train_in_tensor.shape().size(0), num_output});
      auto data_type = core23::ToScalarType<T>::value;

      train_tensors_[i] = this->output_tensors_[0];
      if (this->output_tensors_.size() == 1) {
        if (acts_[i] == Activation_t::Relu) {
          mask_tensors_[i] = core23::Tensor(
              core23::TensorParams().data_type(data_type).shape(shape).device(device).buffer_params(
                  buffer_params));
        }
        dact_tensors_[i] = core23::Tensor(
            core23::TensorParams().data_type(data_type).shape(shape).device(device).buffer_params(
                buffer_params));
      }
    }

    output_mask_[i] = (acts_[i] == Activation_t::Relu) && (i != num_layers - 1);
  }
}

template <typename T>
void MLPLayer<T>::fprop(bool is_train) {
  HugeCTR::tracing::ScopedRange _scope("MLPLayer::fprop");
  CudaDeviceContext context(this->get_device_id());
  int num_layers = num_outputs_.size();
  // Phase 20.6 (2026-05-17): per-MLP-sub-layer GpuPhase tracing. One
  // event per sub-layer (NV-equivalent granularity: mlp_fwd L1, L3, etc.).
  // Lazily registered, capture-safe (hipStreamIsCapturing check in
  // ScopedGpuPhase skips during graph capture).
  using HugeCTR::tracing::GpuPhase;
  using HugeCTR::tracing::PerfettoEmitter;
  using HugeCTR::tracing::ScopedGpuPhase;
  static thread_local std::unordered_map<const MLPLayer<T>*, std::vector<GpuPhase*>> fwd_subs;
  const bool nt_on = HugeCTR::tracing::native_trace_enabled();
  std::vector<GpuPhase*>* subs = nullptr;
  if (nt_on) {
    auto it = fwd_subs.find(this);
    if (it == fwd_subs.end()) {
      auto& v = fwd_subs[this];
      v.reserve(num_layers);
      for (int j = 0; j < num_layers; ++j) {
        char nm[48];
        std::snprintf(nm, sizeof(nm), "mlp/sub%d_fprop", j);
        v.push_back(PerfettoEmitter::instance().register_phase(this->get_gpu().get_local_id(), nm));
      }
      subs = &v;
      // NOTE: MLPLayer is only used when HCTR_USE_FUSED_MLP=1. The default
      // DLRM-DCNv2 config uses Layer_t.InnerProduct + Layer_t.ReLU, in
      // which case this branch never fires. Per-FC-layer coverage is
      // already provided by core23_network.cpp's L/* phases.
    } else {
      subs = &it->second;
    }
  }
  for (int i = 0; i < num_layers; i++) {
    GpuPhase* sub_phase = (subs && i < (int)subs->size()) ? (*subs)[i] : nullptr;
    ScopedGpuPhase _sub(sub_phase, this->get_gpu().get_stream(), "default");
    const T* kernel = kernels_[i].template data<T>();
    const T* bottom =
        i == 0 ? this->input_tensors_[0].template data<T>() : train_tensors_[i - 1].template data<T>();
    T* top_fprop = train_tensors_[i].template data<T>();

    // ROCm port Phase 14i.5 (2026-05-13): CK-Tile fused fwd path.
    // Routes to the CK-Tile kernel when HCTR_USE_CK_TILE_MLP=1 + T=__half + shape supported.
    //
    // STATUS: kernel COMPILES + INTEGRATES + RUNS end-to-end (verified). However,
    // numerical output diverges from baseline (loss 0.605 vs 0.283) due to a tensor
    // LAYOUT mismatch between HCTR's row-major weight tensor (K×N row-major =
    // N×K col-major) and our CK-Tile config's expected col-major B with stride K.
    // Debugging in progress — fix is either (a) custom transpose layer, or
    // (b) different CK-Tile pipeline that accepts row-major B (RowMajor B with
    // current pipeline tested but produced zeros — likely policy specialization
    // gap). See README §Phase 14i.5b for status.
    //
    // Routing block kept compiled-in but gated additionally on HCTR_CK_TILE_FORCE
    // env var (NOT just HCTR_USE_CK_TILE_MLP) to avoid breaking production runs.
    // Once layout fix lands, set HCTR_CK_TILE_FORCE=1 to engage.
    bool used_cktile = false;
    if constexpr (std::is_same<T, __half>::value) {
      static const bool force = []() {
        const char* env = std::getenv("HCTR_CK_TILE_FORCE");
        return env != nullptr && std::atoi(env) == 1;
      }();
      if (force && cktile::is_enabled()) {
        const auto& bot_shape = (i == 0 ? this->input_tensors_[0].shape() : train_tensors_[i - 1].shape());
        const int64_t M = bot_shape.size(0);
        const int64_t K = bot_shape.size(1);
        const int64_t N = num_outputs_[i];
        if (cktile::is_shape_supported((int)M, (int)N, (int)K) && use_bias_[i]) {
          const __half* bias_ptr = biases_[i].template data<__half>();
          auto stream = this->get_gpu().get_stream();

          // Phase 14n.2 weight pre-transpose: HCTR's kernel is K×N row-major;
          // CK-Tile expects K×N col-major. Per-layer scratch buffer was
          // allocated in initialize() (before graph capture starts).
          // Transpose runs every fwd (~10us/layer), buffer reused across iters.
          __half* kernel_T = cktile_kernel_T_buffers_[i];
          if (kernel_T != nullptr) {
            hctr_cktile_transpose_weight_fp16(
                reinterpret_cast<const __half*>(kernel), kernel_T,
                (int)K, (int)N, stream);

          hipError_t err = (acts_[i] == Activation_t::Relu)
              ? hctr_cktile_gemm_bias_relu_fp16(reinterpret_cast<const __half*>(bottom),
                                                 kernel_T,
                                                 bias_ptr,
                                                 reinterpret_cast<__half*>(top_fprop),
                                                 (int)M, (int)N, (int)K, stream)
              : hctr_cktile_gemm_bias_fp16(reinterpret_cast<const __half*>(bottom),
                                            kernel_T,
                                            bias_ptr,
                                            reinterpret_cast<__half*>(top_fprop),
                                            (int)M, (int)N, (int)K, stream);
          if (err == hipSuccess) {
            used_cktile = true;
            // Phase 14n.7 (2026-05-14): legacy layer_functors_.fprop uses
            // hipBLASLt's RELU_AUX_BIAS epilogue which writes a BIT-PACKED
            // uint8_t mask (1 bit per element, bit (i & 7) of byte at
            // (i/8) + j*aux_ld in cublas col-major view). HCTR's bprop_drelu
            // kernels read this format. We replicate via the
            // hctr_cktile_pack_relu_mask_fp16 helper (validated independently
            // — see /home/chcai/cktile_mask_test.cpp; matches CPU reference
            // exactly on 884736 bytes for top[1] shape).
            if (acts_[i] == Activation_t::Relu && output_mask_[i]) {
              void* mask_out = mask_tensors_[i].template data<T>();
              int aux_ld = (int)((N + 7) / 8);  // bytes per col in cublas col-major view
              (void)hctr_cktile_pack_relu_mask_fp16(
                  reinterpret_cast<const __half*>(top_fprop),
                  mask_out, (int)M, (int)N, aux_ld, stream);
            }
          }
          }
        }
      }
    }
    if (!used_cktile) {
      layer_functors_.fprop(kernel, bottom, top_fprop, layer_desc_[i], layer_algo_[i],
                            this->get_gpu().get_cublaslt_handle(), this->get_gpu().get_stream());
    }
    if (i == num_layers - 1 && acts_[i] == Activation_t::Relu) {
      T* mask_out = mask_tensors_[i].template data<T>();
      int64_t len = train_tensors_[i].num_elements();
      HCTR_LIB_THROW(hipMemcpyAsync(mask_out, top_fprop, len * sizeof(T), hipMemcpyDeviceToDevice,
                                     this->get_gpu().get_stream()));
    }
  }
}

template <typename T>
void MLPLayer<T>::bprop() {
  HugeCTR::tracing::ScopedRange _scope("MLPLayer::bprop");
  CudaDeviceContext context(this->get_device_id());

  int num_layers = num_outputs_.size();
  // Phase 20.6: per-sub-layer bprop GpuPhase.
  using HugeCTR::tracing::GpuPhase;
  using HugeCTR::tracing::PerfettoEmitter;
  using HugeCTR::tracing::ScopedGpuPhase;
  static thread_local std::unordered_map<const MLPLayer<T>*, std::vector<GpuPhase*>> bwd_subs;
  const bool nt_on = HugeCTR::tracing::native_trace_enabled();
  std::vector<GpuPhase*>* subs = nullptr;
  if (nt_on) {
    auto it = bwd_subs.find(this);
    if (it == bwd_subs.end()) {
      auto& v = bwd_subs[this];
      v.reserve(num_layers);
      for (int j = 0; j < num_layers; ++j) {
        char nm[48];
        std::snprintf(nm, sizeof(nm), "mlp/sub%d_bprop", j);
        v.push_back(PerfettoEmitter::instance().register_phase(this->get_gpu().get_local_id(), nm));
      }
      subs = &v;
    } else {
      subs = &it->second;
    }
  }
  for (int i = num_layers - 1; i >= 0; i--) {
    GpuPhase* sub_phase = (subs && i < (int)subs->size()) ? (*subs)[i] : nullptr;
    ScopedGpuPhase _sub(sub_phase, this->get_gpu().get_stream(), "default");
    const auto& bottom_tensor_dim =
        i == 0 ? this->input_tensors_[0].shape() : train_tensors_[i - 1].shape();
    int64_t batch_size = bottom_tensor_dim.size(0);
    int64_t top_size = num_outputs_[i];

    const T* kernel = kernels_[i].template data<T>();
    const T* train_top = train_tensors_[i].template data<T>();

    // Only the last layer needs the mask of itself to get the grad.
    const T* mask_top = (i == num_layers - 1 && acts_[i] == Activation_t::Relu)
                            ? mask_tensors_[i].template data<T>()
                            : nullptr;

    T* grad_top =
        acts_[i] == Activation_t::None ? train_tensors_[i].template data<T>() : dact_tensors_[i].template data<T>();
    T* kernel_grad = kernels_grad_[i].template data<T>();
    T* bottom =
        i == 0 ? this->input_tensors_[0].template data<T>() : train_tensors_[i - 1].template data<T>();
    bool enable_async_wgrad = async_wgrad_;
    T* bottom_bprop = nullptr;
    if (i != 0) {
      bottom_bprop = acts_[i - 1] == Activation_t::None ? train_tensors_[i - 1].template data<T>()
                                                        : dact_tensors_[i - 1].template data<T>();
    } else {
      if (this->input_tensors_.size() == 1) {
        // train_in_tensor
        bottom_bprop = this->input_tensors_[0].template data<T>();
        enable_async_wgrad = false;
      } else {
        bottom_bprop = this->input_tensors_[1].template data<T>();
      }
    }
    // Phase 14p.2 (2026-05-14): CK-Tile dgrad path. Math:
    //   bottom_bprop[m, n] = sum_k grad_top[m, k] * kernel[n, k]
    // K_dgrad = top_size (out_size = contraction dim), N_dgrad = bottom_size.
    // CK-Tile config supports M>=64, N_dgrad>=64, K_dgrad>=32. Layers that fail
    // the predicate fall back to hipBLASLt's dgrad (skip_dgrad stays false).
    bool used_cktile_dgrad = false;
    if constexpr (std::is_same<T, __half>::value) {
      static const bool ck_dgrad_on = []() {
        const char* env1 = std::getenv("HCTR_USE_CK_TILE_MLP");
        const char* env2 = std::getenv("HCTR_USE_CK_TILE_DGRAD");
        return env1 && std::atoi(env1) == 1 && env2 && std::atoi(env2) == 1;
      }();
      const bool head_skipped_anyway = (i == 0 && skip_head_dgrad_);
      const int64_t bottom_size_i = bottom_tensor_dim.size(1);
      if (ck_dgrad_on && !head_skipped_anyway && bottom_bprop != nullptr &&
          cktile::is_shape_supported((int)batch_size, (int)bottom_size_i, (int)top_size)) {
        // Run hipBLASLt wgrad+drelu first (skip_dgrad=true to avoid double-write
        // of bottom_bprop). Then CK-Tile dgrad fills bottom_bprop.
        layer_functors_.bprop(kernel, bottom, train_top, mask_top, batch_size * top_size, grad_top,
                              bottom_bprop, kernel_grad, layer_desc_[i], layer_algo_[i],
                              this->get_gpu().get_cublaslt_handle(), this->get_gpu().get_stream(),
                              this->get_gpu().get_comp_overlap_stream(), event_overlap_,
                              enable_async_wgrad, /*skip_dgrad=*/true);
        hipError_t err = hctr_cktile_gemm_dgrad_fp16(
            reinterpret_cast<const __half*>(grad_top),
            reinterpret_cast<const __half*>(kernel),
            reinterpret_cast<__half*>(bottom_bprop),
            (int)batch_size, (int)bottom_size_i, (int)top_size,
            this->get_gpu().get_stream());
        if (err == hipSuccess) {
          used_cktile_dgrad = true;
        } else {
          // CK-Tile rejected — re-run hipBLASLt dgrad as fallback. No double
          // wgrad cost since wgrad already ran above.
          // dgrad GEMM (matches bprop_dgrad in fused_fc_layer_functors.cu line 164):
          //   bottom_bprop = grad_top @ kernel  (HCTR's bprop signature; alpha=1, beta=0)
          // Done by re-calling layer_functors_.bprop with mask=nullptr (already
          // applied) + skip_dgrad=false would double-do wgrad. Easier:
          // execute a one-shot hipblasGemmEx fallback inline.
          const __half alpha = __float2half(1.f), beta = __float2half(0.f);
          (void)hipblasGemmEx(
              this->get_gpu().get_cublas_handle(), HIPBLAS_OP_T, HIPBLAS_OP_N,
              (int)bottom_size_i, (int)batch_size, (int)top_size,
              &alpha,
              reinterpret_cast<const __half*>(kernel),    HIP_R_16F, (int)top_size,
              reinterpret_cast<const __half*>(grad_top),  HIP_R_16F, (int)top_size,
              &beta,
              reinterpret_cast<__half*>(bottom_bprop),    HIP_R_16F, (int)bottom_size_i,
              HIPBLAS_COMPUTE_16F, HIPBLAS_GEMM_DEFAULT);
        }
      }
    }
    if (!used_cktile_dgrad) {
      layer_functors_.bprop(kernel, bottom, train_top, mask_top, batch_size * top_size, grad_top,
                            bottom_bprop, kernel_grad, layer_desc_[i], layer_algo_[i],
                            this->get_gpu().get_cublaslt_handle(), this->get_gpu().get_stream(),
                            this->get_gpu().get_comp_overlap_stream(), event_overlap_,
                            enable_async_wgrad, i == 0 ? skip_head_dgrad_ : false);
    }
  }

  if (async_wgrad_) {
    HCTR_LIB_THROW(hipEventRecord(event_overlap_, this->get_gpu().get_comp_overlap_stream()));
    HCTR_LIB_THROW(hipStreamWaitEvent(this->get_gpu().get_stream(), event_overlap_));
  }
}

template <typename T>
void MLPLayer<T>::initialize() {
  CudaDeviceContext context(this->get_device_id());

  HCTR_LIB_THROW(hipEventCreate(&event_overlap_));
  event_overlap_created_ = true;

  int num_layers = num_outputs_.size();
  for (int i = 0; i < num_layers; i++) {
    const auto& bottom_tensor_dim =
        i == 0 ? this->input_tensors_[0].shape() : train_tensors_[i - 1].shape();
    int64_t batch_size = bottom_tensor_dim.size(0);
    int64_t input_size = bottom_tensor_dim.size(1);
    int64_t output_size = num_outputs_[i];

    const T* bias_ptr = nullptr;
    if (use_bias_[i]) {
      bias_ptr = biases_[i].template data<T>();
    }

    T* mask_out_ptr = nullptr;
    bool output_mask = output_mask_[i];
    if (output_mask) {
      mask_out_ptr = mask_tensors_[i].template data<T>();
    }
    layer_desc_[i].set_fprop_attr(bias_ptr, acts_[i], mask_out_ptr, batch_size, input_size,
                                  output_size, enable_tf32_compute_);

    T* mask_in_ptr = nullptr;
    if (i > 0) {
      if (acts_[i - 1] == Activation_t::Relu) {
        mask_in_ptr = mask_tensors_[i - 1].template data<T>();
      }
    }
    T* dbias_bottom_ptr = nullptr;
    T* dbias_top_ptr = nullptr;
    // If there is no ReLu, then dbias should be fused with wgrad.
    // Compute the bias gradient for this layer.
    if (fuse_wb_ || i == num_layers - 1 || acts_[i] == Activation_t::None) {
      if (use_bias_[i]) {
        dbias_top_ptr = db_tensors_[i].template data<T>();
      }
    }
    // Compute the bias gradient for bottom layer. For the last layer of MLP, it should compute both
    // gradients.
    if (!fuse_wb_ && i > 0 && use_bias_[i - 1] && acts_[i - 1] != Activation_t::None) {
      dbias_bottom_ptr = db_tensors_[i - 1].template data<T>();
    }
    layer_desc_[i].set_bprop_attr(dbias_bottom_ptr, dbias_top_ptr, mask_in_ptr, batch_size,
                                  input_size, output_size, enable_tf32_compute_);
    layer_algo_[i].set_fprop_algo(layer_desc_[i], this->get_gpu().get_cublaslt_handle());
    layer_algo_[i].set_bprop_algo(layer_desc_[i], this->get_gpu().get_cublaslt_handle());

    // Phase 14n.2: pre-allocate weight pre-transpose scratch buffer for
    // CK-Tile MLP kernel. Only allocate when CK-Tile is enabled (saves
    // memory in the common case). Allocation happens BEFORE graph capture.
    if constexpr (std::is_same<T, __half>::value) {
      static const bool ck_tile_on = []() {
        const char* env1 = std::getenv("HCTR_USE_CK_TILE_MLP");
        const char* env2 = std::getenv("HCTR_CK_TILE_FORCE");
        return env1 != nullptr && std::atoi(env1) == 1
            && env2 != nullptr && std::atoi(env2) == 1;
      }();
      if (ck_tile_on) {
        size_t bytes = sizeof(__half) * input_size * output_size;
        HCTR_LIB_THROW(hipMalloc(&cktile_kernel_T_buffers_[i], bytes));
        cktile_kernel_T_allocated_[i] = true;
      }
      // Phase 14p.2: pre-allocate zero-bias scratch for CK-Tile dgrad
      // (HCTR_USE_CK_TILE_DGRAD=1). Sized to max in_size we'll see; for
      // DLRM-DCNv2 the max is 1024 (top[1] in_size). Use 4096 as safety
      // margin. Idempotent across MLPLayer instances and across devices
      // (per-device buffer indexed inside cktile_mlp_kernel.cu).
      static const bool ck_dgrad_on = []() {
        const char* env1 = std::getenv("HCTR_USE_CK_TILE_MLP");
        const char* env2 = std::getenv("HCTR_USE_CK_TILE_DGRAD");
        return env1 && std::atoi(env1) == 1 && env2 && std::atoi(env2) == 1;
      }();
      if (ck_dgrad_on) {
        (void)hctr_cktile_init_dgrad_zero_bias(/*max_n=*/4096);
      }
    }
  }
}

template <typename T>
void MLPLayer<T>::search_algorithm() {
  CudaDeviceContext context(this->get_device_id());
  int num_layers = num_outputs_.size();
  for (int i = 0; i < num_layers; i++) {
    T* kernel = kernels_[i].template data<T>();
    T* bottom =
        i == 0 ? this->input_tensors_[0].template data<T>() : train_tensors_[i - 1].template data<T>();
    T* top = train_tensors_[i].template data<T>();

    const auto& bottom_tensor_dim =
        i == 0 ? this->input_tensors_[0].shape() : train_tensors_[i - 1].shape();
    int64_t batch_size = bottom_tensor_dim.size(0);
    int64_t input_size = bottom_tensor_dim.size(1);
    int64_t output_size = num_outputs_[i];

    layer_functors_.search_algorithm(
        bottom, top, kernel, batch_size, input_size, output_size, layer_desc_[i], layer_algo_[i],
        this->get_gpu().get_cublaslt_handle(), this->get_gpu().get_stream());
  }
}

template <typename T>
std::unique_ptr<DataSimulator> MLPLayer<T>::get_uniform_initializer(const int index) {
  int i = index / 2;
  int64_t bottom_dim =
      i == 0 ? this->input_tensors_[0].shape().size(1) : train_tensors_[i - 1].shape().size(1);
  float limit = sqrt(1.0f / (bottom_dim));
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

template <typename T>
std::unique_ptr<DataSimulator> MLPLayer<T>::get_xavier_uniform_initializer(const int index) {
  int i = index / 2;
  int64_t bottom_dim =
      i == 0 ? this->input_tensors_[0].shape().size(1) : train_tensors_[i - 1].shape().size(1);
  int64_t top_dim = train_tensors_[i].shape().size(1);
  // fan_avg for weight
  // fan_out for bias
  auto fan_mode = i % 2 ? data_simu::Mode_t::Fan_out : data_simu::Mode_t::Fan_avg;
  return std::make_unique<VarianceScalingSimulator>(
      1.f, fan_mode, data_simu::Distribution_t::Uniform, bottom_dim, top_dim);
}
template <typename T>
std::unique_ptr<DataSimulator> MLPLayer<T>::get_xavier_norm_initializer(const int index) {
  int i = index / 2;
  int64_t bottom_dim =
      i == 0 ? this->input_tensors_[0].shape().size(1) : train_tensors_[i - 1].shape().size(1);
  int64_t top_dim = train_tensors_[i].shape().size(1);
  // fan_avg for weight
  // fan_out for bias
  auto fan_mode = i % 2 ? data_simu::Mode_t::Fan_out : data_simu::Mode_t::Fan_avg;
  return std::make_unique<VarianceScalingSimulator>(1.f, fan_mode, data_simu::Distribution_t::Norm,
                                                    bottom_dim, top_dim);
}

template <typename T>
std::unique_ptr<DataSimulator> MLPLayer<T>::get_default_initializer(const int index) {
  return this->get_uniform_initializer(index);
}

// HugeCTR ROCm port: explicit instantiations to ensure all symbols emit.
template class MLPLayer<float>;
template class MLPLayer<__half>;

}  // namespace HugeCTR
