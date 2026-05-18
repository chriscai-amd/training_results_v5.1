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

#include <array>
#include <layers/fully_connected_layer_half.hpp>
#include <perfetto_emitter.hpp>
#include <unordered_map>
#include <utility>
#include <utils.cuh>
#include <utils.hpp>

namespace HugeCTR {

FullyConnectedLayer<__half>::FullyConnectedLayer(const core23::Tensor& bottom_tensor,
                                                 const core23::Tensor& top_tensor,
                                                 const std::shared_ptr<GPUResource>& gpu_resource,
                                                 std::vector<Initializer_t> initializer_types)
    : TrainableLayer<__half>({bottom_tensor}, {top_tensor}, gpu_resource, initializer_types),
      falgo_b_(HIPBLAS_GEMM_DEFAULT),
      falgo_k_(HIPBLAS_GEMM_DEFAULT),
      balgo_b_(HIPBLAS_GEMM_DEFAULT),
      balgo_k_(HIPBLAS_GEMM_DEFAULT),
      balgo_x_(HIPBLAS_GEMM_DEFAULT) {
  const auto& bottom_tensor_dim = bottom_tensor.shape();
  const auto& top_tensor_dim = top_tensor.shape();

  if (bottom_tensor_dim.dims() != top_tensor_dim.dims()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "input or output tensor don't have same dimensions");
  }
  int64_t in_batch_size = 1;
  int64_t out_batch_size = 1;
  int64_t input_size = bottom_tensor_dim.size(bottom_tensor_dim.dims() - 1);
  int64_t output_size = top_tensor_dim.size(top_tensor_dim.dims() - 1);

  for (int64_t idx = 0; idx < bottom_tensor_dim.dims() - 1; idx++) {
    in_batch_size = in_batch_size * bottom_tensor_dim.size(idx);
    out_batch_size = out_batch_size * top_tensor_dim.size(idx);
  }

  if (in_batch_size != out_batch_size) {
    HCTR_OWN_THROW(Error_t::WrongInput, "size of input / output tensor doesn't match");
  }

  core23::Shape kernel_dim = {input_size, output_size};
  core23::Shape bias_dim = {1, output_size};
  core23::Shape identity_dim = {1, in_batch_size};

  this->set_weight(0, kernel_dim);
  this->set_weight(1, bias_dim);
  this->set_wgrad(0, kernel_dim);
  this->set_wgrad(1, bias_dim);

  core23::BufferParams blobs_buffer_params = {};
  blobs_buffer_params.channel = GetBlobsBufferChannel();
  core23::Device device(core23::DeviceType::GPU, gpu_resource->get_device_id());

  identity_tensor_ = core23::Tensor(core23::TensorParams()
                                        .data_type(core23::ToScalarType<__half>::value)
                                        .shape(identity_dim)
                                        .device(device)
                                        .buffer_params(blobs_buffer_params));
}

void FullyConnectedLayer<__half>::fprop(bool is_train) {
  CudaDeviceContext context(get_device_id());

  const __half* kernel = this->get_weight(0).template data<__half>();
  const __half* bias = this->get_weight(1).template data<__half>();
  const __half* bottom = get_bottom_tensor(is_train).template data<__half>();
  const __half* identity = identity_tensor_.template data<__half>();
  auto top_tensor = this->output_tensors_[0];
  __half* top = top_tensor.template data<__half>();

  const auto& bottom_tensor_dim = get_bottom_tensor(is_train).shape();
  const auto& top_tensor_dim = top_tensor.shape();

  int64_t in_batch_size = 1;
  int64_t input_size = bottom_tensor_dim.size(bottom_tensor_dim.dims() - 1);
  int64_t output_size = top_tensor_dim.size(top_tensor_dim.dims() - 1);

  for (int64_t idx = 0; idx < bottom_tensor_dim.dims() - 1; idx++) {
    in_batch_size = in_batch_size * bottom_tensor_dim.size(idx);
  }

  // Phase 20.6 + 20.7 (2026-05-17): per-GEMM GpuPhase tracing with
  // PyTorch-profiler-style M/N/K metadata. NV captures mlp_fwd as
  // separate kernels (bias GEMM + kernel GEMM). One pair of GpuPhases
  // per FC layer instance, lazily registered. Each FC layer emits 2
  // fprop events per iter with GEMM shape inlined into args.
  using HugeCTR::tracing::GpuPhase;
  using HugeCTR::tracing::PerfettoEmitter;
  using HugeCTR::tracing::ScopedGpuPhase;
  static thread_local std::unordered_map<const FullyConnectedLayer<__half>*,
                                          std::pair<GpuPhase*, GpuPhase*>> fwd_gemm_phases;
  GpuPhase* p_bias_gemm = nullptr;
  GpuPhase* p_kernel_gemm = nullptr;
  // Phase 20.7: per-GEMM events require DETAIL >= 2 (highest fidelity).
  if (HugeCTR::tracing::native_trace_enabled() &&
      HugeCTR::tracing::native_trace_detail() >= 2) {
    auto it = fwd_gemm_phases.find(this);
    if (it == fwd_gemm_phases.end()) {
      int lid = this->get_gpu().get_local_id();
      // bias GEMM: top = identity[1xM] * bias[1xN] (broadcast)
      // Hipblas: M=output_size, N=in_batch_size, K=1
      char bias_args[160], kernel_args[160];
      std::snprintf(bias_args, sizeof(bias_args),
                    "\"kernel\":\"hipblasGemmEx(bias)\","
                    "\"M\":%ld,\"N\":%ld,\"K\":1,"
                    "\"dtype\":\"fp16\",\"compute\":\"fp32\",\"op\":\"NN\"",
                    output_size, in_batch_size);
      // kernel GEMM: top = kernel[MxK] * bottom[KxN]
      // Hipblas: M=output_size, N=in_batch_size, K=input_size
      std::snprintf(kernel_args, sizeof(kernel_args),
                    "\"kernel\":\"hipblasGemmEx(weight)\","
                    "\"M\":%ld,\"N\":%ld,\"K\":%ld,"
                    "\"dtype\":\"fp16\",\"compute\":\"fp32\",\"op\":\"NN\"",
                    output_size, in_batch_size, input_size);
      auto bp = PerfettoEmitter::instance().register_phase(lid, "fc/bias_gemm",
                                                            std::string(bias_args));
      auto kp = PerfettoEmitter::instance().register_phase(lid, "fc/kernel_gemm",
                                                            std::string(kernel_args));
      fwd_gemm_phases[this] = {bp, kp};
      p_bias_gemm = bp; p_kernel_gemm = kp;
    } else {
      p_bias_gemm = it->second.first;
      p_kernel_gemm = it->second.second;
    }
  }

  const float alpha = 1.0f;
  const float beta_b = 0.0f;
  const float beta_k = 1.0f;

  hipStream_t fc_stream = get_gpu().get_stream();
  {
    ScopedGpuPhase _p(p_bias_gemm, fc_stream, "default");
    HCTR_LIB_THROW(hipblasGemmEx(get_gpu().get_cublas_handle(), HIPBLAS_OP_N, HIPBLAS_OP_N, output_size,
                                in_batch_size, 1, &alpha, bias, HIP_R_16F, output_size, identity,
                                HIP_R_16F, 1, &beta_b, top, HIP_R_16F, output_size, HIPBLAS_COMPUTE_32F,
                                falgo_b_));
  }
  {
    ScopedGpuPhase _p(p_kernel_gemm, fc_stream, "default");
    HCTR_LIB_THROW(hipblasGemmEx(get_gpu().get_cublas_handle(), HIPBLAS_OP_N, HIPBLAS_OP_N, output_size,
                                in_batch_size, input_size, &alpha, kernel, HIP_R_16F, output_size,
                                bottom, HIP_R_16F, input_size, &beta_k, top, HIP_R_16F, output_size,
                                HIPBLAS_COMPUTE_32F, falgo_k_));
  }
}

void FullyConnectedLayer<__half>::bprop() {
  CudaDeviceContext context(get_device_id());

  const __half* kernel = this->get_weight(0).template data<__half>();
  auto top_tensor = this->output_tensors_[0];
  const __half* top = top_tensor.template data<__half>();
  const __half* identity = identity_tensor_.template data<__half>();
  __half* kernel_grad = this->get_wgrad(0).template data<__half>();
  __half* bias_grad = this->get_wgrad(1).template data<__half>();
  __half* bottom = get_bottom_tensor(true).template data<__half>();

  const auto& bottom_tensor_dim = get_bottom_tensor(true).shape();
  const auto& top_tensor_dim = top_tensor.shape();

  int64_t in_batch_size = 1;
  int64_t input_size = bottom_tensor_dim.size(bottom_tensor_dim.dims() - 1);
  int64_t output_size = top_tensor_dim.size(top_tensor_dim.dims() - 1);

  for (int64_t idx = 0; idx < bottom_tensor_dim.dims() - 1; idx++) {
    in_batch_size = in_batch_size * bottom_tensor_dim.size(idx);
  }

  const float alpha = 1.0f;
  const float beta_b = 0.0f;
  const float beta_k = 1.0f;
  const float beta_x = 0.0f;

  // ROCm port (Phase 14n, 2026-05-13): route bias_grad + kernel_grad (wgrad)
  // GEMMs onto computation_stream_2_ via cublas_handle_wgrad_ when
  // HCTR_INNERPRODUCT_ASYNC_WGRAD=1 (default). This unlocks 2-way stream
  // parallelism: wgrad runs on stream 2, dgrad on default stream.
  //
  // Both wgrad GEMMs read `top` (upstream grad) and bias_grad/kernel_grad
  // are independent output buffers, so safe to dispatch on parallel stream.
  // dgrad (line 3) MUST stay on default stream because `bottom` (its output)
  // is the input for the previous layer's bprop on the default stream.
  //
  // Event sync at end ensures wgrad is observable to subsequent ops
  // (exchange_wgrad allreduce). With CUDA graphs (HCTR_USE_CUDA_GRAPH=1)
  // the event-recorded edges become part of the captured graph.
  // Default OFF: alone this is -3% at bs=1× because per-layer event sync
  // overhead (4 layers × 2 events ≈ 80us) exceeds wgrad parallel-with-dgrad
  // savings (~60us at bs=1×). Also, CUDA graph capture may flatten the
  // logical stream split — even with cublas_handle_wgrad_ bound to
  // computation_stream_2_, the wgrad GEMM kernels still observed on the
  // default stream in rocprofv3 trace (Phase 14n.dbg verification).
  // Becomes net positive after Item A1 (CK-Tile) cuts critical-stream kernels
  // and exposes more parallel work for the wgrad stream to absorb.
  static const bool use_async_wgrad = []() {
    const char* env = std::getenv("HCTR_INNERPRODUCT_ASYNC_WGRAD");
    return env != nullptr && std::atoi(env) == 1;
  }();

  hipblasHandle_t wgrad_handle = use_async_wgrad
      ? get_gpu().get_cublas_handle_wgrad()
      : get_gpu().get_cublas_handle();
  hipStream_t default_stream = get_gpu().get_stream();
  hipStream_t wgrad_stream = use_async_wgrad
      ? get_gpu().get_comp_overlap_stream()
      : default_stream;

  // (1) Make wgrad stream wait for default stream's prior work (so `top`
  //     is ready before wgrad GEMMs start reading it). Uses pre-created
  //     event_overlap_ so this works under CUDA graph capture.
  if (use_async_wgrad) {
    HCTR_LIB_THROW(hipEventRecord(event_overlap_, default_stream));
    HCTR_LIB_THROW(hipStreamWaitEvent(wgrad_stream, event_overlap_, 0));
    // Phase 14n.6 (TESTED, didn't help): explicit hipblasSetStream before
    // each GEMM doesn't move the wgrad kernels off the default stream
    // either. Confirms the hipblas/HIP graph-capture stream pinning issue
    // is deeper than per-call stream binding.
  }

  // Phase 20.6 + 20.7 (2026-05-17): per-GEMM bprop GpuPhase tracing
  // with PyTorch-profiler-style M/N/K metadata.
  using HugeCTR::tracing::GpuPhase;
  using HugeCTR::tracing::PerfettoEmitter;
  using HugeCTR::tracing::ScopedGpuPhase;
  static thread_local std::unordered_map<const FullyConnectedLayer<__half>*,
                                          std::array<GpuPhase*, 3>> bwd_gemm_phases;
  GpuPhase* p_bias_grad = nullptr;
  GpuPhase* p_kernel_grad = nullptr;
  GpuPhase* p_dgrad = nullptr;
  // Phase 20.7: per-GEMM bprop events require DETAIL >= 2.
  if (HugeCTR::tracing::native_trace_enabled() &&
      HugeCTR::tracing::native_trace_detail() >= 2) {
    auto it = bwd_gemm_phases.find(this);
    if (it == bwd_gemm_phases.end()) {
      int lid = this->get_gpu().get_local_id();
      char bg_args[180], kg_args[180], dg_args[180];
      // bias_grad: GEMM N,N (output_size, 1, in_batch_size) -> bias_grad[N]
      std::snprintf(bg_args, sizeof(bg_args),
                    "\"kernel\":\"hipblasGemmEx(bias_grad)\","
                    "\"M\":%ld,\"N\":1,\"K\":%ld,"
                    "\"dtype\":\"fp16\",\"compute\":\"fp32\",\"op\":\"NN\"",
                    output_size, in_batch_size);
      // kernel_grad (wgrad): N,T (output_size, input_size, in_batch_size)
      //   = out[OxB] * bottom[BxI]^T -> wgrad[OxI]
      std::snprintf(kg_args, sizeof(kg_args),
                    "\"kernel\":\"hipblasGemmEx(wgrad)\","
                    "\"M\":%ld,\"N\":%ld,\"K\":%ld,"
                    "\"dtype\":\"fp16\",\"compute\":\"fp32\",\"op\":\"NT\"",
                    output_size, input_size, in_batch_size);
      // dgrad: T,N (input_size, in_batch_size, output_size)
      //   = weight[IxO]^T * out[OxB] -> dgrad[IxB]
      std::snprintf(dg_args, sizeof(dg_args),
                    "\"kernel\":\"hipblasGemmEx(dgrad)\","
                    "\"M\":%ld,\"N\":%ld,\"K\":%ld,"
                    "\"dtype\":\"fp16\",\"compute\":\"fp32\",\"op\":\"TN\"",
                    input_size, in_batch_size, output_size);
      auto bg = PerfettoEmitter::instance().register_phase(lid, "fc/bias_grad",
                                                            std::string(bg_args));
      auto kg = PerfettoEmitter::instance().register_phase(lid, "fc/kernel_grad",
                                                            std::string(kg_args));
      auto dg = PerfettoEmitter::instance().register_phase(lid, "fc/dgrad",
                                                            std::string(dg_args));
      bwd_gemm_phases[this] = {bg, kg, dg};
      p_bias_grad = bg; p_kernel_grad = kg; p_dgrad = dg;
    } else {
      p_bias_grad = it->second[0];
      p_kernel_grad = it->second[1];
      p_dgrad = it->second[2];
    }
  }

  // bias_grad GEMM (on wgrad_handle / computation_stream_2_)
  {
    ScopedGpuPhase _p(p_bias_grad, wgrad_stream, "default");
    HCTR_LIB_THROW(hipblasGemmEx(wgrad_handle, HIPBLAS_OP_N, HIPBLAS_OP_N, output_size,
                                1, in_batch_size, &alpha, top, HIP_R_16F, output_size, identity,
                                HIP_R_16F, in_batch_size, &beta_b, bias_grad, HIP_R_16F,
                                output_size, HIPBLAS_COMPUTE_32F, balgo_b_));
  }

  // kernel_grad GEMM (wgrad, on wgrad_handle / computation_stream_2_)
  {
    ScopedGpuPhase _p(p_kernel_grad, wgrad_stream, "default");
    HCTR_LIB_THROW(hipblasGemmEx(wgrad_handle, HIPBLAS_OP_N, HIPBLAS_OP_T, output_size,
                                input_size, in_batch_size, &alpha, top, HIP_R_16F, output_size,
                                bottom, HIP_R_16F, input_size, &beta_k, kernel_grad, HIP_R_16F,
                                output_size, HIPBLAS_COMPUTE_32F, balgo_k_));
  }

  // dgrad GEMM (stays on default stream — bottom is consumed by prev layer's bprop)
  ScopedGpuPhase _p_dgrad(p_dgrad, default_stream, "default");
  HCTR_LIB_THROW(hipblasGemmEx(get_gpu().get_cublas_handle(), HIPBLAS_OP_T, HIPBLAS_OP_N, input_size,
                              in_batch_size, output_size, &alpha, kernel, HIP_R_16F, output_size,
                              top, HIP_R_16F, output_size, &beta_x, bottom, HIP_R_16F, input_size,
                              HIPBLAS_COMPUTE_32F, balgo_x_));

  // (2) Make default stream wait for wgrad completion so subsequent
  //     allreduce / exchange_wgrad sees the new gradients.
  if (use_async_wgrad) {
    HCTR_LIB_THROW(hipEventRecord(event_overlap_, wgrad_stream));
    HCTR_LIB_THROW(hipStreamWaitEvent(default_stream, event_overlap_, 0));
  }
}

void FullyConnectedLayer<__half>::initialize() {
  CudaDeviceContext context(get_device_id());

  __half* identity = identity_tensor_.template data<__half>();
  const auto& bottom_tensor_dim = get_bottom_tensor(true).shape();
  int64_t m = 1;
  for (int64_t idx = 0; idx < bottom_tensor_dim.dims() - 1; idx++) {
    m = m * bottom_tensor_dim.size(idx);
  }
  // Initialize identity vector
  initialize_array<<<(m - 1) / 1024 + 1, 1024, 0, get_gpu().get_stream()>>>(identity, m,
                                                                            __float2half(1.0f));

  // ROCm-port (Phase 14n): pre-create cross-stream event for async wgrad sync.
  // Created with hipEventDisableTiming for lower overhead under CUDA graph capture.
  if (!event_overlap_created_) {
    HCTR_LIB_THROW(hipEventCreateWithFlags(&event_overlap_, hipEventDisableTiming));
    event_overlap_created_ = true;
  }
}

void FullyConnectedLayer<__half>::search_algorithm() {
  // Set to the CUDA device where this layer assigned to
  CudaDeviceContext context(get_device_id());

  const int64_t repeat_num = 100;

  // Device Tensors to be used
  __half* bottom = get_bottom_tensor(true).template data<__half>();
  __half* top = this->output_tensors_[0].template data<__half>();
  __half* identity = identity_tensor_.template data<__half>();
  __half* kernel = this->get_weight(0).template data<__half>();
  __half* bias = this->get_weight(1).template data<__half>();
  __half* kernel_grad = this->get_wgrad(0).template data<__half>();
  __half* bias_grad = this->get_wgrad(1).template data<__half>();

  // Tensor dim
  const auto& bottom_tensor_dim = get_bottom_tensor(true).shape();
  const auto& top_tensor_dim = this->output_tensors_[0].shape();

  int64_t in_batch_size = 1;
  int64_t input_size = bottom_tensor_dim.size(bottom_tensor_dim.dims() - 1);
  int64_t output_size = top_tensor_dim.size(top_tensor_dim.dims() - 1);

  for (int64_t idx = 0; idx < bottom_tensor_dim.dims() - 1; idx++) {
    in_batch_size = in_batch_size * bottom_tensor_dim.size(idx);
  }

  // Record time for each algorithm
  float shortestTime = std::numeric_limits<float>::max();
  float time;
  hipEvent_t start, stop;
  HCTR_LIB_THROW(hipEventCreate(&start));
  HCTR_LIB_THROW(hipEventCreate(&stop));

  // Start, end for search
  const hipblasGemmAlgo_t startAlgo = HIPBLAS_GEMM_DEFAULT;
  const hipblasGemmAlgo_t endAlgo = HIPBLAS_GEMM_DEFAULT;

  // Search all the algorithm for falgo_b_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Record start event
    HCTR_LIB_THROW(hipEventRecord(start, get_gpu().get_stream()));
    for (int64_t i = 0; i < repeat_num && status == HIPBLAS_STATUS_SUCCESS; ++i) {
      status = hipblasGemmEx(get_gpu().get_cublas_handle(), HIPBLAS_OP_N, HIPBLAS_OP_N, output_size,
                            in_batch_size, 1, &alpha, bias, HIP_R_16F, output_size, identity,
                            HIP_R_16F, 1, &beta, top, HIP_R_16F, output_size, HIPBLAS_COMPUTE_32F,
                            static_cast<hipblasGemmAlgo_t>(testAlgo));
    }
    HCTR_LIB_THROW(hipEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(hipEventSynchronize(stop));
    HCTR_LIB_THROW(hipEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this algorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != HIPBLAS_STATUS_SUCCESS) {
      // HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for fprop_b, skipped.\n",
      // testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      falgo_b_ = static_cast<hipblasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = std::numeric_limits<float>::max();

  // Search all the algorithm for falgo_k_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 1.0f;

    // Record start event
    HCTR_LIB_THROW(hipEventRecord(start, get_gpu().get_stream()));
    for (int64_t i = 0; i < repeat_num && status == HIPBLAS_STATUS_SUCCESS; ++i) {
      status = hipblasGemmEx(get_gpu().get_cublas_handle(), HIPBLAS_OP_N, HIPBLAS_OP_N, output_size,
                            in_batch_size, input_size, &alpha, kernel, HIP_R_16F, output_size,
                            bottom, HIP_R_16F, input_size, &beta, top, HIP_R_16F, output_size,
                            HIPBLAS_COMPUTE_32F, static_cast<hipblasGemmAlgo_t>(testAlgo));
    }
    HCTR_LIB_THROW(hipEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(hipEventSynchronize(stop));
    HCTR_LIB_THROW(hipEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this algorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != HIPBLAS_STATUS_SUCCESS) {
      // HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for fprop, skipped.\n",
      // testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      falgo_k_ = static_cast<hipblasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = std::numeric_limits<float>::max();

  // Search all the algorithm for balgo_b_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Record start event
    HCTR_LIB_THROW(hipEventRecord(start, get_gpu().get_stream()));
    for (int64_t i = 0; i < repeat_num && status == HIPBLAS_STATUS_SUCCESS; ++i) {
      status = hipblasGemmEx(get_gpu().get_cublas_handle(), HIPBLAS_OP_N, HIPBLAS_OP_N, output_size, 1,
                            in_batch_size, &alpha, top, HIP_R_16F, output_size, identity,
                            HIP_R_16F, in_batch_size, &beta, bias_grad, HIP_R_16F, output_size,
                            HIPBLAS_COMPUTE_32F, static_cast<hipblasGemmAlgo_t>(testAlgo));
    }
    HCTR_LIB_THROW(hipEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(hipEventSynchronize(stop));
    HCTR_LIB_THROW(hipEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this algorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != HIPBLAS_STATUS_SUCCESS) {
      // HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for bprop_W, skipped.\n",
      // testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      balgo_b_ = static_cast<hipblasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = std::numeric_limits<float>::max();

  // Search all the algorithm for balgo_k_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 1.0f;

    // Record start event
    HCTR_LIB_THROW(hipEventRecord(start, get_gpu().get_stream()));
    for (int64_t i = 0; i < repeat_num && status == HIPBLAS_STATUS_SUCCESS; ++i) {
      status = hipblasGemmEx(get_gpu().get_cublas_handle(), HIPBLAS_OP_N, HIPBLAS_OP_T, output_size,
                            input_size, in_batch_size, &alpha, top, HIP_R_16F, output_size, bottom,
                            HIP_R_16F, input_size, &beta, kernel_grad, HIP_R_16F, output_size,
                            HIPBLAS_COMPUTE_32F, static_cast<hipblasGemmAlgo_t>(testAlgo));
    }
    HCTR_LIB_THROW(hipEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(hipEventSynchronize(stop));
    HCTR_LIB_THROW(hipEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this algorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != HIPBLAS_STATUS_SUCCESS) {
      // HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for bprop_W, skipped.\n",
      // testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      balgo_k_ = static_cast<hipblasGemmAlgo_t>(testAlgo);
    }
  }

  // Reset shortestTime
  shortestTime = std::numeric_limits<float>::max();

  // Search all the algorithm for balgo_x_
  for (int testAlgo = startAlgo; testAlgo <= endAlgo; testAlgo++) {
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Record start event
    HCTR_LIB_THROW(hipEventRecord(start, get_gpu().get_stream()));
    for (int64_t i = 0; i < repeat_num && status == HIPBLAS_STATUS_SUCCESS; ++i) {
      status = hipblasGemmEx(get_gpu().get_cublas_handle(), HIPBLAS_OP_T, HIPBLAS_OP_N, input_size,
                            in_batch_size, output_size, &alpha, kernel, HIP_R_16F, output_size,
                            top, HIP_R_16F, output_size, &beta, bottom, HIP_R_16F, input_size,
                            HIPBLAS_COMPUTE_32F, static_cast<hipblasGemmAlgo_t>(testAlgo));
    }

    HCTR_LIB_THROW(hipEventRecord(stop, get_gpu().get_stream()));
    HCTR_LIB_THROW(hipEventSynchronize(stop));
    HCTR_LIB_THROW(hipEventElapsedTime(&time, start, stop));
    // Avg Time(ms) for this algorithm for fprop GEMM
    time = time / repeat_num;
    // Skip if the algorithm is supported for fprop configuration
    if (status != HIPBLAS_STATUS_SUCCESS) {
      // HCTR_LOG(INFO, WORLD, "The algorithms %d is not supported for bprop_Xn, skipped.\n",
      // testAlgo);
      continue;
    }
    // Record the optimal time and algorithm
    if (time < shortestTime) {
      shortestTime = time;
      balgo_x_ = static_cast<hipblasGemmAlgo_t>(testAlgo);
    }
  }

  // Print selection information
  // HCTR_LOG(INFO, WORLD,
  //     "The algorithm selection for falgo_b_, falgo_k_, balgo_b_, balgo_k_, balgo_x_ are: %d, %d,
  //     "
  //     "%d, %d and %d.\n",
  //     (int)falgo_b_ - HIPBLAS_GEMM_DEFAULT, (int)falgo_k_ -
  //     HIPBLAS_GEMM_DEFAULT, (int)balgo_b_ - HIPBLAS_GEMM_DEFAULT, (int)balgo_k_
  //     - HIPBLAS_GEMM_DEFAULT, (int)balgo_x_ - HIPBLAS_GEMM_DEFAULT);

  // Output msg
  // HCTR_LOG(INFO, ROOT, "The fully-connected layer has finished choosing the algorithm for cublas
  // Gemm.\n"); Clean-up
  HCTR_LIB_THROW(hipEventDestroy(start));
  HCTR_LIB_THROW(hipEventDestroy(stop));
}  // namespace HugeCTR

std::unique_ptr<DataSimulator> FullyConnectedLayer<__half>::get_uniform_initializer(
    const int index) {
  int64_t bottom_dim =
      get_bottom_tensor(true).shape().size(get_bottom_tensor(true).shape().dims() - 1);
  auto top_tensor = this->output_tensors_[0];
  int64_t top_dim = top_tensor.shape().size(top_tensor.shape().dims() - 1);

  float limit = 1.0f / ((0 == index ? bottom_dim : 0) + top_dim);
  return std::make_unique<UniformDataSimulator>(-1 * limit, limit);
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<__half>::get_xavier_uniform_initializer(
    const int index) {
  int64_t bottom_dim =
      get_bottom_tensor(true).shape().size(get_bottom_tensor(true).shape().dims() - 1);
  auto top_tensor = this->output_tensors_[0];
  int64_t top_dim = top_tensor.shape().size(top_tensor.shape().dims() - 1);

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Uniform,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<__half>::get_xavier_norm_initializer(
    const int index) {
  int64_t bottom_dim =
      get_bottom_tensor(true).shape().size(get_bottom_tensor(true).shape().dims() - 1);
  auto top_tensor = this->output_tensors_[0];
  int64_t top_dim = top_tensor.shape().size(top_tensor.shape().dims() - 1);

  return std::make_unique<VarianceScalingSimulator>(1.f, data_simu::Mode_t::Fan_avg,
                                                    data_simu::Distribution_t::Norm,
                                                    0 == index ? bottom_dim : 0, top_dim);
}

std::unique_ptr<DataSimulator> FullyConnectedLayer<__half>::get_default_initializer(
    const int index) {
  int64_t bottom_dim =
      get_bottom_tensor(true).shape().size(get_bottom_tensor(true).shape().dims() - 1);
  auto top_tensor = this->output_tensors_[0];
  int64_t top_dim = top_tensor.shape().size(top_tensor.shape().dims() - 1);

  std::unique_ptr<DataSimulator> simu(nullptr);
  if (0 == index) {
    simu.reset(new VarianceScalingSimulator(1.f, data_simu::Mode_t::Fan_avg,
                                            data_simu::Distribution_t::Norm, bottom_dim, top_dim));
  } else if (1 == index) {
    float stddev = sqrt(1.f / top_dim);
    simu.reset(new GaussianDataSimulator(0, stddev, -2 * stddev, 2 * stddev));
  } else {
    HCTR_OWN_THROW(Error_t::OutOfBound, "index != {0, 1}.");
  }

  return simu;
}

template class FullyConnectedLayer<__half>;

}  // namespace HugeCTR
