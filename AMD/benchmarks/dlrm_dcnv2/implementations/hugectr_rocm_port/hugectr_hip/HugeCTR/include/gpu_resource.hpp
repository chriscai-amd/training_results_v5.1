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
#pragma once

#include <hipblaslt/hipblaslt.h>
#include <hipblas/hipblas.h>
/* HugeCTR ROCm port: <hipDNN.h> removed */
#include <hiprand/hiprand.h>
#include <rccl/rccl.h>

#include <stream_event_manager.hpp>
#include <utils.hpp>

namespace HugeCTR {

/**
 * @brief GPU resource allocated on a target gpu.
 *
 * This class implement unified resource management on the target GPU.
 */
class GPUResource {
  const int device_id_;
  const size_t local_id_;
  const size_t global_id_;
  std::string stream_name_;
  hipStream_t memcpy_stream_; /**< cuda stream for data copy */
  hipStream_t p2p_stream_;    /**< cuda stream for broadcast copy */
  hiprandGenerator_t replica_uniform_curand_generator_;
  hiprandGenerator_t replica_variant_curand_generator_;
  hipblasHandle_t cublas_handle_;
  hipblasHandle_t cublas_handle_wgrad_;
#ifndef HUGECTR_ROCM_PORT
  hipdnnHandle_t cudnn_handle_;
#endif
  hipblasLtHandle_t cublaslt_handle_;
  ncclComm_t comm_;
  size_t sm_count_;
  int max_thread_per_sm_;
  int cc_major_;
  int cc_minor_;
  hipStream_t computation_stream_2_;

  hipEvent_t wait_wgrad_event_;

  StreamEventManager stream_event_manager_;

 public:
  GPUResource(int device_id, size_t local_id, size_t global_id,
              unsigned long long replica_uniform_seed, unsigned int long long replica_variant_seed,
              const ncclComm_t& comm);
  GPUResource(const GPUResource&) = delete;
  GPUResource& operator=(const GPUResource&) = delete;
  ~GPUResource();

  const hipStream_t& get_stream(const std::string& name, int priority = 0);
  const hipEvent_t& get_event(const std::string& name);

  int get_device_id() const { return device_id_; }
  int get_local_id() const { return local_id_; }
  size_t get_global_id() const { return global_id_; }
  const hipStream_t& get_stream() const { return stream_event_manager_.get_stream(stream_name_); }
  std::string get_current_stream_name() const { return stream_name_; }
  void set_stream(const std::string& name, int priority = 0) {
    hipStream_t current_stream =
        stream_event_manager_.get_stream(name, hipStreamNonBlocking, priority);
    stream_name_ = name;
    HCTR_LIB_THROW(hiprandSetStream(replica_variant_curand_generator_, current_stream));
    HCTR_LIB_THROW(hipblasSetStream(cublas_handle_, current_stream));
#ifndef HUGECTR_ROCM_PORT
    HCTR_LIB_THROW(hipdnnSetStream(cudnn_handle_, current_stream));
#endif
  }
  const hipStream_t& get_memcpy_stream() const { return memcpy_stream_; }
  const hipStream_t& get_p2p_stream() const { return p2p_stream_; }
  const hipStream_t& get_comp_overlap_stream() const { return computation_stream_2_; }
  const hiprandGenerator_t& get_replica_uniform_curand_generator() const {
    return replica_uniform_curand_generator_;
  }
  const hiprandGenerator_t& get_replica_variant_curand_generator() const {
    return replica_variant_curand_generator_;
  }
  const hipblasHandle_t& get_cublas_handle() const { return cublas_handle_; }
  const hipblasHandle_t& get_cublas_handle_wgrad() const { return cublas_handle_wgrad_; }
  const hipblasLtHandle_t& get_cublaslt_handle() const { return cublaslt_handle_; }
#ifndef HUGECTR_ROCM_PORT
  const hipdnnHandle_t& get_cudnn_handle() const { return cudnn_handle_; }
#endif
  const ncclComm_t& get_nccl() const { return comm_; }
  size_t get_sm_count() const { return sm_count_; }
  int get_max_thread_per_sm() const { return max_thread_per_sm_; }
  int get_cc_major() const { return cc_major_; }
  int get_cc_minor() const { return cc_minor_; }
  bool support_nccl() const { return comm_ != nullptr; }

  void set_wgrad_event_sync(const hipStream_t& sync_stream) const;
  void wait_on_wgrad_event(const hipStream_t& sync_stream) const;
};

class StreamContext {
  std::shared_ptr<GPUResource> local_gpu_;
  std::string origin_stream_name_;

 public:
  StreamContext(std::shared_ptr<GPUResource> local_gpu, const std::string& new_stream_name,
                int priority = 0)
      : local_gpu_(local_gpu), origin_stream_name_(local_gpu->get_current_stream_name()) {
    local_gpu_->set_stream(new_stream_name, priority);
  }
  ~StreamContext() { local_gpu_->set_stream(origin_stream_name_, 0); }
};

}  // namespace HugeCTR
