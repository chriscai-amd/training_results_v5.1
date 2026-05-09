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
#include <gpu_resource.hpp>
#include <utils.hpp>

namespace HugeCTR {

GPUResource::GPUResource(int device_id, size_t local_id, size_t global_id,
                         unsigned long long replica_uniform_seed,
                         unsigned long long replica_variant_seed, const ncclComm_t& comm)
    : device_id_(device_id),
      local_id_(local_id),
      global_id_(global_id),
      stream_name_("default"),
      comm_(comm) {
  CudaDeviceContext context(device_id);
  HCTR_LIB_THROW(
      hiprandCreateGenerator(&replica_variant_curand_generator_, HIPRAND_RNG_PSEUDO_DEFAULT));
  HCTR_LIB_THROW(
      hiprandSetPseudoRandomGeneratorSeed(replica_variant_curand_generator_, replica_variant_seed));
  HCTR_LIB_THROW(hipblasCreate(&cublas_handle_));
#ifndef HUGECTR_ROCM_PORT
  HCTR_LIB_THROW(hipdnnCreate(&cudnn_handle_));
#endif

  set_stream(stream_name_, 0);
  hipStream_t computation_stream_ = stream_event_manager_.get_stream(stream_name_);
  memcpy_stream_ = stream_event_manager_.get_stream("memcpy_stream_", hipStreamNonBlocking);
  computation_stream_2_ = stream_event_manager_.get_stream("computation_stream_2_");
  p2p_stream_ = stream_event_manager_.get_stream("p2p_stream_", hipStreamNonBlocking);
  wait_wgrad_event_ = stream_event_manager_.get_event("wgrad_event", hipEventDefault);

  HCTR_LIB_THROW(
      hiprandCreateGenerator(&replica_uniform_curand_generator_, HIPRAND_RNG_PSEUDO_DEFAULT));
  HCTR_LIB_THROW(
      hiprandSetPseudoRandomGeneratorSeed(replica_uniform_curand_generator_, replica_uniform_seed));
  HCTR_LIB_THROW(hiprandSetStream(replica_uniform_curand_generator_, computation_stream_));

  HCTR_LIB_THROW(hiprandSetStream(replica_variant_curand_generator_, computation_stream_));
  HCTR_LIB_THROW(hipblasCreate(&cublas_handle_wgrad_));
  HCTR_LIB_THROW(hipblasLtCreate(&cublaslt_handle_));
  HCTR_LIB_THROW(hipblasSetStream(cublas_handle_, computation_stream_));
  HCTR_LIB_THROW(hipblasSetStream(cublas_handle_wgrad_, computation_stream_2_));
#ifndef HUGECTR_ROCM_PORT
  HCTR_LIB_THROW(hipdnnSetStream(cudnn_handle_, computation_stream_));
#endif

  int sm_count;
  HCTR_LIB_THROW(hipDeviceGetAttribute(&sm_count, hipDeviceAttributeMultiprocessorCount, device_id));
  sm_count_ = sm_count;
  int max_thread_per_sm;
  HCTR_LIB_THROW(hipDeviceGetAttribute(&max_thread_per_sm, hipDeviceAttributeMaxThreadsPerMultiProcessor,
                                        device_id));
  max_thread_per_sm_ = max_thread_per_sm;

  HCTR_LIB_THROW(hipDeviceGetAttribute(&cc_major_, hipDeviceAttributeComputeCapabilityMajor, device_id));
  HCTR_LIB_THROW(hipDeviceGetAttribute(&cc_minor_, hipDeviceAttributeComputeCapabilityMinor, device_id));
}

GPUResource::~GPUResource() {
  try {
    CudaDeviceContext context(device_id_);
    HCTR_LIB_THROW(ncclCommDestroy(comm_));
    HCTR_LIB_THROW(hiprandDestroyGenerator(replica_uniform_curand_generator_));
    HCTR_LIB_THROW(hiprandDestroyGenerator(replica_variant_curand_generator_));
    HCTR_LIB_THROW(hipblasDestroy(cublas_handle_));
    HCTR_LIB_THROW(hipblasDestroy(cublas_handle_wgrad_));
#ifndef HUGECTR_ROCM_PORT
    HCTR_LIB_THROW(hipdnnDestroy(cudnn_handle_));
#endif
  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
  }
}

void GPUResource::set_wgrad_event_sync(const hipStream_t& sync_stream) const {
  HCTR_LIB_THROW(hipEventRecord(wait_wgrad_event_, sync_stream));
  return;
}

void GPUResource::wait_on_wgrad_event(const hipStream_t& sync_stream) const {
  HCTR_LIB_THROW(hipStreamWaitEvent(sync_stream, wait_wgrad_event_));
  return;
}

const hipStream_t& GPUResource::get_stream(const std::string& name, int priority) {
  return stream_event_manager_.get_stream(name, hipStreamNonBlocking, priority);
}

const hipEvent_t& GPUResource::get_event(const std::string& name) {
  return stream_event_manager_.get_event(name, hipEventDisableTiming);
}

}  // namespace HugeCTR
