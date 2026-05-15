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
#pragma once

#include <hip/hip_runtime_api.h>
/* HugeCTR ROCm port: <hipDNN.h> removed */
#include <getopt.h>
#include <rccl/rccl.h>
#include <numa.h>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <common.hpp>
#include <ctime>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef ENABLE_MPI
#include <mpi.h>
#endif

namespace HugeCTR {

template <typename T>
inline void ArgConvertor(std::string arg, T& ret);

template <>
inline void ArgConvertor(std::string arg, int& ret) {
  ret = std::stoi(arg);
}

template <>
inline void ArgConvertor(std::string arg, size_t& ret) {
  ret = std::stoul(arg);
}

template <>
inline void ArgConvertor(std::string arg, float& ret) {
  ret = std::stof(arg);
}

template <>
inline void ArgConvertor(std::string arg, std::vector<int>& ret) {
  ret.clear();
  std::istringstream is(arg);
  for (int i; is >> i;) {
    ret.push_back(i);
    if (is.peek() == ',') is.ignore();
  }
}

template <>
inline void ArgConvertor(std::string arg, std::vector<size_t>& ret) {
  ret.clear();
  std::istringstream is(arg);
  for (size_t i; is >> i;) {
    ret.push_back(i);
    if (is.peek() == ',') is.ignore();
  }
}

template <>
inline void ArgConvertor(std::string arg, std::string& ret) {
  ret = arg;
}

struct ArgParser {
 private:
  static std::string get_arg_(const std::string target, int argc, char** argv) {
    std::vector<std::string> tokens;
    for (int i = 1; i < argc; ++i) tokens.push_back(std::string(argv[i]));
    std::vector<std::string>::const_iterator itr;
    const std::string option = "--" + target;
    itr = std::find(tokens.begin(), tokens.end(), option);
    if (itr != tokens.end() && ++itr != tokens.end()) {
      return *itr;
    }
    static const std::string empty_string("");
    return empty_string;
  }

 public:
  template <typename T>
  static T get_arg(const std::string target, int argc, char** argv) {
    auto arg = get_arg_(target, argc, argv);
    if (arg.empty()) {
      HCTR_OWN_THROW(Error_t::WrongInput, "Cannot find target string: " + target);
    }
    T ret;
    ArgConvertor<T>(arg, ret);
    return ret;
  }
  template <typename T>
  static T get_arg(const std::string target, int argc, char** argv, T default_val) {
    auto arg = get_arg_(target, argc, argv);
    if (arg.empty()) {
      HCTR_LOG_S(INFO, ROOT) << "Cannot find target string: " << target
                             << " use default value:" << std::endl;
      return default_val;
    }
    T ret;
    ArgConvertor<T>(arg, ret);
    return ret;
  }
  static bool has_arg(const std::string target, int argc, char** argv) {
    std::vector<std::string> tokens;
    for (int i = 1; i < argc; ++i) tokens.push_back(std::string(argv[i]));
    const std::string option = "--" + target;
    return std::find(tokens.begin(), tokens.end(), option) != tokens.end();
  }
};

template <typename T>
std::string vec_to_string(std::vector<T> vec) {
  std::string ret;
  for (auto& elem : vec) {
    ret = ret + std::to_string(elem) + ", ";
  }
  return ret.substr(0, ret.size() - 2);
}

/**
 * CPU Timer.
 */
class Timer {
 public:
  void start() {
    m_StartTime_ = std::chrono::steady_clock::now();
    m_bRunning_ = true;
  }
  void stop() {
    m_EndTime_ = std::chrono::steady_clock::now();
    m_bRunning_ = false;
  }
  double elapsedMilliseconds() { return elapsed().count() / 1000.0; }
  double elapsedMicroseconds() { return elapsed().count(); }
  double elapsedSeconds() { return elapsed().count() / 1000000.0; }

 private:
  std::chrono::microseconds elapsed() {
    std::chrono::time_point<std::chrono::steady_clock> endTime;
    if (m_bRunning_) {
      endTime = std::chrono::steady_clock::now();
    } else {
      endTime = m_EndTime_;
    }
    return std::chrono::duration_cast<std::chrono::microseconds>(endTime - m_StartTime_);
  }

  std::chrono::time_point<std::chrono::steady_clock> m_StartTime_{};
  std::chrono::time_point<std::chrono::steady_clock> m_EndTime_{};
  bool m_bRunning_ = false;
};

/**
 * GPU Timer
 */
class GPUTimer {
 private:
  hipEvent_t start_;
  hipEvent_t stop_;
  hipStream_t stream_;

 public:
  GPUTimer() {
    hipEventCreate(&start_);
    hipEventCreate(&stop_);
  }

  ~GPUTimer() {
    hipEventDestroy(start_);
    hipEventDestroy(stop_);
  }

  void start(hipStream_t st = 0) {
    stream_ = st;
    hipEventRecord(start_, stream_);
  }

  float stop() {
    float milliseconds = 0;
    hipEventRecord(stop_, stream_);
    hipEventSynchronize(stop_);
    hipEventElapsedTime(&milliseconds, start_, stop_);
    return milliseconds;
  }
};

/**
 * Helper class for switching device
 */
class CudaDeviceContext {
  int original_device_;

 public:
  CudaDeviceContext() { HCTR_LIB_THROW(hipGetDevice(&original_device_)); }
  CudaDeviceContext(int device) : CudaDeviceContext() {
    if (device != original_device_) {
      set_device(device);
    }
  }
  ~CudaDeviceContext() noexcept(false) { set_device(original_device_); }

  void set_device(int device) const { HCTR_LIB_THROW(hipSetDevice(device)); }

  void check_device(int device) const {
    HCTR_CHECK_HINT(device == original_device_,
                    "The device id in the context is not consistent with configuration");
  }
};

/**
 * Helper class for switching device and the associated NUMA domain.
 */
class CudaCPUDeviceContext {
 private:
  int original_device_;

 public:
  CudaCPUDeviceContext() { HCTR_LIB_THROW(hipGetDevice(&original_device_)); }
  CudaCPUDeviceContext(int device_id) : CudaCPUDeviceContext() {
    auto node_it = device_id_to_numa_node_.find(device_id);
    assert(node_it != device_id_to_numa_node_.end());
    HCTR_LIB_THROW(hipSetDevice(device_id));

    int node = node_it->second;
    if (node >= 0) {
      numa_run_on_node(node);
      numa_set_preferred(node);
    }
  }

  static void init_cpu_mapping(std::vector<int> device_ids) {
    constexpr int pci_id_len = 16;
    char pci_id[pci_id_len];

    device_id_to_numa_node_.clear();
    numa_node_to_device_ids_.clear();

    auto cpu_mask = numa_allocate_cpumask();

    auto select_node = [](const bitmask* nvml_cpus) -> int {
      for (int cpu = 0; cpu < numa_num_possible_cpus(); cpu++) {
        if (numa_bitmask_isbitset(nvml_cpus, cpu)) {
          return numa_node_of_cpu(cpu);
        }
      }
      return -1;
    };

    {
      HCTR_LOG_S(INFO, ROOT) << "Device to NUMA mapping:" << std::endl;

      for (auto device_id : device_ids) {
#ifndef HUGECTR_ROCM_PORT
        nvmlDevice_t handle;
        HCTR_LIB_THROW(hipDeviceGetPCIBusId(pci_id, pci_id_len, device_id));
        HCTR_LIB_THROW(nvmlDeviceGetHandleByPciBusId_v2(pci_id, &handle));
        HCTR_LIB_THROW(nvmlDeviceGetCpuAffinity(
            handle, cpu_mask->size / (sizeof(unsigned long) * 8), cpu_mask->maskp));
        int node = select_node(cpu_mask);
#else
        // HugeCTR ROCm port: NVML CPU-affinity query is unavailable on AMD.
        // Fall back to "no NUMA preference" -> node = -1. Performance may be
        // suboptimal on multi-socket systems but correctness is preserved.
        (void)pci_id; (void)pci_id_len; (void)cpu_mask; (void)select_node;
        int node = -1;
#endif
        device_id_to_numa_node_[device_id] = node;
        HCTR_LOG_S(INFO, ROOT) << "  GPU " << device_id << " -> "
                               << " node " << node << std::endl;

        if (numa_node_to_device_ids_.count(node) == 0) {
          numa_node_to_device_ids_[node] = std::vector<int>{device_id};
        } else {
          numa_node_to_device_ids_[node].push_back(device_id);
        }
      }
    }

    numa_bitmask_free(cpu_mask);
  }

  ~CudaCPUDeviceContext() { HCTR_LIB_CHECK_(hipSetDevice(original_device_)); }

 public:
  static std::unordered_map<int, int> device_id_to_numa_node_;
  static std::unordered_map<int, std::vector<int>> numa_node_to_device_ids_;
};

/**
 * Get total product from dims.
 */
inline size_t get_size_from_dims(const std::vector<size_t>& dims) {
  size_t matrix_size = 1;
  for (auto iter = dims.begin(); iter != dims.end(); iter++) {
    matrix_size = matrix_size * iter[0];
  }
  return matrix_size;
}

/**
 * Find the item from a map according to str and pass by opt.
 * @return true: found / false: not found
 **/
template <typename ITEM_TYPE>
bool find_item_in_map(ITEM_TYPE& item, const std::string& str,
                      const std::map<std::string, ITEM_TYPE>& item_map) {
  typename std::map<std::string, ITEM_TYPE>::const_iterator it = item_map.find(str);
  if (it == item_map.end()) {
    return false;
  } else {
    item = it->second;
    return true;
  }
}

inline void set_affinity(std::thread& t, int core) {
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core, &cpuset);
  int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    std::ostringstream os;
    os << "Error calling pthread_setaffinity_np: " << rc;
    HCTR_OWN_THROW(Error_t::WrongInput, os.str());
  }
  return;
}

inline void set_affinity(std::thread& t, std::set<int> set, bool excluded) {
  if (set.empty()) {
    auto get_core_id = [](int i) { return (i % 8) * 16 + (i / 8) % 2 + (i / 16) * 128; };
    for (int i = 0; i < 31; i++) {
      int core = get_core_id(i);
      set.insert(core);
    }
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  for (int core = 0; core < 256; core++) {
    if (excluded) {
      if (set.find(core) == set.end()) {
        CPU_SET(core, &cpuset);
      }
    } else {
      if (set.find(core) != set.end()) {
        CPU_SET(core, &cpuset);
      }
    }
  }
  int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
  if (rc != 0) {
    std::ostringstream os;
    os << "Error calling pthread_setaffinity_np: " << rc;
    HCTR_OWN_THROW(Error_t::WrongInput, os.str());
  }
  return;
}

template <typename TOUT, typename TIN>
struct TypeConvert;

template <>
struct TypeConvert<float, float> {
  static __host__ float convert(const float val) { return val; }
};

template <>
struct TypeConvert<__half, float> {
  static __host__ __half convert(const float val) { return __float2half(val); }
};

template <>
struct TypeConvert<float, __half> {
  static __host__ float convert(const __half val) { return __half2float(val); }
};

// HugeCTR ROCm port: cuDNN/MIOpen excluded from build, so this dispatch trait
// is reduced to a forward declaration. None of the layers we build instantiate
// it; if a layer that needs cuDNN is ever re-enabled, this must be re-defined
// against MIOpen's miopenDataType_t enum.
template <typename T>
struct CudnnDataType;

#ifndef HUGECTR_ROCM_PORT
template <>
struct CudnnDataType<float> {
  static hipdnnDataType_t getType() { return HIPDNN_DATA_FLOAT; }
};

template <>
struct CudnnDataType<__half> {
  static hipdnnDataType_t getType() { return HIPDNN_DATA_HALF; }
};
#endif

#ifdef ENABLE_MPI
template <typename TypeKey>
struct ToMpiType;

template <>
struct ToMpiType<long long> {
  static MPI_Datatype T() { return MPI_LONG_LONG; }
};

template <>
struct ToMpiType<unsigned int> {
  static MPI_Datatype T() { return MPI_UNSIGNED; }
};

template <>
struct ToMpiType<float> {
  static MPI_Datatype T() { return MPI_FLOAT; }
};

#endif
template <typename TIN, typename TOUT>
void convert_array_on_device(TOUT* out, const TIN* in, size_t num_elements,
                             const hipStream_t& stream);
template <typename T>
struct NcclDataType;
template <>
struct NcclDataType<int> {
  static ncclDataType_t getType() { return ncclInt32; }
};
template <>
struct NcclDataType<long long> {
  static ncclDataType_t getType() { return ncclInt64; }
};
template <>
struct NcclDataType<unsigned int> {
  static ncclDataType_t getType() { return ncclUint32; }
};
template <>
struct NcclDataType<unsigned long long> {
  static ncclDataType_t getType() { return ncclUint64; }
};
template <>
struct NcclDataType<float> {
  static ncclDataType_t getType() { return ncclFloat32; }
};
template <>
struct NcclDataType<__half> {
  static ncclDataType_t getType() { return ncclHalf; }
};

template <typename TypeKey>
void data_to_unique_categories(TypeKey* value, const TypeKey* rowoffset,
                               const TypeKey* emmbedding_offsets, int num_tables,
                               int num_rowoffsets, const hipStream_t& stream);

template <typename TypeKey>
void data_to_unique_categories(TypeKey* value, const TypeKey* rowoffset,
                               const TypeKey* emmbedding_offsets, int num_tables,
                               int num_rowoffsets, const hipStream_t& stream);

template <typename TypeKey>
void data_to_unique_categories(TypeKey* value, const TypeKey* emmbedding_offsets, int num_tables,
                               int nnz, const hipStream_t& stream);

// Redistribute keys: from table first to sample first
template <typename TypeKey>
void distribute_keys_for_inference(TypeKey* out, const TypeKey* in, size_t batchsize,
                                   const std::vector<std::vector<TypeKey>>& h_reader_row_ptrs_list,
                                   const std::vector<size_t>& slot_num_for_tables) {
  const size_t num_tables = h_reader_row_ptrs_list.size();
  std::vector<size_t> h_embedding_offset_sample_first(batchsize * num_tables + 1, 0);
  std::vector<size_t> h_embedding_offset_table_first(batchsize * num_tables + 1, 0);
  for (size_t i = 0; i < batchsize; i++) {
    for (size_t j = 0; j < num_tables; j++) {
      const size_t num_of_feature = h_reader_row_ptrs_list[j][(i + 1) * slot_num_for_tables[j]] -
                                    h_reader_row_ptrs_list[j][i * slot_num_for_tables[j]];
      h_embedding_offset_sample_first[i * num_tables + j + 1] =
          h_embedding_offset_sample_first[i * num_tables + j] + num_of_feature;
    }
  }

  for (size_t j = 0; j < num_tables; j++) {
    for (size_t i = 0; i < batchsize; i++) {
      const size_t num_of_feature = h_reader_row_ptrs_list[j][(i + 1) * slot_num_for_tables[j]] -
                                    h_reader_row_ptrs_list[j][i * slot_num_for_tables[j]];
      h_embedding_offset_table_first[j * batchsize + i + 1] =
          h_embedding_offset_table_first[j * batchsize + i] + num_of_feature;
    }
  }

  for (size_t i = 0; i < batchsize; i++) {
    for (size_t j = 0; j < num_tables; j++) {
      const size_t num_keys = h_embedding_offset_sample_first[i * num_tables + j + 1] -
                              h_embedding_offset_sample_first[i * num_tables + j];
      for (size_t k = 0; k < num_keys; k++) {
        out[h_embedding_offset_sample_first[i * num_tables + j] + k] =
            in[h_embedding_offset_table_first[j * batchsize + i] + k];
      }
    }
  }
}

void calc_embedding_offset(size_t* d_embedding_offset, const int* d_row_ptrs,
                           const size_t* d_row_ptrs_offset, const size_t* d_slot_num_for_tables,
                           size_t num_tables, size_t batch_size, bool sample_first,
                           hipStream_t stream);

template <typename TypeHashKey>
void convert_keys_to_table_first(TypeHashKey* d_out, const TypeHashKey* d_in,
                                 size_t* d_embedding_offset_table_first,
                                 size_t* d_embedding_offset_sample_first, size_t num_tables,
                                 size_t batch_size, hipStream_t stream);

template <typename TypeHashKey>
void distribute_keys_per_table_on_device(TypeHashKey* d_out, const TypeHashKey* d_in,
                                         const int* d_row_ptrs, size_t batchsize,
                                         const std::vector<size_t>& slot_num_for_tables,
                                         hipStream_t stream) {
  const size_t num_tables = slot_num_for_tables.size();

  size_t* d_slot_num_for_tables;
  hipMallocManaged(&d_slot_num_for_tables, num_tables * sizeof(size_t));
  for (size_t i = 0; i < slot_num_for_tables.size(); i++) {
    d_slot_num_for_tables[i] = slot_num_for_tables[i];
  }

  size_t* d_row_ptrs_offset;
  hipMallocManaged(&d_row_ptrs_offset, (num_tables + 1) * sizeof(size_t));
  for (size_t i = 0; i < num_tables; i++) {
    d_row_ptrs_offset[i + 1] = d_row_ptrs_offset[i] + batchsize * slot_num_for_tables[i] + 1;
  }

  size_t* d_embedding_offset_sample_first;
  size_t* d_embedding_offset_table_first;
  hipMallocAsync(&d_embedding_offset_sample_first, (batchsize * num_tables + 1) * sizeof(size_t),
                  stream);
  hipMallocAsync(&d_embedding_offset_table_first, (batchsize * num_tables + 1) * sizeof(size_t),
                  stream);

  calc_embedding_offset(d_embedding_offset_sample_first, d_row_ptrs, d_row_ptrs_offset,
                        d_slot_num_for_tables, num_tables, batchsize, true, stream);

  calc_embedding_offset(d_embedding_offset_table_first, d_row_ptrs, d_row_ptrs_offset,
                        d_slot_num_for_tables, num_tables, batchsize, false, stream);

  convert_keys_to_table_first(d_out, d_in, d_embedding_offset_table_first,
                              d_embedding_offset_sample_first, num_tables, batchsize, stream);

  HCTR_LIB_THROW(hipStreamSynchronize(stream));
  hipFree(d_slot_num_for_tables);
  hipFree(d_row_ptrs_offset);
  hipFree(d_embedding_offset_sample_first);
  hipFree(d_embedding_offset_table_first);
}

// Redistribute keys: from sample first to table first
template <typename TypeHashKey>
void distribute_keys_per_table(TypeHashKey* h_out, const TypeHashKey* h_in, const int* h_row_ptrs,
                               size_t batchsize, const std::vector<size_t>& slot_num_for_tables) {
  const size_t num_tables = slot_num_for_tables.size();
  std::vector<size_t> row_ptrs_offset(num_tables + 1, 0);
  for (size_t i = 0; i < num_tables; i++) {
    row_ptrs_offset[i + 1] = row_ptrs_offset[i] + batchsize * slot_num_for_tables[i] + 1;
  }
  std::vector<size_t> h_embedding_offset_sample_first(batchsize * num_tables + 1, 0);
  std::vector<size_t> h_embedding_offset_table_first(batchsize * num_tables + 1, 0);
  for (size_t i = 0; i < batchsize; i++) {
    for (size_t j = 0; j < num_tables; j++) {
      const int* const h_row_ptrs_per_table = h_row_ptrs + row_ptrs_offset[j];
      const size_t num_of_feature = h_row_ptrs_per_table[(i + 1) * slot_num_for_tables[j]] -
                                    h_row_ptrs_per_table[i * slot_num_for_tables[j]];
      h_embedding_offset_sample_first[i * num_tables + j + 1] =
          h_embedding_offset_sample_first[i * num_tables + j] + num_of_feature;
    }
  }

  for (size_t j = 0; j < num_tables; j++) {
    for (size_t i = 0; i < batchsize; i++) {
      const int* const h_row_ptrs_per_table = h_row_ptrs + row_ptrs_offset[j];
      const size_t num_of_feature = h_row_ptrs_per_table[(i + 1) * slot_num_for_tables[j]] -
                                    h_row_ptrs_per_table[i * slot_num_for_tables[j]];
      h_embedding_offset_table_first[j * batchsize + i + 1] =
          h_embedding_offset_table_first[j * batchsize + i] + num_of_feature;
    }
  }

  for (size_t i = 0; i < batchsize; ++i) {
    for (size_t j = 0; j < num_tables; ++j) {
      const size_t num_keys = h_embedding_offset_sample_first[i * num_tables + j + 1] -
                              h_embedding_offset_sample_first[i * num_tables + j];
      for (size_t k = 0; k < num_keys; k++) {
        h_out[h_embedding_offset_table_first[j * batchsize + i] + k] =
            h_in[h_embedding_offset_sample_first[i * num_tables + j] + k];
      }
    }
  }
}

template <typename T>
void inc_var(volatile T* x, hipStream_t stream);

inline int64_t divup(int64_t x, int64_t y) { return (x + y - 1) / y; }

template <class T>
inline void hash_combine(std::size_t& s, const T& v) {
  std::hash<T> h;
  s ^= h(v) + 0x9e3779b9 + (s << 6) + (s >> 2);
}
}  // namespace HugeCTR
