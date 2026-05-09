// HugeCTR ROCm port: stub implementations for the StaticHashTable<...> template
// only. The full .cu source is excluded because amd_hip_fp8.h annotates its
// conversion operators as __host__ only, breaking template instantiation in
// __device__ context. The stubs throw at runtime if hit; DLRM-DCNv2 doesn't
// exercise the FP8 cache codepaths in practice.

#include <atomic>
#include <mutex>
#include <thread>

#include <static_hash_table.hpp>
#include <hash_functions.cuh>

#include <stdexcept>
#include <cstdint>

#define HUGECTR_ROCM_STUB_THROW()                                              \
  throw std::runtime_error(                                                    \
      "HugeCTR ROCm port: gpu_cache::StaticHashTable stubbed; FP8-using "      \
      "kernels disabled on ROCm 7.0 due to a __device__ annotation gap "      \
      "in amd_hip_fp8.h.")

namespace gpu_cache {

template <typename Key, typename Value, typename OutValue, unsigned int TileSize,
          unsigned int GroupSize, typename Hasher>
StaticHashTable<Key, Value, OutValue, TileSize, GroupSize, Hasher>::StaticHashTable(
    size_type capacity, int value_dim, bool enable_pagelock_in, Hasher hash)
    : table_keys_(nullptr),
      table_indices_(nullptr),
      key_capacity_(capacity),
      table_values_(nullptr),
      quant_scales_(nullptr),
      value_capacity_(capacity),
      value_dim_(value_dim),
      size_(0),
      enable_pagelock(enable_pagelock_in),
      hash_(hash) {}

template <typename Key, typename Value, typename OutValue, unsigned int TileSize,
          unsigned int GroupSize, typename Hasher>
StaticHashTable<Key, Value, OutValue, TileSize, GroupSize, Hasher>::~StaticHashTable() = default;

template <typename Key, typename Value, typename OutValue, unsigned int TileSize,
          unsigned int GroupSize, typename Hasher>
void StaticHashTable<Key, Value, OutValue, TileSize, GroupSize, Hasher>::clear(hipStream_t) {
  size_ = 0;
}

template <typename Key, typename Value, typename OutValue, unsigned int TileSize,
          unsigned int GroupSize, typename Hasher>
void StaticHashTable<Key, Value, OutValue, TileSize, GroupSize, Hasher>::insert(
    const Key*, const Value*, size_type, hipStream_t, const float*) {
  HUGECTR_ROCM_STUB_THROW();
}

template <typename Key, typename Value, typename OutValue, unsigned int TileSize,
          unsigned int GroupSize, typename Hasher>
void StaticHashTable<Key, Value, OutValue, TileSize, GroupSize, Hasher>::lookup(
    const Key*, OutValue*, int, OutValue, hipStream_t) {
  HUGECTR_ROCM_STUB_THROW();
}

// Explicit instantiations -- expand the matrix to also cover FP8 value/output
// combinations referenced by HugeCTR's high-level paths.
#include <hip/hip_fp8.h>
using fp8_e4m3 = __hip_fp8_e4m3_fnuz;

#define HCTR_INST(K, V, O) \
  template class StaticHashTable<K, V, O, 4u, 16u, MurmurHash3_32<K>>;

#define HCTR_INST_K(K)         \
  HCTR_INST(K, float,   float) \
  HCTR_INST(K, float,   __half)\
  HCTR_INST(K, float,   fp8_e4m3) \
  HCTR_INST(K, __half,  float) \
  HCTR_INST(K, __half,  __half)\
  HCTR_INST(K, __half,  fp8_e4m3) \
  HCTR_INST(K, fp8_e4m3, float) \
  HCTR_INST(K, fp8_e4m3, __half) \
  HCTR_INST(K, fp8_e4m3, fp8_e4m3)

// Cover all integer key widths/signs the dynamic linker may demand.
// On AMD64 Linux, `int64_t==long`, so include `long long`/`unsigned long long`
// separately as distinct types.
HCTR_INST_K(uint32_t)
HCTR_INST_K(int32_t)
HCTR_INST_K(int64_t)
HCTR_INST_K(uint64_t)
HCTR_INST_K(long long)
HCTR_INST_K(unsigned long long)

#undef HCTR_INST_K
#undef HCTR_INST

}  // namespace gpu_cache
