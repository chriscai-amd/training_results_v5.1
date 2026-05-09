// HugeCTR-on-ROCm embedding-lookup runtime test.
// Uses the legacy HashTable<key,val> class (from include/hashtable/nv_hashtable.hpp)
// which is the simplest end-to-end exercise of the embedding storage path:
//   * Create GPU hash table (HierarchicalKV-based on AMD)
//   * Insert N <key, value> pairs
//   * Look up the same N keys, verify values
//
// This exercises the actual HugeCTR embedding storage code path on AMD MI350X
// kernels, hash function (MurmurHash3), atomic CAS hash insert, atomic add
// counter, slab-style allocation -- all the parts that worried us about
// wave32 vs wave64 correctness.

#include <hip/hip_runtime.h>

#include <hashtable/nv_hashtable.hpp>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#define CHECK_HIP(call)                                                       \
  do {                                                                        \
    hipError_t e = (call);                                                    \
    if (e != hipSuccess) {                                                    \
      std::fprintf(stderr, "[FAIL] %s -> %s\n", #call, hipGetErrorString(e)); \
      std::exit(1);                                                           \
    }                                                                         \
  } while (0)

int main() {
  using KeyType = long long;
  using ValType = unsigned long;
  using HugeCTR::HashTable;

  std::cout << "==== HugeCTR-on-ROCm embedding lookup runtime test ====" << std::endl;

  CHECK_HIP(hipSetDevice(0));
  hipDeviceProp_t props{};
  CHECK_HIP(hipGetDeviceProperties(&props, 0));
  std::cout << "[ok] GPU 0 = " << props.name << "  warpSize=" << props.warpSize << std::endl;

  hipStream_t stream;
  CHECK_HIP(hipStreamCreate(&stream));

  // Stress sizes -- enough to exercise multiple wavefronts and hash collisions.
  // 1M keys / 4M capacity = 25% load factor; collisions guaranteed.
  constexpr size_t N = 1u << 20;       // 1,048,576 keys
  constexpr size_t CAPACITY = 1u << 22; // 4,194,304 slots

  // --- Build host data: keys = 0..N-1 (shuffled), values = (key * 7) + 13 ----
  std::vector<KeyType> h_keys(N);
  std::vector<ValType> h_vals(N);
  std::vector<KeyType> h_query(N);
  std::vector<ValType> h_result(N, std::numeric_limits<ValType>::max());

  std::mt19937 rng(42);
  for (size_t i = 0; i < N; ++i) {
    h_keys[i] = static_cast<KeyType>(i);
    h_vals[i] = static_cast<ValType>(i) * 7ull + 13ull;
  }
  std::shuffle(h_keys.begin(), h_keys.end(), rng);
  // Reorder values to match shuffled keys
  for (size_t i = 0; i < N; ++i) {
    h_vals[i] = static_cast<ValType>(h_keys[i]) * 7ull + 13ull;
  }
  // Query order: original 0..N-1
  for (size_t i = 0; i < N; ++i) h_query[i] = static_cast<KeyType>(i);

  // --- Allocate device buffers ----------------------------------------------
  KeyType* d_keys = nullptr;
  ValType* d_vals = nullptr;
  KeyType* d_query = nullptr;
  ValType* d_result = nullptr;
  CHECK_HIP(hipMalloc(&d_keys, N * sizeof(KeyType)));
  CHECK_HIP(hipMalloc(&d_vals, N * sizeof(ValType)));
  CHECK_HIP(hipMalloc(&d_query, N * sizeof(KeyType)));
  CHECK_HIP(hipMalloc(&d_result, N * sizeof(ValType)));

  CHECK_HIP(hipMemcpy(d_keys, h_keys.data(), N * sizeof(KeyType), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_vals, h_vals.data(), N * sizeof(ValType), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(d_query, h_query.data(), N * sizeof(KeyType), hipMemcpyHostToDevice));

  // --- Create the hash table on GPU -----------------------------------------
  std::cout << "[..] constructing HashTable<long long, unsigned long>(capacity="
            << CAPACITY << ")" << std::endl;
  HashTable<KeyType, ValType> table(CAPACITY);
  std::cout << "[ok] HashTable constructed" << std::endl;

  // --- Insert ---------------------------------------------------------------
  std::cout << "[..] inserting " << N << " <key,value> pairs" << std::endl;
  table.insert(d_keys, d_vals, N, stream);
  CHECK_HIP(hipStreamSynchronize(stream));
  std::cout << "[ok] insert(" << N << ") completed; size=" << table.get_size(stream) << std::endl;

  if (table.get_size(stream) != N) {
    std::cerr << "[FAIL] expected size=" << N << " got " << table.get_size(stream) << std::endl;
    return 1;
  }

  // --- Lookup ---------------------------------------------------------------
  std::cout << "[..] looking up " << N << " keys" << std::endl;
  table.get(d_query, d_result, N, stream);
  CHECK_HIP(hipStreamSynchronize(stream));
  std::cout << "[ok] get(" << N << ") completed" << std::endl;

  CHECK_HIP(hipMemcpy(h_result.data(), d_result, N * sizeof(ValType), hipMemcpyDeviceToHost));

  // --- Verify ---------------------------------------------------------------
  size_t mismatches = 0;
  for (size_t i = 0; i < N; ++i) {
    ValType expect = static_cast<ValType>(i) * 7ull + 13ull;
    if (h_result[i] != expect) {
      if (mismatches < 5) {
        std::cerr << "  mismatch at i=" << i << ": got " << h_result[i] << " want " << expect
                  << std::endl;
      }
      ++mismatches;
    }
  }
  if (mismatches > 0) {
    std::cerr << "[FAIL] " << mismatches << " / " << N << " lookup mismatches" << std::endl;
    return 1;
  }
  std::cout << "[ok] all " << N << " lookups returned correct values" << std::endl;

  CHECK_HIP(hipFree(d_keys));
  CHECK_HIP(hipFree(d_vals));
  CHECK_HIP(hipFree(d_query));
  CHECK_HIP(hipFree(d_result));
  CHECK_HIP(hipStreamDestroy(stream));

  std::cout << "==== EMBEDDING LOOKUP TEST PASSED on " << props.name << " ====" << std::endl;
  return 0;
}
