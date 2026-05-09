// Minimal HugeCTR-on-ROCm runtime smoke test.
//
// Goals (in order of difficulty, each a checkpoint):
//   1) Link against the 4 ported libs and load them at runtime.
//   2) Initialize a HugeCTR::core23::Device for GPU 0 (MI350X).
//   3) Allocate a Tensor on the device, fill it from host, copy back, verify.
//   4) Use the ported MLCommon::LinAlg shim to run a unaryOp/binaryOp on GPU.
//
// Build via runtime/CMakeLists.txt; run on the reserved compute node.

#include <hip/hip_runtime.h>

#include <core23/buffer_params.hpp>
#include <core23/device.hpp>
#include <core23/device_type.hpp>
#include <core23/data_type.hpp>
#include <core23/shape.hpp>
#include <core23/tensor.hpp>
#include <core23/tensor_params.hpp>
#include <core23/logger.hpp>
#include <prims/mlcommon_linalg_hip.cuh>

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#define CHECK_HIP(call)                                                                  \
  do {                                                                                   \
    hipError_t e = (call);                                                               \
    if (e != hipSuccess) {                                                               \
      std::fprintf(stderr, "[FAIL] %s -> %s\n", #call, hipGetErrorString(e));            \
      std::exit(1);                                                                      \
    }                                                                                    \
  } while (0)

int main() {
  using namespace HugeCTR;

  std::cout << "==== HugeCTR-on-ROCm smoke test ====" << std::endl;

  // --- 1. Raw HIP sanity: enumerate devices ----------------------------------
  int dev_count = 0;
  CHECK_HIP(hipGetDeviceCount(&dev_count));
  std::cout << "[ok] hipGetDeviceCount = " << dev_count << " GPUs" << std::endl;
  if (dev_count == 0) {
    std::cerr << "[FAIL] no GPUs visible to HIP" << std::endl;
    return 1;
  }
  hipDeviceProp_t props{};
  CHECK_HIP(hipGetDeviceProperties(&props, 0));
  std::cout << "[ok] GPU 0 = " << props.name
            << "  VRAM=" << (props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0)) << " GB"
            << "  warpSize=" << props.warpSize << std::endl;

  CHECK_HIP(hipSetDevice(0));

  // --- 2. Construct a HugeCTR core23::Device for GPU 0 -----------------------
  core23::Device device(core23::DeviceType::GPU, 0);
  std::cout << "[ok] core23::Device constructed for GPU 0" << std::endl;

  // --- 3. Allocate a Tensor on that device, fill, verify ---------------------
  const int64_t N = 1024;
  core23::Shape shape({N});

  core23::TensorParams params = core23::TensorParams()
                                    .device(device)
                                    .shape(shape)
                                    .data_type(core23::ScalarType::Float);
  core23::Tensor t(params);

  // Touch the storage (forces lazy alloc on devices that defer it).
  void* dptr = t.data();
  std::cout << "[ok] core23::Tensor allocated  size=" << t.num_bytes()
            << "B  device_ptr=" << dptr << std::endl;

  // Fill with a simple ramp on host, copy to device, copy back, verify.
  std::vector<float> host_in(N);
  for (int i = 0; i < N; ++i) host_in[i] = static_cast<float>(i);
  CHECK_HIP(hipMemcpy(dptr, host_in.data(), N * sizeof(float), hipMemcpyHostToDevice));
  std::vector<float> host_out(N, -1.0f);
  CHECK_HIP(hipMemcpy(host_out.data(), dptr, N * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) {
    if (host_out[i] != host_in[i]) {
      std::cerr << "[FAIL] tensor roundtrip mismatch at " << i
                << ": got " << host_out[i] << " want " << host_in[i] << std::endl;
      return 1;
    }
  }
  std::cout << "[ok] tensor host->device->host roundtrip verified" << std::endl;

  // --- 4. Run a HugeCTR-shim kernel on the device ----------------------------
  // y = x * 2 + 1 via unaryOp (uses our ROCm-port MLCommon::LinAlg shim).
  float* d_in = static_cast<float*>(dptr);
  float* d_out = nullptr;
  CHECK_HIP(hipMalloc(&d_out, N * sizeof(float)));
  MLCommon::LinAlg::unaryOp(
      d_out, d_in, static_cast<std::size_t>(N),
      [] __device__(float v) { return v * 2.0f + 1.0f; });
  CHECK_HIP(hipDeviceSynchronize());

  CHECK_HIP(hipMemcpy(host_out.data(), d_out, N * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) {
    float expect = host_in[i] * 2.0f + 1.0f;
    if (host_out[i] != expect) {
      std::cerr << "[FAIL] unaryOp mismatch at " << i
                << ": got " << host_out[i] << " want " << expect << std::endl;
      return 1;
    }
  }
  std::cout << "[ok] MLCommon::LinAlg::unaryOp(y = 2x+1) verified for N=" << N << std::endl;

  // --- 5. binaryOp: z = x + y -----------------------------------------------
  float* d_z = nullptr;
  CHECK_HIP(hipMalloc(&d_z, N * sizeof(float)));
  MLCommon::LinAlg::binaryOp(
      d_z, d_in, d_out, static_cast<std::size_t>(N),
      [] __device__(float a, float b) { return a + b; });
  CHECK_HIP(hipDeviceSynchronize());
  CHECK_HIP(hipMemcpy(host_out.data(), d_z, N * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < N; ++i) {
    float expect = host_in[i] + (host_in[i] * 2.0f + 1.0f);
    if (host_out[i] != expect) {
      std::cerr << "[FAIL] binaryOp mismatch at " << i
                << ": got " << host_out[i] << " want " << expect << std::endl;
      return 1;
    }
  }
  std::cout << "[ok] MLCommon::LinAlg::binaryOp(z = x + (2x+1)) verified" << std::endl;

  CHECK_HIP(hipFree(d_out));
  CHECK_HIP(hipFree(d_z));

  std::cout << "==== ALL CHECKS PASSED on " << props.name << " ====" << std::endl;
  return 0;
}
