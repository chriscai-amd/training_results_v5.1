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

#include <functional>
#include <memory>

namespace HugeCTR {
namespace core23 {

class CUDAStream final {
 public:
  CUDAStream() : stream_(nullptr) {}
  CUDAStream(hipStream_t outer_stream) : stream_(new hipStream_t) { *stream_ = outer_stream; }
  CUDAStream(int flags, int priority = 0)
      : stream_(
            [flags, priority]() {
              hipStream_t* stream = new hipStream_t;
              hipStreamCreateWithPriority(stream, flags, priority);
              return stream;
            }(),
            [](hipStream_t* stream) {
              hipStreamDestroy(*stream);
              delete stream;
            }) {}
  CUDAStream(const CUDAStream&) = default;
  CUDAStream(CUDAStream&&) = default;
  CUDAStream& operator=(const CUDAStream&) = default;
  CUDAStream& operator=(CUDAStream&&) = delete;
  ~CUDAStream() = default;

  hipStream_t operator()() const { return stream(); }
  explicit operator hipStream_t() const noexcept { return stream(); }

 private:
  hipStream_t stream() const { return stream_ ? *stream_ : 0; }
  std::shared_ptr<hipStream_t> stream_;
};

inline bool operator==(CUDAStream lhs, CUDAStream rhs) { return lhs() == rhs(); }

inline bool operator!=(CUDAStream lhs, CUDAStream rhs) { return !(lhs() == rhs()); }

}  // namespace core23
}  // namespace HugeCTR
