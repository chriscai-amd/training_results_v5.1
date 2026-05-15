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

#include <hiprand/hiprand.h>

#include <core23/hip_stream.hpp>
#include <core23/device.hpp>
#include <functional>
#include <memory>

namespace HugeCTR {
namespace core23 {

class CURANDGenerator final {
 public:
  CURANDGenerator(const Device& device, unsigned long long seed = 0,
                  hiprandRngType_t rng_type = HIPRAND_RNG_PSEUDO_DEFAULT)
      : generator_(
            [device, rng_type, seed]() {
              hiprandGenerator_t* generator = new hiprandGenerator_t;
              if (device.type() == DeviceType::CPU) {
                hiprandCreateGeneratorHost(generator, rng_type);
              } else {
                hiprandCreateGenerator(generator, rng_type);
              }
              hiprandSetPseudoRandomGeneratorSeed(*generator, seed);
              return generator;
            }(),
            [](hiprandGenerator_t* generator) {
              hiprandDestroyGenerator(*generator);
              delete generator;
            }) {}
  CURANDGenerator(const CURANDGenerator&) = default;
  CURANDGenerator(CURANDGenerator&&) = delete;
  CURANDGenerator& operator=(const CURANDGenerator&) = default;
  CURANDGenerator& operator=(CURANDGenerator&&) = delete;
  ~CURANDGenerator() = default;

  void set_stream(CUDAStream stream) { hiprandSetStream(generator(), stream()); }

  hiprandGenerator_t operator()() const { return generator(); }
  explicit operator hiprandGenerator_t() const noexcept { return generator(); }

 private:
  hiprandGenerator_t generator() const { return generator_ ? *generator_ : 0; }
  std::shared_ptr<hiprandGenerator_t> generator_;
};

inline bool operator==(CURANDGenerator lhs, CURANDGenerator rhs) { return lhs() == rhs(); }

inline bool operator!=(CURANDGenerator lhs, CURANDGenerator rhs) { return !(lhs() == rhs()); }

}  // namespace core23
}  // namespace HugeCTR
