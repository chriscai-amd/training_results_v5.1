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

#include <utils.hpp>
#include <cstdlib>

namespace HugeCTR {

class StreamEventManager {
 public:
  HCTR_DISALLOW_COPY_AND_MOVE(StreamEventManager);

  StreamEventManager() {}

  ~StreamEventManager() {
    for (auto& s : stream_map_) {
      hipStreamDestroy(s.second);
    }
    for (auto& e : event_map_) {
      hipEventDestroy(e.second);
    }
  }

  const hipStream_t& get_stream(const std::string& key, unsigned int flags = 0, int priority = 0) {
    if (stream_map_.find(key) == stream_map_.end()) {
      hipStream_t stream;
      // ROCm-port Technique A (2026-05-13): CU-mask the "mp" and "dp" streams so
      // RCCL kernels (which fire on the current stream) don't saturate all 304 CUs
      // and block concurrent compute on the "default" stream. AMD analog of NV's
      // NVLS hardware multicast: limits RCCL CU footprint so MLP GEMMs can run in
      // parallel. Trace evidence: bs1x exposed RCCL = 0.94 ms (22% of iter); 0% of
      // FWD a2a hides behind compute today. See README §Phase 14g.
      // Phase 18 (2026-05-15): also apply CU mask to the dedicated RCCL streams
      // introduced in Phase 15 ("rccl_emb_ar" and "rccl_mlp_wgrad"). Phase 14g
      // showed CU-masking alone was flat because RCCL still shared the default
      // stream with MLP; combining with Phase 15 stream isolation should now
      // compound by leaving more CUs available for MLP compute on default.
      static const int cu_mask_bits = []() {
        const char* env = std::getenv("HCTR_RCCL_CU_MASK_BITS");
        return env ? std::atoi(env) : 0;
      }();
      const bool should_mask = cu_mask_bits > 0 &&
          (key == "mp" || key == "dp" ||
           key == "rccl_emb_ar" || key == "rccl_mlp_wgrad");
      if (should_mask) {
        // gfx950 has 304 CUs; allocate enough bits then build contiguous low-bit mask
        constexpr int N_DWORDS = 10;  // 320 bits (covers 304 CUs)
        uint32_t mask[N_DWORDS] = {0};
        int bits_left = cu_mask_bits;
        for (int i = 0; i < N_DWORDS && bits_left > 0; ++i) {
          const int b = (bits_left >= 32) ? 32 : bits_left;
          mask[i] = (b == 32) ? 0xFFFFFFFFu : ((1u << b) - 1);
          bits_left -= b;
        }
        HCTR_LIB_THROW(hipExtStreamCreateWithCUMask(&stream, N_DWORDS, mask));
      } else {
        HCTR_LIB_THROW(hipStreamCreateWithPriority(&stream, flags, priority));
      }
      stream_map_[key] = stream;
    }
    return stream_map_.at(key);
  }

  const hipStream_t& get_stream(const std::string& key) const {
    HCTR_CHECK_HINT(stream_map_.find(key) != stream_map_.end(),
                    "StreamEventManager not contain stream %s", key.c_str());
    return stream_map_.at(key);
  }

  hipEvent_t& get_event(const std::string& key, unsigned int flags = 0) {
    if (event_map_.find(key) == event_map_.end()) {
      hipEvent_t event;
      HCTR_LIB_THROW(hipEventCreateWithFlags(&event, flags));
      event_map_[key] = event;
    }
    return event_map_[key];
  }

 private:
  std::unordered_map<std::string, hipStream_t> stream_map_;
  std::unordered_map<std::string, hipEvent_t> event_map_;
};

}  // namespace HugeCTR