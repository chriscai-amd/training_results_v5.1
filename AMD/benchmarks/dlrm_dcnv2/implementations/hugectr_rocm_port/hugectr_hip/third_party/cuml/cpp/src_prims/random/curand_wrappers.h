/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <hiprand.h>

namespace MLCommon {
namespace Random {

/** check for hiprand runtime API errors and assert accordingly */
#define CURAND_CHECK(call)                                        \
  do {                                                            \
    hiprandStatus_t status = call;                                 \
    ASSERT(status == HIPRAND_STATUS_SUCCESS,                       \
           "FAIL: hiprand-call='%s'. Reason:%d\n", #call, status); \
  } while (0)

/**
 * @defgroup normal hiprand normal random number generation operations
 * @{
 */
template <typename T>
hiprandStatus_t hiprandGenerateNormal(hiprandGenerator_t generator, T *outputPtr,
                                    size_t n, T mean, T stddev);

template <>
inline hiprandStatus_t hiprandGenerateNormal(hiprandGenerator_t generator,
                                           float *outputPtr, size_t n,
                                           float mean, float stddev) {
  return hiprandGenerateNormal(generator, outputPtr, n, mean, stddev);
}

template <>
inline hiprandStatus_t hiprandGenerateNormal(hiprandGenerator_t generator,
                                           double *outputPtr, size_t n,
                                           double mean, double stddev) {
  return hiprandGenerateNormalDouble(generator, outputPtr, n, mean, stddev);
}
/** @} */

};  // end namespace Random
};  // end namespace MLCommon
