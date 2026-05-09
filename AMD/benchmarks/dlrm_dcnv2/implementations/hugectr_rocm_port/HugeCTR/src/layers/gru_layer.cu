#include <prims/mlcommon_linalg_hip.cuh>
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

#include <algorithm>
#include <common.hpp>
#include <functional>
#include <gpu_resource.hpp>
#include <include/utils.cuh>
#include <layers/gru_layer.hpp>
// HugeCTR ROCm port: cuml linalg replaced by hand-written HIP shim
// HugeCTR ROCm port: cuml linalg replaced by hand-written HIP shim
// HugeCTR ROCm port: cuml linalg replaced by hand-written HIP shim
#include <utils.cuh>
#include <utils.hpp>
#include <vector>

namespace HugeCTR {

template <typename T>
GRULayer<T>::GRULayer(const core23::Tensor& in_tensor, const core23::Tensor& out_tensor,
                      int64_t hiddenSize, int64_t batch_size, int64_t SeqLength,
                      int64_t embedding_vec_size, const std::shared_ptr<GPUResource>& gpu_resource,
                      std::vector<Initializer_t> initializer_types)
    : TrainableLayer<T>({in_tensor}, {out_tensor}, gpu_resource, initializer_types) {
  try {
    CudaDeviceContext context(this->get_device_id());
    // check the in_tensor and out_tensor
    const auto& in_tensor_dim = in_tensor.shape();
    const auto& out_tensor_dim = out_tensor.shape();

    // 2. dim match?
    // seqLength = in_tensor_dim[1];
    // m = out_tensor_dim[1];
    // miniBatch = in_tensor_dim[0];
    // HCTR_LOG(INFO, WORLD, "m %lu n %lu k %lu \n ", m, n,k);
    hiddenSize_ = hiddenSize;
    miniBatch = batch_size;
    seqLength_ = SeqLength;
    embedding_vec_size_ = embedding_vec_size;

    inputTensorSize = miniBatch * seqLength_ * embedding_vec_size_;
    outputTensorSize = miniBatch * seqLength_ * hiddenSize_;
    hiddenTensorSize = miniBatch * hiddenSize_;

    // weightSpaceSize = m*k + m*m + 1*m; //include W, U weight matrixs and bias vector.

    // HCTR_LIB_THROW(hipdnnSetTensor4dDescriptorEx(hDesc, data_type, n, 1, 1, n,
    //  n, 1, 1, 1));

    // HCTR_LIB_THROW(hipdnnSetTensor4dDescriptorEx(cDesc, data_type, 1, n, m, n,
    //  n, 1, 1, 1));
    seqLengthArray = new int[miniBatch];

    for (size_t i = 0; i < miniBatch; i++) {
      seqLengthArray[i] = seqLength_;
    }

    // cudnnHandle= get_gpu().get_cudnn_handle();
    HCTR_LIB_THROW(hipdnnCreate(&cudnnHandle));
    data_type = CudnnDataType<T>::getType();
    HCTR_LIB_THROW(hipdnnCreateRNNDescriptor(&rnnDesc));
    HCTR_LIB_THROW(cudnnCreateRNNDataDescriptor(&in_Desc));
    HCTR_LIB_THROW(cudnnCreateRNNDataDescriptor(&out_Desc));
    HCTR_LIB_THROW(hipdnnCreateTensorDescriptor(&cDesc));
    HCTR_LIB_THROW(hipdnnCreateTensorDescriptor(&hDesc));
    HCTR_LIB_THROW(hipdnnCreateDropoutDescriptor(&dropoutDesc));

    HCTR_LIB_THROW(cudnnSetRNNDataDescriptor(
        in_Desc,                                   // cudnnRNNDataDescriptor_t RNNDataDesc,
        data_type,                                 // hipdnnDataType_t dataType,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,  // CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                                   // //cudnnRNNDataLayout_t layout,
        seqLength_,                                // int maxSeqLength,
        miniBatch,                                 // int batchSize,
        embedding_vec_size_,                       // int vectorSize,
        seqLengthArray,                            // const int seqLengthArray[],
        NULL                                       // void *paddingFill
        ));

    HCTR_LIB_THROW(cudnnSetRNNDataDescriptor(
        out_Desc,                                  // cudnnRNNDataDescriptor_t RNNDataDesc,
        data_type,                                 // hipdnnDataType_t dataType,
        CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_UNPACKED,  // CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED,
                                                   // //cudnnRNNDataLayout_t layout,
        seqLength_,                                // int maxSeqLength,
        miniBatch,                                 // int batchSize,
        hiddenSize_,                               // int vectorSize,
        seqLengthArray,                            // const int seqLengthArray[],
        NULL                                       // void *paddingFill
        ));
    dimHidden[0] = 1 * 1;
    dimHidden[1] = miniBatch;
    dimHidden[2] = hiddenSize_;
    strideHidden[0] = dimHidden[1] * dimHidden[2];
    strideHidden[1] = dimHidden[2];
    strideHidden[2] = 1;
    HCTR_LIB_THROW(hipdnnSetTensorNdDescriptor(hDesc, data_type, 3, dimHidden, strideHidden));
    HCTR_LIB_THROW(hipdnnSetTensorNdDescriptor(cDesc, data_type, 3, dimHidden, strideHidden));

    HCTR_LIB_THROW(hipdnnDropoutGetStatesSize(cudnnHandle, &stateSize));
    HCTR_LIB_THROW(hipMalloc(&states, stateSize));
    seed = 0;  // 1337ull;
    HCTR_LIB_THROW(
        hipdnnSetDropoutDescriptor(dropoutDesc, cudnnHandle, dropout, states, stateSize, seed));

    HCTR_LIB_THROW(cudnnSetRNNDescriptor_v8(
        rnnDesc,
        HIPDNN_RNN_ALGO_STANDARD,    // hipdnnRNNAlgo_t algo,
        HIPDNN_GRU,                  // hipdnnRNNMode_t cellMode,
        HIPDNN_RNN_WITH_BIAS,  // hipdnnRNNBiasMode_t biasMode,
        HIPDNN_UNIDIRECTIONAL,       // hipdnnDirectionMode_t dirMode,
        HIPDNN_LINEAR_INPUT,         // HIPDNN_SKIP_INPUT, //HIPDNN_LINEAR_INPUT, //hipdnnRNNInputMode_t
                             // inputMode, HIPDNN_SKIP_INPUT :without multiplying input by the weight
                             // matrix
        data_type,             // hipdnnDataType_t dataType,
        data_type,             // hipdnnDataType_t mathPrec,
        HIPDNN_TENSOR_OP_MATH,  // HIPDNN_DEFAULT_MATH , //hipdnnMathType_t mathType,
        embedding_vec_size_,   // int32_t embedding_vec_size, When the inputMode=HIPDNN_SKIP_INPUT,
                               // the embedding_vec_size should match the hiddenSize value
        hiddenSize_,           // int32_t hiddenSize,
        hiddenSize_,           // int32_t projSize,
        1,                     // int32_t numLayers, BIDIRECTIONAL=2
        dropoutDesc,           // hipdnnDropoutDescriptor_t dropoutDesc,
        CUDNN_RNN_PADDED_IO_DISABLED  // uint32_t auxFlags
        ));

    // const int seqLengthArray[in_tensor_dim[0]] = { [0...10] = int(in_tensor_dim[1]) };
    // const int seqLengthArray[m] ={n,n....n};
    // for(int i=0; i<in_tensor_dim[1]; i++)
    // = { [0 . . . 3 ] = 3 };

    HCTR_LIB_THROW(cudnnGetRNNWeightSpaceSize(cudnnHandle, rnnDesc, &weightSpaceSize));
    HCTR_LIB_THROW(cudnnGetRNNTempSpaceSizes(cudnnHandle, rnnDesc, CUDNN_FWD_MODE_TRAINING, in_Desc,
                                             &workSpaceSize, &reserveSpaceSize));
    // std::vector<size_t> weight_dim = {weightSpaceSize/sizeof(T), 1};
    // std::vector<size_t> dx_dim =  {inputTensorSize, 1};
    // std::vector<size_t> dy_dim =  {outputTensorSize, 1};
    // std::vector<size_t> dhx_dim = {hiddenTensorSize, 1};
    // std::vector<size_t> dhy_dim = {hiddenTensorSize, 1};
    // std::vector<size_t> dcx_dim = {hiddenTensorSize, 1};
    // std::vector<size_t> dcy_dim = {hiddenTensorSize, 1};

    core23::Shape weight_dim = {1, static_cast<int64_t>(weightSpaceSize / sizeof(T))};
    core23::Shape hx_dim = {1, static_cast<int64_t>(hiddenTensorSize)};
    core23::Shape dx_dim = {1, static_cast<int64_t>(inputTensorSize)};
    core23::Shape dy_dim = {1, static_cast<int64_t>(outputTensorSize)};
    core23::Shape dhx_dim = {1, static_cast<int64_t>(hiddenTensorSize)};
    core23::Shape dhy_dim = {1, static_cast<int64_t>(hiddenTensorSize)};
    core23::Shape dweigths_dim = {1, static_cast<int64_t>(weightSpaceSize / sizeof(T))};
    // HCTR_LOG(INFO, WORLD, "weighsize %zu\n", weightSpaceSize/sizeof(T));

    this->set_weight(0, weight_dim);
    this->set_weight(1, hx_dim);
    this->set_wgrad(0, dx_dim);
    this->set_wgrad(1, dy_dim);
    this->set_wgrad(2, dhx_dim);
    this->set_wgrad(3, dhy_dim);
    this->set_wgrad(4, dweigths_dim);

    HCTR_LIB_THROW(hipMalloc((void**)&devSeqLengthArray, miniBatch * sizeof(int)));
    HCTR_LIB_THROW(hipMemcpy(devSeqLengthArray, seqLengthArray, miniBatch * sizeof(int),
                              hipMemcpyHostToDevice));
    HCTR_LIB_THROW(hipMalloc((void**)&weightSpace, weightSpaceSize));
    HCTR_LIB_THROW(hipMalloc((void**)&workSpace, workSpaceSize));
    HCTR_LIB_THROW(hipMalloc((void**)&reserveSpace, reserveSpaceSize));
    // HCTR_LIB_THROW(hipMalloc((void **)&dweightSpace, weightSpaceSize));

  } catch (const std::runtime_error& rt_err) {
    HCTR_LOG_S(ERROR, WORLD) << rt_err.what() << std::endl;
    throw;
  }
}

//#define KERAS_CHECK
template <typename T>
void GRULayer<T>::fprop(bool is_train) {
  CudaDeviceContext context(this->get_device_id());

  core23::Tensor& in_tensor = get_in_tensors(is_train)[0];
  core23::Tensor& out_tensor = this->output_tensors_[0];

  T* weight = this->get_weight(0).template data<T>();
  T* hx = this->get_weight(1).template data<T>();

  T* in = in_tensor.template data<T>();
  T* out = out_tensor.template data<T>();

#ifdef KERAS_CHECK
  hipdnnTensorDescriptor_t wDesc;
  hipdnnTensorDescriptor_t bDesc;
  HCTR_LIB_THROW(hipdnnCreateTensorDescriptor(&wDesc));
  HCTR_LIB_THROW(hipdnnCreateTensorDescriptor(&bDesc));

  // core23::Tensor linLayerMat;
  // core23::Tensor linLayerBias;
  numLinearLayers = 6;  // cellMode == HIPDNN_GRU
  for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++) {
    T* linLayerMat = NULL;
    T* linLayerBias = NULL;
    int nbDims = 0;
    int dim[3] = {0, 0, 0}, stride[3];
    int layer = 0;
    // HCTR_LOG(INFO, WORLD, "weightSpaceSize %zu\n", weightSpaceSize);
    HCTR_LIB_THROW(cudnnGetRNNWeightParams(cudnnHandle, rnnDesc, layer, weightSpaceSize,
                                           weights_[0].get_ptr(),  // weightSpace,
                                           linLayerID, wDesc,
                                           (void**)&linLayerMat,  //.get_ptr(),
                                           bDesc,
                                           (void**)&linLayerBias  //.get_ptr()
                                           ));

    if (linLayerMat) {
      HCTR_LIB_THROW(hipdnnGetTensorNdDescriptor(wDesc, 3, &data_type, &nbDims, dim, stride));
      size_t w = dim[0] * dim[1] * dim[2];
      T* h_weights = new T[w];
      HCTR_LIB_THROW(hipMemcpy(h_weights, linLayerMat, sizeof(T) * w, hipMemcpyDeviceToHost));

      HCTR_LOG(INFO, ROOT, "W_%d %zu ", linLayerID, w);
      for (unsigned int i = 0; i < w; i++) {
        HCTR_PRINT(INFO, "%f ", h_weights[i]);
      }
      HCTR_PRINT(INFO, "\n");

      delete[] h_weights;
    }

    if (linLayerBias) {
      HCTR_LIB_THROW(hipdnnGetTensorNdDescriptor(bDesc, 3, &data_type, &nbDims, dim, stride));
      size_t w = dim[0] * dim[1] * dim[2];
      T* h_weights = new T[w];
      HCTR_LIB_THROW(hipMemcpy(h_weights, linLayerBias, sizeof(T) * w, hipMemcpyDeviceToHost));

      HCTR_LOG(INFO, ROOT, "B_%d %zu ", linLayerID, w);
      for (unsigned int i = 0; i < w; i++) {
        HCTR_PRINT(INFO, "%f ", h_weights[i]);
      }
      HCTR_PRINT(INFO, "\n");

      delete[] h_weights;
    }
  }

  HCTR_LIB_THROW(hipdnnDestroyTensorDescriptor(wDesc));
  HCTR_LIB_THROW(hipdnnDestroyTensorDescriptor(bDesc));
#endif
  // CUDNN GRU
  // T tmp[hiddenTensorSize];
  // HCTR_LIB_THROW(hipMemcpy(tmp, weight + weightSpaceSize/sizeof(T), sizeof(T) *
  // hiddenTensorSize, hipMemcpyDeviceToHost)); for(size_t i=0;i<hiddenTensorSize;i++)
  //  if(tmp[i] != 0.0)
  //    HCTR_LOG(INFO, WORLD, "tmp[i] %f\n", tmp[i]);
  HCTR_LIB_THROW(cudnnRNNForward(
      cudnnHandle, rnnDesc, CUDNN_FWD_MODE_TRAINING, devSeqLengthArray,
      in_Desc,   // xDesc,
      in,        // x, input data pointer
      out_Desc,  // yDesc,
      out,       // y, output data pointer
      hDesc,
      NULL,   // hx, Input. Pointer to the GPU buffer with the RNN initial hidden state, NULL:
              // initialized zero.
      NULL,   // hy,  Output. Pointer to the GPU buffer where the final RNN hidden state should be
              // stored. NULL: not saved.
      cDesc,  // cDesc, Input. A tensor descriptor, for LSTM networks only.
      NULL,   // cx,
      NULL,   // cy,
      weightSpaceSize,
      weight,         // weightSpace, The weight space buffer holds all RNN weight matrices and bias
                      // vectors
      workSpaceSize,  // size_t workSpaceSize,
      workSpace,      // workSpace,
      reserveSpaceSize,
      reserveSpace  // reserveSpace
      ));

  // HCTR_LOG(INFO, WORLD, "forward end\n\n");
  // hipdnnDestroy(cudnnHandle);
}

template <typename T>
void GRULayer<T>::bprop() {
  CudaDeviceContext context(this->get_device_id());
  core23::Tensor& in_tensor = get_in_tensors(true)[0];
  core23::Tensor& out_tensor = this->output_tensors_[0];

  T* weight = this->get_weight(0).template data<T>();
  T* in = in_tensor.template data<T>();
  T* out = out_tensor.template data<T>();
  T* dx = this->get_wgrad(0).template data<T>();
  T* dy = this->get_wgrad(1).template data<T>();
  T* dhx = this->get_wgrad(2).template data<T>();
  T* dhy = this->get_wgrad(3).template data<T>();
  T* dweightSpace = this->get_wgrad(4).template data<T>();

  HCTR_LIB_THROW(cudnnRNNBackwardData_v8(cudnnHandle,        // hipdnnHandle_t handle,
                                         rnnDesc,            // hipdnnRNNDescriptor_t rnnDesc,
                                         devSeqLengthArray,  // const int32_t devSeqLengths[],
                                         out_Desc,           // cudnnRNNDataDescriptor_t yDesc,
                                         out,                // const void *y, input
                                         dy,                 // const void *dy, input
                                         in_Desc,            // cudnnRNNDataDescriptor_t xDesc,
                                         dx,                 // void *dx, output
                                         hDesc,              // hipdnnTensorDescriptor_t hDesc,
                                         NULL,               // hx, //const void *hx, input
                                         NULL,               // const void *dhy, input
                                         dhx,                // void *dhx, output
                                         cDesc,              // hipdnnTensorDescriptor_t cDesc,
                                         NULL,  // cx, //const void *cx, for LSTM only, input
                                         NULL,  // const void *dcy, for LSTM only, input
                                         NULL,  // void *dcx, output
                                         weightSpaceSize,
                                         weight,  // weightSpace,
                                         workSpaceSize, workSpace, reserveSpaceSize, reserveSpace));

  // hipdnnRNNBackwardWeights adds to the data in dw.
  HCTR_LIB_THROW(hipMemset(dweightSpace, 0, weightSpaceSize));
  // T* h_hx=NULL;
  // hipMemcpy(h_hx,hx,hiddenTensorSize*sizeof(T),hipMemcpyDeviceToHost );
  // for(unsigned int i=0;i<hiddenTensorSize;i++)
  //    HCTR_LOG(INFO, WORLD, "hx %f \n",h_hx[i]);

  HCTR_LIB_THROW(cudnnRNNBackwardWeights_v8(
      cudnnHandle, rnnDesc, CUDNN_WGRAD_MODE_ADD, devSeqLengthArray, in_Desc, in, hDesc,
      NULL,  // hx,
      out_Desc,
      out,  // output
      weightSpaceSize,
      dweightSpace,  // output
      workSpaceSize, workSpace, reserveSpaceSize, reserveSpace));
  HCTR_LIB_THROW(hipFree(workSpace));
  HCTR_LIB_THROW(hipFree(reserveSpace));
  HCTR_LIB_THROW(hipFree(weightSpace));  // hipFree(dweightSpace);
  // hipFree(x); hipFree(y); hipFree(hx);
  HCTR_LIB_THROW(hipFree(states));
  HCTR_LIB_THROW(cudnnDestroyRNNDataDescriptor(in_Desc));
  HCTR_LIB_THROW(cudnnDestroyRNNDataDescriptor(out_Desc));
  delete[] seqLengthArray;

  HCTR_LIB_THROW(hipdnnDestroyTensorDescriptor(hDesc));
  HCTR_LIB_THROW(hipdnnDestroyTensorDescriptor(cDesc));

  HCTR_LIB_THROW(hipdnnDestroyDropoutDescriptor(dropoutDesc));
  HCTR_LIB_THROW(hipdnnDestroyRNNDescriptor(rnnDesc));
}

template class GRULayer<float>;
// template class GRULayer<__half>;
}  // namespace HugeCTR
