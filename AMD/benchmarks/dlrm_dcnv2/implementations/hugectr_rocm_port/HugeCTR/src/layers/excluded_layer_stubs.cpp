// HugeCTR ROCm port: stubs for excluded layers (GRU, MultiHeadAttention,
// BatchNorm, Dropout, Interaction). The actual .cu sources for these layers
// were excluded because they depend on cuDNN/MIOpen RNN/Dropout/BN APIs (not
// on AMD), CUDA WMMA tensor cores (interaction), or hipBLASLt epilogues we
// don't have. Their constructors and fprop/bprop methods are still referenced
// from add_dense_layer.cpp / model.cpp etc., so we provide stubs that throw
// at runtime if any DLRM-DCNv2 path actually instantiates them.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdexcept>

#include <layers/multi_head_attention_layer.hpp>
#include <layers/gru_layer.hpp>
#include <layers/dropout_layer.hpp>
#include <layers/batch_norm_layer.hpp>
#include <layers/interaction_layer.hpp>

namespace HugeCTR {

namespace {
[[noreturn]] inline void throw_layer_unsupported(const char* name) {
  throw std::runtime_error(
      std::string("HugeCTR ROCm port: layer '") + name +
      "' is stubbed (excluded from the AMD/ROCm build). DLRM-DCNv2 baseline "
      "doesn't require this layer; if your model needs it, port the relevant "
      ".cu source against rocWMMA/MIOpen/hipBLASLt as appropriate.");
}
}  // namespace

// ---- MultiHeadAttentionLayer ----------------------------------------------
template <typename T>
MultiHeadAttentionLayer<T>::MultiHeadAttentionLayer(
    const std::vector<core23::Tensor>& /*input_tensors*/,
    std::vector<core23::Tensor>& /*output_tensors*/, int /*num_attention_heads*/,
    bool transpose_b, const std::shared_ptr<GPUResource>& gpu_resource,
    bool use_mixed_precision, bool enable_tf32_compute)
    : Layer({}, {}, gpu_resource),
      enable_tf32_compute_(enable_tf32_compute),
      use_mixed_precision_(use_mixed_precision),
      num_(0),
      dims_(0),
      transpose_b_(transpose_b),
      num_head_(0) {}

template <typename T>
void MultiHeadAttentionLayer<T>::fprop(bool /*is_train*/) {
  throw_layer_unsupported("MultiHeadAttentionLayer");
}

template <typename T>
void MultiHeadAttentionLayer<T>::bprop() {
  throw_layer_unsupported("MultiHeadAttentionLayer");
}

template class MultiHeadAttentionLayer<float>;
template class MultiHeadAttentionLayer<__half>;

// ---- InteractionLayer ------------------------------------------------------
template <typename T>
InteractionLayer<T>::InteractionLayer(const core23::Tensor& /*input_bottom_mlp_tensor*/,
                                       const core23::Tensor& /*input_embeddings*/,
                                       core23::Tensor& /*out_tensor*/,
                                       const std::shared_ptr<GPUResource>& gpu_resource,
                                       bool use_mixed_precision, bool enable_tf32_compute)
    : Layer({}, {}, gpu_resource),
      enable_tf32_compute_(enable_tf32_compute),
      use_mixed_precision_(use_mixed_precision),
      separate_Y_and_dY_(false) {}

template <typename T>
InteractionLayer<T>::InteractionLayer(const core23::Tensor& /*input_bottom_mlp_tensor*/,
                                       const core23::Tensor& /*input_embeddings*/,
                                       core23::Tensor& /*out_tensor*/, core23::Tensor& /*grad_tensor*/,
                                       const std::shared_ptr<GPUResource>& gpu_resource,
                                       bool use_mixed_precision, bool enable_tf32_compute)
    : Layer({}, {}, gpu_resource),
      enable_tf32_compute_(enable_tf32_compute),
      use_mixed_precision_(use_mixed_precision),
      separate_Y_and_dY_(true) {}

template <typename T>
InteractionLayer<T>::~InteractionLayer() = default;

template <typename T>
void InteractionLayer<T>::init(const core23::Tensor&, const core23::Tensor&, core23::Tensor&,
                                core23::Tensor&, const std::shared_ptr<GPUResource>&) {
  throw_layer_unsupported("InteractionLayer");
}

template <typename T>
void InteractionLayer<T>::fprop_generic(bool /*is_train*/) {
  throw_layer_unsupported("InteractionLayer");
}

template <typename T>
void InteractionLayer<T>::fprop(bool /*is_train*/) {
  throw_layer_unsupported("InteractionLayer");
}

template <typename T>
void InteractionLayer<T>::bprop_generic() {
  throw_layer_unsupported("InteractionLayer");
}

template <typename T>
void InteractionLayer<T>::bprop() {
  throw_layer_unsupported("InteractionLayer");
}

template class InteractionLayer<float>;
template class InteractionLayer<__half>;

// ---- GRULayer --------------------------------------------------------------
template <typename T>
GRULayer<T>::GRULayer(const core23::Tensor& /*in_tensor*/, const core23::Tensor& /*out_tensor*/,
                      int64_t /*hiddenSize*/, int64_t /*batch_size*/, int64_t /*SeqLength*/,
                      int64_t /*embedding_vec_size*/,
                      const std::shared_ptr<GPUResource>& gpu_resource,
                      std::vector<Initializer_t> initializer_types)
    : TrainableLayer<T>({}, {}, gpu_resource, initializer_types) {}

template <typename T>
void GRULayer<T>::fprop(bool /*is_train*/) {
  throw_layer_unsupported("GRULayer");
}

template <typename T>
void GRULayer<T>::bprop() {
  throw_layer_unsupported("GRULayer");
}

template class GRULayer<float>;

// ---- DropoutLayer ----------------------------------------------------------
template <typename T>
DropoutLayer<T>::DropoutLayer(const core23::Tensor& /*in_tensor*/,
                               const core23::Tensor& /*out_tensor*/, float rate,
                               const std::shared_ptr<GPUResource>& gpu_resource)
    : Layer({}, {}, gpu_resource), rate_(rate), scale_(1.0f / (1.0f - rate)),
      cudnn_status_(nullptr), reserveSpaceSizeInBytes_(0) {}

template <typename T>
DropoutLayer<T>::~DropoutLayer() = default;

template <typename T>
void DropoutLayer<T>::fprop(bool /*is_train*/) { throw_layer_unsupported("DropoutLayer"); }

template <typename T>
void DropoutLayer<T>::bprop() { throw_layer_unsupported("DropoutLayer"); }

template class DropoutLayer<float>;
template class DropoutLayer<__half>;

// ---- BatchNormLayer --------------------------------------------------------
template <typename T>
BatchNormLayer<T>::BatchNormLayer(const core23::Tensor& /*in_tensor*/,
                                    const core23::Tensor& /*out_tensor*/,
                                    const Params& params,
                                    const std::shared_ptr<GPUResource>& gpu_resource,
                                    std::vector<Initializer_t> initializer_types)
    : TrainableLayer<T, true>({}, {}, gpu_resource, initializer_types), params_(params) {}

template <typename T>
BatchNormLayer<T>::~BatchNormLayer() = default;

template <typename T>
void BatchNormLayer<T>::initialize() { /* no-op stub */ }

template <typename T>
void BatchNormLayer<T>::fprop(bool /*is_train*/) { throw_layer_unsupported("BatchNormLayer"); }

template <typename T>
void BatchNormLayer<T>::bprop() { throw_layer_unsupported("BatchNormLayer"); }

template <typename T>
std::string BatchNormLayer<T>::get_no_trained_params_in_string() { return "{}"; }

template <typename T>
std::vector<core23::Tensor> BatchNormLayer<T>::get_non_trainable_params_as_tensors() { return {}; }

template <typename T>
std::unique_ptr<DataSimulator> BatchNormLayer<T>::get_default_initializer(const int /*index*/) {
  return nullptr;
}

template class BatchNormLayer<float>;
template class BatchNormLayer<__half>;

}  // namespace HugeCTR
