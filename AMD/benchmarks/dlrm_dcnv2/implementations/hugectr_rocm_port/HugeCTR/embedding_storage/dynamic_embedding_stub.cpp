// HugeCTR ROCm port: stub for embedding::DynamicEmbeddingTable.
// The full .cu uses NVIDIA's cuCollections (cuco::dynamic_map) which we did not
// vendor. HugeCTR's embedding planner instantiates this table for some sharding
// strategies; DLRM-DCNv2 with the default planner uses HierarchicalKV-backed
// RaggedStaticEmbedding instead, so the runtime path doesn't hit these stubs
// in practice. They satisfy the dynamic linker; calls will throw if reached.

#include <embedding_storage/dynamic_embedding.hpp>

#include <stdexcept>

namespace embedding {

#define HCTR_DYN_STUB() \
  throw std::runtime_error( \
      "HugeCTR ROCm port: embedding::DynamicEmbeddingTable is stubbed " \
      "(cuCollections not vendored); use the HierarchicalKV-backed " \
      "embedding storage path instead.")

DynamicEmbeddingTable::DynamicEmbeddingTable(
    const HugeCTR::GPUResource & /*gpu_resource*/,
    std::shared_ptr<CoreResourceManager> core,
    const std::vector<EmbeddingTableParam> & /*table_params*/,
    const EmbeddingCollectionParam & /*ebc_param*/, size_t /*grouped_table_id*/,
    const HugeCTR::OptParams &opt_param)
    : core_(std::move(core)),
      key_type_(),
      table_(nullptr),
      opt_param_(opt_param),
      table_opt_states_(nullptr),
      opt_state_view_(),
      weight_view_() {}

std::vector<size_t> DynamicEmbeddingTable::remap_id_space(const core23::Tensor &, hipStream_t) {
  HCTR_DYN_STUB();
  return {};
}

std::vector<size_t> DynamicEmbeddingTable::remap_id_space(const std::vector<int> &) {
  HCTR_DYN_STUB();
  return {};
}

void DynamicEmbeddingTable::lookup(const core23::Tensor &, size_t, const core23::Tensor &,
                                    size_t, const core23::Tensor &, core23::Tensor &) {
  HCTR_DYN_STUB();
}

void DynamicEmbeddingTable::update(const core23::Tensor &, const core23::Tensor &,
                                    const core23::Tensor &, const core23::Tensor &,
                                    const core23::Tensor &) {
  HCTR_DYN_STUB();
}

void DynamicEmbeddingTable::assign(const core23::Tensor &, size_t, const core23::Tensor &,
                                    size_t, const core23::Tensor &, core23::Tensor &,
                                    const core23::Tensor &) {
  HCTR_DYN_STUB();
}

void DynamicEmbeddingTable::load(core23::Tensor &, core23::Tensor &, core23::Tensor &,
                                  core23::Tensor &, core23::Tensor &) {
  HCTR_DYN_STUB();
}

void DynamicEmbeddingTable::dump(core23::Tensor *, core23::Tensor *, core23::Tensor *,
                                  core23::Tensor *, core23::Tensor *) {
  HCTR_DYN_STUB();
}

void DynamicEmbeddingTable::dump_by_id(core23::Tensor *, core23::Tensor *, int) {
  HCTR_DYN_STUB();
}

void DynamicEmbeddingTable::load_by_id(core23::Tensor *, core23::Tensor *, int) {
  HCTR_DYN_STUB();
}

size_t DynamicEmbeddingTable::size() const { return 0; }
size_t DynamicEmbeddingTable::capacity() const { return 0; }
size_t DynamicEmbeddingTable::key_num() const { return 0; }
std::vector<size_t> DynamicEmbeddingTable::size_per_table() const { return {}; }
std::vector<size_t> DynamicEmbeddingTable::capacity_per_table() const { return {}; }
std::vector<size_t> DynamicEmbeddingTable::key_num_per_table() const { return {}; }
std::vector<int> DynamicEmbeddingTable::table_ids() const { return {}; }
std::vector<int> DynamicEmbeddingTable::table_evsize() const { return {}; }
void DynamicEmbeddingTable::clear() { /* no-op */ }
void DynamicEmbeddingTable::evict(const core23::Tensor &, size_t, const core23::Tensor &,
                                   size_t, const core23::Tensor &) {
  HCTR_DYN_STUB();
}

}  // namespace embedding
