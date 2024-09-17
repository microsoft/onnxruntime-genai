#include "captured_graph_pool.h"
#include "../generators.h"
#include "model.h"

namespace Generators {

void CapturedGraphInfoRecycler::operator()(CapturedGraphInfo* captured_graph_info) {
  if (captured_graph_info) {
    auto pool = captured_graph_info->pool_.lock();

    if (pool) {
      // Return the graph to the pool if available
      pool->AddCapturedGraph(CapturedGraphInfoPtr(captured_graph_info));
    } else {
      // If the pool has already been destroyed, we simply destroy the graph
      delete captured_graph_info;
    }
  }
}

CapturedGraphInfoPtr CapturedGraphPool::ReserveCapturedGraph(const Model& model, const GeneratorParams& params) const {
  if (!params.use_cuda_graph || (model.device_type_ != DeviceType::CUDA && model.device_type_ != DeviceType::DML)) {
    return nullptr;
  }

  // Multiple generators can reserve graphs in parallel, so we need to make it thread saf
  std::unique_lock lock(captured_graph_mutex_);

  auto key = std::make_unique<CapturedGraphKey>(params.max_batch_size, params.search.max_length, params.search.num_beams, params.extra_inputs);
  auto& captured_graphs = captured_graphs_map_[*key];

  // If no graphs are available, create a graph with a new ID
  if (captured_graphs.empty()) {
    auto new_captured_graph = CapturedGraphInfoPtr(new CapturedGraphInfo);

    // Create a unique annotation id
    new_captured_graph->index_ = current_graph_annotation_id_++;

    // We can unlock the mutex here since we don't access state that is subject to changes after this point
    lock.unlock();

    new_captured_graph->max_batch_size_ = params.max_batch_size;
    new_captured_graph->max_length_ = params.search.max_length;
    new_captured_graph->num_beams_ = params.search.num_beams;
    new_captured_graph->pool_ = shared_from_this();

    // Create the static buffer for the input ids
    size_t max_beam_batch_size = static_cast<size_t>(params.search.num_beams) * params.max_batch_size;
    new_captured_graph->sb_input_ids_ = std::make_unique<StaticBuffer>(allocator_device_, max_beam_batch_size);

#if USE_DML
    if (model.device_type_ == DeviceType::DML) {
      new_captured_graph->sb_input_ids_int32_ = std::make_unique<StaticBuffer>(allocator_device_, max_beam_batch_size);
    }
#endif

    // Create the static buffers for the cache
    int layer_count = config_->model.decoder.num_hidden_layers;
    new_captured_graph->sb_kv_caches_.reserve(layer_count * 2);

    for (int i = 0; i < layer_count * 2; ++i) {
      new_captured_graph->sb_kv_caches_.push_back(std::make_unique<StaticBuffer>(allocator_device_, max_beam_batch_size));
    }

    // Create the static buffer for the position ids, if needed
    if (session_info_->HasInput(config_->model.decoder.inputs.position_ids)) {
      new_captured_graph->sb_position_ids_ = std::make_unique<StaticBuffer>(allocator_device_, max_beam_batch_size);
    }

    // Create the static buffer for the attention mask, if needed
    if (session_info_->HasInput(config_->model.decoder.inputs.attention_mask)) {
      new_captured_graph->sb_attention_mask_ = std::make_unique<StaticBuffer>(allocator_device_, max_beam_batch_size);

#if USE_DML
      // DML currently needs an additional static buffer for the mask
      if (model.device_type_ == DeviceType::DML) {
        new_captured_graph->sb_attention_mask_next_ = std::make_unique<StaticBuffer>(allocator_device_, max_beam_batch_size);
      }
#endif
    }

    auto output_type = session_info_->GetOutputDataType(config_->model.decoder.outputs.logits);

    if (output_type == Ort::TypeToTensorType<float>) {
      new_captured_graph->sb_logits32_ = std::make_unique<StaticBuffer>(allocator_device_, max_beam_batch_size);
    }

    if (output_type == Ort::TypeToTensorType<Ort::Float16_t>) {
      new_captured_graph->sb_logits16_ = std::make_unique<StaticBuffer>(allocator_device_, max_beam_batch_size);
    }

    // Create the extra inputs
    for (const auto& extra_input : params.extra_inputs) {
      auto first_dim = extra_input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo()->GetShape()[0];
      new_captured_graph->sb_extra_inputs_[extra_input.name] = std::make_unique<StaticBuffer>(allocator_device_, first_dim);
    }

    // Create the input embeddings if needed
    if (!model.config_->model.embedding.filename.empty()) {
      new_captured_graph->sb_embeddings_ = std::make_unique<StaticBuffer>(allocator_device_, max_beam_batch_size);
    }

    new_captured_graph->key_ = std::move(key);

    return new_captured_graph;
  }

  // We found a graph, so take it from the pool and return it to the caller
  auto captured_graph = std::move(captured_graphs.front());
  captured_graphs.pop_front();
  return captured_graph;
}

void CapturedGraphPool::AddCapturedGraph(CapturedGraphInfoPtr&& captured_graph) const {
  std::unique_lock lock(captured_graph_mutex_);
  captured_graphs_map_[*captured_graph->key_].push_back(std::move(captured_graph));
}
}  // namespace Generators
