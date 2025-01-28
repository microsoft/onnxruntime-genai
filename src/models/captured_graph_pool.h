#pragma once

#include <vector>
#include <list>
#include <mutex>
#include <unordered_map>
#include "static_buffer.h"
#include "../generators.h"

// From boost http://www.boost.org/doc/libs/1_35_0/doc/html/hash/combine.html
template <class T>
inline void hash_combine(size_t& seed, T const& v) {
  seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct InputKey {
  InputKey(std::string name, ONNXTensorElementDataType tensor_type, std::vector<int64_t> tensor_shape)
      : name_(std::move(name)),
        tensor_type_(tensor_type),
        tensor_shape_(std::move(tensor_shape)) {}

  bool operator==(const InputKey& other) const {
    return name_ == other.name_ && tensor_type_ == other.tensor_type_ && tensor_shape_ == other.tensor_shape_;
  }

  std::string name_;
  ONNXTensorElementDataType tensor_type_;
  std::vector<int64_t> tensor_shape_;
};

struct CapturedGraphKey {
  CapturedGraphKey(int max_batch_size, int max_length, int num_beams, const std::vector<Generators::GeneratorParams::Input>& extra_inputs)
      : max_batch_size_(max_batch_size),
        max_length_(max_length),
        num_beams_(num_beams) {
    extra_inputs_.reserve(extra_inputs.size());

    for (const auto& extra_input : extra_inputs) {
      auto type_and_shape_info = extra_input.tensor->ort_tensor_->GetTensorTypeAndShapeInfo();
      extra_inputs_.emplace_back(extra_input.name, type_and_shape_info->GetElementType(), type_and_shape_info->GetShape());
    }

    // We want to compare sorted inputs since the order doesn't matter
    std::sort(extra_inputs_.begin(), extra_inputs_.end(), [](const InputKey& a, const InputKey& b) {
      return a.name_.compare(b.name_) < 0;
    });
  }

  bool operator==(const CapturedGraphKey& other) const {
    return max_batch_size_ == other.max_batch_size_ &&
           max_length_ == other.max_length_ &&
           num_beams_ == other.num_beams_ &&
           extra_inputs_ == other.extra_inputs_;
  }

  int max_batch_size_;
  int max_length_;
  int num_beams_;
  std::vector<InputKey> extra_inputs_;
};

template <typename T>
struct std::hash<std::vector<T>> {
  std::size_t operator()(const std::vector<T>& container) const noexcept {
    size_t seed = 0;

    for (const T& element : container) {
      hash_combine(seed, element);
    }

    return seed;
  }
};

template <>
struct std::hash<InputKey> {
  std::size_t operator()(const InputKey& extra_input) const noexcept {
    size_t seed = 0;
    hash_combine(seed, extra_input.name_);
    hash_combine(seed, extra_input.tensor_type_);
    hash_combine(seed, extra_input.tensor_shape_);
    return seed;
  }
};

template <>
struct std::hash<CapturedGraphKey> {
  std::size_t operator()(const CapturedGraphKey& captured_graph_key) const noexcept {
    size_t seed = 0;
    hash_combine(seed, captured_graph_key.max_batch_size_);
    hash_combine(seed, captured_graph_key.max_length_);
    hash_combine(seed, captured_graph_key.num_beams_);
    hash_combine(seed, captured_graph_key.extra_inputs_);
    return seed;
  }
};

namespace Generators {
struct CapturedGraphInfo;
struct Config;
struct SessionInfo;
struct Model;

struct CapturedGraphInfoRecycler {
  void operator()(CapturedGraphInfo* captured_graph_info);
};

using CapturedGraphInfoPtr = std::unique_ptr<CapturedGraphInfo, CapturedGraphInfoRecycler>;

class CapturedGraphPool : public std::enable_shared_from_this<CapturedGraphPool> {
 public:
  CapturedGraphPool(const Config* config, const SessionInfo* session_info, Ort::Allocator* allocator_device)
      : config_(config),
        session_info_(session_info),
        allocator_device_(allocator_device){};

  void AddCapturedGraph(CapturedGraphInfoPtr&& captured_graph) const;
  CapturedGraphInfoPtr ReserveCapturedGraph(const Model& model, const GeneratorParams& params) const;

 private:
  // Map from batch_size/max_length to a list of captured graphs
  mutable std::unordered_map<CapturedGraphKey, std::list<CapturedGraphInfoPtr>> captured_graphs_map_;
  mutable std::mutex captured_graph_mutex_;

  // 0 is reserved for internal usage in cuda graphs, so we start from 1
  mutable int current_graph_annotation_id_ = 1;
  const Config* config_;
  const SessionInfo* session_info_;
  Ort::Allocator* allocator_device_;
};

struct CapturedGraphInfo {
  std::weak_ptr<const CapturedGraphPool> pool_;
  int max_batch_size_;
  int max_length_;
  int num_beams_;
  int index_;
  std::unique_ptr<Generators::StaticBuffer> sb_input_ids_;
  std::vector<std::unique_ptr<Generators::StaticBuffer>> sb_kv_caches_;
  std::unique_ptr<Generators::StaticBuffer> sb_logits16_;
  std::unique_ptr<Generators::StaticBuffer> sb_logits32_;
  std::unique_ptr<Generators::StaticBuffer> sb_position_ids_;
  std::unique_ptr<Generators::StaticBuffer> sb_attention_mask_;
  std::unordered_map<std::string, std::unique_ptr<Generators::StaticBuffer>> sb_extra_inputs_;
  std::unique_ptr<Generators::StaticBuffer> sb_embeddings_;
  std::unique_ptr<CapturedGraphKey> key_;

  // Generates a unique annotation ID across different captured graph objects. This is necessary because different
  // generators could be alive at the same time and run the same batch size but with different static buffers, so
  // they need to have different annotation IDs.
  int GenerateUniqueAnnotationID(int batch_size) const {
    // Keep the upper half (minus 1 for the sign bit) of the bits for the unique ID, and keep the lower half for the batch
    // size. This should give us 32,767 values for the index and 65,535 values for the batch size, which is more than enough.
    int bit_shift = sizeof(int) * 8 / 2;
    return (index_ << bit_shift) | batch_size;
  }
};
}  // namespace Generators
