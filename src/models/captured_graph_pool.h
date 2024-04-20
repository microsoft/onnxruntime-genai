#pragma once

#include <list>
#include <mutex>
#include <unordered_map>
#include "static_buffer.h"

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
  CapturedGraphInfoPtr ReserveCapturedGraph(const Model& model, int max_batch_size) const;

 private:
  // Map from batch_size to a list of captured graphs
  mutable std::unordered_map<int, std::list<CapturedGraphInfoPtr>> captured_graphs_map_;
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
  int index_;
  std::unique_ptr<Generators::StaticBuffer> sb_input_ids_;
  std::vector<std::unique_ptr<Generators::StaticBuffer>> sb_kv_caches_;
  std::unique_ptr<Generators::StaticBuffer> sb_logits16_;
  std::unique_ptr<Generators::StaticBuffer> sb_logits32_;
  std::unique_ptr<Generators::StaticBuffer> sb_position_ids_;
  std::unique_ptr<Generators::StaticBuffer> sb_attention_mask_;

#if USE_DML
  std::unique_ptr<Generators::StaticBuffer> sb_attention_mask_next_;
  std::unique_ptr<Generators::StaticBuffer> sb_input_ids_int32_;
#endif

  // Generates a unique annotation ID across different captured graph objects. This is necessary because different
  // generators could be alive at the same time and run the same batch size but with different static buffers, so
  // they need to have different annotation IDs.
  int GenerateUniqueAnnotationID(int batch_size) {
    // Keep the upper half (minus 1 for the sign bit) of the bits for the unique ID, and keep the lower half for the batch
    // size. This should give us 32,767 values for the index and 65,535 values for the batch size, which is more than enough.
    int bit_shift = sizeof(int) * 8 / 2;
    return (index_ << bit_shift) | batch_size;
  }
};
}  // namespace Generators
