#pragma once

#include <vector>
#include <list>
#include <mutex>
#include <unordered_map>
#include "static_buffer.h"

// From boost http://www.boost.org/doc/libs/1_35_0/doc/html/hash/combine.html
template <class T>
inline void hash_combine(size_t& seed, T const& v) {
  seed ^= std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

struct tuple_hash {
  size_t operator()(const std::tuple<int, int, int>& p) const {
    size_t seed = 0;
    auto [first, second, third] = p;
    hash_combine(seed, first);
    hash_combine(seed, second);
    hash_combine(seed, third);
    return seed;
  }
};

namespace Generators {
struct CapturedGraphInfo;
struct Config;
struct SessionInfo;
struct Model;
struct GeneratorParams;

struct CapturedGraphInfoRecycler {
  void operator()(CapturedGraphInfo* captured_graph_info);
};

using CapturedGraphInfoPtr = std::unique_ptr<CapturedGraphInfo, CapturedGraphInfoRecycler>;

class CapturedGraphPool : public std::enable_shared_from_this<CapturedGraphPool> {
 public:
  CapturedGraphPool(const Config* config, const SessionInfo* session_info, OrtAllocator* allocator_device)
      : config_(config),
        session_info_(session_info),
        allocator_device_(allocator_device){};

  void AddCapturedGraph(CapturedGraphInfoPtr&& captured_graph) const;
  CapturedGraphInfoPtr ReserveCapturedGraph(const Model& model, const GeneratorParams& params) const;

 private:
  // Map from batch_size/max_length to a list of captured graphs
  mutable std::unordered_map<std::tuple<int, int, int>, std::list<CapturedGraphInfoPtr>, tuple_hash> captured_graphs_map_;
  mutable std::mutex captured_graph_mutex_;

  // 0 is reserved for internal usage in cuda graphs, so we start from 1
  mutable int current_graph_annotation_id_ = 1;
  const Config* config_;
  const SessionInfo* session_info_;
  OrtAllocator* allocator_device_;
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
