// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <mutex>
#include <unordered_map>
#include "cuda_topk.h"

namespace Generators {
namespace cuda {

namespace {  // Anonymous namespace to hide implementation details

// A key to uniquely identify a benchmark configuration.
struct TopkBenchmarkCacheKey {
  int device_id;
  int batch_size;
  int vocab_size;
  int k;

  bool operator==(const TopkBenchmarkCacheKey& other) const {
    return device_id == other.device_id &&
           batch_size == other.batch_size &&
           vocab_size == other.vocab_size &&
           k == other.k;
  }
};

// Hasher for the custom key struct, required for std::unordered_map.
struct TopkBenchmarkCacheKeyHash {
  std::size_t operator()(const TopkBenchmarkCacheKey& key) const {
    // A simple hash combination function.
    size_t h1 = std::hash<int>{}(key.device_id);
    size_t h2 = std::hash<int>{}(key.batch_size);
    size_t h3 = std::hash<int>{}(key.vocab_size);
    size_t h4 = std::hash<int>{}(key.k);
    return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
  }
};

// The global cache manager.
class TopkBenchmarkCacheManager {
 public:
  // Gets the cached algorithm for a specific configuration.
  TopkAlgo Get(int device_id, int batch_size, int vocab_size, int k) {
    std::lock_guard<std::mutex> lock(mutex_);
    TopkBenchmarkCacheKey key{device_id, batch_size, vocab_size, k};

    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }

    return TopkAlgo::UNKNOWN;
  }

  // Sets the algorithm for a specific configuration.
  void Set(int device_id, int batch_size, int vocab_size, int k, TopkAlgo algo) {
    std::lock_guard<std::mutex> lock(mutex_);
    TopkBenchmarkCacheKey key{device_id, batch_size, vocab_size, k};
    cache_[key] = algo;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<TopkBenchmarkCacheKey, TopkAlgo, TopkBenchmarkCacheKeyHash> cache_;
};

TopkBenchmarkCacheManager& GetCache() {
  static TopkBenchmarkCacheManager g_topk_benchmark_cache;
  return g_topk_benchmark_cache;
}

}  // namespace

// Public-facing functions to access the cache.
inline TopkAlgo GetTopkBenchmarkCache(int device_id, int batch_size, int vocab_size, int k) {
  return GetCache().Get(device_id, batch_size, vocab_size, k);
}

inline void SetTopkBenchmarkCache(int device_id, int batch_size, int vocab_size, int k, TopkAlgo algo) {
  GetCache().Set(device_id, batch_size, vocab_size, k, algo);
}

}  // namespace cuda
}  // namespace Generators
