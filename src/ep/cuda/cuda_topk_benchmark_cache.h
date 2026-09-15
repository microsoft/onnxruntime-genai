// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <mutex>
#include <unordered_map>
#include "cuda_topk.h"

namespace Generators {
namespace cuda {

// This file implements a singleton-based, thread-safe, persistent cache for Top-K benchmark results.
// When the online benchmark determines the fastest algorithm for a specific problem size
// (device, batch, vocab, k), the result is stored here. Subsequent calls with the same
// parameters can retrieve the best algorithm directly without re-benchmarking.

namespace {  // Anonymous namespace to hide implementation details

// A key to uniquely identify a benchmark configuration in the cache map.
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

// Custom hasher for the key struct, required for use with `std::unordered_map`.
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

// The global cache manager class, implemented as a singleton.
class TopkBenchmarkCacheManager {
 public:
  // Gets the cached algorithm for a specific configuration.
  // Returns TopkAlgo::UNKNOWN if not found.
  TopkAlgo Get(int device_id, int batch_size, int vocab_size, int k) {
    std::lock_guard<std::mutex> lock(mutex_);
    TopkBenchmarkCacheKey key{device_id, batch_size, vocab_size, k};

    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }

    return TopkAlgo::UNKNOWN;
  }

  // Sets (or updates) the algorithm for a specific configuration.
  void Set(int device_id, int batch_size, int vocab_size, int k, TopkAlgo algo) {
    std::lock_guard<std::mutex> lock(mutex_);
    TopkBenchmarkCacheKey key{device_id, batch_size, vocab_size, k};
    cache_[key] = algo;
  }

 private:
  std::mutex mutex_;  // Protects cache access from multiple host threads.
  std::unordered_map<TopkBenchmarkCacheKey, TopkAlgo, TopkBenchmarkCacheKeyHash> cache_;
};

// Provides access to the singleton instance of the cache manager.
TopkBenchmarkCacheManager& GetCache() {
  static TopkBenchmarkCacheManager g_topk_benchmark_cache;
  return g_topk_benchmark_cache;
}

}  // namespace

// Public-facing functions to access the global cache.
inline TopkAlgo GetTopkBenchmarkCache(int device_id, int batch_size, int vocab_size, int k) {
  return GetCache().Get(device_id, batch_size, vocab_size, k);
}

inline void SetTopkBenchmarkCache(int device_id, int batch_size, int vocab_size, int k, TopkAlgo algo) {
  GetCache().Set(device_id, batch_size, vocab_size, k, algo);
}

}  // namespace cuda
}  // namespace Generators
