// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <mutex>
#include <unordered_map>

namespace Generators {
namespace cuda {

namespace {  // Anonymous namespace to hide implementation details

// A key to uniquely identify a benchmark configuration.
struct TopkBenchmarkCacheKey {
  int device_id;
  int batch_size;
  int vocab_size;

  bool operator==(const TopkBenchmarkCacheKey& other) const {
    return device_id == other.device_id &&
           batch_size == other.batch_size &&
           vocab_size == other.vocab_size;
  }
};

// Hasher for the custom key struct, required for std::unordered_map.
struct TopkBenchmarkCacheKeyHash {
  std::size_t operator()(const TopkBenchmarkCacheKey& k) const {
    // A simple hash combination function.
    size_t h1 = std::hash<int>{}(k.device_id);
    size_t h2 = std::hash<int>{}(k.batch_size);
    size_t h3 = std::hash<int>{}(k.vocab_size);
    return h1 ^ (h2 << 1) ^ (h3 << 2);
  }
};

using TopkAlgoCachePtr = std::shared_ptr<std::array<TopkAlgo, kMaxBenchmarkK + 1>>;

// The global cache manager.
class TopkBenchmarkCacheManager {
 public:
  // Gets (or creates) the algorithm cache for a specific configuration.
  TopkAlgoCachePtr GetOrCreateCache(int device_id, int batch_size, int vocab_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    TopkBenchmarkCacheKey key{device_id, batch_size, vocab_size};

    auto it = cache_.find(key);
    if (it != cache_.end()) {
      return it->second;
    }

    // Cache miss: create a new cache entry, initialize it with UNKNOWN, and return it.
    auto new_cache_array = std::make_shared<std::array<TopkAlgo, kMaxBenchmarkK + 1>>();
    new_cache_array->fill(TopkAlgo::UNKNOWN);

    cache_[key] = new_cache_array;
    return new_cache_array;
  }

 private:
  std::mutex mutex_;
  std::unordered_map<TopkBenchmarkCacheKey, TopkAlgoCachePtr, TopkBenchmarkCacheKeyHash> cache_;
};

// The single static instance of our cache.
static TopkBenchmarkCacheManager g_topk_benchmark_cache;

}  // namespace

// Public-facing function to access the cache.
std::shared_ptr<std::array<TopkAlgo, kMaxBenchmarkK + 1>> GetTopkBenchmarkCache(int device_id, int batch_size, int vocab_size) {
  return g_topk_benchmark_cache.GetOrCreateCache(device_id, batch_size, vocab_size);
}

}  // namespace cuda
}  // namespace Generators
