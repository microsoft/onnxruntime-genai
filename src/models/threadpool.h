// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <vector>
#include <thread>

namespace Generators {

struct ThreadPool {
  ThreadPool(size_t num_threads);

  void Compute(const std::function<void(size_t)>& func);

 private:
  size_t num_threads_;
  std::vector<std::thread> threads_;
};

}  // namespace Generators
