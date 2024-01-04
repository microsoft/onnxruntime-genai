#include "generators.h"
namespace Generators {

void top_k_indices(std::span<int32_t> top_k, std::span<const float> inputs) {
  int32_t k = static_cast<int32_t>(top_k.size());
  assert(k <= inputs.size());  // Use a smaller top_k span if k is larger than inputs

  // Min heap to store pairs of (element, index)
  std::priority_queue<std::pair<float, int32_t>, std::vector<std::pair<float, int32_t>>, std::greater<>> pq;

  // Add first k elements into the heap
  for (int32_t i = 0; i < k; i++) {
    pq.push(std::make_pair(inputs[i], i));
  }

  // For the rest of the elements we already have k, so remove the smallest on each iteration
  for (int32_t i = k; i < inputs.size(); i++) {
    // Entry is smaller than the smallest, so don't bother
    if (inputs[i] <= pq.top().first)
      continue;

    pq.pop();
    pq.push(std::make_pair(inputs[i], i));
  }

  for (int i = 0; i < k; i++) {
    top_k[k - i - 1] = pq.top().second;
    pq.pop();
  }
}

}  // namespace Generators
