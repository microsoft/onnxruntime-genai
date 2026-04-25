#include "generators.h"

namespace Generators {

void softmax(std::span<float> values) {
  float max = *std::max_element(values.data(), values.data() + values.size());
  // Fused: compute exp and accumulate sum in a single pass
  float sum = 0.0f;
  for (auto& v : values) {
    v = std::exp(v - max);
    sum += v;
  }
  if (sum > 0.0f) {
    float inv_sum = 1.0f / sum;
    for (auto& v : values)
      v *= inv_sum;
  }
}

void log_softmax(std::span<float> values) {
  float max = *std::max_element(values.data(), values.data() + values.size());
  // Fused: scale and compute sum of exponentials in a single pass
  float sum = 0.0f;
  for (auto& v : values) {
    v -= max;
    sum += std::exp(v);
  }
  float log_sum = std::log(sum);
  for (auto& v : values)
    v -= log_sum;
}

}  // namespace Generators
