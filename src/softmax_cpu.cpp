#include "generators.h"

namespace Generators {

void softmax(std::span<float> values) {
  float max = *std::max_element(values.data(), values.data() + values.size());
  std::transform(values.begin(), values.end(), values.begin(), [max](float v) { return std::exp(v - max); });
  float sum = std::accumulate(values.begin(), values.end(), 0.0f);
  std::transform(values.begin(), values.end(), values.begin(), [sum](float v) { return v / sum; });
}

void log_softmax(std::span<float> values) {
  float max = *std::max_element(values.data(), values.data() + values.size());
  std::vector<float> scaled(values.begin(), values.end());
  std::transform(values.begin(), values.end(), scaled.begin(), [max](float v) { return std::exp(v - max); });

  float sum = std::accumulate(scaled.begin(), scaled.end(), 0.0f);
  float log_max = std::log(sum);
  std::transform(values.begin(), values.end(), values.begin(), [max, log_max](float v) { return v - max - log_max; });
}

}  // namespace Generators
