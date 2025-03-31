#pragma once

namespace Generators {

void SoftmaxWithMax(std::span<float> scores, float temperature, float max_score) {
  // Subtract max score and scale by temperature
  std::transform(scores.begin(), scores.end(), scores.begin(), [max_score, temperature](float score) { return std::exp((score - max_score) / temperature); });

  // Compute sum of exponentials
  float const exp_sum = std::accumulate(scores.begin(), scores.end(), 0.0f);

  // Divide each score by the sum of exponentials
  std::transform(scores.begin(), scores.end(), scores.begin(), [exp_sum](float score) { return score / exp_sum; });
}

void Softmax(std::span<float> scores, float temperature) {
  const float max_score = *std::max_element(scores.begin(), scores.end());

  SoftmaxWithMax(scores, temperature, max_score);
}

void LogSoftMax(std::span<float> scores, float temperature) {
  float const max_score = *std::max_element(scores.begin(), scores.end());

  // Subtract max score and scale by temperature
  std::transform(scores.begin(), scores.end(), scores.begin(), [max_score, temperature](float score) { return (score - max_score) / temperature; });

  // Compute sum of exponentials
  float const exp_sum = std::accumulate(scores.begin(), scores.end(), 0.0f, [](float a, float b) { return a + std::exp(b); });

  // Subtract log of sum of exponentials from each score
  std::transform(scores.begin(), scores.end(), scores.begin(), [exp_sum](float score) { return score - std::log(exp_sum); });
}

}  // namespace Generators
