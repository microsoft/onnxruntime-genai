#pragma once

namespace Generators {

void SoftmaxWithMax(std::span<float> scores, float temperature, float max_score) {
  // Fused: compute exp and accumulate sum in a single pass
  float inv_temp = 1.0f / temperature;
  float exp_sum = 0.0f;
  for (auto& score : scores) {
    score = std::exp((score - max_score) * inv_temp);
    exp_sum += score;
  }

  // Normalize
  if (exp_sum > 0.0f) {
    float inv_sum = 1.0f / exp_sum;
    for (auto& score : scores)
      score *= inv_sum;
  }
}

void Softmax(std::span<float> scores, float temperature) {
  const float max_score = *std::max_element(scores.begin(), scores.end());

  SoftmaxWithMax(scores, temperature, max_score);
}

void LogSoftMax(std::span<float> scores, float temperature) {
  float const max_score = *std::max_element(scores.begin(), scores.end());

  // Fused: scale and compute sum of exponentials in a single pass
  float inv_temp = 1.0f / temperature;
  float exp_sum = 0.0f;
  for (auto& score : scores) {
    score = (score - max_score) * inv_temp;
    exp_sum += std::exp(score);
  }

  // Subtract log of sum of exponentials
  float const log_sum_exp = std::log(exp_sum);
  for (auto& score : scores)
    score -= log_sum_exp;
}

}  // namespace Generators
