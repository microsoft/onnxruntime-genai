
#if 0
#include "generators.h"
#include <algorithm>
#include <random>

int top_k_sampling(std::span<float> scores, float temperature, int k) {
  std::vector<std::pair<float, int>> top_k;
  std::default_random_engine generator;

  for (int i = 0; i < scores.size(); ++i) {
    scores[i] = std::exp(scores[i] / temperature);

    if (top_k.size() < k || scores[i] > top_k.back().first) {
      if (top_k.size() == k) {
        top_k.pop_back();
      }

      auto it = std::upper_bound(top_k.begin(), top_k.end(), std::make_pair(scores[i], i),
                                 { return a.first < b.first; });
      top_k.insert(it, {scores[i], i});
    }
  }

  // Create a distribution over the top K tokens
  std::vector<float> top_k_probs;
  for (auto& pair : top_k) {
    top_k_probs.push_back(pair.first);
  }

  // Sample a token from the distribution
  std::discrete_distribution<int> dist(top_k_probs.begin(), top_k_probs.end());
  int sampled_index = dist(generator);

  return top_k[sampled_index].second;
}

void MultinomialComputeShared(const Tensor& X,
                              const int64_t batch_size,
                              const int64_t num_classes,
                              const int64_t num_samples,
                              std::default_random_engine& generator,
                              Tensor& Y) {

  // implementation copied from Tensorflow with some changes such as using the std::uniform_real_distribution
  // instead of the Philox RNG.
  Eigen::array<int64_t, 2> X_dims = {{batch_size, num_classes}};
  ConstMatrix<float> logits = ConstMatrix<float>(X.Data<float>(), X_dims);

  Eigen::array<int64_t, 2> Y_dims = {{batch_size, num_samples}};
  Matrix<OutputType> output = Matrix<OutputType>(Y.MutableData<OutputType>(), Y_dims);

  // BEGIN create temporary tensor
  auto cdf_data = static_cast<double*>(alloc->Alloc(SafeInt<size_t>(sizeof(double)) * num_classes));
  BufferUniquePtr cdf_buffer(cdf_data, BufferDeleter(std::move(alloc)));
  Eigen::array<int64_t, 1> cdf_dims = {{num_classes}};
  auto cdf = EigenVector<double>(cdf_data, cdf_dims);
  // END create temporary tensor

  std::uniform_real_distribution<double> dist(0.0, 1.0);  // TODO: should this be initialized per batch?

  for (int64_t b = 0; b < batch_size; ++b) {
    const float* logits_row = &(logits(b, 0));
    // Takes an along-class maximum (for numerical stability).
    float maxx = std::numeric_limits<float>::lowest();
    for (int64_t j = 0; j < num_classes; ++j) {
      if (Eigen::numext::isfinite(logits_row[j])) {
        maxx = std::max(maxx, logits_row[j]);
      }
    }
    const auto max_logit = static_cast<double>(maxx);

    // Precompute cumulative probability distribution across classes.
    // Note: This isn't normalized.
    cdf = (logits.chip<0>(b).cast<double>() - max_logit).exp();
    double running_total = 0;
    for (int64_t j = 0; j < num_classes; ++j) {
      if (Eigen::numext::isfinite(logits_row[j])) {
        running_total += cdf(j);
      }
      cdf(j) = running_total;
    }
    // Generate each sample.
    const double* cdf_begin = cdf.data();
    const double* cdf_end = cdf.data() + num_classes;
    for (int64_t j = 0; j < num_samples; ++j) {
      const double to_find = dist(generator) * running_total;
      auto found_iter = std::upper_bound(cdf_begin, cdf_end, to_find);
      output(b, j) = static_cast<OutputType>(std::distance(cdf_begin, found_iter));
    }
  }

  return Status::OK();
}

#endif
