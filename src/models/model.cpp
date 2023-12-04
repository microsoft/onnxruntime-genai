#include "../generators.h"
#include "../search.h"
#if USE_CUDA
#include "../search_cuda.h"
#endif
#include "model.h"
#include "gpt_common.h"
#include "gpt_cpu.h"
#include "gpt_cuda.h"
#include "llama_cpu.h"
#include "llama_cuda.h"

namespace Generators {

Model::Model(OrtEnv& ort_env, const char* config_path, const ProviderOptions* provider_options) : config_{config_path} {
  auto session_options = OrtSessionOptions::Create();

  if (provider_options) {
#if USE_CUDA
    if (auto* options = std::get_if<OrtCUDAProviderOptions>(provider_options)) {
      cuda_stream_ = reinterpret_cast<cudaStream_t>(options->user_compute_stream);
      session_options->AppendExecutionProvider_CUDA(*options);
      device_type_ = DeviceType::CUDA;
    }
#endif
  }

  if (config_.model_type == "gpt2") {
    impl_ = std::make_unique<Gpt_Model>(ort_env, config_, *session_options);
  } else if (config_.model_type == "llama") {
    impl_llama_ = std::make_unique<Llama_Model>(ort_env, config_, *session_options);
  } else
    throw std::runtime_error("Unsupported model_type in config.json: " + config_.model_type);
}

Model::~Model() = default;

std::unique_ptr<State> Model::CreateState(std::span<int32_t> sequence_lengths, const SearchParams& params) {
  if (impl_llama_) {
    if (device_type_ == DeviceType::CUDA)
      return std::make_unique<Llama_Cuda>(*impl_llama_, sequence_lengths, params);
    else
      return std::make_unique<Llama_State>(*impl_llama_, sequence_lengths, params);

  } else {
    if (device_type_ == DeviceType::CUDA)
      return std::make_unique<Gpt_Cuda>(*impl_, sequence_lengths, params);
    else
      return std::make_unique<Gpt_State>(*impl_, sequence_lengths, params);
  }
}

#if USE_CUDA
std::vector<int32_t> CudaSpanToVector(std::span<int32_t> span) {
  std::vector<int32_t> v(span.size());
  cudaMemcpy(v.data(), span.data(), span.size_bytes(), cudaMemcpyDeviceToHost);
  return v;
}
#endif

std::vector<int32_t> SpanToVector(std::span<int32_t> span) {
  return std::vector<int32_t>(span.begin(), span.end());
}

std::vector<int32_t> Model::Generate(const SearchParams& params) {
#if USE_CUDA
  if (device_type_ == DeviceType::CUDA) {
    if (params.num_beams == 1) {
      Generators::GreedySearch_Cuda search{params};
      auto state = CreateState(search.sequence_lengths_, params);

      while (!search.IsDone()) {
        search.SetLogits(state->Run(search.GetSequenceLength(), search.GetNextTokens()));

        if (config_.top_p < 1.0f) {
          search.SampleTopP(config_.top_p, config_.temperature);
        } else if (config_.top_k > 1) {
          search.SampleTopK(config_.top_k, config_.temperature);
        } else
          search.SelectTop();
      }
      return CudaSpanToVector(search.GetSequences().GetSequence(0));
    } else {
      Generators::BeamSearch_Cuda search{params};
      auto state = CreateState(search.sequence_lengths_, params);

      while (!search.IsDone()) {
        search.SetLogits(state->Run(search.GetSequenceLength(), search.GetNextTokens()));
        search.SelectTop();
      }
      return CudaSpanToVector(search.GetSequences().GetSequence(0));
    }
  }
#endif

  if (params.num_beams == 1) {
    Generators::GreedySearch search{params};
    auto state = CreateState(search.sequence_lengths_, params);

    while (!search.IsDone()) {
      search.SetLogits(state->Run(search.GetSequenceLength(), search.GetNextTokens()));

      if (config_.top_p < 1.0f) {
        search.SampleTopP(config_.top_p, config_.temperature);
      } else if (config_.top_k > 1) {
        search.SampleTopK(config_.top_k, config_.temperature);
      } else
        search.SelectTop();
    }
    return SpanToVector(search.GetSequences().GetSequence(0));
  } else {
    Generators::BeamSearch search{params};
    auto state = CreateState(search.sequence_lengths_, params);

    while (!search.IsDone()) {
      search.SetLogits(state->Run(search.GetSequenceLength(), search.GetNextTokens()));
      search.SelectTop();
    }
    return SpanToVector(search.GetSequences().GetSequence(0));
  }
}

}  // namespace Generators
