// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <array>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <functional>
#include "span.h"
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>
#if USE_CUDA
#include <cuda_runtime.h>
#include "cuda_common.h"
#else
// If we don't include cuda_runtime.h, we define this to avoid lots of extra #ifdefs
using cudaStream_t = void*;
#endif

#include "smartptrs.h"
#include "models/onnxruntime_api.h"
#include "models/debugging.h"
#include "config.h"
#include "logging.h"

namespace Generators {
struct Model;
struct State;
struct Search;

// OgaSequences are a vector of int32 vectors
using TokenSequences = std::vector<std::vector<int32_t>>;

enum struct DeviceType {
  CPU,
  CUDA,
  DML,
};

struct GeneratorParams : std::enable_shared_from_this<GeneratorParams> {
  GeneratorParams() = default;  // This constructor is only used if doing a custom model handler vs built-in
  GeneratorParams(const Model& model);

  Config::Search search;

  // Read only values copied from model
  int pad_token_id{};
  int eos_token_id{};
  int vocab_size{};
  int context_length{};

  int batch_size{1};
  int max_batch_size{0};
  bool use_cuda_graph{};
  int sequence_length{};
  int BatchBeamSize() const { return search.num_beams * batch_size; }

  DeviceType device_type{DeviceType::CPU};
  cudaStream_t cuda_stream{};

#if 0
  struct Bert {
    std::span<const int32_t> input_ids;  // Array of [batchsize][sequence_length]
  };

  struct Gpt {
    using Gpt=Bert;
  };

  struct T5 {
    std::span<const int32_t> encoder_input_ids;  // Array of [batchsize][sequence_length]
    std::span<const int32_t> decoder_input_ids;  // Array of [batchsize][sequence_length]
  };
  using Bart=T5;

#endif

  // TODO: Move this to a separate GPT struct
  std::span<const int32_t> input_ids;  // Array of [batchsize][sequence_length]

  struct Whisper {
    std::unique_ptr<OrtValue> input_features;  // float32 [batch_size, number_of_mels, something that is 3000]
  };

  std::variant<Whisper> inputs;

  std::vector<int32_t> input_ids_owner;  // Backing memory of input_ids in some cases

  std::shared_ptr<GeneratorParams> external_owner_;  // Set to 'this' when created by the C API to preserve lifetime

  struct Input {
    std::string name;
    std::unique_ptr<OrtValue> value;
  };

  // A list of extra model inputs that will be matched at runtime based on name
  std::vector<Input> extra_inputs;

  void TryGraphCapture(int max_bs);

 private:
  bool is_cuda_graph_enabled_{};
};

struct Generator {
  Generator(const Model& model, const GeneratorParams& params);

  bool IsDone() const;
  void ComputeLogits();
  void GenerateNextToken();

  RoamingArray<int32_t> GetSequence(int index) const;

  std::shared_ptr<const Model> model_;
  std::unique_ptr<State> state_;
  std::unique_ptr<Search> search_;
  bool computed_logits_{};  // Set to true in ComputeLogits() and false after appending a token to ensure a 1 to 1 call ratio
};

struct OrtGlobals {
  OrtGlobals();

  std::unique_ptr<OrtEnv> env_;
#if USE_CUDA
  std::unique_ptr<OrtMemoryInfo> memory_info_cuda_;
  std::unique_ptr<Ort::Allocator> allocator_cuda_;
#endif
 private:
  OrtGlobals(const OrtGlobals&) = delete;
  void operator=(const OrtGlobals&) = delete;
};

std::unique_ptr<OrtGlobals>& GetOrtGlobals();
void Shutdown();  // Do this once at exit, Ort code will fail after this call
OrtEnv& GetOrtEnv();

std::shared_ptr<Model> CreateModel(OrtEnv& ort_env, const char* config_path);
std::shared_ptr<GeneratorParams> CreateGeneratorParams(const Model& model);
std::shared_ptr<GeneratorParams> CreateGeneratorParams();  // For benchmarking purposes only
std::unique_ptr<Generator> CreateGenerator(const Model& model, const GeneratorParams& params);
std::vector<std::vector<int32_t>> Generate(const Model& model, const GeneratorParams& params);  // Uses CreateGenerator and a simple loop to return the entire sequence

float Float16ToFloat32(uint16_t v);  // v is a IEEE 752-2008 binary16 format, 1 sign bit, 5 bit exponent, 10 bit fraction
uint16_t Float32ToFloat16(float v);  // Opposite direction of above

void top_k_indices(std::span<int32_t> top_k, std::span<const float> inputs);

}  // namespace Generators
