// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <vector>

#ifndef _WIN32
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <gtest/gtest.h>

#include "dflash2_drafter.h"
#include "engine/cache_manager.h"
#include "engine/engine.h"
#include "engine/paged_key_value_cache.h"
#include "engine/step_plan.h"
#include "ep/webgpu/session_options.h"
#include "models/decoder_only.h"
#include "models/io/kv_cache.h"
#include "models/session_options.h"
#include "ort_genai.h"
#include "search.h"

namespace Generators::test {
namespace {

namespace fs_std = std::filesystem;

class ScopedSnapshotDirectory {
 public:
  explicit ScopedSnapshotDirectory(std::string_view name) {
    std::random_device random;
    for (size_t attempt = 0; attempt < 32; ++attempt) {
      const auto candidate = fs_std::temp_directory_path() /
                             (std::string{name} + "-" + std::to_string(random()) + "-" + std::to_string(random()));
      if (fs_std::create_directory(candidate)) {
        path_ = candidate;
        return;
      }
    }
    throw std::system_error(std::make_error_code(std::errc::file_exists),
                            "Unable to create snapshot test directory.");
  }

  ScopedSnapshotDirectory(const ScopedSnapshotDirectory&) = delete;
  ScopedSnapshotDirectory& operator=(const ScopedSnapshotDirectory&) = delete;

  ~ScopedSnapshotDirectory() {
    std::error_code error;
    fs_std::remove_all(path_, error);
    if (error) ADD_FAILURE() << "Unable to remove " << path_ << ": " << error.message();
  }

  const fs_std::path& Path() const { return path_; }

 private:
  fs_std::path path_;
};

Config MakeDflash2Config() {
  Config config;
  config.engine.dynamic_batching.emplace();
  config.engine.dynamic_batching->prefix_caching = true;
  config.engine.dynamic_batching->snapshot_cache.ram_bytes = 1024 * 1024;
  config.model.vocab_size = 128;
  auto& decoder = config.model.decoder;
  decoder.filename = "target.onnx";
  decoder.sliding_window = Config::Model::Decoder::SlidingWindow{4096};
  decoder.state_groups.emplace();
  decoder.state_groups->push_back(Config::Model::Decoder::StateGroup{
      Config::Model::Decoder::StateGroupKind::FixedConv});

  auto& dflash2 = config.model.dflash2;
  dflash2.filename = "dflash2.onnx";
  dflash2.num_hidden_layers = 3;
  dflash2.num_key_value_heads = 2;
  dflash2.head_size = 8;
  dflash2.block_size = 4;
  dflash2.num_draft_tokens = 3;
  dflash2.selector_top_k = 2;
  dflash2.mask_token_id = 31;
  dflash2.sliding_window = 17;
  return config;
}

std::shared_ptr<PrefixSnapshotCache> MakeSnapshotCache(const Config& config) {
  const auto& batching = *config.engine.dynamic_batching;
  const auto& limits = batching.snapshot_cache;
  return std::make_shared<PrefixSnapshotCache>(
      std::array<size_t, 3>{limits.vram_bytes, limits.ram_bytes, limits.disk_bytes},
      batching.max_snapshots.value_or(std::numeric_limits<size_t>::max()),
      fs_std::path{fs::path{limits.directory}.c_str()});
}

TEST(ConfigTests, FullProviderNameUpdatesCanonicalOptions) {
  Config config;
  SetProviderOption(config, "webgpu", "validationMode", "basic");
  SetProviderOption(config, "WebGpuExecutionProvider", "adapterIndex", "3");

  ASSERT_EQ(config.model.decoder.session_options.provider_options.size(), 1u);
  const auto& provider_options = config.model.decoder.session_options.provider_options.front();
  EXPECT_EQ(provider_options.name, "WebGPU");
  ASSERT_EQ(provider_options.options.size(), 2u);
  const auto find_option = [&provider_options](std::string_view name) {
    return std::find_if(provider_options.options.begin(), provider_options.options.end(),
                        [name](const auto& option) { return option.first == name; });
  };
  const auto validation_mode = find_option("validationMode");
  ASSERT_NE(validation_mode, provider_options.options.end());
  EXPECT_EQ(validation_mode->second, "basic");
  const auto adapter_index = find_option("adapterIndex");
  ASSERT_NE(adapter_index, provider_options.options.end());
  EXPECT_EQ(adapter_index->second, "3");
}

struct TensorMetadata {
  ONNXTensorElementDataType data_type;
  std::vector<int64_t> shape;
};

class SnapshotCountingBuffer final : public DeviceBuffer {
 public:
  SnapshotCountingBuffer(std::shared_ptr<DeviceBuffer> inner, size_t& downloads, size_t& uploads)
      : inner_{std::move(inner)}, downloads_{downloads}, uploads_{uploads} {
    p_device_ = inner_->p_device_;
    size_in_bytes_ = inner_->size_in_bytes_;
  }
  const char* GetType() const override { return inner_->GetType(); }
  void AllocateCpu() override {
    inner_->AllocateCpu();
    p_cpu_ = inner_->p_cpu_;
  }
  void CopyDeviceToCpu() override {
    ++downloads_;
    inner_->CopyDeviceToCpu();
    p_cpu_ = inner_->p_cpu_;
  }
  void CopyCpuToDevice() override {
    ++uploads_;
    inner_->CopyCpuToDevice();
  }
  void CopyFrom(size_t destination, DeviceBuffer& source, size_t offset, size_t bytes) override {
    inner_->CopyFrom(destination, source, offset, bytes);
  }
  void Zero() override { inner_->Zero(); }

 private:
  std::shared_ptr<DeviceBuffer> inner_;
  size_t& downloads_;
  size_t& uploads_;
};

class CheckpointAllocationDevice final : public DeviceInterface {
 public:
  CheckpointAllocationDevice(DeviceInterface& inner, bool host_snapshots,
                             Ort::Allocator* allocator = nullptr)
      : inner_{inner}, type_{host_snapshots ? DeviceType::CUDA : inner.GetType()}, allocator_{allocator} {}

  CheckpointAllocationDevice(DeviceInterface& inner, DeviceType type)
      : inner_{inner}, type_{type}, allocator_{nullptr} {}

  DeviceType GetType() const override { return type_; }
  void InitOrt(const OrtApi& api, Ort::Allocator& allocator) override {
    inner_.InitOrt(api, allocator);
  }
  Ort::Allocator& GetAllocator() override { return allocator_ ? *allocator_ : inner_.GetAllocator(); }
  std::unique_ptr<OrtMemoryInfo> GetMemoryInfo() const override {
    return inner_.GetMemoryInfo();
  }
  std::string GetExecutionProviderName() const override {
    return inner_.GetExecutionProviderName();
  }
  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return inner_.AllocateBase(size);
  }
  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* memory, size_t size) override {
    if (type_ == DeviceType::WEBGPU) {
      wrapped_memory.push_back(memory);
      return std::make_shared<SnapshotCountingBuffer>(
          inner_.WrapMemoryBase(memory, size), buffer_downloads, buffer_uploads);
    }
    return inner_.WrapMemoryBase(memory, size);
  }
  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override {
    return inner_.CreateGreedy(params);
  }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override {
    return inner_.CreateBeam(params);
  }
  std::unique_ptr<KeyValueCache> CreateKeyValueCache(State& state) override {
    return inner_.CreateKeyValueCache(state);
  }
  void Synchronize() override {
    ++synchronizations;
    inner_.Synchronize();
  }
  bool SupportsTransactionalFixedState() const override {
    return inner_.SupportsTransactionalFixedState();
  }
  bool SupportsCompactStateReplay() const override {
    return inner_.SupportsCompactStateReplay();
  }
  void ReplayStateUpdates(const StateUpdateReplayDesc* descriptors, size_t count) override {
    inner_.ReplayStateUpdates(descriptors, count);
  }
  std::shared_ptr<void> AllocatePinnedHost(size_t bytes) override {
    ++host_allocations;
    if (fail_host_allocation) {
      throw std::bad_alloc{};
    }
    void* memory = inner_.GetAllocator().Alloc(bytes);
    return std::shared_ptr<void>(memory, [this](void* pointer) {
      inner_.GetAllocator().Free(pointer);
    });
  }
  void CopyDeviceToHost(void* host, const void* device, size_t bytes) override {
    downloaded_bytes += bytes;
    std::memcpy(host, device, bytes);
  }
  void CopyHostToDevice(void* device, const void* host, size_t bytes) override {
    uploaded_bytes += bytes;
    std::memcpy(device, host, bytes);
  }
  size_t downloaded_bytes{};
  size_t uploaded_bytes{};
  size_t host_allocations{};
  bool fail_host_allocation{};
  size_t buffer_downloads{};
  size_t buffer_uploads{};
  size_t synchronizations{};
  std::vector<void*> wrapped_memory;

 private:
  DeviceInterface& inner_;
  DeviceType type_;
  Ort::Allocator* allocator_;
};

class FakeModelStateMetadata final : public ModelStateMetadata {
 public:
  void AddInput(std::string name, ONNXTensorElementDataType data_type,
                std::vector<int64_t> shape) {
    inputs_.insert_or_assign(
        std::move(name), TensorMetadata{data_type, std::move(shape)});
  }

  void AddOutput(std::string name, ONNXTensorElementDataType data_type,
                 std::vector<int64_t> shape) {
    outputs_.insert_or_assign(
        std::move(name), TensorMetadata{data_type, std::move(shape)});
  }

  bool HasInput(const std::string& name) const override { return inputs_.contains(name); }
  bool HasOutput(const std::string& name) const override { return outputs_.contains(name); }

  void RemoveInput(const std::string& name) { inputs_.erase(name); }

  ONNXTensorElementDataType GetInputDataType(const std::string& name) const override {
    return inputs_.at(name).data_type;
  }

  ONNXTensorElementDataType GetOutputDataType(const std::string& name) const override {
    return outputs_.at(name).data_type;
  }

  std::vector<int64_t> GetInputShape(const std::string& name) const override {
    return inputs_.at(name).shape;
  }

  std::vector<int64_t> GetOutputShape(const std::string& name) const override {
    return outputs_.at(name).shape;
  }

 private:
  std::unordered_map<std::string, TensorMetadata> inputs_;
  std::unordered_map<std::string, TensorMetadata> outputs_;
};

std::pair<FakeModelStateMetadata, FakeModelStateMetadata> MakeCompatibleMetadata(
    bool with_q_row_map = true) {
  FakeModelStateMetadata target;
  target.AddOutput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  FakeModelStateMetadata drafter;
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  drafter.AddInput("input_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {-1});
  if (with_q_row_map) {
    drafter.AddInput("q_row_map", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1});
  }
  for (const auto* name : {"qkv_row_map", "block_row_index",
                           "cumulative_sequence_lengths", "past_sequence_lengths"}) {
    drafter.AddInput(name, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1});
  }
  drafter.AddInput("block_table", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, -1});
  drafter.AddInput("attention_metadata", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {3});
  for (int layer = 0; layer < 3; ++layer) {
    for (const auto* kind : {"key", "value"}) {
      drafter.AddInput("past_key_values." + std::to_string(layer) + "." + kind,
                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 8, 2, 8});
      drafter.AddOutput("present." + std::to_string(layer) + "." + kind,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 8, 2, 8});
    }
  }
  drafter.AddOutput("draft_candidate_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, 3, 2});
  drafter.AddOutput("draft_scores", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 3, 2, 2});
  return {std::move(target), std::move(drafter)};
}

fs_std::path WriteDsparkConfig(std::string_view section_name) {
  const auto root = fs_std::temp_directory_path() /
                    ("ortgenai_dspark_config_" + std::string{section_name});
  std::error_code error;
  fs_std::remove_all(root, error);
  fs_std::create_directories(root);

  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << "{ \"model\": { \"type\": \"tiny-test-model\","
         " \"vocab_size\": 16, \"context_length\": 32,"
         " \"decoder\": { \"filename\": \"model.onnx\" }, \""
      << section_name
      << "\": { \"filename\": \"dspark.onnx\", \"num_hidden_layers\": 1,"
         " \"num_key_value_heads\": 2, \"head_size\": 8, \"block_size\": 4,"
         " \"num_draft_tokens\": 4, \"selector_top_k\": 2 } }, \"search\": {} }";
  return root;
}

fs_std::path WriteDuplicateBlockDrafterConfig() {
  const auto root = fs_std::temp_directory_path() / "ortgenai_duplicate_block_drafter_config";
  std::error_code error;
  fs_std::remove_all(root, error);
  fs_std::create_directories(root);

  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << R"({"model":{"type":"tiny-test-model","vocab_size":16,"context_length":32,)"
         R"("decoder":{"filename":"model.onnx"},)"
         R"("dflash2":{"filename":"dflash2.onnx"},)"
         R"("dspark":{"filename":"dspark.onnx"}},"search":{}})";
  return root;
}

fs_std::path WriteIncompleteBlockDrafterConfig() {
  const auto root = fs_std::temp_directory_path() / "ortgenai_incomplete_block_drafter_config";
  std::error_code error;
  fs_std::remove_all(root, error);
  fs_std::create_directories(root);

  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << R"({"model":{"type":"tiny-test-model","vocab_size":16,"context_length":32,)"
         R"("decoder":{"filename":"model.onnx"},"dflash2":{ }},"search":{}})";
  return root;
}

// Builds a minimal ONNX drafter graph that consumes `anchor_row_index` and `aux_hidden_states`.
// For each query block b in [0, query_batch):
//   - draft token 0 reports `anchor_row_index[b]`
//   - draft token 1 reports `static_cast<int32_t>(aux_hidden_states[anchor_row_index[b], 0])`
//   - draft token 2 reports `static_cast<int32_t>(past_key_values.0.key[0, 0, 0, 0])`
// and `present.0.key[0, 0, 0, 0]` is updated to `aux_hidden_states[anchor_row_index[0], 0]`, so
// even a non-drafting step (whose lattice is discarded) records the row gathered by its borrowed
// placeholder block for the next step to observe.
fs_std::path WriteAnchorRowIndexDrafterModel() {
  auto varint = [](uint64_t v) {
    std::string out;
    while (v >= 0x80u) {
      out.push_back(static_cast<char>((v & 0x7fu) | 0x80u));
      v >>= 7;
    }
    out.push_back(static_cast<char>(v));
    return out;
  };
  auto vf = [&](uint32_t field, uint64_t v) {
    return varint(static_cast<uint64_t>(field) << 3) + varint(v);
  };
  auto bf = [&](uint32_t field, std::string_view bytes) {
    std::string out = varint((static_cast<uint64_t>(field) << 3) | 2u);
    out += varint(bytes.size());
    out.append(bytes.data(), bytes.size());
    return out;
  };
  auto value_info = [&](std::string_view name, uint64_t elem_type,
                        std::initializer_list<int64_t> dims) {
    std::string shape;
    for (const int64_t d : dims) {
      const std::string dim =
          d < 0 ? bf(2, d == -2 ? "q" : "d") : vf(1, static_cast<uint64_t>(d));
      shape += bf(1, dim);
    }
    const std::string tensor_type = vf(1, elem_type) + bf(2, shape);
    return bf(1, name) + bf(2, bf(1, tensor_type));
  };
  auto int64_init = [&](std::string_view name, std::initializer_list<int64_t> dims,
                        std::initializer_list<int64_t> vals) {
    std::string t;
    for (const int64_t d : dims) t += vf(1, static_cast<uint64_t>(d));
    t += vf(2, 7);  // TensorProto::INT64
    for (const int64_t v : vals) t += vf(7, static_cast<uint64_t>(v));
    t += bf(8, name);
    return t;
  };
  auto node = [&](std::string_view op, std::initializer_list<std::string_view> ins,
                  std::initializer_list<std::string_view> outs,
                  std::string_view attr_name = {}, int64_t attr_int = 0) {
    std::string n;
    for (const auto i : ins) n += bf(1, i);
    for (const auto o : outs) n += bf(2, o);
    n += bf(4, op);
    if (!attr_name.empty()) {
      const std::string attr =
          bf(1, attr_name) + vf(3, static_cast<uint64_t>(attr_int)) + vf(20, 2);
      n += bf(5, attr);
    }
    return n;
  };

  constexpr uint64_t kFloat = 1, kInt32 = 6, kInt64 = 7;
  std::string graph;
  for (const auto& n : {
           node("Gather", {"aux_hidden_states", "anchor_row_index"}, {"aa"}, "axis", 0),
           node("Cast", {"aa"}, {"ai"}, "to", static_cast<int64_t>(kInt32)),
           node("Unsqueeze", {"anchor_row_index", "a1"}, {"ac"}),
           node("Sub", {"ac", "ac"}, {"zc"}),
           node("GatherND", {"past_key_values.0.key", "i0"}, {"pk"}),
           node("Cast", {"pk"}, {"pi"}, "to", static_cast<int64_t>(kInt32)),
           node("Add", {"zc", "pi"}, {"pc"}),
           node("Concat", {"ac", "ai", "pc"}, {"dv"}, "axis", 1),
           node("Unsqueeze", {"dv", "a2"}, {"dc"}),
           node("Concat", {"dc", "dc"}, {"draft_candidate_ids"}, "axis", 2),
           node("Sub", {"draft_candidate_ids", "draft_candidate_ids"}, {"zi"}),
           node("Cast", {"zi"}, {"zf"}, "to", static_cast<int64_t>(kFloat)),
           node("Unsqueeze", {"zf", "a3"}, {"zs"}),
           node("Concat", {"zs", "zs"}, {"draft_scores"}, "axis", 3),
           node("Gather", {"aa", "a0"}, {"fa"}, "axis", 0),
           node("Squeeze", {"fa", "a1"}, {"fv"}),
           node("Sub", {"pk", "pk"}, {"pz"}),
           node("Add", {"fv", "pz"}, {"fvu"}),
           node("ScatterND", {"past_key_values.0.key", "i0", "fvu"}, {"present.0.key"}),
           node("Identity", {"past_key_values.0.value"}, {"present.0.value"}),
       }) {
    graph += bf(1, n);
  }
  graph += bf(2, "anchor_dflash2");
  for (const auto& init : {
           int64_init("a0", {1}, {0}),
           int64_init("a1", {1}, {1}),
           int64_init("a2", {1}, {2}),
           int64_init("a3", {1}, {3}),
           int64_init("i0", {1, 4}, {0, 0, 0, 0}),
       }) {
    graph += bf(5, init);
  }
  for (const auto& vi : {
           value_info("aux_hidden_states", kFloat, {-1, 1}),
           value_info("input_ids", kInt64, {-1}),
           value_info("q_row_map", kInt32, {-1}),
           value_info("qkv_row_map", kInt32, {-1}),
           value_info("block_row_index", kInt32, {-1}),
           value_info("anchor_row_index", kInt32, {-2}),
           value_info("cumulative_sequence_lengths", kInt32, {-1}),
           value_info("past_sequence_lengths", kInt32, {-1}),
           value_info("block_table", kInt32, {-1, -1}),
           value_info("attention_metadata", kInt32, {3}),
           value_info("past_key_values.0.key", kFloat, {-1, 4, 1, 1}),
           value_info("past_key_values.0.value", kFloat, {-1, 4, 1, 1}),
       }) {
    graph += bf(11, vi);
  }
  for (const auto& vo : {
           value_info("draft_candidate_ids", kInt32, {-2, 3, 2}),
           value_info("draft_scores", kFloat, {-2, 3, 2, 2}),
           value_info("present.0.key", kFloat, {-1, 4, 1, 1}),
           value_info("present.0.value", kFloat, {-1, 4, 1, 1}),
       }) {
    graph += bf(12, vo);
  }

  const std::string model_bytes = vf(1, 8) + bf(7, graph) + bf(8, bf(1, "") + vf(2, 17));
  const auto root = fs_std::temp_directory_path() / "ortgenai_dflash_anchor_drafter";
  std::error_code error;
  fs_std::remove_all(root, error);
  fs_std::create_directories(root);
  std::ofstream out(root / "dflash2_anchor.onnx", std::ios::binary);
  out.write(model_bytes.data(), static_cast<std::streamsize>(model_bytes.size()));
  return root;
}

}  // namespace

TEST(Dflash2ConfigTest, RequiresDrafterFilename) {
  auto config = MakeDflash2Config();
  config.model.dflash2.filename.clear();
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresCompleteGeometry) {
  auto config = MakeDflash2Config();
  config.model.dflash2.num_key_value_heads = 0;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresPositiveSelectorTopK) {
  auto config = MakeDflash2Config();
  config.model.dflash2.selector_top_k = 0;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, ParsesIndependentSamplingOptions) {
  const auto root = fs_std::temp_directory_path() / "ortgenai_dflash_independent_sampling";
  std::error_code error;
  fs_std::remove_all(root, error);
  fs_std::create_directories(root);
  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << R"({"model":{"type":"tiny-test-model","vocab_size":128,"context_length":32,)"
         R"("decoder":{"filename":"model.onnx"},"dflash2":{"filename":"dflash2.onnx",)"
         R"("num_hidden_layers":1,"num_key_value_heads":2,"head_size":8,"block_size":4,)"
         R"("num_draft_tokens":3,"selector_top_k":4,"mask_token_id":31,"sliding_window":17,)"
         R"("independent_sampling":true,"sampling_temperature":0.1,"sampling_top_p":0.95,)"
         R"("sampling_min_p":0.3}},"search":{}})";
  out.close();

  Config config(fs::path{root.string()}, "");
  EXPECT_TRUE(config.model.dflash2.independent_sampling);
  EXPECT_FLOAT_EQ(config.model.dflash2.sampling_temperature, 0.1f);
  EXPECT_FLOAT_EQ(config.model.dflash2.sampling_top_p, 0.95f);
  EXPECT_FLOAT_EQ(config.model.dflash2.sampling_min_p, 0.3f);
}

TEST(Dflash2ConfigTest, ParsesSampledProposal) {
  using Proposal = Config::Model::Dflash2::SampledProposal;
  auto parse = [](std::string_view name, std::string_view options) {
    const auto root = fs_std::temp_directory_path() / std::string{name};
    std::error_code error;
    fs_std::remove_all(root, error);
    fs_std::create_directories(root);
    std::ofstream out(root / "genai_config.json", std::ios::binary);
    out << R"({"model":{"type":"tiny-test-model","vocab_size":128,"context_length":32,)"
           R"("decoder":{"filename":"model.onnx"},"dflash2":{"filename":"dflash2.onnx",)"
           R"("num_hidden_layers":1,"num_key_value_heads":2,"head_size":8,"block_size":4,)"
           R"("num_draft_tokens":3,"selector_top_k":4,"mask_token_id":31,"sliding_window":17)"
        << options << R"(}},"search":{}})";
    out.close();
    return Config(fs::path{root.string()}, "");
  };

  EXPECT_EQ(parse("ortgenai_dflash_proposal_default", "").model.dflash2.sampled_proposal,
            Proposal::Lattice);
  EXPECT_EQ(parse("ortgenai_dflash_proposal_none", R"(,"sampled_proposal":"none")")
                .model.dflash2.sampled_proposal,
            Proposal::None);
  const auto lattice = parse("ortgenai_dflash_proposal_lattice", R"(,"sampled_proposal":"lattice")");
  EXPECT_EQ(lattice.model.dflash2.sampled_proposal, Proposal::Lattice);
  EXPECT_FALSE(lattice.model.dflash2.independent_sampling);
  EXPECT_EQ(parse("ortgenai_dflash_proposal_greedy", R"(,"sampled_proposal":"greedy_path")")
                .model.dflash2.sampled_proposal,
            Proposal::GreedyPath);
  EXPECT_EQ(parse("ortgenai_dflash_proposal_alias", R"(,"independent_sampling":true)")
                .model.dflash2.sampled_proposal,
            Proposal::Independent);
  EXPECT_THROW(parse("ortgenai_dflash_proposal_bad", R"(,"sampled_proposal":"beam")"),
               std::runtime_error);

  // Both spellings resolve the same way in either key order, and a contradiction is rejected.
  for (const bool proposal_first : {true, false}) {
    auto both = [&](std::string_view name, std::string_view proposal, bool independent) {
      const std::string proposal_key = R"(,"sampled_proposal":")" + std::string{proposal} + "\"";
      const std::string independent_key =
          std::string{R"(,"independent_sampling":)"} + (independent ? "true" : "false");
      return parse(std::string{name} + (proposal_first ? "_pf" : "_if"),
                   proposal_first ? proposal_key + independent_key
                                  : independent_key + proposal_key);
    };
    SCOPED_TRACE(proposal_first ? "sampled_proposal first" : "independent_sampling first");
    EXPECT_EQ(both("ortgenai_dflash_proposal_none_false", "none", false)
                  .model.dflash2.sampled_proposal,
              Proposal::None);
    const auto independent = both("ortgenai_dflash_proposal_indep_true", "independent", true);
    EXPECT_EQ(independent.model.dflash2.sampled_proposal, Proposal::Independent);
    EXPECT_TRUE(independent.model.dflash2.independent_sampling);
    EXPECT_THROW(both("ortgenai_dflash_proposal_none_true", "none", true), std::runtime_error);
    EXPECT_THROW(both("ortgenai_dflash_proposal_indep_false", "independent", false),
                 std::runtime_error);
  }
  EXPECT_EQ(parse("ortgenai_dflash_proposal_legacy_false", R"(,"independent_sampling":false)")
                .model.dflash2.sampled_proposal,
            Proposal::Lattice);
}

TEST(Dflash2ConfigTest, LatticeWalkConditionsOnTheSampledPredecessor) {
  // Step 0 always draws candidate 1. Step 1's row 0 would pick candidate 0 and row 1 candidate 1,
  // so only a walk that follows the draw lands on 21.
  Dflash2Lattice lattice;
  lattice.top_k = 2;
  lattice.candidate_ids = {10, 11, 20, 21};
  lattice.scores = {-100.0f, 0.0f, -100.0f, 0.0f,
                    0.0f, -100.0f, -100.0f, 0.0f};
  std::mt19937 rng{7};
  std::vector<int32_t> tokens;
  std::vector<TargetTokenSelection> distributions;
  Dflash2SampleLatticePath(lattice, 2, /*temperature=*/1.0f, /*top_k=*/20, /*top_p=*/1.0f, rng,
                           tokens, distributions);
  EXPECT_EQ(tokens, (std::vector<int32_t>{11, 21}));
  ASSERT_EQ(distributions.size(), 2u);
  EXPECT_EQ(distributions[1].indices.front(), 21);
  EXPECT_NEAR(distributions[1].probs.front(), 1.0f, 1e-6f);

  Dflash2SampleLatticePath(lattice, 1, 1.0f, 20, 1.0f, rng, tokens, distributions);
  EXPECT_EQ(tokens, (std::vector<int32_t>{11}));
}

TEST(Dflash2ConfigTest, LatticeWalkTruncatesLikeTheTarget) {
  Dflash2Lattice lattice;
  lattice.top_k = 4;
  lattice.candidate_ids = {10, 11, 12, 13};
  lattice.scores.assign(16, 0.0f);
  const std::array<float, 4> row{2.0f, 4.0f, 1.0f, 3.0f};
  std::copy(row.begin(), row.end(), lattice.scores.begin());
  std::mt19937 rng{3};
  std::vector<int32_t> tokens;
  std::vector<TargetTokenSelection> distributions;

  // Top-k 3, then top-p 0.7 keeps the two leading candidates, renormalised at temperature 1.
  Dflash2SampleLatticePath(lattice, 1, 1.0f, 3, 0.7f, rng, tokens, distributions);
  ASSERT_EQ(distributions.size(), 1u);
  EXPECT_EQ(distributions[0].indices, (std::vector<int32_t>{11, 13}));
  EXPECT_NEAR(distributions[0].probs[0], 0.7310586f, 1e-6f);
  EXPECT_NEAR(distributions[0].probs[1], 0.2689414f, 1e-6f);

  // Top-p applies to the top-k renormalized mass: 0.731 of the top two already covers 0.7, while
  // the full row would give the leader only 0.644 and keep a second candidate.
  Dflash2SampleLatticePath(lattice, 1, 1.0f, 2, 0.7f, rng, tokens, distributions);
  EXPECT_EQ(distributions[0].indices, (std::vector<int32_t>{11}));
  EXPECT_EQ(tokens, (std::vector<int32_t>{11}));

  Dflash2Lattice non_finite = lattice;
  non_finite.scores[1] = std::numeric_limits<float>::infinity();
  EXPECT_THROW(Dflash2SampleLatticePath(non_finite, 1, 1.0f, 0, 1.0f, rng, tokens, distributions),
               std::runtime_error);

  // Temperature 2 on the full row, with the sample frequencies matching the recorded q.
  std::array<int, 4> counts{};
  constexpr int kDraws = 20000;
  for (int i = 0; i < kDraws; ++i) {
    Dflash2SampleLatticePath(lattice, 1, 2.0f, 0, 1.0f, rng, tokens, distributions);
    ++counts[static_cast<size_t>(tokens[0] - 10)];
  }
  ASSERT_EQ(distributions[0].indices, (std::vector<int32_t>{11, 13, 10, 12}));
  for (size_t i = 0; i < distributions[0].indices.size(); ++i) {
    const auto token = static_cast<size_t>(distributions[0].indices[i] - 10);
    EXPECT_NEAR(static_cast<double>(counts[token]) / kDraws, distributions[0].probs[i], 0.015);
  }
}

TEST(Dflash2ConfigTest, ParsesAnchorRowIndexInput) {
  const auto root = fs_std::temp_directory_path() / "ortgenai_dflash_anchor_row_index";
  std::error_code error;
  fs_std::remove_all(root, error);
  fs_std::create_directories(root);
  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << R"({"model":{"type":"tiny-test-model","vocab_size":128,"context_length":32,)"
         R"("decoder":{"filename":"model.onnx"},"dflash2":{"filename":"dflash2.onnx",)"
         R"("num_hidden_layers":1,"num_key_value_heads":2,"head_size":8,"block_size":4,)"
         R"("num_draft_tokens":3,"selector_top_k":4,"mask_token_id":31,"sliding_window":17,)"
         R"("inputs":{"anchor_row_index":"anchor_rows"}}},"search":{}})";
  out.close();

  Config config(fs::path{root.string()}, "");
  EXPECT_EQ(config.model.dflash2.inputs.anchor_row_index, "anchor_rows");
  // Absent by default: DFlash 2 / DSpark drafters do not read an anchor row.
  EXPECT_TRUE(MakeDflash2Config().model.dflash2.inputs.anchor_row_index.empty());
}

TEST(Dflash2ConfigTest, AcceptsAnchorRowIndexInput) {
  auto config = MakeDflash2Config();
  config.model.dflash2.inputs.anchor_row_index = "anchor_row_index";
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddInput("anchor_row_index", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1});
  EXPECT_NO_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8));
}

TEST(Dflash2ConfigTest, RejectsMissingOrMistypedAnchorRowIndexInput) {
  auto config = MakeDflash2Config();
  config.model.dflash2.inputs.anchor_row_index = "anchor_row_index";
  {
    auto [target, drafter] = MakeCompatibleMetadata();
    EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
  }
  {
    auto [target, drafter] = MakeCompatibleMetadata();
    drafter.AddInput("anchor_row_index", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, {-1});
    EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
  }
  {
    auto [target, drafter] = MakeCompatibleMetadata();
    drafter.AddInput("anchor_row_index", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {2});
    EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
  }
  {
    auto [target, drafter] = MakeCompatibleMetadata();
    config.model.dflash2.inputs.anchor_row_index = config.model.dflash2.inputs.qkv_row_map;
    EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
  }
}

TEST(Dflash2ConfigTest, RejectsSamplingTemperatureOutsideFloatRange) {
  const auto root = fs_std::temp_directory_path() /
                    "ortgenai_dflash_sampling_temperature_out_of_range";
  std::error_code error;
  fs_std::remove_all(root, error);
  fs_std::create_directories(root);
  std::ofstream out(root / "genai_config.json", std::ios::binary);
  out << R"({"model":{"type":"tiny-test-model","vocab_size":128,"context_length":32,)"
         R"("decoder":{"filename":"model.onnx"},"dflash2":{"filename":"dflash2.onnx",)"
         R"("num_hidden_layers":1,"num_key_value_heads":2,"head_size":8,"block_size":4,)"
         R"("num_draft_tokens":3,"selector_top_k":4,"mask_token_id":31,"sliding_window":17,)"
         R"("independent_sampling":true,"sampling_temperature":1e39}},"search":{}})";
  out.close();

  EXPECT_THROW(Config(fs::path{root.string()}, ""), std::runtime_error);
}

TEST(Dflash2ConfigTest, BuildsReferenceIndependentDistribution) {
  const std::array<int32_t, 4> candidates{10, 11, 12, 13};
  const std::array<float, 4> logits{4.0f, 3.0f, 2.0f, 1.0f};
  const auto distribution = Dflash2IndependentDraftDistribution(
      candidates.data(), logits.data(), candidates.size(),
      /*temperature=*/1.0f, /*top_p=*/0.95f, /*min_p=*/0.3f);
  ASSERT_EQ(distribution.indices, (std::vector<int32_t>{10, 11}));
  ASSERT_EQ(distribution.probs.size(), 2u);
  EXPECT_NEAR(distribution.probs[0], 0.7310586f, 1e-6f);
  EXPECT_NEAR(distribution.probs[1], 0.2689414f, 1e-6f);
}

TEST(Dflash2ConfigTest, SortsSelectorScoresBeforeApplyingProbabilityFilters) {
  const std::array<int32_t, 4> candidates{10, 11, 12, 13};
  const std::array<float, 4> logits{2.0f, 4.0f, 1.0f, 3.0f};
  const auto distribution = Dflash2IndependentDraftDistribution(
      candidates.data(), logits.data(), candidates.size(),
      /*temperature=*/1.0f, /*top_p=*/0.7f, /*min_p=*/0.2f);
  ASSERT_EQ(distribution.indices, (std::vector<int32_t>{11, 13}));
  ASSERT_EQ(distribution.probs.size(), 2u);
  EXPECT_NEAR(distribution.probs[0], 0.7310586f, 1e-6f);
  EXPECT_NEAR(distribution.probs[1], 0.2689414f, 1e-6f);
}

TEST(Dflash2ConfigTest, RejectsAsynchronousExecution) {
  auto config = MakeDflash2Config();
  config.model.dflash2.run_options = Config::RunOptions{
      {"disable_synchronize_execution_providers", "1"}};
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, RejectsSimultaneousMtpDrafter) {
  auto config = MakeDflash2Config();
  config.model.mtp.filename = "mtp.onnx";
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, AllowsDisabledMtpMetadata) {
  auto config = MakeDflash2Config();
  config.model.mtp.filename = "mtp.onnx";
  config.model.mtp.enabled = false;
  EXPECT_NO_THROW(CreateDflash2Config(config));
}

TEST(Dflash2ConfigTest, PreservesTargetProviderOptions) {
  auto config = MakeDflash2Config();
  config.model.decoder.session_options.providers = {"cuda"};
  config.model.decoder.session_options.provider_options.push_back(
      {"cuda", {{"device_id", "1"}, {"arena_extend_strategy", "kSameAsRequested"}}});
  config.model.dflash2.session_options.emplace();
  config.model.dflash2.session_options->provider_options.push_back(
      {"cuda", {{"arena_extend_strategy", "kNextPowerOfTwo"}}});

  const auto projected = CreateDflash2Config(config);
  const auto& session_options = projected->model.decoder.session_options;
  ASSERT_EQ(session_options.providers.size(), 1u);
  EXPECT_EQ(session_options.providers[0], "cuda");
  ASSERT_EQ(session_options.provider_options.size(), 1u);
  ASSERT_EQ(session_options.provider_options[0].options.size(), 2u);
  EXPECT_EQ(session_options.provider_options[0].options[0],
            Config::NamedString("arena_extend_strategy", "kNextPowerOfTwo"));
  EXPECT_EQ(session_options.provider_options[0].options[1],
            Config::NamedString("device_id", "1"));
}

TEST(Dflash2ConfigTest, WebGpuMatmulAccumulationDefaultsAndOverrides) {
  if (FindRegisteredEpDevices("WebGpuExecutionProvider").empty()) {
    GTEST_SKIP() << "No WebGPU EP device is registered.";
  }
  for (const char* model_kind : {"plain", "dflash2", "dspark"}) {
    for (const char* batching : {"none", "dynamic", "static"}) {
      for (int max_batch_size : {1, 2}) {
        for (const char* explicit_value : {"", "0", "1"}) {
          SCOPED_TRACE(::testing::Message() << model_kind << " " << batching << " batch="
                                            << max_batch_size << " override=" << explicit_value);
          auto config = MakeDflash2Config();
          const std::string_view kind{model_kind}, batch{batching};
          if (kind == "plain") config.model.dflash2.filename.clear();
          config.model.dflash2.is_dspark = kind == "dspark";
          if (batch == "dynamic") {
            config.engine.dynamic_batching.emplace().max_batch_size = max_batch_size;
          } else if (batch == "static") {
            config.engine.static_batching.emplace().max_batch_size = max_batch_size;
          }
          Config::ProviderOptions provider{"WebGPU", {}};
          if (*explicit_value) {
            provider.options.emplace_back("enableMatmulFp32Accumulation", explicit_value);
          }
          // Projection must retain the same default/override for the auxiliary drafter session.
          for (bool drafter : {false, true}) {
            if (drafter && kind != "dflash2") continue;
            const auto projected = drafter ? CreateDflash2Config(config) : nullptr;
            auto options = OrtSessionOptions::Create();
            WebGPUExecutionProvider::AppendExecutionProvider(*options, provider, projected ? *projected : config);
            const char* key = "ep.webgpuexecutionprovider.enableMatmulFp32Accumulation";
            const bool default_enabled = kind == "dflash2" || (batch != "none" && max_batch_size > 1);
            const bool specified = *explicit_value || default_enabled;
            ASSERT_EQ(options->HasConfigEntry(key), specified);
            if (specified) {
              size_t size{};
              Ort::ThrowOnError(Ort::api->GetSessionConfigEntry(options.get(), key, nullptr, &size));
              std::string value(size, '\0');
              Ort::ThrowOnError(Ort::api->GetSessionConfigEntry(options.get(), key, value.data(), &size));
              EXPECT_STREQ(value.c_str(), *explicit_value ? explicit_value : "1");
            }
          }
          // Defaults must not mutate the user-provided options, including an explicit opt-out.
          EXPECT_EQ(provider.options.size(), *explicit_value ? 1u : 0u);
        }
      }
    }
  }
}

TEST(Dflash2ConfigTest, DrafterSessionOverridesTargetConfigEntries) {
  auto config = MakeDflash2Config();
  config.model.decoder.session_options.config_entries.push_back(
      {"ep.cuda.fpa_intb_gemm", "1"});
  config.model.dflash2.session_options.emplace();
  config.model.dflash2.session_options->config_entries.push_back(
      {"ep.cuda.fpa_intb_gemm", "0"});

  const auto projected = CreateDflash2Config(config);
  const auto& entries = projected->model.decoder.session_options.config_entries;
  ASSERT_EQ(entries.size(), 1u);
  EXPECT_EQ(entries[0], Config::NamedString("ep.cuda.fpa_intb_gemm", "0"));
}

TEST(Dflash2ConfigTest, AcceptsCompatibleAuxiliaryHiddenStates) {
  const auto config = MakeDflash2Config();
  const auto [target, drafter] = MakeCompatibleMetadata();
  EXPECT_NO_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8));
}

TEST(Dflash2ConfigTest, AcceptsOmittedAttentionMetadata) {
  auto config = MakeDflash2Config();
  config.model.dflash2.inputs.attention_metadata.clear();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.RemoveInput("attention_metadata");
  EXPECT_NO_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8));
}

// Dflash2Drafter::AllocateCache() allocates only key and value buffers from the logical head size
// and binds only past_key_names/past_value_names, so a declared scale template would be silently
// dropped and the drafter session would run against unbound, uninitialized scales. Reject it.
TEST(Dflash2ConfigTest, RejectsScaleNameTemplates) {
  const auto metadata = MakeCompatibleMetadata();
  for (int field = 0; field < 4; ++field) {
    auto config = MakeDflash2Config();
    auto& inputs = config.model.dflash2.inputs;
    auto& outputs = config.model.dflash2.outputs;
    switch (field) {
      case 0:
        inputs.past_key_scale_names = "draft_past.%d.key_scale";
        break;
      case 1:
        inputs.past_value_scale_names = "draft_past.%d.value_scale";
        break;
      case 2:
        outputs.present_key_scale_names = "draft_present.%d.key_scale";
        break;
      default:
        outputs.present_value_scale_names = "draft_present.%d.value_scale";
        break;
    }
    try {
      static_cast<void>(ValidateDflash2ModelCompatibility(config, metadata.first, metadata.second, 8));
      FAIL() << "Expected a quantized block-drafter cache to be rejected for field " << field;
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string{error.what()}.find("not supported"), std::string::npos)
          << "field " << field << ": " << error.what();
    }
  }
}

TEST(Dflash2ConfigTest, UsesConfiguredTargetOutput) {
  auto config = MakeDflash2Config();
  config.model.dflash2.main_aux_hidden_states = "custom_aux_hidden_states";
  auto [target, drafter] = MakeCompatibleMetadata();
  target.AddOutput("custom_aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  EXPECT_NO_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8));
}

TEST(Dflash2ConfigTest, RequiresConfiguredTargetOutput) {
  const auto config = MakeDflash2Config();
  FakeModelStateMetadata target;
  FakeModelStateMetadata drafter;
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresConfiguredDrafterInput) {
  const auto config = MakeDflash2Config();
  FakeModelStateMetadata target;
  target.AddOutput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  const FakeModelStateMetadata drafter;
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresTwoDimensionalAuxiliaryTensors) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  target.AddOutput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 4, 16});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);

  std::tie(target, drafter) = MakeCompatibleMetadata();
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 4, 16});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresMatchingAuxiliaryWidth) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 32});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresMatchingAuxiliaryType) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddInput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 64});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresDynamicPackedDimensions) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  target.AddOutput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {1, 64});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);

  std::tie(target, drafter) = MakeCompatibleMetadata();
  drafter.AddInput("q_row_map", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {1});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);

  std::tie(target, drafter) = MakeCompatibleMetadata();
  drafter.AddInput("block_table", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, 8});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresUniqueRuntimeInputNames) {
  auto config = MakeDflash2Config();
  config.model.dflash2.inputs.qkv_row_map = config.model.dflash2.inputs.q_row_map;
  const auto [target, drafter] = MakeCompatibleMetadata();
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, AcceptsPackedQkvDrafterWithoutQueryRowMap) {
  const auto config = MakeDflash2Config();
  const auto [target, drafter] = MakeCompatibleMetadata(false);
  EXPECT_NO_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8));
}

TEST(Dflash2ConfigTest, RequiresCompleteDrafterContract) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddOutput("draft_scores", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 3, 2});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);

  std::tie(target, drafter) = MakeCompatibleMetadata();
  drafter.AddInput("past_key_values.1.key", ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
                   {-1, -1, 4, 8});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresUniqueEmbeddingInputName) {
  auto config = MakeDflash2Config();
  config.model.embedding.filename = "embedding.onnx";
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddInput(config.model.dflash2.inputs.embeddings, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16, {-1, 64});
  EXPECT_NO_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8));
  for (const auto* alias : {"aux_hidden_states", "input_ids", "past_key_values.0.key", ""}) {
    config.model.dflash2.inputs.embeddings = alias;
    EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error) << alias;
  }
}

TEST(Dflash2ConfigTest, ReservesEmbeddingGrowthPeak) {
  EXPECT_EQ(Dflash2Drafter::EmbeddingReservedBytes(2, 4, 64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            3u * 2 * 4 * 64 * 2);
  EXPECT_EQ(Dflash2Drafter::EmbeddingReservedBytes(3, 8, 64, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
            3u * 3 * 8 * 64 * 4);
  EXPECT_THROW(Dflash2Drafter::EmbeddingReservedBytes(std::numeric_limits<size_t>::max(), 2, 64,
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
               std::runtime_error);
  EXPECT_THROW(Dflash2Drafter::EmbeddingReservedBytes(std::numeric_limits<size_t>::max() / 2, 1, 1,
                                                      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
               std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresOneDraftPerNonAnchorBlockRow) {
  auto config = MakeDflash2Config();
  config.model.dflash2.num_draft_tokens = 1;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresMaskTokenInsideVocabulary) {
  auto config = MakeDflash2Config();
  config.model.vocab_size = 0;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);

  config.model.vocab_size = 128;
  config.model.dflash2.mask_token_id = -1;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);

  config.model.dflash2.mask_token_id = config.model.vocab_size;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);

  config.model.dflash2.mask_token_id = 0;
  EXPECT_NO_THROW(CreateDflash2Config(config));

  config.model.dflash2.mask_token_id = config.model.vocab_size - 1;
  EXPECT_NO_THROW(CreateDflash2Config(config));
}

TEST(Dflash2ConfigTest, AcceptsOneDraftPerDsparkBlockRow) {
  auto config = MakeDflash2Config();
  config.model.dflash2.is_dspark = true;
  config.model.dflash2.num_draft_tokens = config.model.dflash2.block_size;
  EXPECT_NO_THROW(CreateDflash2Config(config));
}

TEST(Dflash2ConfigTest, RejectsDsparkGeometryForDflash2) {
  auto config = MakeDflash2Config();
  config.model.dflash2.num_draft_tokens = config.model.dflash2.block_size;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, RejectsDflash2GeometryForDspark) {
  auto config = MakeDflash2Config();
  config.model.dflash2.is_dspark = true;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, RejectsFullAttentionForDflash2) {
  auto config = MakeDflash2Config();
  config.model.dflash2.sliding_window = 0;
  EXPECT_THROW(CreateDflash2Config(config), std::runtime_error);
}

TEST(Dflash2ConfigTest, ParsesDsparkAlias) {
  EXPECT_NO_THROW(OgaConfig::Create(WriteDsparkConfig("dspark").string().c_str()));
  EXPECT_THROW(OgaConfig::Create(WriteDsparkConfig("dspark2").string().c_str()), std::exception);
}

TEST(Dflash2ConfigTest, RejectsBothBlockDrafterAliases) {
  EXPECT_THROW(OgaConfig::Create(WriteDuplicateBlockDrafterConfig().string().c_str()),
               std::runtime_error);
}

TEST(Dflash2ConfigTest, AllowsSameBlockDrafterAliasInOverlays) {
  const auto root = WriteDsparkConfig("dspark");
  EXPECT_NO_THROW(Config(fs::path{root.string()},
                         R"({"model":{"dspark":{"selector_top_k":3}}})"));

  auto config = OgaConfig::Create(root.string().c_str());
  EXPECT_NO_THROW(config->Overlay(R"({"model":{"dspark":{"selector_top_k":3}}})"));
}

TEST(Dflash2ConfigTest, RejectsDifferentBlockDrafterAliasInOverlay) {
  auto config = OgaConfig::Create(WriteDsparkConfig("dspark").string().c_str());
  EXPECT_THROW(config->Overlay(R"({"model":{"dflash2":{"selector_top_k":3}}})"),
               std::runtime_error);

  config = OgaConfig::Create(WriteIncompleteBlockDrafterConfig().string().c_str());
  EXPECT_THROW(config->Overlay(R"({"model":{"dspark":{"filename":"dspark.onnx"}}})"),
               std::runtime_error);
}

TEST(Dflash2ConfigTest, ProjectsDrafterWithoutTargetState) {
  const auto projected = CreateDflash2Config(MakeDflash2Config());
  const auto& decoder = projected->model.decoder;
  EXPECT_EQ(decoder.filename, "dflash2.onnx");
  EXPECT_EQ(decoder.num_hidden_layers, 3);
  EXPECT_EQ(decoder.num_key_value_heads, 2);
  EXPECT_EQ(decoder.head_size, 8);
  EXPECT_FALSE(decoder.sliding_window.has_value());
  EXPECT_FALSE(decoder.state_groups.has_value());
}

TEST(Dflash2ConfigTest, AccountsForWindowedPagedCache) {
  const auto config = MakeDflash2Config();
  EXPECT_EQ(Dflash2Drafter::BytesPerBlock(
                config, 16, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            3072u);
  EXPECT_EQ(Dflash2Drafter::PoolBlocks(config, 8, 3), 15u);
  EXPECT_EQ(Dflash2Drafter::PoolBytes(
                config, 8, 15, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            23040u);
  EXPECT_EQ(Dflash2Drafter::PrefixCheckpointBytes(
                config, 8, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
            7680u);
}

TEST(Dflash2ConfigTest, HostCheckpointDoesNotConsumeDeviceBudget) {
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite");
  auto& batching = *model->config_->engine.dynamic_batching;
  batching.prefix_caching = true;
  batching.num_blocks = 1;
  const size_t block_bytes = PagedKeyValueCacheBytesPerBlock(model);
  batching.snapshot_cache.ram_bytes = 2 * block_bytes;
  auto manager = CacheManager::Create(model, 0, 0, 2 * block_bytes);
  EXPECT_EQ(manager->DraftCheckpointCapacity(), 1u);
  EXPECT_EQ(manager->DraftCheckpointVramBytes(), 0u);
  EXPECT_EQ(manager->Snapshot().total_blocks, 1u);
  EXPECT_THROW(CacheManager::Create(model, 0, block_bytes, 2 * block_bytes),
               std::runtime_error);
}

TEST(Dflash2ConfigTest, RejectsWindowedPoolByteOverflow) {
  const auto config = MakeDflash2Config();
  EXPECT_THROW(
      Dflash2Drafter::PoolBytes(
          config, 8, std::numeric_limits<size_t>::max(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16),
      std::runtime_error);
}

TEST(Dflash2ConfigTest, BillsFullAttentionCachePerTargetBlock) {
  auto config = MakeDflash2Config();
  config.model.dflash2.is_dspark = true;
  config.model.dflash2.num_draft_tokens = config.model.dflash2.block_size;
  config.model.dflash2.sliding_window = 0;
  EXPECT_EQ(Dflash2Drafter::PoolBlocks(config, 8, 3), 0u);
  EXPECT_EQ(Dflash2Drafter::BytesPerBlock(
                config, 8, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT),
            3072u);
}

TEST(Dflash2ConfigTest, ReservesEveryFullAttentionQuerySpillBlock) {
  EXPECT_EQ(Dflash2Drafter::FullAttentionPoolBlocks(100, 4, 4, 3), 103u);
  EXPECT_EQ(Dflash2Drafter::FullAttentionPoolBlocks(100, 4, 5, 3), 106u);
  EXPECT_EQ(Dflash2Drafter::FullAttentionPoolBlocks(100, 4, 9, 3), 109u);
  EXPECT_EQ(Dflash2Drafter::FullAttentionReservedBytes(4, 9, 3, 128), 1152u);
}

TEST(Dflash2ConfigTest, RejectsFullAttentionPoolOverflow) {
  EXPECT_THROW(
      Dflash2Drafter::FullAttentionPoolBlocks(
          std::numeric_limits<size_t>::max(), 4, 4, 1),
      std::runtime_error);
  EXPECT_THROW(
      Dflash2Drafter::FullAttentionReservedBytes(
          4, 4, 2, std::numeric_limits<size_t>::max()),
      std::runtime_error);
}

TEST(Dflash2ConfigTest, RunsFullAttentionDsparkAcrossRequestLifecycles) {
  auto config = MakeDflash2Config();
  config.config_path = fs::path{MODEL_PATH "engine/synthetic-dspark"};
  auto& dspark = config.model.dflash2;
  dspark.filename = "dspark.onnx";
  dspark.is_dspark = true;
  dspark.num_hidden_layers = 1;
  dspark.num_key_value_heads = 1;
  dspark.head_size = 1;
  dspark.block_size = 4;
  dspark.num_draft_tokens = 4;
  dspark.sliding_window = 0;

  auto model = std::make_shared<Dflash2Model>(CreateDflash2Config(config), GetOrtEnv());
  Dflash2Drafter drafter{model, /*paged_block_size=*/4, /*num_blocks=*/10,
                         /*max_requests=*/2};

  int request_a_id = 0;
  int request_b_id = 0;
  int request_d_id = 0;
  int request_e_id = 0;
  auto* request_a = reinterpret_cast<Request*>(&request_a_id);
  auto* request_b = reinterpret_cast<Request*>(&request_b_id);
  auto* request_d = reinterpret_cast<Request*>(&request_d_id);
  auto* request_e = reinterpret_cast<Request*>(&request_e_id);

  auto* device = GetDeviceInterface(DeviceType::CPU);
  Tensor first_aux{device, Ort::TypeToTensorType<float>};
  const std::array<int64_t, 2> first_aux_shape{18, 1};
  first_aux.CreateTensor(first_aux_shape);
  const std::array first_feeds{
      Dflash2Drafter::Feed{.request = request_a, .aux_row_begin = 0, .aux_row_count = 9, .first_position = 0, .anchor_token = 11, .draft_eligible = true, .wants_drafts = true},
      Dflash2Drafter::Feed{.request = request_b, .aux_row_begin = 9, .aux_row_count = 9, .first_position = 0, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true},
  };
  std::vector<std::vector<int32_t>> drafts;
  drafter.Propose(first_aux, first_feeds, drafts);
  ASSERT_EQ(drafts.size(), 2u);
  EXPECT_EQ(drafts[0], (std::vector<int32_t>{6, 0, 13, 64}));
  EXPECT_EQ(drafts[1], (std::vector<int32_t>{22, 0, 13, 64}));

  Tensor growth_aux{device, Ort::TypeToTensorType<float>};
  const std::array<int64_t, 2> growth_aux_shape{8, 1};
  growth_aux.CreateTensor(growth_aux_shape);
  const std::array growth_feeds{
      Dflash2Drafter::Feed{.request = request_a, .aux_row_begin = 0, .aux_row_count = 4, .first_position = 9, .anchor_token = 13, .draft_eligible = true, .wants_drafts = true},
      Dflash2Drafter::Feed{.request = request_b, .aux_row_begin = 4, .aux_row_count = 4, .first_position = 9, .anchor_token = 14, .draft_eligible = true, .wants_drafts = true},
  };
  drafter.Propose(growth_aux, growth_feeds, drafts);
  ASSERT_EQ(drafts.size(), 2u);
  EXPECT_EQ(drafts[0], (std::vector<int32_t>{14, 9, 8, 44}));
  EXPECT_EQ(drafts[1], (std::vector<int32_t>{31, 9, 8, 44}));

  // Rewind releases only the selected request. It can replay from position zero while the peer
  // continues from its existing drafter state.
  drafter.Release(request_a);
  Tensor reused_aux{device, Ort::TypeToTensorType<float>};
  const std::array<int64_t, 2> reused_aux_shape{10, 1};
  reused_aux.CreateTensor(reused_aux_shape);
  const std::array reused_feeds{
      Dflash2Drafter::Feed{.request = request_a, .aux_row_begin = 0, .aux_row_count = 9, .first_position = 0, .anchor_token = 15, .draft_eligible = true, .wants_drafts = true},
      Dflash2Drafter::Feed{.request = request_b, .aux_row_begin = 9, .aux_row_count = 1, .first_position = 13, .anchor_token = 16, .draft_eligible = true, .wants_drafts = true},
  };
  drafter.Propose(reused_aux, reused_feeds, drafts);
  ASSERT_EQ(drafts.size(), 2u);
  EXPECT_EQ(drafts[0], (std::vector<int32_t>{13, 0, 13, 32}));
  EXPECT_EQ(drafts[1], (std::vector<int32_t>{31, 13, 5, 32}));
  EXPECT_EQ(drafter.AdmissionMisses(), 0u);

  Tensor failed_aux{device, Ort::TypeToTensorType<float>};
  const std::array<int64_t, 2> failed_aux_shape{1, 1};
  failed_aux.CreateTensor(failed_aux_shape);
  const std::array failed_feed{
      Dflash2Drafter::Feed{.request = request_a, .aux_row_begin = 0, .aux_row_count = 1, .first_position = 10, .anchor_token = 17, .draft_eligible = true, .wants_drafts = true},
  };
  EXPECT_THROW(drafter.Propose(failed_aux, failed_feed, drafts), std::logic_error);
  drafter.ReleaseAll();

  const std::array recovered_feeds{
      Dflash2Drafter::Feed{.request = request_d, .aux_row_begin = 0, .aux_row_count = 9, .first_position = 0, .anchor_token = 17, .draft_eligible = true, .wants_drafts = true},
      Dflash2Drafter::Feed{.request = request_e, .aux_row_begin = 9, .aux_row_count = 9, .first_position = 0, .anchor_token = 18, .draft_eligible = true, .wants_drafts = true},
  };
  drafter.Propose(first_aux, recovered_feeds, drafts);
  ASSERT_EQ(drafts.size(), 2u);
  EXPECT_EQ(drafts[0][1], 0);
  EXPECT_EQ(drafts[0][2], 13);
  EXPECT_EQ(drafts[0][3], 64);
  EXPECT_EQ(drafts[1][1], 0);
  EXPECT_EQ(drafts[1][2], 13);
  EXPECT_EQ(drafts[1][3], 64);
  EXPECT_EQ(drafter.AdmissionMisses(), 0u);
}

TEST(Dflash2ConfigTest, RunsPackedQkvDrafterWithoutQueryRowMap) {
  auto config = MakeDflash2Config();
  config.config_path = fs::path{MODEL_PATH "engine/synthetic-dspark"};
  auto& dspark = config.model.dflash2;
  dspark.filename = "dspark_packed_qkv.onnx";
  dspark.is_dspark = true;
  dspark.num_hidden_layers = 1;
  dspark.num_key_value_heads = 1;
  dspark.head_size = 1;
  dspark.block_size = 4;
  dspark.num_draft_tokens = 4;
  dspark.sliding_window = 0;

  auto model = std::make_shared<Dflash2Model>(CreateDflash2Config(config), GetOrtEnv());
  ASSERT_FALSE(model->session_info_.HasInput(dspark.inputs.q_row_map));
  Dflash2Drafter drafter{model, /*paged_block_size=*/4, /*num_blocks=*/10,
                         /*max_requests=*/2};

  int request_a_id = 0;
  int request_b_id = 0;
  auto* request_a = reinterpret_cast<Request*>(&request_a_id);
  auto* request_b = reinterpret_cast<Request*>(&request_b_id);
  Tensor aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  aux.CreateTensor(std::array<int64_t, 2>{18, 1});
  const std::array feeds{
      Dflash2Drafter::Feed{.request = request_a, .aux_row_begin = 0, .aux_row_count = 9, .first_position = 0, .anchor_token = 11, .draft_eligible = true, .wants_drafts = true},
      Dflash2Drafter::Feed{.request = request_b, .aux_row_begin = 9, .aux_row_count = 9, .first_position = 0, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true},
  };
  std::vector<std::vector<int32_t>> drafts;
  ASSERT_TRUE(drafter.Propose(aux, feeds, drafts));
  ASSERT_EQ(drafts.size(), 2u);
  // The last column sums qkv_row_map, a permutation of the 26 packed block and context rows.
  EXPECT_EQ(drafts[0], (std::vector<int32_t>{6, 0, 13, 325}));
  EXPECT_EQ(drafts[1], (std::vector<int32_t>{22, 0, 13, 325}));
}

class Dflash2StorageRestoreTest : public ::testing::TestWithParam<bool> {};

TEST(Dflash2ConfigTest, RunsDrafterWithAnchorRowIndex) {
  const auto root = WriteAnchorRowIndexDrafterModel();
  auto config = MakeDflash2Config();
  config.config_path = fs::path{root.string()};
  auto& draft = config.model.dflash2;
  draft.filename = "dflash2_anchor.onnx";
  draft.num_hidden_layers = 1;
  draft.num_key_value_heads = 1;
  draft.head_size = 1;
  draft.block_size = 4;
  draft.num_draft_tokens = 3;
  draft.selector_top_k = 2;
  draft.sliding_window = 4;  // context_window_ = sliding_window + block_size = 8.
  draft.inputs.anchor_row_index = "anchor_row_index";

  auto model = std::make_shared<Dflash2Model>(CreateDflash2Config(config), GetOrtEnv());
  ASSERT_TRUE(model->session_info_.HasInput(draft.inputs.anchor_row_index));
  FakeModelStateMetadata target;
  target.AddOutput("aux_hidden_states", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 1});
  EXPECT_EQ(ValidateDflash2ModelCompatibility(config, target, model->session_info_, 4),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  const size_t ring_blocks = Dflash2Drafter::PoolBlocks(config, 4, 3);
  Dflash2Drafter drafter{model, /*paged_block_size=*/4, ring_blocks, /*max_requests=*/3};

  int req_a_id = 0, req_b_id = 0, req_c_id = 0;
  auto* req_a = reinterpret_cast<Request*>(&req_a_id);
  auto* req_b = reinterpret_cast<Request*>(&req_b_id);
  auto* req_c = reinterpret_cast<Request*>(&req_c_id);
  auto* device = GetDeviceInterface(DeviceType::CPU);

  auto make_aux = [&](std::initializer_list<float> values) {
    auto aux = std::make_unique<Tensor>(device, Ort::TypeToTensorType<float>);
    aux->CreateTensor(std::array<int64_t, 2>{static_cast<int64_t>(values.size()), 1});
    auto span = aux->GetDeviceSpan<float>();
    std::copy(values.begin(), values.end(), span.CpuSpan().begin());
    span.CopyCpuToDevice();
    return aux;
  };
  std::vector<std::vector<int32_t>> drafts;

  // 1. Per-query-block ordering for unequal context lengths: req_a ingests 3 rows (packed 0..2,
  //    anchor row 2 = 102) and req_b ingests 5 rows (packed 3..7, anchor row 7 = 107).
  {
    auto aux = make_aux({100.0f, 101.0f, 102.0f, 103.0f, 104.0f, 105.0f, 106.0f, 107.0f});
    const std::array feeds{
        Dflash2Drafter::Feed{.request = req_a, .aux_row_begin = 0, .aux_row_count = 3, .first_position = 0, .anchor_token = 11, .draft_eligible = true, .wants_drafts = true},
        Dflash2Drafter::Feed{.request = req_b, .aux_row_begin = 3, .aux_row_count = 5, .first_position = 0, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true},
    };
    ASSERT_TRUE(drafter.Propose(*aux, feeds, drafts));
    ASSERT_EQ(drafts.size(), 2u);
    EXPECT_EQ(drafts[0], (std::vector<int32_t>{2, 102, 0}));
    EXPECT_EQ(drafts[1], (std::vector<int32_t>{7, 107, 0}));
  }

  // 2. Window trimming: req_a supplies 10 rows (> context_window_ = 8), so its leading 2 rows
  //    (200, 201) are dropped and packed rows 0..7 hold 202..209 (anchor row 7 = 209); req_b's
  //    2 rows then pack into 8..9 (anchor row 9 = 211).
  {
    auto aux = make_aux({200.0f, 201.0f, 202.0f, 203.0f, 204.0f, 205.0f,
                         206.0f, 207.0f, 208.0f, 209.0f, 210.0f, 211.0f});
    const std::array feeds{
        Dflash2Drafter::Feed{.request = req_a, .aux_row_begin = 0, .aux_row_count = 10, .first_position = 3, .anchor_token = 13, .draft_eligible = true, .wants_drafts = true},
        Dflash2Drafter::Feed{.request = req_b, .aux_row_begin = 10, .aux_row_count = 2, .first_position = 5, .anchor_token = 14, .draft_eligible = true, .wants_drafts = true},
    };
    ASSERT_TRUE(drafter.Propose(*aux, feeds, drafts));
    ASSERT_EQ(drafts.size(), 2u);
    EXPECT_EQ(drafts[0], (std::vector<int32_t>{7, 209, 102}));
    EXPECT_EQ(drafts[1], (std::vector<int32_t>{9, 211, 102}));
  }

  // 3. Mixed context-only and drafting feeds: req_a (2 rows, packed 0..1) and req_c (1 row,
  //    packed 5) ingest context without drafting, while req_b (3 rows, packed 2..4) drafts from
  //    packed anchor row 4 (= 304).
  {
    auto aux = make_aux({300.0f, 301.0f, 302.0f, 303.0f, 304.0f, 305.0f});
    const std::array feeds{
        Dflash2Drafter::Feed{.request = req_a, .aux_row_begin = 0, .aux_row_count = 2, .first_position = 13, .anchor_token = 15, .draft_eligible = false, .wants_drafts = false},
        Dflash2Drafter::Feed{.request = req_b, .aux_row_begin = 2, .aux_row_count = 3, .first_position = 7, .anchor_token = 16, .draft_eligible = true, .wants_drafts = true},
        Dflash2Drafter::Feed{.request = req_c, .aux_row_begin = 5, .aux_row_count = 1, .first_position = 0, .anchor_token = 17, .draft_eligible = true, .wants_drafts = false},
    };
    ASSERT_TRUE(drafter.Propose(*aux, feeds, drafts));
    ASSERT_EQ(drafts.size(), 3u);
    EXPECT_TRUE(drafts[0].empty());
    EXPECT_EQ(drafts[1], (std::vector<int32_t>{4, 304, 209}));
    EXPECT_TRUE(drafts[2].empty());
  }

  // 4. Borrowed placeholder's row-0 fallback: when no feed wants drafts and the first served feed
  //    (req_a) carries 0 context rows, the borrowed query block falls back to packed row 0 (400)
  //    of req_b's context rows (400, 401), which the graph records in present.0.key for the next
  //    drafting step to read back in slot 2.
  {
    auto aux = make_aux({400.0f, 401.0f});
    const std::array feeds{
        Dflash2Drafter::Feed{.request = req_a, .aux_row_begin = 0, .aux_row_count = 0, .first_position = 15, .anchor_token = 18, .draft_eligible = false, .wants_drafts = false},
        Dflash2Drafter::Feed{.request = req_b, .aux_row_begin = 0, .aux_row_count = 2, .first_position = 10, .anchor_token = 19, .draft_eligible = false, .wants_drafts = false},
    };
    ASSERT_TRUE(drafter.Propose(*aux, feeds, drafts));
    ASSERT_EQ(drafts.size(), 2u);
    EXPECT_TRUE(drafts[0].empty());
    EXPECT_TRUE(drafts[1].empty());

    auto next_aux = make_aux({500.0f});
    const std::array next_feeds{
        Dflash2Drafter::Feed{.request = req_a, .aux_row_begin = 0, .aux_row_count = 1, .first_position = 15, .anchor_token = 20, .draft_eligible = true, .wants_drafts = true},
    };
    ASSERT_TRUE(drafter.Propose(*next_aux, next_feeds, drafts));
    ASSERT_EQ(drafts.size(), 1u);
    EXPECT_EQ(drafts[0], (std::vector<int32_t>{0, 500, 400}));
  }

  // A real query block without a context row (or a placeholder step with zero total context rows)
  // is rejected rather than indexing out of bounds.
  {
    auto aux = make_aux({600.0f});
    const std::array missing_draft_context{
        Dflash2Drafter::Feed{.request = req_a, .aux_row_begin = 0, .aux_row_count = 0, .first_position = 16, .anchor_token = 21, .draft_eligible = true, .wants_drafts = true},
        Dflash2Drafter::Feed{.request = req_b, .aux_row_begin = 0, .aux_row_count = 1, .first_position = 12, .anchor_token = 22, .draft_eligible = false, .wants_drafts = false},
    };
    EXPECT_THROW(drafter.Propose(*aux, missing_draft_context, drafts), std::logic_error);

    const std::array zero_total_context{
        Dflash2Drafter::Feed{.request = req_a, .aux_row_begin = 0, .aux_row_count = 0, .first_position = 16, .anchor_token = 21, .draft_eligible = false, .wants_drafts = false},
    };
    EXPECT_THROW(drafter.Propose(*aux, zero_total_context, drafts), std::logic_error);
  }
}

TEST_P(Dflash2StorageRestoreTest, ExecutesWindowedRestoreAcrossRingWrap) {
  auto config = MakeDflash2Config();
  if (GetParam()) {
    config.engine.dynamic_batching->snapshot_cache.ram_bytes = 0;
    config.engine.dynamic_batching->snapshot_cache.disk_bytes = 1024 * 1024;
  }
  config.config_path = fs::path{MODEL_PATH "engine/synthetic-dspark"};
  auto& draft = config.model.dflash2;
  draft.filename = "dflash2.onnx";
  draft.num_hidden_layers = 1;
  draft.num_key_value_heads = 1;
  draft.head_size = 1;
  draft.block_size = 4;
  draft.num_draft_tokens = 3;
  draft.selector_top_k = 2;
  draft.sliding_window = 8;

  auto model = std::make_shared<Dflash2Model>(CreateDflash2Config(config), GetOrtEnv());
  const size_t ring_blocks = Dflash2Drafter::PoolBlocks(config, 4, 1);
  Dflash2Drafter drafter{model, /*paged_block_size=*/4, ring_blocks * 2,
                         /*max_requests=*/2};
  drafter.InitializeSnapshotCache(MakeSnapshotCache(config));
  int source_id = 0, restored_id = 0, peer_id = 0, extra_id = 0, remap_id = 0;
  auto* source = reinterpret_cast<Request*>(&source_id);
  auto* restored = reinterpret_cast<Request*>(&restored_id);
  auto* peer = reinterpret_cast<Request*>(&peer_id);
  auto* extra = reinterpret_cast<Request*>(&extra_id);
  auto* remap = reinterpret_cast<Request*>(&remap_id);
  Tensor aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  aux.CreateTensor(std::array<int64_t, 2>{8, 1});
  std::vector<std::vector<int32_t>> proposals;

  for (size_t position = 0; position < 24; position += 8) {
    const std::array feeds{Dflash2Drafter::Feed{
        .request = source, .aux_row_count = 8, .first_position = position, .anchor_token = 11, .draft_eligible = true, .wants_drafts = true}};
    ASSERT_TRUE(drafter.Propose(aux, feeds, proposals));
  }
  model->config_->engine.dynamic_batching->prefix_caching.reset();
  auto checkpoint = drafter.CapturePrefix(source, 24);
  ASSERT_NE(checkpoint, nullptr);
  model->config_->engine.dynamic_batching->prefix_caching = false;
  EXPECT_EQ(drafter.CapturePrefix(source, 24), nullptr);
  model->config_->engine.dynamic_batching->prefix_caching.reset();
  EXPECT_EQ(checkpoint->ring_blocks, ring_blocks);
  Tensor next_aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  next_aux.CreateTensor(std::array<int64_t, 2>{1, 1});
  next_aux.GetByteSpan().Zero();
  const std::array target_only{Dflash2Drafter::Feed{
      .request = peer, .aux_row_count = 1, .first_position = 24, .anchor_token = 13, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(next_aux, target_only, proposals));
  EXPECT_TRUE(proposals.front().empty());
  EXPECT_TRUE(drafter.CanCapturePrefix(source, 24));
  EXPECT_FALSE(drafter.CanCapturePrefix(source, 23));
  EXPECT_FALSE(drafter.CanCapturePrefix(peer, 25));
  std::weak_ptr<const Dflash2PrefixCheckpoint> indexed_checkpoint = checkpoint;
  if (drafter.CanCapturePrefix(peer, 25)) {
    checkpoint.reset();
  }
  ASSERT_FALSE(indexed_checkpoint.expired());
  drafter.Release(peer);

  const std::array uninterrupted{Dflash2Drafter::Feed{
      .request = source, .aux_row_count = 1, .first_position = 24, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(next_aux, uninterrupted, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  const auto expected = proposals.front();
  ASSERT_FALSE(expected.empty());
  EXPECT_EQ(expected.front(), 24);

  drafter.Release(source);
  const std::array remap_feed{Dflash2Drafter::Feed{
      .request = remap, .aux_row_count = 1, .anchor_token = 13, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(next_aux, remap_feed, proposals));
  const std::array resumed{Dflash2Drafter::Feed{
      .request = restored, .prefix_checkpoint = checkpoint, .aux_row_count = 1, .first_position = 24, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(next_aux, resumed, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  EXPECT_EQ(proposals.front(), expected);
  EXPECT_EQ(drafter.AdmissionMisses(), 0u);

  drafter.Release(remap);
  drafter.Release(restored);
  ASSERT_TRUE(drafter.Propose(next_aux, resumed, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  EXPECT_EQ(proposals.front(), expected);
  drafter.Release(restored);

  auto wrong_position = resumed;
  wrong_position.front().first_position = 23;
  const std::array with_peer{
      wrong_position.front(),
      Dflash2Drafter::Feed{
          .request = peer, .aux_row_count = 1, .anchor_token = 13, .draft_eligible = true, .wants_drafts = true}};
  EXPECT_TRUE(drafter.Propose(next_aux, with_peer, proposals));
  EXPECT_TRUE(proposals.front().empty());
  EXPECT_FALSE(proposals.back().empty());
  const std::array second_peer{Dflash2Drafter::Feed{
      .request = extra, .aux_row_count = 1, .anchor_token = 14, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(next_aux, second_peer, proposals));
  const std::array full_pool{
      Dflash2Drafter::Feed{
          .request = peer, .aux_row_count = 1, .first_position = 1, .anchor_token = 15, .draft_eligible = true, .wants_drafts = true},
      resumed.front()};
  EXPECT_TRUE(drafter.Propose(next_aux, full_pool, proposals));
  EXPECT_FALSE(proposals.front().empty());
  EXPECT_TRUE(proposals.back().empty());
  EXPECT_EQ(drafter.AdmissionMisses(), 1u);

  drafter.ReleaseAll();
  ASSERT_TRUE(drafter.Propose(next_aux, resumed, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  EXPECT_FALSE(proposals.front().empty());

  drafter.ReleaseAll();
  auto alternate = resumed;
  alternate.front().request = remap;
  alternate.front().anchor_token = 13;
  alternate.front().aux_row_count = 2;
  Tensor alternate_aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  alternate_aux.CreateTensor(std::array<int64_t, 2>{2, 1});
  alternate_aux.GetByteSpan().Zero();
  ASSERT_TRUE(drafter.Propose(alternate_aux, alternate, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  const auto alternate_expected = proposals.front();
  ASSERT_FALSE(alternate_expected.empty());
  ASSERT_NE(alternate_expected, expected);
  drafter.ReleaseAll();

  Tensor paired_aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  paired_aux.CreateTensor(std::array<int64_t, 2>{3, 1});
  paired_aux.GetByteSpan().Zero();
  auto paired_alternate = alternate.front();
  paired_alternate.aux_row_begin = 1;
  const std::array paired{resumed.front(), paired_alternate};
  ASSERT_TRUE(drafter.Propose(paired_aux, paired, proposals));
  ASSERT_EQ(proposals.size(), 2u);
  EXPECT_EQ(proposals[0], expected);
  EXPECT_EQ(proposals[1], alternate_expected);
  drafter.ReleaseAll();

  auto missing_state = std::make_shared<Dflash2PrefixCheckpoint>();
  missing_state->token_count = checkpoint->token_count;
  missing_state->ring_blocks = checkpoint->ring_blocks;
  missing_state->shapes = checkpoint->shapes;
  missing_state->type = checkpoint->type;
  auto damaged_cache = MakeSnapshotCache(config);
  ASSERT_TRUE(damaged_cache->Allocate(missing_state, model->p_device_kvcache_));
  for (size_t layer = 0; layer < checkpoint->shapes.size(); ++layer) {
    Tensor copy{GetDeviceInterface(DeviceType::CPU), checkpoint->type};
    copy.CreateTensor(checkpoint->shapes[layer]);
    checkpoint->Read(layer, 0, copy, 0, copy.GetByteSpan().size());
    if (layer == 0) copy.GetByteSpan().Zero();
    missing_state->Write(layer, 0, copy, 0, copy.GetByteSpan().size());
  }
  drafter.ReleaseAll();
  auto damaged = resumed;
  damaged.front().prefix_checkpoint = missing_state;
  ASSERT_TRUE(drafter.Propose(next_aux, damaged, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  ASSERT_FALSE(proposals.front().empty());
  EXPECT_NE(proposals.front().front(), expected.front());

  drafter.ReleaseAll();
  ++missing_state->shapes.front().back();
  EXPECT_THROW(drafter.Propose(next_aux, damaged, proposals), std::logic_error);

  auto invalid = std::make_shared<Dflash2PrefixCheckpoint>();
  invalid->token_count = checkpoint->token_count;
  invalid->ring_blocks = checkpoint->ring_blocks;
  damaged.front().prefix_checkpoint = std::move(invalid);
  EXPECT_FALSE(drafter.Propose(next_aux, damaged, proposals));
  EXPECT_TRUE(proposals.front().empty());
}

INSTANTIATE_TEST_SUITE_P(RamAndDisk, Dflash2StorageRestoreTest, ::testing::Bool());

TEST(Dflash2ConfigTest, SnapshotRestoreIoFailurePreservesHealthyDraftContextAndAllowsRetry) {
  auto config = MakeDflash2Config();
  config.config_path = fs::path{MODEL_PATH "engine/synthetic-dspark"};
  auto& draft = config.model.dflash2;
  draft.filename = "dflash2.onnx";
  draft.num_hidden_layers = 1;
  draft.num_key_value_heads = 1;
  draft.head_size = 1;
  draft.block_size = 4;
  draft.num_draft_tokens = 3;
  draft.selector_top_k = 2;
  draft.sliding_window = 8;
  ScopedSnapshotDirectory test_directory{"oga-dflash-io"};
  const auto& parent = test_directory.Path();
  auto& limits = config.engine.dynamic_batching->snapshot_cache;
  limits.ram_bytes = 0;
  limits.disk_bytes = 1024 * 1024;
  limits.directory = parent.string();
  auto model = std::make_shared<Dflash2Model>(CreateDflash2Config(config), GetOrtEnv());
  const size_t ring = Dflash2Drafter::PoolBlocks(config, 4, 1);
  auto storage = MakeSnapshotCache(config);
  Dflash2Drafter drafter{model, 4, ring * 2, 2};
  drafter.InitializeSnapshotCache(storage);
  int source_id = 0, restored_id = 0;
  auto* source = reinterpret_cast<Request*>(&source_id);
  auto* restored = reinterpret_cast<Request*>(&restored_id);
  Tensor aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  aux.CreateTensor(std::array<int64_t, 2>{8, 1});
  aux.GetByteSpan().Zero();
  std::vector<std::vector<int32_t>> proposals;
  const std::array initial{Dflash2Drafter::Feed{
      .request = source, .aux_row_count = 8, .anchor_token = 11, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(aux, initial, proposals));
  auto checkpoint = drafter.CapturePrefix(source, 8);
  ASSERT_NE(checkpoint, nullptr);
  const auto directory = fs_std::directory_iterator(parent)->path();
  fs_std::resize_file(directory / "payload", 0);

  Tensor next_aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  next_aux.CreateTensor(std::array<int64_t, 2>{1, 1});
  next_aux.GetByteSpan().Zero();
  const std::array failed_restore{Dflash2Drafter::Feed{
      .request = restored, .prefix_checkpoint = checkpoint, .aux_row_count = 1, .first_position = 8, .anchor_token = 13, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(next_aux, failed_restore, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  EXPECT_TRUE(proposals.front().empty());
  EXPECT_TRUE(drafter.CanCapturePrefix(source, 8));
  EXPECT_EQ(checkpoint->state, PrefixSnapshotCache::State::Evicted);
  const std::array healthy{Dflash2Drafter::Feed{
      .request = source, .aux_row_count = 1, .first_position = 8, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(next_aux, healthy, proposals));
  ASSERT_EQ(proposals.size(), 1u);
  EXPECT_FALSE(proposals.front().empty());
  EXPECT_TRUE(drafter.CanCapturePrefix(source, 9));
  auto retried = drafter.CapturePrefix(source, 9);
  ASSERT_NE(retried, nullptr);
  retried.reset();
  EXPECT_TRUE(fs_std::remove(parent));
}

TEST(Dflash2ConfigTest, MissingWindowedPrefixRebuildsBeforeDraftingOrCheckpointing) {
  auto config = MakeDflash2Config();
  config.config_path = fs::path{MODEL_PATH "engine/synthetic-dspark"};
  auto& draft = config.model.dflash2;
  draft.filename = "dflash2.onnx";
  draft.num_hidden_layers = 1;
  draft.num_key_value_heads = 1;
  draft.head_size = 1;
  draft.block_size = 4;
  draft.num_draft_tokens = 3;
  draft.selector_top_k = 2;
  draft.sliding_window = 8;
  auto model = std::make_shared<Dflash2Model>(CreateDflash2Config(config), GetOrtEnv());
  Dflash2Drafter drafter{model, 4, Dflash2Drafter::PoolBlocks(config, 4, 1), 1};
  drafter.InitializeSnapshotCache(MakeSnapshotCache(config));
  int request_id = 0;
  auto* request = reinterpret_cast<Request*>(&request_id);
  Tensor aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  aux.CreateTensor(std::array<int64_t, 2>{4, 1});
  aux.GetByteSpan().Zero();
  std::vector<std::vector<int32_t>> proposals;
  for (size_t position = 24; position < 36; position += 4) {
    const std::array feeds{Dflash2Drafter::Feed{
        .request = request, .aux_row_count = 4, .first_position = position, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true}};
    ASSERT_TRUE(drafter.Propose(aux, feeds, proposals));
    ASSERT_EQ(proposals.size(), 1u);
    if (position < 32) {
      EXPECT_TRUE(proposals.front().empty());
      EXPECT_FALSE(drafter.CanCapturePrefix(request, position + 4));
      EXPECT_EQ(drafter.CapturePrefix(request, position + 4), nullptr);
    } else {
      EXPECT_FALSE(proposals.front().empty());
      EXPECT_TRUE(drafter.CanCapturePrefix(request, position + 4));
      EXPECT_NE(drafter.CapturePrefix(request, position + 4), nullptr);
    }
  }
  drafter.ReleaseAll();
  const std::array restarted{Dflash2Drafter::Feed{
      .request = request, .aux_row_count = 4, .first_position = 64, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(aux, restarted, proposals));
  EXPECT_TRUE(proposals.front().empty());
  EXPECT_FALSE(drafter.CanCapturePrefix(request, 68));
}

TEST(Dflash2ConfigTest, CapturesAndRestoresExtendedWindowedPrefix) {
  auto config = MakeDflash2Config();
  config.config_path = fs::path{MODEL_PATH "engine/synthetic-dspark"};
  auto& draft = config.model.dflash2;
  draft.filename = "dflash2.onnx";
  draft.num_hidden_layers = 1;
  draft.num_key_value_heads = 1;
  draft.head_size = 1;
  draft.block_size = 4;
  draft.num_draft_tokens = 3;
  draft.selector_top_k = 2;
  draft.sliding_window = 8;

  auto model = std::make_shared<Dflash2Model>(CreateDflash2Config(config), GetOrtEnv());
  const size_t ring_blocks = Dflash2Drafter::PoolBlocks(config, 4, 1);
  Dflash2Drafter drafter{model, /*paged_block_size=*/4, ring_blocks * 2, /*max_requests=*/2};
  drafter.InitializeSnapshotCache(MakeSnapshotCache(config));
  int source_id = 0, replay_id = 0;
  auto* source = reinterpret_cast<Request*>(&source_id);
  auto* replay = reinterpret_cast<Request*>(&replay_id);
  Tensor aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  aux.CreateTensor(std::array<int64_t, 2>{8, 1});
  std::vector<std::vector<int32_t>> proposals;

  for (size_t position = 0; position < 24; position += 8) {
    const std::array feeds{Dflash2Drafter::Feed{
        .request = source, .aux_row_count = 8, .first_position = position, .anchor_token = 11, .draft_eligible = true, .wants_drafts = true}};
    ASSERT_TRUE(drafter.Propose(aux, feeds, proposals));
  }
  auto initial = drafter.CapturePrefix(source, 24);
  ASSERT_NE(initial, nullptr);
  initial.reset();
  const std::array extension{Dflash2Drafter::Feed{
      .request = source, .aux_row_count = 8, .first_position = 24, .anchor_token = 11, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(aux, extension, proposals));
  auto extended = drafter.CapturePrefix(source, 32);
  ASSERT_NE(extended, nullptr);

  Tensor next_aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  next_aux.CreateTensor(std::array<int64_t, 2>{1, 1});
  next_aux.GetByteSpan().Zero();
  const std::array uninterrupted{Dflash2Drafter::Feed{
      .request = source, .aux_row_count = 1, .first_position = 32, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(next_aux, uninterrupted, proposals));
  const auto expected = proposals.front();
  ASSERT_FALSE(expected.empty());

  drafter.Release(source);
  const std::array restored{Dflash2Drafter::Feed{
      .request = replay, .prefix_checkpoint = extended, .aux_row_count = 1, .first_position = 32, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true}};
  ASSERT_TRUE(drafter.Propose(next_aux, restored, proposals));
  EXPECT_EQ(proposals.front(), expected);
  drafter.Release(replay);
  ASSERT_TRUE(drafter.Propose(next_aux, restored, proposals));
  EXPECT_EQ(proposals.front(), expected);
}

TEST(Dflash2ConfigTest, CapturesUniformShapesWithinBlockTableLimit) {
  EXPECT_TRUE(Dflash2GraphCaptureAllowed(true, true, 8, 8));
  EXPECT_FALSE(Dflash2GraphCaptureAllowed(true, false, 8, 8));
  EXPECT_FALSE(Dflash2GraphCaptureAllowed(false, true, 8, 8));
  EXPECT_FALSE(Dflash2GraphCaptureAllowed(true, true, 9, 8));
}

TEST(Dflash2ConfigTest, PinsTheBlockTableWidthOnlyWhenOptedInOrDeviceSelected) {
  // Generic Engine drafters (even with max_requests == 1) keep bucketed widths unless opted in.
  EXPECT_FALSE(Dflash2PinsBlockTableWidth(/*capture_enabled=*/true, /*windowed=*/true,
                                          /*pinned_policy=*/false, /*device_selection=*/false));
  EXPECT_TRUE(Dflash2PinsBlockTableWidth(true, true, /*pinned_policy=*/true, /*device_selection=*/false));
  EXPECT_TRUE(Dflash2PinsBlockTableWidth(true, true, /*pinned_policy=*/false, /*device_selection=*/true));
  EXPECT_FALSE(Dflash2PinsBlockTableWidth(false, true, true, true));
  EXPECT_FALSE(Dflash2PinsBlockTableWidth(true, false, true, true));

  constexpr size_t kBlock = 16;
  EXPECT_TRUE(Dflash2CapturesPinnedStep(/*drafts_wanted=*/true, /*num_ctx_rows=*/kBlock, kBlock,
                                        /*served=*/1, /*device_selection=*/true));
  EXPECT_FALSE(Dflash2CapturesPinnedStep(false, 1, kBlock, 1, true));
  EXPECT_FALSE(Dflash2CapturesPinnedStep(true, kBlock + 1, kBlock, 1, true));
}

TEST(Dflash2ConfigTest, TrackedDsparkIngestsSampledTurnsAndResumesDrafting) {
  auto config = MakeDflash2Config();
  config.config_path = fs::path{MODEL_PATH "engine/synthetic-dspark"};
  auto& dspark = config.model.dflash2;
  dspark.filename = "dspark.onnx";
  dspark.is_dspark = true;
  dspark.num_hidden_layers = 1;
  dspark.num_key_value_heads = 1;
  dspark.head_size = 1;
  dspark.block_size = 4;
  dspark.num_draft_tokens = 4;
  dspark.sliding_window = 0;

  auto model = std::make_shared<Dflash2Model>(CreateDflash2Config(config), GetOrtEnv());
  Dflash2Drafter drafter{model, /*paged_block_size=*/4, /*num_blocks=*/6,
                         /*max_requests=*/1};

  int tracked_id = 0;
  int untracked_id = 0;
  auto* tracked = reinterpret_cast<Request*>(&tracked_id);
  auto* untracked = reinterpret_cast<Request*>(&untracked_id);
  auto* device = GetDeviceInterface(DeviceType::CPU);
  Tensor aux{device, Ort::TypeToTensorType<float>};
  const std::array<int64_t, 2> aux_shape{1, 1};
  aux.CreateTensor(aux_shape);
  std::vector<std::vector<int32_t>> drafts;

  const std::array initial{
      Dflash2Drafter::Feed{.request = tracked, .aux_row_count = 1, .anchor_token = 11, .draft_eligible = true, .wants_drafts = true},
  };
  EXPECT_TRUE(drafter.Propose(aux, initial, drafts));
  ASSERT_EQ(drafts.size(), 1u);
  EXPECT_FALSE(drafts[0].empty());

  const std::array sampled{
      Dflash2Drafter::Feed{.request = tracked, .aux_row_count = 1, .first_position = 1, .anchor_token = 12, .draft_eligible = false, .wants_drafts = false},
  };
  EXPECT_TRUE(drafter.Propose(aux, sampled, drafts));
  ASSERT_EQ(drafts.size(), 1u);
  EXPECT_TRUE(drafts[0].empty());

  const std::array resumed{
      Dflash2Drafter::Feed{.request = tracked, .aux_row_count = 1, .first_position = 2, .anchor_token = 13, .draft_eligible = true, .wants_drafts = true},
  };
  EXPECT_TRUE(drafter.Propose(aux, resumed, drafts));
  ASSERT_EQ(drafts.size(), 1u);
  EXPECT_FALSE(drafts[0].empty());

  drafter.Release(tracked);
  const std::array ineligible{
      Dflash2Drafter::Feed{.request = untracked, .aux_row_count = 1, .anchor_token = 14, .draft_eligible = false, .wants_drafts = false},
  };
  EXPECT_FALSE(drafter.Propose(aux, ineligible, drafts));
  EXPECT_EQ(drafter.AdmissionMisses(), 0u);
  EXPECT_FALSE(drafter.Propose(aux, {}, drafts));
}

TEST(Dflash2ConfigTest, RequiresMatchingCacheTypes) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddInput("past_key_values.1.value", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                   {-1, 8, 2, 8});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresDynamicCachePoolDimension) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddInput("past_key_values.1.value", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                   {32, 8, 2, 8});
  drafter.AddOutput("present.1.value", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
                    {32, 8, 2, 8});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RejectsUnsupportedCacheType) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  for (int layer = 0; layer < 3; ++layer) {
    for (const auto* kind : {"key", "value"}) {
      drafter.AddInput("past_key_values." + std::to_string(layer) + "." + kind,
                       ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, 8, 2, 8});
      drafter.AddOutput("present." + std::to_string(layer) + "." + kind,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, 8, 2, 8});
    }
  }
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresUniqueCacheBindings) {
  auto config = MakeDflash2Config();
  config.model.dflash2.inputs.past_value_names = config.model.dflash2.inputs.past_key_names;
  const auto [target, drafter] = MakeCompatibleMetadata();
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresUniqueRuntimeOutputNames) {
  auto config = MakeDflash2Config();
  config.model.dflash2.block_size = 9;
  config.model.dflash2.num_draft_tokens = 8;
  config.model.dflash2.head_size = 2;
  config.model.dflash2.outputs.scores = "present.0.key";
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddOutput("draft_candidate_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, 8, 2});
  for (int layer = 0; layer < 3; ++layer) {
    for (const auto* kind : {"key", "value"}) {
      drafter.AddInput("past_key_values." + std::to_string(layer) + "." + kind,
                       ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 8, 2, 2});
      drafter.AddOutput("present." + std::to_string(layer) + "." + kind,
                        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-1, 8, 2, 2});
    }
  }
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);

  config = MakeDflash2Config();
  config.model.dflash2.outputs.scores = config.model.dflash2.outputs.candidate_ids;
  std::tie(target, drafter) = MakeCompatibleMetadata();
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresMatchingPagedBlockSize) {
  const auto config = MakeDflash2Config();
  const auto [target, drafter] = MakeCompatibleMetadata();
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 16),
               std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresMatchingLatticeGeometry) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddOutput("draft_candidate_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {-1, 4, 2});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RequiresDynamicMatchingLatticeBatchDimension) {
  const auto config = MakeDflash2Config();
  auto [target, drafter] = MakeCompatibleMetadata();
  drafter.AddOutput("draft_candidate_ids", ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, {4, 3, 2});
  drafter.AddOutput("draft_scores", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {4, 3, 2, 2});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);

  std::tie(target, drafter) = MakeCompatibleMetadata();
  drafter.AddOutput("draft_scores", ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, {-2, 3, 2, 2});
  EXPECT_THROW(ValidateDflash2ModelCompatibility(config, target, drafter, 8), std::runtime_error);
}

TEST(Dflash2ConfigTest, RejectsInvalidCachePoolGeometry) {
  const auto config = MakeDflash2Config();
  EXPECT_THROW(Dflash2Drafter::PoolBlocks(config, 0, 3), std::runtime_error);
  EXPECT_THROW(Dflash2Drafter::PoolBlocks(config, 8, std::numeric_limits<size_t>::max()),
               std::runtime_error);
}

TEST(Dflash2ConfigTest, CapsDraftWidthBySessionAndTurnLimits) {
  EXPECT_EQ(Dflash2DraftWidth(7, 5, 8, 10, 9), 1u);
  EXPECT_EQ(Dflash2DraftWidth(7, 5, 8, 20, 3), 2u);
  EXPECT_EQ(Dflash2DraftWidth(7, 5, 8, 20, 1), 0u);
  EXPECT_EQ(Dflash2DraftWidth(7, 5, 9, 10, 9), 0u);
}

TEST(Dflash2ConfigTest, GraphBlockTableLimitIncludesWorstCaseQuerySpill) {
  EXPECT_EQ(Dflash2GraphBlockTableColumnLimit(/*context_length=*/1024,
                                              /*paged_block_size=*/128,
                                              /*query_block_size=*/8),
            9u);
  EXPECT_EQ(Dflash2GraphBlockTableColumnLimit(/*context_length=*/1024,
                                              /*paged_block_size=*/128,
                                              /*query_block_size=*/129),
            10u);
}

TEST(Dflash2ConfigTest, ReusesProposalBufferUntilAStepOutgrowsIt) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  constexpr auto type = Ort::TypeToTensorType<int32_t>;
  std::unique_ptr<Tensor> slot;

  Dflash2StepTensor(slot, device, type, {2, 4});
  EXPECT_EQ(slot->GetElementCount(), 8u);
  const void* buffer = slot->buffer_;
  ASSERT_NE(buffer, nullptr);

  // A narrower step reshapes a view over the same buffer instead of allocating.
  Dflash2StepTensor(slot, device, type, {1, 3});
  EXPECT_EQ(slot->GetElementCount(), 3u);
  EXPECT_EQ(slot->GetShape(), (std::vector<int64_t>{1, 3}));
  EXPECT_EQ(slot->buffer_, buffer);

  // Tensor rejects a static shape larger than its buffer, so growth must start a new one.
  EXPECT_NO_THROW(Dflash2StepTensor(slot, device, type, {4, 8}));
  EXPECT_EQ(slot->GetElementCount(), 32u);
  const void* grown = slot->buffer_;

  Dflash2StepTensor(slot, device, type, {2, 4});
  EXPECT_EQ(slot->buffer_, grown);
}

TEST(Dflash2ConfigTest, ReportsWhenAProposalBufferMoves) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  constexpr auto type = Ort::TypeToTensorType<int32_t>;
  std::unique_ptr<Tensor> slot;

  // A CUDA graph records the address it was captured against, so the caller has to learn about
  // every move to stop replaying a graph that now points at a freed buffer.
  bool reallocated = false;
  Dflash2StepTensor(slot, device, type, {2, 4}, &reallocated);
  EXPECT_TRUE(reallocated);
  const void* buffer = slot->buffer_;

  reallocated = false;
  Dflash2StepTensor(slot, device, type, {1, 3}, &reallocated);
  EXPECT_FALSE(reallocated);

  std::unique_ptr<Tensor> displaced;
  Dflash2StepTensor(slot, device, type, {4, 8}, &reallocated, &displaced);
  EXPECT_TRUE(reallocated);
  ASSERT_NE(displaced, nullptr);
  EXPECT_EQ(displaced->buffer_, buffer);

  reallocated = false;
  Dflash2StepTensor(slot, device, type, {2, 4}, &reallocated);
  EXPECT_FALSE(reallocated);
}

TEST(Dflash2ConfigTest, FailedReplacementPreservesTheLiveProposalBuffer) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  std::unique_ptr<Tensor> slot;
  Dflash2StepTensor(slot, device, Ort::TypeToTensorType<int32_t>, {2, 4});
  const void* buffer = slot->buffer_;

  // The type change stages a replacement, while the invalid dimension makes CreateTensor fail.
  EXPECT_ANY_THROW(
      Dflash2StepTensor(slot, device, Ort::TypeToTensorType<float>, {0, -1}));
  EXPECT_EQ(slot->GetType(), Ort::TypeToTensorType<int32_t>);
  EXPECT_EQ(slot->buffer_, buffer);
}

TEST(Dflash2ConfigTest, AmortizesProposalBufferGrowth) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  constexpr auto type = Ort::TypeToTensorType<int32_t>;
  std::unique_ptr<Tensor> slot;

  Dflash2StepTensor(slot, device, type, {2, 4});
  // A full-attention drafter's block table gains one column every `block_size` committed tokens,
  // so growth doubles the buffer instead of tracking the shape exactly.
  Dflash2StepTensor(slot, device, type, {2, 5});
  const void* grown = slot->buffer_;
  ASSERT_NE(grown, nullptr);

  Dflash2StepTensor(slot, device, type, {2, 6});
  EXPECT_EQ(slot->buffer_, grown);
  Dflash2StepTensor(slot, device, type, {2, 8});
  EXPECT_EQ(slot->buffer_, grown);
  EXPECT_EQ(slot->GetElementCount(), 16u);
}

TEST(Dflash2ConfigTest, ReusesProposalBufferForAnEmptyStep) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  constexpr auto type = Ort::TypeToTensorType<int32_t>;
  std::unique_ptr<Tensor> slot;

  // A step can serve feeds that carry a query block but no context rows.
  EXPECT_NO_THROW(Dflash2StepTensor(slot, device, type, {0, 64}));
  EXPECT_EQ(slot->GetElementCount(), 0u);
  EXPECT_NO_THROW(Dflash2StepTensor(slot, device, type, {2, 64}));
  EXPECT_EQ(slot->GetElementCount(), 128u);
}

TEST(Dflash2ConfigTest, RejectsProposalShapesThatOverflow) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  std::unique_ptr<Tensor> slot;
  const int64_t huge = std::numeric_limits<int64_t>::max();
  EXPECT_THROW(Dflash2StepTensor(slot, device, Ort::TypeToTensorType<int32_t>, {huge, huge}),
               std::runtime_error);
}

TEST(Dflash2ConfigTest, ReplacesProposalBufferWhenTheElementTypeChanges) {
  auto* device = GetDeviceInterface(DeviceType::CPU);
  std::unique_ptr<Tensor> slot;

  Dflash2StepTensor(slot, device, Ort::TypeToTensorType<int32_t>, {8});
  // A static Tensor keeps the type it was constructed with, so reusing the buffer for another type
  // would silently hand the session the wrong element type.
  Dflash2StepTensor(slot, device, Ort::TypeToTensorType<float>, {4});
  EXPECT_EQ(slot->GetType(), Ort::TypeToTensorType<float>);
  EXPECT_EQ(slot->GetElementCount(), 4u);
}

TEST(Dflash2ConfigTest, JoinsEligibleTurnsAtZeroOrWithWindowedContext) {
  EXPECT_TRUE(Dflash2CanJoin(/*draft_eligible=*/true, /*first_position=*/0));
  EXPECT_FALSE(Dflash2CanJoin(/*draft_eligible=*/false, /*first_position=*/0));
  EXPECT_FALSE(Dflash2CanJoin(/*draft_eligible=*/true, /*first_position=*/1));
  EXPECT_TRUE(Dflash2CanJoin(/*draft_eligible=*/true, /*first_position=*/1, /*context_window=*/12));
  EXPECT_FALSE(Dflash2CanJoin(/*draft_eligible=*/false, /*first_position=*/1, /*context_window=*/12));
}

class PrefixSnapshotTransferTest : public ::testing::TestWithParam<bool> {};

TEST_P(PrefixSnapshotTransferTest, RemapsSelectedBlocksAndPreservesUntouchedBytes) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  CheckpointAllocationDevice device{*cpu, GetParam()};
  PrefixSnapshotCache storage{{0, 12, 0}, 1};
  constexpr auto type = Ort::TypeToTensorType<uint8_t>;
  auto checkpoint = std::make_shared<PrefixSnapshotCache::Entry>();
  checkpoint->shapes = {{3, 4}};
  checkpoint->type = type;
  ASSERT_TRUE(storage.Allocate(checkpoint, &device));
  Tensor source{&device, type}, restored{&device, type};
  source.CreateTensor(std::array<int64_t, 2>{6, 4});
  restored.CreateTensor(std::array<int64_t, 2>{5, 4});
  std::array<uint8_t, 24> values{};
  std::iota(values.begin(), values.end(), uint8_t{1});
  source.GetByteSpan().CopyFromCpu(values);
  std::memset(restored.GetMutableRawData(), 99, 20);
  const std::array<int32_t, 3> original{4, 1, 3}, relocated{2, 0, 4};
  for (size_t block = 0; block < original.size(); ++block) {
    checkpoint->Write(0, block * 4, source, original[block] * 4, 4);
    checkpoint->Read(0, block * 4, restored, relocated[block] * 4, 4);
  }
  EXPECT_EQ(device.downloaded_bytes, GetParam() ? 12u : 0u);
  EXPECT_EQ(device.uploaded_bytes, GetParam() ? 12u : 0u);
  const auto* copied = restored.GetData<uint8_t>();
  for (size_t block = 0; block < original.size(); ++block) {
    for (size_t byte = 0; byte < 4; ++byte) {
      EXPECT_EQ(copied[relocated[block] * 4 + byte], values[original[block] * 4 + byte]);
    }
  }
  for (size_t offset : {4u, 5u, 6u, 7u, 12u, 13u, 14u, 15u}) {
    EXPECT_EQ(copied[offset], 99);
  }
  EXPECT_THROW(checkpoint->Write(0, 0, source, 24, 4), std::out_of_range);
  EXPECT_EQ(device.downloaded_bytes, GetParam() ? 12u : 0u);
}

INSTANTIATE_TEST_SUITE_P(CpuAndHostTransfers, PrefixSnapshotTransferTest, ::testing::Bool());

TEST(Dflash2ConfigTest, HostSnapshotProtectsLeasesAndOutlivesDrafter) {
  auto config = MakeDflash2Config();
  config.config_path = fs::path{MODEL_PATH "engine/synthetic-dspark"};
  auto& draft = config.model.dflash2;
  draft.filename = "dflash2.onnx";
  draft.num_hidden_layers = 1;
  draft.num_key_value_heads = 1;
  draft.head_size = 1;
  draft.block_size = 4;
  draft.num_draft_tokens = 3;
  draft.selector_top_k = 2;
  draft.sliding_window = 8;
  auto model = std::make_shared<Dflash2Model>(CreateDflash2Config(config), GetOrtEnv());
  auto* cpu = model->p_device_kvcache_;
  CheckpointAllocationDevice device{*cpu, true};
  std::shared_ptr<const Dflash2PrefixCheckpoint> checkpoint;
  std::weak_ptr<void> memory;
  {
    Dflash2Drafter drafter{model, 4, Dflash2Drafter::PoolBlocks(config, 4, 1), 1};
    drafter.InitializeSnapshotCache(MakeSnapshotCache(config));
    int request_id{};
    auto* request = reinterpret_cast<Request*>(&request_id);
    Tensor aux{cpu, Ort::TypeToTensorType<float>};
    aux.CreateTensor(std::array<int64_t, 2>{8, 1});
    aux.GetByteSpan().Zero();
    const std::array feeds{Dflash2Drafter::Feed{
        .request = request, .aux_row_count = 8, .anchor_token = 11, .draft_eligible = true, .wants_drafts = true}};
    std::vector<std::vector<int32_t>> drafts;
    ASSERT_TRUE(drafter.Propose(aux, feeds, drafts));
    model->p_device_kvcache_ = &device;
    device.fail_host_allocation = true;
    EXPECT_THROW(drafter.CapturePrefix(request, 8), std::bad_alloc);
    device.fail_host_allocation = false;
    checkpoint = drafter.CapturePrefix(request, 8);
    ASSERT_NE(checkpoint, nullptr);
    ASSERT_EQ(checkpoint->host_buffers.size(), 2u);
    EXPECT_EQ(checkpoint->caches.front()->p_device_->GetType(), DeviceType::CPU);
    memory = checkpoint->host_buffers.front();
    const auto* address = checkpoint->host_buffers.front().get();
    const auto allocations = device.host_allocations;
    auto lease = checkpoint;
    EXPECT_EQ(drafter.CapturePrefix(request, 8), nullptr);
    lease.reset();
    checkpoint = drafter.CapturePrefix(request, 8);
    ASSERT_NE(checkpoint, nullptr);
    EXPECT_EQ(checkpoint->host_buffers.front().get(), address);
    EXPECT_EQ(device.host_allocations, allocations);
    memory = checkpoint->host_buffers.front();
    model->p_device_kvcache_ = cpu;
  }
  EXPECT_FALSE(memory.expired());
  EXPECT_NE(checkpoint->caches.front()->GetRawData(), nullptr);
  checkpoint.reset();
  EXPECT_TRUE(memory.expired());
}

TEST(Dflash2ConfigTest, SnapshotProducersShareCountBudgetAndLeaseProtection) {
  auto config = MakeDflash2Config();
  config.config_path = fs::path{MODEL_PATH "engine/synthetic-dspark"};
  config.engine.dynamic_batching->max_snapshots = 1;
  auto& draft = config.model.dflash2;
  draft.filename = "dflash2.onnx";
  draft.num_hidden_layers = 1;
  draft.num_key_value_heads = 1;
  draft.head_size = 1;
  draft.block_size = 4;
  draft.num_draft_tokens = 3;
  draft.selector_top_k = 2;
  draft.sliding_window = 8;
  auto model = std::make_shared<Dflash2Model>(CreateDflash2Config(config), GetOrtEnv());
  const size_t pool_blocks = Dflash2Drafter::PoolBlocks(config, 4, 1);
  Dflash2Drafter first{model, 4, pool_blocks, 1};
  Dflash2Drafter second{model, 4, pool_blocks, 1};
  auto cache = MakeSnapshotCache(config);
  EXPECT_THROW(first.InitializeSnapshotCache(nullptr), std::invalid_argument);
  first.InitializeSnapshotCache(cache);
  second.InitializeSnapshotCache(cache);
  EXPECT_THROW(first.InitializeSnapshotCache(cache), std::logic_error);
  int first_id{}, second_id{};
  auto* first_request = reinterpret_cast<Request*>(&first_id);
  auto* second_request = reinterpret_cast<Request*>(&second_id);
  Tensor aux{GetDeviceInterface(DeviceType::CPU), Ort::TypeToTensorType<float>};
  aux.CreateTensor(std::array<int64_t, 2>{8, 1});
  aux.GetByteSpan().Zero();
  const std::array first_feeds{Dflash2Drafter::Feed{
      .request = first_request, .aux_row_count = 8, .anchor_token = 11, .draft_eligible = true, .wants_drafts = true}};
  const std::array second_feeds{Dflash2Drafter::Feed{
      .request = second_request, .aux_row_count = 8, .anchor_token = 12, .draft_eligible = true, .wants_drafts = true}};
  std::vector<std::vector<int32_t>> proposals;
  ASSERT_TRUE(first.Propose(aux, first_feeds, proposals));
  ASSERT_TRUE(second.Propose(aux, second_feeds, proposals));
  auto checkpoint = first.CapturePrefix(first_request, 8);
  ASSERT_NE(checkpoint, nullptr);
  ASSERT_EQ(checkpoint->caches.size(), 2u);
  auto lease = checkpoint;
  EXPECT_EQ(second.CapturePrefix(second_request, 8), nullptr);
  EXPECT_EQ(cache->Used(), (std::array<size_t, 3>{0, checkpoint->bytes, 0}));
  const auto* address = checkpoint->caches.front()->GetRawData();
  lease.reset();
  auto replacement = second.CapturePrefix(second_request, 8);
  ASSERT_NE(replacement, nullptr);
  ASSERT_EQ(replacement->caches.size(), 2u);
  EXPECT_EQ(checkpoint->state, PrefixSnapshotCache::State::Evicted);
  EXPECT_EQ(replacement->caches.front()->GetRawData(), address);
  EXPECT_EQ(cache->Used(), (std::array<size_t, 3>{0, replacement->bytes, 0}));
}

namespace {

std::shared_ptr<PrefixSnapshotCache::Entry> MakeStorageEntry(PrefixSnapshotCache& cache,
                                                             DeviceInterface* device,
                                                             std::string_view component = {},
                                                             size_t component_limit =
                                                                 std::numeric_limits<size_t>::max()) {
  auto entry = std::make_shared<PrefixSnapshotCache::Entry>();
  entry->shapes = {{16}};
  entry->type = Ort::TypeToTensorType<uint8_t>;
  if (!cache.Allocate(entry, device, component, component_limit)) return nullptr;
  Tensor source{GetDeviceInterface(DeviceType::CPU), entry->type};
  source.CreateTensor(std::array<int64_t, 1>{32});
  std::array<uint8_t, 32> values{};
  std::iota(values.begin(), values.end(), uint8_t{1});
  source.GetByteSpan().CopyFromCpu(values);
  entry->Write(0, 0, source, 7, 16);
  device->Synchronize();
  return entry;
}

void CheckStorageEntry(const PrefixSnapshotCache::Entry& entry) {
  Tensor restored{GetDeviceInterface(DeviceType::CPU), entry.type};
  restored.CreateTensor(std::array<int64_t, 1>{32});
  std::memset(restored.GetMutableRawData(), 99, 32);
  entry.Read(0, 0, restored, 5, 16);
  const auto* bytes = restored.GetData<uint8_t>();
  for (size_t index = 0; index < 32; ++index) {
    EXPECT_EQ(bytes[index], index >= 5 && index < 21 ? index + 3 : 99);
  }
}

}  // namespace

TEST(PrefixSnapshotCacheTest, ZeroAndUndersizedBudgetsRefuseStorage) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  PrefixSnapshotCache disabled{{0, 0, 0}, 8};
  EXPECT_EQ(MakeStorageEntry(disabled, cpu), nullptr);
  PrefixSnapshotCache undersized{{0, 15, 15}, 8};
  EXPECT_EQ(MakeStorageEntry(undersized, cpu), nullptr);
  PrefixSnapshotCache no_entries{{0, 1024, 1024}, 0};
  EXPECT_EQ(MakeStorageEntry(no_entries, cpu), nullptr);
  PrefixSnapshotCache invalid_vram{{16, 0, 0}, 1};
  EXPECT_THROW(MakeStorageEntry(invalid_vram, cpu), std::runtime_error);
}

TEST(PrefixSnapshotCacheTest, DiagnosticsReportTierMovesAndLeasedCapacityOnlyWhenEnabled) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  CheckpointAllocationDevice device{*cpu, true};
  for (bool enabled : {false, true}) {
    const auto previous_logging = g_log;
    SetLogBool("enabled", enabled);
    testing::internal::CaptureStderr();
    PrefixSnapshotCache cache{{16, 16, 16}, 8};
    auto first = MakeStorageEntry(cache, &device);
    auto second = MakeStorageEntry(cache, &device);
    auto third = MakeStorageEntry(cache, &device);
    auto fourth = MakeStorageEntry(cache, &device);
    auto second_lease = second;
    auto third_lease = third;
    auto fourth_lease = fourth;
    auto refused = MakeStorageEntry(cache, &device);
    const auto log = testing::internal::GetCapturedStderr();
    g_log = previous_logging;
    EXPECT_EQ(first->state, PrefixSnapshotCache::State::Evicted);
    EXPECT_EQ(refused, nullptr);
    CheckStorageEntry(*second_lease);
    CheckStorageEntry(*third_lease);
    CheckStorageEntry(*fourth_lease);
    if (!enabled) {
      EXPECT_EQ(log.find("engine_snapshot_cache"), std::string::npos);
      continue;
    }
    EXPECT_NE(log.find("action=configured"), std::string::npos);
    EXPECT_NE(log.find("action=demoted reason=tier_capacity tier=1 bytes=16"), std::string::npos);
    EXPECT_NE(log.find("action=demoted reason=tier_capacity tier=2 bytes=16"), std::string::npos);
    EXPECT_NE(log.find("action=move_staged"), std::string::npos);
    EXPECT_NE(log.find("used_vram_bytes=16 used_ram_bytes=16 used_disk_bytes=16"), std::string::npos);
    EXPECT_NE(log.find("limit_vram_bytes=16 limit_ram_bytes=16 limit_disk_bytes=16"), std::string::npos);
    EXPECT_NE(log.find("action=evicted reason=tier_capacity"), std::string::npos);
    EXPECT_NE(log.find("action=refused reason=tier_capacity_leased"), std::string::npos);
    EXPECT_NE(log.find("elapsed_us="), std::string::npos);
  }
}

TEST(PrefixSnapshotCacheTest, DiagnosticsDistinguishBudgetAndCountRefusals) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  const auto previous_logging = g_log;
  SetLogBool("enabled", true);
  testing::internal::CaptureStderr();
  PrefixSnapshotCache undersized{{0, 15, 0}, 8};
  auto too_large = MakeStorageEntry(undersized, cpu);
  PrefixSnapshotCache count_limited{{0, 64, 0}, 1};
  auto first = MakeStorageEntry(count_limited, cpu, "dflash2", 1);
  auto lease = first;
  auto component_refused = MakeStorageEntry(count_limited, cpu, "dflash2", 1);
  auto shared_refused = MakeStorageEntry(count_limited, cpu, "other");
  lease.reset();
  auto replacement = MakeStorageEntry(count_limited, cpu, "other");
  const auto log = testing::internal::GetCapturedStderr();
  g_log = previous_logging;
  EXPECT_EQ(too_large, nullptr);
  EXPECT_EQ(component_refused, nullptr);
  EXPECT_EQ(shared_refused, nullptr);
  ASSERT_NE(replacement, nullptr);
  CheckStorageEntry(*replacement);
  EXPECT_NE(log.find("action=refused reason=payload_exceeds_tiers"), std::string::npos);
  EXPECT_NE(log.find("action=refused reason=component_count_leased"), std::string::npos);
  EXPECT_NE(log.find("action=refused reason=shared_count_leased"), std::string::npos);
  EXPECT_NE(log.find("action=reused reason=count"), std::string::npos);
}

TEST(PrefixSnapshotCacheTest, WebGpuIgnoresByteBudgetsButProtectsLeasesAndCountLimits) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  CheckpointAllocationDevice device{*cpu, DeviceType::WEBGPU};
  for (const auto limits : {std::array<size_t, 3>{0, 0, 0}, std::array<size_t, 3>{1, 1, 1}}) {
    PrefixSnapshotCache cache{limits, 2};
    auto first = MakeStorageEntry(cache, &device, "dflash2", 1);
    ASSERT_NE(first, nullptr);
    EXPECT_EQ(first->tier, PrefixSnapshotCache::Tier::Vram);
    EXPECT_EQ(first->caches.front()->p_device_, &device);
    CheckStorageEntry(*first);
    auto lease = first;
    EXPECT_EQ(MakeStorageEntry(cache, &device, "dflash2", 1), nullptr);
    lease.reset();
    auto replacement = MakeStorageEntry(cache, &device, "dflash2", 1);
    ASSERT_NE(replacement, nullptr);
    EXPECT_EQ(first->state, PrefixSnapshotCache::State::Evicted);
    CheckStorageEntry(*replacement);

    auto other = MakeStorageEntry(cache, &device, "other");
    ASSERT_NE(other, nullptr);
    auto third = MakeStorageEntry(cache, &device, "third");
    ASSERT_NE(third, nullptr);
    EXPECT_EQ(replacement->state, PrefixSnapshotCache::State::Evicted);
    EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{32, 0, 0}));
  }
  PrefixSnapshotCache no_entries{{1, 1, 1}, 0};
  EXPECT_EQ(MakeStorageEntry(no_entries, &device), nullptr);
  PrefixSnapshotCache cache{{0, 0, 0}, 1};
  EXPECT_EQ(MakeStorageEntry(cache, &device, "dflash2", 0), nullptr);
}

TEST(PrefixSnapshotCacheTest, WebGpuRangedCopiesPreserveBufferHandles) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  CheckpointAllocationDevice device{*cpu, DeviceType::WEBGPU};
  PrefixSnapshotCache cache{{0, 0, 0}, 1};
  auto entry = MakeStorageEntry(cache, &device);
  ASSERT_NE(entry, nullptr);
  Tensor source{&device, Ort::TypeToTensorType<uint8_t>};
  Tensor destination{&device, Ort::TypeToTensorType<uint8_t>};
  source.CreateTensor(std::array<int64_t, 1>{32});
  destination.CreateTensor(std::array<int64_t, 1>{32});
  std::iota(source.GetMutableData<uint8_t>(), source.GetMutableData<uint8_t>() + 32, uint8_t{1});
  std::memset(destination.GetMutableRawData(), 99, 32);
  device.wrapped_memory.clear();
  entry->Write(0, 3, source, 7, 8);
  entry->Read(0, 3, destination, 5, 8);
  for (void* memory : device.wrapped_memory) {
    EXPECT_TRUE(memory == source.GetMutableRawData() ||
                memory == destination.GetMutableRawData() ||
                memory == entry->caches.front()->GetMutableRawData());
  }
  const auto* bytes = destination.GetData<uint8_t>();
  for (size_t index = 0; index < 32; ++index) {
    EXPECT_EQ(bytes[index], index >= 5 && index < 13 ? index + 3 : 99);
  }
}

TEST(PrefixSnapshotCacheTest, WebGpuBatchesBlockCopiesOncePerBackingBuffer) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  CheckpointAllocationDevice device{*cpu, DeviceType::WEBGPU};
  PrefixSnapshotCache cache{{0, 0, 0}, 1};
  auto entry = MakeStorageEntry(cache, &device);
  ASSERT_NE(entry, nullptr);
  Tensor source{&device, Ort::TypeToTensorType<uint8_t>};
  Tensor destination{&device, Ort::TypeToTensorType<uint8_t>};
  source.CreateTensor(std::array<int64_t, 1>{32});
  destination.CreateTensor(std::array<int64_t, 1>{32});
  std::iota(source.GetMutableData<uint8_t>(), source.GetMutableData<uint8_t>() + 32, uint8_t{1});
  std::memset(destination.GetMutableRawData(), 99, 32);
  const std::array ranges{PrefixSnapshotCache::Entry::Range{0, 16, 8},
                          PrefixSnapshotCache::Entry::Range{8, 0, 8}};
  device.buffer_downloads = device.buffer_uploads = 0;
  entry->Write(0, source, ranges);
  EXPECT_EQ(device.buffer_downloads, 1u);
  EXPECT_EQ(device.buffer_uploads, 1u);
  device.buffer_downloads = device.buffer_uploads = 0;
  entry->Read(0, destination, ranges);
  EXPECT_EQ(device.buffer_downloads, 2u);
  EXPECT_EQ(device.buffer_uploads, 1u);
  const auto* values = destination.GetData<uint8_t>();
  for (size_t index = 0; index < 32; ++index) {
    EXPECT_EQ(values[index], index < 8 || (index >= 16 && index < 24) ? index + 1 : 99);
  }
}

TEST(PrefixSnapshotCacheTest, DiskBatchesRemappedBlocksAcrossTransferBufferBoundaries) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  CheckpointAllocationDevice device{*cpu, true};
  constexpr size_t block_bytes = 4096, blocks = 513, bytes = block_bytes * blocks;
  PrefixSnapshotCache cache{{0, 0, bytes}, 1};
  auto entry = std::make_shared<PrefixSnapshotCache::Entry>();
  entry->shapes = {{static_cast<int64_t>(bytes)}};
  entry->type = Ort::TypeToTensorType<uint8_t>;
  ASSERT_TRUE(cache.Allocate(entry, &device));
  Tensor source{&device, entry->type};
  Tensor destination{&device, entry->type};
  source.CreateTensor(std::array<int64_t, 1>{static_cast<int64_t>(bytes + 32)});
  destination.CreateTensor(source.GetShape());
  std::iota(source.GetMutableData<uint8_t>(), source.GetMutableData<uint8_t>() + bytes + 32, uint8_t{1});
  std::memset(destination.GetMutableRawData(), 99, bytes + 32);
  std::vector<PrefixSnapshotCache::Entry::Range> ranges;
  for (size_t block = 0; block < blocks; ++block) {
    ranges.push_back({block * block_bytes, (blocks - block - 1) * block_bytes + 16, block_bytes});
  }
  entry->Write(0, source, ranges);
  entry->Flush();
  EXPECT_EQ(device.downloaded_bytes, bytes);
  EXPECT_EQ(device.synchronizations, 3u);
  device.synchronizations = 0;
  entry->Read(0, destination, ranges);
  EXPECT_EQ(device.uploaded_bytes, bytes);
  EXPECT_EQ(device.synchronizations, 3u);
  EXPECT_TRUE(std::equal(source.GetData<uint8_t>() + 16, source.GetData<uint8_t>() + bytes + 16,
                         destination.GetData<uint8_t>() + 16));
  for (size_t index = 0; index < 16; ++index) {
    EXPECT_EQ(destination.GetData<uint8_t>()[index], 99);
    EXPECT_EQ(destination.GetData<uint8_t>()[bytes + 16 + index], 99);
  }
}

TEST(PrefixSnapshotCacheTest, LegacyDeviceStorageReusesSnapshotsWithoutTierBudgets) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  for (const bool cuda : {false, true}) {
    CheckpointAllocationDevice device{*cpu, cuda};
    PrefixSnapshotCache cache{{0, 0, 0}, 1, {}, true};
    auto first = MakeStorageEntry(cache, &device);
    ASSERT_NE(first, nullptr);
    CheckStorageEntry(*first);
    auto replacement = MakeStorageEntry(cache, &device);
    ASSERT_NE(replacement, nullptr);
    EXPECT_EQ(first->state, PrefixSnapshotCache::State::Evicted);
    CheckStorageEntry(*replacement);
    EXPECT_EQ(device.host_allocations, 0u);
  }
}

TEST(PrefixSnapshotCacheTest, WebGpuReservationHonorsCountsAndPreservesTargetCapacity) {
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite");
  auto* cpu = model->p_device_kvcache_;
  CheckpointAllocationDevice device{*cpu, DeviceType::WEBGPU};
  model->p_device_kvcache_ = &device;
  auto& batching = *model->config_->engine.dynamic_batching;
  batching.prefix_caching = true;
  batching.max_batch_size = 8;
  batching.dflash2_max_snapshots = 4;
  batching.num_blocks = 7;
  const size_t block_bytes = PagedKeyValueCacheBytesPerBlock(model);
  for (const size_t budget : {0u, 1u}) {
    batching.snapshot_cache.vram_bytes = budget;
    batching.snapshot_cache.ram_bytes = budget;
    batching.snapshot_cache.disk_bytes = budget;
    for (const size_t count : {0u, 1u, 4u}) {
      batching.max_snapshots = count;
      auto manager = CacheManager::Create(model, 0, 0, 2 * block_bytes);
      const size_t expected_count = std::min(count, size_t{3});
      EXPECT_EQ(manager->DraftCheckpointCapacity(), expected_count);
      EXPECT_EQ(manager->DraftCheckpointVramBytes(), expected_count * 2 * block_bytes);
      EXPECT_EQ(manager->Snapshot().total_blocks, 7 - expected_count * 2);
    }
    batching.prefix_caching = false;
    auto disabled = CacheManager::Create(model, 0, 0, 2 * block_bytes);
    EXPECT_EQ(disabled->DraftCheckpointCapacity(), 0u);
    EXPECT_EQ(disabled->Snapshot().total_blocks, 7u);
    batching.prefix_caching = true;
  }
  model->p_device_kvcache_ = cpu;
}

TEST(PrefixSnapshotCacheTest, WebGpuIgnoredBudgetsWarnOncePerEngineWhenEnabled) {
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite");
  auto* cpu = model->p_device_kvcache_;
  CheckpointAllocationDevice device{*cpu, DeviceType::WEBGPU};
  model->p_device_kvcache_ = &device;
  auto& batching = *model->config_->engine.dynamic_batching;
  for (const auto [enabled, warning, budget] :
       {std::tuple{false, true, 1u}, std::tuple{true, false, 1u},
        std::tuple{true, true, 0u}, std::tuple{true, true, 1u}}) {
    batching.snapshot_cache.vram_bytes = budget;
    batching.snapshot_cache.ram_bytes = budget;
    batching.snapshot_cache.disk_bytes = budget;
    const auto previous_logging = g_log;
    SetLogBool("enabled", enabled);
    SetLogBool("warning", warning);
    testing::internal::CaptureStderr();
    {
      Engine engine{model};
    }
    const auto log = testing::internal::GetCapturedStderr();
    g_log = previous_logging;
    const auto message = log.find("WebGPU does not support tiered prefix snapshot byte budgets");
    if (enabled && warning && budget != 0) {
      EXPECT_NE(message, std::string::npos);
      EXPECT_EQ(log.find("WebGPU does not support tiered prefix snapshot byte budgets", message + 1),
                std::string::npos);
    } else {
      EXPECT_EQ(message, std::string::npos);
    }
  }
  model->p_device_kvcache_ = cpu;
}

TEST(PrefixSnapshotCacheTest, OrtAllocationFailurePreservesAccountingAndAllowsRetry) {
  auto model = std::dynamic_pointer_cast<DecoderOnly_Model>(
      CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite"));
  ASSERT_NE(model, nullptr);
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  auto allocator = Ort::Allocator::Create(*model->session_decoder_, cpu->GetAllocator().GetInfo());
  static_cast<OrtAllocator&>(*allocator).Alloc =
      [](OrtAllocator*, size_t) -> void* { throw std::bad_alloc{}; };
  CheckpointAllocationDevice failing_device{*cpu, true, allocator.get()};
  PrefixSnapshotCache cache{{16, 0, 0}, 1};
  auto entry = std::make_shared<PrefixSnapshotCache::Entry>();
  entry->shapes = {{16}};
  entry->type = Ort::TypeToTensorType<uint8_t>;
  EXPECT_THROW(cache.Allocate(entry, &failing_device), std::bad_alloc);
  EXPECT_EQ(entry->state, PrefixSnapshotCache::State::Empty);
  EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{0, 0, 0}));

  CheckpointAllocationDevice device{*cpu, true};
  ASSERT_TRUE(cache.Allocate(entry, &device));
  EXPECT_EQ(entry->state, PrefixSnapshotCache::State::Resident);
  EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{16, 0, 0}));
}

TEST(PrefixSnapshotCacheTest, VramReservationPreservesTargetBlockAndConfiguredLimit) {
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite");
  auto& batching = *model->config_->engine.dynamic_batching;
  batching.prefix_caching = true;
  batching.max_batch_size = 8;
  batching.max_snapshots = 8;
  const size_t block_bytes = PagedKeyValueCacheBytesPerBlock(model);
  const size_t snapshot_bytes = 2 * block_bytes;
  batching.snapshot_cache.vram_bytes = 16 * block_bytes;
  auto* cpu = model->p_device_kvcache_;
  CheckpointAllocationDevice device{*cpu, true};
  model->p_device_kvcache_ = &device;
  bool enabled = false;
  for (const size_t blocks : {4u, 3u, 2u}) {
    batching.num_blocks = blocks;
    auto manager = CacheManager::Create(model, 0, 0, snapshot_bytes, &enabled);
    EXPECT_EQ(enabled, blocks >= 3);
    EXPECT_EQ(manager->DraftCheckpointVramBytes(), blocks >= 3 ? snapshot_bytes : 0u);
    EXPECT_EQ(manager->Snapshot().total_blocks, blocks >= 3 ? blocks - 2 : blocks);
    EXPECT_EQ(batching.snapshot_cache.vram_bytes, 16 * block_bytes);
  }
  model->p_device_kvcache_ = cpu;
}

TEST(PagedKeyValueCacheManifestTest, SnapshotSizingReportsOneBlockMinimumAndBudgetReduction) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  CheckpointAllocationDevice device{*cpu, true};
  auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite");
  model->p_device_kvcache_ = &device;
  auto& batching = *model->config_->engine.dynamic_batching;
  batching.prefix_caching = true;
  batching.max_batch_size = 4;
  batching.num_blocks = 9;
  const size_t block_bytes = PagedKeyValueCacheBytesPerBlock(model);
  batching.snapshot_cache.vram_bytes = 8 * block_bytes;
  const auto previous_logging = g_log.enabled;
  SetLogBool("enabled", true);
  testing::internal::CaptureStderr();
  batching.dflash2_max_snapshots = 1;
  auto smaller = CacheManager::Create(model, 0, 0, 2 * block_bytes);
  batching.dflash2_max_snapshots = 4;
  auto larger = CacheManager::Create(model, 0, 0, 2 * block_bytes);
  auto reduced = CacheManager::Create(model, 0, 0, 3 * block_bytes);
  const auto log = testing::internal::GetCapturedStderr();
  SetLogBool("enabled", previous_logging);

  EXPECT_EQ(smaller->DraftCheckpointCapacity(), 1u);
  EXPECT_EQ(smaller->Snapshot().total_blocks, 7u);
  EXPECT_EQ(larger->DraftCheckpointCapacity(), 4u);
  EXPECT_EQ(larger->Snapshot().total_blocks, 1u);
  EXPECT_EQ(reduced->DraftCheckpointCapacity(), 2u);
  EXPECT_EQ(reduced->Snapshot().total_blocks, 3u);
  EXPECT_NE(log.find("requested=1, effective=1; target cache: blocks=7, block_size=4, bytes=" +
                     std::to_string(7 * block_bytes)),
            std::string::npos);
  EXPECT_NE(log.find("requested=4, effective=4; target cache: blocks=1, block_size=4, bytes=" +
                     std::to_string(block_bytes)),
            std::string::npos);
  EXPECT_NE(log.find("requested=4, effective=2; target cache: blocks=3, block_size=4, bytes=" +
                     std::to_string(3 * block_bytes)),
            std::string::npos);
  EXPECT_NE(log.find("draft snapshot reservation: bytes=" + std::to_string(8 * block_bytes)),
            std::string::npos);
}

TEST(PrefixSnapshotCacheTest, DemotesVramToRamToDiskAndEvictsLeastRecentlyUsed) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  CheckpointAllocationDevice device{*cpu, true};
  PrefixSnapshotCache cache{{16, 16, 16}, 8};
  auto first = MakeStorageEntry(cache, &device);
  ASSERT_NE(first, nullptr);
  EXPECT_EQ(first->tier, PrefixSnapshotCache::Tier::Vram);
  auto second = MakeStorageEntry(cache, &device);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(first->tier, PrefixSnapshotCache::Tier::Ram);
  EXPECT_EQ(second->tier, PrefixSnapshotCache::Tier::Vram);
  auto third = MakeStorageEntry(cache, &device);
  ASSERT_NE(third, nullptr);
  EXPECT_EQ(first->tier, PrefixSnapshotCache::Tier::Disk);
  EXPECT_EQ(second->tier, PrefixSnapshotCache::Tier::Ram);
  EXPECT_EQ(third->tier, PrefixSnapshotCache::Tier::Vram);
  EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{16, 16, 16}));
  CheckStorageEntry(*first);
  CheckStorageEntry(*second);
  CheckStorageEntry(*third);
  auto fourth = MakeStorageEntry(cache, &device);
  ASSERT_NE(fourth, nullptr);
  EXPECT_EQ(first->state, PrefixSnapshotCache::State::Evicted);
  EXPECT_EQ(second->tier, PrefixSnapshotCache::Tier::Disk);
  EXPECT_EQ(third->tier, PrefixSnapshotCache::Tier::Ram);
  EXPECT_EQ(fourth->tier, PrefixSnapshotCache::Tier::Vram);
  CheckStorageEntry(*second);
  CheckStorageEntry(*third);
  CheckStorageEntry(*fourth);
  EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{16, 16, 16}));
  second.reset();
  third.reset();
  fourth.reset();
  EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{0, 0, 0}));
}

TEST(PrefixSnapshotCacheTest, LeasesProtectTierAndCountCapacity) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  PrefixSnapshotCache cache{{0, 16, 16}, 2};
  auto first = MakeStorageEntry(cache, cpu);
  auto lease = first;
  auto second = MakeStorageEntry(cache, cpu);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(first->tier, PrefixSnapshotCache::Tier::Ram);
  EXPECT_EQ(second->tier, PrefixSnapshotCache::Tier::Disk);
  auto second_lease = second;
  EXPECT_EQ(MakeStorageEntry(cache, cpu), nullptr);
  EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{0, 16, 16}));
  CheckStorageEntry(*lease);
  lease.reset();
  auto third = MakeStorageEntry(cache, cpu);
  ASSERT_NE(third, nullptr);
  EXPECT_EQ(first->state, PrefixSnapshotCache::State::Evicted);
  EXPECT_EQ(second->state, PrefixSnapshotCache::State::Resident);
  CheckStorageEntry(*second_lease);
}

TEST(PrefixSnapshotCacheTest, ComponentCountsAreIndependentWithinSharedBudgetsAndGlobalCount) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  PrefixSnapshotCache cache{{0, 64, 0}, 3};
  auto dflash = MakeStorageEntry(cache, cpu, "dflash2", 1);
  auto other_first = MakeStorageEntry(cache, cpu, "other", 2);
  ASSERT_NE(dflash, nullptr);
  ASSERT_NE(other_first, nullptr);
  auto dflash_lease = dflash;
  EXPECT_EQ(MakeStorageEntry(cache, cpu, "dflash2", 1), nullptr);
  auto other_second = MakeStorageEntry(cache, cpu, "other", 2);
  ASSERT_NE(other_second, nullptr);
  EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{0, 48, 0}));

  auto other_lease = other_first;
  auto global_replacement = MakeStorageEntry(cache, cpu, "third", 2);
  ASSERT_NE(global_replacement, nullptr);
  EXPECT_EQ(other_second->state, PrefixSnapshotCache::State::Evicted);
  EXPECT_EQ(dflash->state, PrefixSnapshotCache::State::Resident);
  EXPECT_EQ(other_first->state, PrefixSnapshotCache::State::Resident);
  EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{0, 48, 0}));
}

TEST(PrefixSnapshotCacheTest, CommittedUseRefreshesCountEvictionRecency) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  PrefixSnapshotCache cache{{0, 64, 0}, 2};
  auto first = MakeStorageEntry(cache, cpu);
  auto second = MakeStorageEntry(cache, cpu);
  cache.Touch(*first);
  auto third = MakeStorageEntry(cache, cpu);
  ASSERT_NE(third, nullptr);
  EXPECT_EQ(first->state, PrefixSnapshotCache::State::Resident);
  EXPECT_EQ(second->state, PrefixSnapshotCache::State::Evicted);
  EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{0, 32, 0}));
  EXPECT_THROW(cache.Allocate(second, cpu), std::logic_error);
}

TEST(PrefixSnapshotCacheTest, CountEvictionPrefersRecencyOverCompatibleStorage) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  PrefixSnapshotCache cache{{0, 64, 0}, 2};
  auto oldest = MakeStorageEntry(cache, cpu);
  ASSERT_NE(oldest, nullptr);
  auto recent = std::make_shared<PrefixSnapshotCache::Entry>();
  recent->shapes = {{8, 2}};
  recent->type = oldest->type;
  ASSERT_TRUE(cache.Allocate(recent, cpu));
  Tensor source{cpu, recent->type};
  source.CreateTensor(std::array<int64_t, 1>{16});
  oldest->Read(0, 0, source, 0, 16);
  recent->Write(0, 0, source, 0, 16);
  cache.Touch(*recent);

  auto replacement = std::make_shared<PrefixSnapshotCache::Entry>();
  replacement->shapes = recent->shapes;
  replacement->type = recent->type;
  ASSERT_TRUE(cache.Allocate(replacement, cpu));
  EXPECT_EQ(oldest->state, PrefixSnapshotCache::State::Evicted);
  ASSERT_EQ(recent->state, PrefixSnapshotCache::State::Resident);
  CheckStorageEntry(*recent);
  EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{0, 32, 0}));
}

TEST(PrefixSnapshotCacheTest, DiskOnlyStorageOutlivesManagerAndCleansUpAfterLastLease) {
#ifndef _WIN32
  struct ScopedUmask {
    mode_t previous{::umask(0)};
    ~ScopedUmask() { ::umask(previous); }
  } scoped_umask;
#endif
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  ScopedSnapshotDirectory test_directory{"oga-snapshot-test"};
  const auto& parent = test_directory.Path();
  std::shared_ptr<PrefixSnapshotCache::Entry> entry;
  {
    PrefixSnapshotCache cache{{0, 0, 16}, 1, parent};
    entry = MakeStorageEntry(cache, cpu);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{0, 0, 16}));
#ifndef _WIN32
    const auto directory = fs_std::directory_iterator(parent)->path();
    EXPECT_EQ(fs_std::status(directory).permissions() & fs_std::perms::mask, fs_std::perms::owner_all);
    EXPECT_EQ(fs_std::status(directory / "payload").permissions() & fs_std::perms::mask,
              fs_std::perms::owner_read | fs_std::perms::owner_write);
#endif
  }
  CheckStorageEntry(*entry);
  EXPECT_FALSE(fs_std::is_empty(parent));
  entry.reset();
  EXPECT_TRUE(fs_std::is_empty(parent));
  EXPECT_TRUE(fs_std::remove(parent));
}

TEST(PrefixSnapshotCacheTest, CleanupFailureRespectsLoggingAndWarningSettings) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  ScopedSnapshotDirectory test_directory{"oga-snapshot-cleanup"};
  const auto& parent = test_directory.Path();
  for (const auto [enabled, warning] :
       {std::pair{false, true}, std::pair{true, false}, std::pair{true, true}}) {
    SCOPED_TRACE(testing::Message() << "enabled=" << enabled << ", warning=" << warning);
    PrefixSnapshotCache cache{{0, 0, 16}, 1, parent};
    auto entry = MakeStorageEntry(cache, cpu);
    ASSERT_NE(entry, nullptr);
    const auto directory = fs_std::directory_iterator(parent)->path();
    ASSERT_TRUE(fs_std::create_directory(directory / "blocker"));

    const auto previous_logging = g_log;
    SetLogBool("enabled", enabled);
    SetLogBool("warning", warning);
    testing::internal::CaptureStderr();
    entry.reset();
    const auto log = testing::internal::GetCapturedStderr();
    g_log = previous_logging;

    if (enabled && warning) {
      EXPECT_NE(log.find("Unable to remove prefix snapshot spill storage"), std::string::npos);
    } else {
      EXPECT_TRUE(log.empty());
    }
    EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{0, 0, 0}));
    EXPECT_FALSE(fs_std::exists(directory / "payload"));
    EXPECT_TRUE(fs_std::remove(directory / "blocker"));
    EXPECT_TRUE(fs_std::remove(directory));
  }
  EXPECT_TRUE(fs_std::remove(parent));
}

TEST(PrefixSnapshotCacheTest, FailedDemotionDisablesDiskAndReclaimsUnleasedVictims) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  ScopedSnapshotDirectory test_directory{"oga-snapshot-failure"};
  const auto& parent = test_directory.Path();
  PrefixSnapshotCache cache{{0, 16, 16}, 8, parent};
  auto first = MakeStorageEntry(cache, cpu);
  ASSERT_NE(first, nullptr);
  ASSERT_TRUE(fs_std::remove(parent));
  auto second = MakeStorageEntry(cache, cpu);
  ASSERT_NE(second, nullptr);
  EXPECT_EQ(first->state, PrefixSnapshotCache::State::Evicted);
  CheckStorageEntry(*second);

  ASSERT_TRUE(fs_std::create_directory(parent));
  auto third = MakeStorageEntry(cache, cpu);
  ASSERT_NE(third, nullptr);
  EXPECT_EQ(second->state, PrefixSnapshotCache::State::Evicted);
  EXPECT_EQ(third->tier, PrefixSnapshotCache::Tier::Ram);
  EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{0, 16, 0}));
  EXPECT_TRUE(fs_std::is_empty(parent));
  CheckStorageEntry(*third);
  first.reset();
  second.reset();
  third.reset();
  EXPECT_TRUE(fs_std::remove(parent));
}

TEST(PrefixSnapshotCacheTest, DiskAllocationFailureIsNotRetriedUntilRecreation) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  ScopedSnapshotDirectory test_directory{"oga-snapshot-allocation-failure"};
  const auto& parent = test_directory.Path();
  PrefixSnapshotCache cache{{0, 0, 16}, 1, parent};
  ASSERT_TRUE(fs_std::remove(parent));
  EXPECT_THROW(MakeStorageEntry(cache, cpu), std::system_error);
  ASSERT_TRUE(fs_std::create_directory(parent));
  EXPECT_EQ(MakeStorageEntry(cache, cpu), nullptr);
  EXPECT_TRUE(fs_std::is_empty(parent));
  PrefixSnapshotCache recreated{{0, 0, 16}, 1, parent};
  auto entry = MakeStorageEntry(recreated, cpu);
  ASSERT_NE(entry, nullptr);
  CheckStorageEntry(*entry);
}

#ifndef _WIN32
// Windows cannot rename a directory containing an open spill file.
TEST(PrefixSnapshotCacheTest, DiskFailurePreservesExistingDiskReadsAndLeases) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  ScopedSnapshotDirectory test_directory{"oga-snapshot-leased-failure"};
  const auto parent = test_directory.Path() / "spill";
  const auto moved = test_directory.Path() / "moved";
  ASSERT_TRUE(fs_std::create_directory(parent));
  PrefixSnapshotCache cache{{0, 16, 32}, 8, parent};
  auto first = MakeStorageEntry(cache, cpu);
  auto second = MakeStorageEntry(cache, cpu);
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  ASSERT_EQ(first->tier, PrefixSnapshotCache::Tier::Disk);
  auto disk_lease = first;
  fs_std::rename(parent, moved);
  auto third = MakeStorageEntry(cache, cpu);
  ASSERT_NE(third, nullptr);
  EXPECT_EQ(second->state, PrefixSnapshotCache::State::Evicted);
  CheckStorageEntry(*disk_lease);
  auto ram_lease = third;
  EXPECT_EQ(MakeStorageEntry(cache, cpu), nullptr);
  EXPECT_EQ(disk_lease->state, PrefixSnapshotCache::State::Resident);
  EXPECT_EQ(ram_lease->state, PrefixSnapshotCache::State::Resident);
  EXPECT_EQ(cache.Used(), (std::array<size_t, 3>{0, 16, 16}));
  CheckStorageEntry(*ram_lease);
  fs_std::rename(moved, parent);
}
#endif

TEST(PrefixSnapshotCacheTest, InvalidDiskParentFailsBeforeTheEngineServesRequests) {
  ScopedSnapshotDirectory test_directory{"oga-snapshot-invalid-parent"};
  const auto missing = test_directory.Path() / "missing";
  const auto file = test_directory.Path() / "file";
  std::ofstream{file}.put('x');
  for (const auto& parent : {missing, file}) {
    auto model = CreateModel(GetOrtEnv(), MODEL_PATH "engine/synthetic-composite");
    auto& limits = model->config_->engine.dynamic_batching->snapshot_cache;
    limits.disk_bytes = 1024 * 1024;
    limits.directory = parent.string();
    try {
      Engine engine{model};
      FAIL() << "Expected invalid disk directory to fail Engine construction";
    } catch (const std::system_error& error) {
      EXPECT_NE(std::string{error.what()}.find("engine.dynamic_batching.snapshot_cache.directory"),
                std::string::npos);
      EXPECT_NE(std::string{error.what()}.find(parent.string()), std::string::npos);
    }
  }
  EXPECT_NO_THROW((PrefixSnapshotCache{{0, 16, 0}, 1, missing}));
  EXPECT_NO_THROW((PrefixSnapshotCache{{0, 0, 16}, 1, missing, true}));
}

#ifndef _WIN32
TEST(PrefixSnapshotCacheTest, UnwritableDiskParentFailsDuringConstruction) {
  if (::geteuid() == 0) GTEST_SKIP() << "Root bypasses directory write permissions.";
  ScopedSnapshotDirectory test_directory{"oga-snapshot-unwritable"};
  fs_std::permissions(test_directory.Path(), fs_std::perms::owner_read | fs_std::perms::owner_exec);
  EXPECT_THROW((PrefixSnapshotCache{{0, 0, 16}, 1, test_directory.Path()}), std::system_error);
  fs_std::permissions(test_directory.Path(), fs_std::perms::owner_all);
}
#endif

TEST(PrefixSnapshotCacheTest, RelativeDiskParentIsResolvedBeforeTheWorkingDirectoryChanges) {
  ScopedSnapshotDirectory test_directory{"oga-snapshot-relative"};
  ScopedSnapshotDirectory other_directory{"oga-snapshot-working-directory"};
  struct ScopedWorkingDirectory {
    fs_std::path previous{fs_std::current_path()};
    ~ScopedWorkingDirectory() { fs_std::current_path(previous); }
  } working_directory;
  fs_std::current_path(test_directory.Path().parent_path());
  const auto relative = fs_std::relative(test_directory.Path(), fs_std::current_path());
  ASSERT_FALSE(relative.empty());
  ASSERT_TRUE(relative.is_relative());
  PrefixSnapshotCache cache{{0, 0, 16}, 1, relative};
  fs_std::current_path(other_directory.Path());
  auto entry = MakeStorageEntry(cache, GetDeviceInterface(DeviceType::CPU));
  ASSERT_NE(entry, nullptr);
  EXPECT_FALSE(fs_std::is_empty(test_directory.Path()));
  EXPECT_TRUE(fs_std::is_empty(other_directory.Path()));
  CheckStorageEntry(*entry);
}

TEST(PrefixSnapshotCacheTest, RejectsInvalidRangesAndReportsTruncatedDiskPayload) {
  auto* cpu = GetDeviceInterface(DeviceType::CPU);
  ScopedSnapshotDirectory test_directory{"oga-snapshot-truncate"};
  const auto& parent = test_directory.Path();
  PrefixSnapshotCache cache{{0, 0, 16}, 1, parent};
  auto entry = MakeStorageEntry(cache, cpu);
  ASSERT_NE(entry, nullptr);
  Tensor restored{cpu, entry->type};
  restored.CreateTensor(std::array<int64_t, 1>{16});
  EXPECT_THROW(entry->Read(1, 0, restored, 0, 1), std::out_of_range);
  EXPECT_THROW(entry->Read(0, 15, restored, 0, 2), std::out_of_range);
  EXPECT_THROW(entry->Read(0, 0, restored, 16, 1), std::out_of_range);
  const auto directory = fs_std::directory_iterator(parent)->path();
  fs_std::resize_file(directory / "payload", 0);
  EXPECT_THROW(entry->Read(0, 0, restored, 0, 16), std::ios_base::failure);
  entry.reset();
  EXPECT_TRUE(fs_std::remove(parent));
}

namespace {

// Captures whatever the action logs as a warning.
template <typename Action>
std::string CapturedWarnings(Action&& action) {
  ScopedSnapshotDirectory test_directory{"draft-width-warning"};
  const auto log_path = test_directory.Path() / "warnings.log";
  SetLogString("filename", log_path.string());
  SetLogBool("enabled", true);
  SetLogBool("warning", true);

  action();

  SetLogString("filename", "");
  SetLogBool("enabled", false);
  std::ifstream stream{log_path};
  std::stringstream contents;
  contents << stream.rdbuf();
  stream.close();
  fs_std::remove(log_path);
  return contents.str();
}

// Captures whatever WarnOnClampedDraftWidth logs for one config.
std::string CapturedDraftWidthWarnings(const Config& config) {
  return CapturedWarnings([&] { WarnOnClampedDraftWidth(config); });
}

}  // namespace

TEST(Dflash2ConfigTest, WarnsWhenTheDrafterCannotSupplyTheConfiguredDraftWidth) {
  Config config = MakeDflash2Config();
  config.speculative.max_draft_tokens = 3;  // equals num_draft_tokens, nothing is clamped
  EXPECT_EQ(CapturedDraftWidthWarnings(config), "");

  config.speculative.max_draft_tokens = 5;
  EXPECT_NE(CapturedDraftWidthWarnings(config).find("model.dflash2.num_draft_tokens"),
            std::string::npos);

  config.model.dflash2.is_dspark = true;
  EXPECT_NE(CapturedDraftWidthWarnings(config).find("model.dspark.num_draft_tokens"),
            std::string::npos);
}

TEST(Dflash2ConfigTest, WarnsWhenARuntimeProfileRaisesDraftWidthBeyondTheDrafter) {
  Config config = MakeDflash2Config();
  config.speculative.max_draft_tokens = 3;
  Config::RuntimeProfile profile;
  profile.id = "large";
  profile.eligibility.minimum_total_device_memory_bytes = 1;
  profile.overlay.speculative.max_draft_tokens = 5;
  config.runtime_profiles.push_back(profile);

  const auto warnings = CapturedWarnings([&] { ApplyRuntimeProfile(config, 1); });
  EXPECT_EQ(config.speculative.max_draft_tokens, 5);
  EXPECT_NE(warnings.find("model.dflash2.num_draft_tokens"), std::string::npos);
}

TEST(Dflash2ConfigTest, DoesNotWarnAboutHostingLimitsAtConfigLoad) {
  Config config = MakeDflash2Config();
  // These bounds depend on which speculative path the Engine hosts, so the Engine reports them.
  config.model.dflash2.num_draft_tokens = 16;
  config.model.decoder.state_update_capacity = 1;
  config.speculative.max_draft_tokens = 16;
  EXPECT_EQ(CapturedDraftWidthWarnings(config), "");
}

TEST(Dflash2ConfigTest, DoesNotWarnAboutDraftWidthWithoutABlockDrafter) {
  Config config = MakeDflash2Config();
  // MTP's ceiling is an ONNX output that no session has loaded yet, so it cannot be checked here.
  config.model.dflash2.filename.clear();
  config.model.mtp.filename = "mtp.onnx";
  config.speculative.max_draft_tokens = 16;
  EXPECT_EQ(CapturedDraftWidthWarnings(config), "");
}

}  // namespace Generators::test
