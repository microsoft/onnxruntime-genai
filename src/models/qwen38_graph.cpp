// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Qwen3.8-Flash-Next NVFP4 CUDA-graph decode + MTP(1): model and device engine. See qwen38_graph.h.
#include "engine/step_plan.h"
#include "env_utils.h"
#include "generator/generators.h"
#include "io/kv_cache.h"
#include "logging.h"
#include "qwen38_graph.h"
#include "qwen38_replay_geometry.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace Generators {

namespace {

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

bool StartsWith(const std::string& value, const char* prefix) {
  return value.rfind(prefix, 0) == 0;
}

size_t Elements(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (int64_t dim : shape) count *= static_cast<size_t>(dim);
  return count;
}

size_t AlignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

// Symbolic / dynamic dimensions of the exported graphs are the batch (1) here.
std::vector<int64_t> FixedShape(std::vector<int64_t> shape) {
  for (int64_t& dim : shape) {
    if (dim <= 0) dim = 1;
  }
  return shape;
}

constexpr int kStatusRingSpinCheck = 1 << 16;

// What the replay validation reads of a tensor: its name, shape and element size. `slot_shape` is the shape of one slot (the
// tensor itself carries the slot dimension of a multi-slot device).
Qwen38ReplayTensor ReplayTensorOf(const std::string& name, const std::vector<int64_t>& slot_shape, const DevTensor& tensor) {
  return {name, slot_shape, Ort::SizeOf(tensor.type)};
}
// The cache inputs of `info` that depend on the KV capacity: every input under `prefix` (and past_position_ids).
std::vector<Qwen38CacheInput> CacheInputsOf(const SessionInfo& info, const char* prefix, bool with_positions) {
  std::vector<Qwen38CacheInput> inputs;
  for (const auto& name : info.GetInputNames()) {
    if (StartsWith(name, prefix) || (with_positions && name == "past_position_ids"))
      inputs.push_back({name, info.GetInputShape(name)});
  }
  return inputs;
}

// Walks the fields of one protobuf message that occupies the bytes [begin, end) of a stream. Only the headers of the fields
// are read: a length-delimited payload is skipped unless ReadPayload() asks for it, so the initializers that make up the bulk
// of a model file are never touched. A malformed message (a truncated field, a length that leaves the message, a group) ends
// the walk with Failed() set.
class ProtoFieldReader {
 public:
  struct Field {
    uint32_t number{};
    bool length_delimited{};
    uint64_t payload_begin{};
    uint64_t payload_size{};
  };

  ProtoFieldReader(std::istream& in, uint64_t begin, uint64_t end) : in_(in), pos_(begin), end_(end) {}

  bool Failed() const { return failed_; }

  // Moves to the next field; false at the end of the message and when it is malformed.
  bool Next(Field& field) {
    if (failed_ || pos_ >= end_) return false;
    uint64_t tag = 0;
    if (!ReadVarint(tag)) return Fail();
    const uint64_t number = tag >> 3;
    if (number == 0 || number > kMaxFieldNumber) return Fail();
    field = Field{static_cast<uint32_t>(number), false, 0, 0};
    switch (tag & 7u) {
      case 0: {  // varint
        uint64_t ignored = 0;
        return ReadVarint(ignored) || Fail();
      }
      case 1:  // 64-bit
        return Skip(8);
      case 2: {  // length-delimited
        uint64_t size = 0;
        if (!ReadVarint(size) || size > end_ - pos_) return Fail();
        field.length_delimited = true;
        field.payload_begin = pos_;
        field.payload_size = size;
        return Skip(size);
      }
      case 5:  // 32-bit
        return Skip(4);
      default:  // groups and unknown wire types
        return Fail();
    }
  }

  // The payload of a length-delimited field that Next() returned.
  bool ReadPayload(const Field& field, std::string& bytes) {
    bytes.resize(static_cast<size_t>(field.payload_size));
    in_.clear();
    in_.seekg(static_cast<std::streamoff>(field.payload_begin));
    in_.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    stream_at_pos_ = false;
    return static_cast<bool>(in_) || Fail();
  }

 private:
  bool Fail() {
    failed_ = true;
    return false;
  }

  bool Skip(uint64_t count) {
    if (count > end_ - pos_) return Fail();
    pos_ += count;
    stream_at_pos_ = false;
    return true;
  }

  bool ReadVarint(uint64_t& value) {
    value = 0;
    for (unsigned shift = 0; shift < 64; shift += 7) {
      if (pos_ >= end_) return false;
      if (!stream_at_pos_) {
        in_.clear();
        in_.seekg(static_cast<std::streamoff>(pos_));
        stream_at_pos_ = true;
      }
      const std::istream::int_type byte = in_.get();
      if (byte == std::istream::traits_type::eof()) return false;
      ++pos_;
      value |= static_cast<uint64_t>(byte & 0x7F) << shift;
      if ((byte & 0x80) == 0) return true;
    }
    return false;
  }

  std::istream& in_;
  uint64_t pos_;
  uint64_t end_;
  bool stream_at_pos_{false};  // whether the stream is positioned at pos_
  bool failed_{false};

  static constexpr uint64_t kMaxFieldNumber = (uint64_t{1} << 29) - 1;  // the protobuf limit
};

}  // namespace

void ValidateQwen38CacheShapes(const std::vector<Qwen38CacheInput>& inputs, int kv_capacity, const std::string& graph,
                               size_t request_slots) {
  const auto fail = [&](const Qwen38CacheInput& input, const std::string& requirement) {
    std::ostringstream message;
    message << "qwen38_graph: model.qwen38_graph.kv_capacity (" << kv_capacity << ") does not match the " << graph
            << " input \"" << input.name << "\", declared as [";
    for (size_t i = 0; i < input.shape.size(); ++i) message << (i ? ", " : "") << input.shape[i];
    message << "]: " << requirement << ". Set kv_capacity to the capacity the graphs were exported with.";
    throw std::runtime_error(message.str());
  };
  for (const auto& input : inputs) {
    const auto& shape = input.shape;
    const size_t dot = input.name.rfind('.');
    const std::string tail = dot == std::string::npos ? input.name : input.name.substr(dot + 1);
    const bool is_cache = input.name.find("past_key_values.") != std::string::npos;
    if (input.name == "past_position_ids") {
      if (shape.size() != 2) fail(input, "expected [batch, kv_capacity]");
      if (shape[1] < kv_capacity) fail(input, "dimension 1 must be static and at least kv_capacity");
      // The plane fills the position ids of slot b at b * kv_capacity, ORT reads them at b * dimension 1.
      if (request_slots > 1 && shape[1] != kv_capacity) {
        fail(input, "with " + std::to_string(request_slots) +
                        " request slots dimension 1 must be exactly kv_capacity: the slots' rows lie kv_capacity apart");
      }
    } else if (is_cache && (tail == "key" || tail == "value")) {
      if (shape.size() != 4) fail(input, "expected [batch, 2, kv_capacity, 256]");
      if (shape[1] != 2 || shape[3] != 256) fail(input, "the hand-off copies 2 KV heads of 256 channels");
      if (shape[2] != kv_capacity) fail(input, "dimension 2 (the cache length) must be static and kv_capacity long");
    } else if (is_cache && tail == "index_key") {
      if (shape.size() != 3) fail(input, "expected [batch, kv_capacity / 4, 128]");
      if (shape[2] != 128) fail(input, "dimension 2 (the index-key width) must be 128");
      if (shape[1] < kv_capacity / kQwen38IndexKeyRatio)
        fail(input, "dimension 1 must be static and hold one key per 4 positions");
    }
  }
}

size_t ResolveQwen38MaxBatchSize(const Config& config) {
  constexpr long long kMaxSlots = Config::Model::Qwen38Graph::kMaxBatchSize;
  const auto checked = [&](long long requested, const char* setting) {
    if (requested < 1 || requested > kMaxSlots) {
      throw std::out_of_range(std::string(setting) + " must be in [1, " + std::to_string(kMaxSlots) +
                              "] for qwen38_flash_graph, got " + std::to_string(requested));
    }
    return static_cast<size_t>(requested);
  };
  if (config.engine.dynamic_batching) {
    return checked(static_cast<long long>(config.engine.dynamic_batching->max_batch_size),
                   "engine.dynamic_batching.max_batch_size");
  }
  if (config.engine.static_batching) {
    return checked(static_cast<long long>(config.engine.static_batching->max_batch_size),
                   "engine.static_batching.max_batch_size");
  }
  // 1 is the default, which leaves the choice to the settings below; any other value is the one asked for (the parser
  // refuses what lies outside [1, kMaxBatchSize], a programmatically built Config is checked here).
  if (config.model.qwen38_graph.max_batch_size != 1) {
    return checked(config.model.qwen38_graph.max_batch_size, "model.qwen38_graph.max_batch_size");
  }
  if (config.search.batch_size > 1) {
    return checked(config.search.batch_size, "search.batch_size");
  }
  // The test hook is read as strictly as the settings above: a value that is not an integer in range is refused, not mapped
  // to one slot (an empty value counts as unset).
  if (const char* env = std::getenv("ORT_QWEN38_MAX_BATCH_SIZE"); env != nullptr && *env != '\0') {
    char* end = nullptr;
    errno = 0;
    const long long parsed = std::strtoll(env, &end, 10);
    if (end == env || *end != '\0' || errno == ERANGE) {
      throw std::out_of_range(std::string("ORT_QWEN38_MAX_BATCH_SIZE must be an integer in [1, ") + std::to_string(kMaxSlots) +
                              "] for qwen38_flash_graph, got \"" + env + "\"");
    }
    return checked(parsed, "ORT_QWEN38_MAX_BATCH_SIZE");
  }
  return 1;
}

bool Qwen38HasSeparatePrefillSessions(const Config& config) {
  const auto& q = config.model.qwen38_graph;
  return q.prefill.enabled && !q.prefill.dense.filename.empty() && q.prefill.dense.filename != config.model.decoder.filename;
}

void ValidateQwen38MultiSlotConfig(const Config& config) {
  const size_t slots = ResolveQwen38MaxBatchSize(config);
  if (slots <= 1) return;
  const auto& q = config.model.qwen38_graph;
  const std::string needs = "qwen38_flash_graph: " + std::to_string(slots) +
                            " request slots (model.qwen38_graph.max_batch_size, engine.dynamic_batching / static_batching "
                            "max_batch_size or search.batch_size) need ";
  if (!q.fold.enabled) {
    throw std::runtime_error(needs +
                             "the folded verify path (model.qwen38_graph.fold.enabled): every other decode path runs one "
                             "request slot. Set max_batch_size to 1 or enable the fold stack.");
  }
  if (!q.prefill.enabled || Qwen38HasSeparatePrefillSessions(config)) {
    throw std::runtime_error(needs +
                             "the unified fast prefill (model.qwen38_graph.prefill.enabled with the fold model as the dense "
                             "graph, that is without a separate prefill.dense.filename): the prompt of a slot is ingested "
                             "chunk by chunk through the fold target, the only prompt path that addresses one slot. Neither "
                             "the decode-graph prompt path (prefill.enabled=false) nor a separate dense prefill graph can. "
                             "Set max_batch_size to 1 or enable the unified prefill.");
  }
}

void ValidateQwen38BatchedTargetOutputs(size_t request_slots, const std::vector<std::string>& output_names) {
  if (request_slots <= 1) return;
  const std::string needs = "qwen38_flash_graph: " + std::to_string(request_slots) +
                            " request slots need a target graph with compact recurrent-state replay (state_update.* "
                            "outputs: every slot keeps its own accepted prefix), but ";
  bool compact_state = false;
  for (const std::string& name : output_names) {
    if (StartsWith(name, "state_update.")) compact_state = true;
    if (StartsWith(name, "present_mid") || StartsWith(name, "head.present_mid")) {
      throw std::runtime_error(needs + "the target is a ring-bank graph (it has the intermediate-state output '" + name +
                               "'). Set max_batch_size to 1 or export the target with compact state updates.");
    }
  }
  if (!compact_state) {
    throw std::runtime_error(needs + "the target graph has none. Set max_batch_size to 1 or export the target with compact state updates.");
  }
}

void ValidateQwen38BatchedVerifyRows(size_t request_slots, int verify_rows) {
  if (request_slots <= 1 || verify_rows > 2) return;
  throw std::runtime_error("qwen38_flash_graph: " + std::to_string(request_slots) +
                           " request slots need a target graph that verifies at least 3 rows per step, but this one verifies " +
                           std::to_string(verify_rows) +
                           ": the batched fold inputs of a 2-row target keep the single-slot control and context sizes (8 and 2 "
                           "words), while the plane prepares 16 control words and 2 * rows context words per slot. Set "
                           "max_batch_size to 1 or export the target with at least 2 draft tokens.");
}

std::optional<std::vector<std::string>> ReadQwen38OnnxOutputNames(const std::filesystem::path& model_file) {
  constexpr uint32_t kModelGraph = 7;     // onnx.ModelProto.graph
  constexpr uint32_t kGraphOutput = 12;   // onnx.GraphProto.output
  constexpr uint32_t kValueInfoName = 1;  // onnx.ValueInfoProto.name
  constexpr uint64_t kMaxValueInfoBytes = uint64_t{1} << 20;

  std::ifstream file(model_file, std::ios::binary);
  if (!file) return std::nullopt;
  file.seekg(0, std::ios::end);
  const std::streamoff file_size = file.tellg();
  if (!file || file_size <= 0) return std::nullopt;

  std::vector<std::string> names;
  bool has_graph = false;
  ProtoFieldReader model(file, 0, static_cast<uint64_t>(file_size));
  ProtoFieldReader::Field field;
  while (model.Next(field)) {
    if (field.number != kModelGraph || !field.length_delimited) continue;
    if (has_graph) return std::nullopt;
    has_graph = true;
    ProtoFieldReader graph(file, field.payload_begin, field.payload_begin + field.payload_size);
    ProtoFieldReader::Field output;
    while (graph.Next(output)) {
      if (output.number != kGraphOutput || !output.length_delimited) continue;
      std::string value_info;
      if (output.payload_size > kMaxValueInfoBytes || !graph.ReadPayload(output, value_info)) return std::nullopt;
      std::istringstream value_info_stream(value_info);
      ProtoFieldReader info(value_info_stream, 0, value_info.size());
      ProtoFieldReader::Field entry;
      std::string name;
      while (info.Next(entry)) {
        if (entry.number == kValueInfoName && entry.length_delimited && !info.ReadPayload(entry, name)) return std::nullopt;
      }
      if (info.Failed() || name.empty()) return std::nullopt;
      names.push_back(std::move(name));
    }
    if (graph.Failed()) return std::nullopt;
  }
  if (model.Failed() || !has_graph || names.empty()) return std::nullopt;
  return names;
}

void ValidateQwen38TargetModelFile(size_t request_slots, const std::filesystem::path& model_file) {
  if (request_slots <= 1) return;
  if (const auto output_names = ReadQwen38OnnxOutputNames(model_file)) {
    ValidateQwen38BatchedTargetOutputs(request_slots, *output_names);
  }
}

size_t DevTensor::Bytes() const {
  return Elements(shape) * Ort::SizeOf(type);
}

int Qwen38GraphAnnotationId(Qwen38GraphRun run, int rows, int phase, int batch) {
  const int run_index = static_cast<int>(run);
  if (run_index < 0 || run_index >= kQwen38GraphRunCount || rows < 1 || rows > kQwen38GraphIdMaxRows || phase < 0 ||
      phase >= kQwen38GraphIdMaxPhases || batch < 1 || batch > kQwen38GraphIdMaxBatch ||
      (batch > 1 && run != Qwen38GraphRun::kTarget)) {
    throw std::out_of_range("qwen38_graph: no CUDA graph id for run=" + std::to_string(run_index) +
                            ", rows=" + std::to_string(rows) + ", phase=" + std::to_string(phase) +
                            ", batch=" + std::to_string(batch));
  }
  constexpr int kSingleSlotIds = kQwen38GraphRunCount * kQwen38GraphIdMaxRows * kQwen38GraphIdMaxPhases;
  const int single_slot = (run_index * kQwen38GraphIdMaxRows + (rows - 1)) * kQwen38GraphIdMaxPhases + phase;
  if (batch == 1) return 1 + single_slot;
  // The batched target runs follow every single-slot id.
  return 1 + kSingleSlotIds + ((batch - 2) * kQwen38GraphIdMaxRows + (rows - 1)) * kQwen38GraphIdMaxPhases + phase;
}

Qwen38DflashWarmupPlan PlanQwen38DflashWarmup(bool warmup, int verify_rows, int num_banks) noexcept {
  if (!warmup || verify_rows < 1 || num_banks < 1) return {};
  return {true, static_cast<size_t>(verify_rows) * static_cast<size_t>(num_banks) *
                    static_cast<size_t>(kQwen38WarmupRunsPerGraph)};
}

bool Qwen38DflashSkipsInputSanitizer() {
  bool skip = true;
  GetEnv("ORT_PAGED_ATTENTION_SKIP_SANITIZER", skip);
  return skip;
}

void ValidateQwen38DflashFeatureLayout(size_t aux_width, ONNXTensorElementDataType aux_type) {
  if (aux_width == kQwen38DflashFeatureWidth && aux_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) return;
  throw std::runtime_error("qwen38_graph: the DFlash drafter reads auxiliary hidden states of " + std::to_string(aux_width) +
                           " values of ONNX element type " + std::to_string(static_cast<int>(aux_type)) +
                           " per row, but the engine produces rows of " + std::to_string(kQwen38DflashFeatureWidth) +
                           " bfloat16 values (ONNX element type " +
                           std::to_string(static_cast<int>(ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)) +
                           "); check model.dflash2.inputs.aux_hidden_states");
}

size_t Qwen38DflashCachePageSize(const ModelStateMetadata& drafter, const std::string& past_key_names) {
  const std::string input = ComposeKeyValueName(past_key_names, 0);
  const std::vector<int64_t> shape = drafter.GetInputShape(input);
  if (shape.size() != 4 || (shape[1] != 16 && shape[1] != 128 && shape[1] != 256)) {
    throw std::runtime_error("DFlash experiment requires cache pages of 16, 128, or 256 tokens in the drafter input " +
                             input);
  }
  return static_cast<size_t>(shape[1]);
}

size_t Qwen38DflashPromptSlotRows(int chunk, int verify_rows) {
  if (chunk < 1 || verify_rows < 1) {
    throw std::invalid_argument("qwen38_graph: the DFlash prompt buffer needs a positive prefill chunk (" +
                                std::to_string(chunk) + ") and verify width (" + std::to_string(verify_rows) + ")");
  }
  return static_cast<size_t>(std::max(chunk, verify_rows));
}

void Qwen38ForgetPromptTokens(std::vector<int32_t>& tokens) noexcept {
  tokens.clear();
  if (tokens.capacity() > kQwen38RetainedPromptTokens) std::vector<int32_t>().swap(tokens);
}

void CheckQwen38DecodeStatus(std::span<const bool> started, const std::function<int64_t(size_t)>& read_bits) {
  std::ostringstream failures;
  size_t failed = 0;
  int64_t first_bits = 0;
  for (size_t slot = 0; slot < started.size(); ++slot) {
    if (!started[slot]) continue;
    const int64_t bits = read_bits(slot);
    if (bits == 0) continue;
    if (failed++ == 0) first_bits = bits;
    failures << (failed > 1 ? ", " : "") << "slot " << slot << "=" << bits;
  }
  if (failed == 0) return;
  if (started.size() == 1) {
    throw std::runtime_error("qwen38_graph: device error bits " + std::to_string(first_bits) + " after decode");
  }
  throw std::runtime_error("qwen38_graph: device error bits after decode (" + failures.str() + ")");
}

// ------------------------------------------------------------------------------------------------
// Engine
// ------------------------------------------------------------------------------------------------
Qwen38GraphEngine::Qwen38GraphEngine(Qwen38GraphModel& model)
    : model_{model},
      plane_{*model.plane_},
      capacity_{model.config_->model.qwen38_graph.kv_capacity},
      vocab_{model.config_->model.vocab_size},
      max_batch_{static_cast<int>(ResolveQwen38MaxBatchSize(*model.config_))},
      ledger_{model.config_->model.qwen38_graph.kv_capacity},
      run_hints_{static_cast<size_t>(max_batch_)} {
  const auto& q = model_.config_->model.qwen38_graph;
  mtp_enabled_ = q.mtp.enabled;
  prefill_head_ = q.mtp.prefill_head;
  keep_hidden_ = mtp_enabled_ && prefill_head_ && model_.config_->model.dflash2.filename.empty();
  fold_ = q.fold.enabled;
  // Test hook, deliberately not a config key: ORT_QWEN38_FORCE_REJECT=1 replaces every draft before its verify.
  if (const char* force_reject_env = std::getenv("ORT_QWEN38_FORCE_REJECT")) {
    force_reject_ = std::atoi(force_reject_env) != 0;
  }
  if (const char* sl_env = std::getenv("ORT_QWEN38_SUFFIX_LOOKUP")) {
    suffix_lookup_mode_ = std::atoi(sl_env);
  }
  if (fold_ && !mtp_enabled_) throw std::runtime_error("qwen38_graph: the fold stack requires model.qwen38_graph.mtp.enabled");
  // The strides of the hand-off and of the head KV prefill come from kv_capacity: reject graphs exported for another
  // capacity before any cache buffer is allocated.
  ValidateQwen38CacheShapes(CacheInputsOf(model_.session_info_, "past_key_values.", true), capacity_, "target",
                            static_cast<size_t>(max_batch_));
  if (fold_) {
    ValidateQwen38CacheShapes(CacheInputsOf(model_.session_info_, "head.past_key_values.", false), capacity_,
                              "fold head");
  }
  stream_ = model_.p_device_->GetCudaStream();
  plane_.Check(plane_.set_device(model_.p_device_->GetDeviceId(nullptr)), "cudaSetDevice");
  memory_info_ = model_.p_device_->GetMemoryInfo();

  result_ring_ = PlaneHostBuffer(plane_, static_cast<size_t>(max_batch_) * kRing * 8 * sizeof(int64_t));
  stage_ = PlaneHostBuffer(plane_, 16 * sizeof(int64_t));
  result_host_ = result_ring_.data();
  stage_host_ = stage_.data();

  // control block (the fold model binds it as input, so it exists before the target bindings)
  // Allocated with kCtlWordsPerSlot (16) int64 words per slot: the fold model reads the verify tokens from them, [0] the pending
  // token, [2] draft_0 and [8] .. [13] draft_1 .. draft_6 for the S=3 .. S=8 multi-draft verify.
  ctl_ = PlaneBuffer(plane_, static_cast<size_t>(max_batch_) * kCtlWordsPerSlot * sizeof(int64_t), stream_);
  ctx_commit_ = PlaneBuffer(plane_, static_cast<size_t>(max_batch_) * 2 * sizeof(int64_t), stream_);
  // Per slot, ctx_a_ holds kMaxVerifyRows contiguous 2-int64 row contexts (rows 0 .. 7); ctx_b_ is row 1's buffer (also aliased
  // at ctx_a_ + 16 bytes).
  ctx_a_ = PlaneBuffer(plane_, static_cast<size_t>(max_batch_) * kMaxVerifyRows * 2 * sizeof(int64_t), stream_);
  ctx_b_ = PlaneBuffer(plane_, static_cast<size_t>(max_batch_) * 2 * sizeof(int64_t), stream_);
  if (fold_) BuildHeadState();  // the fold model binds the head state
  BuildTarget();
  const auto ple_start = Clock::now();
  const std::filesystem::path model_root{model_.config_->config_path.string()};
  ple_ = std::make_unique<Qwen38PleTable>(plane_, model_root / q.ple_table, model_root, capacity_);
  ple_register_seconds = SecondsSince(ple_start);
  params_dev_ = PlaneBuffer(plane_, sizeof(Qwen38PleParams), stream_);
  plane_.Check(plane_.memcpy_h2d(params_dev_.data(), &ple_->params, sizeof(Qwen38PleParams), stream_), "PLE params");
  SyncStream();  // ple_->params stays alive; the copy above is complete once the stream drains
  if (ple_width_ != ple_->params.n_heads * ple_->params.head_dim)
    throw std::runtime_error("qwen38_graph: PLE table width does not match the model's ple_embeddings input");

  scratch_i32_ = PlaneBuffer(plane_, sizeof(int32_t), stream_);
  part_v_ = PlaneBuffer(plane_, kArgmaxBlocks * sizeof(float), stream_);
  part_i_ = PlaneBuffer(plane_, kArgmaxBlocks * sizeof(int32_t), stream_);
  const size_t slots = static_cast<size_t>(capacity_) + kMaxVerifyRows + 1;
  token_slots_ = slots;
  d_tokens_ = PlaneBuffer(plane_, static_cast<size_t>(max_batch_) * slots * sizeof(int64_t), stream_);
  d_arange64_ = PlaneBuffer(plane_, slots * sizeof(int64_t), stream_);
  d_arange32_ = PlaneBuffer(plane_, slots * sizeof(int32_t), stream_);
  d_last_hidden_ = PlaneBuffer(plane_, static_cast<size_t>(max_batch_) * static_cast<size_t>(hidden_width_) * 2 * kMaxVerifyRows, stream_);
  if (keep_hidden_) d_prompt_hidden_ = PlaneBuffer(plane_, slots * hidden_width_ * 2, stream_);
  {
    std::vector<int64_t> a64(slots);
    std::vector<int32_t> a32(slots);
    for (size_t i = 0; i < slots; ++i) {
      a64[i] = static_cast<int64_t>(i);
      a32[i] = static_cast<int32_t>(i);
    }
    plane_.Check(plane_.memcpy_h2d(d_arange64_.data(), a64.data(), slots * 8, stream_), "arange64");
    plane_.Check(plane_.memcpy_h2d(d_arange32_.data(), a32.data(), slots * 4, stream_), "arange32");
    SyncStream();  // host vectors die here
  }
  if (fold_ && model_.session_head_r2_)
    BuildHeadR2();
  else if (mtp_enabled_ && !fold_)
    BuildHead();
  if (q.prefill.enabled) BuildFastPrefill();
  if (!model_.config_->model.dflash2.filename.empty()) {
    if (model_.config_->model.dflash2.is_dspark)
      throw std::runtime_error("qwen38_flash_graph does not support model.dspark");
    if (!fold_ || !fast_ || !fast_->unified)
      throw std::runtime_error("DFlash experiment requires unified folded prefill");
    if (!model_.session_info_.HasOutput("aux_hidden_states"))
      throw std::runtime_error("DFlash experiment requires the target graph to expose aux_hidden_states");
    if (model_.config_->model.dflash2.num_draft_tokens != verify_rows_ - 1)
      throw std::runtime_error("DFlash draft count must match target prefix-state width");
    auto draft_config = CreateDflash2Config(*model_.config_);
    draft_config->model.embedding.filename.clear();
    dflash_model_ = std::make_shared<Dflash2Model>(
        std::move(draft_config), GetOrtEnv(), nullptr,
        [this](OrtSessionOptions& opts) {
          opts.AddConfigEntry("session.paged_attention.skip_input_sanitizer",
                              Qwen38DflashSkipsInputSanitizer() ? "1" : "0");
          if (model_.pool_) model_.pool_->AddTo(opts, "dflash2");
          if (!model_.embed_values_.empty()) {
            opts.AddInitializer("model.embed_tokens.weight", *model_.embed_values_.front());
            if (model_.embed_is_int8_ && model_.embed_values_.size() >= 2) {
              opts.AddInitializer("model.embed_tokens.weight.row_scales", *model_.embed_values_[1]);
            }
          }
        });
    const size_t page_size = Qwen38DflashCachePageSize(dflash_model_->session_info_,
                                                       dflash_model_->config_->model.dflash2.inputs.past_key_names);
    dflash_ = std::make_unique<Dflash2Drafter>(
        dflash_model_, page_size,
        Dflash2Drafter::PoolBlocks(*model_.config_, page_size, static_cast<size_t>(max_batch_)),
        static_cast<size_t>(max_batch_));
    dflash_->SetPinnedBlockTableWidth(true);
    ValidateQwen38DflashFeatureLayout(dflash_->AuxHiddenSize(), dflash_->AuxType());
    // The slots are drafted one after the other without draining the stream in between.
    dflash_->SetFreshStagingMirrors(max_batch_ > 1);
    dflash_prompt_slot_rows_ = Qwen38DflashPromptSlotRows(fast_->chunk, verify_rows_);
    d_dflash_prompt_ =
        PlaneBuffer(plane_, static_cast<size_t>(max_batch_) * dflash_prompt_slot_rows_ * kDflashWidth * 2, stream_);
    d_dflash_step_aux_ = PlaneBuffer(plane_, static_cast<size_t>(max_batch_) * kMaxVerifyRows * kDflashWidth * 2, stream_);
    const auto& dflash_config = model_.config_->model.dflash2;
    const size_t dflash_context_window =
        dflash_config.sliding_window > 0 ? static_cast<size_t>(dflash_config.sliding_window + dflash_config.block_size) : 0;
    for (auto& features : slot_dflash_prompt_) features = Qwen38DflashPromptFeatures{dflash_context_window};
  }
  Reset();
  if (dflash_) {
    const auto start = Clock::now();
    run_hints_.RequireDeterministicMoe();
    const Qwen38DflashWarmupPlan warmup = PlanQwen38DflashWarmup(q.warmup, verify_rows_, num_banks_);
    if (warmup.drafter && fast_ && fast_->unified) {
      if (max_batch_ > 1) {
        // Pre-size DFlash2 StepTensors for max_batch_ before capturing single-slot / multi-slot graphs.
        const size_t total_aux_rows = static_cast<size_t>(max_batch_) * static_cast<size_t>(verify_rows_);
        const int64_t aux_shape[2] = {static_cast<int64_t>(total_aux_rows), static_cast<int64_t>(kDflashWidth)};
        Tensor aux_warm(model_.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
        aux_warm.ort_tensor_ = OrtValue::CreateTensor(
            *memory_info_, d_dflash_prompt_.data(), total_aux_rows * kDflashWidth * 2,
            std::span<const int64_t>(aux_shape, 2), ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
        std::vector<Dflash2Drafter::Feed> warm_feeds(max_batch_);
        for (int b = 0; b < max_batch_; ++b) {
          warm_feeds[b].request = const_cast<Request*>(SlotRequestKey(b));
          warm_feeds[b].aux_row_begin = static_cast<size_t>(b * verify_rows_);
          warm_feeds[b].aux_row_count = static_cast<size_t>(verify_rows_);
          warm_feeds[b].first_position = 0;
          warm_feeds[b].anchor_token = 1000;
          warm_feeds[b].draft_eligible = true;
          warm_feeds[b].wants_drafts = true;
        }
        std::vector<std::vector<int32_t>> warm_drafts;
        dflash_->Propose(aux_warm, warm_feeds, warm_drafts, nullptr, nullptr, true);
        dflash_->ReleaseAll();
        SyncStream();
      }
      if (fast_->chunk > 512) {
        Draft(d_dflash_prompt_.data(), static_cast<size_t>(fast_->chunk), static_cast<size_t>(capacity_), 1000, false);
        dflash_->ReleaseAll();
      }
      const int warm_len = std::min<int>({fast_->chunk, 512, capacity_ - 3});
      if (warm_len >= 1) {
        std::vector<int32_t> prompt(warm_len, 1000);
        FastAppend(prompt);
        FastHandoff();
        size_t pos = 0;
        if (const auto saved = slot_dflash_prompt_[0].TakeSavedChunk()) {
          for (int rep = 0; rep < 5; ++rep) {
            dflash_->ReleaseAll();
            Draft(d_dflash_prompt_.data(), saved->rows, saved->position, 1000, true);
          }
          pos = saved->position + saved->rows;
        }
        for (size_t r = 1; r <= static_cast<size_t>(verify_rows_); ++r) {
          for (int rep = 0; rep < 5; ++rep) {
            Draft(d_dflash_prompt_.data(), r, pos, 1000, true);
            pos += r;
          }
        }
        SyncStream();
        Reset();
      }
    }
    if (warmup.target_tokens > 0) {
      const size_t rotation_tokens = static_cast<size_t>(verify_rows_) * static_cast<size_t>(num_banks_);
      const int max_step_prompt = capacity_ - (verify_rows_ + 1);
      if (max_step_prompt >= static_cast<int>(rotation_tokens)) {
        const size_t batch_tokens = (static_cast<size_t>(max_step_prompt) / rotation_tokens) * rotation_tokens;
        for (size_t done = 0; done < warmup.target_tokens; done += batch_tokens) {
          const size_t n = std::min(batch_tokens, warmup.target_tokens - done);
          std::vector<int32_t> dummy(n);
          for (size_t i = 0; i < n; ++i) dummy[i] = static_cast<int32_t>(1000 + done + i);
          StepPrefill(dummy);
          Reset();
        }
        if (fold_ && max_batch_ > 1) {
          // The batched graphs are captured after the arenas ORT shrinks once the prompt state has moved to the decode graphs.
          // Each (batch size, bank) graph is run as often as a single-slot graph, enough for ORT to capture it and replay it:
          // one run would leave the capture to the first live cycle that reaches it.
          const int runs_per_graph = Qwen38WarmupRuns(IsGraphCaptureEnabled(model_.config_->model.decoder.session_options));
          FastHandoff();
          for (int B = 2; B <= max_batch_; ++B) {
            for (int s = 0; s < B; ++s) {
              MuteSlot(s);
            }
            PrepBatch(B);
            for (int k = 0; k < num_banks_; ++k) {
              Direction& d = batch_dirs_[B - 1][k];
              for (int run = 0; run < runs_per_graph; ++run) {
                plane_.Check(plane_.device_memset(banks_[k].arena.data(), 0, banks_[k].arena.size(), stream_),
                             "clear warmup bank");
                SyncStream();
                RunSession(*model_.session_target_, d.run_options.get(), *d.binding, nullptr,
                           Qwen38FirstSlots(static_cast<size_t>(B)));
                SyncStream();
              }
            }
          }
          Reset();
        }
      }
    }
    warmup_seconds = SecondsSince(start);
  }
  if (q.warmup && !dflash_) {
    const auto start = Clock::now();
    Warmup();
    warmup_seconds = SecondsSince(start);
  }
  if (g_log.enabled)
    Log("qwen38_graph") << "engine ready: ple register " << ple_register_seconds << " s, warmup " << warmup_seconds
                        << " s, mtp=" << mtp_enabled_ << " fold=" << fold_ << " fast_prefill=" << (fast_ != nullptr) << std::endl;
}

Qwen38GraphEngine::~Qwen38GraphEngine() {
  Drain();
  // IO bindings and OrtValues reference the buffers: release them first
  for (auto& row : dirs_)
    for (auto& d : row) d.binding.reset();
  for (auto& b_row : batch_dirs_)
    for (auto& d : b_row) d.binding.reset();
  for (auto& h : head_)
    for (auto& d : h.dirs) d.binding.reset();
  for (auto& d : head_r2_.dirs) d.binding.reset();
  fast_.reset();
}

void Qwen38GraphEngine::Drain() noexcept {
  try {
    SyncStream();
  } catch (...) {
  }
}

std::unique_ptr<DevTensor> Qwen38GraphEngine::MakeTensor(void* ptr, std::vector<int64_t> shape,
                                                         ONNXTensorElementDataType type) {
  auto tensor = std::make_unique<DevTensor>();
  tensor->ptr = static_cast<uint8_t*>(ptr);
  tensor->shape = std::move(shape);
  tensor->type = type;
  tensor->value = OrtValue::CreateTensor(*memory_info_, ptr, tensor->Bytes(), tensor->shape, type);
  return tensor;
}

DevTensor* Qwen38GraphEngine::NewDeviceTensor(std::map<std::string, std::unique_ptr<DevTensor>>& owner,
                                              const std::string& name, std::vector<int64_t> shape,
                                              ONNXTensorElementDataType type) {
  const size_t bytes = Elements(shape) * Ort::SizeOf(type);
  owned_.emplace_back(plane_, bytes, stream_);
  owner[name] = MakeTensor(owned_.back().data(), std::move(shape), type);
  return owner[name].get();
}

void Qwen38GraphEngine::SyncStream() {
  plane_.Check(plane_.stream_sync(stream_), "cudaStreamSynchronize");
}

void Qwen38GraphEngine::BuildTarget() {
  OrtSession& session = *model_.session_target_;
  const SessionInfo& info = model_.session_info_;
  const auto in_names = session.GetInputNames();
  const auto out_names = session.GetOutputNames();

  enum class Kind { kStore,
                    kShared,
                    kBank,
                    kRow,
                    kFold,
                    kHeadIn };
  std::map<std::string, Kind> kinds;
  std::vector<std::string> bank_names;
  ple_layer_ = -1;
  for (const auto& name : in_names) {
    if (fold_ && StartsWith(name, "fold.")) {
      kinds[name] = Kind::kFold;
    } else if (fold_ && StartsWith(name, "head.past_key_values.0.")) {
      kinds[name] = Kind::kHeadIn;
    } else if (name == "past_position_ids") {
      kinds[name] = Kind::kStore;
    } else if (StartsWith(name, "past_key_values.")) {
      const size_t dot = name.find('.', 16);
      const std::string suffix = name.substr(dot + 1);
      if (suffix == "key" || suffix == "value")
        kinds[name] = Kind::kStore;
      else if (suffix == "index_key")
        kinds[name] = Kind::kShared;
      else {
        kinds[name] = Kind::kBank;
        bank_names.push_back(name);
      }
    } else {
      kinds[name] = Kind::kRow;
      if (StartsWith(name, "ple_embeddings.")) {
        if (ple_layer_ >= 0) throw std::runtime_error("qwen38_graph: exactly one ple_embeddings input is supported");
        ple_layer_ = std::stoi(name.substr(15));
        ple_width_ = static_cast<int>(info.GetInputShape(name).back());
      }
    }
  }
  if (ple_layer_ < 0) throw std::runtime_error("qwen38_graph: target has no ple_embeddings input");
  // PrepRow fills inputs_embeds from the token ids (the plane's embedding gather over the device-mapped embedding table); a
  // target that takes them cannot run without either, and must never run on zero embeddings.
  if (info.HasInput("inputs_embeds")) {
    if (!plane_.embed_gather && (model_.embed_is_int8_ || !plane_.embed_rows))
      throw std::runtime_error(
          "qwen38_graph: the target takes inputs_embeds but the plane library lacks the qwen38_plane_embed_gather "
          "export (a BF16 embedding table can also use the older qwen38_plane_embed_rows export)");
    if (!model_.embed_mapping_)
      throw std::runtime_error(
          "qwen38_graph: the target takes inputs_embeds, which are gathered from the embedding table mapped from "
          "<model.embedding.filename>.data (at least vocab_size * hidden_size * 2 bytes, or vocab_size * hidden_size + "
          "vocab_size * 4 bytes, plus at most 4096, for a per-row INT8 table); that file is missing, truncated or of "
          "another size");
  }

  // Output facts
  const std::string hidden_name = info.HasOutput("hidden_states_out") ? "hidden_states_out" : "hidden_states";
  hidden_width_ = static_cast<int>(info.GetOutputShape(hidden_name).back());
  const auto val_shape = info.GetOutputShape("validation.aggregate");
  status_count_ = (!val_shape.empty() && val_shape.at(0) > 0) ? static_cast<size_t>(val_shape.at(0)) : 24;
  if (!info.HasOutput("next_tokens") || !info.HasOutput("logits"))
    throw std::runtime_error("qwen38_graph: target must expose logits and next_tokens outputs");
  if (const auto metadata = session.GetModelMetadata()) {
    const std::string labels = metadata->LookupCustomMetadataMap("experiment.validation_status_names");
    bool inside = false;
    std::string current;
    for (char c : labels) {
      if (c == '"') {
        if (inside) status_labels_.push_back(current), current.clear();
        inside = !inside;
      } else if (inside) {
        current.push_back(c);
      }
    }
  }
  if ((!val_shape.empty() && val_shape.at(0) <= 0) && !status_labels_.empty()) {
    status_count_ = status_labels_.size();
  }

  auto with_batch = [](std::vector<int64_t> shape, int batch) {
    if (!shape.empty()) shape[0] = batch;
    return shape;
  };

  // Store (KV, positions) and shared (index_key): one in-place buffer each, bound as input and output.
  for (const auto& name : in_names) {
    if (kinds[name] == Kind::kStore) {
      const auto slot_shape = FixedShape(info.GetInputShape(name));
      const auto dtype = info.GetInputDataType(name);
      store_slot_shape_[name] = slot_shape;
      store_slot_bytes_[name] = Elements(slot_shape) * Ort::SizeOf(dtype);
      NewDeviceTensor(store_, name, with_batch(slot_shape, max_batch_), dtype);
    } else if (kinds[name] == Kind::kShared) {
      const auto slot_shape = FixedShape(info.GetInputShape(name));
      const auto dtype = info.GetInputDataType(name);
      shared_slot_shape_[name] = slot_shape;
      shared_slot_bytes_[name] = Elements(slot_shape) * Ort::SizeOf(dtype);
      NewDeviceTensor(shared_, name, with_batch(slot_shape, max_batch_), dtype);
    }
  }

  // Detect multi-draft verification width from target graph outputs (state_update.* -> S = capacity+1,
  // present_mid3 -> S=4, present_mid2 -> S=3, else S=2).
  verify_rows_ = 2;
  compact_state_updates_ = false;
  state_update_capacity_ = 0;
  for (const auto& name : out_names) {
    if (StartsWith(name, "state_update.")) {
      compact_state_updates_ = true;
      const auto su_shape = FixedShape(info.GetOutputShape(name));
      const auto su_dtype = info.GetOutputDataType(name);
      su_slot_shape_[name] = su_shape;
      su_slot_bytes_[name] = Elements(su_shape) * Ort::SizeOf(su_dtype);
      NewDeviceTensor(state_update_tensors_, name, with_batch(su_shape, max_batch_), su_dtype);
      if (name.find(".conv_value") != std::string::npos || name.find(".indexer_value") != std::string::npos) {
        if (su_shape.size() >= 2 && su_shape[1] > 0) {
          state_update_capacity_ = std::max(state_update_capacity_, static_cast<int>(su_shape[1]));
        }
      }
    } else if (StartsWith(name, "present_mid3.") || StartsWith(name, "head.present_mid3.")) {
      verify_rows_ = std::max(verify_rows_, 4);
    } else if (StartsWith(name, "present_mid2.") || StartsWith(name, "head.present_mid2.")) {
      verify_rows_ = std::max(verify_rows_, 3);
    }
  }
  slot_ledgers_.assign(max_batch_, Qwen38DecodeLedger{capacity_});
  if (compact_state_updates_) {
    if (state_update_capacity_ <= 0 || state_update_capacity_ + 1 > kMaxVerifyRows)
      throw std::runtime_error("qwen38_graph: invalid compact state_update_capacity " +
                               std::to_string(state_update_capacity_));
    verify_rows_ = state_update_capacity_ + 1;
    num_banks_ = 2;
    ledger_.SetCompactPingPong(true, verify_rows_);
    for (auto& sl : slot_ledgers_) sl.SetCompactPingPong(true, verify_rows_);
  } else {
    num_banks_ = fold_ ? (verify_rows_ + 1) : kBanks;
    ledger_.SetNumBanks(num_banks_);
    for (auto& sl : slot_ledgers_) sl.SetNumBanks(num_banks_);
  }
  ValidateQwen38BatchedVerifyRows(static_cast<size_t>(max_batch_), verify_rows_);
  // A verify step writes verify_rows_ rows past the committed length, so a store no longer than ReservedTail() could not hold
  // even an empty request: model.qwen38_graph.kv_capacity is validated against the smallest tail (kReservedTail) only.
  if (capacity_ <= ReservedTail())
    throw std::runtime_error("model.qwen38_graph.kv_capacity must exceed " + std::to_string(ReservedTail()) +
                             ", the positions a verify step of " + std::to_string(verify_rows_) + " rows keeps free");
  // Recurrent banks: flat 256-byte-aligned arenas, identical layout in every bank (the extra scratch bank is only used by S=1).
  size_t arena_bytes = 0;
  std::map<std::string, size_t> offsets;
  for (const auto& name : bank_names) {
    arena_bytes = AlignUp(arena_bytes, 256);
    offsets[name] = arena_bytes;
    const auto slot_shape = FixedShape(info.GetInputShape(name));
    const auto dtype = info.GetInputDataType(name);
    bank_slot_shape_[name] = slot_shape;
    bank_slot_bytes_[name] = Elements(slot_shape) * Ort::SizeOf(dtype);
    arena_bytes += static_cast<size_t>(max_batch_) * bank_slot_bytes_[name];
  }
  arena_bytes = AlignUp(arena_bytes, 256);
  const int alloc_banks = fold_ ? num_banks_ : (num_banks_ + 1);
  for (int b = 0; b < alloc_banks; ++b) {
    Bank& bank = banks_[b];
    bank.arena = PlaneBuffer(plane_, arena_bytes, stream_);
    for (const auto& name : bank_names)
      bank.tensors[name] = MakeTensor(bank.arena.data() + offsets[name],
                                      with_batch(bank_slot_shape_[name], max_batch_),
                                      info.GetInputDataType(name));
  }

  if (compact_state_updates_) {
    if (!plane_.replay_state_updates)
      throw std::runtime_error("qwen38_graph: plane library lacks qwen38_plane_replay_state_updates");
    std::vector<std::string> state_update_names;
    state_update_names.reserve(state_update_tensors_.size());
    for (const auto& [su_name, _] : state_update_tensors_) state_update_names.push_back(su_name);
    RequireQwen38ReplayCoverage(bank_names, state_update_names);
    for (int k = 0; k < 2; ++k) {
      const Bank& src_bank = banks_[k];
      const Bank& dst_bank = banks_[k ^ 1];
      // The replay kernels see ONE slot at a time: every descriptor addresses the slot's slice of the batched tensors, and
      // the validator is given the batch-1 shape of the slot (the slot dimension belongs to the device, not to the state).
      for (int s = 0; s < max_batch_; ++s) {
        std::vector<Qwen38StateReplayDesc> fast_descs;
        std::vector<Qwen38StateReplayDesc> generic_descs;
        for (const auto& [su_name, su_tensor] : state_update_tensors_) {
          const size_t dot1 = su_name.find('.');
          const size_t dot2 = su_name.find('.', dot1 + 1);
          const std::string lid_str = su_name.substr(dot1 + 1, dot2 - dot1 - 1);
          const std::string kind_str = su_name.substr(dot2 + 1);
          const size_t su_off = static_cast<size_t>(s) * su_slot_bytes_.at(su_name);
          // The tensors the replay of this layer reads from the source bank and writes to the destination bank; a package that
          // lacks one fails with its name instead of a bare out_of_range.
          const auto source_tensor = [&](const std::string& name) -> const DevTensor& {
            return RequireQwen38ReplayTensor(src_bank.tensors, name, lid_str, "bank " + std::to_string(k));
          };
          const auto destination_tensor = [&](const std::string& name) -> const DevTensor& {
            return RequireQwen38ReplayTensor(dst_bank.tensors, name, lid_str, "bank " + std::to_string(k ^ 1));
          };
          const auto state_of = [&](const std::string& name, const DevTensor& tensor) {
            return ReplayTensorOf(name, bank_slot_shape_.at(name), tensor);
          };
          const auto update_of = [&] { return ReplayTensorOf(su_name, su_slot_shape_.at(su_name), *su_tensor); };
          if (kind_str == "recurrent_capsule") {
            const std::string state_name = "past_key_values." + lid_str + ".recurrent_state";
            const DevTensor& src_t = source_tensor(state_name);
            const DevTensor& dst_t = destination_tensor(state_name);
            const size_t st_off = static_cast<size_t>(s) * bank_slot_bytes_.at(state_name);
            Qwen38ReplayRequest request;
            request.kind = Qwen38ReplayKind::kGatedDeltaNet;
            request.capacity = state_update_capacity_;
            request.state = state_of(state_name, src_t);
            request.update = update_of();
            const Qwen38ReplayGeometry geometry = ValidateQwen38ReplayDescriptor(request);
            const uint64_t cap = static_cast<uint64_t>(state_update_capacity_);
            const float* capsule = reinterpret_cast<const float*>(su_tensor->ptr + su_off);
            const float* decay = capsule;
            const float* key = decay + cap * geometry.channel_count;
            const float* delta = key + cap * geometry.key_head_count * geometry.key_width;
            Qwen38StateReplayDesc d{};
            d.source_state = src_t.ptr + st_off;
            d.destination_state = dst_t.ptr + st_off;
            d.decay = decay;
            d.key = key;
            d.delta = delta;
            d.channel_count = geometry.channel_count;
            d.state_width = geometry.state_width;
            d.key_width = geometry.key_width;
            d.key_head_count = geometry.key_head_count;
            d.capacity = static_cast<uint32_t>(state_update_capacity_);
            d.kept_count = 1;
            d.element_size = geometry.element_size;
            d.kind = 2;
            fast_descs.push_back(d);
            replay_fast_blocks_ = std::max(replay_fast_blocks_, geometry.blocks);
          } else if (kind_str == "conv_value" || kind_str == "ple_conv_value") {
            const std::string state_tail = (kind_str == "ple_conv_value") ? "ple_conv_state" : "conv_state";
            const std::string state_name = "past_key_values." + lid_str + "." + state_tail;
            const DevTensor& src_t = source_tensor(state_name);
            const DevTensor& dst_t = destination_tensor(state_name);
            const size_t st_off = static_cast<size_t>(s) * bank_slot_bytes_.at(state_name);
            Qwen38ReplayRequest request;
            request.kind = Qwen38ReplayKind::kConvolution;
            request.capacity = state_update_capacity_;
            request.state = state_of(state_name, src_t);
            request.update = update_of();
            const Qwen38ReplayGeometry geometry = ValidateQwen38ReplayDescriptor(request);
            Qwen38StateReplayDesc d{};
            d.source_state = src_t.ptr + st_off;
            d.destination_state = dst_t.ptr + st_off;
            d.value = su_tensor->ptr + su_off;
            d.channel_count = geometry.channel_count;
            d.state_width = geometry.state_width;
            d.capacity = static_cast<uint32_t>(state_update_capacity_);
            d.kept_count = 1;
            d.element_size = geometry.element_size;
            d.kind = 1;
            generic_descs.push_back(d);
          } else if (kind_str == "indexer_value") {
            const std::string buf_name = "past_key_values." + lid_str + ".index_buffer";
            const std::string len_name = "past_key_values." + lid_str + ".index_lengths";
            const DevTensor& src_buf = source_tensor(buf_name);
            const DevTensor& dst_buf = destination_tensor(buf_name);
            const DevTensor& src_len = source_tensor(len_name);
            const DevTensor& dst_len = destination_tensor(len_name);
            const size_t buf_off = static_cast<size_t>(s) * bank_slot_bytes_.at(buf_name);
            const size_t len_off = static_cast<size_t>(s) * bank_slot_bytes_.at(len_name);
            Qwen38ReplayRequest request;
            request.kind = Qwen38ReplayKind::kIndexer;
            request.capacity = state_update_capacity_;
            request.state = state_of(buf_name, src_buf);
            request.update = update_of();
            request.lengths = state_of(len_name, src_len);
            request.compress_ratio = kQwen38IndexKeyRatio;
            const Qwen38ReplayGeometry geometry = ValidateQwen38ReplayDescriptor(request);
            Qwen38StateReplayDesc d{};
            d.value = su_tensor->ptr + su_off;
            d.source_aux_state = src_buf.ptr + buf_off;
            d.destination_aux_state = dst_buf.ptr + buf_off;
            d.source_lengths = reinterpret_cast<const int32_t*>(src_len.ptr + len_off);
            d.destination_lengths = reinterpret_cast<int32_t*>(dst_len.ptr + len_off);
            d.state_width = geometry.state_width;
            d.state_capacity = 0;
            d.aux_capacity = geometry.aux_capacity;
            d.capacity = static_cast<uint32_t>(state_update_capacity_);
            d.kept_count = 1;
            d.element_size = geometry.element_size;
            d.compress_ratio = static_cast<uint32_t>(kQwen38IndexKeyRatio);
            d.kind = 4;
            generic_descs.push_back(d);
            if (k == 0) {
              // Bank 0 as the source and bank 1 as the destination cover both banks of the slot (see ResetSlotIndexLengths).
              slot_index_lengths_[s].push_back(reinterpret_cast<int32_t*>(src_len.ptr + len_off));
              slot_index_lengths_[s].push_back(reinterpret_cast<int32_t*>(dst_len.ptr + len_off));
            }
          } else {
            throw std::runtime_error("qwen38_graph: unsupported state_update output " + su_name);
          }
        }
        replay_fast_count_ = static_cast<int>(fast_descs.size());
        replay_generic_count_ = static_cast<int>(generic_descs.size());
        std::vector<Qwen38StateReplayDesc> all_descs = std::move(fast_descs);
        all_descs.insert(all_descs.end(), generic_descs.begin(), generic_descs.end());
        const size_t desc_bytes = all_descs.size() * sizeof(Qwen38StateReplayDesc);
        slot_replay_descs_dev_[k][s] = PlaneBuffer(plane_, desc_bytes, stream_);
        plane_.Check(plane_.memcpy_h2d(slot_replay_descs_dev_[k][s].data(), all_descs.data(), desc_bytes, stream_),
                     "upload slot state replay descs");
        if (s == 0) {
          replay_descs_dev_[k] = PlaneBuffer(plane_, desc_bytes, stream_);
          plane_.Check(plane_.memcpy_h2d(replay_descs_dev_[k].data(), all_descs.data(), desc_bytes, stream_),
                       "upload state replay descs");
        }
      }
    }
    SyncStream();
  }

  if (fold_) {
    // fold.* control inputs alias the device control block; fold.ptrs = (pinned result ring address, ring size)
    fold_ptrs_ = PlaneBuffer(plane_, 2 * sizeof(int64_t), stream_);
    const int64_t ptrs[2] = {static_cast<int64_t>(reinterpret_cast<intptr_t>(result_host_)), kRing};
    plane_.Check(plane_.memcpy_h2d(fold_ptrs_.data(), ptrs, sizeof(ptrs), stream_), "fold ptrs");
    SyncStream();  // ptrs lives on the stack
    const auto i64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    fold_tensors_["fold.ctl"] = MakeTensor(ctl_.data(), {verify_rows_ > 2 ? 16 : 8}, i64);
    fold_tensors_["fold.ctx_commit"] = MakeTensor(ctx_commit_.data(), {2}, i64);
    fold_tensors_["fold.ctx_a"] = MakeTensor(ctx_a_.data(), {verify_rows_ > 2 ? 2 * verify_rows_ : 2}, i64);
    fold_tensors_["fold.ctx_b"] = MakeTensor(ctx_a_.data() + 16, {2}, i64);
    fold_tensors_["fold.ptrs"] = MakeTensor(fold_ptrs_.data(), {2}, i64);
  }

  // Per-row-count inputs
  const int min_rows = fold_ ? verify_rows_ : 1;
  const int max_rows = fold_ ? verify_rows_ : 2;
  for (int rows = min_rows; rows <= max_rows; ++rows) {
    RowInputs& in = inputs_[rows - 1];
    auto set64 = [&](DevTensor* t, const std::vector<int64_t>& v) {
      plane_.Check(plane_.memcpy_h2d(t->ptr, v.data(), v.size() * 8, stream_), "input init");
    };
    auto set32 = [&](DevTensor* t, const std::vector<int32_t>& v) {
      plane_.Check(plane_.memcpy_h2d(t->ptr, v.data(), v.size() * 4, stream_), "input init");
    };
    std::vector<int64_t> ones(rows, 1), arange(rows);
    for (int r = 0; r < rows; ++r) arange[r] = r;
    in.input_ids = NewDeviceTensor(in.tensors, "input_ids", {1, rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)->ptr;
    set64(NewDeviceTensor(in.tensors, "attention_mask", {1, rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64), ones);
    in.position_ids = NewDeviceTensor(in.tensors, "position_ids", {1, rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)->ptr;
    in.committed = NewDeviceTensor(in.tensors, "committed_length", {1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)->ptr;
    set32(NewDeviceTensor(in.tensors, "sequence_length_minus_one", {1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
          {rows - 1});
    set32(NewDeviceTensor(in.tensors, "cumulative_sequence_lengths", {2}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
          {0, rows});
    if (info.HasInput("state_update_capture_count"))
      set32(NewDeviceTensor(in.tensors, "state_update_capture_count", {1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
            {rows - 1});
    if (info.HasInput("logits_indices"))
      set64(NewDeviceTensor(in.tensors, "logits_indices", {rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64), arange);
    if (info.HasInput("inputs_embeds")) {
      const int embed_width = model_.config_->model.decoder.hidden_size;
      in.inputs_embeds = NewDeviceTensor(in.tensors, "inputs_embeds", {1, rows, embed_width},
                                         ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)
                             ->ptr;
    }
    in.ple = NewDeviceTensor(in.tensors, "ple_embeddings." + std::to_string(ple_layer_), {1, rows, ple_width_},
                             ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)
                 ->ptr;
  }
  SyncStream();

  // One IO binding per (rows, ring phase). Graph k reads bank k; S=1 writes bank k+1 (present_mid -> scratch bank),
  // S=R writes present_mid -> bank k+1, present_mid2 -> bank k+2, present_mid3 -> bank k+3, and present -> bank k+R
  // (or bank k^1 when compact_state_updates_ is enabled).
  const auto dtype_logits = info.GetOutputDataType("logits");
  for (int rows = min_rows; rows <= max_rows; ++rows) {
    for (int k = 0; k < num_banks_; ++k) {
      Direction& d = dirs_[rows - 1][k];
      d.binding = OrtIoBinding::Create(session);
      Bank& in_bank = banks_[k];
      Bank& out_bank = compact_state_updates_ ? banks_[k ^ 1] : banks_[(k + rows) % num_banks_];
      Bank& mid1_bank = rows >= 2 ? banks_[(k + 1) % num_banks_] : banks_[num_banks_];
      Bank& mid2_bank = rows >= 3 ? banks_[(k + 2) % num_banks_] : banks_[num_banks_];
      Bank& mid3_bank = rows >= 4 ? banks_[(k + 3) % num_banks_] : banks_[num_banks_];
      std::map<std::string, std::unique_ptr<DevTensor>> own;
      auto add_own = [&](const std::string& name, std::vector<int64_t> shape, ONNXTensorElementDataType type) {
        return NewDeviceTensor(own, name, std::move(shape), type);
      };
      auto add_view = [&](const std::string& name, void* ptr, std::vector<int64_t> shape,
                          ONNXTensorElementDataType type) {
        own[name] = MakeTensor(ptr, std::move(shape), type);
        return own[name].get();
      };
      for (const auto& name : in_names) {
        DevTensor* t = nullptr;
        switch (kinds[name]) {
          case Kind::kStore:
            t = max_batch_ == 1 ? store_[name].get()
                                : add_view("in." + name, store_[name]->ptr, store_slot_shape_[name], store_[name]->type);
            break;
          case Kind::kShared:
            t = max_batch_ == 1 ? shared_[name].get()
                                : add_view("in." + name, shared_[name]->ptr, shared_slot_shape_[name], shared_[name]->type);
            break;
          case Kind::kBank:
            t = max_batch_ == 1 ? in_bank.tensors[name].get()
                                : add_view("in." + name, in_bank.tensors[name]->ptr, bank_slot_shape_[name],
                                           in_bank.tensors[name]->type);
            break;
          case Kind::kFold:
            t = fold_tensors_.at(name).get();
            break;
          case Kind::kHeadIn: {
            const std::string tail = name.substr(std::string("head.past_key_values.0.").size());
            if (tail == "index_buffer" || tail == "index_lengths")
              t = head_banks_[k].tensors.at("past_key_values.0." + tail).get();
            else
              t = head_store_.at("past_key_values.0." + tail).get();
            break;
          }
          case Kind::kRow: {
            auto found = inputs_[rows - 1].tensors.find(name);
            if (found == inputs_[rows - 1].tensors.end())
              throw std::runtime_error("qwen38_graph: unsupported target input " + name);
            t = found->second.get();
            break;
          }
        }
        d.binding->BindInput(name.c_str(), *t->value);
      }
      for (const auto& name : out_names) {
        DevTensor* t = nullptr;
        auto state_for = [&](const std::string& tail, Bank& bank, const std::string& tag) -> DevTensor* {
          const std::string input_name = "past_key_values." + tail;
          auto it = kinds.find(input_name);
          if (it == kinds.end()) throw std::runtime_error("qwen38_graph: no input for output " + name);
          if (it->second == Kind::kStore) {
            return max_batch_ == 1 ? store_[input_name].get()
                                   : add_view(tag, store_[input_name]->ptr, store_slot_shape_[input_name],
                                              store_[input_name]->type);
          }
          if (it->second == Kind::kShared) {
            return max_batch_ == 1 ? shared_[input_name].get()
                                   : add_view(tag, shared_[input_name]->ptr, shared_slot_shape_[input_name],
                                              shared_[input_name]->type);
          }
          return max_batch_ == 1 ? bank.tensors[input_name].get()
                                 : add_view(tag, bank.tensors[input_name]->ptr, bank_slot_shape_[input_name],
                                            bank.tensors[input_name]->type);
        };
        auto head_state_for = [&](const std::string& tail, int bank_index) -> DevTensor* {
          if (tail == "index_buffer" || tail == "index_lengths")
            return head_banks_[bank_index % num_banks_].tensors.at("past_key_values.0." + tail).get();
          return head_store_.at("past_key_values.0." + tail).get();
        };
        if (fold_ && StartsWith(name, "head.present_mid.0."))
          t = head_state_for(name.substr(19), k + 1);
        else if (fold_ && StartsWith(name, "head.present_mid2.0."))
          t = head_state_for(name.substr(20), k + 2);
        else if (fold_ && StartsWith(name, "head.present_mid3.0."))
          t = head_state_for(name.substr(20), k + 3);
        else if (fold_ && StartsWith(name, "head.present_s1.0."))
          t = head_state_for(name.substr(18), k + rows);
        else if (fold_ && StartsWith(name, "head.present_s2.0."))
          t = head_state_for(name.substr(18), k + rows);
        else if (fold_ && StartsWith(name, "head.present.0."))
          t = head_state_for(name.substr(15), k + rows);
        else if (fold_ && name == "head.validation.aggregate")
          t = add_own(name, FixedShape(info.GetOutputShape(name)), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
        else if (fold_ && name == "head.next_token")
          t = add_own(name, {1, 1, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        else if (fold_ && name == "fold.tail_done")
          t = add_own(name, {1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
        else if (name == "present_position_ids")
          t = max_batch_ == 1 ? store_["past_position_ids"].get()
                              : add_view(name, store_["past_position_ids"]->ptr,
                                         store_slot_shape_["past_position_ids"], store_["past_position_ids"]->type);
        else if (StartsWith(name, "present_mid."))
          t = state_for(name.substr(12), mid1_bank, name);
        else if (StartsWith(name, "present_mid2."))
          t = state_for(name.substr(13), mid2_bank, name);
        else if (StartsWith(name, "present_mid3."))
          t = state_for(name.substr(13), mid3_bank, name);
        else if (StartsWith(name, "state_update."))
          t = max_batch_ == 1 ? state_update_tensors_.at(name).get()
                              : add_view(name, state_update_tensors_.at(name)->ptr, su_slot_shape_.at(name),
                                         state_update_tensors_.at(name)->type);
        else if (StartsWith(name, "present."))
          t = state_for(name.substr(8), out_bank, name);
        else if (name == "logits")
          t = add_own(name, {1, rows, vocab_}, dtype_logits);
        else if (name == hidden_name)
          t = add_own(name, {1, rows, hidden_width_}, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
        else if (name == "validation.aggregate")
          t = add_own(name, {static_cast<int64_t>(status_count_)}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
        else if (name == "next_tokens")
          t = add_own(name, {1, rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        else if (name == "aux_hidden_states") {
          t = add_own(name, {1, rows, int64_t(kDflashWidth)}, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
          d.auxiliary = t->ptr;
        } else
          throw std::runtime_error("qwen38_graph: unsupported target output " + name);
        d.binding->BindOutput(name.c_str(), *t->value);
      }
      d.logits = own["logits"]->ptr;
      d.hidden = own[hidden_name]->ptr;
      d.status = own["validation.aggregate"]->ptr;
      d.tokens = own["next_tokens"]->ptr;
      d.status_count = status_count_;
      for (auto& entry : own) output_tensors_.push_back(std::move(entry.second));
      d.run_options = OrtRunOptions::Create();
      const int id = Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, rows, k);
      d.run_options->AddConfigEntry("gpu_graph_id", std::to_string(id).c_str());
      d.run_options->AddConfigEntry("disable_synchronize_execution_providers", "1");
    }
  }

  // Build multi-batch (B = 2 .. max_batch_) bindings for verify_rows_ when fold_ is enabled.
  if (fold_ && max_batch_ > 1) {
    const int rows = verify_rows_;
    for (int B = 2; B <= max_batch_; ++B) {
      RowInputs& in = batch_inputs_[B - 1];
      auto set64 = [&](DevTensor* t, const std::vector<int64_t>& v) {
        plane_.Check(plane_.memcpy_h2d(t->ptr, v.data(), v.size() * 8, stream_), "batch input init");
      };
      auto set32 = [&](DevTensor* t, const std::vector<int32_t>& v) {
        plane_.Check(plane_.memcpy_h2d(t->ptr, v.data(), v.size() * 4, stream_), "batch input init");
      };
      std::vector<int64_t> ones(B * rows, 1), arange(B * rows);
      for (int r = 0; r < B * rows; ++r) arange[r] = r;
      std::vector<int32_t> cu_seq(B + 1), cap_cnt(B, rows - 1);
      for (int b = 0; b <= B; ++b) cu_seq[b] = b * rows;

      in.input_ids = NewDeviceTensor(in.tensors, "input_ids", {B, rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)->ptr;
      set64(NewDeviceTensor(in.tensors, "attention_mask", {B, rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64), ones);
      in.position_ids = NewDeviceTensor(in.tensors, "position_ids", {B, rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)->ptr;
      in.committed = NewDeviceTensor(in.tensors, "committed_length", {B}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)->ptr;
      set32(NewDeviceTensor(in.tensors, "sequence_length_minus_one", {1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
            {rows - 1});
      set32(NewDeviceTensor(in.tensors, "cumulative_sequence_lengths", {B + 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
            cu_seq);
      if (info.HasInput("state_update_capture_count"))
        set32(NewDeviceTensor(in.tensors, "state_update_capture_count", {B}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32),
              cap_cnt);
      if (info.HasInput("logits_indices"))
        set64(NewDeviceTensor(in.tensors, "logits_indices", {B * rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64), arange);
      if (info.HasInput("inputs_embeds")) {
        const int embed_width = model_.config_->model.decoder.hidden_size;
        in.inputs_embeds = NewDeviceTensor(in.tensors, "inputs_embeds", {B, rows, embed_width},
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)
                               ->ptr;
      }
      in.ple = NewDeviceTensor(in.tensors, "ple_embeddings." + std::to_string(ple_layer_), {B, rows, ple_width_},
                               ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)
                   ->ptr;

      for (int k = 0; k < num_banks_; ++k) {
        Direction& d = batch_dirs_[B - 1][k];
        d.binding = OrtIoBinding::Create(session);
        Bank& in_bank = banks_[k];
        Bank& out_bank = compact_state_updates_ ? banks_[k ^ 1] : banks_[(k + rows) % num_banks_];
        std::map<std::string, std::unique_ptr<DevTensor>> own;
        auto add_own = [&](const std::string& name, std::vector<int64_t> shape, ONNXTensorElementDataType type) {
          return NewDeviceTensor(own, name, std::move(shape), type);
        };
        auto add_view = [&](const std::string& name, void* ptr, std::vector<int64_t> shape,
                            ONNXTensorElementDataType type) {
          own[name] = MakeTensor(ptr, std::move(shape), type);
          return own[name].get();
        };
        const auto i64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        for (const auto& name : in_names) {
          DevTensor* t = nullptr;
          switch (kinds[name]) {
            case Kind::kStore:
              t = add_view("in." + name, store_[name]->ptr, with_batch(store_slot_shape_[name], B), store_[name]->type);
              break;
            case Kind::kShared:
              t = add_view("in." + name, shared_[name]->ptr, with_batch(shared_slot_shape_[name], B), shared_[name]->type);
              break;
            case Kind::kBank:
              t = add_view("in." + name, in_bank.tensors[name]->ptr, with_batch(bank_slot_shape_[name], B),
                           in_bank.tensors[name]->type);
              break;
            case Kind::kFold:
              // The slots lie kCtlWordsPerSlot and 2 * verify_rows_ words apart, as the plane prepares them
              // (ValidateQwen38BatchedVerifyRows keeps verify_rows_ above 2 for every batch).
              if (name == "fold.ctl")
                t = add_view(name, ctl_.data(), {static_cast<int64_t>(B) * static_cast<int64_t>(kCtlWordsPerSlot)}, i64);
              else if (name == "fold.ctx_commit")
                t = add_view(name, ctx_commit_.data(), {B * 2}, i64);
              else if (name == "fold.ctx_a")
                t = add_view(name, ctx_a_.data(), {B * 2 * verify_rows_}, i64);
              else
                t = fold_tensors_.at(name).get();
              break;
            case Kind::kRow: {
              auto found = in.tensors.find(name);
              if (found == in.tensors.end())
                throw std::runtime_error("qwen38_graph: unsupported batch target input " + name);
              t = found->second.get();
              break;
            }
            default:
              throw std::runtime_error("qwen38_graph: unsupported batch target input kind " + name);
          }
          d.binding->BindInput(name.c_str(), *t->value);
        }
        const size_t batch_status_count = status_count_ + static_cast<size_t>(12 * (B - 1));
        for (const auto& name : out_names) {
          DevTensor* t = nullptr;
          auto state_for = [&](const std::string& tail, Bank& bank, const std::string& tag) -> DevTensor* {
            const std::string input_name = "past_key_values." + tail;
            auto it = kinds.find(input_name);
            if (it == kinds.end()) throw std::runtime_error("qwen38_graph: no input for output " + name);
            if (it->second == Kind::kStore)
              return add_view(tag, store_[input_name]->ptr, with_batch(store_slot_shape_[input_name], B),
                              store_[input_name]->type);
            if (it->second == Kind::kShared)
              return add_view(tag, shared_[input_name]->ptr, with_batch(shared_slot_shape_[input_name], B),
                              shared_[input_name]->type);
            return add_view(tag, bank.tensors[input_name]->ptr, with_batch(bank_slot_shape_[input_name], B),
                            bank.tensors[input_name]->type);
          };
          if (name == "fold.tail_done")
            t = add_own(name, {B}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
          else if (name == "present_position_ids")
            t = add_view(name, store_["past_position_ids"]->ptr,
                         with_batch(store_slot_shape_["past_position_ids"], B), store_["past_position_ids"]->type);
          else if (StartsWith(name, "state_update."))
            t = add_view(name, state_update_tensors_.at(name)->ptr, with_batch(su_slot_shape_.at(name), B),
                         state_update_tensors_.at(name)->type);
          else if (StartsWith(name, "present."))
            t = state_for(name.substr(8), out_bank, name);
          else if (name == "logits")
            t = add_own(name, {B, rows, vocab_}, dtype_logits);
          else if (name == hidden_name)
            t = add_own(name, {B, rows, hidden_width_}, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
          else if (name == "validation.aggregate")
            t = add_own(name, {static_cast<int64_t>(batch_status_count)}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
          else if (name == "next_tokens")
            t = add_own(name, {B, rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
          else if (name == "aux_hidden_states") {
            t = add_own(name, {B, rows, int64_t(kDflashWidth)}, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
            d.auxiliary = t->ptr;
          } else
            throw std::runtime_error("qwen38_graph: unsupported batch target output " + name);
          d.binding->BindOutput(name.c_str(), *t->value);
        }
        d.logits = own["logits"]->ptr;
        d.hidden = own[hidden_name]->ptr;
        d.status = own["validation.aggregate"]->ptr;
        d.tokens = own["next_tokens"]->ptr;
        d.status_count = batch_status_count;
        for (auto& entry : own) output_tensors_.push_back(std::move(entry.second));
        d.run_options = OrtRunOptions::Create();
        const int id = Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, rows, k, B);
        d.run_options->AddConfigEntry("gpu_graph_id", std::to_string(id).c_str());
        d.run_options->AddConfigEntry("disable_synchronize_execution_providers", "1");
      }
    }
    SyncStream();
  }
  past_pos_ = store_["past_position_ids"]->ptr;
}

void Qwen38GraphEngine::BuildHead() {
  head_[0].rows = 1;
  head_[0].session = model_.session_head_s1_.get();
  head_[1].rows = 2;
  head_[1].session = model_.session_head_s2_.get();
  for (auto& h : head_) h.info.Add(*h.session);
  for (const HeadSession& h : head_)
    ValidateQwen38CacheShapes(CacheInputsOf(h.info, "past_key_values.", false), capacity_,
                              h.rows == 1 ? "head (S=1)" : "head (S=2)");
  BuildHeadState();

  for (HeadSession& h : head_) {
    const int rows = h.rows;
    OrtSession& session = *h.session;
    const auto bf16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
    auto* tokens = NewDeviceTensor(h.tensors, "tokens", {1, rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    auto* positions = NewDeviceTensor(h.tensors, "positions", {1, rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    auto* hidden = NewDeviceTensor(h.tensors, "hidden", {1, rows, hidden_width_}, bf16);
    auto* committed = NewDeviceTensor(h.tensors, "committed", {rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    h.tokens = tokens->ptr;
    h.positions = positions->ptr;
    h.hidden = hidden->ptr;
    h.committed = committed->ptr;
    auto* logits = NewDeviceTensor(h.tensors, "logits", {1, 1, vocab_}, h.info.GetOutputDataType("logits"));
    auto* hidden_out = NewDeviceTensor(h.tensors, "hidden_states_out", {1, 1, hidden_width_}, bf16);
    auto* next_token = NewDeviceTensor(h.tensors, "next_token", {1, 1, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    h.status_count = static_cast<size_t>(h.info.GetOutputShape("validation.aggregate").at(0));
    auto* status = NewDeviceTensor(h.tensors, "validation.aggregate", {static_cast<int64_t>(h.status_count)},
                                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    h.next_token = next_token->ptr;
    h.status = status->ptr;

    // Row views: rows==1 binds the whole buffers, rows==2 binds [1,1] slices named input_ids.0 / .1 ...
    std::map<std::string, std::unique_ptr<DevTensor>> views;
    if (rows == 1) {
      views["input_ids"] = MakeTensor(tokens->ptr, {1, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      views["position_ids"] = MakeTensor(positions->ptr, {1, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      views["hidden_states"] = MakeTensor(hidden->ptr, {1, 1, hidden_width_}, bf16);
      views["committed_length"] = MakeTensor(committed->ptr, {1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    } else {
      for (int r = 0; r < rows; ++r) {
        const std::string s = "." + std::to_string(r);
        views["input_ids" + s] = MakeTensor(tokens->ptr + 8 * r, {1, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        views["position_ids" + s] = MakeTensor(positions->ptr + 8 * r, {1, 1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
        views["hidden_states" + s] =
            MakeTensor(hidden->ptr + static_cast<size_t>(r) * hidden_width_ * 2, {1, 1, hidden_width_}, bf16);
        views["committed_length" + s] = MakeTensor(committed->ptr + 4 * r, {1}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
      }
    }
    const auto in_names = session.GetInputNames();
    const auto out_names = session.GetOutputNames();
    for (int dir = 0; dir < 2; ++dir) {
      Direction& d = h.dirs[dir];
      d.binding = OrtIoBinding::Create(session);
      HeadBank& bank_in = head_banks_[dir];
      HeadBank& bank_out = head_banks_[1 - dir];
      for (const auto& name : in_names) {
        DevTensor* t = nullptr;
        if (views.count(name))
          t = views[name].get();
        else if (head_store_.count(name))
          t = head_store_[name].get();
        else if (bank_in.tensors.count(name))
          t = bank_in.tensors[name].get();
        else
          throw std::runtime_error("qwen38_graph: unsupported head input " + name);
        d.binding->BindInput(name.c_str(), *t->value);
      }
      for (const auto& name : out_names) {
        DevTensor* t = nullptr;
        const std::string in_name = "past_key_values." + name.substr(name.find('.') + 1);
        if (name == "logits")
          t = logits;
        else if (name == "hidden_states_out")
          t = hidden_out;
        else if (name == "next_token")
          t = next_token;
        else if (name == "validation.aggregate")
          t = status;
        else if (StartsWith(name, "present_mid."))
          t = head_store_.at(in_name).get();
        else if (StartsWith(name, "present.")) {
          if (head_store_.count(in_name))
            t = head_store_[in_name].get();
          else
            t = bank_out.tensors.at(in_name).get();
        } else
          throw std::runtime_error("qwen38_graph: unsupported head output " + name);
        d.binding->BindOutput(name.c_str(), *t->value);
      }
      d.run_options = OrtRunOptions::Create();
      const int id = Qwen38GraphAnnotationId(Qwen38GraphRun::kHead, rows, dir);
      d.run_options->AddConfigEntry("gpu_graph_id", std::to_string(id).c_str());
      d.run_options->AddConfigEntry("disable_synchronize_execution_providers", "1");
    }
    for (auto& view : views) head_views_.push_back(std::move(view.second));
  }
  SyncStream();
}

// Head KV / compressed index keys in place, indexer small state in 2 ping-pong banks (S=1/S=2 head sessions) or in the 3-bank
// ring of the fold graph.
void Qwen38GraphEngine::BuildHeadState() {
  const SessionInfo& info = fold_ ? model_.session_info_ : head_[0].info;
  const std::string prefix = fold_ ? "head.past_key_values.0." : "past_key_values.0.";
  if (!model_.config_->model.dflash2.filename.empty() && !info.HasInput(prefix + "key")) return;
  for (const char* tail : {"key", "value", "index_key"})
    NewDeviceTensor(head_store_, std::string("past_key_values.0.") + tail, FixedShape(info.GetInputShape(prefix + tail)),
                    info.GetInputDataType(prefix + tail));
  const int banks = fold_ ? kMaxBanks : 2;
  for (int b = 0; b < banks; ++b) {
    for (const char* tail : {"index_buffer", "index_lengths"})
      NewDeviceTensor(head_banks_[b].tensors, std::string("past_key_values.0.") + tail,
                      FixedShape(info.GetInputShape(prefix + tail)), info.GetInputDataType(prefix + tail));
  }
}

void Qwen38GraphEngine::ClearHead() {
  for (auto& entry : head_store_) plane_.Check(plane_.device_memset(entry.second->ptr, 0, entry.second->Bytes(), stream_), "clear head");
  for (auto& bank : head_banks_)
    for (auto& entry : bank.tensors)
      plane_.Check(plane_.device_memset(entry.second->ptr, 0, entry.second->Bytes(), stream_), "clear head bank");
  head_length_ = 0;
  head_turn_ = 0;
  head_cur_ = 0;
}

void Qwen38GraphEngine::MarkContextFailed() noexcept { model_.PoisonExecutionContext(); }

bool Qwen38GraphEngine::ContextIsHealthy() const noexcept { return model_.IsExecutionContextHealthy(); }

void Qwen38GraphEngine::Reset() {
  SyncStream();
  if (dflash_) dflash_->ReleaseAll();
  for (auto& features : slot_dflash_prompt_) features.Reset();
  for (auto& bank : banks_)
    if (bank.arena.data()) plane_.Check(plane_.device_memset(bank.arena.data(), 0, bank.arena.size(), stream_), "clear bank");
  for (auto& entry : store_) plane_.Check(plane_.device_memset(entry.second->ptr, 0, entry.second->Bytes(), stream_), "clear store");
  for (auto& entry : shared_) plane_.Check(plane_.device_memset(entry.second->ptr, 0, entry.second->Bytes(), stream_), "clear index");
  plane_.Check(plane_.device_memset(ctl_.data(), 0, ctl_.size(), stream_), "clear ctl");
  std::memset(result_host_, 0, static_cast<size_t>(max_batch_) * kRing * 8 * sizeof(int64_t));
  stage_host_[0] = stage_host_[1] = ple_->params.eos;
  for (int s = 0; s < max_batch_; ++s) {
    plane_.Check(plane_.memcpy_h2d(ctx_commit_slot(s), stage_host_, 16, stream_), "ctx init");
  }
  if (mtp_enabled_) ClearHead();
  SyncStream();
  FastReset();
  run_hints_.BeginAllTurns();
  seq_ = 0;
  bank_phase_ = 0;
  inflight_batch_ = 1;
  ledger_.Reset();
  for (int s = 0; s < max_batch_; ++s) {
    if (static_cast<size_t>(s) < slot_ledgers_.size()) slot_ledgers_[s].Reset();
    slot_seq_[s] = 0;
    slot_pending_[s] = -1;
    slot_deferred_aux_rows_[s] = 0;
    slot_deferred_aux_pos_[s] = 0;
    slot_decode_started_[s] = false;
  }
  inflight_ = false;
  decode_started_ = false;
  pending_ = -1;
  Qwen38ForgetPromptTokens(prompt_tokens_);
  suffix_drafter_.Reset();
  last_hidden_ptr_ = nullptr;
}

void Qwen38GraphEngine::ResetSlot(size_t slot_idx) {
  if (slot_idx >= static_cast<size_t>(max_batch_)) {
    throw std::out_of_range("qwen38_graph: slot index out of range");
  }
  if (max_batch_ == 1 && slot_idx == 0) {
    Reset();
    return;
  }
  const int s = static_cast<int>(slot_idx);
  SyncStream();
  if (dflash_) dflash_->Release(SlotRequestKey(s));
  const int alloc_banks = fold_ ? num_banks_ : (num_banks_ + 1);
  for (int b = 0; b < alloc_banks; ++b) {
    for (const auto& [name, t] : banks_[b].tensors) {
      const size_t sb = bank_slot_bytes_.at(name);
      plane_.Check(plane_.device_memset(t->ptr + static_cast<size_t>(s) * sb, 0, sb, stream_), "clear slot bank");
    }
  }
  for (const auto& [name, t] : store_) {
    const size_t sb = store_slot_bytes_.at(name);
    plane_.Check(plane_.device_memset(t->ptr + static_cast<size_t>(s) * sb, 0, sb, stream_), "clear slot store");
  }
  for (const auto& [name, t] : shared_) {
    const size_t sb = shared_slot_bytes_.at(name);
    plane_.Check(plane_.device_memset(t->ptr + static_cast<size_t>(s) * sb, 0, sb, stream_), "clear slot shared");
  }
  plane_.Check(plane_.device_memset(ctl_slot(s), 0, kCtlWordsPerSlot * sizeof(int64_t), stream_), "clear slot ctl");
  std::memset(result_ring_slot(s), 0, kRing * 8 * sizeof(int64_t));
  stage_host_[0] = stage_host_[1] = ple_->params.eos;
  plane_.Check(plane_.memcpy_h2d(ctx_commit_slot(s), stage_host_, 16, stream_), "slot ctx init");
  const int64_t muted_val = 1;
  WriteCtlSlot(s, 3, {&muted_val, 1});
  SyncStream();
  run_hints_.BeginTurn(slot_idx);
  slot_ledgers_[s].Reset();
  slot_seq_[s] = 0;
  slot_pending_[s] = -1;
  slot_dflash_prompt_[s].Reset();
  slot_deferred_aux_rows_[s] = 0;
  slot_deferred_aux_pos_[s] = 0;
  slot_decode_started_[s] = false;
  if (s == 0) {
    seq_ = 0;
    ledger_.Reset();
    pending_ = -1;
    Qwen38ForgetPromptTokens(prompt_tokens_);
    decode_started_ = false;
  }
}

void Qwen38GraphEngine::PrepRow(int rows, int row, const void* tok, const void* pos, int add, const void* ctx_in,
                                void* ctx_out) {
  const RowInputs& in = inputs_[rows - 1];
  plane_.Check(plane_.prep(stream_, params_dev_.data(), tok, pos, add, ctx_in, ctx_out, in.input_ids + 8 * row,
                           in.position_ids + 8 * row, row == 0 ? in.committed : scratch_i32_.data(), past_pos_,
                           in.ple + 2 * static_cast<size_t>(ple_width_) * row, ctl_.data()),
               "prep");
  if (row + 1 == rows && in.inputs_embeds) {
    plane_.EmbedRows(stream_, model_.EmbeddingTable(), in.input_ids, in.inputs_embeds, rows);
  }
}

// The fold rows of the first `batch` request slots in one call of the plane's batched prep.
void Qwen38GraphEngine::PrepBatch(int batch) {
  const RowInputs& in = (batch == 1) ? inputs_[verify_rows_ - 1] : batch_inputs_[batch - 1];
  plane_.PrepBatched(stream_, params_dev_.data(), verify_rows_, batch, ctx_commit_.data(), ctx_a_.data(), in.input_ids,
                     in.position_ids, in.committed, past_pos_, in.ple, ctl_.data());
  if (in.inputs_embeds) {
    plane_.EmbedRows(stream_, model_.EmbeddingTable(), in.input_ids, in.inputs_embeds, batch * verify_rows_);
  }
}

void Qwen38GraphEngine::RunTarget(int rows, int phase, const Qwen38PromptRunHints* prompt) {
  Direction& d = dirs_[rows - 1][phase % num_banks_];
  RunSession(*model_.session_target_, d.run_options.get(), *d.binding, prompt);
}

void Qwen38GraphEngine::RunSession(OrtSession& session, const OrtRunOptions* run_options, const OrtIoBinding& binding,
                                   const Qwen38PromptRunHints* prompt, Qwen38SlotMask participants) {
  RunWithQwen38OrtHints(run_hints_, prompt, [&] { session.Run(run_options, binding); }, participants);
}

void Qwen38GraphEngine::ExpectPrefillTokensSlot(size_t slot, size_t total_tokens) {
  if (slot >= static_cast<size_t>(max_batch_)) throw std::out_of_range("qwen38_graph: slot index out of range");
  run_hints_.ExpectPrefillTokens(slot, total_tokens);
}

void Qwen38GraphEngine::ReplayStateUpdates(int src_bank, size_t kept_count) {
  ReplayStateUpdatesSlot(src_bank, 0, kept_count);
}

void Qwen38GraphEngine::ReplayStateUpdatesSlot(int src_bank, int slot, size_t kept_count) {
  if (!compact_state_updates_ || kept_count >= static_cast<size_t>(verify_rows_)) return;
  const int b = src_bank % 2;
  plane_.Check(plane_.replay_state_updates(stream_, slot_replay_descs_dev_[b][slot].data(), replay_fast_count_,
                                           replay_generic_count_, replay_fast_blocks_, static_cast<int>(kept_count)),
               "replay compact state updates");
}

int64_t Qwen38GraphEngine::ReadCtl(int index) {
  return ReadCtlSlot(0, index);
}

void Qwen38GraphEngine::WriteCtl(int first, std::span<const int64_t> values) {
  WriteCtlSlot(0, first, values);
}

int64_t Qwen38GraphEngine::ReadCtlSlot(int slot, int index) {
  plane_.Check(plane_.memcpy_d2h(stage_host_, ctl_slot(slot) + 8 * index, 8, stream_), "read ctl");
  SyncStream();
  return stage_host_[0];
}

void Qwen38GraphEngine::WriteCtlSlot(int slot, int first, std::span<const int64_t> values) {
  std::memcpy(stage_host_ + 4, values.data(), values.size() * 8);
  plane_.Check(plane_.memcpy_h2d(ctl_slot(slot) + 8 * first, stage_host_ + 4, values.size() * 8, stream_), "write ctl");
  SyncStream();
}

void Qwen38GraphEngine::CheckDeviceStatus(const char* where, const Direction* dir) {
  std::ostringstream message;
  message << "qwen38_graph: device validation failed during " << where;
  if (dir) {
    std::vector<int32_t> codes(dir->status_count);
    plane_.Check(plane_.memcpy_d2h(codes.data(), dir->status, codes.size() * 4, stream_), "read status");
    SyncStream();
    for (size_t i = 0; i < codes.size(); ++i) {
      if (codes[i] != 0)
        message << " [" << (i < status_labels_.size() ? status_labels_[i] : std::to_string(i)) << "=" << codes[i] << "]";
    }
  }
  throw std::runtime_error(message.str());
}

// ---- prompt ingestion -------------------------------------------------------------------------
size_t EffectiveQwen38PrefillChunkTokens(const Config::Model::Qwen38Graph& q) {
  constexpr int kStepChunk = 2048;   // prompt tokens per Engine step through the decode graphs
  constexpr int kDenseChunk = 8192;  // prompt tokens per call of the dense prefill session
  if (q.prefill.chunk > 0) return static_cast<size_t>(q.prefill.chunk);
  return static_cast<size_t>(q.prefill.enabled ? kDenseChunk : kStepChunk);
}

void ValidateQwen38PrefillChunk(const Config::Model::Qwen38Graph& q) {
  if (q.prefill.chunk < 0)
    throw std::runtime_error("model.qwen38_graph.prefill.chunk must not be negative (0 selects the default)");
  const size_t chunk = EffectiveQwen38PrefillChunkTokens(q);
  if (q.prefill.enabled) {
    if (chunk > static_cast<size_t>(std::max(q.kv_capacity, 0)))
      throw std::runtime_error("model.qwen38_graph.prefill.chunk must be in [1, kv_capacity]");
  } else if ((chunk % 2) != 0) {
    throw std::runtime_error(
        "model.qwen38_graph.prefill.chunk must be even when the prompt is ingested by the decode graphs "
        "(prefill.enabled is false)");
  }
}

size_t Qwen38GraphEngine::PrefillChunkTokens() const {
  return fast_ ? static_cast<size_t>(fast_->chunk) : EffectiveQwen38PrefillChunkTokens(model_.config_->model.qwen38_graph);
}

void ValidateQwen38PromptTokens(std::span<const int32_t> tokens, int vocab_size) {
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (tokens[i] < 0 || tokens[i] >= vocab_size) {
      throw std::invalid_argument("qwen38_graph: prompt token " + std::to_string(i) + " has id " +
                                  std::to_string(tokens[i]) + ", outside [0, " + std::to_string(vocab_size) +
                                  ") (model.vocab_size)");
    }
  }
}

// ctl[3] = mute: the fold graph runs without its decode bookkeeping while it ingests prompt rows.
void Qwen38GraphEngine::MutedPrefill(bool muted) {
  MutedPrefillSlot(0, muted);
}

void Qwen38GraphEngine::MutedPrefillSlot(int slot, bool muted) {
  if (!fold_) return;
  const int64_t value = muted ? 1 : 0;
  WriteCtlSlot(slot, 3, {&value, 1});
}

void Qwen38GraphEngine::MuteSlot(int slot) {
  if (!fold_ || slot < 0 || slot >= max_batch_) return;
  const int64_t state[4] = {0, 0, 0, 1};
  WriteCtlSlot(slot, 0, std::span<const int64_t>(state, 4));
  ResetSlotIndexLengths(slot);
}

// A muted slot holds no tokens, so the lengths of its indexer buffers are zero in both banks. The replay kernels keep at least
// one transition (qwen38_plane_replay_state_updates rejects a kept count of zero), so the lengths are cleared here instead.
void Qwen38GraphEngine::ResetSlotIndexLengths(int slot) {
  for (int32_t* lengths : slot_index_lengths_[slot]) {
    plane_.Check(plane_.device_memset(lengths, 0, 2 * sizeof(int32_t), stream_), "clear the indexer lengths of a muted slot");
  }
}

void Qwen38GraphEngine::PrefillChunk(std::span<const int32_t> tokens) {
  // Before any device copy or state change: a rejected chunk leaves the engine as it was.
  ValidateQwen38PromptTokens(tokens, vocab_);
  if (fast_)
    FastAppend(tokens);
  else
    StepPrefill(tokens);
  if (suffix_lookup_mode_ > 0) {
    suffix_drafter_.AppendSpan(tokens);
  }
}

void Qwen38GraphEngine::PrefillChunkSlot(size_t slot_idx, std::span<const int32_t> tokens) {
  if (slot_idx >= static_cast<size_t>(max_batch_)) {
    throw std::out_of_range("qwen38_graph: slot index out of range");
  }
  if (max_batch_ == 1 && slot_idx == 0) {
    PrefillChunk(tokens);
    return;
  }
  if (!fast_ || !fast_->unified) {
    throw std::runtime_error("qwen38_graph: multi-slot prefill requires unified fast prefill");
  }
  FastAppendSlot(static_cast<int>(slot_idx), tokens);
}

int32_t Qwen38GraphEngine::PendingTokenSlot(size_t slot_idx) const {
  if (slot_idx >= static_cast<size_t>(max_batch_)) {
    throw std::out_of_range("qwen38_graph: slot index out of range");
  }
  return (max_batch_ == 1 && slot_idx == 0) ? pending_ : slot_pending_[slot_idx];
}

void Qwen38GraphEngine::StepPrefill(std::span<const int32_t> tokens) {
  if (decode_started_) throw std::logic_error("qwen38_graph: PrefillChunk after decoding started");
  const size_t n = tokens.size();
  if (n == 0) throw std::logic_error("qwen38_graph: empty prefill chunk");
  const int R = fold_ ? verify_rows_ : 2;
  ledger_.RequireRoom(n, "the prompt", static_cast<size_t>(R + 1));
  MutedPrefill(true);
  const size_t base = ledger_.Length();
  const Qwen38PromptRunHints prompt_run{run_hints_.ExpectedPrefillTokens(), 0};
  {
    std::vector<int64_t> chunk(n + R, 0);  // trailing 0 = dummy rows of a partial tail
    for (size_t i = 0; i < n; ++i) chunk[i] = tokens[i];
    plane_.Check(plane_.memcpy_h2d(d_tokens_.data() + 8 * base, chunk.data(), (n + R) * 8, stream_), "upload prompt");
  }
  prompt_tokens_.insert(prompt_tokens_.end(), tokens.begin(), tokens.end());
  const uint8_t* tp = d_tokens_.data();
  const uint8_t* pp = d_arange64_.data();
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  size_t i = 0;
  const Direction* last = nullptr;
  int last_row = 0;
  while (i < n) {
    const size_t rem = n - i;
    const size_t step = std::min<size_t>(rem, static_cast<size_t>(R));
    const size_t gi = base + i;
    for (int r = 0; r < R; ++r) {
      const void* cin = (r == 0) ? ctx_commit_.data() : (ctx_a_.data() + 16 * (r - 1));
      void* cout = (step == static_cast<size_t>(R) && r == R - 1) ? ctx_commit_.data() : (ctx_a_.data() + 16 * r);
      PrepRow(R, r, tp + 8 * (gi + r), pp + 8 * (gi + r), 0, cin, cout);
    }
    RunTarget(R, ledger_.Bank(), &prompt_run);
    Direction& d = dirs_[R - 1][ledger_.Bank()];
    plane_.or_status(stream_, d.status, static_cast<int>(d.status_count), ctl_.data());
    if (step < static_cast<size_t>(R)) {
      plane_.Check(plane_.memcpy_d2d(ctx_commit_.data(), ctx_a_.data() + 16 * (step - 1), 16, stream_), "ctx commit");
      if (compact_state_updates_) {
        ReplayStateUpdates(ledger_.Bank(), step);
      }
    }
    if (keep_hidden_)
      plane_.Check(plane_.memcpy_d2d(d_prompt_hidden_.data() + gi * row_bytes, d.hidden, step * row_bytes, stream_),
                   "keep hidden");
    ledger_.Advance(step);
    i += step;
    last = &d;
    last_row = static_cast<int>(step - 1);
  }
  plane_.Check(plane_.memcpy_d2d(d_last_hidden_.data(), last->hidden + static_cast<size_t>(last_row) * row_bytes, row_bytes,
                                 stream_),
               "last hidden");
  plane_.Check(plane_.memcpy_d2h(stage_host_ + 4, last->tokens + 8 * last_row, 8, stream_), "read prediction");
  SyncStream();
  if (ReadCtl(5) != 0) CheckDeviceStatus("prefill", last);
  pending_ = static_cast<int32_t>(stage_host_[4]);  // ReadCtl reuses stage_host_[0]; slot 4 is untouched
  MutedPrefill(false);                              // (WriteCtl uses stage_host_ + 4, so pending_ is read first)
}

// ---- head -------------------------------------------------------------------------------------
void Qwen38GraphEngine::RunHeadOnce(HeadSession& head, int rows) {
  (void)rows;
  Direction& d = head.dirs[head_turn_ % 2];
  RunSession(*head.session, d.run_options.get(), *d.binding);
}

void Qwen38GraphEngine::EnqueueHead(int rows, const uint8_t* hidden_src) {
  HeadSession& head = head_[rows - 1];
  if (head_length_ + rows > capacity_) throw std::runtime_error("qwen38_graph: head cache capacity exceeded");
  plane_.head_prep(stream_, ctl_.data(), rows, head_length_, hidden_width_, hidden_src, head.hidden, head.tokens,
                   head.positions, head.committed);
  RunHeadOnce(head, rows);
  plane_.head_post(stream_, head.next_token, ctl_.data());
  plane_.or_status(stream_, head.status, static_cast<int>(head.status_count), ctl_.data());
  head_length_ += rows;
  head_turn_ += 1;
}

void Qwen38GraphEngine::HeadPrefill() {
  // Prompt context for the head: rows p = 1..P-1 are (token p, target hidden p-1); S=2 calls on pairs, S=1 for an odd tail.
  const size_t n = prompt_tokens_.size();
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  size_t p = 1;
  while (p < n) {
    const int rows = p + 1 < n ? 2 : 1;
    HeadSession& head = head_[rows - 1];
    plane_.Check(plane_.memcpy_d2d(head.tokens, d_tokens_.data() + 8 * p, 8 * rows, stream_), "head tokens");
    plane_.Check(plane_.memcpy_d2d(head.positions, d_arange64_.data() + 8 * p, 8 * rows, stream_), "head positions");
    plane_.Check(plane_.memcpy_d2d(head.hidden, d_prompt_hidden_.data() + (p - 1) * row_bytes, rows * row_bytes, stream_),
                 "head hidden");
    plane_.Check(plane_.memcpy_d2d(head.committed, d_arange32_.data() + 4 * (p - 1), 4 * rows, stream_), "head committed");
    RunHeadOnce(head, rows);
    plane_.or_status(stream_, head.status, static_cast<int>(head.status_count), ctl_.data());
    head_length_ += rows;
    head_turn_ += 1;
    p += rows;
  }
  SyncStream();
  if (ReadCtl(5) != 0) CheckDeviceStatus("head prefill", nullptr);
}

// ---- fold: standalone R=2 head, prompt context by pairs, first draft ----------------------------------------------------
void Qwen38GraphEngine::BuildHeadR2() {
  HeadR2Session& h = head_r2_;
  h.session = model_.session_head_r2_.get();
  h.info.Add(*h.session);
  ValidateQwen38CacheShapes(CacheInputsOf(h.info, "past_key_values.", false), capacity_, "head_r2");
  const auto bf16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  const auto i64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  const auto i32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  h.tokens = NewDeviceTensor(h.tensors, "input_ids", {1, 2}, i64)->ptr;
  h.positions = NewDeviceTensor(h.tensors, "position_ids", {1, 2}, i64)->ptr;
  h.hidden = NewDeviceTensor(h.tensors, "hidden_states", {1, 2, hidden_width_}, bf16)->ptr;
  h.committed = NewDeviceTensor(h.tensors, "committed_length", {1}, i32)->ptr;
  NewDeviceTensor(h.tensors, "logits", {1, 2, vocab_}, h.info.GetOutputDataType("logits"));
  NewDeviceTensor(h.tensors, "hidden_states_out", {1, 2, hidden_width_}, bf16);
  h.next_token = NewDeviceTensor(h.tensors, "next_token", {1, 2, 1}, i64)->ptr;
  h.status_count = static_cast<size_t>(h.info.GetOutputShape("validation.aggregate").at(0));
  h.status = NewDeviceTensor(h.tensors, "validation.aggregate", {static_cast<int64_t>(h.status_count)}, i32)->ptr;
  const auto in_names = h.session->GetInputNames();
  const auto out_names = h.session->GetOutputNames();
  for (int k = 0; k < num_banks_; ++k) {
    Direction& d = h.dirs[k];
    d.binding = OrtIoBinding::Create(*h.session);
    HeadBank& in_bank = head_banks_[k];
    HeadBank& mid_bank = head_banks_[(k + 1) % num_banks_];
    HeadBank& out_bank = head_banks_[(k + 2) % num_banks_];
    for (const auto& name : in_names) {
      DevTensor* t = nullptr;
      if (h.tensors.count(name))
        t = h.tensors[name].get();
      else if (head_store_.count(name))
        t = head_store_[name].get();
      else if (in_bank.tensors.count(name))
        t = in_bank.tensors[name].get();
      else
        throw std::runtime_error("qwen38_graph: unsupported head_r2 input " + name);
      d.binding->BindInput(name.c_str(), *t->value);
    }
    for (const auto& name : out_names) {
      DevTensor* t = nullptr;
      const std::string state = "past_key_values." + name.substr(name.find('.') + 1);
      if (h.tensors.count(name)) {
        t = h.tensors[name].get();
      } else if (StartsWith(name, "present_mid.")) {
        t = mid_bank.tensors.at(state).get();
      } else if (StartsWith(name, "present.")) {
        if (head_store_.count(state))
          t = head_store_[state].get();
        else
          t = out_bank.tensors.at(state).get();
      } else {
        throw std::runtime_error("qwen38_graph: unsupported head_r2 output " + name);
      }
      d.binding->BindOutput(name.c_str(), *t->value);
    }
    d.run_options = OrtRunOptions::Create();
    d.run_options->AddConfigEntry("gpu_graph_id",
                                  std::to_string(Qwen38GraphAnnotationId(Qwen38GraphRun::kHeadR2, 2, k)).c_str());
    d.run_options->AddConfigEntry("disable_synchronize_execution_providers", "1");
  }
  SyncStream();
}

// One standalone head call at the current ring phase: rows (tokens[i], hidden_i) at positions[i], head cache length
// head_length_; commits `commit_rows` (1 or 2) rows and rotates the ring.
void Qwen38GraphEngine::CallHeadR2(const int64_t tokens[2], const int64_t positions[2], const uint8_t* hidden0,
                                   const uint8_t* hidden1, int commit_rows) {
  HeadR2Session& h = head_r2_;
  if (head_length_ + 2 > capacity_) throw std::runtime_error("qwen38_graph: head cache capacity exceeded");
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  plane_.Check(plane_.memcpy_h2d(h.tokens, tokens, 16, stream_), "head tokens");
  plane_.Check(plane_.memcpy_h2d(h.positions, positions, 16, stream_), "head positions");
  const int32_t committed = head_length_;
  plane_.Check(plane_.memcpy_h2d(h.committed, &committed, 4, stream_), "head committed");
  plane_.Check(plane_.memcpy_d2d(h.hidden, hidden0, row_bytes, stream_), "head hidden");
  plane_.Check(plane_.memcpy_d2d(h.hidden + row_bytes, hidden1, row_bytes, stream_), "head hidden");
  Direction& d = h.dirs[head_cur_ % num_banks_];
  RunSession(*h.session, d.run_options.get(), *d.binding);
  plane_.or_status(stream_, h.status, static_cast<int>(h.status_count), ctl_.data());
  head_cur_ = (head_cur_ + commit_rows) % num_banks_;
  head_length_ += commit_rows;
}

// Prompt rows p = 1..P-1 are (token p, target hidden p-1) at position p; pairs, an odd tail commits one row.
void Qwen38GraphEngine::HeadPrefillFold() {
  const size_t n = prompt_tokens_.size();
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  size_t p = 1;
  while (p < n) {
    const bool two = p + 1 < n;
    const size_t r1 = two ? p + 1 : p;
    const int64_t tokens[2] = {prompt_tokens_[p], prompt_tokens_[r1]};
    const int64_t positions[2] = {static_cast<int64_t>(p), static_cast<int64_t>(r1)};
    CallHeadR2(tokens, positions, d_prompt_hidden_.data() + (p - 1) * row_bytes, d_prompt_hidden_.data() + (r1 - 1) * row_bytes,
               two ? 2 : 1);
    p += two ? 2 : 1;
  }
  SyncStream();
  if (ReadCtl(5) != 0) CheckDeviceStatus("head prefill", nullptr);
}

// Head call on (pending, last hidden) @ L with one committed row: the draft goes to ctl[2]; the head ring is rotated to the
// verify ring phase; ctl = (pending, L, draft, ..., head length). For S=3 .. S=8, the chained speculative drafts go to ctl[8] .. ctl[13].
void Qwen38GraphEngine::FoldFirstDraft() {
  const size_t length = ledger_.Length();
  const int bank = ledger_.Bank();
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  const int64_t tokens[2] = {pending_, pending_};
  const int64_t positions[2] = {static_cast<int64_t>(length), static_cast<int64_t>(length) + 1};
  CallHeadR2(tokens, positions, d_last_hidden_.data(), d_last_hidden_.data(), 1);
  plane_.Check(plane_.memcpy_d2d(ctl_.data() + 16, head_r2_.next_token, 8, stream_), "first draft");
  SyncStream();
  if (ReadCtl(5) != 0) CheckDeviceStatus("first draft", nullptr);
  if (head_cur_ != bank) {
    for (auto& entry : head_banks_[head_cur_].tensors)
      plane_.Check(plane_.memcpy_d2d(head_banks_[bank].tensors.at(entry.first)->ptr, entry.second->ptr, entry.second->Bytes(),
                                     stream_),
                   "rotate head ring");
    head_cur_ = bank;
  }
  const int committed_head_length = head_length_;
  if (verify_rows_ >= 3) {
    uint8_t* h_out = head_r2_.tensors.at("hidden_states_out")->ptr;
    plane_.Check(plane_.memcpy_d2d(d_last_hidden_.data() + row_bytes, h_out, row_bytes, stream_), "save h_mtp_1");
    const int64_t d1 = ReadCtl(2);
    const int64_t tok2[2] = {d1, d1};
    const int64_t pos2[2] = {static_cast<int64_t>(length) + 1, static_cast<int64_t>(length) + 2};
    CallHeadR2(tok2, pos2, d_last_hidden_.data() + row_bytes, d_last_hidden_.data() + row_bytes, 1);
    plane_.Check(plane_.memcpy_d2d(ctl_.data() + 64, head_r2_.next_token, 8, stream_), "second draft");
    SyncStream();
    if (ReadCtl(5) != 0) CheckDeviceStatus("second draft", nullptr);
    if (verify_rows_ >= 4) {
      plane_.Check(plane_.memcpy_d2d(d_last_hidden_.data() + 2 * row_bytes, h_out, row_bytes, stream_), "save h_mtp_2");
      const int64_t d2 = ReadCtl(8);
      const int64_t tok3[2] = {d2, d2};
      const int64_t pos3[2] = {static_cast<int64_t>(length) + 2, static_cast<int64_t>(length) + 3};
      CallHeadR2(tok3, pos3, d_last_hidden_.data() + 2 * row_bytes, d_last_hidden_.data() + 2 * row_bytes, 1);
      plane_.Check(plane_.memcpy_d2d(ctl_.data() + 72, head_r2_.next_token, 8, stream_), "third draft");
      SyncStream();
      if (ReadCtl(5) != 0) CheckDeviceStatus("third draft", nullptr);
    }
    head_cur_ = bank;
    head_length_ = committed_head_length;
  }
  const int64_t state[2] = {pending_, static_cast<int64_t>(length)};
  WriteCtl(0, std::span<const int64_t>(state, 2));
  const int64_t head_length = head_length_;
  WriteCtl(7, std::span<const int64_t>(&head_length, 1));
}

// Test hook: the verify of the next cycle sees a draft that differs from the head's, so the cycle commits exactly the greedy
// token of verify row 0. A coincidence with that token is accepted by the graph and stays valid.
void Qwen38GraphEngine::ForceRejectDraft() {
  SyncStream();
  constexpr int kDraftCtlSlots[7] = {2, 8, 9, 10, 11, 12, 13};
  for (int d = 0; d < verify_rows_ - 1 && d < 7; ++d) {
    const int slot = kDraftCtlSlots[d];
    const int64_t forced = (ReadCtl(slot) + 1) % vocab_;
    WriteCtl(slot, {&forced, 1});
  }
}

void Qwen38GraphEngine::EnqueueFold() {
  if (force_reject_) ForceRejectDraft();
  ledger_.RequireRoom(verify_rows_, "the next verify step");
  const RowInputs& in = inputs_[verify_rows_ - 1];
  plane_.Check(plane_.prep(stream_, params_dev_.data(), ctl_.data(), ctl_.data() + 8, -verify_rows_,
                           ctx_commit_.data(), ctx_a_.data(), in.input_ids, in.position_ids, in.committed, past_pos_,
                           in.ple, ctl_.data()),
               "prepare the fold rows");
  if (in.inputs_embeds) {
    plane_.EmbedRows(stream_, model_.EmbeddingTable(), in.input_ids, in.inputs_embeds, verify_rows_);
  }
  RunTarget(verify_rows_, ledger_.Bank());
  Direction& d = dirs_[verify_rows_ - 1][ledger_.Bank()];
  last_hidden_ptr_ = d.hidden;
  inflight_dir_ = &d;
  ++seq_;
  inflight_ = true;
}

void Qwen38GraphEngine::CheckDecodeStatus() {
  // ctl[5] of a slot ORs the validation status of every call of its decode: the target's (fold: the whole fused cycle) and,
  // outside the fold stack, the standalone head's, which the cycle record's status does not carry. Every slot has its own
  // word, so every slot that started a decode is read, not only slot 0. The stream is idle once ReadCtlSlot returns, so
  // this also holds after Drain() while the last cycle's record (inflight_) is unconsumed.
  std::array<bool, kMaxBatch> started{};
  for (int s = 0; s < max_batch_; ++s) {
    started[static_cast<size_t>(s)] = s == 0 ? decode_started_ : slot_decode_started_[s];
  }
  CheckQwen38DecodeStatus(std::span<const bool>{started.data(), static_cast<size_t>(max_batch_)},
                          [this](size_t slot) { return ReadCtlSlot(static_cast<int>(slot), 5); });
}

// ---- decode -----------------------------------------------------------------------------------
void Qwen38GraphEngine::EnqueueVerify() {
  ledger_.RequireRoom(2, "the next verify step");
  const uint8_t* ctl = ctl_.data();
  PrepRow(2, 0, ctl, ctl + 8, 0, ctx_commit_.data(), ctx_a_.data());
  PrepRow(2, 1, ctl + 16, ctl + 8, 1, ctx_a_.data(), ctx_b_.data());
  RunTarget(2, ledger_.Bank());
  Direction& d = dirs_[1][ledger_.Bank()];
  plane_.post_tok(stream_, d.tokens, ctl_.data(), ctx_commit_.data(), ctx_a_.data(), ctx_b_.data(), d.status,
                  static_cast<int>(d.status_count), result_host_, kRing);
  last_hidden_ptr_ = d.hidden;
  inflight_dir_ = &d;
  ++seq_;
  inflight_ = true;
}

void Qwen38GraphEngine::EnqueueStep() {
  ledger_.RequireRoom(1, "the next step");
  const uint8_t* ctl = ctl_.data();
  PrepRow(1, 0, ctl, ctl + 8, 0, ctx_commit_.data(), ctx_a_.data());
  RunTarget(1, ledger_.Bank());
  Direction& d = dirs_[0][ledger_.Bank()];
  plane_.argmax_part(stream_, d.logits, vocab_, part_v_.data(), part_i_.data(), kArgmaxBlocks);
  plane_.post1(stream_, part_v_.data(), part_i_.data(), kArgmaxBlocks, ctl_.data(), ctx_commit_.data(), ctx_a_.data(),
               d.status, static_cast<int>(d.status_count), result_host_, kRing);
  last_hidden_ptr_ = d.hidden;
  inflight_dir_ = &d;
  ++seq_;
  inflight_ = true;
}

void Qwen38GraphEngine::Draft(const uint8_t* features, size_t rows, size_t position,
                              int32_t anchor, bool propose) {
  DraftSlot(0, features, rows, position, anchor, propose);
}

void Qwen38GraphEngine::DraftSlot(int slot, const uint8_t* features, size_t rows, size_t position,
                                  int32_t anchor, bool propose) {
  if (!features || rows == 0) throw std::runtime_error("DFlash experiment requires committed feature rows");
  if (!plane_.dflash_select) SyncStream();
  const int64_t shape[2] = {int64_t(rows), int64_t(kDflashWidth)};
  Tensor auxiliary(model_.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  auxiliary.ort_tensor_ = OrtValue::CreateTensor(*memory_info_, const_cast<uint8_t*>(features),
                                                 rows * kDflashWidth * 2, std::span<const int64_t>(shape, 2),
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  Dflash2Drafter::Feed feed{};
  feed.request = const_cast<Request*>(SlotRequestKey(slot));
  feed.aux_row_count = rows;
  feed.first_position = position;
  feed.anchor_token = anchor;
  feed.draft_eligible = true;
  feed.wants_drafts = propose;
  std::vector<std::vector<int32_t>> drafts;
  if (!dflash_->Propose(auxiliary, {&feed, 1}, drafts, nullptr, nullptr, plane_.dflash_select != nullptr))
    throw std::runtime_error("DFlash experiment could not ingest the committed target features");
  if (propose) {
    if (plane_.dflash_select) {
      plane_.Check(plane_.dflash_select(stream_, dflash_->CandidateIds().GetRawData(),
                                        dflash_->Scores().GetRawData(), ctl_slot(slot),
                                        verify_rows_ - 1, model_.config_->model.dflash2.selector_top_k),
                   "select DFlash drafts on device");
      return;
    }
    if (drafts.size() != 1 || drafts[0].size() != size_t(verify_rows_ - 1))
      throw std::runtime_error("DFlash experiment returned an incomplete proposal");
    constexpr int slots[7] = {2, 8, 9, 10, 11, 12, 13};
    for (size_t i = 0; i < drafts[0].size(); ++i) {
      const int64_t d = drafts[0][i];
      WriteCtlSlot(slot, slots[i], {&d, 1});
    }
  }
}

void Qwen38GraphEngine::BeginDecode(bool run_cycle) {
  if (decode_started_) throw std::logic_error("qwen38_graph: decoding already started");
  if (ledger_.Length() == 0) throw std::logic_error("qwen38_graph: BeginDecode before the prompt was ingested");
  decode_started_ = true;
  run_hints_.ExpectPrefillTokens(0);
  if (dflash_) {
    if (!run_cycle) return;
    FastHandoff();
    const int64_t state[2] = {pending_, static_cast<int64_t>(ledger_.Length())};
    WriteCtl(0, std::span<const int64_t>(state, 2));
    const auto last_chunk = slot_dflash_prompt_[0].TakeSavedChunk();
    if (!last_chunk) throw std::runtime_error("DFlash experiment lacks prompt features");
    Draft(d_dflash_prompt_.data(), last_chunk->rows, last_chunk->position, pending_, true);
    EnqueueFold();
    return;
  }
  if (fold_) {
    if (!run_cycle) return;
    ClearHead();
    SyncStream();
    if (fast_) {
      FastHandoff();
      if (prefill_head_) FastHeadKvPrefill();
    } else if (prefill_head_) {
      HeadPrefillFold();
    }
    FoldFirstDraft();
    EnqueueFold();
    return;
  }
  if (mtp_enabled_ && run_cycle) {
    ClearHead();
    SyncStream();
    if (prefill_head_) HeadPrefill();
  }
  const int64_t state[2] = {pending_, static_cast<int64_t>(ledger_.Length())};
  WriteCtl(0, std::span<const int64_t>(state, 2));
  if (!run_cycle) return;
  if (mtp_enabled_) {
    EnqueueHead(1, d_last_hidden_.data());
    EnqueueVerify();
  } else {
    EnqueueStep();
  }
}

Qwen38CycleRecord Qwen38GraphEngine::WaitCycle() {
  if (!inflight_) throw std::logic_error("qwen38_graph: no cycle in flight");
  slot_seq_[0] = seq_;
  Qwen38CycleRecord result = WaitCycleSlot(0);
  inflight_ = false;
  return result;
}

Qwen38CycleRecord Qwen38GraphEngine::WaitCycleSlot(int slot_idx) {
  const uint64_t expected_seq = slot_seq_[slot_idx];
  const size_t ring_offset = ((expected_seq - 1) % kRing) * 8;
  volatile const int64_t* record = result_ring_slot(slot_idx) + ring_offset;
  const auto start = Clock::now();
  uint32_t spins = 0;
  while (record[0] != static_cast<int64_t>(expected_seq)) {
#if defined(_MSC_VER) && defined(_M_X64)
    _mm_pause();
#elif defined(__x86_64__) || defined(_M_X64)
    __builtin_ia32_pause();
#endif
    if (((++spins) & (kStatusRingSpinCheck - 1)) == 0) {
      if (SecondsSince(start) > 180.0) throw std::runtime_error("qwen38_graph: timeout waiting for the cycle record");
      std::this_thread::yield();
    }
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  Qwen38CycleRecord result;
  const uint64_t raw1 = static_cast<uint64_t>(record[1]);
  const uint64_t raw2 = static_cast<uint64_t>(record[2]);
  const uint64_t raw3 = static_cast<uint64_t>(record[3]);
  const uint64_t raw5 = static_cast<uint64_t>(record[5]);
  const uint64_t raw7 = static_cast<uint64_t>(record[7]);
  result.pred0 = static_cast<int64_t>(static_cast<int32_t>(raw1 & 0xFFFFFFFFu));
  result.pred1 = static_cast<int64_t>(static_cast<int32_t>(raw2 & 0xFFFFFFFFu));
  const int64_t acc_count = mtp_enabled_ ? static_cast<int64_t>(static_cast<int32_t>(raw3 & 0xFFFFFFFFu)) : 0;
  result.accepted = acc_count > 0;
  result.accepted_count = static_cast<int>(acc_count);
  result.proposed_count = mtp_enabled_ ? (verify_rows_ - 1) : 1;
  const int64_t status = record[4];
  result.new_pending = static_cast<int64_t>(static_cast<int32_t>(raw5 & 0xFFFFFFFFu));
  result.new_length = record[6];
  if (verify_rows_ >= 3) {
    result.pred2 = static_cast<int64_t>(static_cast<int32_t>(raw7 & 0xFFFFFFFFu));
    result.pred3 = static_cast<int64_t>(static_cast<int32_t>(raw7 >> 32));
  }
  if (verify_rows_ >= 5) {
    result.pred4 = static_cast<int64_t>(static_cast<int32_t>(raw1 >> 32));
    result.pred5 = static_cast<int64_t>(static_cast<int32_t>(raw2 >> 32));
    result.pred6 = static_cast<int64_t>(static_cast<int32_t>(raw3 >> 32));
    result.pred7 = static_cast<int64_t>(static_cast<int32_t>(raw5 >> 32));
  }
  if (status != 0) CheckDeviceStatus("decode", inflight_dir_);
  return result;
}

void Qwen38GraphEngine::BeginDecodeSlot(size_t slot_idx, bool want_more) {
  if (slot_idx >= static_cast<size_t>(max_batch_)) {
    throw std::out_of_range("qwen38_graph: slot index out of range");
  }
  if (max_batch_ == 1 && slot_idx == 0) {
    BeginDecode(want_more);
    return;
  }
  const int s = static_cast<int>(slot_idx);
  if (slot_decode_started_[s]) throw std::logic_error("qwen38_graph: slot decoding already started");
  if (slot_ledgers_[s].Length() == 0) throw std::logic_error("qwen38_graph: BeginDecodeSlot before prompt was ingested");
  slot_decode_started_[s] = true;
  if (s == 0) decode_started_ = true;
  if (!want_more) {
    MuteSlot(s);
    return;
  }
  FastHandoff(slot_idx);
  const int64_t state[4] = {slot_pending_[s], static_cast<int64_t>(slot_ledgers_[s].Length()), 0, 0};
  WriteCtlSlot(s, 0, std::span<const int64_t>(state, 4));
  if (dflash_) {
    const auto last_chunk = slot_dflash_prompt_[s].TakeSavedChunk();
    if (!last_chunk) throw std::runtime_error("DFlash experiment lacks slot prompt features");
    DraftSlot(s, d_dflash_prompt_slot(s), last_chunk->rows, last_chunk->position, slot_pending_[s], true);
  }
}

void Qwen38GraphEngine::DraftDeferredSlots(std::span<const size_t> active_slots) {
  if (!dflash_) return;
  for (const size_t slot_idx : active_slots) {
    const int s = static_cast<int>(slot_idx);
    if (slot_deferred_aux_rows_[s] > 0) {
      DraftSlot(s, d_dflash_step_aux_slot(s), slot_deferred_aux_rows_[s], slot_deferred_aux_pos_[s], slot_pending_[s], true);
      slot_deferred_aux_rows_[s] = 0;
    }
  }
}

void Qwen38GraphEngine::EnqueueBatchDecode(std::span<const size_t> active_slots) {
  if (active_slots.empty()) return;
  if (max_batch_ == 1 && active_slots.size() == 1 && active_slots[0] == 0) {
    EnqueueFold();
    return;
  }
  DraftDeferredSlots(active_slots);
  size_t max_slot = 0;
  for (const size_t s : active_slots) {
    max_slot = std::max(max_slot, s);
    slot_ledgers_[s].RequireRoom(verify_rows_, "the next verify step");
  }
  const int B = static_cast<int>(max_slot + 1);
  for (int s = 0; s < B; ++s) {
    const bool is_active = std::find(active_slots.begin(), active_slots.end(), static_cast<size_t>(s)) != active_slots.end();
    if (!is_active) {
      MuteSlot(s);
    }
  }
  PrepBatch(B);
  Direction& d = (B == 1) ? dirs_[verify_rows_ - 1][bank_phase_ % num_banks_]
                          : batch_dirs_[B - 1][bank_phase_ % num_banks_];
  RunSession(*model_.session_target_, d.run_options.get(), *d.binding, nullptr, Qwen38SlotMaskOf(active_slots));
  last_hidden_ptr_ = d.hidden;
  inflight_dir_ = &d;
  inflight_batch_ = B;
  for (const size_t s : active_slots) {
    ++slot_seq_[s];
    if (s == 0) seq_ = slot_seq_[0];
  }
  inflight_ = true;
}

void Qwen38GraphEngine::WaitBatchDecode(std::span<const size_t> active_slots,
                                        std::span<Qwen38CycleRecord> records) {
  if (!inflight_) throw std::logic_error("qwen38_graph: no batch cycle in flight");
  if (records.size() < active_slots.size()) {
    throw std::invalid_argument("qwen38_graph: insufficient record buffer for WaitBatchDecode");
  }
  for (size_t i = 0; i < active_slots.size(); ++i) {
    records[i] = WaitCycleSlot(static_cast<int>(active_slots[i]));
  }
  inflight_ = false;
}

void Qwen38GraphEngine::AdvanceBatchDecode(std::span<const size_t> active_slots,
                                           std::span<const Qwen38CycleRecord> records,
                                           std::span<const uint8_t> want_more,
                                           bool defer_next_cycle) {
  if (max_batch_ == 1 && !defer_next_cycle && active_slots.size() == 1 && active_slots[0] == 0) {
    Advance(records[0], want_more[0] != 0);
    return;
  }
  const int prev_bank = bank_phase_;
  std::vector<size_t> next_active_slots;
  std::vector<size_t> next_active_indices;
  next_active_slots.reserve(active_slots.size());
  next_active_indices.reserve(active_slots.size());

  for (size_t i = 0; i < active_slots.size(); ++i) {
    const int s = static_cast<int>(active_slots[i]);
    const size_t step = records[i].TokenCount();
    if (compact_state_updates_ && step < static_cast<size_t>(verify_rows_)) {
      ReplayStateUpdatesSlot(prev_bank, s, step);
    }
    slot_ledgers_[s].Advance(step);
    slot_ledgers_[s].CheckDeviceLength(records[i].new_length);
    slot_pending_[s] = static_cast<int32_t>(records[i].new_pending);
    if (s == 0) {
      ledger_ = slot_ledgers_[0];
      pending_ = slot_pending_[0];
    }
    if (want_more[i] != 0) {
      next_active_slots.push_back(active_slots[i]);
      next_active_indices.push_back(i);
    } else {
      MuteSlot(s);
    }
  }
  bank_phase_ = (bank_phase_ + 1) % num_banks_;
  if (next_active_slots.empty()) {
    return;
  }
  if (defer_next_cycle) {
    if (dflash_ && inflight_dir_ && inflight_dir_->auxiliary) {
      const size_t row_bytes = kDflashWidth * 2;
      for (const size_t idx : next_active_indices) {
        const int s = static_cast<int>(active_slots[idx]);
        const size_t step = records[idx].TokenCount();
        const uint8_t* src = inflight_dir_->auxiliary + static_cast<size_t>(s * verify_rows_) * row_bytes;
        plane_.Check(plane_.memcpy_d2d(d_dflash_step_aux_slot(s), src, step * row_bytes, stream_),
                     "save deferred slot aux");
        slot_deferred_aux_rows_[s] = step;
        slot_deferred_aux_pos_[s] = slot_ledgers_[s].Length() - step;
      }
      SyncStream();
    }
    return;
  }
  if (dflash_) {
    if (!plane_.dflash_select) SyncStream();
    const size_t total_aux_rows = static_cast<size_t>(inflight_batch_ * verify_rows_);
    const int64_t aux_shape[2] = {static_cast<int64_t>(total_aux_rows), static_cast<int64_t>(kDflashWidth)};
    Tensor auxiliary(model_.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
    auxiliary.ort_tensor_ = OrtValue::CreateTensor(
        *memory_info_, inflight_dir_->auxiliary, total_aux_rows * kDflashWidth * 2,
        std::span<const int64_t>(aux_shape, 2), ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
    std::vector<Dflash2Drafter::Feed> feeds;
    feeds.reserve(next_active_indices.size());
    for (const size_t idx : next_active_indices) {
      const int s = static_cast<int>(active_slots[idx]);
      const size_t step = records[idx].TokenCount();
      Dflash2Drafter::Feed feed{};
      feed.request = const_cast<Request*>(SlotRequestKey(s));
      feed.aux_row_begin = static_cast<size_t>(s * verify_rows_);
      feed.aux_row_count = step;
      feed.first_position = slot_ledgers_[s].Length() - step;
      feed.anchor_token = slot_pending_[s];
      feed.draft_eligible = true;
      feed.wants_drafts = true;
      feeds.push_back(feed);
    }
    std::vector<std::vector<int32_t>> drafts;
    if (!dflash_->Propose(auxiliary, feeds, drafts, nullptr, nullptr, plane_.dflash_select != nullptr)) {
      throw std::runtime_error("DFlash experiment could not ingest batched target features");
    }
    const int num_spec = verify_rows_ - 1;
    const int top_k = model_.config_->model.dflash2.selector_top_k;
    for (size_t f_idx = 0; f_idx < next_active_slots.size(); ++f_idx) {
      const int s = static_cast<int>(next_active_slots[f_idx]);
      if (plane_.dflash_select) {
        const int32_t* cand_ptr =
            static_cast<const int32_t*>(dflash_->CandidateIds().GetRawData()) + f_idx * num_spec * top_k;
        const float* score_ptr =
            static_cast<const float*>(dflash_->Scores().GetRawData()) + f_idx * num_spec * top_k * top_k;
        plane_.Check(plane_.dflash_select(stream_, cand_ptr, score_ptr, ctl_slot(s), num_spec, top_k),
                     "select batched DFlash drafts on device");
      } else {
        constexpr int slots[7] = {2, 8, 9, 10, 11, 12, 13};
        for (size_t j = 0; j < drafts[f_idx].size(); ++j) {
          const int64_t d_val = drafts[f_idx][j];
          WriteCtlSlot(s, slots[j], {&d_val, 1});
        }
      }
    }
  }
  EnqueueBatchDecode(next_active_slots);
}

void Qwen38GraphEngine::Advance(const Qwen38CycleRecord& record, bool want_more) {
  const int prev_bank = ledger_.Bank();
  const size_t step = record.TokenCount();
  if (compact_state_updates_ && step < static_cast<size_t>(verify_rows_)) {
    ReplayStateUpdates(prev_bank, step);
  }
  ledger_.Advance(step);
  ledger_.CheckDeviceLength(record.new_length);
  if (fold_) head_length_ += static_cast<int>(step);
  if (suffix_lookup_mode_ > 0) {
    const int64_t preds[7] = {record.pred0, record.pred1, record.pred2, record.pred3,
                              record.pred4, record.pred5, record.pred6};
    suffix_drafter_.AppendToken(static_cast<int32_t>(pending_));
    for (int i = 0; i < record.accepted_count && i < 7; ++i) {
      suffix_drafter_.AppendToken(static_cast<int32_t>(preds[i]));
    }
    pending_ = static_cast<int32_t>(record.new_pending);
  }
  if (!want_more) return;
  if (dflash_) {
    pending_ = static_cast<int32_t>(record.new_pending);
    Draft(inflight_dir_->auxiliary, step, ledger_.Length() - step, pending_, true);
    EnqueueFold();
    return;
  }
  if (fold_) {
    if (suffix_lookup_mode_ == 1 || (suffix_lookup_mode_ == 2 && !record.accepted)) {
      int32_t lookup_drafts[7] = {-1, -1, -1, -1, -1, -1, -1};
      constexpr int kDraftCtlSlots[7] = {2, 8, 9, 10, 11, 12, 13};
      const int max_d = verify_rows_ - 1;
      const int n_d = suffix_drafter_.ProposeDrafts(static_cast<int32_t>(record.new_pending), max_d, lookup_drafts, 3);
      for (int i = 0; i < n_d && i < verify_rows_ - 1 && i < 7; ++i) {
        const int64_t di = lookup_drafts[i];
        WriteCtl(kDraftCtlSlots[i], {&di, 1});
      }
    }
    head_cur_ = ledger_.Bank();
    EnqueueFold();
    return;
  }
  if (mtp_enabled_) {
    EnqueueHead(record.accepted ? 2 : 1, last_hidden_ptr_);
    EnqueueVerify();
  } else {
    EnqueueStep();
  }
}

void Qwen38GraphEngine::Warmup() {
  // ORT captures a graph id after min_num_runs eager runs: touch every (rows, phase) and head direction enough times.
  // The Runs of the warmup carry ORT_QMOE_IN_WARMUP and, from here on, every Run of the engine asks for deterministic
  // MoE tactics: the engine keeps that state and publishes it around each Run only (qwen38_ort_env.h), also when this
  // throws.
  struct WarmupFlag {
    explicit WarmupFlag(Qwen38OrtRunHints& hints) : hints_{hints} { hints_.SetWarmup(true); }
    ~WarmupFlag() { hints_.SetWarmup(false); }
    WarmupFlag(const WarmupFlag&) = delete;
    WarmupFlag& operator=(const WarmupFlag&) = delete;
    Qwen38OrtRunHints& hints_;
  } warmup_flag{run_hints_};
  const int R = fold_ ? verify_rows_ : 2;
  const int max_prompt = capacity_ - Qwen38DecodeDevice::kReservedTail - (R + 1);
  if (max_prompt < R) return;
  const bool target_captures_graph = IsGraphCaptureEnabled(model_.config_->model.decoder.session_options);
  if (fast_ && fast_->unified) {
    const int warm_len = std::min<int>({fast_->chunk, 512, max_prompt});
    if (warm_len >= 2) {
      std::vector<int32_t> prompt(warm_len, 1000);
      FastAppend(prompt);
      ClearHead();
      SyncStream();
      FastHandoff();
      if (prefill_head_) FastHeadKvPrefill();
      SyncStream();
      Reset();
    }
  }
  const int runs_per_phase = Qwen38WarmupRuns(target_captures_graph);
  int dummy_tok = 1000;
  for (int run = 0; run < runs_per_phase; ++run) {
    for (int phase = 0; phase < num_banks_; ++phase) {
      if (ledger_.Length() + static_cast<size_t>(R) > static_cast<size_t>(max_prompt)) {
        Reset();
      }
      ledger_.SetBank(phase);
      std::vector<int32_t> dummy(R);
      for (int i = 0; i < R; ++i) dummy[i] = dummy_tok++;
      StepPrefill(dummy);  // S=R target ids; fold: the muted cycle graphs
    }
  }
  if (fold_) {
    const auto& q = model_.config_->model.qwen38_graph;
    const bool head_captures_graph =
        Qwen38HeadR2CapturesGraph(model_.config_->model.decoder.session_options, q.fold.head_r2.session_options);
    const int runs_per_head_phase = Qwen38WarmupRuns(head_captures_graph);
    ClearHead();
    SyncStream();
    int head_dummy_tok = 1000;
    for (int run = 0; run < runs_per_head_phase; ++run) {
      for (int phase = 0; phase < num_banks_; ++phase) {
        // A reset for the capacity restarts the ring at phase 0; pick the phase explicitly so that each one is run
        // exactly runs_per_head_phase times whatever kv_capacity is.
        if (head_length_ + 2 > capacity_) {
          ClearHead();
          SyncStream();
        }
        head_cur_ = phase;
        const int64_t tokens[2] = {head_dummy_tok, head_dummy_tok}, positions[2] = {head_length_ + 1, head_length_ + 2};
        ++head_dummy_tok;
        CallHeadR2(tokens, positions, d_last_hidden_.data(), d_last_hidden_.data(), 1);
      }
    }
    SyncStream();
    if (ReadCtl(5) != 0) CheckDeviceStatus("warmup (head)", nullptr);
    ClearHead();
    SyncStream();
    if (fast_ && !fast_->unified) {
      // dense / head-KV shapes: arena growth and first-use costs of the prefill sessions (one dummy prompt per length)
      std::mt19937 rng(12345);
      for (int length : model_.config_->model.qwen38_graph.prefill.warmup_lengths) {
        if (length < 2 || length > max_prompt) continue;
        Reset();
        std::vector<int32_t> prompt(length);
        for (auto& t : prompt) t = static_cast<int32_t>(1000 + rng() % 199000);
        FastAppend(prompt);
        ClearHead();
        SyncStream();
        FastHandoff();
        if (prefill_head_) FastHeadKvPrefill();
        SyncStream();
      }
    }
    Reset();
    return;
  }
  const size_t max_usable = static_cast<size_t>(capacity_ - Qwen38DecodeDevice::kReservedTail);
  for (int i = 0; i < 24; ++i) {  // S=1 target ids
    const int target_phase = i % num_banks_;
    if (ledger_.Length() == 0 || ledger_.Length() + 1 > max_usable) {
      Reset();
      const std::vector<int32_t> pair = {1000 + i, 1001 + i};
      StepPrefill(pair);
    }
    ledger_.SetBank(target_phase);
    const int64_t state[2] = {pending_, static_cast<int64_t>(ledger_.Length())};
    WriteCtl(0, std::span<const int64_t>(state, 2));
    EnqueueStep();
    WaitCycle();
    ledger_.Advance(1);
  }
  if (mtp_enabled_) {
    ClearHead();
    SyncStream();
    for (int i = 0; i < 24; ++i) {
      if (head_length_ + 1 > capacity_) {
        ClearHead();
        SyncStream();
      }
      EnqueueHead(1, d_last_hidden_.data());
    }
    for (int i = 0; i < 24; ++i) {
      if (head_length_ + 2 > capacity_) {
        ClearHead();
        SyncStream();
      }
      EnqueueHead(2, d_last_hidden_.data());
    }
    SyncStream();
    if (ReadCtl(5) != 0) CheckDeviceStatus("warmup", nullptr);
  }
  Reset();
}

// ------------------------------------------------------------------------------------------------
// Model
// ------------------------------------------------------------------------------------------------
Config::SessionOptions ResolveQwen38AuxiliarySessionOptions(const Config::SessionOptions& parent,
                                                            const std::optional<Config::SessionOptions>& configured,
                                                            bool inherit) {
  Config::SessionOptions options = configured.value_or(Config::SessionOptions{});
  for (const auto& po : options.provider_options) {
    if (std::find(options.providers.begin(), options.providers.end(), po.name) == options.providers.end()) {
      options.providers.push_back(po.name);
    }
  }
  if (inherit) {
    InheritSessionOptionsFromParent(parent, options);
    return options;
  }
  if (options.providers.empty()) {
    options.providers = parent.providers;
  }
  for (const auto& parent_provider : parent.provider_options) {
    if (std::find(options.providers.begin(), options.providers.end(), parent_provider.name) == options.providers.end()) {
      options.providers.push_back(parent_provider.name);
    }
    auto child_provider = std::find_if(options.provider_options.begin(), options.provider_options.end(),
                                       [&parent_provider](const Config::ProviderOptions& provider) {
                                         return provider.name == parent_provider.name;
                                       });
    if (child_provider == options.provider_options.end()) {
      options.provider_options.push_back(
          Config::ProviderOptions{parent_provider.name, {}, parent_provider.device_filtering_options});
      child_provider = std::prev(options.provider_options.end());
    } else if (!child_provider->device_filtering_options) {
      child_provider->device_filtering_options = parent_provider.device_filtering_options;
    }
    for (const auto& parent_option : parent_provider.options) {
      if (parent_option.first == "enable_cuda_graph" || parent_option.first == "enable_graph_capture" ||
          parent_option.first == "enableGraphCapture") {
        continue;
      }
      const bool overridden = std::any_of(child_provider->options.begin(), child_provider->options.end(),
                                          [&parent_option](const Config::NamedString& option) {
                                            return option.first == parent_option.first;
                                          });
      if (!overridden) {
        child_provider->options.push_back(parent_option);
      }
    }
  }
  return options;
}

bool Qwen38HeadR2CapturesGraph(const Config::SessionOptions& decoder,
                               const std::optional<Config::SessionOptions>& head_r2) {
  return IsGraphCaptureEnabled(ResolveQwen38AuxiliarySessionOptions(decoder, head_r2, /*inherit=*/true));
}

Qwen38EmbeddingFormat RequireQwen38PackageSupportedByPlane(const Qwen38Plane& plane, const Config& config,
                                                           const std::filesystem::path& model_folder) {
  // EnqueueFold prepares the rows of a verify step with one negative-pos_add call; a library without the batch would prepare
  // the first row and leave the others stale, so a package with the fold stack needs the capability.
  if (config.model.qwen38_graph.fold.enabled) {
    plane.RequireCapabilities(QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH,
                              "the folded MTP verify path (model.qwen38_graph.fold)");
  }
  // The batched prep entry point (qwen38_plane_prep_batched) prepares the rows of every slot of a step with explicit row and
  // batch arguments; only the packages that run several slots need it.
  if (ResolveQwen38MaxBatchSize(config) > 1) {
    plane.RequireCapabilities(QWEN38_PLANE_CAP_BATCHED_PREP,
                              "several request slots (model.qwen38_graph.max_batch_size > 1)");
  }
  const std::filesystem::path ple_table_path = model_folder / config.model.qwen38_graph.ple_table;
  RequireQwen38PleTableSupported(plane, ParseQwen38PleDescription(ple_table_path), ple_table_path);
  if (config.model.embedding.filename.empty()) return Qwen38EmbeddingFormat::kNone;
  const std::filesystem::path embed_data_path = model_folder / (config.model.embedding.filename + ".data");
  std::error_code size_error;
  const auto embed_file_bytes = std::filesystem::file_size(embed_data_path, size_error);
  if (size_error) return Qwen38EmbeddingFormat::kNone;
  return SelectQwen38EmbeddingFormat(plane, embed_data_path, static_cast<size_t>(embed_file_bytes),
                                     config.model.vocab_size, config.model.decoder.hidden_size);
}

namespace {

// Validates the qwen38_flash_graph configuration and checks the package's PLE and embedding formats against the plane
// library before Model::Model runs EnsureDeviceOrtInit or AddSharedInitializers.
std::unique_ptr<Config> PreflightQwen38GraphModelConfig(std::unique_ptr<Config> config) {
  if (!config) return config;
  const auto& q = config->model.qwen38_graph;
  if (!q.IsConfigured() || q.plane_library.empty() || q.ple_table.empty())
    throw std::runtime_error(
        "qwen38_flash_graph requires model.qwen38_graph.{head.filename_s1|fold.head_r2.filename,plane_library,ple_table}");
  if (q.fold.enabled) {
    if (!q.mtp.enabled) throw std::runtime_error("model.qwen38_graph.fold requires model.qwen38_graph.mtp.enabled");
    if (q.fold.head_r2.filename.empty() && config->model.dflash2.filename.empty())
      throw std::runtime_error("model.qwen38_graph.fold.head_r2.filename is required");
  } else if (q.mtp.enabled && q.head.filename_s2.empty()) {
    throw std::runtime_error("model.qwen38_graph.head.filename_s2 is required when MTP is enabled");
  }
  if (q.prefill.enabled && !q.fold.enabled)
    throw std::runtime_error("model.qwen38_graph.prefill requires fold.enabled");
  const bool separate_prefill_sessions = Qwen38HasSeparatePrefillSessions(*config);
  if (separate_prefill_sessions && (q.prefill.head_kv.filename.empty() || q.prefill.constants.empty()))
    throw std::runtime_error(
        "model.qwen38_graph.prefill with a separate dense graph requires prefill.{head_kv.filename,constants}");
  ValidateQwen38PrefillChunk(q);
  if (!config->model.dflash2.filename.empty() && config->model.dflash2.is_dspark)
    throw std::runtime_error("qwen38_flash_graph does not support model.dspark");
  if (q.kv_capacity <= Qwen38DecodeDevice::kReservedTail)
    throw std::runtime_error("model.qwen38_graph.kv_capacity must exceed " +
                             std::to_string(Qwen38DecodeDevice::kReservedTail));

  const std::filesystem::path root =
      std::filesystem::weakly_canonical(std::filesystem::path{config->config_path.string()});
  const std::filesystem::path plane_path = std::filesystem::weakly_canonical(root / q.plane_library);
  const std::string root_prefix = (root / "").string();
  if (plane_path.string().compare(0, root_prefix.size(), root_prefix) != 0)
    throw std::runtime_error("model.qwen38_graph.plane_library must resolve inside the model folder: " +
                             plane_path.string());
  std::error_code ec;
  if (std::filesystem::exists(ResolveWindowsSharedLibraryPath(plane_path), ec) ||
      std::filesystem::exists(plane_path, ec)) {
    Qwen38Plane plane{plane_path};
    (void)RequireQwen38PackageSupportedByPlane(plane, *config, root);
  }
  return config;
}

}  // namespace

Qwen38GraphModel::Qwen38GraphModel(std::unique_ptr<Config> config, OrtEnv& ort_env)
    : Model{PreflightQwen38GraphModelConfig(std::move(config))} {
  const auto& q = config_->model.qwen38_graph;
  const bool separate_prefill_sessions =
      q.prefill.enabled && !q.prefill.dense.filename.empty() &&
      q.prefill.dense.filename != config_->model.decoder.filename;
  if (!p_device_ || p_device_->GetType() != DeviceType::CUDA)
    throw std::runtime_error("qwen38_flash_graph requires the CUDA execution provider");
  // Several request slots need the fold stack and the unified prefill: refuse the rest here, not at the first decode.
  ValidateQwen38MultiSlotConfig(*config_);

  // Token selection is done by the device loop; the Engine's Search only does bookkeeping (EOS, max length), on the CPU so
  // that committing a token never synchronises the CUDA stream the look-ahead cycle runs on.
  p_device_scoring_ = GetDeviceInterface(DeviceType::CPU);

  const auto start = Clock::now();
  const std::filesystem::path root = std::filesystem::weakly_canonical(std::filesystem::path{config_->config_path.string()});
  const std::filesystem::path plane_path = std::filesystem::weakly_canonical(root / q.plane_library);
  plane_ = std::make_unique<Qwen38Plane>(plane_path);
  plane_->Check(plane_->set_device(p_device_->GetDeviceId(nullptr)), "cudaSetDevice");

  // An INT8 package needs a plane library that can decode it: refuse it now, before the weights are uploaded and the sessions
  // are created, instead of letting the library read INT8 values as FP8 shards or as a bfloat16 table.
  const Qwen38EmbeddingFormat embed_format = RequireQwen38PackageSupportedByPlane(*plane_, *config_, root);

  // The target graph decides how the slots are batched: choose it now (the MTP-drafts test hook may pick another one), and for
  // several slots read its output names from the model file, so that a target the batched device cannot serve is refused
  // before any weight is uploaded. The check against the session below stays authoritative.
  std::string target_filename = config_->model.decoder.filename;
  if (const char* drafts_env = std::getenv("ORT_QWEN38_MTP_DRAFTS")) {
    const int d = std::atoi(drafts_env);
    if (d >= 1 && d <= 3) {
      const std::filesystem::path parent = std::filesystem::path(target_filename).parent_path();
      const std::string alt_name = "model_s" + std::to_string(d + 1) + ".onnx";
      const std::string alt = (parent.empty() ? std::filesystem::path(alt_name) : (parent / alt_name)).string();
      if (std::filesystem::exists(root / alt)) {
        target_filename = alt;
      } else if (std::filesystem::exists(root / ("target/" + alt_name))) {
        target_filename = "target/" + alt_name;
      }
    }
  }
  if (!config_->memory_assets_only_ && config_->model_data_spans_.count(target_filename) == 0) {
    ValidateQwen38TargetModelFile(ResolveQwen38MaxBatchSize(*config_), root / target_filename);
  }

  // Device copy of the initializers several sessions have in common (optional)
  if (!q.prefill.weights_pool.empty()) {
    pool_ = std::make_unique<Qwen38WeightPool>(*plane_, root / q.prefill.weights_pool, root, *p_device_->GetMemoryInfo());
    if (g_log.enabled) Log("qwen38_graph") << "weight pool " << SecondsSince(start) << " s" << std::endl;
    pool_->AddTo(*session_options_, "target");
  }

  // Standard CPU embedding.onnx session (Gather on CPU EP) + zero-copy UVA mapping of embedding.onnx.data for decode graphs.
  if (!config_->model.embedding.filename.empty()) {
    embedding_session_options_ = OrtSessionOptions::Create();
    Config::SessionOptions cpu_options = config_->model.embedding.session_options.value_or(Config::SessionOptions{});
    if (!cpu_options.intra_op_num_threads) cpu_options.intra_op_num_threads = 1;
    CreateSessionOptionsFromConfig(cpu_options, *embedding_session_options_, false, true, false, false);
    session_embedding_ = CreateSession(ort_env, config_->model.embedding.filename, embedding_session_options_.get());
    embedding_run_options_ = OrtRunOptions::Create();
    if (config_->model.embedding.run_options) {
      for (const auto& [key, value] : *config_->model.embedding.run_options) {
        embedding_run_options_->AddConfigEntry(key.c_str(), value.c_str());
      }
    }
    // Only the BF16 or the INT8 vocab_size x hidden_size table is mapped: nothing may gather rows from any other file.
    if (embed_format != Qwen38EmbeddingFormat::kNone) {
      const std::filesystem::path embed_data_path = root / (config_->model.embedding.filename + ".data");
      embed_mapping_ = std::make_unique<Qwen38FileMapping>(*plane_, embed_data_path, true);
      const std::vector<int64_t> embed_shape{config_->model.vocab_size, config_->model.decoder.hidden_size};
      const Qwen38EmbeddingFormat mapped_format = SelectQwen38EmbeddingFormat(
          *plane_, embed_data_path, embed_mapping_->size(), embed_shape[0], embed_shape[1]);
      if (mapped_format != embed_format) {
        const size_t mapped_bytes = embed_mapping_->size();
        embed_mapping_.reset();
        throw std::runtime_error("qwen38_graph: the mapped embedding table " + embed_data_path.string() +
                                 " changed size (" + std::to_string(mapped_bytes) +
                                 " bytes) and no longer matches the validated embedding format");
      }
      if (embed_format == Qwen38EmbeddingFormat::kBf16) {
        embed_is_int8_ = false;
        embed_values_.push_back(OrtValue::CreateTensor(
            *p_device_->GetMemoryInfo(), embed_mapping_->device_data(),
            Qwen38EmbeddingTableBytes(embed_format, embed_shape[0], embed_shape[1]), embed_shape,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16));
        for (const char* name : {"model.embed_tokens.weight", "head.mtp.embed_tokens.weight"}) {
          session_options_->AddInitializer(name, *embed_values_.back());
        }
      } else {
        embed_is_int8_ = true;
        const size_t table_i8_bytes = static_cast<size_t>(embed_shape[0]) * static_cast<size_t>(embed_shape[1]);
        embed_values_.push_back(OrtValue::CreateTensor(
            *p_device_->GetMemoryInfo(), embed_mapping_->device_data(), table_i8_bytes, embed_shape,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8));
        for (const char* name : {"model.embed_tokens.weight", "head.mtp.embed_tokens.weight"}) {
          session_options_->AddInitializer(name, *embed_values_.back());
        }
        const std::vector<int64_t> scale_shape{config_->model.vocab_size, 1};
        const size_t scale_bytes = static_cast<size_t>(scale_shape[0]) * sizeof(float);
        embed_values_.push_back(OrtValue::CreateTensor(
            *p_device_->GetMemoryInfo(), embed_mapping_->device_data() + table_i8_bytes, scale_bytes, scale_shape,
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT));
        session_options_->AddInitializer("model.embed_tokens.weight.row_scales", *embed_values_.back());
      }
    }
  }

  session_target_ = CreateSession(ort_env, target_filename, session_options_.get());
  session_info_.Add(*session_target_);
  // The outputs of the session are authoritative (the model file may not be a protobuf one): refuse a ring-bank target for
  // several slots before anything else is built.
  ValidateQwen38BatchedTargetOutputs(ResolveQwen38MaxBatchSize(*config_), session_target_->GetOutputNames());
  if (g_log.enabled) Log("qwen38_graph") << "target session " << SecondsSince(start) << " s" << std::endl;

  auto make_options = [&](const std::optional<Config::SessionOptions>& configured, bool inherit, const char* label) {
    Config::SessionOptions options =
        ResolveQwen38AuxiliarySessionOptions(config_->model.decoder.session_options, configured, inherit);
    auto result = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(options, *result, /*is_primary_session_options=*/false,
                                   /*disable_graph_capture=*/!inherit, /*cloned_from_parent=*/false,
                                   /*append_providers=*/true);
    if (pool_) pool_->AddTo(*result, label);
    if (!embed_values_.empty() &&
        (std::string(label) == "mtp" || std::string(label) == "drafter" || std::string(label) == "head_r2" ||
         std::string(label) == "head")) {
      result->AddInitializer("mtp.embed_tokens.weight", *embed_values_.front());
    }
    return result;
  };

  if (q.fold.enabled) {
    if (!q.fold.head_r2.filename.empty() && config_->model.dflash2.filename.empty()) {
      head_r2_session_options_ = make_options(q.fold.head_r2.session_options, true, "mtp");
      session_head_r2_ = CreateSession(ort_env, q.fold.head_r2.filename, head_r2_session_options_.get());
    }
    if (separate_prefill_sessions) {
      // the dense / head-KV sessions are not graph captured and do not inherit the decoder's provider options
      dense_session_options_ = make_options(q.prefill.dense.session_options, false, "dense");
      head_kv_session_options_ = make_options(q.prefill.head_kv.session_options, false, "head_kv");
      session_dense_ = CreateSession(ort_env, q.prefill.dense.filename, dense_session_options_.get());
      session_head_kv_ = CreateSession(ort_env, q.prefill.head_kv.filename, head_kv_session_options_.get());
      qsa_constants_ = std::make_unique<Qwen38QsaConstants>(*plane_, root / q.prefill.constants, root, p_device_->GetCudaStream());
    }
    if (g_log.enabled) Log("qwen38_graph") << "head / prefill sessions " << SecondsSince(start) << " s" << std::endl;
  } else if (q.mtp.enabled) {
    head_session_options_ = make_options(q.head.session_options, true, "head");
    session_head_s1_ = CreateSession(ort_env, q.head.filename_s1, head_session_options_.get());
    session_head_s2_ = CreateSession(ort_env, q.head.filename_s2, head_session_options_.get());
    if (g_log.enabled) Log("qwen38_graph") << "head sessions " << SecondsSince(start) << " s" << std::endl;
  }

  engine_ = std::make_unique<Qwen38GraphEngine>(*this);
  auto& capabilities = resolved_capabilities_;
  capabilities.token_selection = BackendTokenSelection::Backend;
  capabilities.admission = engine_->MaxBatchSize() > 1 ? BackendAdmission::IndependentSlots : BackendAdmission::WholeBatch;
  capabilities.kv_storage = BackendKvStorage::PersistentContiguous;
  capabilities.additional_state = BackendAdditionalState::PrivateBanks;
  capabilities.continuation = BackendContinuation::ReplayHistory;
  capabilities.rewind = BackendRewind::ReplayHistory;
  capabilities.supports_sampling = false;
  capabilities.supports_penalties = false;
  capabilities.supports_min_generated_tokens = false;
  capabilities.supports_guidance = false;
  capabilities.supports_stop_strings = true;
  capabilities.max_slots = engine_->MaxBatchSize();
  capabilities.configured_max_batch_size = capabilities.max_slots;
  const uint64_t kv_limit = static_cast<uint64_t>(engine_->MaxSequenceLength());
  capabilities.max_request_length =
      config_->model.context_length > 0
          ? std::min<uint64_t>(kv_limit, static_cast<uint64_t>(config_->model.context_length))
          : kv_limit;
  capabilities.max_result_tokens = kMaxGeneratedTokensPerStep;
  if (config_->engine.dynamic_batching)
    capabilities.max_scheduled_tokens = config_->engine.dynamic_batching->max_scheduled_tokens;
}

Qwen38GraphModel::~Qwen38GraphModel() = default;

Qwen38EmbeddingTable Qwen38GraphModel::EmbeddingTable() const {
  Qwen38EmbeddingTable table;
  table.data = embed_mapping_ ? embed_mapping_->device_data() : nullptr;
  table.rows = config_->model.vocab_size;
  table.width = config_->model.decoder.hidden_size;
  table.int8 = embed_is_int8_;
  return table;
}

void Qwen38GraphModel::RunCpuEmbedding(const int64_t* ids_host, size_t num_tokens, void* dst_host) const {
  if (!session_embedding_) throw std::runtime_error("qwen38_graph: model.embedding session is not initialized");
  const auto& emb_cfg = config_->model.embedding;
  const auto& cpu_mem = GetDeviceInterface(DeviceType::CPU)->GetAllocator().GetInfo();
  const int64_t id_shape[1] = {static_cast<int64_t>(num_tokens)};
  const int64_t out_shape[2] = {static_cast<int64_t>(num_tokens),
                                static_cast<int64_t>(config_->model.decoder.hidden_size)};
  const size_t out_bytes = num_tokens * static_cast<size_t>(config_->model.decoder.hidden_size) * sizeof(uint16_t);
  auto in_val = OrtValue::CreateTensor(cpu_mem, const_cast<int64_t*>(ids_host), num_tokens * sizeof(int64_t),
                                       std::span<const int64_t>(id_shape, 1), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
  auto out_val = OrtValue::CreateTensor(cpu_mem, dst_host, out_bytes, std::span<const int64_t>(out_shape, 2),
                                        ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  const char* in_name = emb_cfg.inputs.input_ids.c_str();
  const char* out_name = emb_cfg.outputs.embeddings.c_str();
  OrtValue* in_ptr = in_val.get();
  OrtValue* out_ptr = out_val.get();
  session_embedding_->Run(embedding_run_options_.get(), &in_name, &in_ptr, 1, &out_name, &out_ptr, 1);
}

std::unique_ptr<State> Qwen38GraphModel::CreateState(DeviceSpan<int32_t>, const GeneratorParams&) const {
  throw std::runtime_error(
      "The qwen38_flash_graph model type is served by the Engine API (og.Engine / OgaEngine) only; it has no Generator API support.");
}

std::shared_ptr<Qwen38DecodeDevice> Qwen38GraphModel::AcquireDevice() {
  if (!IsExecutionContextHealthy())
    throw std::runtime_error("qwen38_flash_graph: the execution context is poisoned; reload the Model before creating another Engine.");
  if (leased_.exchange(true))
    throw std::runtime_error(
        "qwen38_flash_graph: the model's CUDA-graph state is already used by another Engine; destroy that Engine first "
        "(one Engine per model).");
  try {
    if (!IsExecutionContextHealthy()) {
      throw std::runtime_error("qwen38_flash_graph: the execution context is poisoned; reload the Model before creating another Engine.");
    }
    // CUDA's current device is per thread. The Engine drives the model from the thread that acquires it, which is not the
    // thread that loaded the model, so select the model's GPU here: its stream and buffers live on that device.
    plane_->Check(plane_->set_device(p_device_->GetDeviceId(nullptr)), "cudaSetDevice");
  } catch (...) {
    leased_.store(false);
    throw;
  }
  Qwen38GraphEngine* engine = engine_.get();
  // The lease keeps the model alive and drains the device before releasing it, so a look-ahead cycle never outlives its Engine.
  std::shared_ptr<Model> keep_alive = shared_from_this();
  return std::shared_ptr<Qwen38DecodeDevice>(engine, [this, keep_alive](Qwen38DecodeDevice* device) {
    device->Drain();
    try {
      device->CheckDecodeStatus();
    } catch (...) {
      PoisonExecutionContext();
    }
    leased_.store(false);
  });
}

}  // namespace Generators
