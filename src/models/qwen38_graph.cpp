// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Qwen3.8-Flash-Next NVFP4 CUDA-graph decode + MTP(1): model and device engine. See qwen38_graph.h.
#include "engine/step_plan.h"
#include "generator/generators.h"
#include "logging.h"
#include "qwen38_graph.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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

// The cache inputs of `info` that depend on the KV capacity: every input under `prefix` (and past_position_ids).
std::vector<Qwen38CacheInput> CacheInputsOf(const SessionInfo& info, const char* prefix, bool with_positions) {
  std::vector<Qwen38CacheInput> inputs;
  for (const auto& name : info.GetInputNames()) {
    if (StartsWith(name, prefix) || (with_positions && name == "past_position_ids"))
      inputs.push_back({name, info.GetInputShape(name)});
  }
  return inputs;
}

}  // namespace

void ValidateQwen38CacheShapes(const std::vector<Qwen38CacheInput>& inputs, int kv_capacity, const std::string& graph) {
  const auto fail = [&](const Qwen38CacheInput& input, const char* requirement) {
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

size_t DevTensor::Bytes() const {
  return Elements(shape) * Ort::SizeOf(type);
}

int Qwen38GraphAnnotationId(Qwen38GraphRun run, int rows, int phase) {
  const int run_index = static_cast<int>(run);
  if (run_index < 0 || run_index >= kQwen38GraphRunCount || rows < 1 || rows > kQwen38GraphIdMaxRows || phase < 0 ||
      phase >= kQwen38GraphIdMaxPhases) {
    throw std::out_of_range("qwen38_graph: no CUDA graph id for run=" + std::to_string(run_index) +
                            ", rows=" + std::to_string(rows) + ", phase=" + std::to_string(phase));
  }
  return 1 + (run_index * kQwen38GraphIdMaxRows + (rows - 1)) * kQwen38GraphIdMaxPhases + phase;
}

Qwen38DflashWarmupPlan PlanQwen38DflashWarmup(bool warmup, int verify_rows, int num_banks) noexcept {
  if (!warmup || verify_rows < 1 || num_banks < 1) return {};
  return {true, static_cast<size_t>(verify_rows) * static_cast<size_t>(num_banks) *
                    static_cast<size_t>(kQwen38WarmupRunsPerGraph)};
}

// ------------------------------------------------------------------------------------------------
// Engine
// ------------------------------------------------------------------------------------------------
Qwen38GraphEngine::Qwen38GraphEngine(Qwen38GraphModel& model)
    : model_{model},
      plane_{*model.plane_},
      capacity_{model.config_->model.qwen38_graph.kv_capacity},
      vocab_{model.config_->model.vocab_size},
      ledger_{model.config_->model.qwen38_graph.kv_capacity} {
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
  ValidateQwen38CacheShapes(CacheInputsOf(model_.session_info_, "past_key_values.", true), capacity_, "target");
  if (fold_) {
    ValidateQwen38CacheShapes(CacheInputsOf(model_.session_info_, "head.past_key_values.", false), capacity_,
                              "fold head");
  }
  stream_ = model_.p_device_->GetCudaStream();
  plane_.Check(plane_.set_device(model_.p_device_->GetDeviceId(nullptr)), "cudaSetDevice");
  memory_info_ = model_.p_device_->GetMemoryInfo();

  result_ring_ = PlaneHostBuffer(plane_, kRing * 8 * sizeof(int64_t));
  stage_ = PlaneHostBuffer(plane_, 16 * sizeof(int64_t));
  result_host_ = result_ring_.data();
  stage_host_ = stage_.data();

  // control block (the fold model binds it as input, so it exists before the target bindings)
  // Allocated with 16 int64 words so slots [8] and [9] hold draft_1 and draft_2 for S=3 / S=4 multi-draft verify.
  ctl_ = PlaneBuffer(plane_, 16 * sizeof(int64_t), stream_);
  ctx_commit_ = PlaneBuffer(plane_, 2 * sizeof(int64_t), stream_);
  // ctx_a_ holds 4 contiguous 2-int64 row contexts (rows 0..3); ctx_b_ is row 1's buffer (also aliased at ctx_a_ + 16).
  ctx_a_ = PlaneBuffer(plane_, kMaxVerifyRows * 2 * sizeof(int64_t), stream_);
  ctx_b_ = PlaneBuffer(plane_, 2 * sizeof(int64_t), stream_);
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
  d_tokens_ = PlaneBuffer(plane_, slots * sizeof(int64_t), stream_);
  d_arange64_ = PlaneBuffer(plane_, slots * sizeof(int64_t), stream_);
  d_arange32_ = PlaneBuffer(plane_, slots * sizeof(int32_t), stream_);
  d_last_hidden_ = PlaneBuffer(plane_, static_cast<size_t>(hidden_width_) * 2 * kMaxVerifyRows, stream_);
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
    if (!fold_ || !fast_ || !fast_->unified)
      throw std::runtime_error("DFlash experiment requires unified folded prefill");
    if (model_.config_->model.dflash2.num_draft_tokens != verify_rows_ - 1)
      throw std::runtime_error("DFlash draft count must match target prefix-state width");
    auto draft_config = CreateDflash2Config(*model_.config_);
    draft_config->model.embedding.filename.clear();
    dflash_model_ = std::make_shared<Dflash2Model>(
        std::move(draft_config), GetOrtEnv(), nullptr,
        [this](OrtSessionOptions& opts) {
          const char* skip_san = std::getenv("ORT_PAGED_ATTENTION_SKIP_SANITIZER");
          opts.AddConfigEntry("session.paged_attention.skip_input_sanitizer",
                              (skip_san != nullptr && std::string(skip_san) == "0") ? "0" : "1");
          if (model_.pool_) model_.pool_->AddTo(opts, "dflash2");
          if (!model_.embed_values_.empty()) {
            opts.AddInitializer("model.embed_tokens.weight", *model_.embed_values_.front());
            if (model_.embed_is_int8_ && model_.embed_values_.size() >= 2) {
              opts.AddInitializer("model.embed_tokens.weight.row_scales", *model_.embed_values_[1]);
            }
          }
        });
    const auto cache_shape = dflash_model_->session_info_.GetInputShape("past_key_values.0.key");
    if (cache_shape.size() != 4 || (cache_shape[1] != 16 && cache_shape[1] != 128 && cache_shape[1] != 256))
      throw std::runtime_error("DFlash experiment requires cache pages of 16, 128, or 256 tokens");
    const size_t page_size = static_cast<size_t>(cache_shape[1]);
    dflash_ = std::make_unique<Dflash2Drafter>(
        dflash_model_, page_size, Dflash2Drafter::PoolBlocks(*model_.config_, page_size, 1), 1);
    d_dflash_prompt_ = PlaneBuffer(plane_, size_t(fast_->chunk) * kDflashWidth * 2, stream_);
  }
  Reset();
  if (dflash_) {
    const auto start = Clock::now();
    run_hints_.RequireDeterministicMoe();
    const Qwen38DflashWarmupPlan warmup = PlanQwen38DflashWarmup(q.warmup, verify_rows_, num_banks_);
    if (warmup.drafter && fast_ && fast_->unified) {
      if (fast_->chunk > 512) {
        Draft(d_dflash_prompt_.data(), static_cast<size_t>(fast_->chunk), static_cast<size_t>(capacity_), 1000, false);
        dflash_->ReleaseAll();
      }
      const int warm_len = std::min<int>(fast_->chunk, 512);
      std::vector<int32_t> prompt(warm_len, 1000);
      FastAppend(prompt);
      FastHandoff();
      size_t pos = 0;
      if (dflash_prompt_rows_ > 0) {
        for (int rep = 0; rep < 5; ++rep) {
          dflash_->ReleaseAll();
          Draft(d_dflash_prompt_.data(), dflash_prompt_rows_, dflash_prompt_position_, 1000, true);
        }
        pos = dflash_prompt_position_ + dflash_prompt_rows_;
        dflash_prompt_rows_ = 0;
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
    if (warmup.target_tokens > 0) {
      std::vector<int32_t> dummy(warmup.target_tokens, 1000);
      StepPrefill(dummy);
      Reset();
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
  status_count_ = static_cast<size_t>(info.GetOutputShape("validation.aggregate").at(0));
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

  // Store (KV, positions) and shared (index_key): one in-place buffer each, bound as input and output.
  for (const auto& name : in_names) {
    if (kinds[name] == Kind::kStore)
      NewDeviceTensor(store_, name, FixedShape(info.GetInputShape(name)), info.GetInputDataType(name));
    else if (kinds[name] == Kind::kShared)
      NewDeviceTensor(shared_, name, FixedShape(info.GetInputShape(name)), info.GetInputDataType(name));
  }

  // Detect multi-draft verification width from target graph outputs (state_update.* -> S = capacity+1,
  // present_mid3 -> S=4, present_mid2 -> S=3, else S=2).
  verify_rows_ = 2;
  compact_state_updates_ = false;
  state_update_capacity_ = 0;
  for (const auto& name : out_names) {
    if (StartsWith(name, "state_update.")) {
      compact_state_updates_ = true;
      NewDeviceTensor(state_update_tensors_, name, FixedShape(info.GetOutputShape(name)), info.GetOutputDataType(name));
      if (name.find(".conv_value") != std::string::npos || name.find(".indexer_value") != std::string::npos) {
        const auto su_shape = FixedShape(info.GetOutputShape(name));
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
  if (compact_state_updates_) {
    if (state_update_capacity_ <= 0 || state_update_capacity_ + 1 > kMaxVerifyRows)
      throw std::runtime_error("qwen38_graph: invalid compact state_update_capacity " +
                               std::to_string(state_update_capacity_));
    verify_rows_ = state_update_capacity_ + 1;
    num_banks_ = 2;
    ledger_.SetCompactPingPong(true, verify_rows_);
  } else {
    num_banks_ = fold_ ? (verify_rows_ + 1) : kBanks;
    ledger_.SetNumBanks(num_banks_);
  }

  // Recurrent banks: flat 256-byte-aligned arenas, identical layout in every bank (the extra scratch bank is only used by S=1).
  size_t arena_bytes = 0;
  std::map<std::string, size_t> offsets;
  for (const auto& name : bank_names) {
    arena_bytes = AlignUp(arena_bytes, 256);
    offsets[name] = arena_bytes;
    arena_bytes += Elements(FixedShape(info.GetInputShape(name))) * Ort::SizeOf(info.GetInputDataType(name));
  }
  arena_bytes = AlignUp(arena_bytes, 256);
  const int alloc_banks = fold_ ? num_banks_ : (num_banks_ + 1);
  for (int b = 0; b < alloc_banks; ++b) {
    Bank& bank = banks_[b];
    bank.arena = PlaneBuffer(plane_, arena_bytes, stream_);
    for (const auto& name : bank_names)
      bank.tensors[name] = MakeTensor(bank.arena.data() + offsets[name], FixedShape(info.GetInputShape(name)),
                                      info.GetInputDataType(name));
  }

  if (compact_state_updates_) {
    if (!plane_.replay_state_updates)
      throw std::runtime_error("qwen38_graph: plane library lacks qwen38_plane_replay_state_updates");
    for (int k = 0; k < 2; ++k) {
      const Bank& src_bank = banks_[k];
      const Bank& dst_bank = banks_[k ^ 1];
      std::vector<Qwen38StateReplayDesc> fast_descs;
      std::vector<Qwen38StateReplayDesc> generic_descs;
      for (const auto& [su_name, su_tensor] : state_update_tensors_) {
        const size_t dot1 = su_name.find('.');
        const size_t dot2 = su_name.find('.', dot1 + 1);
        const std::string lid_str = su_name.substr(dot1 + 1, dot2 - dot1 - 1);
        const std::string kind_str = su_name.substr(dot2 + 1);
        if (kind_str == "recurrent_capsule") {
          const std::string state_name = "past_key_values." + lid_str + ".recurrent_state";
          const DevTensor& src_t = *src_bank.tensors.at(state_name);
          const DevTensor& dst_t = *dst_bank.tensors.at(state_name);
          const uint64_t heads_v = static_cast<uint64_t>(src_t.shape.at(1));
          const uint64_t width_v = static_cast<uint64_t>(src_t.shape.at(2));
          const uint64_t width_k = static_cast<uint64_t>(src_t.shape.at(3));
          const uint64_t cap = static_cast<uint64_t>(state_update_capacity_);
          const uint64_t total_width = static_cast<uint64_t>(su_tensor->shape.at(1));
          const uint64_t per_step = total_width / cap;
          const uint64_t heads_k = (per_step - heads_v - heads_v * width_v) / width_k;
          const float* capsule = reinterpret_cast<const float*>(su_tensor->ptr);
          const float* decay = capsule;
          const float* key = decay + cap * heads_v;
          const float* delta = key + cap * heads_k * width_k;
          Qwen38StateReplayDesc d{};
          d.source_state = src_t.ptr;
          d.destination_state = dst_t.ptr;
          d.decay = decay;
          d.key = key;
          d.delta = delta;
          d.channel_count = heads_v;
          d.state_width = width_v;
          d.key_width = width_k;
          d.key_head_count = heads_k;
          d.capacity = static_cast<uint32_t>(state_update_capacity_);
          d.kept_count = 1;
          d.element_size = 4;
          d.kind = 2;
          fast_descs.push_back(d);
          const int blocks = static_cast<int>(heads_v * ((width_v + 31) / 32));
          replay_fast_blocks_ = std::max(replay_fast_blocks_, blocks);
        } else if (kind_str == "conv_value" || kind_str == "ple_conv_value") {
          const std::string state_tail = (kind_str == "ple_conv_value") ? "ple_conv_state" : "conv_state";
          const std::string state_name = "past_key_values." + lid_str + "." + state_tail;
          const DevTensor& src_t = *src_bank.tensors.at(state_name);
          const DevTensor& dst_t = *dst_bank.tensors.at(state_name);
          Qwen38StateReplayDesc d{};
          d.source_state = src_t.ptr;
          d.destination_state = dst_t.ptr;
          d.value = su_tensor->ptr;
          d.channel_count = static_cast<uint64_t>(src_t.shape.at(1));
          d.state_width = static_cast<uint64_t>(src_t.shape.at(2));
          d.capacity = static_cast<uint32_t>(state_update_capacity_);
          d.kept_count = 1;
          d.element_size = static_cast<uint32_t>(Ort::SizeOf(src_t.type));
          d.kind = 1;
          generic_descs.push_back(d);
        } else if (kind_str == "indexer_value") {
          const std::string buf_name = "past_key_values." + lid_str + ".index_buffer";
          const std::string len_name = "past_key_values." + lid_str + ".index_lengths";
          const DevTensor& src_buf = *src_bank.tensors.at(buf_name);
          const DevTensor& dst_buf = *dst_bank.tensors.at(buf_name);
          const DevTensor& src_len = *src_bank.tensors.at(len_name);
          const DevTensor& dst_len = *dst_bank.tensors.at(len_name);
          Qwen38StateReplayDesc d{};
          d.value = su_tensor->ptr;
          d.source_aux_state = src_buf.ptr;
          d.destination_aux_state = dst_buf.ptr;
          d.source_lengths = reinterpret_cast<const int32_t*>(src_len.ptr);
          d.destination_lengths = reinterpret_cast<int32_t*>(dst_len.ptr);
          d.state_width = static_cast<uint64_t>(src_buf.shape.at(2));
          d.state_capacity = 0;
          d.aux_capacity = static_cast<uint64_t>(src_buf.shape.at(1));
          d.capacity = static_cast<uint32_t>(state_update_capacity_);
          d.kept_count = 1;
          d.element_size = static_cast<uint32_t>(Ort::SizeOf(src_buf.type));
          d.compress_ratio = 4;
          d.kind = 4;
          generic_descs.push_back(d);
        } else {
          throw std::runtime_error("qwen38_graph: unsupported state_update output " + su_name);
        }
      }
      replay_fast_count_ = static_cast<int>(fast_descs.size());
      replay_generic_count_ = static_cast<int>(generic_descs.size());
      std::vector<Qwen38StateReplayDesc> all_descs = std::move(fast_descs);
      all_descs.insert(all_descs.end(), generic_descs.begin(), generic_descs.end());
      const size_t desc_bytes = all_descs.size() * sizeof(Qwen38StateReplayDesc);
      replay_descs_dev_[k] = PlaneBuffer(plane_, desc_bytes, stream_);
      plane_.Check(plane_.memcpy_h2d(replay_descs_dev_[k].data(), all_descs.data(), desc_bytes, stream_),
                   "upload state replay descs");
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
    fold_tensors_["fold.ctl"] = MakeTensor(ctl_.data(), info.GetInputShape("fold.ctl"), i64);
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
      for (const auto& name : in_names) {
        DevTensor* t = nullptr;
        switch (kinds[name]) {
          case Kind::kStore:
            t = store_[name].get();
            break;
          case Kind::kShared:
            t = shared_[name].get();
            break;
          case Kind::kBank:
            t = in_bank.tensors[name].get();
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
        auto state_for = [&](const std::string& tail, Bank& bank) -> DevTensor* {
          const std::string input_name = "past_key_values." + tail;
          auto it = kinds.find(input_name);
          if (it == kinds.end()) throw std::runtime_error("qwen38_graph: no input for output " + name);
          if (it->second == Kind::kStore) return store_[input_name].get();
          if (it->second == Kind::kShared) return shared_[input_name].get();
          return bank.tensors[input_name].get();
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
          t = store_["past_position_ids"].get();
        else if (StartsWith(name, "present_mid."))
          t = state_for(name.substr(12), mid1_bank);
        else if (StartsWith(name, "present_mid2."))
          t = state_for(name.substr(13), mid2_bank);
        else if (StartsWith(name, "present_mid3."))
          t = state_for(name.substr(13), mid3_bank);
        else if (StartsWith(name, "state_update."))
          t = state_update_tensors_.at(name).get();
        else if (StartsWith(name, "present."))
          t = state_for(name.substr(8), out_bank);
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
  dflash_prompt_rows_ = 0;
  dflash_ingested_in_turn_ = false;
  for (auto& bank : banks_)
    if (bank.arena.data()) plane_.Check(plane_.device_memset(bank.arena.data(), 0, bank.arena.size(), stream_), "clear bank");
  for (auto& entry : store_) plane_.Check(plane_.device_memset(entry.second->ptr, 0, entry.second->Bytes(), stream_), "clear store");
  for (auto& entry : shared_) plane_.Check(plane_.device_memset(entry.second->ptr, 0, entry.second->Bytes(), stream_), "clear index");
  plane_.Check(plane_.device_memset(ctl_.data(), 0, ctl_.size(), stream_), "clear ctl");
  std::memset(result_host_, 0, kRing * 8 * sizeof(int64_t));
  stage_host_[0] = stage_host_[1] = ple_->params.eos;
  plane_.Check(plane_.memcpy_h2d(ctx_commit_.data(), stage_host_, 16, stream_), "ctx init");
  if (mtp_enabled_) ClearHead();
  SyncStream();
  FastReset();
  run_hints_.BeginTurn();
  seq_ = 0;
  ledger_.Reset();
  inflight_ = false;
  decode_started_ = false;
  pending_ = -1;
  prompt_tokens_.clear();
  suffix_drafter_.Reset();
  last_hidden_ptr_ = nullptr;
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

void Qwen38GraphEngine::RunTarget(int rows, int phase, const Qwen38PromptRunHints* prompt) {
  Direction& d = dirs_[rows - 1][phase % num_banks_];
  RunSession(*model_.session_target_, d.run_options.get(), *d.binding, prompt);
}

void Qwen38GraphEngine::RunSession(OrtSession& session, const OrtRunOptions* run_options, const OrtIoBinding& binding,
                                   const Qwen38PromptRunHints* prompt) {
  RunWithQwen38OrtHints(run_hints_, prompt, [&] { session.Run(run_options, binding); });
}

void Qwen38GraphEngine::ReplayStateUpdates(int src_bank, size_t kept_count) {
  if (!compact_state_updates_ || kept_count == 0 || kept_count >= static_cast<size_t>(verify_rows_)) return;
  const int b = src_bank % 2;
  plane_.Check(plane_.replay_state_updates(stream_, replay_descs_dev_[b].data(), replay_fast_count_,
                                           replay_generic_count_, replay_fast_blocks_, static_cast<int>(kept_count)),
               "replay compact state updates");
}

int64_t Qwen38GraphEngine::ReadCtl(int index) {
  plane_.Check(plane_.memcpy_d2h(stage_host_, ctl_.data() + 8 * index, 8, stream_), "read ctl");
  SyncStream();
  return stage_host_[0];
}

void Qwen38GraphEngine::WriteCtl(int first, std::span<const int64_t> values) {
  std::memcpy(stage_host_ + 4, values.data(), values.size() * 8);
  plane_.Check(plane_.memcpy_h2d(ctl_.data() + 8 * first, stage_host_ + 4, values.size() * 8, stream_), "write ctl");
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
  if (!fold_) return;
  const int64_t value = muted ? 1 : 0;
  WriteCtl(3, std::span<const int64_t>(&value, 1));
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
// verify ring phase; ctl = (pending, L, draft, ..., head length). For S=3 / S=4, chained speculative drafts d2/d3 go to ctl[8]/ctl[9].
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
  if (!decode_started_) return;
  // ctl[5] ORs the validation status of every call of the decode: the target's (fold: the whole fused cycle) and,
  // outside the fold stack, the standalone head's, which the cycle record's status does not carry. The stream is idle
  // once ReadCtl returns, so this also holds after Drain() while the last cycle's record (inflight_) is unconsumed.
  const int64_t bits = ReadCtl(5);
  if (bits != 0) throw std::runtime_error("qwen38_graph: device error bits " + std::to_string(bits) + " after decode");
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
  if (!features || rows == 0) throw std::runtime_error("DFlash experiment requires committed feature rows");
  if (!plane_.dflash_select) SyncStream();
  const int64_t shape[2] = {int64_t(rows), int64_t(kDflashWidth)};
  Tensor auxiliary(model_.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  auxiliary.ort_tensor_ = OrtValue::CreateTensor(*memory_info_, const_cast<uint8_t*>(features),
                                                 rows * kDflashWidth * 2, std::span<const int64_t>(shape, 2),
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  Dflash2Drafter::Feed feed{};
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
                                        dflash_->Scores().GetRawData(), ctl_.data(),
                                        verify_rows_ - 1, model_.config_->model.dflash2.selector_top_k),
                   "select DFlash drafts on device");
      return;
    }
    if (drafts.size() != 1 || drafts[0].size() != size_t(verify_rows_ - 1))
      throw std::runtime_error("DFlash experiment returned an incomplete proposal");
    constexpr int slots[7] = {2, 8, 9, 10, 11, 12, 13};
    for (size_t i = 0; i < drafts[0].size(); ++i) {
      const int64_t d = drafts[0][i];
      WriteCtl(slots[i], {&d, 1});
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
    if (!dflash_prompt_rows_) throw std::runtime_error("DFlash experiment lacks prompt features");
    Draft(d_dflash_prompt_.data(), dflash_prompt_rows_, dflash_prompt_position_, pending_, true);
    dflash_prompt_rows_ = 0;
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
  const size_t slot = ((seq_ - 1) % kRing) * 8;
  volatile const int64_t* record = result_host_ + slot;
  const auto start = Clock::now();
  uint32_t spins = 0;
  while (record[0] != static_cast<int64_t>(seq_)) {
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
  inflight_ = false;
  if (status != 0) CheckDeviceStatus("decode", inflight_dir_);
  return result;
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
  const int runs_per_phase = target_captures_graph ? kQwen38WarmupRunsPerGraph : 1;
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
    const int runs_per_head_phase = head_captures_graph ? kQwen38WarmupRunsPerGraph : 1;
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
    if (q.fold.head_r2.filename.empty() && config_->model.dflash2.filename.empty())
      throw std::runtime_error("model.qwen38_graph.fold.head_r2.filename is required");
  } else if (q.mtp.enabled && q.head.filename_s2.empty()) {
    throw std::runtime_error("model.qwen38_graph.head.filename_s2 is required when MTP is enabled");
  }
  if (q.prefill.enabled && !q.fold.enabled)
    throw std::runtime_error("model.qwen38_graph.prefill requires fold.enabled");
  const bool separate_prefill_sessions =
      q.prefill.enabled && !q.prefill.dense.filename.empty() &&
      q.prefill.dense.filename != config->model.decoder.filename;
  if (separate_prefill_sessions && (q.prefill.head_kv.filename.empty() || q.prefill.constants.empty()))
    throw std::runtime_error(
        "model.qwen38_graph.prefill with a separate dense graph requires prefill.{head_kv.filename,constants}");
  ValidateQwen38PrefillChunk(q);
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
  session_target_ = CreateSession(ort_env, target_filename, session_options_.get());
  session_info_.Add(*session_target_);
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
  capabilities.admission = BackendAdmission::WholeBatch;
  capabilities.kv_storage = BackendKvStorage::PersistentContiguous;
  capabilities.additional_state = BackendAdditionalState::PrivateBanks;
  capabilities.continuation = BackendContinuation::ReplayHistory;
  capabilities.rewind = BackendRewind::ReplayHistory;
  capabilities.supports_sampling = false;
  capabilities.supports_penalties = false;
  capabilities.supports_min_generated_tokens = false;
  capabilities.supports_guidance = false;
  capabilities.supports_stop_strings = true;
  capabilities.max_slots = 1;
  capabilities.configured_max_batch_size = 1;
  const uint64_t kv_limit = static_cast<uint64_t>(engine_->Capacity() - engine_->ReservedTail());
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
