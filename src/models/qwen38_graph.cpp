// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Qwen3.8-Flash-Next NVFP4 CUDA-graph decode + MTP(1): model, engine and state. See qwen38_graph.h.
#if defined(__linux__)
#include "generator/generators.h"
#include "qwen38_graph.h"
#include <limits>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

namespace Generators {

namespace {

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

bool Verbose() {
  static const bool verbose = [] {
    const char* value = std::getenv("QWEN38_GRAPH_VERBOSE");
    return value != nullptr && value[0] == '1';
  }();
  return verbose;
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

float Bf16ToFloat(uint16_t bits) {
  const uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

constexpr int kStatusRingSpinCheck = 1 << 16;

}  // namespace

// ------------------------------------------------------------------------------------------------
// Plane library
// ------------------------------------------------------------------------------------------------
Qwen38Plane::Qwen38Plane(const Qwen38PlaneFunctions& functions) : Qwen38PlaneFunctions{functions} {
  if (!sizeof_params || !set_device || !error_string || !malloc_ || !free_ || !host_alloc || !host_free ||
      !host_register || !host_unregister || !memset_ || !memcpy_h2d || !memcpy_d2h || !memcpy_d2d ||
      !stream_sync || !prep || !argmax_part || !post_tok || !post1 || !head_prep || !head_post || !or_status ||
      !memcpy2d_d2d || !host_get_device_pointer || !embed_rows || !dflash_select || !argmax_finish ||
      !stream_create || !stream_destroy || !ple_rows || !pool_keys || !transpose_f32)
    throw std::runtime_error("qwen38_graph: CUDA add-on returned an incomplete data-plane interface");
  if (sizeof_params() != static_cast<int>(sizeof(Qwen38PleParams)))
    throw std::runtime_error("qwen38_graph: CUDA add-on PleParams layout differs from the host definition");
}

void Qwen38Plane::Check(int error, const char* what) const {
  if (error != 0)
    throw std::runtime_error(std::string("qwen38_graph: CUDA error in ") + what + ": " + error_string(error));
}

PlaneBuffer::PlaneBuffer(Qwen38Plane& plane, size_t bytes, void* stream) : plane_{&plane}, bytes_{bytes} {
  void* p = nullptr;
  plane.Check(plane.malloc_(&p, std::max<size_t>(bytes, 16)), "cudaMalloc");
  std::unique_ptr<void, int (*)(void*)> allocation{p, plane.free_};
  ptr_ = static_cast<uint8_t*>(p);
  plane.Check(plane.memset_(ptr_, 0, bytes_, stream), "cudaMemsetAsync");
  allocation.release();
}

PlaneBuffer::~PlaneBuffer() {
  if (ptr_ && plane_) plane_->free_(ptr_);
}

PlaneBuffer::PlaneBuffer(PlaneBuffer&& other) noexcept : plane_{other.plane_}, ptr_{other.ptr_}, bytes_{other.bytes_} {
  other.ptr_ = nullptr;
  other.bytes_ = 0;
}

PlaneBuffer& PlaneBuffer::operator=(PlaneBuffer&& other) noexcept {
  if (this != &other) {
    if (ptr_ && plane_) plane_->free_(ptr_);
    plane_ = other.plane_;
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

size_t DevTensor::Bytes() const {
  return Elements(shape) * Ort::SizeOf(type);
}

// ------------------------------------------------------------------------------------------------
// PLE table
// ------------------------------------------------------------------------------------------------
Qwen38PleTable::Qwen38PleTable(Qwen38Plane& plane, const std::string& table_path, const std::string& root, int capacity) {
  std::ifstream in(table_path);
  if (!in) throw std::runtime_error("qwen38_graph: cannot open PLE table " + table_path);
  std::string line;
  std::getline(in, line);
  if (line != "qwen38_ple_table 1") throw std::runtime_error("qwen38_graph: unsupported PLE table header: " + line);
  struct Shard {
    std::string path;
    unsigned long long offset;
    long long rows;
  };
  std::vector<Shard> shards;
  while (std::getline(in, line)) {
    std::istringstream fields(line);
    std::string key;
    fields >> key;
    if (key == "n_heads") {
      if (!(fields >> params.n_heads) || params.n_heads <= 0 || params.n_heads > 32)
        throw std::runtime_error("qwen38_graph: invalid PLE head count in " + table_path);
    } else if (key == "head_dim")
      fields >> params.head_dim;
    else if (key == "heads_per_ngram")
      fields >> params.heads_per_ngram;
    else if (key == "eos")
      fields >> params.eos;
    else if (key == "scale")
      fields >> params.scale;
    else if (key == "mult") {
      for (long long& m : params.mult) fields >> m;
    } else if (key == "vocab") {
      if (params.n_heads <= 0 || params.n_heads > 32)
        throw std::runtime_error("qwen38_graph: PLE head count must precede vocab in " + table_path);
      for (int h = 0; h < params.n_heads; ++h) fields >> params.vocab[h];
    } else if (key == "offset") {
      if (params.n_heads <= 0 || params.n_heads > 32)
        throw std::runtime_error("qwen38_graph: PLE head count must precede offset in " + table_path);
      for (int h = 0; h < params.n_heads; ++h) fields >> params.offset[h];
    } else if (key == "shard") {
      Shard shard{};
      if (!(fields >> shard.path >> shard.offset >> shard.rows) || shard.rows <= 0)
        throw std::runtime_error("qwen38_graph: malformed PLE shard in " + table_path);
      shards.push_back(shard);
    }
  }
  if (params.n_heads <= 0 || params.n_heads > 32 || params.head_dim <= 0 || shards.empty() || shards.size() > 256)
    throw std::runtime_error("qwen38_graph: malformed PLE table " + table_path);
  params.n_shards = static_cast<int>(shards.size());
  params.capacity = capacity;
  const auto start = Clock::now();
  long long first_row = 0;
  const long page = sysconf(_SC_PAGESIZE);
  for (size_t i = 0; i < shards.size(); ++i) {
    const Shard& shard = shards[i];
    auto mapping = std::make_unique<Mapping>(plane);
    if (shard.path.find("..") != std::string::npos)
      throw std::runtime_error("qwen38_graph: PLE shard path must not contain '..': " + shard.path);
    const std::string full = shard.path[0] == '/' ? shard.path : root + "/" + shard.path;
    const int fd = open(full.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("qwen38_graph: cannot open PLE shard file " + full);
    const size_t bytes = static_cast<size_t>(shard.rows) * static_cast<size_t>(params.head_dim);
    const size_t aligned = shard.offset / page * page;
    const size_t delta = shard.offset - aligned;
    void* base = mmap(nullptr, bytes + delta, PROT_READ, MAP_SHARED, fd, static_cast<off_t>(aligned));
    close(fd);
    if (base == MAP_FAILED) throw std::runtime_error("qwen38_graph: mmap failed for " + full);
    mapping->base = base;
    mapping->length = bytes + delta;
    auto* rows = static_cast<unsigned char*>(base) + delta;
    // cudaHostRegisterMapped | cudaHostRegisterReadOnly: the gather kernel reads the table zero-copy over PCIe.
    plane.Check(plane.host_register(rows, bytes, 2 | 8), "cudaHostRegister(PLE shard)");
    mapping->registered = rows;
    mappings_.push_back(std::move(mapping));
    params.shard_ptr[i] = rows;
    first_row += shard.rows;
    params.shard_end[i] = first_row;
  }
  register_seconds = SecondsSince(start);
}

Qwen38PleTable::Mapping::~Mapping() {
  if (registered) plane.host_unregister(registered);
  if (base) munmap(base, length);
}

Qwen38PleTable::~Qwen38PleTable() = default;
// ------------------------------------------------------------------------------------------------
// Engine
// ------------------------------------------------------------------------------------------------
Qwen38GraphEngine::Qwen38GraphEngine(Qwen38GraphModel& model)
    : model_{model},
      plane_{*model.plane_},
      capacity_{model.config_->model.qwen38_graph.kv_capacity},
      vocab_{model.config_->model.vocab_size} {
  const auto& q = model_.config_->model.qwen38_graph;
  mtp_enabled_ = q.mtp.enabled;
  prefill_head_ = q.mtp.prefill_head;
  skip_refeed_ = q.mtp.skip_refeed;
  keep_hidden_ = mtp_enabled_ && prefill_head_ && model_.config_->model.dflash2.filename.empty();
  fold_ = q.fold.enabled;
  force_reject_ = q.fold.force_reject;
  suffix_lookup_mode_ = q.suffix_lookup;
  if (const char* value = std::getenv("ORT_QWEN38_SUFFIX_LOOKUP")) {
    const std::string mode{value};
    if (mode != "0" && mode != "1" && mode != "2")
      throw std::runtime_error("qwen38_graph: ORT_QWEN38_SUFFIX_LOOKUP must be 0, 1, or 2");
    suffix_lookup_mode_ = mode[0] - '0';
  }
  if (suffix_lookup_mode_ != 0 && (!fold_ || !mtp_enabled_))
    throw std::runtime_error("qwen38_graph: suffix lookup requires folded MTP decode");
  if (fold_ && !mtp_enabled_) throw std::runtime_error("qwen38_graph: the fold stack requires model.qwen38_graph.mtp.enabled");
  for (const auto& name : model_.session_target_->GetOutputNames()) {
    if (StartsWith(name, "present_mid3."))
      verify_rows_ = 4;
    else if (StartsWith(name, "present_mid2."))
      verify_rows_ = std::max(verify_rows_, 3);
  }
  if (verify_rows_ > 2 && !fold_)
    throw std::runtime_error("qwen38_graph: multi-draft prefix states require folded MTP");
  num_banks_ = fold_ ? verify_rows_ + 1 : kBanks;
  stream_ = model_.p_device_->GetCudaStream();
  plane_.Check(plane_.set_device(0), "cudaSetDevice");
  memory_info_ = model_.p_device_->GetMemoryInfo();

  void* host = nullptr;
  plane_.Check(plane_.host_alloc(&host, kRing * 8 * sizeof(int64_t)), "cudaHostAlloc(result ring)");
  result_host_ = static_cast<int64_t*>(host);
  plane_.Check(plane_.host_alloc(&host, 16 * sizeof(int64_t)), "cudaHostAlloc(stage)");
  stage_host_ = static_cast<int64_t*>(host);
  std::memset(result_host_, 0, kRing * 8 * sizeof(int64_t));
  std::memset(stage_host_, 0, 16 * sizeof(int64_t));

  // control block (the fold model binds it as input, so it exists before the target bindings)
  ctl_ = PlaneBuffer(plane_, 16 * sizeof(int64_t), stream_);
  ctx_commit_ = PlaneBuffer(plane_, 2 * sizeof(int64_t), stream_);
  ctx_a_ = PlaneBuffer(plane_, kMaxVerifyRows * 2 * sizeof(int64_t), stream_);
  ctx_b_ = PlaneBuffer(plane_, 2 * sizeof(int64_t), stream_);
  if (fold_) BuildHeadState();  // the fold model binds the head state
  BuildTarget();
  const auto ple_start = Clock::now();
  ple_ = std::make_unique<Qwen38PleTable>(plane_, model_.config_->config_path.string() + "/" + q.ple_table,
                                          model_.config_->config_path.string(), capacity_);
  ple_register_seconds = SecondsSince(ple_start);
  params_dev_ = PlaneBuffer(plane_, sizeof(Qwen38PleParams), stream_);
  plane_.Check(plane_.memcpy_h2d(params_dev_.data(), &ple_->params, sizeof(Qwen38PleParams), stream_), "PLE params");
  SyncStream();  // ple_->params stays alive; the copy above is complete once the stream drains
  if (ple_width_ != ple_->params.n_heads * ple_->params.head_dim)
    throw std::runtime_error("qwen38_graph: PLE table width does not match the model's ple_embeddings input");

  scratch_i32_ = PlaneBuffer(plane_, sizeof(int32_t), stream_);
  part_v_ = PlaneBuffer(plane_, kArgmaxBlocks * sizeof(float), stream_);
  part_i_ = PlaneBuffer(plane_, kArgmaxBlocks * sizeof(int32_t), stream_);
  buffered_logits_ = PlaneBuffer(plane_, static_cast<size_t>(verify_rows_ - 1) * vocab_ * sizeof(uint16_t), stream_);
  const size_t slots = static_cast<size_t>(capacity_) + 3;
  d_tokens_ = PlaneBuffer(plane_, slots * sizeof(int64_t), stream_);
  d_arange64_ = PlaneBuffer(plane_, slots * sizeof(int64_t), stream_);
  d_arange32_ = PlaneBuffer(plane_, slots * sizeof(int32_t), stream_);
  d_last_hidden_ = PlaneBuffer(plane_, static_cast<size_t>(hidden_width_) * kMaxVerifyRows * 2, stream_);
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
  if (fold_ && model_.config_->model.dflash2.filename.empty())
    BuildHeadR2();
  else if (mtp_enabled_ && !fold_)
    BuildHead();
  if (q.prefill.enabled) BuildFastPrefill();
  if (!model_.config_->model.dflash2.filename.empty()) {
    if (!fold_ || !fast_ || !q.prefill.unified || suffix_lookup_mode_ != 0 || q.warmup)
      throw std::runtime_error("DFlash experiment requires unified folded prefill, no suffix lookup, and warmup=false");
    if (model_.config_->model.dflash2.num_draft_tokens != verify_rows_ - 1)
      throw std::runtime_error("DFlash draft count must match target prefix-state width");
    auto draft_config = CreateDflash2Config(*model_.config_);
    draft_config->model.embedding.filename.clear();
    dflash_model_ = std::make_shared<Dflash2Model>(std::move(draft_config), GetOrtEnv());
    if (dflash_model_->p_device_->GetCudaStream() != stream_)
      throw std::runtime_error("DFlash device pipeline requires a shared CUDA stream");
    const auto cache_shape = dflash_model_->session_info_.GetInputShape("past_key_values.0.key");
    if (cache_shape.size() != 4 || (cache_shape[1] != 16 && cache_shape[1] != 128))
      throw std::runtime_error("DFlash experiment requires cache pages of 16 or 128 tokens");
    const size_t page_size = static_cast<size_t>(cache_shape[1]);
    dflash_ = std::make_unique<Dflash2Drafter>(
        dflash_model_, page_size, Dflash2Drafter::PoolBlocks(*model_.config_, page_size, 1), 1);
    d_dflash_prompt_ = PlaneBuffer(plane_, size_t(fast_->chunk) * kDflashWidth * 2, stream_);
  }
  Reset();
  if (dflash_) {
    const auto start = Clock::now();
    std::vector<int32_t> dummy(verify_rows_ * num_banks_ * 8, 1000);
    StepPrefill(dummy);
    Reset();
    warmup_seconds = SecondsSince(start);
  }
  if (q.warmup) {
    const auto start = Clock::now();
    Warmup();
    warmup_seconds = SecondsSince(start);
  }
  if (Verbose())
    std::cerr << "[qwen38_graph] engine ready: ple register " << ple_register_seconds << " s, warmup " << warmup_seconds
              << " s, mtp=" << mtp_enabled_ << std::endl;
}

Qwen38GraphEngine::~Qwen38GraphEngine() {
  try {
    SyncStream();
  } catch (...) {
  }
  // IO bindings and OrtValues reference the buffers: release them first
  for (auto& row : dirs_)
    for (auto& d : row) d.binding.reset();
  for (auto& h : head_)
    for (auto& d : h.dirs) d.binding.reset();
  for (auto& d : head_r2_.dirs) d.binding.reset();
  fast_.reset();
  if (result_host_) plane_.host_free(result_host_);
  if (stage_host_) plane_.host_free(stage_host_);
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
  const auto& q = model_.config_->model.qwen38_graph;
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
  if (fold_) {
    for (const auto& name : bank_names) {
      for (int prefix = 1; prefix < verify_rows_; ++prefix) {
        const std::string tag = prefix == 1 ? "present_mid." : "present_mid" + std::to_string(prefix) + ".";
        if (!info.HasOutput(tag + name.substr(16)))
          throw std::runtime_error("qwen38_graph: missing prefix state " + tag + name.substr(16));
      }
    }
    const auto ctl_shape = info.GetInputShape("fold.ctl");
    if (ctl_shape != std::vector<int64_t>{16} &&
        (verify_rows_ > 2 || ctl_shape != std::vector<int64_t>{8}))
      throw std::runtime_error("qwen38_graph: folded control shape does not match verification width");
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

  // S=1 needs a scratch bank; folded S=2 only uses the three-bank ring.
  size_t arena_bytes = 0;
  std::map<std::string, size_t> offsets;
  for (const auto& name : bank_names) {
    arena_bytes = AlignUp(arena_bytes, 256);
    offsets[name] = arena_bytes;
    arena_bytes += Elements(FixedShape(info.GetInputShape(name))) * Ort::SizeOf(info.GetInputDataType(name));
  }
  arena_bytes = AlignUp(arena_bytes, 256);
  const int num_banks = fold_ ? num_banks_ : num_banks_ + 1;
  for (int b = 0; b < num_banks; ++b) {
    Bank& bank = banks_[b];
    bank.arena = PlaneBuffer(plane_, arena_bytes, stream_);
    for (const auto& name : bank_names)
      bank.tensors[name] = MakeTensor(bank.arena.data() + offsets[name], FixedShape(info.GetInputShape(name)),
                                      info.GetInputDataType(name));
  }

  if (fold_) {
    // fold.* control inputs alias the device control block; fold.ptrs = (pinned result ring address, ring size)
    fold_ptrs_ = PlaneBuffer(plane_, 2 * sizeof(int64_t), stream_);
    const int64_t ptrs[2] = {static_cast<int64_t>(reinterpret_cast<intptr_t>(result_host_)), kRing};
    plane_.Check(plane_.memcpy_h2d(fold_ptrs_.data(), ptrs, sizeof(ptrs), stream_), "fold ptrs");
    SyncStream();
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
    if (info.HasInput("logits_indices"))
      set64(NewDeviceTensor(in.tensors, "logits_indices", {rows}, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64), arange);
    in.ple = NewDeviceTensor(in.tensors, "ple_embeddings." + std::to_string(ple_layer_), {1, rows, ple_width_},
                             ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)
                 ->ptr;
  }
  SyncStream();

  // One IO binding per (rows, ring phase). Graph k reads bank k; S=1 writes bank k+1 (present_mid -> scratch bank),
  // S=2 writes present_mid -> bank k+1 and present -> bank k+2.
  const auto dtype_logits = info.GetOutputDataType("logits");
  for (int rows = min_rows; rows <= max_rows; ++rows) {
    for (int k = 0; k < num_banks_; ++k) {
      Direction& d = dirs_[rows - 1][k];
      d.binding = OrtIoBinding::Create(session);
      Bank& in_bank = banks_[k];
      Bank& out_bank = banks_[(k + rows) % num_banks_];
      Bank& mid_bank = rows > 1 ? banks_[(k + 1) % num_banks_] : banks_[num_banks_];
      std::map<std::string, std::unique_ptr<DevTensor>> own;
      auto add_own = [&](const std::string& name, std::vector<int64_t> shape, ONNXTensorElementDataType type) {
        return NewDeviceTensor(own, name, std::move(shape), type);
      };
      for (const auto& name : in_names) {
        if (name == "prefill_numerics") {
          d.binding->BindInput(name.c_str(), PrefillNumerics(false));
          continue;
        }
        if (StartsWith(name, "prefill_cache.")) {
          d.binding->BindInput(name.c_str(), NumericalCache(name, info, false));
          continue;
        }
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
        if (StartsWith(name, "prefill_cache_out.")) {
          d.binding->BindOutput(name.c_str(), NumericalCache("prefill_cache." + name.substr(18), info, false));
          continue;
        }
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
        if (fold_ && StartsWith(name, "head.present_mid3.0."))
          t = head_state_for(name.substr(20), k + 3);
        else if (fold_ && StartsWith(name, "head.present_mid2.0."))
          t = head_state_for(name.substr(20), k + 2);
        else if (fold_ && StartsWith(name, "head.present_mid.0."))
          t = head_state_for(name.substr(19), k + 1);
        else if (fold_ && (StartsWith(name, "head.present_s1.0.") || StartsWith(name, "head.present_s2.0.")))
          t = head_state_for(name.substr(18), k);
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
        else if (StartsWith(name, "present_mid3."))
          t = state_for(name.substr(13), banks_[(k + 3) % num_banks_]);
        else if (StartsWith(name, "present_mid2."))
          t = state_for(name.substr(13), banks_[(k + 2) % num_banks_]);
        else if (StartsWith(name, "present_mid."))
          t = state_for(name.substr(12), mid_bank);
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
      const int id = fold_ ? q.fold.target_graph_id + k : (rows == 1 ? q.target_s1_graph_id : q.target_s2_graph_id) + k;
      d.run_options->AddConfigEntry("gpu_graph_id", std::to_string(id).c_str());
      d.run_options->AddConfigEntry("disable_synchronize_execution_providers", "1");
    }
  }
  past_pos_ = store_["past_position_ids"]->ptr;
}

void Qwen38GraphEngine::BuildHead() {
  const auto& q = model_.config_->model.qwen38_graph;
  head_[0].rows = 1;
  head_[0].session = model_.session_head_s1_.get();
  head_[1].rows = 2;
  head_[1].session = model_.session_head_s2_.get();
  for (auto& h : head_) h.info.Add(*h.session);
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
        if (name == "prefill_numerics") {
          d.binding->BindInput(name.c_str(), PrefillNumerics(false));
          continue;
        }
        if (StartsWith(name, "prefill_cache.")) {
          d.binding->BindInput(name.c_str(), NumericalCache(name, h.info, false, true));
          continue;
        }
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
        if (StartsWith(name, "prefill_cache_out.")) {
          d.binding->BindOutput(name.c_str(), NumericalCache("prefill_cache." + name.substr(18), h.info, false, true));
          continue;
        }
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
      const int id = (rows == 1 ? q.head_s1_graph_id : q.head_s2_graph_id) + dir;
      d.run_options->AddConfigEntry("gpu_graph_id", std::to_string(id).c_str());
      d.run_options->AddConfigEntry("disable_synchronize_execution_providers", "1");
    }
    for (auto& view : views) head_views_.push_back(std::move(view.second));
  }
  SyncStream();
}

// Head KV / compressed index keys in place, indexer small state in 2 ping-pong banks (S=1/S=2 head sessions) or in the
// 3-bank ring of the fold graph.
void Qwen38GraphEngine::BuildHeadState() {
  const SessionInfo& info = fold_ ? model_.session_info_ : head_[0].info;
  const std::string prefix = fold_ ? "head.past_key_values.0." : "past_key_values.0.";
  if (!model_.config_->model.dflash2.filename.empty() && !info.HasInput(prefix + "key")) return;
  for (const char* tail : {"key", "value", "index_key"})
    NewDeviceTensor(head_store_, std::string("past_key_values.0.") + tail, FixedShape(info.GetInputShape(prefix + tail)),
                    info.GetInputDataType(prefix + tail));
  const int banks = fold_ ? num_banks_ : 2;
  for (int b = 0; b < banks; ++b) {
    for (const char* tail : {"index_buffer", "index_lengths"})
      NewDeviceTensor(head_banks_[b].tensors, std::string("past_key_values.0.") + tail,
                      FixedShape(info.GetInputShape(prefix + tail)), info.GetInputDataType(prefix + tail));
  }
}

void Qwen38GraphEngine::ClearHead() {
  for (auto& entry : head_store_) plane_.Check(plane_.memset_(entry.second->ptr, 0, entry.second->Bytes(), stream_), "clear head");
  for (auto& bank : head_banks_)
    for (auto& entry : bank.tensors)
      plane_.Check(plane_.memset_(entry.second->ptr, 0, entry.second->Bytes(), stream_), "clear head bank");
  head_length_ = 0;
  head_turn_ = 0;
  head_cur_ = 0;
}

void Qwen38GraphEngine::Reset() {
  SyncStream();
  if (dflash_) dflash_->ReleaseAll();
  dflash_prompt_rows_ = 0;
  for (auto& bank : banks_) {
    if (bank.arena.data())
      plane_.Check(plane_.memset_(bank.arena.data(), 0, bank.arena.size(), stream_), "clear bank");
  }
  for (auto& entry : store_) plane_.Check(plane_.memset_(entry.second->ptr, 0, entry.second->Bytes(), stream_), "clear store");
  for (auto& entry : shared_) plane_.Check(plane_.memset_(entry.second->ptr, 0, entry.second->Bytes(), stream_), "clear index");
  plane_.Check(plane_.memset_(ctl_.data(), 0, ctl_.size(), stream_), "clear ctl");
  std::memset(result_host_, 0, kRing * 8 * sizeof(int64_t));
  stage_host_[0] = stage_host_[1] = ple_->params.eos;
  plane_.Check(plane_.memcpy_h2d(ctx_commit_.data(), stage_host_, 16, stream_), "ctx init");
  if (mtp_enabled_) ClearHead();
  SyncStream();
  FastReset();
  seq_ = 0;
  cur_ = 0;
  length_ = 0;
  inflight_ = false;
  decode_started_ = false;
  pending_ = -1;
  prompt_tokens_.clear();
  suffix_drafter_.Reset();
  suffix_lookup_attempted_ = false;
  suffix_lookup_hit_ = false;
  suffix_lookup_tokens_proposed_ = 0;
  last_logits_ptr_ = nullptr;
  last_hidden_ptr_ = nullptr;
  stats_ = {};
}

void Qwen38GraphEngine::PrepRow(int rows, int row, const void* tok, const void* pos, int add, const void* ctx_in,
                                void* ctx_out) {
  const RowInputs& in = inputs_[rows - 1];
  plane_.prep(stream_, params_dev_.data(), tok, pos, add, ctx_in, ctx_out, in.input_ids + 8 * row,
              in.position_ids + 8 * row, row == 0 ? in.committed : scratch_i32_.data(), past_pos_,
              in.ple + 2 * static_cast<size_t>(ple_width_) * row, ctl_.data());
}

void Qwen38GraphEngine::RunTarget(int rows, int phase) {
  Direction& d = dirs_[rows - 1][phase % num_banks_];
  model_.session_target_->Run(d.run_options.get(), *d.binding);
}

int64_t Qwen38GraphEngine::ReadCtl(int index) {
  if (index < 0 || index >= 16) throw std::runtime_error("qwen38_graph: invalid control read");
  plane_.Check(plane_.memcpy_d2h(stage_host_, ctl_.data() + 8 * index, 8, stream_), "read ctl");
  SyncStream();
  return stage_host_[0];
}

void Qwen38GraphEngine::WriteCtl(int first, std::span<const int64_t> values) {
  if (first < 0 || first > 16 || values.size() > static_cast<size_t>(16 - first) || values.size() > 12)
    throw std::runtime_error("qwen38_graph: control write exceeds staging or control storage");
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
void Qwen38GraphEngine::MutedPrefillBegin() {
  if (!fold_) return;
  const int64_t one = 1;  // ctl[3] = mute: the fold graph runs without its decode bookkeeping
  WriteCtl(3, {&one, 1});
}

void Qwen38GraphEngine::MutedPrefillEnd() {
  if (!fold_) return;
  const int64_t zero = 0;
  WriteCtl(3, {&zero, 1});
}

void Qwen38GraphEngine::PrefillChunk(std::span<const int32_t> tokens) {
  if (fast_)
    FastAppend(tokens);
  else
    StepPrefill(tokens);
  if (suffix_lookup_mode_ != 0) suffix_drafter_.AppendSpan(tokens);
}

// Prompt ingestion by S=2 graph steps (the fold graph runs muted).
void Qwen38GraphEngine::StepPrefill(std::span<const int32_t> tokens) {
  if (decode_started_) throw std::runtime_error("qwen38_graph: AppendTokens after generation started is not supported");
  MutedPrefillBegin();
  const size_t n = tokens.size();
  if (n == 0) throw std::runtime_error("qwen38_graph: empty prefill chunk");
  const int rows = fold_ ? verify_rows_ : 2;
  if (length_ + n + std::max(rows, 3) > static_cast<size_t>(capacity_))
    throw std::runtime_error("qwen38_graph: prompt exceeds the KV capacity of " + std::to_string(capacity_));
  const size_t base = length_;
  {
    std::vector<int64_t> chunk(n + rows - 1, 0);
    for (size_t i = 0; i < n; ++i) chunk[i] = tokens[i];
    plane_.Check(plane_.memcpy_h2d(d_tokens_.data() + 8 * base, chunk.data(), chunk.size() * 8, stream_), "upload prompt");
  }
  prompt_tokens_.insert(prompt_tokens_.end(), tokens.begin(), tokens.end());
  const uint8_t* tp = d_tokens_.data();
  const uint8_t* pp = d_arange64_.data();
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  size_t i = 0;
  const Direction* last = nullptr;
  int last_row = 0;
  while (i < n) {
    const size_t step = std::min(n - i, static_cast<size_t>(rows));
    const size_t gi = base + i;
    for (int row = 0; row < rows; ++row) {
      const void* ctx_in = row == 0 ? ctx_commit_.data() : ctx_a_.data() + 16 * (row - 1);
      void* ctx_out = step == static_cast<size_t>(rows) && row == rows - 1
                          ? ctx_commit_.data()
                          : ctx_a_.data() + 16 * row;
      PrepRow(rows, row, tp + 8 * (gi + row), pp + 8 * (gi + row), 0, ctx_in, ctx_out);
    }
    RunTarget(rows, cur_);
    Direction& d = dirs_[rows - 1][cur_ % num_banks_];
    plane_.or_status(stream_, d.status, static_cast<int>(d.status_count), ctl_.data());
    if (step < static_cast<size_t>(rows))
      plane_.Check(plane_.memcpy_d2d(ctx_commit_.data(), ctx_a_.data() + 16 * (step - 1), 16, stream_), "ctx commit");
    if (keep_hidden_)
      plane_.Check(plane_.memcpy_d2d(d_prompt_hidden_.data() + gi * row_bytes, d.hidden, step * row_bytes, stream_),
                   "keep hidden");
    cur_ = static_cast<int>((cur_ + step) % num_banks_);
    i += step;
    last = &d;
    last_row = static_cast<int>(step) - 1;
  }
  plane_.Check(plane_.memcpy_d2d(d_last_hidden_.data(), last->hidden + static_cast<size_t>(last_row) * row_bytes, row_bytes,
                                 stream_),
               "last hidden");
  plane_.Check(plane_.memcpy_d2h(stage_host_ + 4, last->tokens + 8 * last_row, 8, stream_), "read prediction");
  SyncStream();
  if (ReadCtl(5) != 0) CheckDeviceStatus("prefill", last);
  pending_ = static_cast<int32_t>(stage_host_[4]);  // ReadCtl reuses stage_host_[0]; slot 4 is untouched
  MutedPrefillEnd();                                // (WriteCtl uses stage_host_ + 4)
  last_logits_ptr_ = last->logits;
  last_logits_row_ = last_row;
  length_ += n;
}

void Qwen38GraphEngine::LastLogits(std::span<float> out) const {
  if (fast_ && !last_logits_host_.empty()) {
    std::copy_n(last_logits_host_.begin(), std::min(out.size(), last_logits_host_.size()), out.begin());
    return;
  }
  if (!last_logits_ptr_) throw std::runtime_error("qwen38_graph: no logits available (no prompt was processed)");
  std::vector<uint16_t> raw(static_cast<size_t>(vocab_));
  plane_.Check(plane_.memcpy_d2h(raw.data(), last_logits_ptr_ + static_cast<size_t>(last_logits_row_) * vocab_ * 2,
                                 raw.size() * 2, stream_),
               "read logits");
  plane_.Check(plane_.stream_sync(stream_), "sync logits");
  for (size_t i = 0; i < raw.size() && i < out.size(); ++i) out[i] = Bf16ToFloat(raw[i]);
}

void Qwen38GraphEngine::SaveBufferedLogits(size_t buffered_tokens) {
  if (!inflight_dir_) throw std::logic_error("qwen38_graph: no completed cycle logits");
  if (buffered_tokens == 0 || buffered_tokens >= static_cast<size_t>(verify_rows_))
    throw std::logic_error("qwen38_graph: incorrect buffered logits count");
  const size_t row_bytes = static_cast<size_t>(vocab_) * sizeof(uint16_t);
  // Preserve rows predicting the remaining emissions before look-ahead reuses the direction.
  plane_.Check(plane_.memcpy_d2d(buffered_logits_.data(), inflight_dir_->logits + row_bytes,
                                 buffered_tokens * row_bytes, stream_),
               "save buffered target logits");
  buffered_logits_count_ = buffered_tokens;
}

void Qwen38GraphEngine::DecodeLogits(size_t buffered_tokens, std::span<float> out) const {
  if (!buffered_tokens && !inflight_)
    throw std::runtime_error("qwen38_graph: logits are unavailable after generation is complete");
  if (buffered_tokens > buffered_logits_count_)
    throw std::logic_error("qwen38_graph: incorrect remaining buffered logits count");
  const uint8_t* source = buffered_tokens
                              ? buffered_logits_.data() +
                                    (buffered_logits_count_ - buffered_tokens) * vocab_ * sizeof(uint16_t)
                              : inflight_dir_->logits;
  std::vector<uint16_t> raw(static_cast<size_t>(vocab_));
  if (out.size() != raw.size()) throw std::logic_error("qwen38_graph: incorrect decode logits size");
  plane_.Check(plane_.memcpy_d2h(raw.data(), source, raw.size() * sizeof(uint16_t), stream_), "read decode logits");
  plane_.Check(plane_.stream_sync(stream_), "sync decode logits");
  for (size_t i = 0; i < raw.size(); ++i) out[i] = Bf16ToFloat(raw[i]);
}

// ---- head -------------------------------------------------------------------------------------
void Qwen38GraphEngine::RunHeadOnce(HeadSession& head, int rows) {
  (void)rows;
  Direction& d = head.dirs[head_turn_ % 2];
  head.session->Run(d.run_options.get(), *d.binding);
}

void Qwen38GraphEngine::EnqueueHead(int rows, const uint8_t* hidden_src) {
  HeadSession& head = head_[rows - 1];
  if (head_length_ + rows > capacity_) throw std::runtime_error("qwen38_graph: head cache capacity exceeded");
  plane_.head_prep(stream_, ctl_.data(), rows, head_length_, hidden_src, head.hidden, head.tokens, head.positions,
                   head.committed);
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

// ---- fold (v12): standalone R=2 head, prompt context by pairs, first draft ---------------------------------------------
void Qwen38GraphEngine::BuildHeadR2() {
  const auto& q = model_.config_->model.qwen38_graph;
  HeadR2Session& h = head_r2_;
  h.session = model_.session_head_r2_.get();
  h.info.Add(*h.session);
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
      if (name == "prefill_numerics") {
        d.binding->BindInput(name.c_str(), PrefillNumerics(false));
        continue;
      }
      if (StartsWith(name, "prefill_cache.")) {
        d.binding->BindInput(name.c_str(), NumericalCache(name, h.info, false, true));
        continue;
      }
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
      if (StartsWith(name, "prefill_cache_out.")) {
        d.binding->BindOutput(name.c_str(), NumericalCache("prefill_cache." + name.substr(18), h.info, false, true));
        continue;
      }
      DevTensor* t = nullptr;
      const std::string state = "past_key_values." + name.substr(name.find('.') + 1);
      if (h.tensors.count(name))
        t = h.tensors[name].get();
      else if (StartsWith(name, "present_mid."))
        t = mid_bank.tensors.at(state).get();
      else if (StartsWith(name, "present.")) {
        if (head_store_.count(state))
          t = head_store_[state].get();
        else
          t = out_bank.tensors.at(state).get();
      } else
        throw std::runtime_error("qwen38_graph: unsupported head_r2 output " + name);
      d.binding->BindOutput(name.c_str(), *t->value);
    }
    d.run_options = OrtRunOptions::Create();
    d.run_options->AddConfigEntry("gpu_graph_id", std::to_string(q.fold.head_r2.graph_id + k).c_str());
    d.run_options->AddConfigEntry("disable_synchronize_execution_providers", "1");
  }
  SyncStream();
}

// One standalone head call at the current ring phase: rows (tokens[i], hidden_i) at positions[i], head cache length
// head_length_; commits `commit_rows` (1 or 2) rows and rotates the ring (HeadR2.call).
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
  h.session->Run(d.run_options.get(), *d.binding);
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
// verify ring phase; ctl = (pending, L, draft, ..., head length).
void Qwen38GraphEngine::FoldFirstDraft() {
  const int64_t tokens[2] = {pending_, pending_};
  const int64_t positions[2] = {static_cast<int64_t>(length_), static_cast<int64_t>(length_) + 1};
  CallHeadR2(tokens, positions, d_last_hidden_.data(), d_last_hidden_.data(), 1);
  plane_.Check(plane_.memcpy_d2d(ctl_.data() + 16, head_r2_.next_token, 8, stream_), "first draft");
  SyncStream();
  if (ReadCtl(5) != 0) CheckDeviceStatus("first draft", nullptr);
  if (head_cur_ != cur_) {
    for (auto& entry : head_banks_[head_cur_].tensors)
      plane_.Check(plane_.memcpy_d2d(head_banks_[cur_].tensors.at(entry.first)->ptr, entry.second->ptr, entry.second->Bytes(), stream_),
                   "rotate head ring");
    head_cur_ = cur_;
  }
  const int committed_head_length = head_length_;
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  for (int draft = 2; draft < verify_rows_; ++draft) {
    auto* previous_hidden = d_last_hidden_.data() + static_cast<size_t>(draft - 1) * row_bytes;
    plane_.Check(plane_.memcpy_d2d(previous_hidden, head_r2_.tensors.at("hidden_states_out")->ptr, row_bytes, stream_),
                 "save chained MTP hidden");
    const int previous_slot = draft == 2 ? 2 : 8;
    const int64_t previous_token = ReadCtl(previous_slot);
    const int64_t next_tokens[2] = {previous_token, previous_token};
    const int64_t next_positions[2] = {static_cast<int64_t>(length_) + draft - 1,
                                       static_cast<int64_t>(length_) + draft};
    CallHeadR2(next_tokens, next_positions, previous_hidden, previous_hidden, 1);
    plane_.Check(plane_.memcpy_d2d(ctl_.data() + 8 * (draft + 6), head_r2_.next_token, 8, stream_),
                 "chained MTP draft");
    SyncStream();
    if (ReadCtl(5) != 0) CheckDeviceStatus("chained draft", nullptr);
  }
  head_cur_ = cur_;
  head_length_ = committed_head_length;
  const int64_t state[2] = {pending_, static_cast<int64_t>(length_)};
  WriteCtl(0, {state, 2});
  const int64_t head_length = head_length_;
  WriteCtl(7, {&head_length, 1});
}

// Test hook: the verify of the next cycle sees a draft that differs from the head's (the python draft_hook), so the cycle
// commits exactly the greedy token of verify row 0. A coincidence with that token is accepted by the graph and stays valid.
void Qwen38GraphEngine::ForceRejectDraft() {
  SyncStream();
  const int64_t forced = (ReadCtl(2) + 1) % vocab_;
  WriteCtl(2, {&forced, 1});
  for (int draft = 2; draft < verify_rows_; ++draft) {
    const int slot = draft + 6;
    const int64_t forced_extra = (ReadCtl(slot) + 1) % vocab_;
    WriteCtl(slot, {&forced_extra, 1});
  }
}

void Qwen38GraphEngine::EnqueueFold() {
  if (force_reject_) ForceRejectDraft();
  if (length_ + verify_rows_ > static_cast<size_t>(capacity_)) throw std::runtime_error("qwen38_graph: KV capacity reached");
  const uint8_t* ctl = ctl_.data();
  constexpr int slots[kMaxVerifyRows] = {0, 2, 8, 9};
  for (int row = 0; row < verify_rows_; ++row) {
    const void* ctx_in = row == 0 ? ctx_commit_.data() : ctx_a_.data() + 16 * (row - 1);
    PrepRow(verify_rows_, row, ctl + 8 * slots[row], ctl + 8, row, ctx_in, ctx_a_.data() + 16 * row);
  }
  RunTarget(verify_rows_, cur_);
  Direction& d = dirs_[verify_rows_ - 1][cur_ % num_banks_];
  last_hidden_ptr_ = d.hidden;
  inflight_dir_ = &d;
  ++seq_;
  inflight_ = true;
}

void Qwen38GraphEngine::CheckErrorBits() {
  if (!decode_started_) return;
  const int64_t bits = ReadCtl(5);
  if (bits != 0) throw std::runtime_error("qwen38_graph: device error bits " + std::to_string(bits) + " after decode");
}

// ---- decode -----------------------------------------------------------------------------------
void Qwen38GraphEngine::EnqueueVerify() {
  if (length_ + 2 > static_cast<size_t>(capacity_)) throw std::runtime_error("qwen38_graph: KV capacity reached");
  const uint8_t* ctl = ctl_.data();
  PrepRow(2, 0, ctl, ctl + 8, 0, ctx_commit_.data(), ctx_a_.data());
  PrepRow(2, 1, ctl + 16, ctl + 8, 1, ctx_a_.data(), ctx_b_.data());
  RunTarget(2, cur_);
  Direction& d = dirs_[1][cur_ % kBanks];
  plane_.post_tok(stream_, d.tokens, ctl_.data(), ctx_commit_.data(), ctx_a_.data(), ctx_b_.data(), d.status,
                  static_cast<int>(d.status_count), result_host_, kRing);
  last_hidden_ptr_ = d.hidden;
  inflight_dir_ = &d;
  ++seq_;
  inflight_ = true;
}

void Qwen38GraphEngine::EnqueueStep() {
  if (length_ + 1 > static_cast<size_t>(capacity_)) throw std::runtime_error("qwen38_graph: KV capacity reached");
  const uint8_t* ctl = ctl_.data();
  PrepRow(1, 0, ctl, ctl + 8, 0, ctx_commit_.data(), ctx_a_.data());
  RunTarget(1, cur_);
  Direction& d = dirs_[0][cur_ % kBanks];
  plane_.argmax_part(stream_, d.logits, vocab_, part_v_.data(), part_i_.data(), kArgmaxBlocks);
  plane_.post1(stream_, part_v_.data(), part_i_.data(), kArgmaxBlocks, ctl_.data(), ctx_commit_.data(), ctx_a_.data(),
               d.status, static_cast<int>(d.status_count), result_host_, kRing);
  last_hidden_ptr_ = d.hidden;
  inflight_dir_ = &d;
  ++seq_;
  inflight_ = true;
}

void Qwen38GraphEngine::BeginDecode(bool run_cycles) {
  if (decode_started_) return;
  if (length_ == 0) throw std::runtime_error("qwen38_graph: GenerateNextToken called before AppendTokens");
  decode_started_ = true;
  if (dflash_) {
    if (!run_cycles) return;
    FastHandoff();
    const int64_t state[2] = {pending_, static_cast<int64_t>(length_)};
    WriteCtl(0, state);
    if (!dflash_prompt_rows_) throw std::runtime_error("DFlash experiment lacks prompt features");
    Draft(d_dflash_prompt_.data(), dflash_prompt_rows_, dflash_prompt_position_, pending_, true);
    dflash_prompt_rows_ = 0;
    EnqueueFold();
    return;
  }
  if (fold_) {
    if (!run_cycles) return;
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
  if (mtp_enabled_ && run_cycles) {
    ClearHead();
    SyncStream();
    if (prefill_head_) HeadPrefill();
  }
  const int64_t state[2] = {pending_, static_cast<int64_t>(length_)};
  WriteCtl(0, {state, 2});
  if (!run_cycles) return;
  if (mtp_enabled_) {
    EnqueueHead(1, d_last_hidden_.data());
    EnqueueVerify();
  } else {
    EnqueueStep();
  }
}

Qwen38CycleResult Qwen38GraphEngine::WaitCycle() {
  if (!inflight_) throw std::logic_error("qwen38_graph: no cycle in flight");
  const size_t slot = ((seq_ - 1) % kRing) * 8;
  volatile const int64_t* record = result_host_ + slot;
  const auto start = Clock::now();
  uint32_t spins = 0;
  while (record[0] != static_cast<int64_t>(seq_)) {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#endif
    if (((++spins) & (kStatusRingSpinCheck - 1)) == 0) {
      if (SecondsSince(start) > 180.0) throw std::runtime_error("qwen38_graph: timeout waiting for the cycle record");
      std::this_thread::yield();
    }
  }
  std::atomic_thread_fence(std::memory_order_acquire);
  Qwen38CycleResult result;
  result.pred0 = record[1];
  result.pred1 = record[2];
  result.proposed_count = mtp_enabled_ ? verify_rows_ - 1 : 1;
  if (record[3] < 0 || record[3] > result.proposed_count)
    throw std::runtime_error("qwen38_graph: invalid device accepted-prefix count");
  result.accepted_count = mtp_enabled_ ? static_cast<int>(record[3]) : 0;
  result.accepted = result.accepted_count > 0;
  result.status = record[4];
  result.new_pending = record[5];
  result.new_length = record[6];
  if (verify_rows_ > 2) {
    const uint64_t packed = static_cast<uint64_t>(record[7]);
    result.pred2 = static_cast<int64_t>(static_cast<int32_t>(packed & 0xffffffffu));
    result.pred3 = static_cast<int64_t>(static_cast<int32_t>(packed >> 32));
  }
  result.suffix_lookup_attempted = suffix_lookup_attempted_;
  result.suffix_lookup_hit = suffix_lookup_hit_;
  result.suffix_lookup_tokens_proposed = suffix_lookup_tokens_proposed_;
  inflight_ = false;
  if (result.status != 0) {
    CheckDeviceStatus("decode", inflight_dir_);
    throw std::runtime_error("qwen38_graph: decode reported device control/validation error " +
                             std::to_string(result.status));
  }
  return result;
}

void Qwen38GraphEngine::Advance(const Qwen38CycleResult& result, bool want_more) {
  const size_t step = mtp_enabled_ ? static_cast<size_t>(result.TokenCount()) : 1;
  cur_ = static_cast<int>((cur_ + step) % num_banks_);
  length_ += step;
  if (fold_) head_length_ += static_cast<int>(step);
  if (static_cast<int64_t>(length_) != result.new_length)
    throw std::runtime_error("qwen38_graph: device length " + std::to_string(result.new_length) +
                             " differs from host length " + std::to_string(length_));
  stats_.cycles += 1;
  stats_.accepted += result.AcceptedCount();
  stats_.tokens += step;
  stats_.suffix_lookup_hits += result.suffix_lookup_hit ? 1 : 0;
  stats_.suffix_lookup_misses += result.suffix_lookup_attempted && !result.suffix_lookup_hit ? 1 : 0;
  suffix_lookup_attempted_ = false;
  suffix_lookup_hit_ = false;
  suffix_lookup_tokens_proposed_ = 0;
  if (suffix_lookup_mode_ != 0) {
    suffix_drafter_.AppendToken(pending_);
    const int64_t predictions[kMaxVerifyRows] = {result.pred0, result.pred1, result.pred2, result.pred3};
    for (int i = 0; i < result.AcceptedCount(); ++i)
      suffix_drafter_.AppendToken(static_cast<int32_t>(predictions[i]));
  }
  pending_ = static_cast<int32_t>(result.new_pending);
  if (!want_more) return;
  if (dflash_) {
    Draft(inflight_dir_->auxiliary, step, length_ - step, pending_, true);
    EnqueueFold();
    return;
  }
  if (fold_) {
    if (suffix_lookup_mode_ == 1 || (suffix_lookup_mode_ == 2 && !result.accepted)) {
      suffix_lookup_attempted_ = true;
      std::array<int32_t, 3> drafts{};
      const size_t count = suffix_drafter_.ProposeDrafts(pending_, std::span<int32_t>(drafts).first(verify_rows_ - 1));
      suffix_lookup_tokens_proposed_ = count;
      constexpr int slots[3] = {2, 8, 9};
      for (size_t i = 0; i < count; ++i) {
        const int64_t draft = drafts[i];
        WriteCtl(slots[i], {&draft, 1});
        suffix_lookup_hit_ = true;
      }
    }
    head_cur_ = cur_;
    EnqueueFold();
    return;
  }
  if (mtp_enabled_) {
    const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
    if (result.accepted) {
      if (skip_refeed_)
        EnqueueHead(1, last_hidden_ptr_ + row_bytes);
      else
        EnqueueHead(2, last_hidden_ptr_);
    } else {
      EnqueueHead(1, last_hidden_ptr_);
    }
    EnqueueVerify();
  } else {
    EnqueueStep();
  }
}

void Qwen38GraphEngine::Drain() {
  SyncStream();
}

void Qwen38GraphEngine::Draft(const uint8_t* features, size_t rows, size_t position,
                              int32_t anchor, bool propose) {
  if (Verbose()) std::cerr << "[DFlash] ingest " << rows << " at " << position << " propose=" << propose << std::endl;
  if (!features || rows == 0) throw std::runtime_error("DFlash experiment requires committed feature rows");
  if (!plane_.dflash_select) SyncStream();
  const int64_t shape[2] = {int64_t(rows), int64_t(kDflashWidth)};
  Tensor auxiliary(model_.p_device_, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  auxiliary.ort_tensor_ = OrtValue::CreateTensor(*memory_info_, const_cast<uint8_t*>(features),
                                                 rows * kDflashWidth * 2, shape,
                                                 ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  Dflash2Drafter::Feed feed{};
  // A single resident request uses the null identity; the drafter never dereferences it.
  feed.aux_row_count = rows;
  feed.first_position = position;
  feed.anchor_token = anchor;
  feed.draft_eligible = true;
  feed.wants_drafts = propose;
  std::vector<std::vector<int32_t>> drafts;
  if (!dflash_->Propose(auxiliary, {&feed, 1}, drafts, nullptr, nullptr, plane_.dflash_select != nullptr))
    throw std::runtime_error("DFlash experiment could not ingest the committed target features");
  if (Verbose()) std::cerr << "[DFlash] proposal finished" << std::endl;
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
    constexpr int slots[3] = {2, 8, 9};
    for (size_t i = 0; i < drafts[0].size(); ++i) {
      stage_host_[4 + i] = drafts[0][i];
      plane_.Check(plane_.memcpy_h2d(ctl_.data() + 8 * slots[i], stage_host_ + 4 + i,
                                     sizeof(int64_t), stream_),
                   "write DFlash draft");
    }
    SyncStream();
  }
}

void Qwen38GraphEngine::Warmup() {
  // ORT captures a graph id after min_num_runs eager runs: touch every (rows, phase) and head direction enough times.
  const int warm_tokens = (fold_ ? verify_rows_ : 2) * num_banks_ * 8;
  std::vector<int32_t> dummy(warm_tokens);
  for (int i = 0; i < warm_tokens; ++i) dummy[i] = 1000 + i;
  StepPrefill(dummy);  // S=2 target ids (24 steps -> 8 runs per phase); fold: muted cycle graphs
  if (fold_) {
    // standalone R=2 head graphs (ids 81-83): one call commits one row, so 18 calls touch every phase 6 times
    ClearHead();
    SyncStream();
    for (int i = 0; i < num_banks_ * 6; ++i) {
      const int64_t tokens[2] = {1000 + i, 1000 + i}, positions[2] = {i + 1, i + 2};
      CallHeadR2(tokens, positions, d_last_hidden_.data(), d_last_hidden_.data(), 1);
    }
    SyncStream();
    if (ReadCtl(5) != 0) CheckDeviceStatus("warmup (head)", nullptr);
    ClearHead();
    SyncStream();
    if (fast_) {
      // dense / head-KV shapes: arena growth and first-use costs of the prefill session (one dummy prompt per length)
      std::mt19937 rng(12345);
      for (int length : model_.config_->model.qwen38_graph.prefill.warmup_lengths) {
        if (length < 2 || static_cast<size_t>(length) + 3 > static_cast<size_t>(capacity_)) continue;
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
  for (int i = 0; i < 24; ++i) {  // S=1 target ids
    const int64_t state[2] = {pending_, static_cast<int64_t>(length_)};
    WriteCtl(0, {state, 2});
    EnqueueStep();
    WaitCycle();
    cur_ = (cur_ + 1) % kBanks;
    length_ += 1;
  }
  if (mtp_enabled_) {
    ClearHead();
    SyncStream();
    for (int i = 0; i < 24; ++i) EnqueueHead(1, d_last_hidden_.data());
    for (int i = 0; i < 24; ++i) EnqueueHead(2, d_last_hidden_.data());
    SyncStream();
    if (ReadCtl(5) != 0) CheckDeviceStatus("warmup", nullptr);
  }
  Reset();
}

// ------------------------------------------------------------------------------------------------
// Model / State
// ------------------------------------------------------------------------------------------------
Qwen38GraphModel::Qwen38GraphModel(std::unique_ptr<Config> config, OrtEnv& ort_env) : Model{std::move(config)} {
  const auto& q = config_->model.qwen38_graph;
  if (!q.IsConfigured() || q.ple_table.empty())
    throw std::runtime_error(
        "qwen38_flash_graph requires a folded/head configuration and model.qwen38_graph.ple_table");
  if (!p_device_ || p_device_->GetType() != DeviceType::CUDA)
    throw std::runtime_error("qwen38_flash_graph requires the CUDA execution provider");
  if (q.fold.enabled) {
    if (!q.mtp.enabled) throw std::runtime_error("model.qwen38_graph.fold requires model.qwen38_graph.mtp.enabled");
    if (q.fold.head_r2.filename.empty() && config_->model.dflash2.filename.empty())
      throw std::runtime_error("model.qwen38_graph.fold.head_r2.filename is required");
  } else if (q.mtp.enabled && q.head.filename_s2.empty()) {
    throw std::runtime_error("model.qwen38_graph.head.filename_s2 is required when MTP is enabled");
  }
  if (q.prefill.unified && !q.prefill.enabled)
    throw std::runtime_error("model.qwen38_graph.prefill.unified requires prefill.enabled");
  if (q.prefill.enabled && (!q.fold.enabled || (!q.prefill.unified &&
                                                (q.prefill.dense.filename.empty() || q.prefill.head_kv.filename.empty() ||
                                                 q.prefill.constants.empty()))))
    throw std::runtime_error(
        "model.qwen38_graph.prefill requires fold.enabled and prefill.{dense.filename,head_kv.filename,constants}");

  // Token selection is done by the engine; the search only does bookkeeping (EOS, max length), on the CPU so that
  // committing a token never synchronises the CUDA stream the look-ahead cycle runs on.
  p_device_scoring_ = GetDeviceInterface(DeviceType::CPU);

  const auto start = Clock::now();
  const std::filesystem::path root = std::filesystem::weakly_canonical(std::filesystem::path(config_->config_path.string()));
  plane_ = std::make_unique<Qwen38Plane>(GetOrtGlobals()->LoadQwen38PlaneFunctions());
  plane_->Check(plane_->set_device(0), "cudaSetDevice");

  // Device copy of the initializers several sessions have in common (optional)
  if (!q.prefill.weights_pool.empty()) {
    pool_ = std::make_unique<Qwen38WeightPool>(*plane_, (root / q.prefill.weights_pool).string(), root.string(),
                                               *p_device_->GetMemoryInfo());
    if (Verbose()) std::cerr << "[qwen38_graph] weight pool " << SecondsSince(start) << " s" << std::endl;
    pool_->AddTo(*session_options_, "target");
  }

  session_target_ = CreateSession(ort_env, config_->model.decoder.filename, session_options_.get());
  session_info_.Add(*session_target_);
  if (Verbose()) std::cerr << "[qwen38_graph] target session " << SecondsSince(start) << " s" << std::endl;

  auto make_options = [&](const std::optional<Config::SessionOptions>& configured, bool inherit,
                          const char* label, bool disable_graph_capture = false) {
    Config::SessionOptions options = configured.value_or(Config::SessionOptions{});
    if (inherit) InheritSessionOptionsFromParent(config_->model.decoder.session_options, options);
    auto result = OrtSessionOptions::Create();
    CreateSessionOptionsFromConfig(options, *result, /*is_primary_session_options=*/false,
                                   disable_graph_capture, /*cloned_from_parent=*/false, /*append_providers=*/true);
    if (pool_) pool_->AddTo(*result, label);
    return result;
  };

  if (q.fold.enabled && config_->model.dflash2.filename.empty()) {
    head_r2_session_options_ = make_options(q.fold.head_r2.session_options, true, "head_r2");
    session_head_r2_ = CreateSession(ort_env, q.fold.head_r2.filename, head_r2_session_options_.get());
    if (q.prefill.enabled && !q.prefill.unified) {
      // the dense / head-KV sessions are not graph captured and do not inherit the decoder's provider options
      dense_session_options_ = make_options(q.prefill.dense.session_options, false, "dense");
      head_kv_session_options_ = make_options(q.prefill.head_kv.session_options, false, "head_kv");
      session_dense_ = CreateSession(ort_env, q.prefill.dense.filename, dense_session_options_.get());
      session_head_kv_ = CreateSession(ort_env, q.prefill.head_kv.filename, head_kv_session_options_.get());
    } else if (q.prefill.enabled && q.prefill.unified && !q.prefill.head_kv.filename.empty()) {
      if (!session_info_.HasInput("prefill_numerics"))
        throw std::runtime_error("qwen38_graph: eager numerical head requires explicit prefill_numerics");
      head_kv_session_options_ = make_options(q.prefill.head_kv.session_options, false, "head_kv", true);
      session_head_kv_ = CreateSession(ort_env, q.prefill.head_kv.filename, head_kv_session_options_.get());
    }
    if (Verbose()) std::cerr << "[qwen38_graph] head / prefill sessions " << SecondsSince(start) << " s" << std::endl;
  } else if (q.mtp.enabled && !q.fold.enabled) {
    head_session_options_ = make_options(q.head.session_options, true, "head");
    session_head_s1_ = CreateSession(ort_env, q.head.filename_s1, head_session_options_.get());
    session_head_s2_ = CreateSession(ort_env, q.head.filename_s2, head_session_options_.get());
    if (Verbose()) std::cerr << "[qwen38_graph] head sessions " << SecondsSince(start) << " s" << std::endl;
  }

  if (q.prefill.enabled && (!q.prefill.unified || session_head_kv_)) {
    void* stream = p_device_->GetCudaStream();
    qsa_constants_ = std::make_unique<Qwen38QsaConstants>(*plane_, (root / q.prefill.constants).string(), root.string(), stream);
  }
  engine_ = std::make_unique<Qwen38GraphEngine>(*this);
}

Qwen38GraphModel::~Qwen38GraphModel() = default;

std::unique_ptr<State> Qwen38GraphModel::CreateState(DeviceSpan<int32_t> /*sequence_lengths*/,
                                                     const GeneratorParams& params) const {
  return std::make_unique<Qwen38GraphState>(*this, params);
}

Qwen38GraphState::Qwen38GraphState(const Qwen38GraphModel& model, const GeneratorParams& params)
    : State{params, model}, model_{model} {
  const auto& search = params.search;
  if (search.batch_size != 1 || search.num_beams != 1)
    throw std::runtime_error("qwen38_flash_graph supports batch_size 1 and num_beams 1 only");
  if (search.do_sample && search.top_k != 1 && search.temperature != 0.0f)
    throw std::runtime_error(
        "qwen38_flash_graph decodes greedily (the MTP loop is lossless only for argmax): set do_sample=false");
  if (search.repetition_penalty != 1.0f || search.no_repeat_ngram_size != 0 || search.min_length != 0)
    throw std::runtime_error("qwen38_flash_graph does not support repetition_penalty, no_repeat_ngram_size or min_length");
  if (!params.guidance_type.empty()) throw std::runtime_error("qwen38_flash_graph does not support constrained decoding");
  if (search.max_length > model.engine_->Capacity() - 4)
    throw std::runtime_error("max_length exceeds the KV capacity of the qwen38_flash_graph model (" +
                             std::to_string(model.engine_->Capacity() - 4) + ")");
  engine_ = model.engine_.get();
  if (!engine_->TryAcquire())
    throw std::runtime_error(
        "qwen38_flash_graph: the model's CUDA-graph state is already leased by another Engine or Generator; "
        "destroy it first (one active decoder per model)");
  try {
    engine_->Reset();
  } catch (...) {
    engine_->Release();
    throw;
  }
}

Qwen38GraphState::~Qwen38GraphState() {
  if (engine_) {
    try {
      engine_->Drain();
    } catch (...) {
    }
    engine_->Release();
  }
}

DeviceSpan<float> Qwen38GraphState::Run(int /*total_length*/, DeviceSpan<int32_t>& next_tokens,
                                        DeviceSpan<int32_t> /*next_indices*/) {
  if (engine_->DecodeStarted())
    throw std::runtime_error(
        "qwen38_flash_graph does not support continuous decoding (AppendTokens after GenerateNextToken). "
        "Create a new Generator or call RewindToLength(0).");
  auto tokens = next_tokens.CpuSpan();
  engine_->PrefillChunk(std::span<const int32_t>(tokens.data(), tokens.size()));
  auto logits = params_->p_device->Allocate<float>(static_cast<size_t>(model_.config_->model.vocab_size));
  engine_->LastLogits(logits.CpuSpan());
  logits.CopyCpuToDevice();
  return logits;
}

void Qwen38GraphState::Finalize(int /*current_length*/) {
  engine_->Drain();
  engine_->CheckErrorBits();
}

void Qwen38GraphState::RewindTo(size_t index) {
  if (index != 0)
    throw std::runtime_error("qwen38_flash_graph supports RewindToLength(0) only (recurrent state cannot be cropped)");
  engine_->Drain();
  engine_->Reset();
}

void Qwen38GraphState::SnapshotState(size_t /*position*/) {
  throw std::runtime_error("qwen38_flash_graph does not support state snapshots");
}

}  // namespace Generators
#endif  // defined(__linux__)
