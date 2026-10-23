// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Qwen3.8-Flash-Next NVFP4 (model.type "qwen38_flash_graph"), fast prefill: shared weight pool, dense chunked prefill,
// dense -> packed state hand-off, batched KV-only head prefill. Port of CG/mtp_runner/prefill (engine.py, handoff.py,
// head_kv.py, shared_weights.py); see tools/qwen38_graph/README.md.
#if defined(__linux__)
#include "generator/generators.h"
#include "qwen38_graph.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <charconv>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <random>
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

float Bf16ToFloat(uint16_t bits) {
  const uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

std::string JoinPath(const std::string& root, const std::string& relative) {
  return !relative.empty() && relative[0] == '/' ? relative : root + "/" + relative;
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// Shared weight pool
// ------------------------------------------------------------------------------------------------
Qwen38WeightPool::Qwen38WeightPool(Qwen38Plane& plane, const std::string& manifest_path, const std::string& root,
                                   const OrtMemoryInfo& memory_info, int threads)
    : plane_{plane} {
  std::ifstream in(manifest_path);
  if (!in) throw std::runtime_error("qwen38_graph: cannot open weight pool manifest " + manifest_path);
  std::string line;
  std::getline(in, line);
  if (line != "qwen38_weights_pool 1")
    throw std::runtime_error("qwen38_graph: unsupported weight pool manifest header: " + line);
  while (std::getline(in, line)) {
    std::istringstream fields(line);
    std::string key;
    fields >> key;
    if (key == "pool") {
      size_t id, rank;
      Entry entry;
      fields >> id >> entry.dtype >> entry.offset >> entry.length >> rank;
      entry.shape.resize(rank);
      for (auto& d : entry.shape) fields >> d;
      fields >> entry.file;
      if (id != entries_.size()) throw std::runtime_error("qwen38_graph: weight pool ids must be consecutive");
      if (entry.file.find("..") != std::string::npos)
        throw std::runtime_error("qwen38_graph: weight pool file must not contain '..': " + entry.file);
      entry.file = JoinPath(root, entry.file);
      entries_.push_back(std::move(entry));
    } else if (key == "session") {
      std::string label, name;
      int id;
      fields >> label >> name >> id;
      if (id < 0 || static_cast<size_t>(id) >= entries_.size())
        throw std::runtime_error("qwen38_graph: weight pool session entry references an unknown pool id");
      sessions_[label].emplace_back(name, id);
    }
  }
  const auto start = Clock::now();
  plane_.Check(plane_.set_device(0), "cudaSetDevice");
  for (auto& entry : entries_) {
    void* p = nullptr;
    plane_.Check(plane_.malloc_(&p, std::max<uint64_t>(entry.length, 16)), "cudaMalloc(weight pool)");
    entry.buffer = {static_cast<uint8_t*>(p), plane_.free_};
    bytes += entry.length;
    entry.value = OrtValue::CreateTensor(memory_info, entry.buffer.get(), static_cast<size_t>(entry.length), entry.shape,
                                         static_cast<ONNXTensorElementDataType>(entry.dtype));
  }

  // Copy the files in parallel: pinned staging buffer per worker, pread from the page cache, async H2D on a private stream.
  constexpr size_t kStage = size_t{64} << 20;
  std::atomic<size_t> next{0};
  std::mutex error_mutex;
  std::string error;
  auto worker = [&]() {
    void* stage = nullptr;
    void* stream = nullptr;
    std::map<std::string, int> fds;
    try {
      plane_.Check(plane_.set_device(0), "cudaSetDevice");
      plane_.Check(plane_.host_alloc(&stage, kStage), "cudaHostAlloc(pool stage)");
      plane_.Check(plane_.stream_create(&stream), "cudaStreamCreate");
      for (size_t i = next.fetch_add(1); i < entries_.size(); i = next.fetch_add(1)) {
        Entry& entry = entries_[i];
        auto [it, inserted] = fds.emplace(entry.file, -1);
        int& fd = it->second;
        if (inserted) fd = open(entry.file.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("qwen38_graph: cannot open weight file " + entry.file);
        for (uint64_t done = 0; done < entry.length;) {
          const size_t n = static_cast<size_t>(std::min<uint64_t>(kStage, entry.length - done));
          size_t got = 0;
          while (got < n) {
            const ssize_t r = pread(fd, static_cast<uint8_t*>(stage) + got, n - got, static_cast<off_t>(entry.offset + done + got));
            if (r <= 0) throw std::runtime_error("qwen38_graph: short read of weight file " + entry.file);
            got += static_cast<size_t>(r);
          }
          plane_.Check(plane_.memcpy_h2d(entry.buffer.get() + done, stage, n, stream), "weight upload");
          plane_.Check(plane_.stream_sync(stream), "weight upload sync");
          done += n;
        }
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (error.empty()) error = e.what();
      next.store(entries_.size());
    }
    for (auto& f : fds)
      if (f.second >= 0) close(f.second);
    if (stream) plane_.stream_destroy(stream);
    if (stage) plane_.host_free(stage);
  };
  if (!plane_.stream_create) throw std::runtime_error("qwen38_graph: the plane library lacks the weight pool helpers");
  if (threads <= 0) throw std::runtime_error("qwen38_graph: weight upload requires at least one worker");
  std::vector<std::thread> pool;
  pool.reserve(static_cast<size_t>(threads));
  try {
    for (int t = 0; t < threads; ++t) pool.emplace_back(worker);
  } catch (...) {
    next.store(entries_.size());
    for (auto& t : pool) t.join();
    throw;
  }
  for (auto& t : pool) t.join();
  if (!error.empty()) throw std::runtime_error(error);
  seconds = SecondsSince(start);
  if (Verbose())
    std::cerr << "[qwen38_graph] weight pool: " << entries_.size() << " initializers, " << bytes / double(1 << 30)
              << " GiB in " << seconds << " s" << std::endl;
}

Qwen38WeightPool::~Qwen38WeightPool() = default;

size_t Qwen38WeightPool::AddTo(OrtSessionOptions& options, const std::string& label) const {
  auto it = sessions_.find(label);
  if (it == sessions_.end()) return 0;
  for (const auto& [name, id] : it->second) options.AddInitializer(name.c_str(), *entries_[id].value);
  return it->second.size();
}

// ------------------------------------------------------------------------------------------------
// QSA constants
// ------------------------------------------------------------------------------------------------
Qwen38QsaConstants::Qwen38QsaConstants(Qwen38Plane& plane, const std::string& index_path, const std::string& root,
                                       void* stream) {
  std::ifstream in(index_path);
  if (!in) throw std::runtime_error("qwen38_graph: cannot open QSA constants index " + index_path);
  std::string line;
  std::getline(in, line);
  if (line != "qwen38_qsa_consts 1") throw std::runtime_error("qwen38_graph: unsupported QSA constants header: " + line);
  struct Item {
    std::string set, name;
    size_t offset, bytes;
  };
  std::vector<Item> items;
  std::string file;
  while (std::getline(in, line)) {
    std::istringstream fields(line);
    std::string key;
    fields >> key;
    if (key == "file")
      fields >> file;
    else if (key == "epsilon")
      fields >> epsilon;
    else if (key == "width")
      fields >> target.width, head.width = target.width;
    else if (key == "tensor") {
      Item item{};
      if (!(fields >> item.set >> item.name >> item.offset >> item.bytes))
        throw std::runtime_error("qwen38_graph: malformed QSA constant in " + index_path);
      items.push_back(item);
    }
  }
  std::ifstream data(JoinPath(root, file), std::ios::binary | std::ios::ate);
  if (!data) throw std::runtime_error("qwen38_graph: cannot open QSA constants data " + file);
  const size_t total = static_cast<size_t>(data.tellg());
  std::vector<uint8_t> host(total);
  data.seekg(0);
  data.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(total));
  if (!data) throw std::runtime_error("qwen38_graph: short read of QSA constants data " + file);
  for (const Item& item : items) {
    if (item.bytes == 0 || item.offset > total || item.bytes > total - item.offset)
      throw std::runtime_error("qwen38_graph: QSA constant range exceeds data blob: " + item.set + "." + item.name);
  }
  buffer_ = PlaneBuffer(plane, total, stream);
  plane.Check(plane.memcpy_h2d(buffer_.data(), host.data(), total, stream), "QSA constants upload");
  plane.Check(plane.stream_sync(stream), "QSA constants sync");
  for (const Item& item : items) {
    Set& set = item.set == "head" ? head : target;
    const uint8_t* ptr = buffer_.data() + item.offset;
    if (item.name == "cos")
      set.cos = ptr;
    else if (item.name == "sin")
      set.sin = ptr;
    else if (item.name.rfind("knorm.", 0) == 0)
      set.knorm[std::stoi(item.name.substr(6))] = ptr;
  }
  if (!target.cos || !target.sin || !head.cos || !head.sin || target.knorm.empty() || head.knorm.empty())
    throw std::runtime_error("qwen38_graph: incomplete QSA constants " + index_path);
}

// ------------------------------------------------------------------------------------------------
// Fast prefill state
// ------------------------------------------------------------------------------------------------

namespace {

constexpr int kRatio = 4;  // PackedQSAIndexer pooling ratio

int LayerOf(const std::string& name) {
  const size_t first = name.find('.');
  const size_t second = name.find('.', first + 1);
  return std::stoi(name.substr(first + 1, second - first - 1));
}

std::string TailOf(const std::string& name) {
  const size_t first = name.find('.');
  const size_t second = name.find('.', first + 1);
  return name.substr(second + 1);
}

}  // namespace

void Qwen38GraphEngine::BuildFastPrefill() {
  const auto& q = model_.config_->model.qwen38_graph;
  if (q.prefill.unified) {
    const bool dflash = !model_.config_->model.dflash2.filename.empty();
    if (!plane_.ple_rows || !model_.session_target_ || (!dflash && !model_.session_head_r2_))
      throw std::runtime_error("qwen38_graph: unified prefill requires folded target/head sessions and ple_rows");
    auto fast = std::make_unique<FastPrefillState>();
    fast->unified = true;
    fast->dense = model_.session_target_.get();
    fast->head_kv = model_.session_head_kv_ ? model_.session_head_kv_.get() : model_.session_head_r2_.get();
    fast->consts = model_.qsa_constants_.get();
    fast->dense_info.Add(*fast->dense);
    if (fast->head_kv) fast->head_kv_info.Add(*fast->head_kv);
    fast->dense_inputs = fast->dense->GetInputNames();
    fast->dense_outputs = fast->dense->GetOutputNames();
    fast->chunk = q.prefill.chunk;
    if (fast->chunk < 1 || fast->chunk > capacity_)
      throw std::runtime_error("qwen38_graph: invalid unified prefill.chunk");
    const size_t slots = static_cast<size_t>(capacity_) + 8;
    if (model_.session_head_kv_)
      fast->raw_keys = PlaneBuffer(plane_, slots * 128 * 2, stream_);
    std::vector<int64_t> ones(slots, 1);
    fast->ones = PlaneBuffer(plane_, slots * 8, stream_);
    plane_.Check(plane_.memcpy_h2d(fast->ones.data(), ones.data(), slots * 8, stream_), "unified attention mask");
    fast->idx = PlaneBuffer(plane_, 64, stream_);
    fast->ples = PlaneBuffer(plane_, static_cast<size_t>(fast->chunk) * ple_width_ * 2, stream_);
    fast->scratch = PlaneBuffer(plane_, std::max<size_t>(4096, 64 + status_count_ * 4), stream_);
    if (d_prompt_hidden_.size() == 0)
      d_prompt_hidden_ = PlaneBuffer(plane_, (dflash ? size_t(fast->chunk) : slots) * hidden_width_ * 2, stream_);
    SyncStream();
    fast->run_options = OrtRunOptions::Create();
    fast->run_options->AddConfigEntry("gpu_graph_id", "-1");
    fast_ = std::move(fast);
    return;
  }
  if (!plane_.ple_rows || !plane_.pool_keys || !plane_.transpose_f32 || !plane_.memcpy2d_d2d)
    throw std::runtime_error("qwen38_graph: the plane library lacks the fast prefill kernels (rebuild qwen38_plane)");
  if (!model_.session_dense_ || !model_.session_head_kv_ || !model_.qsa_constants_)
    throw std::runtime_error("qwen38_graph: prefill.enabled requires the dense, head_kv sessions and the QSA constants");
  auto fast = std::make_unique<FastPrefillState>();
  fast->dense = model_.session_dense_.get();
  fast->head_kv = model_.session_head_kv_.get();
  fast->dense_info.Add(*fast->dense);
  fast->head_kv_info.Add(*fast->head_kv);
  fast->chunk = q.prefill.chunk;
  fast->consts = model_.qsa_constants_.get();
  if (fast->chunk < 1 || fast->chunk > capacity_) throw std::runtime_error("qwen38_graph: invalid prefill.chunk");
  fast->dense_inputs = fast->dense->GetInputNames();
  fast->dense_outputs = fast->dense->GetOutputNames();

  const size_t slots = static_cast<size_t>(capacity_) + 8;
  {
    std::vector<int64_t> ones(slots, 1);
    fast->ones = PlaneBuffer(plane_, slots * 8, stream_);
    plane_.Check(plane_.memcpy_h2d(fast->ones.data(), ones.data(), slots * 8, stream_), "ones");
    SyncStream();
  }
  fast->idx = PlaneBuffer(plane_, 8, stream_);
  fast->ples = PlaneBuffer(plane_, static_cast<size_t>(fast->chunk) * ple_width_ * 2, stream_);
  fast->raw_keys = PlaneBuffer(plane_, slots * 128 * 2, stream_);
  fast->scratch = PlaneBuffer(plane_, 256, stream_);
  if (d_prompt_hidden_.size() == 0) d_prompt_hidden_ = PlaneBuffer(plane_, slots * hidden_width_ * 2, stream_);

  // Initial dense state: zero-length KV / index keys / positions, zero recurrent and conv states.
  const auto& info = fast->dense_info;
  for (const auto& name : fast->dense_inputs) {
    if (name != "past_position_ids" && name.rfind("past_key_values.", 0) != 0) continue;
    std::vector<int64_t> shape = info.GetInputShape(name);
    const std::string tail = name == "past_position_ids" ? "positions" : TailOf(name);
    for (int64_t& d : shape)
      if (d <= 0) d = 1;
    if (tail == "positions")
      shape = {1, 0};
    else if (tail == "key" || tail == "value")
      shape[2] = 0;
    else if (tail == "index_key")
      shape[1] = 0;
    const auto type = info.GetInputDataType(name);
    size_t elements = 1;
    for (int64_t d : shape) elements *= static_cast<size_t>(d);
    const size_t bytes = elements * Ort::SizeOf(type);
    fast->zero_buffers.emplace_back(plane_, std::max<size_t>(bytes, 16), stream_);
    fast->initial[name] = OrtValue::CreateTensor(*memory_info_, fast->zero_buffers.back().data(), bytes, shape, type);
  }
  SyncStream();
  fast->run_options = OrtRunOptions::Create();
  fast_ = std::move(fast);
}

void Qwen38GraphEngine::FastReset() {
  if (!fast_) return;
  fast_->states.clear();
  fast_->logits.reset();
  fast_->dense_length = 0;
  fast_->timings = {};
}

// One call = whole chunk loop over `tokens`, continuing the dense state of earlier calls (chunked AppendTokens).
void Qwen38GraphEngine::FastAppend(std::span<const int32_t> tokens) {
  FastPrefillState& f = *fast_;
  if (f.unified) {
    FastUnifiedAppend(tokens);
    return;
  }
  if (decode_started_) throw std::runtime_error("qwen38_graph: AppendTokens after generation started is not supported");
  const size_t n = tokens.size();
  if (n == 0) throw std::runtime_error("qwen38_graph: empty prefill chunk");
  if (length_ + n + 3 > static_cast<size_t>(capacity_))
    throw std::runtime_error("qwen38_graph: prompt exceeds the KV capacity of " + std::to_string(capacity_));
  const auto start = Clock::now();
  const size_t base = length_;
  {
    std::vector<int64_t> chunk(n);
    for (size_t i = 0; i < n; ++i) chunk[i] = tokens[i];
    plane_.Check(plane_.memcpy_h2d(d_tokens_.data() + 8 * base, chunk.data(), n * 8, stream_), "upload prompt");
    SyncStream();  // the host vector dies here
  }
  prompt_tokens_.insert(prompt_tokens_.end(), tokens.begin(), tokens.end());
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  const auto i64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  const auto bf16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  const std::string ple_name = "ple_embeddings." + std::to_string(ple_layer_);

  for (size_t lo = 0; lo < n; lo += static_cast<size_t>(f.chunk)) {
    const size_t m = std::min(static_cast<size_t>(f.chunk), n - lo);
    const size_t pos = base + lo;  // == f.dense_length
    plane_.ple_rows(stream_, params_dev_.data(), d_tokens_.data() + 8 * pos, static_cast<int>(m), ctx_commit_.data(),
                    ctx_b_.data(), f.ples.data());
    plane_.Check(plane_.memcpy_d2d(ctx_commit_.data(), ctx_b_.data(), 16, stream_), "ctx commit");
    const int64_t last_row = static_cast<int64_t>(m) - 1;
    plane_.Check(plane_.memcpy_h2d(f.idx.data(), &last_row, 8, stream_), "logits_indices");

    auto binding = OrtIoBinding::Create(*f.dense);
    std::vector<std::unique_ptr<OrtValue>> views;
    auto view = [&](void* ptr, std::vector<int64_t> shape, ONNXTensorElementDataType type) -> OrtValue& {
      size_t count = 1;
      for (int64_t d : shape) count *= static_cast<size_t>(d);
      views.push_back(OrtValue::CreateTensor(*memory_info_, ptr, count * Ort::SizeOf(type), shape, type));
      return *views.back();
    };
    for (const auto& name : f.dense_inputs) {
      if (name == "input_ids")
        binding->BindInput(name.c_str(), view(d_tokens_.data() + 8 * pos, {1, int64_t(m)}, i64));
      else if (name == "attention_mask")
        binding->BindInput(name.c_str(), view(f.ones.data(), {1, int64_t(pos + m)}, i64));
      else if (name == "position_ids")
        binding->BindInput(name.c_str(), view(d_arange64_.data() + 8 * pos, {1, int64_t(m)}, i64));
      else if (name == "logits_indices")
        binding->BindInput(name.c_str(), view(f.idx.data(), {1}, i64));
      else if (name == ple_name)
        binding->BindInput(name.c_str(), view(f.ples.data(), {1, int64_t(m), ple_width_}, bf16));
      else {
        auto it = f.states.find(name);
        OrtValue* value = it != f.states.end() ? it->second.get() : f.initial.at(name).get();
        binding->BindInput(name.c_str(), *value);
      }
    }
    for (const auto& name : f.dense_outputs) {
      if (name == "hidden_states")
        binding->BindOutput(name.c_str(), view(d_prompt_hidden_.data() + pos * row_bytes, {1, int64_t(m), hidden_width_}, bf16));
      else
        binding->BindOutput(name.c_str(), *memory_info_);
    }
    f.dense->Run(f.run_options.get(), *binding);
    SyncStream();

    const auto names = binding->GetOutputNames();
    auto values = binding->GetOutputValues();
    std::map<std::string, std::unique_ptr<OrtValue>> next;
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == "present_position_ids")
        next["past_position_ids"] = std::move(values[i]);
      else if (names[i].rfind("present.", 0) == 0)
        next["past_key_values." + names[i].substr(8)] = std::move(values[i]);
      else if (names[i] == "logits")
        f.logits = std::move(values[i]);
    }
    f.states = std::move(next);
    f.dense_length += m;
  }
  length_ += n;

  // Last prompt row: logits (host argmax, smallest index on ties like the device argmax), last hidden row.
  const auto shape_info = f.logits->GetTensorTypeAndShapeInfo();
  const auto shape = shape_info->GetShape();
  size_t rows = 1;
  for (size_t d = 0; d + 1 < shape.size(); ++d) rows *= static_cast<size_t>(shape[d]);
  if (shape.back() != vocab_) throw std::runtime_error("qwen38_graph: unexpected dense logits shape");
  std::vector<uint16_t> raw(static_cast<size_t>(vocab_));
  plane_.Check(plane_.memcpy_d2h(raw.data(), static_cast<uint8_t*>(f.logits->GetTensorMutableRawData()) + (rows - 1) * static_cast<size_t>(vocab_) * 2,
                                 raw.size() * 2, stream_),
               "read logits");
  plane_.Check(plane_.memcpy_d2d(d_last_hidden_.data(), d_prompt_hidden_.data() + (length_ - 1) * row_bytes, row_bytes, stream_),
               "last hidden");
  SyncStream();
  last_logits_host_.resize(raw.size());
  int best = 0;
  float best_value = -INFINITY;
  for (size_t i = 0; i < raw.size(); ++i) {
    const float v = Bf16ToFloat(raw[i]);
    last_logits_host_[i] = v;
    if (v > best_value) {
      best_value = v;
      best = static_cast<int>(i);
    }
  }
  pending_ = best;
  last_logits_ptr_ = nullptr;
  f.timings.dense += SecondsSince(start);
}

// Dense state (KV, positions, GDN / conv / PLE state, raw index keys) -> packed decode state: KV rows into the fixed store,
// GDN state transposed into ring bank 0, PLE / conv state copied, raw index keys pooled into the shared compressed keys
// (+ pending rows and lengths). Mirrors handoff.py (HandoffWriter.write).
const OrtValue& Qwen38GraphEngine::PrefillNumerics(bool prefill) {
  const size_t index = prefill ? 1 : 0;
  if (!prefill_numerics_values_[index]) {
    auto cpu = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    prefill_numerics_values_[index] = OrtValue::CreateTensor(
        *cpu, &prefill_numerics_constants_[index], sizeof(int32_t), std::vector<int64_t>{1},
        ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
  }
  return *prefill_numerics_values_[index];
}

const OrtValue& Qwen38GraphEngine::NumericalCache(
    const std::string& name, const SessionInfo& info, bool prefill, bool head) {
  const std::string key = (head ? "head/" : "target/") + name;
  auto& tensors = prefill ? fast_->numerical_cache : numerical_decode_cache_;
  if (auto found = tensors.find(key); found != tensors.end()) return *found->second->value;
  auto shape = info.GetInputShape(name);
  const auto type = info.GetInputDataType(name);
  const bool scoring = shape.size() == 3 && shape[2] == 128 &&
                       type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  if (!scoring && (shape.size() != 4 || shape[1] <= 0 || shape[3] <= 0 ||
                   type != ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16))
    throw std::runtime_error("qwen38_graph: invalid numerical shadow cache " + name);
  shape[0] = 1;
  if (scoring) {
    if (!name.ends_with(".scoring"))
      throw std::runtime_error("qwen38_graph: invalid numerical scoring state " + name);
    const std::string source = name.substr(14, name.size() - 14 - 8);
    const auto original_shape = info.GetInputShape(source);
    if (original_shape.size() != 3 || original_shape[1] <= 0 || original_shape[2] != 128)
      throw std::runtime_error("qwen38_graph: invalid numerical scoring state source " + source);
    shape[1] = prefill ? original_shape[1] : 0;
  } else
    shape[2] = prefill ? capacity_ : 0;
  uint8_t* ptr = nullptr;
  if (prefill) {
    size_t elements = 1;
    for (auto dim : shape) elements *= static_cast<size_t>(dim);
    const size_t bytes = elements * Ort::SizeOf(type);
    fast_->numerical_buffers.emplace_back(plane_, bytes, stream_);
    ptr = fast_->numerical_buffers.back().data();
    plane_.Check(plane_.memset_(ptr, 0, bytes, stream_), "initialize dense numerical KV");
  }
  tensors[key] = MakeTensor(ptr, std::move(shape), type);
  return *tensors.at(key)->value;
}

void Qwen38GraphEngine::FastUnifiedAppend(std::span<const int32_t> tokens) {
  if (dflash_ && dflash_prompt_rows_) {
    Draft(d_dflash_prompt_.data(), dflash_prompt_rows_, dflash_prompt_position_, 0, false);
    dflash_prompt_rows_ = 0;
  }
  auto& f = *fast_;
  if (decode_started_) throw std::runtime_error("qwen38_graph: AppendTokens after generation started is not supported");
  if (tokens.empty()) throw std::runtime_error("qwen38_graph: empty prefill chunk");
  if (length_ + tokens.size() + 3 > static_cast<size_t>(capacity_))
    throw std::runtime_error("qwen38_graph: prompt exceeds the KV capacity");
  const auto start = Clock::now();
  const size_t base = length_;
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  const auto i64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  const auto i32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  const auto bf16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  const std::string ple_name = "ple_embeddings." + std::to_string(ple_layer_);
  std::vector<int64_t> ids(tokens.begin(), tokens.end());
  plane_.Check(plane_.memcpy_h2d(d_tokens_.data() + base * 8, ids.data(), ids.size() * 8, stream_), "upload prompt");
  prompt_tokens_.insert(prompt_tokens_.end(), tokens.begin(), tokens.end());
  MutedPrefillBegin();
  try {
    for (size_t lo = 0; lo < tokens.size(); lo += static_cast<size_t>(f.chunk)) {
      const size_t m = std::min(static_cast<size_t>(f.chunk), tokens.size() - lo);
      const size_t pos = base + lo;
      const int32_t meta[] = {static_cast<int32_t>(pos), static_cast<int32_t>(m - 1), 0, static_cast<int32_t>(m)};
      plane_.Check(plane_.memcpy_h2d(f.idx.data(), meta, sizeof(meta), stream_), "unified prefill metadata");
      plane_.ple_rows(stream_, params_dev_.data(), d_tokens_.data() + pos * 8, static_cast<int>(m), ctx_commit_.data(),
                      ctx_b_.data(), f.ples.data());
      plane_.Check(plane_.memcpy_d2d(ctx_commit_.data(), ctx_b_.data(), 16, stream_), "commit PLE context");
      plane_.Check(plane_.memcpy_d2d(past_pos_ + pos * 8, d_arange64_.data() + pos * 8, m * 8, stream_), "prompt positions");
      // The folded head executes only its two-row tail while muted; do not expose stale index lengths to it.
      if (auto it = head_banks_[0].tensors.find("past_key_values.0.index_lengths"); it != head_banks_[0].tensors.end())
        plane_.Check(plane_.memset_(it->second->ptr, 0, it->second->Bytes(), stream_), "clear muted head index lengths");
      SyncStream();
      auto binding = OrtIoBinding::Create(*f.dense);
      std::vector<std::unique_ptr<OrtValue>> views;
      auto view = [&](void* ptr, std::vector<int64_t> shape, ONNXTensorElementDataType type) -> OrtValue& {
        size_t count = 1;
        for (int64_t d : shape) count *= static_cast<size_t>(d);
        views.push_back(OrtValue::CreateTensor(*memory_info_, ptr, count * Ort::SizeOf(type), shape, type));
        return *views.back();
      };
      // The target mixer sees the unpadded Slice tail. FoldMid2 independently pads
      // its head hidden bridge; padding logits_indices would read beyond an S=1 tail.
      const size_t logit_rows = std::min(m, static_cast<size_t>(verify_rows_));
      for (const auto& name : f.dense_inputs) {
        if (name == "prefill_numerics")
          binding->BindInput(name.c_str(), PrefillNumerics(true));
        else if (name.rfind("prefill_cache.", 0) == 0)
          binding->BindInput(name.c_str(), NumericalCache(name, f.dense_info, true));
        else if (name == "input_ids")
          binding->BindInput(name.c_str(), view(d_tokens_.data() + pos * 8, {1, int64_t(m)}, i64));
        else if (name == "attention_mask")
          binding->BindInput(name.c_str(), view(f.ones.data(), {1, int64_t(m)}, i64));
        else if (name == "position_ids")
          binding->BindInput(name.c_str(), view(d_arange64_.data() + pos * 8, {1, int64_t(m)}, i64));
        else if (name == "logits_indices")
          binding->BindInput(name.c_str(), view(d_arange64_.data(), {int64_t(logit_rows)}, i64));
        else if (name == "committed_length")
          binding->BindInput(name.c_str(), view(f.idx.data(), {1}, i32));
        else if (name == "sequence_length_minus_one")
          binding->BindInput(name.c_str(), view(f.idx.data() + 4, {1}, i32));
        else if (name == "cumulative_sequence_lengths")
          binding->BindInput(name.c_str(), view(f.idx.data() + 8, {2}, i32));
        else if (name == ple_name)
          binding->BindInput(name.c_str(), view(f.ples.data(), {1, int64_t(m), ple_width_}, bf16));
        else if (fold_tensors_.count(name))
          binding->BindInput(name.c_str(), *fold_tensors_.at(name)->value);
        else if (name.rfind("head.past_key_values.0.", 0) == 0) {
          const std::string state = name.substr(5);
          binding->BindInput(name.c_str(), head_store_.count(state) ? *head_store_.at(state)->value
                                                                    : *head_banks_[0].tensors.at(state)->value);
        } else if (store_.count(name))
          binding->BindInput(name.c_str(), *store_.at(name)->value);
        else if (shared_.count(name))
          binding->BindInput(name.c_str(), *shared_.at(name)->value);
        else if (banks_[0].tensors.count(name))
          binding->BindInput(name.c_str(), *banks_[0].tensors.at(name)->value);
        else
          throw std::runtime_error("qwen38_graph: unsupported unified prefill input " + name);
      }
      for (const auto& name : f.dense_outputs) {
        if (name.rfind("prefill_cache_out.", 0) == 0)
          binding->BindOutput(name.c_str(), NumericalCache("prefill_cache." + name.substr(18), f.dense_info, true));
        else if (name == "hidden_states")
          binding->BindOutput(name.c_str(), view(d_prompt_hidden_.data() + (dflash_ ? 0 : pos * row_bytes),
                                                 {1, int64_t(m), hidden_width_}, bf16));
        else if (name == "aux_hidden_states") {
          if (!dflash_) throw std::runtime_error("Unexpected DFlash auxiliary prefill output");
          binding->BindOutput(name.c_str(), view(d_dflash_prompt_.data(), {1, int64_t(m), int64_t(kDflashWidth)}, bf16));
        } else if (name == "validation.aggregate")
          binding->BindOutput(name.c_str(), view(f.scratch.data() + 64, {int64_t(status_count_)}, i32));
        else if (name == "present_position_ids")
          binding->BindOutput(name.c_str(), *store_.at("past_position_ids")->value);
        else if (name.rfind("present_mid", 0) == 0) {
          const std::string state = "past_key_values." + name.substr(name.find('.') + 1);
          const size_t dot = name.find('.');
          const std::string suffix = name.substr(11, dot - 11);
          const int prefix = suffix.empty() ? 1 : std::stoi(suffix);
          if (prefix < 1 || prefix >= verify_rows_)
            throw std::runtime_error("qwen38_graph: invalid unified prefix output " + name);
          binding->BindOutput(name.c_str(), *banks_[prefix + 1].tensors.at(state)->value);
        } else if (name.rfind("present.", 0) == 0) {
          const std::string state = "past_key_values." + name.substr(8);
          binding->BindOutput(name.c_str(), store_.count(state)    ? *store_.at(state)->value
                                            : shared_.count(state) ? *shared_.at(state)->value
                                                                   : *banks_[1].tensors.at(state)->value);
        } else if (name.rfind("head.present", 0) == 0) {
          const size_t dot = name.find('.', 5);
          const std::string state = "past_key_values." + name.substr(dot + 1);
          int bank = 1;
          if (name.rfind("head.present_mid", 0) == 0) {
            const std::string suffix = name.substr(16, dot - 16);
            const int prefix = suffix.empty() ? 1 : std::stoi(suffix);
            if (prefix < 1 || prefix >= verify_rows_)
              throw std::runtime_error("qwen38_graph: invalid unified head prefix output " + name);
            bank = prefix + 1;
          }
          binding->BindOutput(name.c_str(), head_store_.count(state) ? *head_store_.at(state)->value
                                                                     : *head_banks_[bank].tensors.at(state)->value);
        } else
          binding->BindOutput(name.c_str(), *memory_info_);
      }
      f.dense->Run(f.run_options.get(), *binding);
      plane_.or_status(stream_, f.scratch.data() + 64, static_cast<int>(status_count_), ctl_.data());
      plane_.Check(plane_.memcpy_d2d(banks_[0].arena.data(), banks_[1].arena.data(), banks_[0].arena.size(), stream_),
                   "commit unified state bank");
      SyncStream();
      if (ReadCtl(5) != 0) CheckDeviceStatus("unified prefill", nullptr);
      const auto names = binding->GetOutputNames();
      auto values = binding->GetOutputValues();
      for (size_t i = 0; i < names.size(); ++i)
        if (names[i] == "logits") f.logits = std::move(values[i]);
      f.dense_length += m;
      if (dflash_) {
        if (lo + m < tokens.size()) {
          Draft(d_dflash_prompt_.data(), m, pos, 0, false);
        } else {
          dflash_prompt_rows_ = m;
          dflash_prompt_position_ = pos;
        }
      }
    }
    length_ += tokens.size();
    const size_t last_hidden_row = dflash_ ? (tokens.size() - 1) % size_t(f.chunk) : length_ - 1;
    plane_.Check(plane_.memcpy_d2d(d_last_hidden_.data(), d_prompt_hidden_.data() + last_hidden_row * row_bytes,
                                   row_bytes, stream_),
                 "last prompt hidden");
    if (!f.logits) throw std::runtime_error("qwen38_graph: unified target has no logits output");
    const auto info = f.logits->GetTensorTypeAndShapeInfo();
    const auto shape = info->GetShape();
    if (shape.empty() || shape.back() != vocab_) throw std::runtime_error("qwen38_graph: unexpected unified logits shape");
    const size_t elements = info->GetElementCount();
    if (dflash_ && plane_.argmax_finish) {
      last_logits_ptr_ = static_cast<uint8_t*>(f.logits->GetTensorMutableRawData());
      last_logits_row_ = static_cast<int>(elements / vocab_) - 1;
      last_logits_host_.clear();
      plane_.argmax_part(stream_, last_logits_ptr_ + size_t(last_logits_row_) * vocab_ * 2,
                         vocab_, part_v_.data(), part_i_.data(), kArgmaxBlocks);
      plane_.Check(plane_.argmax_finish(stream_, part_v_.data(), part_i_.data(), kArgmaxBlocks,
                                        scratch_i32_.data()),
                   "prefill argmax");
      int32_t prediction = -1;
      plane_.Check(plane_.memcpy_d2h(&prediction, scratch_i32_.data(), sizeof(prediction), stream_),
                   "read prefill prediction");
      SyncStream();
      pending_ = prediction;
    } else {
      std::vector<uint16_t> raw(static_cast<size_t>(vocab_));
      plane_.Check(plane_.memcpy_d2h(raw.data(), static_cast<uint8_t*>(f.logits->GetTensorMutableRawData()) + (elements - vocab_) * 2, raw.size() * 2, stream_), "read unified logits");
      SyncStream();
      last_logits_host_.resize(raw.size());
      for (size_t i = 0; i < raw.size(); ++i) last_logits_host_[i] = Bf16ToFloat(raw[i]);
      pending_ = static_cast<int32_t>(std::max_element(last_logits_host_.begin(), last_logits_host_.end()) - last_logits_host_.begin());
    }
  } catch (...) {
    MutedPrefillEnd();
    throw;
  }
  MutedPrefillEnd();
  f.timings.dense += SecondsSince(start);
}

void Qwen38GraphEngine::FastUnifiedHeadPrefill() {
  auto& f = *fast_;
  const auto start = Clock::now();
  const size_t rows = length_ > 0 ? length_ - 1 : 0;
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  const auto i64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  const auto i32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  const auto bf16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  head_cur_ = 0;
  for (size_t lo = 0; lo < rows; lo += static_cast<size_t>(f.chunk)) {
    const size_t m = std::min(static_cast<size_t>(f.chunk), rows - lo);
    const int32_t committed = static_cast<int32_t>(lo);
    plane_.Check(plane_.memcpy_h2d(f.idx.data(), &committed, sizeof(committed), stream_), "head committed length");
    SyncStream();
    auto binding = OrtIoBinding::Create(*f.head_kv);
    std::vector<std::unique_ptr<OrtValue>> views;
    auto view = [&](void* ptr, std::vector<int64_t> shape, ONNXTensorElementDataType type) -> OrtValue& {
      size_t count = 1;
      for (int64_t d : shape) count *= static_cast<size_t>(d);
      views.push_back(OrtValue::CreateTensor(*memory_info_, ptr, count * Ort::SizeOf(type), shape, type));
      return *views.back();
    };
    for (const auto& name : f.head_kv->GetInputNames()) {
      if (name == "prefill_numerics")
        binding->BindInput(name.c_str(), PrefillNumerics(true));
      else if (name.rfind("prefill_cache.", 0) == 0)
        binding->BindInput(name.c_str(), NumericalCache(name, f.head_kv_info, true, true));
      else if (name == "input_ids")
        binding->BindInput(name.c_str(), view(d_tokens_.data() + (lo + 1) * 8, {1, int64_t(m)}, i64));
      else if (name == "position_ids")
        binding->BindInput(name.c_str(), view(d_arange64_.data() + (lo + 1) * 8, {1, int64_t(m)}, i64));
      else if (name == "hidden_states")
        binding->BindInput(name.c_str(), view(d_prompt_hidden_.data() + lo * row_bytes, {1, int64_t(m), hidden_width_}, bf16));
      else if (name == "committed_length")
        binding->BindInput(name.c_str(), view(f.idx.data(), {1}, i32));
      else if (head_store_.count(name))
        binding->BindInput(name.c_str(), *head_store_.at(name)->value);
      else if (head_banks_[0].tensors.count(name))
        binding->BindInput(name.c_str(), *head_banks_[0].tensors.at(name)->value);
      else
        throw std::runtime_error("qwen38_graph: unsupported unified head input " + name);
    }
    for (const auto& name : f.head_kv->GetOutputNames()) {
      const std::string state = "past_key_values." + name.substr(name.find('.') + 1);
      if (name.rfind("prefill_cache_out.", 0) == 0)
        binding->BindOutput(name.c_str(), NumericalCache("prefill_cache." + name.substr(18), f.head_kv_info, true, true));
      else if (name == "validation.aggregate")
        binding->BindOutput(name.c_str(), view(f.scratch.data(), {int64_t(head_r2_.status_count)}, i32));
      else if (name.rfind("present", 0) == 0) {
        const int bank = name.rfind("present_mid", 0) == 0 ? 2 : 1;
        binding->BindOutput(name.c_str(), head_store_.count(state) ? *head_store_.at(state)->value
                                                                   : *head_banks_[bank].tensors.at(state)->value);
      } else
        binding->BindOutput(name.c_str(), *memory_info_);
    }
    f.head_kv->Run(f.run_options.get(), *binding);
    plane_.or_status(stream_, f.scratch.data(), static_cast<int>(head_r2_.status_count), ctl_.data());
    for (const auto& [name, tensor] : head_banks_[1].tensors)
      plane_.Check(plane_.memcpy_d2d(head_banks_[0].tensors.at(name)->ptr, tensor->ptr, tensor->Bytes(), stream_),
                   "commit unified head state");
    SyncStream();
    if (ReadCtl(5) != 0) CheckDeviceStatus("unified head prefill", nullptr);
  }
  head_length_ = static_cast<int>(rows);
  f.timings.head += SecondsSince(start);
}

void Qwen38GraphEngine::FastHandoff() {
  FastPrefillState& f = *fast_;
  if (f.unified) {
    if (f.dense_length != length_) throw std::runtime_error("qwen38_graph: unified state length mismatch");
    cur_ = 0;
    return;
  }
  const auto start = Clock::now();
  const size_t P = length_;
  if (f.dense_length != P) throw std::runtime_error("qwen38_graph: dense state length differs from the prompt length");
  auto dense_ptr = [&](const std::string& name) -> uint8_t* {
    auto it = f.states.find(name);
    if (it == f.states.end()) throw std::runtime_error("qwen38_graph: dense state lacks " + name);
    return static_cast<uint8_t*>(it->second->GetTensorMutableRawData());
  };
  for (auto& [name, t] : store_) {
    const std::string source_name =
        t->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 ? name + ".quantized" : name;
    const uint8_t* src = dense_ptr(source_name);
    if (name == "past_position_ids") {
      plane_.Check(plane_.memcpy_d2d(t->ptr, src, P * 8, stream_), "handoff positions");
    } else {
      // [1, 2, cap, 256] <- [1, 2, P, 256]
      const auto source_type = f.states.at(source_name)->GetTensorTypeAndShapeInfo()->GetElementType();
      if (source_type != t->type)
        throw std::runtime_error("qwen38_graph: prefill/decode cache type mismatch for " + name);
      const size_t element_size = Ort::SizeOf(t->type);
      const size_t head_dst = static_cast<size_t>(capacity_) * 256 * element_size;
      const size_t head_src = P * 256 * element_size;
      for (int h = 0; h < 2; ++h)
        plane_.Check(plane_.memcpy_d2d(t->ptr + h * head_dst, src + h * head_src, head_src, stream_), "handoff kv");
    }
  }
  const size_t blocks = P / kRatio, pending = P % kRatio;
  for (auto& [name, t] : banks_[0].tensors) {
    const std::string tail = TailOf(name);
    if (tail == "index_buffer" || tail == "index_lengths") continue;
    const uint8_t* src = dense_ptr(name);
    if (tail == "recurrent_state") {
      // dense [V, K] -> packed [K, V]; shape [1, heads, 128, 128] fp32
      plane_.transpose_f32(stream_, src, t->ptr, static_cast<int>(t->shape.back()),
                           static_cast<int>(t->Bytes() / (4 * t->shape.back() * t->shape.back())));
    } else {
      plane_.Check(plane_.memcpy_d2d(t->ptr, src, t->Bytes(), stream_), "handoff state");
    }
  }
  const auto& qc = *f.consts;
  const int32_t lengths[2] = {static_cast<int32_t>(blocks), static_cast<int32_t>(pending)};
  for (auto& [name, shared] : shared_) {
    const int layer = LayerOf(name);
    const uint8_t* raw = dense_ptr(name);  // raw per-token keys bf16 [1, P, 128]
    const std::string base = "past_key_values." + std::to_string(layer) + ".";
    auto knorm = qc.target.knorm.find(layer);
    if (knorm == qc.target.knorm.end()) throw std::runtime_error("qwen38_graph: no QSA key norm for layer " + std::to_string(layer));
    plane_.pool_keys(stream_, raw, static_cast<int>(blocks), knorm->second, qc.target.cos, qc.target.sin, qc.target.width,
                     qc.epsilon, 0, shared->ptr);
    if (pending)
      plane_.Check(plane_.memcpy_d2d(banks_[0].tensors.at(base + "index_buffer")->ptr, raw + blocks * kRatio * 128 * 2,
                                     pending * 128 * 2, stream_),
                   "handoff index buffer");
    plane_.Check(plane_.memcpy_h2d(banks_[0].tensors.at(base + "index_lengths")->ptr, lengths, 8, stream_), "handoff index lengths");
  }
  SyncStream();  // Complete copies before the stack lengths and dense outputs are released.
  f.states.clear();
  f.logits.reset();
  cur_ = 0;
  f.timings.handoff += SecondsSince(start);
}

// Batched head prompt prefill: KV-only head model over rows p = 1..P-1 = (token p, target hidden p-1) in chunks, written to
// head slot p-1, then the pooled index keys / pending rows / lengths. Mirrors head_kv.py.
void Qwen38GraphEngine::FastHeadKvPrefill() {
  FastPrefillState& f = *fast_;
  if (f.unified && !model_.session_head_kv_) {
    FastUnifiedHeadPrefill();
    return;
  }
  const auto start = Clock::now();
  const size_t P = length_;
  if (P < 2) {
    head_length_ = 0;
    return;
  }
  const size_t rows = P - 1;
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  const auto i64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  const auto bf16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  uint8_t* key = head_store_.at("past_key_values.0.key")->ptr;
  uint8_t* value = head_store_.at("past_key_values.0.value")->ptr;
  const auto cache_type = head_store_.at("past_key_values.0.key")->type;
  if (head_store_.at("past_key_values.0.value")->type != cache_type)
    throw std::runtime_error("qwen38_graph: head key/value cache types differ");
  const size_t row_size = 256 * Ort::SizeOf(cache_type);
  const size_t head_stride = static_cast<size_t>(capacity_) * row_size;
  for (size_t lo = 0; lo < rows; lo += static_cast<size_t>(f.chunk)) {
    const size_t m = std::min(static_cast<size_t>(f.chunk), rows - lo);
    auto binding = OrtIoBinding::Create(*f.head_kv);
    std::vector<std::unique_ptr<OrtValue>> views;
    auto view = [&](void* ptr, std::vector<int64_t> shape, ONNXTensorElementDataType type) -> OrtValue& {
      size_t count = 1;
      for (int64_t d : shape) count *= static_cast<size_t>(d);
      views.push_back(OrtValue::CreateTensor(*memory_info_, ptr, count * Ort::SizeOf(type), shape, type));
      return *views.back();
    };
    for (const auto& name : f.head_kv->GetInputNames()) {
      if (name == "input_ids")
        binding->BindInput(name.c_str(), view(d_tokens_.data() + 8 * (1 + lo), {1, int64_t(m)}, i64));
      else if (name == "attention_mask")
        binding->BindInput(name.c_str(), view(f.ones.data(), {1, int64_t(m)}, i64));
      else if (name == "position_ids")
        binding->BindInput(name.c_str(), view(d_arange64_.data() + 8 * (1 + lo), {1, int64_t(m)}, i64));
      else if (name == "past_position_ids")
        binding->BindInput(name.c_str(), view(f.scratch.data(), {1, 0}, i64));
      else if (name == "past_key_values.0.index_key")
        binding->BindInput(name.c_str(), view(f.scratch.data(), {1, 0, 128}, bf16));
      else if (name == "hidden_states")
        binding->BindInput(name.c_str(), view(d_prompt_hidden_.data() + lo * row_bytes, {1, int64_t(m), hidden_width_}, bf16));
      else
        throw std::runtime_error("qwen38_graph: unsupported head_kv input " + name);
    }
    for (const char* name : {"key", "value", "index_key"}) binding->BindOutput(name, *memory_info_);
    f.head_kv->Run(f.run_options.get(), *binding);
    const auto names = binding->GetOutputNames();
    auto values = binding->GetOutputValues();
    for (size_t i = 0; i < names.size(); ++i) {
      uint8_t* src = static_cast<uint8_t*>(values[i]->GetTensorMutableRawData());
      const auto shape = values[i]->GetTensorTypeAndShapeInfo()->GetShape();
      if ((names[i] == "key" || names[i] == "value") &&
          values[i]->GetTensorTypeAndShapeInfo()->GetElementType() != cache_type)
        throw std::runtime_error("qwen38_graph: head prefill/decode cache type mismatch for " + names[i]);
      if (names[i] == "key") {
        if (shape.size() == 4 && shape[1] == 2 && shape[2] == int64_t(m) && shape[3] == 256) {
          // [1, 2, m, 256]
          for (int h = 0; h < 2; ++h)
            plane_.Check(plane_.memcpy_d2d(key + h * head_stride + lo * row_size, src + h * m * row_size,
                                           m * row_size, stream_),
                         "head key");
        } else if (shape.size() == 3 && shape[1] == int64_t(m) && shape[2] == 512) {
          // [1, m, 512] -> per head h columns [256 h, 256 (h+1))
          for (int h = 0; h < 2; ++h)
            plane_.Check(plane_.memcpy2d_d2d(key + h * head_stride + lo * row_size, row_size, src + h * row_size,
                                             2 * row_size, row_size, m, stream_),
                         "head key");
        } else {
          std::ostringstream message;
          message << "qwen38_graph: unexpected head_kv key shape [";
          for (int64_t d : shape) message << d << ' ';
          throw std::runtime_error(message.str() + "]");
        }
      } else if (names[i] == "value") {
        // [1, m, 512] -> per head h columns [256 h, 256 (h+1))
        if (shape.size() != 3 || shape[1] != int64_t(m) || shape[2] != 512)
          throw std::runtime_error("qwen38_graph: unexpected head_kv value shape");
        for (int h = 0; h < 2; ++h)
          plane_.Check(plane_.memcpy2d_d2d(value + h * head_stride + lo * row_size, row_size, src + h * row_size,
                                           2 * row_size, row_size, m, stream_),
                       "head value");
      } else if (names[i] == "index_key") {
        if (shape.size() != 3 || shape[1] != int64_t(m) || shape[2] != 128)
          throw std::runtime_error("qwen38_graph: unexpected head_kv index_key shape");
        plane_.Check(plane_.memcpy_d2d(f.raw_keys.data() + lo * 256, src, m * 256, stream_), "head raw keys");
      }
    }
    SyncStream();  // the output values die here
  }
  const size_t blocks = rows / kRatio, pending = rows % kRatio;
  const auto& qc = f.consts->head;
  plane_.pool_keys(stream_, f.raw_keys.data(), static_cast<int>(blocks), qc.knorm.at(0), qc.cos, qc.sin, qc.width,
                   f.consts->epsilon, 0, head_store_.at("past_key_values.0.index_key")->ptr);
  HeadBank& bank = head_banks_[head_cur_];
  if (pending)
    plane_.Check(plane_.memcpy_d2d(bank.tensors.at("past_key_values.0.index_buffer")->ptr, f.raw_keys.data() + blocks * kRatio * 256,
                                   pending * 256, stream_),
                 "head index buffer");
  const int32_t lengths[2] = {static_cast<int32_t>(blocks), static_cast<int32_t>(pending)};
  plane_.Check(plane_.memcpy_h2d(bank.tensors.at("past_key_values.0.index_lengths")->ptr, lengths, 8, stream_), "head lengths");
  SyncStream();
  head_length_ = static_cast<int>(rows);
  f.timings.head += SecondsSince(start);
}

}  // namespace Generators
#endif  // defined(__linux__)
