// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Qwen3.8-Flash-Next NVFP4 (model.type "qwen38_flash_graph"), fast prefill: shared weight pool, dense chunked prefill,
// dense -> packed state hand-off, batched KV-only head prefill. See qwen38_graph.h and tools/qwen38_graph/README.md.
#include "generator/generators.h"
#include "logging.h"
#include "qwen38_graph.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

namespace Generators {

namespace {

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

float Bf16ToFloat(uint16_t bits) {
  const uint32_t widened = static_cast<uint32_t>(bits) << 16;
  float value;
  std::memcpy(&value, &widened, sizeof(value));
  return value;
}

void SetQwen38EnvVar(const char* name, const char* value) {
#if defined(_WIN32)
  _putenv_s(name, value);
#else
  setenv(name, value, 1);
#endif
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// Shared weight pool
// ------------------------------------------------------------------------------------------------
Qwen38WeightPool::Qwen38WeightPool(Qwen38Plane& plane, const std::filesystem::path& manifest_path,
                                   const std::filesystem::path& root, const OrtMemoryInfo& memory_info, int threads)
    : plane_{plane},
      cpu_memory_info_{OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)} {
  Qwen38WeightPoolManifest manifest = ParseQwen38WeightPoolManifest(manifest_path);
  sessions_ = std::move(manifest.sessions);
  for (auto& source : manifest.entries) {
    Entry entry;
    entry.dtype = source.dtype;
    entry.offset = source.offset;
    entry.length = source.length;
    entry.shape = std::move(source.shape);
    entry.file = root / source.file;
    entry.placement = std::move(source.placement);
    entries_.push_back(std::move(entry));
  }

  const char* cache_env = std::getenv("ORT_QMOE_HOST_CACHE_MIB");
  const bool bounded_host_cache = cache_env && std::string(cache_env) != "0";
  const char* pin_pool = std::getenv("ORTGENAI_PIN_CPU_POOL");
  if (bounded_host_cache && pin_pool && std::string(pin_pool) != "0")
    throw std::runtime_error("Bounded host cache cannot be combined with ORTGENAI_PIN_CPU_POOL");
  const auto start = Clock::now();
  std::vector<size_t> device_indices;
  std::map<std::filesystem::path, std::vector<std::pair<size_t, size_t>>> raw_ranges;
  constexpr size_t kPage = 4096;

  for (size_t idx = 0; idx < entries_.size(); ++idx) {
    auto& entry = entries_[idx];
    if (entry.placement == "cpu" || entry.placement == "host_mapped") {
      auto key = std::make_pair(entry.file, false);
      auto it = file_mappings_.find(key);
      if (it == file_mappings_.end()) {
        it = file_mappings_.emplace(key, std::make_unique<Qwen38FileMapping>(plane_, entry.file, false)).first;
      }
      Qwen38FileMapping& mapping = *it->second;
      if (entry.offset > mapping.size() || entry.length > mapping.size() - entry.offset) {
        throw std::runtime_error("qwen38_graph: weight pool entry exceeds mapped file size: " + entry.file.string());
      }
      entry.ptr = mapping.data() + entry.offset;
      entry.owns_device_ptr = false;
      const size_t aligned_start = (static_cast<size_t>(entry.offset) / kPage) * kPage;
      const size_t aligned_end =
          std::min(mapping.size(), ((static_cast<size_t>(entry.offset + entry.length) + kPage - 1) / kPage) * kPage);
      // Large CPU expert banks remain pageable in bounded mode (and on Windows when no separate dense UVA session exists).
      // Global scales and UVA-only tensors still need registration for the dense session's GPU reads.
#if defined(_WIN32)
      const bool skip_large_cpu_pin =
          (bounded_host_cache || (sessions_.count("dense") == 0 && (!pin_pool || std::string(pin_pool) != "1")));
#else
      const bool skip_large_cpu_pin = bounded_host_cache;
#endif
      if (aligned_end > aligned_start &&
          !(skip_large_cpu_pin && entry.placement == "cpu" && entry.length >= (size_t{1} << 20))) {
        raw_ranges[entry.file].emplace_back(aligned_start, aligned_end);
      }
      if (entry.placement == "host_mapped") {
        host_mapped_bytes += entry.length;
        entry.value = OrtValue::CreateTensor(memory_info, entry.ptr, static_cast<size_t>(entry.length), entry.shape,
                                             static_cast<ONNXTensorElementDataType>(entry.dtype));
      } else {
        cpu_bytes += entry.length;
        entry.value = OrtValue::CreateTensor(*cpu_memory_info_, entry.ptr, static_cast<size_t>(entry.length), entry.shape,
                                             static_cast<ONNXTensorElementDataType>(entry.dtype));
        entry.gpu_mapped_value = OrtValue::CreateTensor(memory_info, entry.ptr, static_cast<size_t>(entry.length),
                                                        entry.shape, static_cast<ONNXTensorElementDataType>(entry.dtype));
      }
    } else {
      void* p = nullptr;
      plane_.Check(plane_.device_alloc(&p, std::max<uint64_t>(entry.length, 16)), "cudaMalloc(weight pool)");
      entry.ptr = static_cast<uint8_t*>(p);
      entry.owns_device_ptr = true;
      bytes += entry.length;
      entry.value = OrtValue::CreateTensor(memory_info, entry.ptr, static_cast<size_t>(entry.length), entry.shape,
                                           static_cast<ONNXTensorElementDataType>(entry.dtype));
      device_indices.push_back(idx);
    }
  }

  // Register only the merged page-aligned active subranges of cpu/host_mapped files for UVA access (skips unreferenced dead bytes).
  try {
    for (auto& [file, intervals] : raw_ranges) {
      if (intervals.empty()) continue;
      std::sort(intervals.begin(), intervals.end());
      std::vector<std::pair<size_t, size_t>> merged;
      for (const auto& iv : intervals) {
        if (merged.empty() || iv.first > merged.back().second) {
          merged.push_back(iv);
        } else {
          merged.back().second = std::max(merged.back().second, iv.second);
        }
      }
      uint8_t* base = file_mappings_.at({file, false})->data();
      for (const auto& [lo, hi] : merged) {
        void* p = base + lo;
        file_mappings_.at({file, false})->PrepareNumaRange(lo, hi - lo);
#if defined(_WIN32)
        if (plane_.host_register(p, hi - lo, 2 | 8) == 0) {
          registered_ranges_.push_back(p);
        } else if (sessions_.count("dense") > 0) {
          plane_.Check(plane_.host_register(p, hi - lo, 2 | 8), "cudaHostRegister(weight pool range)");
        }
#else
        plane_.Check(plane_.host_register(p, hi - lo, 2 | 8), "cudaHostRegister(weight pool range)");
        registered_ranges_.push_back(p);
#endif
      }
    }
    if (plane_.host_get_device_pointer) {
      for (auto& entry : entries_) {
        if (entry.placement == "host_mapped" || entry.placement == "cpu") {
          void* dev_ptr = nullptr;
          if (plane_.host_get_device_pointer(&dev_ptr, entry.ptr, 0) == 0 && dev_ptr != nullptr &&
              dev_ptr != entry.ptr) {
            if (entry.placement == "host_mapped") {
              entry.value = OrtValue::CreateTensor(memory_info, dev_ptr, static_cast<size_t>(entry.length), entry.shape,
                                                   static_cast<ONNXTensorElementDataType>(entry.dtype));
            } else {
              entry.gpu_mapped_value =
                  OrtValue::CreateTensor(memory_info, dev_ptr, static_cast<size_t>(entry.length), entry.shape,
                                         static_cast<ONNXTensorElementDataType>(entry.dtype));
            }
          }
        }
      }
    }
  } catch (...) {
    for (void* p : registered_ranges_) plane_.host_unregister(p);
    registered_ranges_.clear();
    for (auto& entry : entries_) {
      entry.value.reset();
      entry.gpu_mapped_value.reset();
      if (entry.owns_device_ptr && entry.ptr) plane_.device_free(entry.ptr);
    }
    entries_.clear();
    file_mappings_.clear();
    throw;
  }

  // Copy the device-placed files in parallel: pinned staging buffer per worker, unbuffered reads from the page cache,
  // async H2D on a private stream.
  constexpr size_t kStage = size_t{64} << 20;
  std::atomic<size_t> next{0};
  std::mutex error_mutex;
  std::string error;
  auto worker = [&]() {
    void* stage = nullptr;
    void* stream = nullptr;
    std::map<std::filesystem::path, std::ifstream> files;
    try {
      plane_.Check(plane_.host_alloc(&stage, kStage), "cudaHostAlloc(pool stage)");
      plane_.Check(plane_.stream_create(&stream), "cudaStreamCreate");
      for (size_t k = next.fetch_add(1); k < device_indices.size(); k = next.fetch_add(1)) {
        Entry& entry = entries_[device_indices[k]];
        auto [found, inserted] = files.try_emplace(entry.file);
        std::ifstream& file = found->second;
        if (inserted) {
          file.rdbuf()->pubsetbuf(nullptr, 0);
          file.open(entry.file, std::ios::binary);
        }
        if (!file) throw std::runtime_error("qwen38_graph: cannot open the weight file " + entry.file.string());
        for (uint64_t done = 0; done < entry.length;) {
          const size_t n = static_cast<size_t>(std::min<uint64_t>(kStage, entry.length - done));
          file.clear();
          file.seekg(static_cast<std::streamoff>(entry.offset + done));
          file.read(static_cast<char*>(stage), static_cast<std::streamsize>(n));
          if (static_cast<size_t>(file.gcount()) != n)
            throw std::runtime_error("qwen38_graph: short read of the weight file " + entry.file.string());
          plane_.Check(plane_.memcpy_h2d(entry.ptr + done, stage, n, stream), "weight upload");
          plane_.Check(plane_.stream_sync(stream), "weight upload sync");
          done += n;
        }
      }
    } catch (const std::exception& e) {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (error.empty()) error = e.what();
      next.store(device_indices.size());
    }
    if (stream) plane_.stream_destroy(stream);
    if (stage) plane_.host_free(stage);
  };
  if (!device_indices.empty()) {
    std::vector<std::thread> workers;
    const int active_threads = std::min<int>(threads, static_cast<int>(device_indices.size()));
    for (int t = 0; t < active_threads; ++t) workers.emplace_back(worker);
    for (auto& t : workers) t.join();
  }
  if (!error.empty()) {
    for (void* p : registered_ranges_) plane_.host_unregister(p);
    registered_ranges_.clear();
    for (auto& entry : entries_) {
      entry.value.reset();
      entry.gpu_mapped_value.reset();
      if (entry.owns_device_ptr && entry.ptr) plane_.device_free(entry.ptr);
    }
    entries_.clear();
    file_mappings_.clear();
    throw std::runtime_error(error);
  }
  seconds = SecondsSince(start);
  if (g_log.enabled)
    Log("qwen38_graph") << "weight pool: " << entries_.size() << " initializers, device=" << bytes / double(1 << 30)
                        << " GiB, cpu=" << cpu_bytes / double(1 << 30)
                        << " GiB, host_mapped=" << host_mapped_bytes / double(1 << 30) << " GiB in "
                        << seconds << " s" << std::endl;
}

Qwen38WeightPool::~Qwen38WeightPool() {
  for (auto& entry : entries_) {
    entry.value.reset();
    entry.gpu_mapped_value.reset();
    if (entry.owns_device_ptr && entry.ptr) plane_.device_free(entry.ptr);
  }
  for (void* p : registered_ranges_) plane_.host_unregister(p);
  registered_ranges_.clear();
  entries_.clear();
  file_mappings_.clear();
}

size_t Qwen38WeightPool::AddTo(OrtSessionOptions& options, const std::string& label) const {
  auto it = sessions_.find(label);
  if (it == sessions_.end() && (label == "head_r2" || label == "mtp" || label == "drafter")) {
    for (const char* alt : {"mtp", "drafter", "head_r2"}) {
      it = sessions_.find(alt);
      if (it != sessions_.end()) break;
    }
  }
  if (it == sessions_.end()) return 0;
  const bool use_gpu_mapped = (label == "dense");
  for (const auto& [name, id] : it->second) {
    const Entry& entry = entries_[id];
    const OrtValue& val = (use_gpu_mapped && entry.gpu_mapped_value) ? *entry.gpu_mapped_value : *entry.value;
    options.AddInitializer(name.c_str(), val);
  }
  return it->second.size();
}

// ------------------------------------------------------------------------------------------------
// QSA constants
// ------------------------------------------------------------------------------------------------
Qwen38QsaConstants::Qwen38QsaConstants(Qwen38Plane& plane, const std::filesystem::path& index_path,
                                       const std::filesystem::path& root, void* stream) {
  const Qwen38QsaConstantsDescription description = ParseQwen38QsaConstantsIndex(index_path);
  epsilon = description.epsilon;
  target.width = head.width = description.width;
  std::ifstream data(root / description.file, std::ios::binary | std::ios::ate);
  if (!data) throw std::runtime_error("qwen38_graph: cannot open the QSA constants data " + description.file);
  const size_t total = static_cast<size_t>(data.tellg());
  std::vector<uint8_t> host(total);
  data.seekg(0);
  data.read(reinterpret_cast<char*>(host.data()), static_cast<std::streamsize>(total));
  buffer_ = PlaneBuffer(plane, total, stream);
  plane.Check(plane.memcpy_h2d(buffer_.data(), host.data(), total, stream), "QSA constants upload");
  plane.Check(plane.stream_sync(stream), "QSA constants sync");  // `host` dies here
  for (const auto& item : description.items) {
    if (item.offset + item.bytes > total) throw std::runtime_error("qwen38_graph: QSA constant outside of the data file: " + item.name);
    Set& set = item.set == "head" ? head : target;
    const uint8_t* ptr = buffer_.data() + item.offset;
    if (item.name == "cos")
      set.cos = ptr;
    else if (item.name == "sin")
      set.sin = ptr;
    else if (item.name.rfind("knorm.", 0) == 0)
      set.knorm[std::stoi(item.name.substr(6))] = ptr;
    else if (item.name.rfind("kscale.", 0) == 0)
      set.kscale[std::stoi(item.name.substr(7))] = ptr;
    else if (item.name.rfind("vscale.", 0) == 0)
      set.vscale[std::stoi(item.name.substr(7))] = ptr;
  }
  if (!target.cos || !head.cos || target.knorm.empty() || head.knorm.empty())
    throw std::runtime_error("qwen38_graph: incomplete QSA constants " + index_path.string());
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
  auto fast = std::make_unique<FastPrefillState>();
  if (!model_.session_dense_) {
    const bool dflash = !model_.config_->model.dflash2.filename.empty();
    if (!model_.session_target_ || (!dflash && !model_.session_head_r2_))
      throw std::runtime_error("qwen38_graph: unified prefill requires target and head_r2 sessions");
    fast->unified = true;
    fast->dense = model_.session_target_.get();
    fast->head_kv = model_.session_head_r2_.get();
    fast->dense_info.Add(*fast->dense);
    if (fast->head_kv) fast->head_kv_info.Add(*fast->head_kv);
    fast->chunk = (q.prefill.chunk > 0) ? std::min<int>(q.prefill.chunk, 2048) : 512;
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
    fast->idx = PlaneBuffer(plane_, 64, stream_);
    fast->ples = PlaneBuffer(plane_, static_cast<size_t>(fast->chunk) * ple_width_ * 2, stream_);
    fast->scratch = PlaneBuffer(plane_, 4096, stream_);
    if (fast->dense_info.HasInput("inputs_embeds") ||
        (fast->head_kv && fast->head_kv_info.HasInput("inputs_embeds"))) {
      const size_t max_emb_bytes =
          static_cast<size_t>(fast->chunk) * static_cast<size_t>(model_.config_->model.decoder.hidden_size) * 2;
      fast->embeds = PlaneBuffer(plane_, max_emb_bytes, stream_);
      fast->embeds_host = PlaneHostBuffer(plane_, max_emb_bytes);
    }
    if (d_prompt_hidden_.size() == 0)
      d_prompt_hidden_ = PlaneBuffer(plane_, (dflash ? size_t(fast->chunk) : slots) * hidden_width_ * 2, stream_);
    SyncStream();
    fast->run_options = OrtRunOptions::Create();
    fast->run_options->AddConfigEntry("gpu_graph_id", "-1");
    fast_ = std::move(fast);
    return;
  }
  if (!model_.session_dense_ || !model_.session_head_kv_ || !model_.qsa_constants_)
    throw std::runtime_error("qwen38_graph: prefill.enabled requires the dense, head_kv sessions and the QSA constants");
  fast->dense = model_.session_dense_.get();
  fast->head_kv = model_.session_head_kv_.get();
  fast->dense_info.Add(*fast->dense);
  fast->head_kv_info.Add(*fast->head_kv);
  fast->chunk = (q.prefill.chunk >= 2048) ? std::min<int>(std::max(q.prefill.chunk, 4096), capacity_) : q.prefill.chunk;
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
  if (fast->dense_info.HasInput("inputs_embeds") || fast->head_kv_info.HasInput("inputs_embeds")) {
    const size_t max_emb_bytes =
        static_cast<size_t>(fast->chunk) * static_cast<size_t>(model_.config_->model.decoder.hidden_size) * 2;
    fast->embeds = PlaneBuffer(plane_, max_emb_bytes, stream_);
    fast->embeds_host = PlaneHostBuffer(plane_, max_emb_bytes);
  }
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
  const std::string shrink_cfg = "gpu:" + std::to_string(model_.p_device_->GetDeviceId(nullptr));
  fast->run_options->AddConfigEntry("memory.enable_memory_arena_shrinkage", shrink_cfg.c_str());
  fast_ = std::move(fast);
}

void Qwen38GraphEngine::FastReset() {
  SetQwen38EnvVar("ORT_QMOE_PREFILL_HANDOFF_DONE", "0");
  SetQwen38EnvVar("ORT_QMOE_PREFILL_NEW_TURN", "1");
  SetQwen38EnvVar("ORT_QMOE_PREFILL_TOTAL_TOKENS", "0");
  if (!fast_) return;
  fast_->states.clear();
  fast_->logits.reset();
  fast_->dense_length = 0;
}

// One call = whole chunk loop over `tokens`, continuing the dense state of earlier calls (chunked AppendTokens).
void Qwen38GraphEngine::FastAppendSlot(int slot, std::span<const int32_t> tokens) {
  FastPrefillState& f = *fast_;
  if (slot_decode_started_[slot]) throw std::logic_error("qwen38_graph: PrefillChunkSlot after decoding started");
  const size_t n = tokens.size();
  if (n == 0) throw std::logic_error("qwen38_graph: empty prefill chunk");
  slot_ledgers_[slot].RequireRoom(n, "the prompt", static_cast<size_t>(verify_rows_ + 1));
  const size_t base = slot_ledgers_[slot].Length();
  size_t total_expected = base + n;
  if (base == 0) {
    SetQwen38EnvVar("ORT_QMOE_PREFILL_NEW_TURN", "1");
  }
  SetQwen38EnvVar("ORT_QMOE_PREFILL_HANDOFF_DONE", "0");
  if (const char* env_tot = std::getenv("ORT_QMOE_PREFILL_TOTAL_TOKENS")) {
    const size_t parsed = static_cast<size_t>(std::strtoull(env_tot, nullptr, 10));
    if (parsed > total_expected) total_expected = parsed;
  }
  const std::string total_str = std::to_string(total_expected);
  SetQwen38EnvVar("ORT_QMOE_PREFILL_TOTAL_TOKENS", total_str.c_str());
  std::vector<int64_t> chunk(n);
  for (size_t i = 0; i < n; ++i) chunk[i] = tokens[i];
  plane_.Check(plane_.memcpy_h2d(d_tokens_slot(slot) + 8 * base, chunk.data(), n * 8, stream_), "upload slot prompt");
  SyncStream();
  if (slot == 0) {
    prompt_tokens_.insert(prompt_tokens_.end(), tokens.begin(), tokens.end());
  }
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  const int embed_width = model_.config_->model.decoder.hidden_size;
  const size_t embed_row_bytes = static_cast<size_t>(embed_width) * 2;
  const auto i64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  const auto i32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  const auto bf16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  const std::string ple_name = "ple_embeddings." + std::to_string(ple_layer_);

  if (dflash_ && slot_dflash_prompt_rows_[slot]) {
    DraftSlot(slot, d_dflash_prompt_slot(slot), slot_dflash_prompt_rows_[slot], slot_dflash_prompt_pos_[slot], 0, false);
    slot_dflash_prompt_rows_[slot] = 0;
    slot_dflash_ingested_[slot] = true;
  }
  const size_t dflash_ctx_win =
      (dflash_ && model_.config_->model.dflash2.sliding_window > 0)
          ? static_cast<size_t>(model_.config_->model.dflash2.sliding_window +
                                model_.config_->model.dflash2.block_size)
          : 0;
  MutedPrefillSlot(slot, true);
  const int cur_bank = bank_phase_ % num_banks_;
  const int scratch_bank = (bank_phase_ + 1) % num_banks_;
  size_t last_m = 1;
  for (size_t lo = 0; lo < n;) {
    const size_t pos = base + lo;
    size_t max_m = static_cast<size_t>(f.chunk);
    if (pos >= 16384 && max_m > 2048) max_m = std::min<size_t>(max_m, 2048);
    const size_t m = std::min(max_m, n - lo);
    last_m = m;
#if defined(_WIN32)
    const char* compact_cap = "64";
#else
    const char* compact_cap = (pos + m <= 2048) ? (m <= 1024 ? "128" : "64") : "128";
#endif
    SetQwen38EnvVar("ORT_QMOE_CURRENT_MAX_COMPACT_EXPERTS", compact_cap);
    plane_.ple_rows(stream_, params_dev_.data(), d_tokens_slot(slot) + 8 * pos, static_cast<int>(m),
                    ctx_commit_slot(slot), ctx_b_slot(slot), f.ples.data());
    plane_.Check(plane_.memcpy_d2d(ctx_commit_slot(slot), ctx_b_slot(slot), 16, stream_), "slot ctx commit");
    plane_.Check(plane_.memcpy_d2d(past_pos_slot(slot) + 8 * pos, d_arange64_.data() + 8 * pos, m * 8, stream_), "slot past_pos");
    const int32_t meta32[4] = {
        static_cast<int32_t>(pos),
        static_cast<int32_t>(m - 1),
        0,
        static_cast<int32_t>(m),
    };
    plane_.Check(plane_.memcpy_h2d(f.idx.data(), meta32, sizeof(meta32), stream_), "unified slot prefill meta");
    if (f.dense_info.HasInput("inputs_embeds")) {
      model_.RunCpuEmbedding(chunk.data() + lo, m, f.embeds_host.data());
      plane_.Check(plane_.memcpy_h2d(f.embeds.data(), f.embeds_host.data(), m * embed_row_bytes, stream_),
                   "upload slot inputs_embeds");
    }
    SyncStream();

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
        binding->BindInput(name.c_str(), view(d_tokens_slot(slot) + 8 * pos, {1, int64_t(m)}, i64));
      else if (name == "inputs_embeds")
        binding->BindInput(name.c_str(), view(f.embeds.data(), {1, int64_t(m), embed_width}, bf16));
      else if (name == "attention_mask")
        binding->BindInput(name.c_str(), view(f.ones.data(), {1, int64_t(m)}, i64));
      else if (name == "position_ids")
        binding->BindInput(name.c_str(), view(d_arange64_.data() + 8 * pos, {1, int64_t(m)}, i64));
      else if (name == "committed_length")
        binding->BindInput(name.c_str(), view(f.idx.data() + 0, {1}, i32));
      else if (name == "sequence_length_minus_one")
        binding->BindInput(name.c_str(), view(f.idx.data() + 4, {1}, i32));
      else if (name == "cumulative_sequence_lengths")
        binding->BindInput(name.c_str(), view(f.idx.data() + 8, {2}, i32));
      else if (name == "state_update_capture_count")
        binding->BindInput(name.c_str(), view(f.idx.data() + 8, {1}, i32));
      else if (name == ple_name)
        binding->BindInput(name.c_str(), view(f.ples.data(), {1, int64_t(m), ple_width_}, bf16));
      else if (name == "fold.ctl")
        binding->BindInput(name.c_str(), view(ctl_slot(slot), {verify_rows_ > 2 ? 16 : 8}, i64));
      else if (name == "fold.ctx_commit")
        binding->BindInput(name.c_str(), view(ctx_commit_slot(slot), {2}, i64));
      else if (name == "fold.ctx_a")
        binding->BindInput(name.c_str(), view(ctx_a_slot(slot), {verify_rows_ > 2 ? 2 * verify_rows_ : 2}, i64));
      else if (name == "fold.ctx_b")
        binding->BindInput(name.c_str(), view(ctx_a_slot(slot) + 16, {2}, i64));
      else if (fold_tensors_.count(name))
        binding->BindInput(name.c_str(), *fold_tensors_.at(name)->value);
      else if (store_.count(name))
        binding->BindInput(
            name.c_str(),
            view(store_.at(name)->ptr + static_cast<size_t>(slot) * store_slot_bytes_.at(name),
                 store_slot_shape_.at(name), store_.at(name)->type));
      else if (shared_.count(name))
        binding->BindInput(
            name.c_str(),
            view(shared_.at(name)->ptr + static_cast<size_t>(slot) * shared_slot_bytes_.at(name),
                 shared_slot_shape_.at(name), shared_.at(name)->type));
      else if (banks_[cur_bank].tensors.count(name))
        binding->BindInput(
            name.c_str(),
            view(banks_[cur_bank].tensors.at(name)->ptr + static_cast<size_t>(slot) * bank_slot_bytes_.at(name),
                 bank_slot_shape_.at(name), banks_[cur_bank].tensors.at(name)->type));
      else
        throw std::runtime_error("qwen38_graph: unsupported unified slot prefill input " + name);
    }
    const size_t tok_rows = std::min<size_t>(m, static_cast<size_t>(verify_rows_));
    for (const auto& name : f.dense_outputs) {
      if (name == "hidden_states" || name == "hidden_states_out")
        binding->BindOutput(name.c_str(), view(d_prompt_hidden_.data(), {1, int64_t(m), hidden_width_}, bf16));
      else if (name == "aux_hidden_states") {
        if (!dflash_) throw std::runtime_error("Unexpected DFlash auxiliary prefill output");
        binding->BindOutput(name.c_str(), view(d_dflash_prompt_slot(slot), {1, int64_t(m), int64_t(kDflashWidth)}, bf16));
      } else if (name == "next_tokens")
        binding->BindOutput(name.c_str(), view(f.scratch.data(), {1, int64_t(tok_rows)}, i64));
      else if (name == "validation.aggregate")
        binding->BindOutput(name.c_str(), view(f.scratch.data() + 64, {int64_t(status_count_)}, i32));
      else if (name == "present_position_ids")
        binding->BindOutput(
            name.c_str(),
            view(store_.at("past_position_ids")->ptr + static_cast<size_t>(slot) * store_slot_bytes_.at("past_position_ids"),
                 store_slot_shape_.at("past_position_ids"), store_.at("past_position_ids")->type));
      else if (name.rfind("state_update.", 0) == 0) {
        binding->BindOutput(
            name.c_str(),
            view(state_update_tensors_.at(name)->ptr + static_cast<size_t>(slot) * su_slot_bytes_.at(name),
                 su_slot_shape_.at(name), state_update_tensors_.at(name)->type));
      } else if (name.rfind("present.", 0) == 0) {
        const std::string in_n = "past_key_values." + name.substr(8);
        if (store_.count(in_n)) {
          binding->BindOutput(
              name.c_str(),
              view(store_.at(in_n)->ptr + static_cast<size_t>(slot) * store_slot_bytes_.at(in_n),
                   store_slot_shape_.at(in_n), store_.at(in_n)->type));
        } else if (shared_.count(in_n)) {
          binding->BindOutput(
              name.c_str(),
              view(shared_.at(in_n)->ptr + static_cast<size_t>(slot) * shared_slot_bytes_.at(in_n),
                   shared_slot_shape_.at(in_n), shared_.at(in_n)->type));
        } else {
          binding->BindOutput(
              name.c_str(),
              view(banks_[scratch_bank].tensors.at(in_n)->ptr + static_cast<size_t>(slot) * bank_slot_bytes_.at(in_n),
                   bank_slot_shape_.at(in_n), banks_[scratch_bank].tensors.at(in_n)->type));
        }
      } else {
        binding->BindOutput(name.c_str(), *memory_info_);
      }
    }
    f.dense->Run(f.run_options.get(), *binding);
    plane_.or_status(stream_, f.scratch.data() + 64, static_cast<int>(status_count_), ctl_slot(slot));
    if (max_batch_ == 1) {
      plane_.Check(plane_.memcpy_d2d(banks_[cur_bank].arena.data(), banks_[scratch_bank].arena.data(),
                                     banks_[cur_bank].arena.size(), stream_),
                   "commit unified bank");
    } else {
      for (const auto& [bname, t_dst] : banks_[cur_bank].tensors) {
        const size_t sb = bank_slot_bytes_.at(bname);
        const uint8_t* src_ptr = banks_[scratch_bank].tensors.at(bname)->ptr + static_cast<size_t>(slot) * sb;
        plane_.Check(plane_.memcpy_d2d(t_dst->ptr + static_cast<size_t>(slot) * sb, src_ptr, sb, stream_),
                     "commit unified slot bank");
      }
    }
    SyncStream();
    if (slot == 0) f.dense_length += m;
    if (dflash_) {
      if (lo + m < n) {
        if (slot_dflash_ingested_[slot] || dflash_ctx_win == 0 || (n - (lo + m) < dflash_ctx_win)) {
          DraftSlot(slot, d_dflash_prompt_slot(slot), m, pos, 0, false);
          slot_dflash_ingested_[slot] = true;
        }
      } else {
        slot_dflash_prompt_rows_[slot] = m;
        slot_dflash_prompt_pos_[slot] = pos;
      }
    }
    lo += m;
  }
  slot_ledgers_[slot].AdvancePrompt(n);
  const size_t last_tok_idx = std::min<size_t>(last_m, static_cast<size_t>(verify_rows_)) - 1;
  plane_.Check(plane_.memcpy_d2h(stage_host_ + 4, f.scratch.data() + 8 * last_tok_idx, 8, stream_), "read slot prediction");
  SyncStream();
  if (ReadCtlSlot(slot, 5) != 0) CheckDeviceStatus("unified prefill", nullptr);
  slot_pending_[slot] = static_cast<int32_t>(stage_host_[4]);
  MutedPrefillSlot(slot, false);
  if (slot == 0) {
    ledger_ = slot_ledgers_[0];
    pending_ = slot_pending_[0];
    dflash_prompt_rows_ = slot_dflash_prompt_rows_[0];
    dflash_prompt_position_ = slot_dflash_prompt_pos_[0];
    dflash_ingested_in_turn_ = slot_dflash_ingested_[0];
  }
}

void Qwen38GraphEngine::FastAppend(std::span<const int32_t> tokens) {
  if (fast_ && fast_->unified && max_batch_ > 1) {
    FastAppendSlot(0, tokens);
    return;
  }
  FastPrefillState& f = *fast_;
  if (decode_started_) throw std::logic_error("qwen38_graph: PrefillChunk after decoding started");
  const size_t n = tokens.size();
  if (n == 0) throw std::logic_error("qwen38_graph: empty prefill chunk");
  ledger_.RequireRoom(n, "the prompt", 3);
  const size_t base = ledger_.Length();
  size_t total_expected = base + n;
  if (base == 0) {
    SetQwen38EnvVar("ORT_QMOE_PREFILL_NEW_TURN", "1");
  }
  SetQwen38EnvVar("ORT_QMOE_PREFILL_HANDOFF_DONE", "0");
  if (const char* env_tot = std::getenv("ORT_QMOE_PREFILL_TOTAL_TOKENS")) {
    const size_t parsed = static_cast<size_t>(std::strtoull(env_tot, nullptr, 10));
    if (parsed > total_expected) total_expected = parsed;
  }
  const std::string total_str = std::to_string(total_expected);
  SetQwen38EnvVar("ORT_QMOE_PREFILL_TOTAL_TOKENS", total_str.c_str());
  std::vector<int64_t> chunk(n);
  for (size_t i = 0; i < n; ++i) chunk[i] = tokens[i];
  plane_.Check(plane_.memcpy_h2d(d_tokens_.data() + 8 * base, chunk.data(), n * 8, stream_), "upload prompt");
  SyncStream();
  prompt_tokens_.insert(prompt_tokens_.end(), tokens.begin(), tokens.end());
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  const int embed_width = model_.config_->model.decoder.hidden_size;
  const size_t embed_row_bytes = static_cast<size_t>(embed_width) * 2;
  const auto i64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  const auto i32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  const auto bf16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  const std::string ple_name = "ple_embeddings." + std::to_string(ple_layer_);

  if (f.unified) {
    if (dflash_ && dflash_prompt_rows_) {
      Draft(d_dflash_prompt_.data(), dflash_prompt_rows_, dflash_prompt_position_, 0, false);
      dflash_prompt_rows_ = 0;
      dflash_ingested_in_turn_ = true;
    }
    const size_t dflash_ctx_win =
        (dflash_ && model_.config_->model.dflash2.sliding_window > 0)
            ? static_cast<size_t>(model_.config_->model.dflash2.sliding_window +
                                  model_.config_->model.dflash2.block_size)
            : 0;
    MutedPrefill(true);
    size_t last_m = 1;
    for (size_t lo = 0; lo < n;) {
      const size_t pos = base + lo;
      size_t max_m = static_cast<size_t>(f.chunk);
      if (pos >= 16384 && max_m > 2048) max_m = std::min<size_t>(max_m, 2048);
      const size_t m = std::min(max_m, n - lo);
      last_m = m;
#if defined(_WIN32)
      const char* compact_cap = "64";
#else
      const char* compact_cap = (pos + m <= 2048) ? (m <= 1024 ? "128" : "64") : "128";
#endif
      SetQwen38EnvVar("ORT_QMOE_CURRENT_MAX_COMPACT_EXPERTS", compact_cap);
      plane_.ple_rows(stream_, params_dev_.data(), d_tokens_.data() + 8 * pos, static_cast<int>(m), ctx_commit_.data(),
                      ctx_b_.data(), f.ples.data());
      plane_.Check(plane_.memcpy_d2d(ctx_commit_.data(), ctx_b_.data(), 16, stream_), "ctx commit");
      plane_.Check(plane_.memcpy_d2d(past_pos_ + 8 * pos, d_arange64_.data() + 8 * pos, m * 8, stream_), "past_pos");
      const int32_t meta32[4] = {
          static_cast<int32_t>(pos),
          static_cast<int32_t>(m - 1),
          0,
          static_cast<int32_t>(m),
      };
      plane_.Check(plane_.memcpy_h2d(f.idx.data(), meta32, sizeof(meta32), stream_), "unified prefill meta");
      if (auto it = head_banks_[0].tensors.find("past_key_values.0.index_lengths"); it != head_banks_[0].tensors.end()) {
        plane_.Check(plane_.device_memset(it->second->ptr, 0, it->second->Bytes(), stream_), "zero muted head len");
      }
      if (f.dense_info.HasInput("inputs_embeds")) {
        model_.RunCpuEmbedding(chunk.data() + lo, m, f.embeds_host.data());
        plane_.Check(plane_.memcpy_h2d(f.embeds.data(), f.embeds_host.data(), m * embed_row_bytes, stream_),
                     "upload inputs_embeds");
      }
      SyncStream();

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
        else if (name == "inputs_embeds")
          binding->BindInput(name.c_str(), view(f.embeds.data(), {1, int64_t(m), embed_width}, bf16));
        else if (name == "attention_mask")
          binding->BindInput(name.c_str(), view(f.ones.data(), {1, int64_t(m)}, i64));
        else if (name == "position_ids")
          binding->BindInput(name.c_str(), view(d_arange64_.data() + 8 * pos, {1, int64_t(m)}, i64));
        else if (name == "committed_length")
          binding->BindInput(name.c_str(), view(f.idx.data() + 0, {1}, i32));
        else if (name == "sequence_length_minus_one")
          binding->BindInput(name.c_str(), view(f.idx.data() + 4, {1}, i32));
        else if (name == "cumulative_sequence_lengths")
          binding->BindInput(name.c_str(), view(f.idx.data() + 8, {2}, i32));
        else if (name == "state_update_capture_count")
          binding->BindInput(name.c_str(), view(f.idx.data() + 8, {1}, i32));
        else if (name == ple_name)
          binding->BindInput(name.c_str(), view(f.ples.data(), {1, int64_t(m), ple_width_}, bf16));
        else if (fold_tensors_.count(name))
          binding->BindInput(name.c_str(), *fold_tensors_.at(name)->value);
        else if (name.rfind("head.past_key_values.0.", 0) == 0) {
          const std::string sub = "past_key_values.0." + name.substr(23);
          OrtValue& val = head_store_.count(sub) ? *head_store_.at(sub)->value : *head_banks_[0].tensors.at(sub)->value;
          binding->BindInput(name.c_str(), val);
        } else if (store_.count(name))
          binding->BindInput(name.c_str(), *store_.at(name)->value);
        else if (shared_.count(name))
          binding->BindInput(name.c_str(), *shared_.at(name)->value);
        else if (banks_[0].tensors.count(name))
          binding->BindInput(name.c_str(), *banks_[0].tensors.at(name)->value);
        else
          throw std::runtime_error("qwen38_graph: unsupported unified target prefill input " + name);
      }
      const size_t tok_rows = std::min<size_t>(m, static_cast<size_t>(verify_rows_));
      for (const auto& name : f.dense_outputs) {
        if (name == "hidden_states" || name == "hidden_states_out")
          binding->BindOutput(name.c_str(), view(d_prompt_hidden_.data() + (dflash_ ? 0 : pos * row_bytes), {1, int64_t(m), hidden_width_}, bf16));
        else if (name == "aux_hidden_states") {
          if (!dflash_) throw std::runtime_error("Unexpected DFlash auxiliary prefill output");
          binding->BindOutput(name.c_str(), view(d_dflash_prompt_.data(), {1, int64_t(m), int64_t(kDflashWidth)}, bf16));
        } else if (name == "next_tokens")
          binding->BindOutput(name.c_str(), view(f.scratch.data(), {1, int64_t(tok_rows)}, i64));
        else if (name == "validation.aggregate")
          binding->BindOutput(name.c_str(), view(f.scratch.data() + 64, {int64_t(status_count_)}, i32));
        else if (name == "present_position_ids")
          binding->BindOutput(name.c_str(), *store_.at("past_position_ids")->value);
        else if (name.rfind("present_mid.", 0) == 0) {
          const std::string in_n = "past_key_values." + name.substr(12);
          binding->BindOutput(name.c_str(), *banks_[2].tensors.at(in_n)->value);
        } else if (name.rfind("present_mid2.", 0) == 0) {
          const std::string in_n = "past_key_values." + name.substr(13);
          binding->BindOutput(name.c_str(), *banks_[3].tensors.at(in_n)->value);
        } else if (name.rfind("present_mid3.", 0) == 0) {
          const std::string in_n = "past_key_values." + name.substr(13);
          binding->BindOutput(name.c_str(), *banks_[4].tensors.at(in_n)->value);
        } else if (name.rfind("state_update.", 0) == 0) {
          binding->BindOutput(name.c_str(), *state_update_tensors_.at(name)->value);
        } else if (name.rfind("present.", 0) == 0) {
          const std::string in_n = "past_key_values." + name.substr(8);
          OrtValue& val = store_.count(in_n) ? *store_.at(in_n)->value
                        : shared_.count(in_n) ? *shared_.at(in_n)->value
                                              : *banks_[1].tensors.at(in_n)->value;
          binding->BindOutput(name.c_str(), val);
        } else if (name.rfind("head.present_mid.0.", 0) == 0) {
          const std::string sub = "past_key_values.0." + name.substr(19);
          OrtValue& val = head_store_.count(sub) ? *head_store_.at(sub)->value : *head_banks_[2].tensors.at(sub)->value;
          binding->BindOutput(name.c_str(), val);
        } else if (name.rfind("head.present_mid2.0.", 0) == 0 || name.rfind("head.present_mid3.0.", 0) == 0) {
          const std::string sub = "past_key_values.0." + name.substr(20);
          OrtValue& val = head_store_.count(sub) ? *head_store_.at(sub)->value : *head_banks_[2].tensors.at(sub)->value;
          binding->BindOutput(name.c_str(), val);
        } else if (name.rfind("head.present_s1.0.", 0) == 0 || name.rfind("head.present_s2.0.", 0) == 0) {
          const std::string sub = "past_key_values.0." + name.substr(18);
          OrtValue& val = head_store_.count(sub) ? *head_store_.at(sub)->value : *head_banks_[1].tensors.at(sub)->value;
          binding->BindOutput(name.c_str(), val);
        } else if (name.rfind("head.present.0.", 0) == 0) {
          const std::string sub = "past_key_values.0." + name.substr(15);
          OrtValue& val = head_store_.count(sub) ? *head_store_.at(sub)->value : *head_banks_[1].tensors.at(sub)->value;
          binding->BindOutput(name.c_str(), val);
        } else {
          binding->BindOutput(name.c_str(), *memory_info_);
        }
      }
      f.dense->Run(f.run_options.get(), *binding);
      plane_.or_status(stream_, f.scratch.data() + 64, static_cast<int>(status_count_), ctl_.data());
      plane_.Check(plane_.memcpy_d2d(banks_[0].arena.data(), banks_[1].arena.data(), banks_[0].arena.size(), stream_),
                   "commit unified bank 0");
      SyncStream();
      f.dense_length += m;
      if (dflash_) {
        if (lo + m < n) {
          if (dflash_ingested_in_turn_ || dflash_ctx_win == 0 || (n - (lo + m) < dflash_ctx_win)) {
            Draft(d_dflash_prompt_.data(), m, pos, 0, false);
            dflash_ingested_in_turn_ = true;
          }
        } else {
          dflash_prompt_rows_ = m;
          dflash_prompt_position_ = pos;
        }
      }
      lo += m;
    }
    ledger_.AdvancePrompt(n);
    const size_t last_tok_idx = std::min<size_t>(last_m, static_cast<size_t>(verify_rows_)) - 1;
    const size_t last_hidden_row = dflash_ ? (last_m - 1) : (ledger_.Length() - 1);
    plane_.Check(plane_.memcpy_d2d(d_last_hidden_.data(), d_prompt_hidden_.data() + last_hidden_row * row_bytes, row_bytes, stream_),
                 "last hidden");
    plane_.Check(plane_.memcpy_d2h(stage_host_ + 4, f.scratch.data() + 8 * last_tok_idx, 8, stream_), "read prediction");
    SyncStream();
    if (ReadCtl(5) != 0) CheckDeviceStatus("unified prefill", nullptr);
    pending_ = static_cast<int32_t>(stage_host_[4]);
    MutedPrefill(false);
    slot_ledgers_[0] = ledger_;
    slot_pending_[0] = pending_;
    slot_dflash_prompt_rows_[0] = dflash_prompt_rows_;
    slot_dflash_prompt_pos_[0] = dflash_prompt_position_;
    slot_dflash_ingested_[0] = dflash_ingested_in_turn_;
    return;
  }

  for (size_t lo = 0; lo < n;) {
    const size_t pos = base + lo;  // == f.dense_length
    size_t max_m = static_cast<size_t>(f.chunk);
    if (pos >= 16384 && max_m > 2048) {
      max_m = std::min<size_t>(max_m, 2048);
    }
    const size_t m = std::min(max_m, n - lo);
    // With Step 3 prefill memory lending:
    // - At m <= 1024 (pos + m <= 2048): 0 layers lent, 128-expert compaction (4 passes) fits in free HBM.
    // - At m == 2048 (pos + m <= 2048): 0 layers lent, 64-expert compaction (8 passes) fits in free HBM.
    // - At pos + m > 2048 (up to 32k): 4,400-7,200 MiB lent, 128-expert compaction (4 passes) with double-buffered DMA.
#if defined(_WIN32)
    const char* compact_cap = "64";
#else
    const char* compact_cap =
        (pos + m <= 2048) ? (m <= 1024 ? "128" : "64") : "128";
#endif
    SetQwen38EnvVar("ORT_QMOE_CURRENT_MAX_COMPACT_EXPERTS", compact_cap);
    plane_.ple_rows(stream_, params_dev_.data(), d_tokens_.data() + 8 * pos, static_cast<int>(m), ctx_commit_.data(),
                    ctx_b_.data(), f.ples.data());
    plane_.Check(plane_.memcpy_d2d(ctx_commit_.data(), ctx_b_.data(), 16, stream_), "ctx commit");
    const int64_t last_row = static_cast<int64_t>(m) - 1;
    plane_.Check(plane_.memcpy_h2d(f.idx.data(), &last_row, 8, stream_), "logits_indices");
    if (f.dense_info.HasInput("inputs_embeds")) {
      model_.RunCpuEmbedding(chunk.data() + lo, m, f.embeds_host.data());
      plane_.Check(plane_.memcpy_h2d(f.embeds.data(), f.embeds_host.data(), m * embed_row_bytes, stream_),
                   "upload inputs_embeds");
    }

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
      else if (name == "inputs_embeds")
        binding->BindInput(name.c_str(), view(f.embeds.data(), {1, int64_t(m), embed_width}, bf16));
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
    lo += m;
  }
  ledger_.AdvancePrompt(n);

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
  plane_.Check(plane_.memcpy_d2d(d_last_hidden_.data(), d_prompt_hidden_.data() + (ledger_.Length() - 1) * row_bytes, row_bytes, stream_),
               "last hidden");
  SyncStream();
  int best = 0;
  float best_value = -INFINITY;
  for (size_t i = 0; i < raw.size(); ++i) {
    const float v = Bf16ToFloat(raw[i]);
    if (v > best_value) {
      best_value = v;
      best = static_cast<int>(i);
    }
  }
  f.logits.reset();
  pending_ = best;
}

// Dense state (KV, positions, GDN / conv / PLE state, raw index keys) -> packed decode state: KV rows into the fixed store,
// GDN state transposed into ring bank 0, PLE / conv state copied, raw index keys pooled into the shared compressed keys
// (+ pending rows and lengths). Mirrors handoff.py (HandoffWriter.write).
void Qwen38GraphEngine::FastHandoff() {
  FastPrefillState& f = *fast_;
  if (f.unified) {
    SetQwen38EnvVar("ORT_QMOE_PREFILL_HANDOFF_DONE", "1");
    SetQwen38EnvVar("ORT_QMOE_PREFILL_TOTAL_TOKENS", "0");
    return;
  }
  const size_t P = ledger_.Length();
  if (ledger_.Bank() != 0) throw std::logic_error("qwen38_graph: the dense hand-off writes bank 0");
  if (f.dense_length != P) throw std::runtime_error("qwen38_graph: dense state length differs from the prompt length");
  auto dense_ptr = [&](const std::string& name) -> uint8_t* {
    auto it = f.states.find(name);
    if (it == f.states.end()) throw std::runtime_error("qwen38_graph: dense state lacks " + name);
    return static_cast<uint8_t*>(it->second->GetTensorMutableRawData());
  };
  const auto& qc = *f.consts;
  for (auto& [name, t] : store_) {
    const uint8_t* src = dense_ptr(name);
    if (name == "past_position_ids") {
      plane_.Check(plane_.memcpy_d2d(t->ptr, src, P * 8, stream_), "handoff positions");
    } else if (t->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16) {
      // [1, 2, cap, 256] <- [1, 2, P, 256]
      const size_t head_dst = static_cast<size_t>(capacity_) * 256 * 2, head_src = P * 256 * 2;
      for (int h = 0; h < 2; ++h)
        plane_.Check(plane_.memcpy_d2d(t->ptr + h * head_dst, src + h * head_src, head_src, stream_), "handoff kv");
    } else if (t->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || t->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN) {
      if (!plane_.quantize_kv)
        throw std::runtime_error("qwen38_graph: plane library lacks qwen38_plane_quantize_kv for quantized KV cache");
      const int layer = LayerOf(name);
      const std::string tail = TailOf(name);
      const auto& smap = (tail == "key") ? qc.target.kscale : qc.target.vscale;
      auto sit = smap.find(layer);
      if (sit == smap.end())
        throw std::runtime_error("qwen38_graph: missing KV scale in qsa_consts for " + name);
      const int mode = (t->type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) ? 1 : 2;
      const size_t head_dst = static_cast<size_t>(capacity_) * 256, head_src = P * 256 * 2;
      for (int h = 0; h < 2; ++h) {
        plane_.quantize_kv(stream_, src + h * head_src, t->ptr + h * head_dst, sit->second + h * 256 * sizeof(float),
                           static_cast<int>(P), 256, 256, mode);
      }
    } else {
      throw std::runtime_error("qwen38_graph: unsupported store tensor dtype for " + name);
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
  SyncStream();  // `lengths` lives on the stack
  f.states.clear();
  f.logits.reset();
  SetQwen38EnvVar("ORT_QMOE_PREFILL_HANDOFF_DONE", "1");
  SetQwen38EnvVar("ORT_QMOE_PREFILL_TOTAL_TOKENS", "0");
}

// Batched head prompt prefill: KV-only head model over rows p = 1..P-1 = (token p, target hidden p-1) in chunks, written to
// head slot p-1, then the pooled index keys / pending rows / lengths. Mirrors head_kv.py.
void Qwen38GraphEngine::FastHeadKvPrefill() {
  FastPrefillState& f = *fast_;
  const size_t P = ledger_.Length();
  if (P < 2) {
    head_length_ = 0;
    return;
  }
  const size_t rows = P - 1;
  const size_t row_bytes = static_cast<size_t>(hidden_width_) * 2;
  const int embed_width = model_.config_->model.decoder.hidden_size;
  const size_t embed_row_bytes = static_cast<size_t>(embed_width) * 2;
  const auto i64 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  const auto i32 = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  const auto bf16 = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  if (f.unified) {
    const size_t head_chunk = static_cast<size_t>(f.chunk);
    head_cur_ = 0;
    for (size_t lo = 0; lo < rows; lo += head_chunk) {
      const size_t m = std::min(head_chunk, rows - lo);
      const int32_t committed = static_cast<int32_t>(lo);
      plane_.Check(plane_.memcpy_h2d(f.idx.data(), &committed, sizeof(committed), stream_), "head committed");
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
        else if (name == "position_ids")
          binding->BindInput(name.c_str(), view(d_arange64_.data() + 8 * (1 + lo), {1, int64_t(m)}, i64));
        else if (name == "hidden_states")
          binding->BindInput(name.c_str(), view(d_prompt_hidden_.data() + lo * row_bytes, {1, int64_t(m), hidden_width_}, bf16));
        else if (name == "committed_length")
          binding->BindInput(name.c_str(), view(f.idx.data(), {1}, i32));
        else if (head_store_.count(name))
          binding->BindInput(name.c_str(), *head_store_.at(name)->value);
        else if (head_banks_[0].tensors.count(name))
          binding->BindInput(name.c_str(), *head_banks_[0].tensors.at(name)->value);
        else
          throw std::runtime_error("qwen38_graph: unsupported unified head_r2 prefill input " + name);
      }
      for (const auto& name : f.head_kv->GetOutputNames()) {
        const std::string state = "past_key_values." + name.substr(name.find('.') + 1);
        if (name == "validation.aggregate")
          binding->BindOutput(name.c_str(), view(f.scratch.data(), {int64_t(head_r2_.status_count)}, i32));
        else if (name.rfind("present_mid.", 0) == 0) {
          OrtValue& val = head_store_.count(state) ? *head_store_.at(state)->value : *head_banks_[2].tensors.at(state)->value;
          binding->BindOutput(name.c_str(), val);
        } else if (name.rfind("present.", 0) == 0) {
          OrtValue& val = head_store_.count(state) ? *head_store_.at(state)->value : *head_banks_[1].tensors.at(state)->value;
          binding->BindOutput(name.c_str(), val);
        } else {
          binding->BindOutput(name.c_str(), *memory_info_);
        }
      }
      f.head_kv->Run(f.run_options.get(), *binding);
      plane_.or_status(stream_, f.scratch.data(), static_cast<int>(head_r2_.status_count), ctl_.data());
      for (auto& entry : head_banks_[1].tensors) {
        plane_.Check(plane_.memcpy_d2d(head_banks_[0].tensors.at(entry.first)->ptr, entry.second->ptr, entry.second->Bytes(), stream_),
                     "commit unified head bank 0");
      }
      SyncStream();
    }
    if (ReadCtl(5) != 0) CheckDeviceStatus("unified head prefill", nullptr);
    head_length_ = static_cast<int>(rows);
    return;
  }
  const DevTensor& key_tensor = *head_store_.at("past_key_values.0.key");
  const DevTensor& value_tensor = *head_store_.at("past_key_values.0.value");
  uint8_t* key = key_tensor.ptr;
  uint8_t* value = value_tensor.ptr;
  const bool kv_quant =
      (key_tensor.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 || key_tensor.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN);
  const int kv_mode = (key_tensor.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8) ? 1 : 2;
  const size_t elem_bytes = kv_quant ? 1 : 2;
  const size_t head_stride = static_cast<size_t>(capacity_) * 256 * elem_bytes;
  const auto& qc = f.consts->head;
  const uint8_t* head_kscale = nullptr;
  const uint8_t* head_vscale = nullptr;
  if (kv_quant) {
    if (!plane_.quantize_kv)
      throw std::runtime_error("qwen38_graph: plane library lacks qwen38_plane_quantize_kv for quantized head KV cache");
    auto ksit = qc.kscale.find(0);
    auto vsit = qc.vscale.find(0);
    if (ksit == qc.kscale.end() || vsit == qc.vscale.end())
      throw std::runtime_error("qwen38_graph: missing head KV scale in qsa_consts");
    head_kscale = ksit->second;
    head_vscale = vsit->second;
  }
  const size_t head_chunk = std::min<size_t>(static_cast<size_t>(f.chunk), 512);
  for (size_t lo = 0; lo < rows; lo += head_chunk) {
    const size_t m = std::min(head_chunk, rows - lo);
    if (f.head_kv_info.HasInput("inputs_embeds")) {
      std::vector<int64_t> h_ids(m);
      for (size_t i = 0; i < m; ++i) h_ids[i] = prompt_tokens_[1 + lo + i];
      model_.RunCpuEmbedding(h_ids.data(), m, f.embeds_host.data());
      plane_.Check(plane_.memcpy_h2d(f.embeds.data(), f.embeds_host.data(), m * embed_row_bytes, stream_),
                   "upload head_kv inputs_embeds");
    }
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
      else if (name == "inputs_embeds")
        binding->BindInput(name.c_str(), view(f.embeds.data(), {1, int64_t(m), embed_width}, bf16));
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
      if (names[i] == "key") {
        if (shape.size() == 4 && shape[1] == 2 && shape[2] == int64_t(m) && shape[3] == 256) {
          // [1, 2, m, 256]
          for (int h = 0; h < 2; ++h) {
            if (!kv_quant) {
              plane_.Check(plane_.memcpy_d2d(key + h * head_stride + lo * 512, src + h * m * 512, m * 512, stream_), "head key");
            } else {
              plane_.quantize_kv(stream_, src + h * m * 512, key + h * head_stride + lo * 256,
                                 head_kscale + h * 256 * sizeof(float), static_cast<int>(m), 256, 256, kv_mode);
            }
          }
        } else if (shape.size() == 3 && shape[1] == int64_t(m) && shape[2] == 512) {
          // [1, m, 512] -> per head h columns [256 h, 256 (h+1))
          for (int h = 0; h < 2; ++h) {
            if (!kv_quant) {
              plane_.Check(plane_.memcpy2d_d2d(key + h * head_stride + lo * 512, 512, src + h * 512, 1024, 512, m, stream_), "head key");
            } else {
              plane_.quantize_kv(stream_, src + h * 512, key + h * head_stride + lo * 256,
                                 head_kscale + h * 256 * sizeof(float), static_cast<int>(m), 512, 256, kv_mode);
            }
          }
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
        for (int h = 0; h < 2; ++h) {
          if (!kv_quant) {
            plane_.Check(plane_.memcpy2d_d2d(value + h * head_stride + lo * 512, 512, src + h * 512, 1024, 512, m, stream_), "head value");
          } else {
            plane_.quantize_kv(stream_, src + h * 512, value + h * head_stride + lo * 256,
                               head_vscale + h * 256 * sizeof(float), static_cast<int>(m), 512, 256, kv_mode);
          }
        }
      } else if (names[i] == "index_key") {
        if (shape.size() != 3 || shape[1] != int64_t(m) || shape[2] != 128)
          throw std::runtime_error("qwen38_graph: unexpected head_kv index_key shape");
        plane_.Check(plane_.memcpy_d2d(f.raw_keys.data() + lo * 256, src, m * 256, stream_), "head raw keys");
      }
    }
    SyncStream();  // the output values die here
  }
  const size_t blocks = rows / kRatio, pending = rows % kRatio;
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
}

}  // namespace Generators
