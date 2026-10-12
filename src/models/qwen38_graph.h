// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Qwen3.8-Flash-Next NVFP4: CUDA-graph decode stack + MTP(1) self-speculative loop (model.type "qwen38_flash_graph").
// See DESIGN.md of the integration run for the architecture; the numerics are those of the research runner
// (CG/mtp_runner: S2Verifier + DeviceHead + libmtp_plane.so), only the orchestration is C++.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "model.h"

namespace Generators {

struct Qwen38GraphModel;

// C ABI of libqwen38_plane.so (CUDA data plane + CUDA runtime helpers), loaded with dlopen so that the core library does
// not link cudart.
struct Qwen38Plane {
  explicit Qwen38Plane(const std::string& path);
  ~Qwen38Plane();
  Qwen38Plane(const Qwen38Plane&) = delete;
  Qwen38Plane& operator=(const Qwen38Plane&) = delete;

  void Check(int error, const char* what) const;

  void* handle{};
  int (*sizeof_params)();
  int (*set_device)(int);
  const char* (*error_string)(int);
  int (*malloc_)(void**, size_t);
  int (*free_)(void*);
  int (*host_alloc)(void**, size_t);
  int (*host_free)(void*);
  int (*host_register)(void*, size_t, unsigned);
  int (*host_unregister)(void*);
  int (*memset_)(void*, int, size_t, void*);
  int (*memcpy_h2d)(void*, const void*, size_t, void*);
  int (*memcpy_d2h)(void*, const void*, size_t, void*);
  int (*memcpy_d2d)(void*, const void*, size_t, void*);
  int (*stream_sync)(void*);
  void (*prep)(void*, const void*, const void*, const void*, int, const void*, void*, void*, void*, void*, void*, void*,
               void*);
  void (*argmax_part)(void*, const void*, int, void*, void*, int);
  void (*post_tok)(void*, const void*, void*, void*, const void*, const void*, const void*, int, void*, int);
  void (*post1)(void*, const void*, const void*, int, void*, void*, const void*, const void*, int, void*, int);
  void (*head_prep)(void*, const void*, int, int, const void*, void*, void*, void*, void*);
  void (*head_post)(void*, const void*, void*);
  void (*or_status)(void*, const void*, int, void*);
};

// Device allocation owned by RAII.
struct PlaneBuffer {
  PlaneBuffer() = default;
  PlaneBuffer(Qwen38Plane& plane, size_t bytes, void* stream);  // zero-initialised
  ~PlaneBuffer();
  PlaneBuffer(PlaneBuffer&& other) noexcept;
  PlaneBuffer& operator=(PlaneBuffer&& other) noexcept;
  PlaneBuffer(const PlaneBuffer&) = delete;
  PlaneBuffer& operator=(const PlaneBuffer&) = delete;

  uint8_t* data() const { return ptr_; }
  size_t size() const { return bytes_; }

 private:
  Qwen38Plane* plane_{};
  uint8_t* ptr_{};
  size_t bytes_{};
};

// A fixed-address device tensor plus the OrtValue that names it in IO bindings.
struct DevTensor {
  uint8_t* ptr{};
  std::vector<int64_t> shape;
  ONNXTensorElementDataType type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::unique_ptr<OrtValue> value;
  size_t Bytes() const;
};

// Host-registered FP8 PLE shards (see ple_table.txt written by tools/qwen38_graph/build_package.py).
struct Qwen38PleParams {
  int n_heads, head_dim, heads_per_ngram, n_shards;
  long long mult[4];
  long long vocab[32];
  long long offset[32];
  long long shard_end[256];
  const unsigned char* shard_ptr[256];
  float scale;
  int pad_;
  long long eos;
  long long capacity;
};

struct Qwen38PleTable {
  Qwen38PleTable(Qwen38Plane& plane, const std::string& table_path, const std::string& root, int capacity);
  ~Qwen38PleTable();
  Qwen38PleTable(const Qwen38PleTable&) = delete;
  Qwen38PleTable& operator=(const Qwen38PleTable&) = delete;

  Qwen38PleParams params{};
  double register_seconds{};

 private:
  struct Mapping {
    void* base{};
    size_t length{};
    void* registered{};
  };
  Qwen38Plane& plane_;
  std::vector<Mapping> mappings_;
};

struct Qwen38CycleResult {
  int64_t pred0{};
  int64_t pred1{};
  bool accepted{};
  int64_t status{};
  int64_t new_pending{};
  int64_t new_length{};
};

struct Qwen38GraphStats {
  size_t cycles{};
  size_t accepted{};
  size_t tokens{};
};

// Owns every device-side object of one model: target + head bindings, state banks, KV store, control block, PLE table.
// One generator at a time may lease it; graphs stay captured across generators.
struct Qwen38GraphEngine {
  explicit Qwen38GraphEngine(Qwen38GraphModel& model);
  ~Qwen38GraphEngine();

  bool TryAcquire() { return !leased_.exchange(true); }
  void Release() { leased_.store(false); }

  // --- prompt ingestion (pluggable: the single entry point a faster prefill replaces) ---
  void Reset();                                          // zero all state, length 0
  void PrefillChunk(std::span<const int32_t> tokens);    // S=2 steps; leaves the argmax of the last row in PendingToken()
  void LastLogits(std::span<float> out) const;           // fp32 copy of the logits row of the last prompt token
  size_t Length() const { return length_; }
  int32_t PendingToken() const { return pending_; }
  bool DecodeStarted() const { return decode_started_; }

  // --- decode ---
  bool mtp_enabled() const { return mtp_enabled_; }
  // Prepares the loop after the prompt (head prefill, first draft) and enqueues the first cycle unless `run_cycles` is false.
  void BeginDecode(bool run_cycles);
  Qwen38CycleResult WaitCycle();                         // spin on the pinned record of the cycle in flight
  // Commits the cycle on the host and, if `want_more`, enqueues head + next cycle (look-ahead).
  void Advance(const Qwen38CycleResult& result, bool want_more);
  void Drain();                                          // wait for all enqueued work
  bool CycleInFlight() const { return inflight_; }
  int Capacity() const { return capacity_; }
  const Qwen38GraphStats& Stats() const { return stats_; }

  double warmup_seconds{};
  double ple_register_seconds{};

 private:
  struct Bank {
    PlaneBuffer arena;
    std::map<std::string, std::unique_ptr<DevTensor>> tensors;  // keyed by input name (past_key_values.*)
  };
  struct Direction {
    std::unique_ptr<OrtIoBinding> binding;
    std::unique_ptr<OrtRunOptions> run_options;
    uint8_t* logits{};
    uint8_t* status{};
    uint8_t* tokens{};
    uint8_t* hidden{};
    size_t status_count{};
  };
  struct RowInputs {  // fixed-address per-row-count inputs of the target
    std::map<std::string, std::unique_ptr<DevTensor>> tensors;
    uint8_t* input_ids{};
    uint8_t* position_ids{};
    uint8_t* committed{};
    uint8_t* ple{};
  };
  struct HeadSession {
    int rows{};
    OrtSession* session{};
    SessionInfo info;
    std::map<std::string, std::unique_ptr<DevTensor>> tensors;  // own inputs/outputs
    uint8_t* tokens{};
    uint8_t* positions{};
    uint8_t* hidden{};
    uint8_t* committed{};
    uint8_t* next_token{};
    uint8_t* status{};
    size_t status_count{};
    Direction dirs[2];
  };

  void BuildTarget();
  void BuildHead();
  std::unique_ptr<DevTensor> MakeTensor(void* ptr, std::vector<int64_t> shape, ONNXTensorElementDataType type);
  DevTensor* NewDeviceTensor(std::map<std::string, std::unique_ptr<DevTensor>>& owner, const std::string& name,
                             std::vector<int64_t> shape, ONNXTensorElementDataType type);
  void Warmup();

  void PrepRow(int rows, int row, const void* tok, const void* pos, int add, const void* ctx_in, void* ctx_out);
  void RunTarget(int rows, int phase);
  void EnqueueVerify();
  void EnqueueStep();
  void EnqueueHead(int rows, const uint8_t* hidden_src);
  void HeadPrefill();
  void RunHeadOnce(HeadSession& head, int rows);
  void ClearHead();
  void CheckDeviceStatus(const char* where, const Direction* dir);
  int64_t ReadCtl(int index);
  void WriteCtl(int first, std::span<const int64_t> values);
  void SyncStream();

  Qwen38GraphModel& model_;
  Qwen38Plane& plane_;
  void* stream_{};
  std::unique_ptr<OrtMemoryInfo> memory_info_;
  const int capacity_;
  const int vocab_;
  int hidden_width_{};
  int ple_width_{};
  int ple_layer_{};
  size_t status_count_{};
  std::vector<std::string> status_labels_;
  bool mtp_enabled_{};
  bool prefill_head_{};
  bool skip_refeed_{};
  static constexpr int kBanks = 3;
  static constexpr int kRing = 64;
  static constexpr int kArgmaxBlocks = 16;

  // target state
  std::map<std::string, std::unique_ptr<DevTensor>> store_;   // in-place KV / positions
  std::map<std::string, std::unique_ptr<DevTensor>> shared_;   // index_key buffers
  Bank banks_[kBanks + 1];                                     // last = scratch sink for S=1 present_mid
  RowInputs inputs_[2];
  Direction dirs_[2][kBanks];                                  // [rows-1][phase]
  std::vector<std::unique_ptr<DevTensor>> output_tensors_;
  std::vector<std::unique_ptr<DevTensor>> head_views_;
  std::deque<PlaneBuffer> owned_;                              // individually allocated tensors
  const Direction* inflight_dir_{};

  // control block / plane buffers
  PlaneBuffer ctl_, ctx_commit_, ctx_a_, ctx_b_, scratch_i32_, part_v_, part_i_, params_dev_;
  PlaneBuffer d_tokens_, d_arange64_, d_arange32_, d_prompt_hidden_, d_last_hidden_;
  int64_t* result_host_{};
  int64_t* stage_host_{};
  std::unique_ptr<Qwen38PleTable> ple_;
  uint8_t* past_pos_{};

  // head
  HeadSession head_[2];  // [0] = S=1, [1] = S=2
  std::map<std::string, std::unique_ptr<DevTensor>> head_store_;
  struct HeadBank {
    PlaneBuffer arena;
    std::map<std::string, std::unique_ptr<DevTensor>> tensors;
  } head_banks_[2];
  int head_length_{};
  int head_turn_{};

  // host state
  int cur_{};
  size_t length_{};
  uint64_t seq_{};
  bool inflight_{};
  bool decode_started_{};
  bool keep_hidden_{};
  int32_t pending_{};
  std::vector<int32_t> prompt_tokens_;
  const uint8_t* last_logits_ptr_{};
  int last_logits_row_{};
  const uint8_t* last_hidden_ptr_{};  // hidden output of the last cycle (rows h0, h1)
  Qwen38GraphStats stats_;
  std::atomic<bool> leased_{false};
};

struct Qwen38GraphModel : Model {
  Qwen38GraphModel(std::unique_ptr<Config> config, OrtEnv& ort_env);
  ~Qwen38GraphModel() override;

  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  std::unique_ptr<OrtSession> session_target_;
  std::unique_ptr<OrtSessionOptions> head_session_options_;
  std::unique_ptr<OrtSession> session_head_s1_, session_head_s2_;
  std::unique_ptr<Qwen38Plane> plane_;
  std::unique_ptr<Qwen38GraphEngine> engine_;
};

struct Qwen38GraphState : State {
  Qwen38GraphState(const Qwen38GraphModel& model, const GeneratorParams& params);
  ~Qwen38GraphState() override;

  DeviceSpan<float> Run(int total_length, DeviceSpan<int32_t>& next_tokens, DeviceSpan<int32_t> next_indices) override;
  void Finalize(int current_length) override;
  void RewindTo(size_t index) override;
  void SnapshotState(size_t position) override;

  Qwen38GraphEngine& engine() { return *engine_; }

  const Qwen38GraphModel& model_;
  Qwen38GraphEngine* engine_{};
};

}  // namespace Generators
