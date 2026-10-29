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
  // CPU resource-lifetime tests inject callbacks instead of loading CUDA.
  Qwen38Plane() = default;
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
  // fast prefill (optional: nullptr in planes built before v12)
  int (*memcpy2d_d2d)(void*, size_t, const void*, size_t, size_t, size_t, void*){};
  int (*stream_create)(void**){};
  int (*stream_destroy)(void*){};
  void (*ple_rows)(void*, const void*, const void*, int, const void*, void*, void*){};
  void (*pool_keys)(void*, const void*, int, const void*, const void*, const void*, int, float, int, void*){};
  void (*transpose_f32)(void*, const void*, void*, int, int){};
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
    explicit Mapping(Qwen38Plane& plane) : plane{plane} {}
    ~Mapping();
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    Qwen38Plane& plane;
    void* base{};
    size_t length{};
    void* registered{};
  };
  std::vector<std::unique_ptr<Mapping>> mappings_;
};

// Device copy of every initializer that several sessions of the stack have in common (decode/fold, head, dense prefill,
// head KV prefill): loaded once from the files of a text manifest (tools/qwen38_graph/build_package_v12.py), handed to each
// session through AddInitializer (ORT uses a user supplied initializer in place when it is on the session's device).
struct Qwen38WeightPool {
  Qwen38WeightPool(Qwen38Plane& plane, const std::string& manifest_path, const std::string& root,
                   const OrtMemoryInfo& memory_info, int threads = 8);
  ~Qwen38WeightPool();
  Qwen38WeightPool(const Qwen38WeightPool&) = delete;
  Qwen38WeightPool& operator=(const Qwen38WeightPool&) = delete;

  // Adds the shared initializers of one session label ("target", "head_r2", "dense", "head_kv", ...) to its options.
  size_t AddTo(OrtSessionOptions& options, const std::string& label) const;
  bool HasLabel(const std::string& label) const { return sessions_.count(label) != 0; }
  size_t bytes{};
  double seconds{};

 private:
  struct Entry {
    int dtype{};
    uint64_t offset{};
    uint64_t length{};
    std::vector<int64_t> shape;
    std::string file;
    std::unique_ptr<uint8_t, int (*)(void*)> buffer{nullptr, nullptr};
    std::unique_ptr<OrtValue> value;
  };
  Qwen38Plane& plane_;
  std::vector<Entry> entries_;
  std::map<std::string, std::vector<std::pair<std::string, int>>> sessions_;
};

// Constants of the PackedQSAIndexer state conversion (key-norm weights, leading-RoPE tables) for the dense -> packed hand-off.
struct Qwen38QsaConstants {
  struct Set {
    std::map<int, const uint8_t*> knorm;  // per QSA layer (bf16 [128])
    const uint8_t* cos{};                 // bf16 [max_pos, width]
    const uint8_t* sin{};
    int width{};
  };
  Qwen38QsaConstants(Qwen38Plane& plane, const std::string& index_path, const std::string& root, void* stream);
  Set target, head;
  float epsilon{};

 private:
  PlaneBuffer buffer_;
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
  void Reset();  // zero all state, length 0
  // Ingests prompt tokens by S=2 graph steps (odd tail commits one row). Contract for a replacement (faster) prefill, per
  // chunk: advance the committed recurrent bank (cur_), the in-place KV/index/position store and the PLE context
  // (ctx_commit_) to length_ + n; append the prompt tokens to d_tokens_ / prompt_tokens_; if keep_hidden_, write the target
  // hidden row of every token to d_prompt_hidden_ (the head is prefilled from it); leave the hidden row of the last token in
  // d_last_hidden_, its argmax in PendingToken() and its logits row reachable through LastLogits().
  void PrefillChunk(std::span<const int32_t> tokens);
  void LastLogits(std::span<float> out) const;  // fp32 copy of the logits row of the last prompt token
  void SaveBufferedLogits();
  void DecodeLogits(bool buffered, std::span<float> out) const;
  size_t Length() const { return length_; }
  int32_t PendingToken() const { return pending_; }
  bool DecodeStarted() const { return decode_started_; }
  bool Fold() const { return fold_; }
  bool FastPrefill() const { return fast_ != nullptr; }
  void CheckErrorBits();  // accumulated device error bits of the whole decode (ctl[5])

  // --- decode ---
  bool mtp_enabled() const { return mtp_enabled_; }
  // Prepares the loop after the prompt (head prefill, first draft) and enqueues the first cycle unless `run_cycles` is false.
  void BeginDecode(bool run_cycles);
  Qwen38CycleResult WaitCycle();  // spin on the pinned record of the cycle in flight
  // Commits the cycle on the host and, if `want_more`, enqueues head + next cycle (look-ahead).
  void Advance(const Qwen38CycleResult& result, bool want_more);
  void Drain();  // wait for all enqueued work
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
  void BuildHeadState();
  void BuildHeadR2();
  void CallHeadR2(const int64_t tokens[2], const int64_t positions[2], const uint8_t* hidden0, const uint8_t* hidden1,
                  int commit_rows);
  void HeadPrefillFold();
  void FoldFirstDraft();
  void EnqueueFold();
  void ForceRejectDraft();
  void BuildFastPrefill();
  void StepPrefill(std::span<const int32_t> tokens);
  void FastAppend(std::span<const int32_t> tokens);
  void FastHandoff();
  void FastHeadKvPrefill();
  void FastReset();
  void MutedPrefillBegin();
  void MutedPrefillEnd();
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
  std::map<std::string, std::unique_ptr<DevTensor>> shared_;  // index_key buffers
  Bank banks_[kBanks + 1];                                    // last = scratch sink for S=1 present_mid
  RowInputs inputs_[2];
  Direction dirs_[2][kBanks];  // [rows-1][phase]
  std::vector<std::unique_ptr<DevTensor>> output_tensors_;
  std::vector<std::unique_ptr<DevTensor>> head_views_;
  std::deque<PlaneBuffer> owned_;  // individually allocated tensors
  const Direction* inflight_dir_{};

  // control block / plane buffers
  PlaneBuffer ctl_, ctx_commit_, ctx_a_, ctx_b_, scratch_i32_, part_v_, part_i_, params_dev_;
  PlaneBuffer d_tokens_, d_arange64_, d_arange32_, d_prompt_hidden_, d_last_hidden_;
  PlaneBuffer buffered_logits_;
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
  } head_banks_[3];  // two ping-pong banks (S=1/S=2 head sessions) or the three ring phases (fold)
  int head_length_{};
  int head_turn_{};
  int head_cur_{};  // fold: head ring phase

  // fold (v12): the target session is the fold model (verify + head + glue); the standalone R=2 head session serves the
  // prompt context (pair calls) and the first draft
  bool fold_{};
  bool force_reject_{};  // fold test hook (config fold.force_reject)
  struct HeadR2Session {
    OrtSession* session{};
    SessionInfo info;
    std::map<std::string, std::unique_ptr<DevTensor>> tensors;
    uint8_t* tokens{};
    uint8_t* positions{};
    uint8_t* hidden{};
    uint8_t* committed{};
    uint8_t* next_token{};
    uint8_t* status{};
    size_t status_count{};
    Direction dirs[3];
  } head_r2_;
  std::map<std::string, std::unique_ptr<DevTensor>> fold_tensors_;  // fold.* inputs (device control block)
  PlaneBuffer fold_ptrs_;

  // fast prefill (dense session + hand-off); null when not configured
  struct FastPrefillState {
    OrtSession* dense{};
    SessionInfo dense_info;
    std::vector<std::string> dense_inputs, dense_outputs;
    OrtSession* head_kv{};
    SessionInfo head_kv_info;
    int chunk{8192};
    Qwen38QsaConstants* consts{};

    PlaneBuffer ones, idx, ples, raw_keys, scratch;  // ones: attention mask; idx: logits_indices; ples: PLE rows of a chunk
    std::deque<PlaneBuffer> zero_buffers;
    std::map<std::string, std::unique_ptr<OrtValue>> initial;  // zero-length / zero initial states of the dense model
    std::map<std::string, std::unique_ptr<OrtValue>> states;   // dense `past` inputs (outputs of the previous chunk)
    size_t dense_length{};
    std::unique_ptr<OrtValue> logits;
    std::unique_ptr<OrtRunOptions> run_options;

    struct Timings {
      double dense{}, handoff{}, head{};
    } timings;
  };
  std::unique_ptr<FastPrefillState> fast_;
  std::vector<float> last_logits_host_;

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

  // Declaration order = reverse destruction order: the sessions go first, then the shared weights / constants they were given,
  // then the plane library.
  std::unique_ptr<Qwen38Plane> plane_;
  std::unique_ptr<Qwen38WeightPool> pool_;
  std::unique_ptr<Qwen38QsaConstants> qsa_constants_;
  std::unique_ptr<OrtSession> session_target_;
  std::unique_ptr<OrtSessionOptions> head_session_options_;
  std::unique_ptr<OrtSession> session_head_s1_, session_head_s2_;
  std::unique_ptr<OrtSessionOptions> head_r2_session_options_, dense_session_options_, head_kv_session_options_;
  std::unique_ptr<OrtSession> session_head_r2_, session_dense_, session_head_kv_;
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
