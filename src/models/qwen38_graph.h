// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Qwen3.8-Flash-Next NVFP4: CUDA-graph decode stack + MTP(1) self-speculative loop (model.type "qwen38_flash_graph").
//
// Qwen38GraphModel owns the ORT sessions (target and MTP head) and the CUDA data plane; Qwen38GraphEngine owns every
// device-side object built on them (recurrent bank ring, in-place KV store, control block, PLE table, IO bindings of the
// captured graphs) and implements Qwen38DecodeDevice, the interface the Engine executor drives
// (src/engine/qwen38_graph_executor.h). The model is served by the Engine API only; there is no Generator state.
#pragma once

#include <atomic>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "model.h"
#include "qwen38_decode_device.h"
#include "qwen38_graph_plane.h"
#include "span.h"

namespace Generators {

struct Qwen38GraphModel;

// A fixed-address device tensor plus the OrtValue that names it in IO bindings.
struct DevTensor {
  uint8_t* ptr{};
  std::vector<int64_t> shape;
  ONNXTensorElementDataType type{ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED};
  std::unique_ptr<OrtValue> value;
  size_t Bytes() const;
};

// Device copy of every initializer that several sessions of the stack have in common (decode/fold, head, dense prefill,
// head KV prefill): loaded once from the files of a text manifest (tools/qwen38_graph/build_package_v12b.py), handed to each
// session through AddInitializer (ORT uses a user supplied initializer in place when it is on the session's device).
struct Qwen38WeightPool {
  Qwen38WeightPool(Qwen38Plane& plane, const std::filesystem::path& manifest_path, const std::filesystem::path& root,
                   const OrtMemoryInfo& memory_info, int threads = 8);
  ~Qwen38WeightPool();
  Qwen38WeightPool(const Qwen38WeightPool&) = delete;
  Qwen38WeightPool& operator=(const Qwen38WeightPool&) = delete;

  // Adds the shared initializers of one session label ("target", "head_r2", "dense", "head_kv", ...) to its options.
  size_t AddTo(OrtSessionOptions& options, const std::string& label) const;
  size_t bytes{};
  double seconds{};

 private:
  struct Entry {
    int dtype{};
    uint64_t offset{};
    uint64_t length{};
    std::vector<int64_t> shape;
    std::filesystem::path file;
    uint8_t* ptr{};
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
  Qwen38QsaConstants(Qwen38Plane& plane, const std::filesystem::path& index_path, const std::filesystem::path& root,
                     void* stream);
  Set target, head;
  float epsilon{};

 private:
  PlaneBuffer buffer_;
};

// Owns every device-side object of one model. Graphs stay captured across Engines (see Qwen38GraphModel::AcquireDevice).
struct Qwen38GraphEngine final : Qwen38DecodeDevice {
  explicit Qwen38GraphEngine(Qwen38GraphModel& model);
  ~Qwen38GraphEngine() override;

  int Capacity() const override { return capacity_; }
  bool MtpEnabled() const override { return mtp_enabled_; }
  void Reset() override;
  void Drain() noexcept override;

  // Ingests prompt tokens: by the dense chunked prefill session when prefill.enabled (FastAppend: the dense state continues
  // across chunks and is handed over to the decode state in BeginDecode), otherwise by S=2 graph steps (StepPrefill; the
  // fold graph runs muted, an odd tail commits one row). Per chunk both leave the prompt tokens in d_tokens_ /
  // prompt_tokens_, the target hidden row of every token in d_prompt_hidden_ (the head is prefilled from it), the hidden row
  // of the last token in d_last_hidden_ and its argmax in PendingToken().
  void PrefillChunk(std::span<const int32_t> tokens) override;
  int32_t PendingToken() const override { return pending_; }
  size_t PrefillChunkTokens() const override;

  void BeginDecode(bool run_cycle) override;
  Qwen38CycleRecord WaitCycle() override;
  void Advance(const Qwen38CycleRecord& record, bool want_more) override;
  void CheckDecodeStatus() override;

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
  void BuildFastPrefill();
  void StepPrefill(std::span<const int32_t> tokens);
  void FastAppend(std::span<const int32_t> tokens);
  void FastHandoff();
  void FastHeadKvPrefill();
  void FastReset();
  void MutedPrefill(bool muted);
  void CallHeadR2(const int64_t tokens[2], const int64_t positions[2], const uint8_t* hidden0, const uint8_t* hidden1,
                  int commit_rows);
  void HeadPrefillFold();
  void FoldFirstDraft();
  void ForceRejectDraft();
  void EnqueueFold();
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
  bool fold_{};
  bool force_reject_{};  // test hook (fold.force_reject): every draft is replaced before its verify
  static constexpr int kBanks = Qwen38DecodeLedger::kBanks;
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
  PlaneHostBuffer result_ring_;  // pinned: the device publishes one 8-word record per cycle, the host spins on word 0
  PlaneHostBuffer stage_;        // pinned staging words for control-block reads/writes
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

  // fold: the decoder session is the fold model (verify + head + glue, one graph launch per cycle); the standalone R=2 head
  // session serves the prompt context (pair calls) and the first draft
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
  std::map<std::string, std::unique_ptr<DevTensor>> fold_tensors_;  // fold.* inputs (the device control block)
  PlaneBuffer fold_ptrs_;

  // fast prefill (dense session + hand-off + batched head KV prefill); null when not configured
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
    std::unique_ptr<OrtValue> logits;  // logits row of the last chunk
    std::unique_ptr<OrtRunOptions> run_options;
  };
  std::unique_ptr<FastPrefillState> fast_;

  // host state
  Qwen38DecodeLedger ledger_;
  uint64_t seq_{};
  bool inflight_{};
  bool decode_started_{};
  bool keep_hidden_{};
  int32_t pending_{};
  std::vector<int32_t> prompt_tokens_;
  const uint8_t* last_hidden_ptr_{};  // hidden output of the last cycle (rows h0, h1)
};

struct Qwen38GraphModel : Model {
  Qwen38GraphModel(std::unique_ptr<Config> config, OrtEnv& ort_env);
  ~Qwen38GraphModel() override;

  // The Engine API is the only way to run this model.
  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  // Leases the device engine to one Engine at a time (it owns the captured graphs, the KV store and the bank ring). The
  // lease drains the device when released and keeps the model alive. Throws when another Engine holds it.
  std::shared_ptr<Qwen38DecodeDevice> AcquireDevice();

  // Declaration order = reverse destruction order: the engine and the sessions go first, then the shared weights / constants
  // they were given, then the plane library.
  std::unique_ptr<Qwen38Plane> plane_;
  std::unique_ptr<Qwen38WeightPool> pool_;
  std::unique_ptr<Qwen38QsaConstants> qsa_constants_;
  std::unique_ptr<OrtSession> session_target_;
  std::unique_ptr<OrtSessionOptions> head_session_options_;
  std::unique_ptr<OrtSession> session_head_s1_, session_head_s2_;
  std::unique_ptr<OrtSessionOptions> head_r2_session_options_, dense_session_options_, head_kv_session_options_;
  std::unique_ptr<OrtSession> session_head_r2_, session_dense_, session_head_kv_;
  std::unique_ptr<Qwen38GraphEngine> engine_;

 private:
  std::atomic<bool> leased_{false};
};

}  // namespace Generators
