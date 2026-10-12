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
#include <map>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "model.h"
#include "qwen38_decode_device.h"
#include "qwen38_graph_plane.h"

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

// Owns every device-side object of one model. Graphs stay captured across Engines (see Qwen38GraphModel::AcquireDevice).
struct Qwen38GraphEngine final : Qwen38DecodeDevice {
  explicit Qwen38GraphEngine(Qwen38GraphModel& model);
  ~Qwen38GraphEngine() override;

  int Capacity() const override { return capacity_; }
  bool MtpEnabled() const override { return mtp_enabled_; }
  void Reset() override;
  void Drain() noexcept override;

  // Ingests prompt tokens by S=2 graph steps (an odd tail commits one row). Contract for a replacement (faster) prefill, per
  // chunk: advance the committed recurrent bank (ledger_), the in-place KV/index/position store and the PLE context
  // (ctx_commit_) to length + n; append the prompt tokens to d_tokens_ / prompt_tokens_; if keep_hidden_, write the target
  // hidden row of every token to d_prompt_hidden_ (the head is prefilled from it); leave the hidden row of the last token in
  // d_last_hidden_ and its argmax in PendingToken().
  void PrefillChunk(std::span<const int32_t> tokens) override;
  int32_t PendingToken() const override { return pending_; }

  void BeginDecode(bool run_cycle) override;
  Qwen38CycleRecord WaitCycle() override;
  void Advance(const Qwen38CycleRecord& record, bool want_more) override;

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
  void RunHeadOnce(HeadSession& head);
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
  } head_banks_[2];
  int head_length_{};
  int head_turn_{};

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

  std::unique_ptr<OrtSession> session_target_;
  std::unique_ptr<OrtSessionOptions> head_session_options_;
  std::unique_ptr<OrtSession> session_head_s1_, session_head_s2_;
  std::unique_ptr<Qwen38Plane> plane_;
  std::unique_ptr<Qwen38GraphEngine> engine_;

 private:
  std::atomic<bool> leased_{false};
};

}  // namespace Generators
