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

#include <array>
#include <atomic>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "engine/backend_capabilities.h"
#include "model.h"
#include "../dflash2_drafter.h"
#include "qwen38_decode_device.h"
#include "qwen38_dflash_prompt.h"
#include "qwen38_graph_plane.h"
#include "qwen38_ort_env.h"
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
  size_t cpu_bytes{};
  size_t host_mapped_bytes{};
  double seconds{};

 private:
  void Cleanup() noexcept;
  struct Entry {
    int dtype{};
    uint64_t offset{};
    uint64_t length{};
    std::vector<int64_t> shape;
    std::filesystem::path file;
    std::string placement{"device"};
    bool host_mapped{false};
    bool owns_device_ptr{false};
    uint8_t* ptr{};
    std::unique_ptr<OrtValue> value;
    std::unique_ptr<OrtValue> gpu_mapped_value;
  };
  Qwen38Plane& plane_;
  std::unique_ptr<OrtMemoryInfo> cpu_memory_info_;
  std::map<std::pair<std::filesystem::path, bool>, std::unique_ptr<Qwen38FileMapping>> file_mappings_;
  std::vector<void*> registered_ranges_;
  std::vector<Entry> entries_;
  std::map<std::string, std::vector<std::pair<std::string, int>>> sessions_;
};

// Resolves auxiliary session options for head/prefill sessions; when `inherit` is false (dense, head_kv), inherits the
// parent's provider selection and device options while stripping graph-capture options.
Config::SessionOptions ResolveQwen38AuxiliarySessionOptions(const Config::SessionOptions& parent,
                                                            const std::optional<Config::SessionOptions>& configured,
                                                            bool inherit);

// Returns the effective prompt chunk size of a Qwen38Graph configuration: prefill.chunk, or when that is 0 the default
// of the prompt path in use (2048 tokens per Engine step through the decode graphs, 8192 per dense chunk when
// prefill.enabled).
size_t EffectiveQwen38PrefillChunkTokens(const Config::Model::Qwen38Graph& q);

// Throws a std::runtime_error unless the effective prompt chunk lies in [1, kv_capacity] (dense prefill) or, when the
// prompt is ingested by the decode graphs (prefill.enabled false), is even so that S=2 pairs stay aligned.
void ValidateQwen38PrefillChunk(const Config::Model::Qwen38Graph& q);

// Checks a model package against the plane library it names, when the model is loaded, before its weights are uploaded and
// before any session is created: an INT8 PLE table (model.qwen38_graph.ple_table with a negative scale) needs the int8_ple
// plane capability, an INT8 embedding table (<model.embedding.filename>.data of vocab_size x hidden_size INT8 rows and
// one float scale per row) needs int8_embedding, and the folded MTP verify path (model.qwen38_graph.fold.enabled), which
// prepares all the rows of a verify step with one call, needs negative_pos_add_batch (qwen38_graph_plane.h). Throws a
// std::runtime_error naming the missing capability, what required it and the library when the library lacks one; BF16 and
// FP8 packages without the fold path need no capability. Returns the layout of the embedding table the model maps from that
// file: kNone when the package has no table of that shape.
Qwen38EmbeddingFormat RequireQwen38PackageSupportedByPlane(const Qwen38Plane& plane, const Config& config,
                                                           const std::filesystem::path& model_folder);

// The CUDA graphs of the decode stack. ORT keeps one captured graph per annotation id in a session, so every run that
// binds different buffers needs an id of its own: one per (row count, ring phase) for the target (decode graphs or the
// fold model), one per (row count, ring phase, number of request slots) for the batched runs of the fold target, one per
// (row count, ping-pong direction) for the S=1/S=2 head sessions and one per ring phase for the standalone R=2 head. The ids
// are derived from the role of the run, they are not configurable, and they are distinct across the sessions of a model.
// 0 is ORT's default annotation and -1 means "do not capture"; neither is produced.
// The DFlash2 drafter is a session of its own that numbers the shapes it captures itself (engine/graph_annotation_ids.h),
// so it takes none of these ids; the S=8 (seven drafts) target of its folded pipeline is a kTarget run of 8 rows.
enum class Qwen38GraphRun { kTarget,
                            kHead,
                            kHeadR2 };
inline constexpr int kQwen38GraphRunCount = 3;     // the enumerators above
inline constexpr int kQwen38GraphIdMaxRows = 8;    // verify rows of the target (S=1 .. S=8)
inline constexpr int kQwen38GraphIdMaxPhases = 8;  // ring banks / ping-pong directions of any decode mode
inline constexpr int kQwen38GraphIdMaxBatch = 8;   // request slots of one batched target run (B = 1 .. 8)
static_assert(static_cast<int>(Qwen38GraphRun::kHeadR2) + 1 == kQwen38GraphRunCount,
              "kQwen38GraphRunCount must count every Qwen38GraphRun");
// `batch` is the number of request slots the run serves: only the fold target runs batched (batch > 1), every other run is
// one slot. The ids of batch 1 do not depend on the batch dimension. Throws std::out_of_range for a run, row count, phase or
// batch beyond the limits above, or a batch above 1 for a run other than kTarget.
int Qwen38GraphAnnotationId(Qwen38GraphRun run, int rows, int phase, int batch = 1);

// ORT captures a graph after a few eager runs of it: the warm-up of a graph runs it this many times, to be safe.
inline constexpr int kQwen38WarmupRunsPerGraph = 6;

// What the DFlash engine runs while its model is created. model.qwen38_graph.warmup=false opts out of all of it, as for the
// other paths: no forward, hence no CUDA graph captured and no startup cost, for the drafter or for the target. With
// warmup=true the drafter ingests a warm-up prompt and drafts every row width (`drafter`) and `target_tokens` dummy prompt
// tokens step through the S=verify_rows target graphs, enough runs of every (rows, ring bank) graph to capture it.
struct Qwen38DflashWarmupPlan {
  bool drafter{};
  size_t target_tokens{};
};
Qwen38DflashWarmupPlan PlanQwen38DflashWarmup(bool warmup, int verify_rows, int num_banks) noexcept;

// Rows of the DFlash prompt-feature buffer that one request slot owns: a prompt chunk of `chunk` rows, but at least the
// `verify_rows` of a verification step, because the warm-up of the drafter views that many rows from the same buffer
// (a prefill.chunk below the verify width would otherwise make it read past the allocation). Throws std::invalid_argument
// unless both are positive.
size_t Qwen38DflashPromptSlotRows(int chunk, int verify_rows);

// The packed QSA indexer pools kQwen38IndexKeyRatio tokens into one compressed index key.
inline constexpr int kQwen38IndexKeyRatio = 4;

// A cache input of an exported graph as the graph declares it (dimensions <= 0 are symbolic).
struct Qwen38CacheInput {
  std::string name;
  std::vector<int64_t> shape;
};

// The engine allocates every KV / position / index-key buffer from the shapes the graphs declare, but its ledger and
// the strides of the dense hand-off and of the head KV prefill come from model.qwen38_graph.kv_capacity. Throws a
// std::runtime_error naming the offending input of `graph` ("target", "head", ...) unless the two agree:
//   past_key_values.<n>.key|value (also head.past_key_values.<n>.*)  [batch, 2, kv_capacity, 256]
//   past_position_ids                                                [batch, >= kv_capacity]
//   past_key_values.<n>.index_key                                    [batch, >= kv_capacity / ratio, width]
// where ratio is kQwen38IndexKeyRatio. Other inputs (recurrent state, index buffers, ...) do not depend on the
// capacity and are ignored.
void ValidateQwen38CacheShapes(const std::vector<Qwen38CacheInput>& inputs, int kv_capacity, const std::string& graph);

// Constants of the PackedQSAIndexer state conversion (key-norm weights, leading-RoPE tables) for the dense -> packed hand-off.
struct Qwen38QsaConstants {
  // fp32 scales per quantized-KV layer: one per channel of the 2 KV heads x 256 channels.
  static constexpr size_t kKvScaleCount = 512;
  struct Set {
    std::map<int, const uint8_t*> knorm;   // per QSA layer (bf16 [128])
    std::map<int, const uint8_t*> kscale;  // per QSA layer (fp32 [kKvScaleCount], optional for INT8/FP8 KV)
    std::map<int, const uint8_t*> vscale;  // per QSA layer (fp32 [kKvScaleCount], optional for INT8/FP8 KV)
    const uint8_t* cos{};                  // bf16 [max_pos, width]
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

// Prompt & output n-gram suffix lookup drafter for multi-draft / hybrid MTP+lookup speculation.
struct Qwen38SuffixDrafter {
  static constexpr size_t kTableSize = 8192;
  static constexpr size_t kTableMask = kTableSize - 1;

  Qwen38SuffixDrafter() { Reset(); }

  void Reset() noexcept {
    tokens_.clear();
    ngram3_pos_.fill(-1);
    ngram3_key_.fill(0);
    ngram2_pos_.fill(-1);
    ngram2_key_.fill(0);
  }

  void AppendToken(int32_t tok) {
    const int32_t next_pos = static_cast<int32_t>(tokens_.size());
    if (tokens_.size() >= 2) {
      const uint64_t k2 = Pack2(tokens_[tokens_.size() - 2], tokens_[tokens_.size() - 1]);
      const size_t slot2 = Hash64(k2) & kTableMask;
      ngram2_key_[slot2] = k2;
      ngram2_pos_[slot2] = next_pos;
    }
    if (tokens_.size() >= 3) {
      const uint64_t k3 = Pack3(tokens_[tokens_.size() - 3], tokens_[tokens_.size() - 2], tokens_[tokens_.size() - 1]);
      const size_t slot3 = Hash64(k3) & kTableMask;
      ngram3_key_[slot3] = k3;
      ngram3_pos_[slot3] = next_pos;
    }
    tokens_.push_back(tok);
  }

  void AppendSpan(std::span<const int32_t> span) {
    tokens_.reserve(tokens_.size() + span.size());
    for (int32_t t : span) AppendToken(t);
  }

  int ProposeDrafts(int32_t pending_token, int max_drafts, int32_t* out_drafts, int min_ngram = 3) const noexcept {
    if (max_drafts <= 0 || out_drafts == nullptr || tokens_.empty()) return 0;
    int32_t match_pos = -1;
    if (tokens_.size() >= 2) {
      const uint64_t k3 = Pack3(tokens_[tokens_.size() - 2], tokens_[tokens_.size() - 1], pending_token);
      const size_t slot3 = Hash64(k3) & kTableMask;
      if (ngram3_pos_[slot3] >= 0 && ngram3_key_[slot3] == k3) {
        match_pos = ngram3_pos_[slot3];
      }
    }
    if (match_pos < 0 && min_ngram <= 2 && !tokens_.empty()) {
      const uint64_t k2 = Pack2(tokens_[tokens_.size() - 1], pending_token);
      const size_t slot2 = Hash64(k2) & kTableMask;
      if (ngram2_pos_[slot2] >= 0 && ngram2_key_[slot2] == k2) {
        match_pos = ngram2_pos_[slot2];
      }
    }
    if (match_pos < 0 || static_cast<size_t>(match_pos) >= tokens_.size()) return 0;
    int count = 0;
    while (count < max_drafts && static_cast<size_t>(match_pos + count) < tokens_.size()) {
      out_drafts[count] = tokens_[match_pos + count];
      ++count;
    }
    return count;
  }

 private:
  static uint64_t Pack2(int32_t a, int32_t b) noexcept {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a)) << 32) | static_cast<uint32_t>(b);
  }
  static uint64_t Pack3(int32_t a, int32_t b, int32_t c) noexcept {
    return (static_cast<uint64_t>(static_cast<uint32_t>(a) & 0x1FFFFFu) << 42) |
           (static_cast<uint64_t>(static_cast<uint32_t>(b) & 0x1FFFFFu) << 21) |
           static_cast<uint64_t>(static_cast<uint32_t>(c) & 0x1FFFFFu);
  }
  static size_t Hash64(uint64_t x) noexcept {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return static_cast<size_t>(x);
  }

  std::vector<int32_t> tokens_;
  std::array<int32_t, kTableSize> ngram3_pos_{};
  std::array<uint64_t, kTableSize> ngram3_key_{};
  std::array<int32_t, kTableSize> ngram2_pos_{};
  std::array<uint64_t, kTableSize> ngram2_key_{};
};

// Owns every device-side object of one model. Graphs stay captured across Engines (see Qwen38GraphModel::AcquireDevice).
size_t ResolveQwen38MaxBatchSize(const Config& config);

struct Qwen38GraphEngine final : Qwen38DecodeDevice {
  explicit Qwen38GraphEngine(Qwen38GraphModel& model);
  ~Qwen38GraphEngine() override;

  int Capacity() const override { return capacity_; }
  int ReservedTail() const override { return std::max(verify_rows_, kReservedTail); }
  bool MtpEnabled() const override { return mtp_enabled_; }
  size_t MaxBatchSize() const override { return static_cast<size_t>(max_batch_); }
  bool SupportsDynamicBatching() const override { return true; }
  void Reset() override;
  void Drain() noexcept override;
  void MarkContextFailed() noexcept override;
  bool ContextIsHealthy() const noexcept override;

  // Ingests prompt tokens: by the dense chunked prefill session when prefill.enabled (FastAppend: the dense state continues
  // across chunks and is handed over to the decode state in BeginDecode), otherwise by S=2 graph steps (StepPrefill; the
  // fold graph runs muted, an odd tail commits one row). Per chunk both leave the prompt tokens in d_tokens_ /
  // prompt_tokens_, the target hidden row of every token in d_prompt_hidden_ (the head is prefilled from it), the hidden row
  // of the last token in d_last_hidden_ and its argmax in PendingToken().
  void PrefillChunk(std::span<const int32_t> tokens) override;
  int32_t PendingToken() const override { return pending_; }
  size_t PrefillChunkTokens() const override;
  void ExpectPrefillTokens(size_t total_tokens) override { run_hints_.ExpectPrefillTokens(total_tokens); }

  void BeginDecode(bool run_cycle) override;
  Qwen38CycleRecord WaitCycle() override;
  void Advance(const Qwen38CycleRecord& record, bool want_more) override;
  void CheckDecodeStatus() override;

  void ResetSlot(size_t slot) override;
  void PrefillChunkSlot(size_t slot, std::span<const int32_t> tokens) override;
  int32_t PendingTokenSlot(size_t slot) const override;
  void BeginDecodeSlot(size_t slot, bool want_more) override;
  void DraftDeferredSlots(std::span<const size_t> active_slots) override;
  void EnqueueBatchDecode(std::span<const size_t> active_slots) override;
  void WaitBatchDecode(std::span<const size_t> active_slots,
                       std::span<Qwen38CycleRecord> records) override;
  void AdvanceBatchDecode(std::span<const size_t> active_slots,
                          std::span<const Qwen38CycleRecord> records,
                          std::span<const uint8_t> want_more,
                          bool defer_next_cycle = false) override;

  double warmup_seconds{};
  double ple_register_seconds{};

 private:
  static constexpr int kMaxBatch = 8;

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
    uint8_t* auxiliary{};
  };
  struct RowInputs {  // fixed-address per-row-count inputs of the target
    std::map<std::string, std::unique_ptr<DevTensor>> tensors;
    uint8_t* input_ids{};
    uint8_t* inputs_embeds{};
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
  void FastAppendSlot(int slot, std::span<const int32_t> tokens);
  void FastHandoff();
  void FastHeadKvPrefill();
  void FastReset();
  void MutedPrefill(bool muted);
  void MutedPrefillSlot(int slot, bool muted);
  void MuteSlot(int slot);
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
  void Draft(const uint8_t* features, size_t rows, size_t position, int32_t anchor, bool propose);
  void DraftSlot(int slot, const uint8_t* features, size_t rows, size_t position, int32_t anchor, bool propose);

  void PrepRow(int rows, int row, const void* tok, const void* pos, int add, const void* ctx_in, void* ctx_out);
  void PrepBatch(int batch);
  void RunTarget(int rows, int phase, const Qwen38PromptRunHints* prompt = nullptr);
  // Every ORT Run of the engine goes through here: it publishes the QMoE hints of run_hints_ for exactly this Run
  // (qwen38_ort_env.h).
  void RunSession(OrtSession& session, const OrtRunOptions* run_options, const OrtIoBinding& binding,
                  const Qwen38PromptRunHints* prompt = nullptr);
  void EnqueueVerify();
  void EnqueueStep();
  void EnqueueHead(int rows, const uint8_t* hidden_src);
  void HeadPrefill();
  void RunHeadOnce(HeadSession& head, int rows);
  void ClearHead();
  void ReplayStateUpdates(int src_bank, size_t kept_count);
  void ReplayStateUpdatesSlot(int src_bank, int slot, size_t kept_count);
  Qwen38CycleRecord WaitCycleSlot(int slot);
  void CheckDeviceStatus(const char* where, const Direction* dir);
  int64_t ReadCtl(int index);
  void WriteCtl(int first, std::span<const int64_t> values);
  int64_t ReadCtlSlot(int slot, int index);
  void WriteCtlSlot(int slot, int first, std::span<const int64_t> values);
  void SyncStream();

  uint8_t* ctl_slot(int slot) const { return ctl_.data() + static_cast<size_t>(slot) * 16 * sizeof(int64_t); }
  uint8_t* ctx_commit_slot(int slot) const { return ctx_commit_.data() + static_cast<size_t>(slot) * 2 * sizeof(int64_t); }
  uint8_t* ctx_a_slot(int slot) const { return ctx_a_.data() + static_cast<size_t>(slot) * kMaxVerifyRows * 2 * sizeof(int64_t); }
  uint8_t* ctx_b_slot(int slot) const { return ctx_b_.data() + static_cast<size_t>(slot) * 2 * sizeof(int64_t); }
  int64_t* result_ring_slot(int slot) const { return result_host_ + static_cast<size_t>(slot) * kRing * 8; }
  uint8_t* d_tokens_slot(int slot) const { return d_tokens_.data() + static_cast<size_t>(slot) * token_slots_ * sizeof(int64_t); }
  uint8_t* d_dflash_prompt_slot(int slot) const {
    return d_dflash_prompt_.data() + static_cast<size_t>(slot) * dflash_prompt_slot_rows_ * kDflashWidth * 2;
  }
  uint8_t* d_dflash_step_aux_slot(int slot) const {
    return d_dflash_step_aux_.data() + static_cast<size_t>(slot) * kMaxVerifyRows * kDflashWidth * 2;
  }
  uint8_t* past_pos_slot(int slot) const { return past_pos_ + static_cast<size_t>(slot) * static_cast<size_t>(capacity_) * sizeof(int64_t); }
  static const Request* SlotRequestKey(int slot) { return reinterpret_cast<const Request*>(static_cast<uintptr_t>(slot + 1)); }

  Qwen38GraphModel& model_;
  Qwen38Plane& plane_;
  void* stream_{};
  std::unique_ptr<OrtMemoryInfo> memory_info_;
  const int capacity_;
  const int vocab_;
  int max_batch_{1};
  size_t token_slots_{};
  int hidden_width_{};
  int ple_width_{};
  int ple_layer_{};
  size_t status_count_{};
  std::vector<std::string> status_labels_;
  bool mtp_enabled_{};
  bool prefill_head_{};
  bool fold_{};
  bool force_reject_{};  // test hook (ORT_QWEN38_FORCE_REJECT): every draft is replaced before its verify
  bool compact_state_updates_{false};
  int state_update_capacity_{0};
  int verify_rows_{2};  // rows verified per target forward pass (2 = 1 draft, 3 = 2 drafts, 4 = 3 drafts, 8 = 7 drafts)
  int num_banks_{Qwen38DecodeLedger::kBanks};
  static constexpr int kBanks = Qwen38DecodeLedger::kBanks;
  static constexpr int kMaxBanks = 5;
  static constexpr int kMaxVerifyRows = 8;
  static_assert(kMaxBanks <= kQwen38GraphIdMaxPhases && kMaxVerifyRows <= kQwen38GraphIdMaxRows &&
                    kMaxBatch <= kQwen38GraphIdMaxBatch,
                "every (row count, ring phase, request slots) of the decode graphs needs a CUDA graph id");
  static constexpr int kRing = 64;
  static constexpr int kArgmaxBlocks = 16;

  // target state
  std::map<std::string, std::unique_ptr<DevTensor>> store_;                 // in-place KV / positions
  std::map<std::string, std::unique_ptr<DevTensor>> shared_;                // index_key buffers
  std::map<std::string, std::unique_ptr<DevTensor>> state_update_tensors_;  // compact state_update.* outputs
  std::map<std::string, size_t> store_slot_bytes_;
  std::map<std::string, size_t> shared_slot_bytes_;
  std::map<std::string, size_t> bank_slot_bytes_;
  std::map<std::string, size_t> su_slot_bytes_;
  std::map<std::string, std::vector<int64_t>> store_slot_shape_;
  std::map<std::string, std::vector<int64_t>> shared_slot_shape_;
  std::map<std::string, std::vector<int64_t>> bank_slot_shape_;
  std::map<std::string, std::vector<int64_t>> su_slot_shape_;
  PlaneBuffer replay_descs_dev_[2];
  PlaneBuffer slot_replay_descs_dev_[2][kMaxBatch];
  int replay_fast_count_{0};
  int replay_generic_count_{0};
  int replay_fast_blocks_{0};
  Bank banks_[kMaxBanks + 1];  // last = scratch sink for S=1 present_mid
  RowInputs inputs_[kMaxVerifyRows];
  Direction dirs_[kMaxVerifyRows][kMaxBanks];  // [rows-1][phase]
  RowInputs batch_inputs_[kMaxBatch];
  Direction batch_dirs_[kMaxBatch][kMaxBanks];
  std::vector<std::unique_ptr<DevTensor>> output_tensors_;
  std::vector<std::unique_ptr<DevTensor>> head_views_;
  std::deque<PlaneBuffer> owned_;  // individually allocated tensors
  const Direction* inflight_dir_{};
  int inflight_batch_{1};

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
  } head_banks_[kMaxBanks];  // two ping-pong banks (S=1/S=2 head sessions) or the ring phases (fold)
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
    Direction dirs[kMaxBanks];
  } head_r2_;
  std::map<std::string, std::unique_ptr<DevTensor>> fold_tensors_;  // fold.* inputs (the device control block)
  PlaneBuffer fold_ptrs_;

  // fast prefill (dense session + hand-off + batched head KV prefill, or unified target/head_r2 prefill); null when not configured
  struct FastPrefillState {
    bool unified{};
    OrtSession* dense{};
    SessionInfo dense_info;
    std::vector<std::string> dense_inputs, dense_outputs;
    OrtSession* head_kv{};
    SessionInfo head_kv_info;
    int chunk{8192};
    Qwen38QsaConstants* consts{};

    PlaneBuffer ones, idx, ples, raw_keys, scratch;  // ones: attention mask; idx: logits_indices; ples: PLE rows of a chunk
    PlaneBuffer embeds;                              // device buffer for CPU embedding.onnx output [1, m, hidden_width_]
    PlaneHostBuffer embeds_host;                     // pinned host staging buffer for CPU embedding.onnx output [m, hidden_width_]
    std::deque<PlaneBuffer> zero_buffers;
    std::map<std::string, std::unique_ptr<OrtValue>> initial;  // zero-length / zero initial states of the dense model
    std::map<std::string, std::unique_ptr<OrtValue>> states;   // dense `past` inputs (outputs of the previous chunk)
    size_t dense_length{};
    std::unique_ptr<OrtValue> logits;  // logits row of the last chunk
    std::unique_ptr<OrtRunOptions> run_options;
  };
  std::unique_ptr<FastPrefillState> fast_;
  std::shared_ptr<Dflash2Model> dflash_model_;
  std::unique_ptr<Dflash2Drafter> dflash_;
  PlaneBuffer d_dflash_prompt_;
  size_t dflash_prompt_slot_rows_{};  // rows one slot's slice of d_dflash_prompt_ holds (Qwen38DflashPromptSlotRows)
  PlaneBuffer d_dflash_step_aux_;     // aux rows of the verify steps, kMaxVerifyRows per slot
  // Which prompt chunks the drafter ingests, one per slot (slot 0 serves the single-slot calls); the window is set with dflash_
  Qwen38DflashPromptFeatures slot_dflash_prompt_[kMaxBatch];
  static constexpr size_t kDflashWidth = 12800;

  // host state
  Qwen38DecodeLedger ledger_;
  std::vector<Qwen38DecodeLedger> slot_ledgers_;
  int bank_phase_{0};
  uint64_t slot_seq_[kMaxBatch]{};
  int32_t slot_pending_[kMaxBatch]{};
  size_t slot_deferred_aux_rows_[kMaxBatch]{};
  size_t slot_deferred_aux_pos_[kMaxBatch]{};
  bool slot_decode_started_[kMaxBatch]{};
  Qwen38SuffixDrafter suffix_drafter_;
  int suffix_lookup_mode_{0};  // 0=off, 1=always prefer 3-gram suffix match, 2=fallback on prior draft reject
  uint64_t seq_{};
  bool inflight_{};
  bool decode_started_{};
  bool keep_hidden_{};
  int32_t pending_{};
  std::vector<int32_t> prompt_tokens_;
  const uint8_t* last_hidden_ptr_{};  // hidden output of the last cycle (rows h0, h1)
  Qwen38OrtRunHints run_hints_;       // what the ORT QMoE kernels are told with each Run (qwen38_ort_env.h)
};

struct Qwen38GraphModel : Model {
  Qwen38GraphModel(std::unique_ptr<Config> config, OrtEnv& ort_env);
  ~Qwen38GraphModel() override;

  // The Engine API is the only way to run this model.
  std::unique_ptr<State> CreateState(DeviceSpan<int32_t> sequence_lengths, const GeneratorParams& params) const override;

  // Leases the device engine to one Engine at a time (it owns the captured graphs, the KV store and the bank ring). The
  // lease drains the device when released and keeps the model alive. Throws when another Engine holds it.
  std::shared_ptr<Qwen38DecodeDevice> AcquireDevice();
  void PoisonExecutionContext() noexcept { context_healthy_.store(false); }
  bool IsExecutionContextHealthy() const noexcept { return context_healthy_.load(); }
  const BackendCapabilities& ResolvedBackendCapabilities() const noexcept { return resolved_capabilities_; }

  // The CPU embedding table that PrepRow gathers rows from (data is null when the model has none; see embed_mapping_).
  Qwen38EmbeddingTable EmbeddingTable() const;

  // Declaration order = reverse destruction order: the engine and the sessions go first, then the shared weights / constants
  // they were given, then the plane library.
  std::unique_ptr<Qwen38Plane> plane_;
  std::unique_ptr<Qwen38WeightPool> pool_;
  std::unique_ptr<Qwen38FileMapping> embed_mapping_;
  bool embed_is_int8_{false};
  std::vector<std::unique_ptr<OrtValue>> embed_values_;
  std::unique_ptr<OrtSessionOptions> embedding_session_options_;
  std::unique_ptr<OrtSession> session_embedding_;
  std::unique_ptr<OrtRunOptions> embedding_run_options_;
  std::unique_ptr<Qwen38QsaConstants> qsa_constants_;
  std::unique_ptr<OrtSession> session_target_;
  std::unique_ptr<OrtSessionOptions> head_session_options_;
  std::unique_ptr<OrtSession> session_head_s1_, session_head_s2_;
  std::unique_ptr<OrtSessionOptions> head_r2_session_options_, dense_session_options_, head_kv_session_options_;
  std::unique_ptr<OrtSession> session_head_r2_, session_dense_, session_head_kv_;
  std::unique_ptr<Qwen38GraphEngine> engine_;

  void RunCpuEmbedding(const int64_t* ids_host, size_t num_tokens, void* dst_host) const;

 private:
  std::atomic<bool> leased_{false};
  std::atomic<bool> context_healthy_{true};
  BackendCapabilities resolved_capabilities_;
};

}  // namespace Generators
