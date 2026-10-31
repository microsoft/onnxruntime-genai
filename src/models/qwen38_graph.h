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
#include <functional>
#include <map>
#include <memory>
#include <optional>
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
// head KV prefill): loaded once from the files of a text manifest (weights_pool.txt, written by the package builders in
// tools/qwen38_graph/, e.g. build_package_engine_v12b.py), handed to each session through AddInitializer (ORT uses a
// user supplied initializer in place when it is on the session's device).
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
    // The entry's whole page-aligned range was added to the ranges to register with CUDA. A CPU entry left out of them (a
    // large one under the bounded host cache) is not pinned even when the page it starts on belongs to the range of a
    // neighbour in the same file, so it must not get a device alias: the alias would reach past that range.
    bool pinned{false};
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
// prompt is ingested by the decode graphs (prefill.enabled false), is even so that S=2 pairs stay aligned. The model calls it
// with the kv_capacity of its configuration, before it has loaded its graph.
void ValidateQwen38PrefillChunk(const Config::Model::Qwen38Graph& q);

// The part of ValidateQwen38PrefillChunk that does not look at kv_capacity: the chunk is not negative and, when the decode
// graphs ingest the prompt, even. Once the graph is loaded, kv_capacity is that graph's capacity, which a context tier can
// leave below the chunk of the package; the device clamps the chunk to it, so the Engine's dependencies check only this part.
void ValidateQwen38PrefillChunkShape(const Config::Model::Qwen38Graph& q);

// Throws std::invalid_argument naming the first prompt id outside [0, vocab_size). The ids are row indices of the
// embedding gather on the device. embed_gather clamps an id outside the table to row 0, so the model would silently embed
// the wrong token; the legacy embed_rows of a library without embed_gather has no bound at all and reads past the mapped
// file (a sticky CUDA error that takes the whole context down, or another row's data). The engine rejects such ids before
// it touches any state.
void ValidateQwen38PromptTokens(std::span<const int32_t> tokens, int vocab_size);

// Whether the session of the standalone R=2 head captures CUDA graphs. That session is created from the decoder's
// options merged into model.qwen38_graph.fold.head_r2.session_options (which may name only a few settings), so the
// answer comes from the merged options, not from the configured ones: Warmup() runs the head the number of times a
// captured graph needs, and a capture that lands in a live request stalls it.
bool Qwen38HeadR2CapturesGraph(const Config::SessionOptions& decoder, const std::optional<Config::SessionOptions>& head_r2);

// The number of request slots the device hosts: engine.dynamic_batching.max_batch_size, else
// model.qwen38_graph.max_batch_size, else search.batch_size, else the ORT_QWEN38_MAX_BATCH_SIZE test hook (the first of
// them that is set, or that asks for more than one slot, where 1 is the default); 1 when none is.
// engine.static_batching is not consulted: it configures the Engine's static path in general, which the single-slot
// backend always ignored, so a package that carries it (with the Engine's default of four or a value of its own) keeps
// its single slot and keeps loading on a stack that cannot serve several. Throws std::out_of_range naming the setting
// unless the value lies in [1, Config::Model::Qwen38Graph::kMaxBatchSize]: the device never clamps a request silently.
// The test hook is held to the same rule: text that is not an integer ("4x", "abc") is refused too, only an empty value
// is unset.
size_t ResolveQwen38MaxBatchSize(const Config& config);

// True when prefill.enabled names a dense prefill graph other than the decoder (model.qwen38_graph.prefill.dense.filename):
// the prompt then runs through sessions of its own. Otherwise the fold target ingests the prompt itself (unified prefill).
bool Qwen38HasSeparatePrefillSessions(const Config& config);

// Throws a std::runtime_error at model creation, before the plane library is loaded and before any weight is uploaded, for a
// configuration that asks for several request slots (ResolveQwen38MaxBatchSize > 1) on a decode stack that cannot serve
// them: slots are decoded by the batched fold target (model.qwen38_graph.fold.enabled) and their prompts are ingested chunk by
// chunk through it (unified fast prefill: prefill.enabled without a separate dense graph). A single slot has no restriction.
void ValidateQwen38MultiSlotConfig(const Config& config);

// Throws a std::runtime_error unless a target graph with these outputs can be run for `request_slots` slots at once: every slot
// keeps its own accepted prefix of the verify rows, which the batched run restores per slot from the compact state_update.*
// outputs, so a ring-bank target (present_mid* outputs, one bank per row count for the whole batch) is refused with a
// diagnostic naming the output. A single slot has no restriction. The model constructor runs it twice: on the outputs read
// from the model file before any weight is uploaded (ValidateQwen38TargetModelFile), and, authoritatively, on the outputs of
// the target session as soon as that exists, before the other sessions are created and any device state is allocated.
void ValidateQwen38BatchedTargetOutputs(size_t request_slots, const std::vector<std::string>& output_names);

// Throws a std::runtime_error when `request_slots` slots are asked of a target that verifies only `verify_rows` <= 2 rows per
// step. The engine sizes the batched fold.ctl / fold.ctx_a inputs of such a target like the single-slot ones (8 and 2 words
// per slot), but the plane prepares the slots 16 control words and 2 * rows context words apart: from the second slot on the
// graph would read the control words of another slot. A single slot has no restriction. The engine constructor runs it once
// the verify width is known, the only place the exported state_update.* shapes are read.
void ValidateQwen38BatchedVerifyRows(size_t request_slots, int verify_rows);

// The names of the outputs of the graph of the ONNX model file `model_file`, in order, read from its protobuf encoding
// (ModelProto.graph, GraphProto.output, ValueInfoProto.name) without reading a single initializer, so that they are known
// before any weight is uploaded. nullopt when the file cannot be opened, is not a protobuf ONNX model (an ORT-format model,
// say) or has no output: the caller then relies on the outputs of the session.
std::optional<std::vector<std::string>> ReadQwen38OnnxOutputNames(const std::filesystem::path& model_file);

// ValidateQwen38BatchedTargetOutputs on the outputs of `model_file`, for several slots only (a single slot reads nothing). A
// file whose outputs cannot be read is accepted here and left to the check against the session.
void ValidateQwen38TargetModelFile(size_t request_slots, const std::filesystem::path& model_file);

// The context tier an ORT_QWEN38_CONTEXT_LENGTH request names ("2k", "8k", "16k", "32k", "64k", "128k" or a token count): the
// request lower-cased, with the counts 2048, 8192, 16384, 32768, 65536 and 131072 folded to the tier of that size. Empty when
// nothing is requested. Throws std::invalid_argument for a value that cannot be part of a graph's file name (anything but
// letters and digits).
std::string NormalizeQwen38ContextTier(const std::string& request);

// Chooses the target graph of a package once, as a file name relative to the model folder. The model constructor hands the
// result to the pre-upload reader of the graph's output names (ValidateQwen38TargetModelFile) and to CreateSession, so both
// inspect the same file. `exists` tells whether a folder-relative file can be loaded (a file of the package, or a graph supplied
// from memory). Every candidate lies next to the decoder (model.decoder.filename, which may sit in a sub-folder such as target/):
//  - `context_request` (ORT_QWEN38_CONTEXT_LENGTH, see NormalizeQwen38ContextTier) takes model_<tier>.onnx, else
//    model_dynamic_ctx.onnx, else the decoder itself. The graph of a package is the 32K one, so the 32k tier tries
//    model_32k.onnx, then the decoder, then model_dynamic_ctx.onnx. The first candidate that exists wins; when none does, a
//    std::runtime_error names every path that was tried. Because the decoder is one of the candidates, a tier that has no graph
//    of its own is not an error: the model constructor logs the graph it took, with the kv_capacity, context_length and
//    max_length that graph gave, when logging is enabled.
//  - Without a context request, `mtp_drafts` (ORT_QWEN38_MTP_DRAFTS, 1..3) takes model_s<drafts + 1>.onnx next to the decoder,
//    else under target/, else the decoder.
// With neither, the decoder is returned as configured and nothing is looked up.
std::string ResolveQwen38TargetGraphFilename(const std::string& decoder_filename, const std::string& context_request,
                                             const std::string& mtp_drafts,
                                             const std::function<bool(const std::string&)>& exists);

// Resolves a context-tier variant for an auxiliary head graph (e.g. fold.head_r2.filename or head.filename_s1 / filename_s2)
// using the same <stem>_<tier><ext> / <stem>_dynamic_ctx<ext> / configured fallback chain as the target graph.
std::string ResolveQwen38AuxiliaryGraphFilename(const std::string& configured_filename, const std::string& context_request,
                                                const std::function<bool(const std::string&)>& exists);

// The context settings of a package once its target graph is known: model.qwen38_graph.kv_capacity, model.context_length and
// search.max_length, the default generation budget of a request (Engine::CreateRequest takes the smaller of it and the engine's
// own ceiling).
struct Qwen38GraphContext {
  int kv_capacity{};
  int context_length{};
  int max_length{};
};

// Follows the capacity of a target graph whose past_position_ids input holds `graph_positions` positions: kv_capacity is that
// count and context_length what is left of it after the rows of the widest verify step. The capacity is always synchronized,
// the generation budget only when the context asks for it. A request for a context tier (`context_requested`) takes the whole
// context, search.max_length = context_length. Without one the configured `configured_max_length` stays: a smaller limit is
// retained, a limit beyond the graph's context is clamped to it, and only a missing limit (<= 0) takes the whole context.
// Throws std::invalid_argument for graph_positions <= 0.
Qwen38GraphContext SyncQwen38GraphContext(int64_t graph_positions, int configured_max_length, bool context_requested);

// Checks a model package against the plane library it names, when the model is loaded, before its weights are uploaded and
// before any session is created: an INT8 PLE table (model.qwen38_graph.ple_table with a negative scale) needs the int8_ple
// plane capability, an INT8 embedding table (<model.embedding.filename>.data of vocab_size x hidden_size INT8 rows and
// one float scale per row) needs int8_embedding, the folded MTP verify path (model.qwen38_graph.fold.enabled), which
// prepares all the rows of a verify step with one call, needs negative_pos_add_batch, and a configuration with several request
// slots (ResolveQwen38MaxBatchSize > 1), which prepares the rows of every slot with one call too, needs batched_prep
// (qwen38_graph_plane.h). Throws a std::runtime_error naming the missing capability, what required it and the library when the
// library lacks one; BF16 and FP8 packages without the fold path or with one slot need no capability. Returns the layout of
// the embedding table the model maps from that file: kNone when the package has no table of that shape.
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

// The runs the warm-up gives one graph of a session: kQwen38WarmupRunsPerGraph when the session captures CUDA graphs (a single
// run never captures a graph, so the capture would land in the first live request that reaches it), else one, which still pays
// the first-use costs of the graph.
inline constexpr int Qwen38WarmupRuns(bool session_captures_graph) noexcept {
  return session_captures_graph ? kQwen38WarmupRunsPerGraph : 1;
}

// What the DFlash engine runs while its model is created. model.qwen38_graph.warmup=false opts out of all of it, as for the
// other paths: no forward, hence no CUDA graph captured and no startup cost, for the drafter or for the target. With
// warmup=true the drafter ingests a warm-up prompt and drafts every row width (`drafter`) and `target_tokens` dummy prompt
// tokens step through the S=verify_rows target graphs, enough runs of every (rows, ring bank) graph to capture it.
struct Qwen38DflashWarmupPlan {
  bool drafter{};
  size_t target_tokens{};
};
Qwen38DflashWarmupPlan PlanQwen38DflashWarmup(bool warmup, int verify_rows, int num_banks) noexcept;

// Whether the DFlash drafter session skips the input sanitizer of ORT's paged attention, which it does by default: the
// engine hands the drafter well-formed inputs. ORT_PAGED_ATTENTION_SKIP_SANITIZER=0 (or false) keeps the sanitizer and
// =1 (or true) skips it; any other non-empty value throws std::invalid_argument, as for every boolean switch of
// models/env_utils.h.
bool Qwen38DflashSkipsInputSanitizer();

// The rows of target features the engine hands the DFlash drafter (the auxiliary hidden states of the target, five layers of
// 2560 values) are this many bfloat16 values wide.
inline constexpr size_t kQwen38DflashFeatureWidth = 12800;

// Dflash2Drafter::Propose reads the packed features with the stride of its own aux_hidden_states input and does not check it
// against the tensor it is given, so a drafter that declares another width or element type than the engine writes would read
// outside every row of the feature buffer. Throws std::runtime_error naming both layouts unless the drafter's input is
// [rows, kQwen38DflashFeatureWidth] of bfloat16.
void ValidateQwen38DflashFeatureLayout(size_t aux_width, ONNXTensorElementDataType aux_type);

// The tokens one page of the DFlash drafter's cache holds: the second dimension of the past key input of its first
// layer, which the drafter graph names by `past_key_names` (model.dflash2.inputs.past_key_names, a layer template) and
// not necessarily "past_key_values.0.key". Throws std::runtime_error, naming the input, unless it is a 4-D cache of
// pages of 16, 128 or 256 tokens (std::runtime_error from the metadata when the graph has no such input).
size_t Qwen38DflashCachePageSize(const ModelStateMetadata& drafter, const std::string& past_key_names);

// Rows of the DFlash prompt-feature buffer that one request slot owns: a prompt chunk of `chunk` rows, but at least the
// `verify_rows` of a verification step, because the warm-up of the drafter views that many rows from the same buffer
// (a prefill.chunk below the verify width would otherwise make it read past the allocation). Throws std::invalid_argument
// unless both are positive.
size_t Qwen38DflashPromptSlotRows(int chunk, int verify_rows);

// Tokens of a prompt that stay allocated once its request is over: a longer prompt gives its buffer back.
inline constexpr size_t kQwen38RetainedPromptTokens = 8192;

// Forgets the prompt a request slot ingested, as the slot starts its next request (Reset and ResetSlot of slot 0 both call
// it). The tokens are only kept for the head prefill of that request, so a long-lived engine holds at most the prompt of the
// request in flight, never those it served before; the buffer of a prompt beyond kQwen38RetainedPromptTokens is released.
void Qwen38ForgetPromptTokens(std::vector<int32_t>& tokens) noexcept;

// The decode status of a device with several slots. The validation bits of a decode accumulate in the ctl[5] word of the slot
// that ran it, so every slot that started a decode (`started[slot]`) is read through `read_bits(slot)`, the slots beyond
// slot 0 included; a slot that never started is not read. Throws a std::runtime_error naming every slot with bits set (a
// device with a single slot keeps the message without a slot).
void CheckQwen38DecodeStatus(std::span<const bool> started, const std::function<int64_t(size_t)>& read_bits);

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
//   past_key_values.<n>.index_key                                    [batch, >= kv_capacity / ratio, 128]
// where ratio is kQwen38IndexKeyRatio. Other inputs (recurrent state, index buffers, ...) do not depend on the
// capacity and are ignored. With more than one request slot the position ids of the slots are kv_capacity apart in the
// buffer the plane fills, so past_position_ids must then be exactly [batch, kv_capacity]: a longer row would be read at the
// wrong offset for every slot but the first, without any error.
void ValidateQwen38CacheShapes(const std::vector<Qwen38CacheInput>& inputs, int kv_capacity, const std::string& graph,
                               size_t request_slots = 1);

// Derives the KV capacity of a loaded target graph from its static past_key_values.<n>.key|value cache length (shape[2]),
// falling back to past_position_ids (shape[1]) only when no static key/value cache input is present. Single-slot graphs may
// pad past_position_ids beyond the KV cache (e.g. 33040 positions for 33028 KV rows), so using the KV cache length preserves
// compatibility with ValidateQwen38CacheShapes.
int64_t ResolveQwen38GraphKvCapacity(const std::vector<Qwen38CacheInput>& inputs);

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
    size_t cos_rows{};
    size_t sin_rows{};
    int width{};
    void RequireRopePositions(size_t blocks, const char* set_name) const;
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
struct Qwen38GraphEngine final : Qwen38DecodeDevice {
  explicit Qwen38GraphEngine(Qwen38GraphModel& model);
  ~Qwen38GraphEngine() override;

  int Capacity() const override { return capacity_; }
  int ReservedTail() const override { return std::max(verify_rows_, kReservedTail); }
  bool MtpEnabled() const override { return mtp_enabled_; }
  size_t MaxBatchSize() const override { return static_cast<size_t>(max_batch_); }
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
  // The prompt length of the turn of `slot`, kept as the slot's own state and handed to ORT only by the Runs that ingest
  // that slot's prompt chunks (qwen38_ort_env.h).
  void ExpectPrefillTokensSlot(size_t slot, size_t total_tokens) override;

  void BeginDecode(bool run_cycle) override;
  Qwen38CycleRecord WaitCycle() override;
  void Advance(const Qwen38CycleRecord& record, bool want_more) override;
  void CheckDecodeStatus() override;

  void ResetSlot(size_t slot) override;
  void PrefillChunkSlot(size_t slot, std::span<const int32_t> tokens) override;
  int32_t PendingTokenSlot(size_t slot) const override;
  void BeginDecodeSlot(size_t slot, bool want_more) override;
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
  static constexpr int kMaxBatch = Config::Model::Qwen38Graph::kMaxBatchSize;
  // The int64 control words of one request slot in ctl_ (the `state` of the plane's prep): the slots lie this far apart for
  // the plane, for the batched fold.ctl views and for every host access.
  static constexpr size_t kCtlWordsPerSlot = QWEN38_PLANE_CTL_WORDS_PER_SLOT;
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
  void FastHandoff(size_t slot_idx = 0);
  void FastHeadKvPrefill();
  void FastReset();
  void MutedPrefill(bool muted);
  void MutedPrefillSlot(int slot, bool muted);
  void MuteSlot(int slot);
  void ResetSlotIndexLengths(int slot);
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
  // Drafts, one slot after the other, the slots whose previous cycle left aux rows behind (EnqueueBatchDecode).
  void DraftDeferredSlots(std::span<const size_t> active_slots);

  void PrepRow(int rows, int row, const void* tok, const void* pos, int add, const void* ctx_in, void* ctx_out);
  void PrepBatch(int batch);
  void RunTarget(int rows, int phase, const Qwen38PromptRunHints* prompt = nullptr);
  // Every ORT Run of the engine goes through here: it publishes the QMoE hints of run_hints_ for exactly this Run
  // (qwen38_ort_env.h). `participants` are the slots the Run serves: only their notices are published and consumed.
  void RunSession(OrtSession& session, const OrtRunOptions* run_options, const OrtIoBinding& binding,
                  const Qwen38PromptRunHints* prompt = nullptr, Qwen38SlotMask participants = Qwen38SlotBit(0));
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

  uint8_t* ctl_slot(int slot) const { return ctl_.data() + static_cast<size_t>(slot) * kCtlWordsPerSlot * sizeof(int64_t); }
  uint8_t* ctx_commit_slot(int slot) const { return ctx_commit_.data() + static_cast<size_t>(slot) * 2 * sizeof(int64_t); }
  uint8_t* ctx_a_slot(int slot) const { return ctx_a_.data() + static_cast<size_t>(slot) * kFoldCtxAWords * sizeof(int64_t); }
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
  static_assert(kMaxBatch <= QWEN38_PLANE_MAX_PREP_BATCH && kMaxVerifyRows <= QWEN38_PLANE_MAX_PREP_ROWS,
                "the plane prepares every slot and row of a step in one call");
  static_assert(static_cast<size_t>(kMaxBatch) <= Qwen38OrtRunHints::kMaxSlots,
                "every request slot needs a bit in the slot mask of a Run");
  static constexpr int kFoldCtlWords = static_cast<int>(kCtlWordsPerSlot);  // int64 words of ctl_ per slot
  static constexpr int kFoldCtxCommitWords = 2;                             // int64 words of ctx_commit_ per slot
  // int64 words of ctx_a_ per slot (the n-gram contexts of the verify rows, 2 per row): always the room of the largest row
  // count, so that the stride does not depend on verify_rows_. It is the stride that the plane's batched prep writes slot
  // s at, that ctx_a_slot() and the per-slot views use, and that SpeculativePrefixCommit derives from the size of its
  // prefixes input (words / slots), which is why the batched view of fold.ctx_a spans every slot at this width.
  static constexpr int kFoldCtxAWords = 2 * kMaxVerifyRows;
  static_assert(kFoldCtxAWords == QWEN38_PLANE_CTX_WORDS_PER_SLOT,
                "the prep kernels write the contexts of slot s at s * QWEN38_PLANE_CTX_WORDS_PER_SLOT");
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
  // The (source, destination) lengths of every indexer layer of a slot, as bank 0 -> bank 1 replays them.
  std::vector<int32_t*> slot_index_lengths_[kMaxBatch];
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
  std::map<std::string, std::unique_ptr<DevTensor>> fold_tensors_;  // fold.* inputs (device control block); the state outputs bind to the same tensors
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
  static constexpr size_t kDflashWidth = kQwen38DflashFeatureWidth;

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
  Qwen38OrtRunHints run_hints_;       // per slot: what the ORT QMoE kernels are told with each Run (qwen38_ort_env.h)
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
