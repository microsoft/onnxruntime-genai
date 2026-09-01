// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <future>
#include <fstream>
#include <iterator>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#include "generator/generators.h"
#include "models/model.h"
#include "models/preprocessing/genai_tokenizer.h"
#if USE_GUIDANCE
#include "llguidance.h"
#endif

#include "constrained_logits_processor.h"

namespace Generators {

namespace {

// Grammar kinds llguidance knows how to build a constraint for. Kept independent of USE_GUIDANCE
// so that ValidateGuidanceRequest() rejects an unsupported guidance_type with the same message
// whether or not this build links llguidance.
constexpr std::array<std::string_view, 3> kSupportedGuidanceTypes{
    "json_schema", "regex", "lark_grammar"};

bool IsSupportedGuidanceType(std::string_view guidance_type) {
  return std::find(kSupportedGuidanceTypes.begin(), kSupportedGuidanceTypes.end(), guidance_type) !=
         kSupportedGuidanceTypes.end();
}

// Shared message for a validly-formed guidance request that this build cannot satisfy. Used by
// every CreateGuidanceLogitsProcessor overload below so a build without use_guidance=true fails
// the same way for the generic Generator and the Engine, and never falls back to a
// warning-and-unconstrained-generation behavior.
std::string UnavailableGuidanceBuildMessage(std::string_view guidance_type) {
  return "Guidance is unavailable in this build (guidance_type=\"" + std::string(guidance_type) +
         "\"). Rebuild with use_guidance=true.";
}

}  // namespace

bool ValidateGuidanceRequest(std::string_view guidance_type, std::string_view guidance_data) {
  const bool has_type = !guidance_type.empty();
  const bool has_data = !guidance_data.empty();
  if (!has_type && !has_data) {
    return false;
  }
  if (has_type != has_data) {
    throw std::runtime_error("Guidance type and data must be provided together.");
  }
  if (!IsSupportedGuidanceType(guidance_type)) {
    throw std::runtime_error("Unsupported guidance type: " + std::string(guidance_type) +
                             " (only json_schema, regex, and lark_grammar are supported).");
  }
  return true;
}

#if USE_GUIDANCE
namespace {

constexpr size_t kMaxCachedGrammars = 64;
constexpr size_t kMaxCachedGrammarKeyBytes = 16 * 1024 * 1024;

bool FitsGrammarCacheKeyBytes(size_t cached_bytes,
                              size_t key_bytes) noexcept {
  return key_bytes <= kMaxCachedGrammarKeyBytes &&
         cached_bytes <= kMaxCachedGrammarKeyBytes - key_bytes;
}

struct LlgTokenizerDeleter {
  void operator()(LlgTokenizer* tokenizer) const {
    llg_free_tokenizer(tokenizer);
  }
};

struct SharedLlgConstraintDeleter {
  void operator()(LlgConstraint* constraint) const {
    llg_free_constraint(constraint);
  }
};

struct TokenizeData {
  std::shared_ptr<Tokenizer> tokenizer;
  size_t prefix_len;
  std::mutex mutex;
};

struct GuidanceTokenizerAsset {
  std::shared_ptr<Tokenizer> tokenizer;
  std::shared_ptr<TokenizeData> tokenize_data;
  // Keep this after tokenize_data: reverse member destruction frees the callback-owning
  // llguidance tokenizer before the raw callback user_data it references.
  std::shared_ptr<LlgTokenizer> llg_tokenizer;
};

}  // namespace

struct GuidanceGrammarAsset {
  // Keep initial_constraint after tokenizer: reverse member destruction frees the constraint
  // before the tokenizer it references.
  std::shared_ptr<const GuidanceTokenizerAsset> tokenizer;
  std::shared_ptr<LlgConstraint> initial_constraint;
  // Serializes clones of the one model-shared initial cursor. Request-local cursors are cloned
  // only on the serialized Engine path and do not use this mutex.
  mutable std::mutex clone_mutex;
};

struct GuidanceCacheState {
  struct GrammarEntry {
    std::shared_future<std::shared_ptr<const GuidanceGrammarAsset>> future;
    std::list<std::string>::iterator lru_iterator;
    bool ready{};
  };

  std::mutex mutex;
  std::shared_future<std::shared_ptr<const GuidanceTokenizerAsset>> tokenizer_future;
  std::unordered_map<std::string, GrammarEntry> grammars;
  std::list<std::string> lru;
  size_t cached_key_bytes{};
  size_t pending_key_bytes{};
  uint64_t tokenizer_initializations{};
  uint64_t grammar_hits{};
  uint64_t grammar_misses{};
  uint64_t grammar_waits{};
  uint64_t grammar_compile_microseconds{};
  uint64_t grammar_evictions{};
};

namespace {

void EvictLeastRecentlyUsedGrammar(GuidanceCacheState& cache) {
  const auto lru_entry = std::prev(cache.lru.end());
  cache.cached_key_bytes -= lru_entry->size();
  cache.grammars.erase(*lru_entry);
  cache.lru.erase(lru_entry);
  ++cache.grammar_evictions;
}

struct ParallelMaskJob {
  struct Output {
    std::promise<std::vector<uint32_t>> promise;
    std::exception_ptr error;
    size_t first_constraint{};
    size_t constraint_count{};
    size_t first_mask_word{};
    size_t mask_word_count{};
    size_t words_per_row{};
  };

  std::vector<std::shared_ptr<const GuidanceGrammarAsset>> grammar_assets;
  std::vector<std::shared_ptr<LlgConstraint>> constraints;
  std::vector<uint32_t> masks;
  std::vector<Output> outputs;
};

void CompleteParallelMaskJob(const void* user_data) noexcept {
  try {
    std::unique_ptr<ParallelMaskJob> job(
        const_cast<ParallelMaskJob*>(static_cast<const ParallelMaskJob*>(user_data)));
    for (auto& output : job->outputs) {
      try {
        for (size_t i = 0; i < output.constraint_count; ++i) {
          if (const char* error =
                  llg_get_error(job->constraints[output.first_constraint + i].get())) {
            throw std::runtime_error("Error computing mask: " + std::string(error));
          }
        }
      } catch (...) {
        output.error = std::current_exception();
      }
    }

    // Mask computation is complete. Drop the job's cursor references before waking request
    // waiters, so their next CommitTokens() sees unique ownership unless a real transaction
    // checkpoint still shares the cursor.
    job->constraints.clear();
    job->grammar_assets.clear();

    for (auto& output : job->outputs) {
      try {
        if (output.error) {
          output.promise.set_exception(output.error);
        } else {
          const auto begin = job->masks.begin() +
                             static_cast<std::ptrdiff_t>(output.first_mask_word);
          output.promise.set_value(
              std::vector<uint32_t>(begin, begin + output.mask_word_count));
        }
      } catch (...) {
        // If value materialization fails, deliver that failure through the same future.
        try {
          output.promise.set_exception(std::current_exception());
        } catch (...) {
          // A promise that was already satisfied cannot strand a waiter.
        }
      }
    }
  } catch (...) {
    // No exception may cross the C callback boundary.
  }
}

std::shared_ptr<const GeneratorParams> SnapshotGuidanceParams(
    const Model& model, const GeneratorParams& params) {
  auto snapshot = std::make_shared<GeneratorParams>(model);
  snapshot->search = params.search;
  snapshot->p_device = params.p_device;
  snapshot->guidance_type = params.guidance_type;
  snapshot->guidance_data = params.guidance_data;
  snapshot->guidance_ff_tokens_enabled = params.guidance_ff_tokens_enabled;
  return snapshot;
}

std::shared_ptr<const GuidanceTokenizerAsset> CreateTokenizerAsset(
    const Model& model, const GeneratorParams& params) {
  auto tokenizer = model.CreateTokenizer();

  const fs::path tokenizer_dir = params.config.ResolvePath(params.config.model.tokenizer_dir);
  const fs::path json_path = tokenizer_dir / GuidanceLogitsProcessor::kDefaultVocabFile;
  std::ifstream json_file(json_path.string(), std::ios::binary);
  if (!json_file) {
    throw std::runtime_error("Unable to open guidance tokenizer data: " + json_path.string());
  }
  std::stringstream json_buffer;
  json_buffer << json_file.rdbuf();
  const std::string json_data = json_buffer.str();

  auto tokenize_data = std::make_shared<TokenizeData>();
  tokenize_data->tokenizer = tokenizer;
  tokenize_data->prefix_len = tokenizer->Encode(GuidanceLogitsProcessor::kTokenizePrefixStr).size();

  LlgTokenizeFn tokenize_fn = +[](const void* user_data, const uint8_t* bytes,
                                  size_t bytes_len, uint32_t* output_tokens,
                                  size_t output_tokens_len) -> size_t {
    auto* data = const_cast<TokenizeData*>(reinterpret_cast<const TokenizeData*>(user_data));
    std::lock_guard lock(data->mutex);
    const auto output_ids = GuidanceLogitsProcessor::tokenize_partial(
        data->tokenizer.get(), data->prefix_len, bytes, bytes_len);
    const size_t output_size = std::min(output_tokens_len, output_ids.size());
    for (size_t i = 0; i < output_size; ++i) {
      output_tokens[i] = static_cast<uint32_t>(output_ids[i]);
    }
    return output_ids.size();
  };

  LlgTokenizerInit tokenizer_init{
      static_cast<uint32_t>(params.config.model.vocab_size),
      static_cast<uint32_t>(params.config.model.eos_token_id[0]),
      nullptr,
      nullptr,
      json_data.c_str(),
      false,
      tokenize_fn,
      false,
      tokenize_data.get(),
  };

  char error_buf[256]{};
  auto* raw_tokenizer = llg_new_tokenizer(&tokenizer_init, error_buf, sizeof(error_buf));
  if (!raw_tokenizer) {
    throw std::runtime_error("Error creating llg_tokenizer: " + std::string(error_buf));
  }

  auto asset = std::make_shared<GuidanceTokenizerAsset>();
  asset->tokenizer = std::move(tokenizer);
  asset->tokenize_data = std::move(tokenize_data);
  asset->llg_tokenizer = std::shared_ptr<LlgTokenizer>(raw_tokenizer, LlgTokenizerDeleter{});
  return asset;
}

std::shared_ptr<const GuidanceTokenizerAsset> GetTokenizerAsset(
    const Model& model, const GeneratorParams& params,
    const std::shared_ptr<GuidanceCacheState>& cache) {
  std::shared_ptr<std::promise<std::shared_ptr<const GuidanceTokenizerAsset>>> promise;
  std::shared_future<std::shared_ptr<const GuidanceTokenizerAsset>> future;
  {
    std::lock_guard lock(cache->mutex);
    if (cache->tokenizer_future.valid()) {
      future = cache->tokenizer_future;
    } else {
      promise = std::make_shared<std::promise<std::shared_ptr<const GuidanceTokenizerAsset>>>();
      future = promise->get_future().share();
      cache->tokenizer_future = future;
    }
  }

  if (promise) {
    try {
      auto asset = CreateTokenizerAsset(model, params);
      promise->set_value(std::move(asset));
      std::lock_guard lock(cache->mutex);
      ++cache->tokenizer_initializations;
    } catch (...) {
      promise->set_exception(std::current_exception());
      std::lock_guard lock(cache->mutex);
      cache->tokenizer_future = {};
      throw;
    }
  }
  return future.get();
}

std::string GrammarCacheKey(const GeneratorParams& params) {
  std::string key;
  key.reserve(params.guidance_type.size() + params.guidance_data.size() + 3);
  key.append(params.guidance_type);
  key.push_back('\0');
  key.append(params.guidance_data);
  key.push_back('\0');
  key.push_back(params.guidance_ff_tokens_enabled && params.search.batch_size == 1 &&
                        params.search.num_beams == 1
                    ? '\1'
                    : '\0');
  return key;
}

std::shared_ptr<const GuidanceGrammarAsset> CompileGrammarAsset(
    const GeneratorParams& params,
    std::shared_ptr<const GuidanceTokenizerAsset> tokenizer_asset) {
  LlgConstraintInit constraint_init;
  llg_constraint_init_set_defaults(&constraint_init, tokenizer_asset->llg_tokenizer.get());
  constraint_init.ff_tokens_ok = params.guidance_ff_tokens_enabled && params.search.batch_size == 1 &&
                                 params.search.num_beams == 1;

  LlgConstraint* constraint = nullptr;
  if (params.guidance_type == "json_schema") {
    constraint = llg_new_constraint_json(&constraint_init, params.guidance_data.data());
  } else if (params.guidance_type == "regex") {
    constraint = llg_new_constraint_regex(&constraint_init, params.guidance_data.data());
  } else if (params.guidance_type == "lark_grammar") {
    constraint = llg_new_constraint_lark(&constraint_init, params.guidance_data.data());
  }

  if (!constraint) {
    throw std::runtime_error("Error creating grammar: llguidance returned a null constraint.");
  }
  if (const char* error = llg_get_error(constraint)) {
    const std::string message(error);
    llg_free_constraint(constraint);
    throw std::runtime_error("Error creating grammar: " + message);
  }

  auto asset = std::make_shared<GuidanceGrammarAsset>();
  asset->tokenizer = std::move(tokenizer_asset);
  asset->initial_constraint =
      std::shared_ptr<LlgConstraint>(constraint, SharedLlgConstraintDeleter{});
  return asset;
}

std::shared_ptr<const GuidanceGrammarAsset> GetGrammarAsset(
    const GeneratorParams& params,
    const std::shared_ptr<GuidanceCacheState>& cache,
    std::shared_ptr<const GuidanceTokenizerAsset> tokenizer_asset) {
  const std::string key = GrammarCacheKey(params);
  std::shared_ptr<std::promise<std::shared_ptr<const GuidanceGrammarAsset>>> promise;
  std::shared_future<std::shared_ptr<const GuidanceGrammarAsset>> future;
  bool cached_entry = false;
  {
    std::lock_guard lock(cache->mutex);
    const auto existing = cache->grammars.find(key);
    if (existing != cache->grammars.end()) {
      future = existing->second.future;
      if (existing->second.ready) {
        ++cache->grammar_hits;
        cache->lru.splice(cache->lru.begin(), cache->lru, existing->second.lru_iterator);
      } else {
        ++cache->grammar_waits;
      }
    } else {
      promise = std::make_shared<std::promise<std::shared_ptr<const GuidanceGrammarAsset>>>();
      future = promise->get_future().share();
      ++cache->grammar_misses;
      const bool cacheable =
          FitsGrammarCacheKeyBytes(cache->pending_key_bytes, key.size());
      while (cacheable && !cache->lru.empty() &&
             !FitsGrammarCacheKeyBytes(cache->cached_key_bytes, key.size())) {
        EvictLeastRecentlyUsedGrammar(*cache);
      }
      if (cacheable &&
          FitsGrammarCacheKeyBytes(cache->cached_key_bytes, key.size())) {
        cache->grammars.emplace(
            key, GuidanceCacheState::GrammarEntry{
                     future, cache->lru.end(), false});
        cache->cached_key_bytes += key.size();
        cache->pending_key_bytes += key.size();
        cached_entry = true;
      }
    }
  }

  if (promise) {
    std::shared_ptr<const GuidanceGrammarAsset> asset;
    try {
      const auto compile_started = std::chrono::steady_clock::now();
      asset = CompileGrammarAsset(params, std::move(tokenizer_asset));
      const auto compile_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(
                                            std::chrono::steady_clock::now() - compile_started)
                                            .count();

      std::lock_guard lock(cache->mutex);
      cache->grammar_compile_microseconds +=
          static_cast<uint64_t>(std::max<int64_t>(0, compile_microseconds));
      if (cached_entry) {
        const auto inserted = cache->grammars.find(key);
        if (inserted == cache->grammars.end()) {
          throw std::logic_error(
              "Pending guidance grammar disappeared from the cache.");
        }
        while (cache->lru.size() >= kMaxCachedGrammars) {
          EvictLeastRecentlyUsedGrammar(*cache);
        }
        cache->lru.push_front(key);
        inserted->second.lru_iterator = cache->lru.begin();
        inserted->second.ready = true;
        cache->pending_key_bytes -= key.size();
      }
    } catch (...) {
      const auto error = std::current_exception();
      if (cached_entry) {
        std::lock_guard lock(cache->mutex);
        const auto inserted = cache->grammars.find(key);
        if (inserted != cache->grammars.end()) {
          if (inserted->second.lru_iterator != cache->lru.end()) {
            cache->lru.erase(inserted->second.lru_iterator);
          } else {
            cache->pending_key_bytes -= inserted->first.size();
          }
          cache->cached_key_bytes -= inserted->first.size();
          cache->grammars.erase(inserted);
        }
      }
      promise->set_exception(error);
      throw;
    }
    promise->set_value(std::move(asset));
  }

  return future.get();
}

}  // namespace

GuidanceLogitsProcessor::GuidanceLogitsProcessor(const State& state)
    : GuidanceLogitsProcessor(state.model_, *state.params_) {}

GuidanceLogitsProcessor::GuidanceLogitsProcessor(
    const Model& model, const GeneratorParams& params)
    : params_(SnapshotGuidanceParams(model, params)),
      eos_token_(params_->config.model.eos_token_id[0]) {
  if (!ValidateGuidanceRequest(params.guidance_type, params.guidance_data)) {
    throw std::runtime_error("GuidanceLogitsProcessor requires guidance type and data.");
  }

  InitializeGuidanceAssets(model);
  InitializeLlgConstraints();
  ComputeMask();
}

void GuidanceLogitsProcessor::InitializeGuidanceAssets(const Model& model) {
  std::shared_ptr<GuidanceCacheState> cache;
  {
    std::lock_guard lock(model.guidance_cache_mutex_);
    if (!model.guidance_cache_) {
      model.guidance_cache_ = std::make_shared<GuidanceCacheState>();
    }
    cache = model.guidance_cache_;
  }

  auto tokenizer_asset = GetTokenizerAsset(model, *params_, cache);
  grammar_asset_ = GetGrammarAsset(*params_, cache, std::move(tokenizer_asset));
}

void GuidanceLogitsProcessor::InitializeLlgConstraints() {
  llg_constraints_.clear();
  llg_constraints_.resize(params_->search.batch_size);
  ff_tokens_batch_.clear();
  ff_tokens_batch_.resize(params_->search.batch_size);

  // Clone the immutable initial cursor for each mutable request row.
  for (int i = 0; i < params_->search.batch_size; i++) {
    LlgConstraint* constraint;
    {
      std::lock_guard lock(grammar_asset_->clone_mutex);
      constraint = llg_clone_constraint(grammar_asset_->initial_constraint.get());
    }
    if (!constraint) {
      throw std::runtime_error("Error cloning cached guidance grammar.");
    }
    llg_constraints_[i] = std::shared_ptr<LlgConstraint>(constraint, LlgConstraintDeleter{});
  }
}

void GuidanceLogitsProcessor::ComputeMask() {
  pending_masks_ = {};
  mask_dirty_ = false;
  mask_words_per_row_ = (params_->config.model.vocab_size + 31) / 32;
  masks_ = ComputeMasks(*params_, eos_token_, llg_constraints_);
}

std::vector<uint32_t> GuidanceLogitsProcessor::ComputeMasks(
    const GeneratorParams& params, uint32_t eos_token,
    const std::vector<std::shared_ptr<LlgConstraint>>& constraints) {
  const size_t words_per_row = (params.config.model.vocab_size + 31) / 32;
  std::vector<uint32_t> masks(constraints.size() * words_per_row, 0);
  for (size_t batch_idx = 0; batch_idx < constraints.size(); ++batch_idx) {
    LlgMaskResult mask_result;
    const auto error = llg_compute_mask(constraints[batch_idx].get(), &mask_result);
    if (error != 0) {
      const char* error_message = llg_get_error(constraints[batch_idx].get());
      throw std::runtime_error("Error computing mask: " +
                               std::string(error_message ? error_message : "unknown llguidance error"));
    }

    auto mask = std::span<uint32_t>(masks).subspan(batch_idx * words_per_row, words_per_row);
    if (mask_result.sample_mask) {
      std::copy_n(mask_result.sample_mask, words_per_row, mask.begin());
    }
    if (mask_result.is_stop) {
      mask[eos_token / 32] |= uint32_t{1} << (eos_token % 32);
    }
  }
  return masks;
}

void GuidanceLogitsProcessor::EnsureMaskReady() {
  if (!pending_masks_.valid()) {
    return;
  }
  auto pending = std::exchange(pending_masks_, {});
  try {
    masks_ = pending.get();
  } catch (...) {
    // Submission and future-delivery failures can be retried. A cursor poisoned by llguidance
    // remains dirty so subsequent attempts continue to fail rather than reuse a stale mask.
    mask_dirty_ = true;
    throw;
  }
}

std::vector<int32_t> GuidanceLogitsProcessor::GetFFTokens(size_t index) {
  if (index >= ff_tokens_batch_.size()) {
    // in case guidance is not being used, return empty vector
    return std::vector<int32_t>();
  }

  auto v = std::vector<int32_t>(ff_tokens_batch_[index]);
  ff_tokens_batch_[index].clear();
  return v;
}

void GuidanceLogitsProcessor::CommitTokens(std::span<int32_t> tokens) {
  EnsureMaskReady();
  for (int i = 0; i < params_->search.batch_size; i++) {
    if (llg_constraints_[i].use_count() != 1) {
      auto* clone = llg_clone_constraint(llg_constraints_[i].get());
      if (!clone) {
        throw std::runtime_error("Error cloning shared guidance cursor before token commit.");
      }
      llg_constraints_[i] = std::shared_ptr<LlgConstraint>(clone, LlgConstraintDeleter{});
    }

    LlgCommitResult commit_result;
    auto error = llg_commit_token(llg_constraints_[i].get(), static_cast<uint32_t>(tokens[i]), &commit_result);
    if (error != 0) {
      std::string error_message = llg_get_error(llg_constraints_[i].get());
      throw std::runtime_error("Error committing tokens: " + error_message);
    }

    auto& ff_tokens = ff_tokens_batch_[i];
    ff_tokens.clear();
    // Store forced tokens (i.e. index >= 1) to process outside of this logits processor
    for (size_t j = 1; j < commit_result.n_tokens; j++) {
      ff_tokens.push_back((int32_t)commit_result.tokens[j]);
    }
  }

  mask_dirty_ = true;
}

void GuidanceLogitsProcessor::EnsureMaskScheduled() {
  if (mask_dirty_) {
    ConstrainedLogitsProcessor* processor = this;
    ScheduleGuidanceMaskComputation(std::span{&processor, size_t{1}});
  }
}

void GuidanceLogitsProcessor::ProcessLogits(DeviceSpan<float> logits) {
  EnsureMaskScheduled();
  EnsureMaskReady();
  if (params_->p_device->GetType() == DeviceType::CUDA || params_->p_device->GetType() == DeviceType::NvTensorRtRtx) {
    if (device_masks_.size() != masks_.size()) {
      device_masks_ = params_->p_device->Allocate<uint32_t>(masks_.size());
    }
    copy(std::span<const uint32_t>{masks_}, device_masks_.CpuSpan());
    device_masks_.CopyCpuToDevice();
    params_->p_device->LaunchAddLogitsMask(logits.Span().data(), params_->search.batch_size,
                                           params_->config.model.vocab_size, device_masks_.Span().data());
    return;
  }
  size_t vocab_index = 0;

  auto logits_span = logits.CpuSpan();
  for (int index = 0; index < params_->search.batch_size; index++) {
    auto subspan = logits_span.subspan(vocab_index, params_->config.model.vocab_size);
    const auto mask = std::span<const uint32_t>(masks_).subspan(
        static_cast<size_t>(index) * mask_words_per_row_, mask_words_per_row_);
    for (size_t i = 0; i < params_->config.model.vocab_size; i++) {
      // Each bit corresponds to one vocabulary token. A set bit allows the token; an unset bit masks its logit to
      // the lowest representable value.
      subspan[i] = mask[i / 32] & (uint32_t{1} << (i % 32)) ? subspan[i] : std::numeric_limits<float>::lowest();
    }
    vocab_index += params_->config.model.vocab_size;
  }
}

std::span<const uint32_t> GuidanceLogitsProcessor::GetReadyMask() {
  EnsureMaskScheduled();
  EnsureMaskReady();
  return masks_;
}

// Reset the LLGuidance constraints and then recompute the mask
void GuidanceLogitsProcessor::Reset() {
  pending_masks_ = {};
  InitializeLlgConstraints();
  ComputeMask();
}

// Independent grammar cursor at the current state. Immutable tokenizer and initial-grammar assets
// remain model-owned. Constraints are shared initially so transaction checkpoints are cheap; every
// production mutation path detaches them first through the COW checks in CommitTokens() and
// ScheduleGuidanceMaskComputation(). A checkpoint never retains in-flight work: if that work fails,
// rollback must restore a dirty cursor that can submit a fresh job rather than the same exceptional
// shared_future.
std::unique_ptr<ConstrainedLogitsProcessor> GuidanceLogitsProcessor::Clone() const {
  auto clone = std::unique_ptr<GuidanceLogitsProcessor>(new GuidanceLogitsProcessor());
  clone->params_ = params_;
  clone->eos_token_ = eos_token_;
  clone->grammar_asset_ = grammar_asset_;
  clone->mask_words_per_row_ = mask_words_per_row_;
  clone->masks_ = masks_;
  clone->mask_dirty_ = mask_dirty_ || pending_masks_.valid();
  clone->ff_tokens_batch_ = ff_tokens_batch_;
  clone->llg_constraints_ = llg_constraints_;
  return clone;
}

std::unique_ptr<ConstrainedLogitsProcessor> GuidanceLogitsProcessor::CloneForNewTurn() const {
  auto clone = std::unique_ptr<GuidanceLogitsProcessor>(new GuidanceLogitsProcessor());
  clone->params_ = params_;
  clone->eos_token_ = eos_token_;
  clone->grammar_asset_ = grammar_asset_;
  clone->InitializeLlgConstraints();
  clone->ComputeMask();
  return clone;
}

void ScheduleGuidanceMaskComputation(
    std::span<ConstrainedLogitsProcessor* const> processors) {
  std::vector<GuidanceLogitsProcessor*> candidates;
  candidates.reserve(processors.size());
  for (auto* processor : processors) {
    auto* guidance = dynamic_cast<GuidanceLogitsProcessor*>(processor);
    if (!guidance || !guidance->mask_dirty_) {
      continue;
    }
    if (guidance->pending_masks_.valid()) {
      throw std::logic_error("A dirty guidance cursor already has pending mask work.");
    }
    for (auto& constraint : guidance->llg_constraints_) {
      if (constraint.use_count() != 1) {
        auto* clone = llg_clone_constraint(constraint.get());
        if (!clone) {
          throw std::runtime_error("Error cloning shared guidance cursor before mask computation.");
        }
        constraint =
            std::shared_ptr<LlgConstraint>(clone, GuidanceLogitsProcessor::LlgConstraintDeleter{});
      }
    }
    candidates.push_back(guidance);
  }
  if (candidates.empty()) {
    return;
  }

  auto job = std::make_unique<ParallelMaskJob>();
  std::vector<std::shared_future<std::vector<uint32_t>>> futures;
  futures.reserve(candidates.size());
  size_t total_mask_words = 0;

  for (auto* guidance : candidates) {
    ParallelMaskJob::Output output;
    output.first_constraint = job->constraints.size();
    output.constraint_count = guidance->llg_constraints_.size();
    output.first_mask_word = total_mask_words;
    output.words_per_row = guidance->mask_words_per_row_;
    output.mask_word_count = output.constraint_count * output.words_per_row;
    total_mask_words += output.mask_word_count;

    futures.push_back(output.promise.get_future().share());
    job->constraints.insert(job->constraints.end(), guidance->llg_constraints_.begin(),
                            guidance->llg_constraints_.end());
    job->grammar_assets.push_back(guidance->grammar_asset_);
    job->outputs.push_back(std::move(output));
  }

  job->masks.assign(total_mask_words, 0);
  std::vector<LlgConstraintStep> steps;
  steps.reserve(job->constraints.size());
  for (const auto& output : job->outputs) {
    for (size_t i = 0; i < output.constraint_count; ++i) {
      steps.push_back(LlgConstraintStep{
          job->constraints[output.first_constraint + i].get(),
          job->masks.data() + output.first_mask_word + i * output.words_per_row,
          output.words_per_row * sizeof(uint32_t),
      });
    }
  }

  for (size_t i = 0; i < candidates.size(); ++i) {
    candidates[i]->pending_masks_ = std::move(futures[i]);
    candidates[i]->mask_dirty_ = false;
  }

  ParallelMaskJob* raw_job = job.release();
  llg_par_compute_mask(steps.data(), steps.size(), raw_job, CompleteParallelMaskJob);
}

GuidanceCacheStats GetGuidanceCacheStats(const Model& model) {
  std::shared_ptr<GuidanceCacheState> cache;
  {
    std::lock_guard lock(model.guidance_cache_mutex_);
    cache = model.guidance_cache_;
  }
  if (!cache) {
    return {};
  }

  std::lock_guard lock(cache->mutex);
  return GuidanceCacheStats{
      cache->tokenizer_initializations,
      cache->grammar_hits,
      cache->grammar_misses,
      cache->grammar_waits,
      cache->grammar_compile_microseconds,
      cache->grammar_evictions,
      static_cast<uint64_t>(cache->lru.size()),
      static_cast<uint64_t>(cache->cached_key_bytes),
  };
}

std::vector<int32_t> GuidanceLogitsProcessor::tokenize_partial(const Tokenizer* tokenizer, const size_t prefix_len,
                                                               const uint8_t* bytes, size_t bytes_len) {
  // add prefix to tokenize for partial tokenization, it will produce ids more stable
  std::string input_string = kTokenizePrefixStr;
  input_string.reserve(bytes_len + 2);
  for (size_t i = 0; i < bytes_len; i++) {
    input_string.push_back(bytes[i]);
  }
  std::vector<int32_t> output_ids = tokenizer->Encode(input_string.c_str());
  return std::vector<int32_t>(output_ids.begin() + prefix_len, output_ids.end());
}

#endif

#if !USE_GUIDANCE
void ScheduleGuidanceMaskComputation(
    std::span<ConstrainedLogitsProcessor* const> /*processors*/) {}

GuidanceCacheStats GetGuidanceCacheStats(const Model& /*model*/) {
  return {};
}
#endif

std::unique_ptr<ConstrainedLogitsProcessor> CreateGuidanceLogitsProcessor(const State& state) {
  return CreateGuidanceLogitsProcessor(state.model_, state.params_);
}

std::unique_ptr<ConstrainedLogitsProcessor> CreateGuidanceLogitsProcessor(
    const Model& model, std::shared_ptr<const GeneratorParams> params) {
  if (!ValidateGuidanceRequest(params->guidance_type, params->guidance_data)) {
    return nullptr;
  }
#if USE_GUIDANCE
  return std::make_unique<GuidanceLogitsProcessor>(model, *params);
#else
  // A valid guidance request in a build without llguidance is a hard error, never a fallback to
  // unconstrained generation: silently ignoring the request would let a caller believe their
  // output was constrained when it was not.
  throw std::runtime_error(UnavailableGuidanceBuildMessage(params->guidance_type));
#endif
}

// Engine-facing overload: a Request exists (and must validate its guidance request) before it is
// necessarily associated with a model, so this checks the request's shape and this build's
// support for it first - neither of which needs a model - and only then requires one.
std::unique_ptr<ConstrainedLogitsProcessor> CreateGuidanceLogitsProcessor(
    std::shared_ptr<const GeneratorParams> params) {
  if (!ValidateGuidanceRequest(params->guidance_type, params->guidance_data)) {
    return nullptr;
  }
#if USE_GUIDANCE
  if (!params->model_) {
    throw std::runtime_error("Guidance requires generator parameters associated with a model.");
  }
  return CreateGuidanceLogitsProcessor(*params->model_, params);
#else
  throw std::runtime_error(UnavailableGuidanceBuildMessage(params->guidance_type));
#endif
}

}  // namespace Generators
