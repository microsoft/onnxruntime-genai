// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "backend_factory.h"
#include "../logging.h"
#include "../models/qwen38_graph.h"
#include "qwen38_graph_executor.h"
#include "decoders/varlen_decoder_io.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace Generators {
namespace {

size_t CheckedMtpAdd(size_t left, size_t right) {
  if (right > std::numeric_limits<size_t>::max() - left)
    throw std::runtime_error("MTP scratch memory bytes overflow size_t.");
  return left + right;
}

size_t CheckedMtpMultiply(size_t left, size_t right) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left)
    throw std::runtime_error("MTP scratch memory bytes overflow size_t.");
  return left * right;
}

size_t IndexShareScratchBytes(const DecoderOnly_Model& model, size_t max_batch) {
  const auto& indexshare = model.index_share_config_;
  const size_t stages = static_cast<size_t>(indexshare.max_draft_tokens - 1);
  const size_t extend_rows = CheckedMtpMultiply(max_batch, static_cast<size_t>(indexshare.max_draft_tokens) + 1);
  const size_t output_capacity = CheckedMtpAdd(static_cast<size_t>(indexshare.base_capacity), stages);
  const size_t batch_ints = CheckedMtpAdd(
      CheckedMtpAdd(CheckedMtpAdd(static_cast<size_t>(indexshare.base_capacity), 2), std::max(size_t{1}, stages)),
      CheckedMtpMultiply(stages, CheckedMtpAdd(output_capacity, 5)));
  const size_t ints = CheckedMtpAdd(
      CheckedMtpMultiply(extend_rows, CheckedMtpAdd(output_capacity, 3)),
      CheckedMtpMultiply(max_batch, batch_ints));
  const size_t projection_rows = CheckedMtpAdd(extend_rows, CheckedMtpMultiply(stages, max_batch));
  size_t bytes = CheckedMtpAdd(CheckedMtpMultiply(ints, sizeof(int32_t)),
                               CheckedMtpMultiply(projection_rows, sizeof(int64_t)));
  const auto& decoder = model.config_->model.decoder;
  size_t update_bank_bytes = 0;
  if (decoder.state_groups) {
    for (const auto& group : *decoder.state_groups) {
      if (!group.state_update) continue;
      std::vector<std::string> templates;
      using Kind = Config::Model::Decoder::StateGroupKind;
      if (group.kind == Kind::FixedIndexer) {
        templates.push_back(decoder.outputs.state_update_indexer_names);
      } else if (group.kind == Kind::FixedConv) {
        templates.push_back(decoder.outputs.state_update_conv_value_names);
      } else if (group.kind == Kind::FixedPle) {
        templates.push_back(decoder.outputs.state_update_ple_token_names);
        templates.push_back(decoder.outputs.state_update_ple_conv_value_names);
      }
      for (const auto& output_template : templates) {
        if (output_template.empty()) continue;
        for (int layer : group.layer_ids) {
          std::string name = output_template;
          const auto placeholder = name.find("%d");
          if (placeholder != std::string::npos) name.replace(placeholder, 2, std::to_string(layer));
          const auto shape = model.session_info_.GetOutputShape(name);
          size_t row_elements = 1;
          for (size_t axis = 1; axis < shape.size(); ++axis) {
            if (shape[axis] <= 0) throw std::runtime_error("IndexShare state-update scratch requires static non-batch geometry.");
            row_elements = CheckedMtpMultiply(row_elements, static_cast<size_t>(shape[axis]));
          }
          update_bank_bytes = CheckedMtpAdd(update_bank_bytes, CheckedMtpMultiply(
                                                                   max_batch, CheckedMtpMultiply(row_elements, Ort::SizeOf(model.session_info_.GetOutputDataType(name)))));
        }
      }
    }
  }
  const size_t state_bytes = CheckedMtpAdd(
      CheckedMtpMultiply(2, FixedStatePool::StateBankBytes(model, max_batch)), update_bank_bytes);
  return CheckedMtpAdd(bytes, CheckedMtpMultiply(stages, state_bytes));
}

EngineDependencies CreateDecoderDependencies(std::shared_ptr<Model> model) {
  // The static batch decoder rebuilds its contiguous cache from the whole sequence every step, so
  // it cannot resume a half-written prompt. Reject a model that configures chunking here rather
  // than surfacing it once a Request is already being admitted.
  if (!model->config_->engine.dynamic_batching &&
      model->config_->search.chunk_size.value_or(0) != 0) {
    throw std::runtime_error(
        "search.chunk_size requires dynamic batching; the static batch scheduler cannot chunk a "
        "prefill.");
  }
  std::shared_ptr<DecoderOnly_Model> mtp_model;
  size_t mtp_bytes_per_block = 0;
  if (model->config_->model.mtp.IsEnabled()) {
    if (!model->config_->engine.dynamic_batching) {
      throw std::runtime_error("An Engine-hosted MTP head requires dynamic batching.");
    }
    auto& target_hidden_states =
        model->config_->model.decoder.outputs.hidden_states;
    target_hidden_states = model->config_->model.mtp.main_hidden_states;
    mtp_model = std::make_shared<DecoderOnly_Model>(
        CreateMtpDecoderConfig(*model->config_), GetOrtEnv());
    mtp_model->InitializeIndexShare(model->config_->model.mtp.index_share, GetOrtEnv());
    ValidateMtpModelCompatibility(
        *model->config_, model->session_info_, mtp_model->session_info_);
    mtp_bytes_per_block = PagedKeyValueCacheBytesPerBlock(mtp_model);
    // The head consumes the target decoder's packed hidden states, so the target must emit them.
    model->config_->engine.hidden_states_output_required = true;
  }

  std::unique_ptr<Dflash2Drafter> dflash2_drafter;
  std::shared_ptr<Dflash2Model> dflash2_model;
  size_t dflash2_bytes_per_block = 0;
  size_t dflash2_reserved_memory_bytes = 0;
  size_t dflash2_max_batch_size = 0;
  size_t dflash2_pool_blocks = 0;
  size_t dflash2_prefix_checkpoint_bytes = 0;
  bool dflash2_prefix_checkpoints_enabled = false;
  if (!model->config_->model.dflash2.filename.empty()) {
    if (!model->config_->engine.dynamic_batching) {
      throw std::runtime_error("An Engine-hosted DFlash 2 drafter requires dynamic batching.");
    }
    auto& dflash2 = model->config_->model.dflash2;
    auto& target_aux_output = model->config_->model.decoder.outputs.aux_hidden_states;
    if (dflash2.main_aux_hidden_states.empty()) {
      dflash2.main_aux_hidden_states = target_aux_output;
    }
    if (dflash2.main_aux_hidden_states.empty()) {
      throw std::runtime_error(
          "model.dflash2.main_aux_hidden_states must name a main-model output.");
    }
    target_aux_output = dflash2.main_aux_hidden_states;
    if (!model->session_info_.HasOutput(target_aux_output)) {
      throw std::runtime_error(
          "model.dflash2.main_aux_hidden_states must name a main-model output.");
    }

    const auto& batching = *model->config_->engine.dynamic_batching;
    const size_t paged_block_size = static_cast<size_t>(batching.block_size);
    dflash2_max_batch_size = static_cast<size_t>(batching.max_batch_size);
    auto decoder_model = std::dynamic_pointer_cast<DecoderOnly_Model>(model);
    dflash2_model = std::make_shared<Dflash2Model>(
        CreateDflash2Config(*model->config_), GetOrtEnv(),
        decoder_model ? decoder_model->cpu_embedding_ : nullptr);
    const auto dflash2_cache_type = ValidateDflash2ModelCompatibility(
        *model->config_, model->session_info_, dflash2_model->session_info_, paged_block_size);
    model->config_->engine.aux_hidden_states_output_required = true;
    dflash2_pool_blocks = Dflash2Drafter::PoolBlocks(
        *model->config_, paged_block_size, dflash2_max_batch_size);
    if (dflash2_pool_blocks != 0) {
      if (batching.num_blocks.has_value()) {
        // Explicit num_blocks bypasses the free-memory probe. CacheManager validates and deducts
        // this fixed pool before either cache pool is allocated.
        dflash2_reserved_memory_bytes = Dflash2Drafter::PoolBytes(
            *model->config_, paged_block_size, dflash2_pool_blocks, dflash2_cache_type);
      } else {
        // Automatic sizing measures free memory after allocating the fixed drafter pool.
        dflash2_drafter = std::make_unique<Dflash2Drafter>(
            dflash2_model, paged_block_size, dflash2_pool_blocks, dflash2_max_batch_size);
      }
    } else {
      // A full-attention drafter mirrors the target pool, so it is billed per target block instead
      // and built below once the target pool's block count is known.
      dflash2_bytes_per_block =
          Dflash2Drafter::BytesPerBlock(*model->config_, paged_block_size, dflash2_cache_type);
      dflash2_reserved_memory_bytes = Dflash2Drafter::FullAttentionReservedBytes(
          paged_block_size, static_cast<size_t>(dflash2.block_size),
          dflash2_max_batch_size, dflash2_bytes_per_block);
    }
    if (dflash2_model->cpu_embedding_) {
      const size_t embedding_bytes = Dflash2Drafter::EmbeddingReservedBytes(
          dflash2_max_batch_size, static_cast<size_t>(dflash2.block_size),
          static_cast<size_t>(dflash2_model->cpu_embedding_->hidden_size_),
          dflash2_model->cpu_embedding_->type_);
      if (embedding_bytes > std::numeric_limits<size_t>::max() - dflash2_reserved_memory_bytes) {
        throw std::runtime_error("DFlash 2 reserved memory bytes overflow size_t.");
      }
      dflash2_reserved_memory_bytes += embedding_bytes;
    }
    if (dflash2_pool_blocks != 0 && mtp_bytes_per_block == 0 && !dflash2.is_dspark &&
        ModelStateManifest{model->config_->model.decoder}.HasFixedStateGroups() &&
        ResolvePrefixCachingEnabled(model, /*auxiliary_bytes_per_block=*/0)) {
      dflash2_prefix_checkpoint_bytes = Dflash2Drafter::PrefixCheckpointBytes(
          *model->config_, paged_block_size, dflash2_cache_type);
    }
  }

  if (dflash2_bytes_per_block > std::numeric_limits<size_t>::max() - mtp_bytes_per_block) {
    throw std::runtime_error("Engine auxiliary cache bytes per block overflow size_t.");
  }
  // SimpleDecoder allocates its persistent capture buffers only after the cache has chosen its
  // block count, so the automatic free-memory probe has to know about them or the paged pool claims
  // the memory they still need. IndexShare's selection and per-stage state scratch are also priced
  // here, even without graph capture. Sizing uses the engine-wide draft ceiling, which is at least what
  // each decoder will ask for. An explicit num_blocks is the caller pinning the pool, so it is left
  // alone.
  size_t graph_buffer_reserved_bytes = 0;
  if (model->config_->engine.dynamic_batching &&
      !model->config_->engine.dynamic_batching->num_blocks.has_value()) {
    const auto reserve_for = [](const std::shared_ptr<Model>& decoder) -> size_t {
      if (!decoder || !IsGraphCaptureEnabled(decoder->config_->model.decoder.session_options)) {
        return 0;
      }
      return VarlenGraphBufferBytes(*decoder, PackedPositionIdPlanes(*decoder),
                                    kMaxDraftTokensPerStep + 1);
    };
    graph_buffer_reserved_bytes = reserve_for(model) + reserve_for(mtp_model);
    if (mtp_model) {
      const size_t max_batch = model->config_->engine.dynamic_batching->max_batch_size;
      const size_t scratch_bytes = mtp_model->index_share_config_.base_capacity > 0
                                       ? IndexShareScratchBytes(*mtp_model, max_batch)
                                       : CheckedMtpMultiply(2, FixedStatePool::StateBankBytes(*mtp_model, max_batch));
      graph_buffer_reserved_bytes = CheckedMtpAdd(graph_buffer_reserved_bytes, scratch_bytes);
    }
  }
  if (graph_buffer_reserved_bytes >
      std::numeric_limits<size_t>::max() - dflash2_reserved_memory_bytes) {
    throw std::runtime_error("Engine reserved memory bytes overflow size_t.");
  }
  std::shared_ptr<CacheManager> cache_manager =
      CacheManager::Create(model, mtp_bytes_per_block + dflash2_bytes_per_block,
                           dflash2_reserved_memory_bytes + graph_buffer_reserved_bytes,
                           dflash2_prefix_checkpoint_bytes,
                           &dflash2_prefix_checkpoints_enabled);
  std::shared_ptr<PrefixSnapshotCache> snapshot_cache;
  const auto& batching = model->config_->engine.dynamic_batching;
  const bool webgpu = model->p_device_kvcache_->GetType() == DeviceType::WEBGPU;
  const bool device_resident = webgpu || (batching && !batching->snapshot_cache.Configured());
  if (webgpu && batching && batching->snapshot_cache.Enabled() && g_log.enabled && g_log.warning) {
    Log("warning",
        "WebGPU does not support tiered prefix snapshot byte budgets; ignoring these limits "
        "and retaining device-resident snapshots subject to count limits.");
  }
  if (batching && batching->prefix_caching.value_or(true) && batching->dflash2_max_snapshots != 0 &&
      (!batching->max_snapshots || *batching->max_snapshots != 0) &&
      (device_resident || batching->snapshot_cache.Enabled())) {
    const auto& limits = batching->snapshot_cache;
    snapshot_cache = std::make_shared<PrefixSnapshotCache>(
        std::array<size_t, 3>{cache_manager->DraftCheckpointVramBytes(),
                              limits.ram_bytes, limits.disk_bytes},
        device_resident ? cache_manager->DraftCheckpointCapacity()
                        : batching->max_snapshots.value_or(std::numeric_limits<size_t>::max()),
        device_resident ? std::filesystem::path{}
                        : std::filesystem::path{fs::path{limits.directory}.c_str()},
        device_resident);
  }
  if (dflash2_model && !dflash2_drafter) {
    const size_t paged_block_size =
        static_cast<size_t>(model->config_->engine.dynamic_batching->block_size);
    if (dflash2_pool_blocks != 0) {
      dflash2_drafter = std::make_unique<Dflash2Drafter>(
          dflash2_model, paged_block_size, dflash2_pool_blocks, dflash2_max_batch_size);
    } else {
      const auto& dflash2 = model->config_->model.dflash2;
      dflash2_drafter = std::make_unique<Dflash2Drafter>(
          dflash2_model, paged_block_size,
          Dflash2Drafter::FullAttentionPoolBlocks(
              cache_manager->Snapshot().total_blocks, paged_block_size,
              static_cast<size_t>(dflash2.block_size), dflash2_max_batch_size),
          dflash2_max_batch_size);
    }
  }
  auto scheduler = Scheduler::Create(model, cache_manager);
  auto model_executor = ModelExecutor::Create(model, cache_manager);

  std::shared_ptr<CacheManager> mtp_cache_manager;
  std::unique_ptr<ModelExecutor> mtp_model_executor;
  if (mtp_model) {
    // Both decoders cover the same resident request set. Fixing the head to the main pool's block
    // count makes the auxiliary bytes included above exact rather than letting it independently
    // consume another gpu_utilization_factor share of currently free memory.
    mtp_model->config_->engine.dynamic_batching->num_blocks =
        cache_manager->Snapshot().total_blocks;
    mtp_cache_manager = CacheManager::Create(mtp_model);
    mtp_model_executor = ModelExecutor::Create(mtp_model, mtp_cache_manager);
  }

  return EngineDependencies{
      std::move(cache_manager), std::move(scheduler), std::move(model_executor),
      std::move(mtp_model), std::move(mtp_cache_manager), std::move(mtp_model_executor),
      std::move(dflash2_drafter), dflash2_prefix_checkpoints_enabled,
      nullptr, std::move(snapshot_cache)};
}

}  // namespace

BackendCapabilities ResolveBackendCapabilities(
    const Model& model, const CacheManager& cache_manager, const Scheduler& scheduler,
    const ModelExecutor& executor, std::optional<BackendCapabilities> declared) {
  BackendCapabilities resolved;
  const bool transactions = cache_manager.SupportsDynamicBatching();
  const bool own_tokens = executor.SelectsOwnTokens();
  if (transactions && own_tokens) {
    throw std::runtime_error("Backend-selected tokens require the nontransactional Engine execution protocol.");
  }
  resolved.execution_protocol = transactions ? BackendExecutionProtocol::Transactional
                                             : BackendExecutionProtocol::Nontransactional;
  resolved.token_selection = own_tokens ? BackendTokenSelection::Backend
                                        : BackendTokenSelection::Engine;
  resolved.admission = (cache_manager.SupportsContinuousMultiSlotDecoding() &&
                        scheduler.SupportsIndependentSlotAdmission())
                           ? BackendAdmission::IndependentSlots
                           : BackendAdmission::WholeBatch;
  resolved.kv_storage = transactions ? BackendKvStorage::Paged
                                     : (own_tokens ? BackendKvStorage::PersistentContiguous
                                                   : BackendKvStorage::RebuiltContiguous);
  if (own_tokens) {
    resolved.additional_state = BackendAdditionalState::PrivateBanks;
    resolved.continuation = BackendContinuation::ReplayHistory;
    resolved.supports_sampling = false;
    resolved.supports_penalties = false;
    resolved.supports_min_generated_tokens = false;
    resolved.supports_guidance = false;
  } else if (ModelStateManifest{model.config_->model.decoder}.HasFixedStateGroups()) {
    resolved.additional_state = BackendAdditionalState::FixedPools;
  }
  resolved.recovery = transactions ? BackendRecovery::TransactionalRollback
                                   : BackendRecovery::FatalAfterExecution;
  resolved.prefix_reuse = cache_manager.PrefixMetrics() != nullptr;
  resolved.supports_stop_strings = transactions || own_tokens;
  resolved.supports_per_turn_seed = transactions && scheduler.SupportsTransactionalSamplerState();
  resolved.supports_delimited_guidance = transactions && !own_tokens;
  resolved.max_slots = cache_manager.MaxBatchSize();
  resolved.max_result_tokens = own_tokens || transactions ? kMaxGeneratedTokensPerStep : 1;

  const auto& engine = model.config_->engine;
  if (engine.dynamic_batching) {
    const auto& batching = *engine.dynamic_batching;
    resolved.configured_max_batch_size = batching.max_batch_size;
    resolved.max_scheduled_tokens = batching.max_scheduled_tokens;
    const size_t block_count = cache_manager.TargetBlockCount();
    const size_t block_size = cache_manager.TargetBlockSize();
    if (block_count != 0 && block_size != 0) {
      if (block_count > (std::numeric_limits<uint64_t>::max() - 1) / block_size) {
        throw std::overflow_error("Engine max_request_length exceeds uint64_t.");
      }
      const uint64_t cache_limit = static_cast<uint64_t>(block_count) * block_size + 1;
      resolved.max_request_length = std::min<uint64_t>(
          static_cast<uint64_t>(model.config_->model.context_length), cache_limit);
    }
  } else {
    resolved.configured_max_batch_size = engine.static_batching
                                             ? engine.static_batching->max_batch_size
                                             : kDefaultStaticBatchSize;
  }
  if (!transactions) {
    if (const size_t rows = cache_manager.MaxStaticBatchSize()) {
      resolved.configured_max_batch_size = rows;
    }
    resolved.max_request_length = cache_manager.MaxStaticRequestLength();
    if (resolved.max_request_length != 0 && model.config_->model.context_length > 0) {
      resolved.max_request_length = std::min<uint64_t>(
          resolved.max_request_length,
          static_cast<uint64_t>(model.config_->model.context_length));
    }
  }

  if (!declared) return resolved;
  const auto& explicit_capabilities = *declared;
  if (explicit_capabilities.UsesTransactions() != transactions ||
      explicit_capabilities.SelectsOwnTokens() != own_tokens) {
    throw std::runtime_error("Engine backend capabilities contradict its cache/executor execution protocol.");
  }
  if (explicit_capabilities.max_slots == 0 ||
      explicit_capabilities.max_slots != cache_manager.MaxBatchSize() ||
      explicit_capabilities.max_result_tokens == 0 ||
      explicit_capabilities.max_result_tokens > kMaxGeneratedTokensPerStep ||
      (!own_tokens && !transactions && explicit_capabilities.max_result_tokens != 1)) {
    throw std::runtime_error("Engine backend capabilities contradict its effective execution limits.");
  }
  if (explicit_capabilities.SupportsIndependentAdmission() &&
      explicit_capabilities.max_slots > 1 &&
      (!cache_manager.SupportsContinuousMultiSlotDecoding() ||
       !scheduler.SupportsIndependentSlotAdmission())) {
    throw std::runtime_error("Engine backend capabilities require unsupported independent slot admission.");
  }
  if ((explicit_capabilities.supports_per_turn_seed && !resolved.supports_per_turn_seed) ||
      (explicit_capabilities.supports_stop_strings && !resolved.supports_stop_strings) ||
      (explicit_capabilities.supports_delimited_guidance && !resolved.supports_delimited_guidance) ||
      (explicit_capabilities.prefix_reuse && !resolved.prefix_reuse) ||
      (!transactions && explicit_capabilities.recovery == BackendRecovery::TransactionalRollback) ||
      // Zero declares no limit of its own (the model's search limit applies), which a finite cache cannot honor.
      (resolved.max_request_length != 0 &&
       (explicit_capabilities.max_request_length == 0 ||
        explicit_capabilities.max_request_length > resolved.max_request_length))) {
    throw std::runtime_error("Engine backend capabilities exceed its collaborators' supported features or capacity.");
  }
  return explicit_capabilities;
}

EngineBackendBundle CreateEngineBackend(std::shared_ptr<Model> model) {
  if (!model) throw std::invalid_argument("Engine backend construction requires a model.");
  EngineDependencies dependencies;
  std::optional<BackendCapabilities> declared;
  if (auto graph_model = std::dynamic_pointer_cast<Qwen38GraphModel>(model)) {
    declared = graph_model->ResolvedBackendCapabilities();
    dependencies = CreateQwen38GraphDependencies(std::move(graph_model));
  } else {
    dependencies = CreateDecoderDependencies(model);
  }
  auto capabilities = ResolveBackendCapabilities(
      *model, *dependencies.cache_manager, *dependencies.scheduler, *dependencies.model_executor,
      declared ? declared : dependencies.backend_capabilities);
  dependencies.backend_capabilities = capabilities;
  return {std::move(dependencies), capabilities};
}

}  // namespace Generators
