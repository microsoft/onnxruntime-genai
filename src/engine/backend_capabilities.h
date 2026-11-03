// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

namespace Generators {

// Internal, resolved execution contract. Admission and storage are independent: fixed slots can
// admit requests continuously without implementing paged transactions or rollback.
enum class BackendExecutionProtocol { Nontransactional,
                                      Transactional };
enum class BackendTokenSelection { Engine,
                                   Backend };
enum class BackendAdmission { WholeBatch,
                              IndependentSlots };
enum class BackendKvStorage { RebuiltContiguous,
                              PersistentContiguous,
                              Paged };
enum class BackendAdditionalState { None,
                                    FixedPools,
                                    PrivateBanks };
enum class BackendContinuation { RetainedState,
                                 ReplayHistory };
enum class BackendRewind { Unsupported,
                           ReplayHistory,
                           CheckpointRestore };
enum class BackendPhases { Serialized,
                           Interleaved,
                           Mixed };
enum class BackendRecovery { FatalAfterExecution,
                             TransactionalRollback };

struct BackendCapabilities {
  BackendExecutionProtocol execution_protocol{BackendExecutionProtocol::Nontransactional};
  BackendTokenSelection token_selection{BackendTokenSelection::Engine};
  BackendAdmission admission{BackendAdmission::WholeBatch};
  BackendKvStorage kv_storage{BackendKvStorage::RebuiltContiguous};
  BackendAdditionalState additional_state{BackendAdditionalState::None};
  BackendContinuation continuation{BackendContinuation::RetainedState};
  BackendRewind rewind{BackendRewind::ReplayHistory};
  BackendPhases phases{BackendPhases::Serialized};
  BackendRecovery recovery{BackendRecovery::FatalAfterExecution};
  bool prefix_reuse{};
  bool supports_sampling{true};
  bool supports_penalties{true};
  bool supports_min_generated_tokens{true};
  bool supports_guidance{true};
  bool supports_stop_strings{};
  bool supports_per_turn_seed{};
  bool supports_delimited_guidance{};

  // Requested/configured values retain their public meanings. Effective limits are finalized
  // after allocating the backend and already account for any private pending-token/reserved tail.
  size_t configured_max_batch_size{};
  size_t max_scheduled_tokens{};
  size_t max_slots{};
  uint64_t max_request_length{};  // Zero retains the legacy model-search-only public limit.
  size_t max_result_tokens{1};

  bool UsesTransactions() const noexcept {
    return execution_protocol == BackendExecutionProtocol::Transactional;
  }
  bool SelectsOwnTokens() const noexcept {
    return token_selection == BackendTokenSelection::Backend;
  }
  bool SupportsIndependentAdmission() const noexcept {
    return admission == BackendAdmission::IndependentSlots;
  }
};

}  // namespace Generators
