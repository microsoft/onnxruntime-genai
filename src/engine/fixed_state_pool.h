// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "../config.h"

#include "onnxruntime_c_api.h"

struct OrtValue;

namespace Generators {

struct Model;
class FixedStatePool;

// Per-request row geometry derived from a fixed-state binding's session metadata.
struct FixedStateGeometry {
  size_t row_element_count{};  // Product of every non-batch (axis >= 1) dimension.
  size_t fixed_batch_size{};   // Fixed axis-0 extent, or 0 when axis 0 is dynamic.
};

// Validates that a fixed-state binding's input and output describe one per-request row of
// identical, statically known, non-batch geometry with a batch axis that is either dynamic or a
// positive fixed extent. Throws std::runtime_error describing the first violation. This is the
// narrow, model-independent geometry contract the pool enforces on top of the manifest's
// input/output compatibility check; it is exposed so it can be exercised directly.
FixedStateGeometry ValidateFixedStateGeometry(std::string_view input_name,
                                              ONNXTensorElementDataType input_type,
                                              std::span<const int64_t> input_shape,
                                              std::string_view output_name,
                                              ONNXTensorElementDataType output_type,
                                              std::span<const int64_t> output_shape);

enum class FixedStateSlotOwnership {
  Free,
  Reserved,
  Committed,
};

struct FixedStateSlotHandle {
  const FixedStatePool* pool{};
  const void* request_id{};
  size_t slot{};
  uint64_t generation{};

  explicit operator bool() const {
    return pool != nullptr && request_id != nullptr;
  }

  bool operator==(const FixedStateSlotHandle&) const = default;
};

struct FixedStateCommittedState {
  FixedStateSlotHandle handle;
  uint64_t committed_tokens{};
};

// One scheduled row of a reservation: the request identity and the processed-token count the
// caller intends to commit for that request after this step. `target_tokens` is recorded against
// the slot at publish (as `committed_tokens`) and must never regress below the slot's currently
// committed value.
struct FixedStateReservationRequest {
  const void* request_id{};
  uint64_t target_tokens{};
  size_t capture_count{};
};

struct FixedStateBinding {
  Config::Model::Decoder::StateGroupKind kind{};
  int layer_id{};
  const char* input_name{};
  OrtValue* input{};
  const char* output_name{};
  OrtValue* output{};
  Config::Model::Decoder::StateUpdateKind state_update_kind{};
  size_t state_update_capacity{};
  const char* state_update_capture_count_name{};
  OrtValue* state_update_capture_count{};
  const char* state_update_active_name{};
  OrtValue* state_update_active{};
  const char* state_update_value_name{};
  OrtValue* state_update_value{};
  const char* state_update_capsule_name{};
  OrtValue* state_update_capsule{};
};

enum class FixedStateReservationState {
  Reserved,
  Prepared,
  Committed,
  Discarded,
  Failed,
};

struct FixedStateSlotSnapshot {
  const void* request_id{};
  size_t slot{};
  uint64_t generation{};
  uint64_t state_generation{};
  uint64_t committed_tokens{};
  FixedStateSlotOwnership ownership{FixedStateSlotOwnership::Free};
};

struct FixedStatePoolSnapshot {
  size_t capacity{};
  size_t free_slots{};
  size_t reserved_slots{};
  size_t committed_slots{};
  size_t persistent_bytes{};
  size_t zeroing_scratch_bytes{};
  size_t active_staging_bytes{};
  bool healthy{true};
  std::vector<FixedStateSlotSnapshot> slots;
};

class FixedStateReservation {
 public:
  FixedStateReservation(FixedStateReservation&& other) noexcept;
  FixedStateReservation& operator=(FixedStateReservation&&) = delete;
  FixedStateReservation(const FixedStateReservation&) = delete;
  FixedStateReservation& operator=(const FixedStateReservation&) = delete;
  ~FixedStateReservation();

  FixedStateReservationState State() const { return state_; }
  std::span<const FixedStateSlotHandle> Handles() const;
  std::span<const FixedStateBinding> Bindings() const;
  std::span<const uint64_t> TargetTokens() const;
  size_t PlannedStagingBytes() const;
  size_t NewSlotCount() const;
  bool CapturesStateUpdates() const;
  // True when model inputs and outputs view the active and inactive persistent banks directly.
  bool UsesDirectBindings() const;
  void CommitPrefix(size_t row, size_t step_tokens, size_t kept_tokens);

  // Commit is split into three phases so a composite Engine transaction can validate and stage all
  // of its resources, synchronize once, and then publish them at a single infallible boundary:
  //
  //   ValidateCommit()  const, throws  - proves every checkpointed slot is exactly where the
  //                                      reservation left it and that the commit cannot overflow or
  //                                      regress committed tokens. Mutates nothing.
  //   PrepareCommit()   throws          - performs all fallible and device work: copies each staged
  //                                      output row into the slot's inactive persistent bank and
  //                                      synchronizes. The active (visible) bank is untouched, so a
  //                                      failure leaves committed state unchanged. Advances the
  //                                      reservation to Prepared.
  //   PublishCommit()   noexcept        - performs no fallible or device work: flips each slot to
  //                                      its freshly written bank, advances state generation and
  //                                      committed tokens, and publishes provisional ownership.
  //
  // Commit() is a convenience wrapper that runs all three phases in order.
  void ValidateCommit() const;
  void PrepareCommit();
  void PublishCommit() noexcept;
  void Commit();
  void Discard();

 private:
  struct Storage;

  FixedStateReservation(FixedStatePool& pool,
                        uint64_t reservation_id,
                        std::unique_ptr<Storage> storage);

  FixedStatePool* pool_{};
  uint64_t reservation_id_{};
  std::unique_ptr<Storage> storage_;
  FixedStateReservationState state_{FixedStateReservationState::Reserved};

  friend class FixedStatePool;
};

class FixedStatePool {
 public:
  FixedStatePool(std::shared_ptr<Model> model, size_t capacity);
  ~FixedStatePool();

  FixedStatePool(const FixedStatePool&) = delete;
  FixedStatePool& operator=(const FixedStatePool&) = delete;

  size_t Capacity() const;
  size_t AvailableSlots() const;
  // Number of slots that currently hold committed ownership. Cheap counterpart to
  // Snapshot().committed_slots for the composite planner's per-step consistency check, so planning
  // does not allocate a full snapshot on the hot path.
  size_t CommittedSlotCount() const;
  size_t PersistentBytes() const;
  size_t ZeroingScratchBytes() const;
  // Retained staging API reports the live input/output binding footprint. Direct bank views overlap
  // PersistentBytes(); fallback and compact-update buffers use dedicated storage.
  size_t ActiveStagingBytes() const;
  // Input/output binding footprint for `row_count` scheduled rows, plus compact capture storage
  // when requested. A pure function of tensor geometry, so composite planning can verify the
  // reservation before its direct-binding eligibility is known.
  size_t PlannedStagingBytes(size_t row_count, bool captures_state_updates = false) const;
  bool SupportsStateUpdates() const;
  size_t StateUpdateCapacity() const;

  FixedStateSlotHandle HandleFor(const void* request_id) const;
  // True when `request_id` currently owns a committed slot. Non-throwing counterpart to HandleFor
  // (which throws when the request owns nothing), for the composite planner's per-row ownership
  // cross-check.
  bool OwnsCommittedSlot(const void* request_id) const;
  std::optional<FixedStateCommittedState> CommittedState(
      const void* request_id) const noexcept;
  // Admits a batch in scheduled row order. Ownership is inferred per request: an identity that
  // already owns a committed slot is treated as resident and keeps that slot; any other identity is
  // admitted provisionally and only becomes discoverable committed ownership on Commit.
  FixedStateReservation Reserve(std::span<const FixedStateReservationRequest> requests);
  void Release(const FixedStateSlotHandle& handle);
  void ValidateRelease(const FixedStateSlotHandle& handle) const;
  // Host-only publication for an unchanged handle accepted by ValidateRelease(). A guard failure
  // is an impossible publication invariant violation and terminates rather than orphaning state.
  void ReleaseValidated(const FixedStateSlotHandle& handle) noexcept;

  uint64_t StateGeneration(const FixedStateSlotHandle& handle) const;
  uint64_t CommittedTokens(const FixedStateSlotHandle& handle) const;
  FixedStatePoolSnapshot Snapshot() const;

 private:
  struct Impl;

  void ValidateCommit(const FixedStateReservation& reservation) const;
  void PrepareCommit(FixedStateReservation& reservation);
  void PublishCommit(FixedStateReservation& reservation) noexcept;
  void ReleaseProvisionalSlots(FixedStateReservation& reservation) noexcept;
  void Discard(FixedStateReservation& reservation) noexcept;
  void Finish(FixedStateReservation& reservation) noexcept;

  std::unique_ptr<Impl> impl_;

  friend class FixedStateReservation;
};

}  // namespace Generators
