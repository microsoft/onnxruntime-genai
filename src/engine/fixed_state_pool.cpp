// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fixed_state_pool.h"

#include <algorithm>
#include <array>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

#include "../generators.h"
#include "../models/model.h"
#include "../models/model_state_manifest.h"
#include "../models/utils.h"

namespace Generators {
namespace {

using StateGroupKind = Config::Model::Decoder::StateGroupKind;

std::string ExpandBinding(const std::string& binding, int layer_id) {
  std::string name{binding};
  name.replace(name.find("%d"), 2, std::to_string(layer_id));
  return name;
}

size_t CheckedMultiply(size_t left, size_t right, std::string_view description) {
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    throw std::runtime_error(
        "Fixed state " + std::string{description} + " exceeds addressable memory.");
  }
  return left * right;
}

size_t CheckedAdd(size_t left, size_t right, std::string_view description) {
  if (right > std::numeric_limits<size_t>::max() - left) {
    throw std::runtime_error(
        "Fixed state " + std::string{description} + " exceeds addressable memory.");
  }
  return left + right;
}

std::vector<int64_t> StorageShape(size_t rows,
                                  std::span<const int64_t> session_shape) {
  std::vector<int64_t> shape{session_shape.begin(), session_shape.end()};
  shape[0] = static_cast<int64_t>(rows);
  return shape;
}

}  // namespace

FixedStateGeometry ValidateFixedStateGeometry(std::string_view input_name,
                                              ONNXTensorElementDataType input_type,
                                              std::span<const int64_t> input_shape,
                                              std::string_view output_name,
                                              ONNXTensorElementDataType output_type,
                                              std::span<const int64_t> output_shape) {
  if (input_type != output_type) {
    throw std::runtime_error(
        "Fixed state binding '" + std::string{input_name} + "' and '" +
        std::string{output_name} + "' have mismatched dtypes.");
  }
  if (input_shape.empty()) {
    throw std::runtime_error(
        "Fixed state input '" + std::string{input_name} +
        "' must have a request batch dimension.");
  }
  if (input_shape.size() != output_shape.size()) {
    throw std::runtime_error(
        "Fixed state binding '" + std::string{input_name} + "' rank " +
        std::to_string(input_shape.size()) + " does not match output '" +
        std::string{output_name} + "' rank " +
        std::to_string(output_shape.size()) + ".");
  }

  // Axis 0 is the per-request batch axis. It may be dynamic (< 0 for symbolic dimensions) or a
  // positive fixed extent, but a fixed zero batch is never serviceable.
  const auto to_fixed = [](int64_t extent) -> size_t {
    return extent > 0 ? static_cast<size_t>(extent) : 0;
  };
  if (input_shape[0] == 0 || output_shape[0] == 0) {
    throw std::runtime_error(
        "Fixed state binding '" + std::string{input_name} +
        "' declares a zero-length batch dimension.");
  }
  const size_t input_fixed_batch = to_fixed(input_shape[0]);
  const size_t output_fixed_batch = to_fixed(output_shape[0]);
  if (input_fixed_batch != 0 && output_fixed_batch != 0 &&
      input_fixed_batch != output_fixed_batch) {
    throw std::runtime_error(
        "Fixed state binding '" + std::string{input_name} +
        "' has asymmetric fixed batch dimensions between input and output.");
  }

  // Every non-batch axis must be statically known, positive, and identical on input and output;
  // rows are copied and gathered by exact byte size, so a dynamic or mismatched non-batch axis
  // cannot be sized.
  size_t row_element_count = 1;
  for (size_t axis = 1; axis < input_shape.size(); ++axis) {
    if (input_shape[axis] <= 0) {
      throw std::runtime_error(
          "Fixed state input '" + std::string{input_name} +
          "' has unsupported dynamic dimension at axis " + std::to_string(axis) + ".");
    }
    if (output_shape[axis] != input_shape[axis]) {
      throw std::runtime_error(
          "Fixed state binding '" + std::string{input_name} +
          "' has incompatible non-batch geometry with output '" +
          std::string{output_name} + "' at axis " + std::to_string(axis) + ".");
    }
    row_element_count = CheckedMultiply(
        row_element_count, static_cast<size_t>(input_shape[axis]), "row element count");
  }

  // A fixed batch on either side constrains the whole binding: even if the input axis is symbolic,
  // a fixed output axis means the session only accepts that exact batch, so the pool must adopt and
  // enforce it rather than treating the batch as free.
  const size_t fixed_batch = input_fixed_batch != 0 ? input_fixed_batch : output_fixed_batch;
  return FixedStateGeometry{row_element_count, fixed_batch};
}

struct FixedStateReservation::Storage {
  std::shared_ptr<Model> model_keepalive;
  std::vector<FixedStateSlotHandle> handles;
  std::vector<bool> provisional;
  std::vector<uint64_t> expected_state_generations;
  std::vector<uint64_t> target_tokens;
  std::vector<std::unique_ptr<OrtValue>> gathered_inputs;
  std::vector<std::unique_ptr<OrtValue>> staged_outputs;
  std::vector<std::unique_ptr<OrtValue>> staged_checkpoints;
  // Per row, the step length and the accepted prefix of it. Zero means "commit the step's final
  // state", which is every row of a non-speculative step.
  std::vector<size_t> commit_step_tokens;
  std::vector<size_t> commit_kept_tokens;
  // The reservation owns the binding name strings so accessors stay valid even after the pool is
  // destroyed; FixedStateBinding::input_name/output_name point into these.
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::vector<std::string> checkpoints_names;
  std::vector<FixedStateBinding> bindings;
  size_t staging_bytes{};
  size_t batch_rows{};
  bool captures_checkpoints{};
};

struct FixedStatePool::Impl {
  struct TensorSpec {
    int layer_id{};
    std::string input_name;
    std::string output_name;
    std::string checkpoints_name;
    Config::Model::Decoder::CheckpointAlignment checkpoint_alignment{};
    size_t checkpoint_count{};
    ONNXTensorElementDataType data_type{};
    std::vector<int64_t> session_shape;
    size_t row_bytes{};
    // Two persistent [capacity, row...] banks per tensor. Each slot reads its currently active bank
    // and a commit stages into the inactive bank, so publish is a bank flip with no device work and
    // a failed prepare cannot corrupt the visible (active) state. See PrepareCommit/PublishCommit.
    std::array<std::unique_ptr<OrtValue>, 2> banks;
    std::unique_ptr<OrtValue> zero_row;  // [1, row...] reusable zeroed gather source.
  };

  struct Slot {
    const void* request_id{};
    uint64_t generation{};
    uint64_t state_generation{};
    uint64_t committed_tokens{};
    uint64_t reservation_id{};
    uint8_t active_bank{};  // Which persistent bank currently holds this slot's visible state.
    FixedStateSlotOwnership ownership{FixedStateSlotOwnership::Free};
  };

  explicit Impl(std::shared_ptr<Model> model_value, size_t capacity_value)
      : model{std::move(model_value)},
        device{model ? model->p_device_kvcache_ : nullptr},
        capacity{capacity_value},
        slots(capacity_value) {}

  Slot* FindSlot(const void* request_id) {
    const auto it = std::find_if(
        slots.begin(), slots.end(),
        [request_id](const Slot& slot) {
          return slot.ownership != FixedStateSlotOwnership::Free &&
                 slot.request_id == request_id;
        });
    return it == slots.end() ? nullptr : &*it;
  }

  const Slot* FindSlot(const void* request_id) const {
    const auto it = std::find_if(
        slots.begin(), slots.end(),
        [request_id](const Slot& slot) {
          return slot.ownership != FixedStateSlotOwnership::Free &&
                 slot.request_id == request_id;
        });
    return it == slots.end() ? nullptr : &*it;
  }

  size_t SlotIndex(const Slot& slot) const {
    return static_cast<size_t>(&slot - slots.data());
  }

  void EnsureHealthy() const {
    if (!healthy) {
      throw std::logic_error(
          "Fixed state pool is unhealthy after a failed commit.");
    }
  }

  void EnsureIdle() const {
    if (active_reservation_id != 0) {
      throw std::logic_error(
          "Fixed state pool already has a live reservation.");
    }
  }

  Slot& ValidateHandle(const FixedStateSlotHandle& handle,
                       FixedStateSlotOwnership required_ownership) {
    if (handle.pool != owner || !handle.request_id ||
        handle.slot >= slots.size()) {
      throw std::runtime_error("Fixed state slot handle does not belong to this pool.");
    }
    auto& slot = slots[handle.slot];
    if (slot.ownership != required_ownership ||
        slot.request_id != handle.request_id ||
        slot.generation != handle.generation) {
      throw std::runtime_error("Fixed state slot handle is stale.");
    }
    return slot;
  }

  const Slot& ValidateCommittedHandle(
      const FixedStateSlotHandle& handle) const {
    if (handle.pool != owner || !handle.request_id ||
        handle.slot >= slots.size()) {
      throw std::runtime_error("Fixed state slot handle does not belong to this pool.");
    }
    const auto& slot = slots[handle.slot];
    if (slot.ownership != FixedStateSlotOwnership::Committed ||
        slot.request_id != handle.request_id ||
        slot.generation != handle.generation) {
      throw std::runtime_error("Fixed state slot handle is stale.");
    }
    return slot;
  }

  FixedStateSlotHandle MakeHandle(const Slot& slot) const {
    return FixedStateSlotHandle{
        owner, slot.request_id, SlotIndex(slot), slot.generation};
  }

  size_t FreeSlotCount() const {
    return static_cast<size_t>(std::count_if(
        slots.begin(), slots.end(), [](const Slot& slot) {
          return slot.ownership == FixedStateSlotOwnership::Free;
        }));
  }

  // Copies a resident slot's visible (active-bank) state into contiguous batch row `batch_row` of
  // the gathered input. Enqueues a device copy only.
  void GatherResidentRow(const TensorSpec& spec, size_t slot_index, uint8_t active_bank,
                         size_t batch_row, OrtValue& gathered) {
    auto destination = ByteWrapTensor(*device, gathered)
                           .subspan(batch_row * spec.row_bytes, spec.row_bytes);
    const auto source = ByteWrapTensor(*device, *spec.banks[active_bank])
                            .subspan(slot_index * spec.row_bytes, spec.row_bytes);
    destination.CopyFrom(source);
  }

  void GatherZeroRow(const TensorSpec& spec, size_t batch_row, OrtValue& gathered) {
    auto destination = ByteWrapTensor(*device, gathered)
                           .subspan(batch_row * spec.row_bytes, spec.row_bytes);
    const auto source = ByteWrapTensor(*device, *spec.zero_row);
    destination.CopyFrom(source);
  }

  // Stages a committed batch row into the slot's inactive persistent bank. The active bank (the
  // visible state) is untouched, so this device copy is fully reversible until PublishCommit flips
  // the slot to `inactive_bank`.
  void StageRowIntoInactiveBank(const TensorSpec& spec, size_t slot_index, uint8_t inactive_bank,
                                size_t batch_row, OrtValue& staged) {
    auto destination = ByteWrapTensor(*device, *spec.banks[inactive_bank])
                           .subspan(slot_index * spec.row_bytes, spec.row_bytes);
    const auto source = ByteWrapTensor(*device, staged)
                            .subspan(batch_row * spec.row_bytes, spec.row_bytes);
    destination.CopyFrom(source);
  }

  // Which checkpoint slot of a `step_tokens`-long step holds the state after its token
  // `kept_tokens - 1`. Left-aligned outputs index from the start of the step, right-aligned ones
  // from its end.
  static size_t CheckpointSlotFor(const TensorSpec& spec, size_t step_tokens, size_t kept_tokens) {
    using Alignment = Config::Model::Decoder::CheckpointAlignment;
    return spec.checkpoint_alignment == Alignment::Left
               ? kept_tokens - 1
               : spec.checkpoint_count - step_tokens + kept_tokens - 1;
  }

  // Same as StageRowIntoInactiveBank but sources one slot of the step's checkpoint series, which is
  // laid out [checkpoint_count, batch_rows, row...].
  void StageCheckpointIntoInactiveBank(const TensorSpec& spec, size_t slot_index,
                                       uint8_t inactive_bank, size_t batch_row, size_t batch_rows,
                                       size_t checkpoint_slot, OrtValue& checkpoints) {
    auto destination = ByteWrapTensor(*device, *spec.banks[inactive_bank])
                           .subspan(slot_index * spec.row_bytes, spec.row_bytes);
    const auto source =
        ByteWrapTensor(*device, checkpoints)
            .subspan((checkpoint_slot * batch_rows + batch_row) * spec.row_bytes, spec.row_bytes);
    destination.CopyFrom(source);
  }

  std::shared_ptr<Model> model;
  DeviceInterface* device{};
  FixedStatePool* owner{};
  size_t capacity{};
  size_t persistent_bytes{};
  size_t zeroing_scratch_bytes{};
  size_t active_staging_bytes{};
  size_t fixed_session_batch_size{};
  size_t checkpoint_count{};
  uint64_t next_reservation_id{1};
  uint64_t active_reservation_id{};
  bool healthy{true};
  FixedStateReservation* active_reservation{};
  std::vector<TensorSpec> tensors;
  std::vector<Slot> slots;
};

FixedStateReservation::FixedStateReservation(
    FixedStatePool& pool,
    uint64_t reservation_id,
    std::unique_ptr<Storage> storage)
    : pool_{&pool},
      reservation_id_{reservation_id},
      storage_{std::move(storage)} {
  pool.impl_->active_reservation = this;
}

FixedStateReservation::FixedStateReservation(
    FixedStateReservation&& other) noexcept
    : pool_{std::exchange(other.pool_, nullptr)},
      reservation_id_{std::exchange(other.reservation_id_, 0)},
      storage_{std::move(other.storage_)},
      state_{std::exchange(other.state_,
                           FixedStateReservationState::Discarded)} {
  if (pool_) {
    pool_->impl_->active_reservation = this;
  }
}

FixedStateReservation::~FixedStateReservation() {
  if (!pool_) {
    return;
  }
  // A reservation holds the pool's single-reservation lock and its staging memory for its whole
  // lifetime; both are released here. An uncommitted (Reserved or Prepared) reservation is discarded
  // first so its provisional slots return to the pool; a committed, discarded, or failed reservation
  // only needs to detach.
  if (state_ == FixedStateReservationState::Reserved ||
      state_ == FixedStateReservationState::Prepared) {
    pool_->Discard(*this);
  }
  pool_->Finish(*this);
}

std::span<const FixedStateSlotHandle> FixedStateReservation::Handles() const {
  return storage_ ? std::span<const FixedStateSlotHandle>{storage_->handles}
                  : std::span<const FixedStateSlotHandle>{};
}

std::span<const FixedStateBinding> FixedStateReservation::Bindings() const {
  return storage_ ? std::span<const FixedStateBinding>{storage_->bindings}
                  : std::span<const FixedStateBinding>{};
}

std::span<const uint64_t> FixedStateReservation::TargetTokens() const {
  return storage_ ? std::span<const uint64_t>{storage_->target_tokens}
                  : std::span<const uint64_t>{};
}

size_t FixedStateReservation::PlannedStagingBytes() const {
  return storage_ ? storage_->staging_bytes : 0;
}

bool FixedStateReservation::CapturesCheckpoints() const {
  return storage_ && storage_->captures_checkpoints;
}

void FixedStateReservation::CommitPrefix(size_t row, size_t step_tokens, size_t kept_tokens) {
  if (!storage_ || !storage_->captures_checkpoints) {
    throw std::logic_error(
        "Committing a prefix requires a reservation that captures checkpoints.");
  }
  if (state_ != FixedStateReservationState::Reserved) {
    throw std::logic_error("Fixed state reservation is no longer accepting prefix commits.");
  }
  if (row >= storage_->handles.size()) {
    throw std::out_of_range("Fixed state prefix commit row is out of range.");
  }
  const size_t window = pool_ ? pool_->CheckpointCount() : 0;
  if (kept_tokens == 0 || kept_tokens > step_tokens || step_tokens > window) {
    throw std::runtime_error(
        "Fixed state prefix commit must keep between one and step_tokens tokens of a step no "
        "longer than the checkpoint window.");
  }
  storage_->commit_step_tokens[row] = step_tokens;
  storage_->commit_kept_tokens[row] = kept_tokens;
}

void FixedStateReservation::ValidateCommit() const {
  if (!pool_ || state_ != FixedStateReservationState::Reserved) {
    throw std::logic_error(
        "Fixed state reservation can only be validated for commit while reserved.");
  }
  pool_->ValidateCommit(*this);
}

void FixedStateReservation::PrepareCommit() {
  if (!pool_ || state_ != FixedStateReservationState::Reserved) {
    throw std::logic_error(
        "Fixed state reservation can only be prepared for commit once, while reserved.");
  }
  pool_->PrepareCommit(*this);
}

void FixedStateReservation::PublishCommit() noexcept {
  if (!pool_) {
    // The pool was destroyed after PrepareCommit; there is nothing left to publish into.
    return;
  }
  // PublishCommit must only be reached after a successful PrepareCommit. Anything else is a caller
  // sequencing bug that would silently diverge fixed state from the rest of a composite commit.
  if (state_ != FixedStateReservationState::Prepared) {
    std::terminate();
  }
  pool_->PublishCommit(*this);
}

void FixedStateReservation::Commit() {
  ValidateCommit();
  PrepareCommit();
  PublishCommit();
}

void FixedStateReservation::Discard() {
  if (state_ == FixedStateReservationState::Committed) {
    throw std::logic_error(
        "Committed fixed state reservation cannot be discarded.");
  }
  // Discarding an already-terminal reservation is a harmless no-op. In particular a Failed
  // reservation already released its provisional slots when PrepareCommit failed, and a reservation
  // whose pool was destroyed has nothing left to release.
  if (!pool_ ||
      state_ == FixedStateReservationState::Discarded ||
      state_ == FixedStateReservationState::Failed) {
    return;
  }
  // Discard is valid from Reserved or Prepared. A Prepared reservation only wrote into inactive
  // banks, so nothing visible has to be undone; only provisional slots are returned to the pool.
  pool_->Discard(*this);
}

FixedStatePool::FixedStatePool(std::shared_ptr<Model> model,
                               const ModelStateManifest& manifest,
                               size_t capacity)
    : impl_{std::make_unique<Impl>(std::move(model), capacity)} {
  if (!impl_->model) {
    throw std::invalid_argument("Fixed state pool requires a model.");
  }
  if (capacity == 0) {
    throw std::invalid_argument(
        "Fixed state pool capacity must be positive.");
  }
  if (capacity > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
    throw std::invalid_argument(
        "Fixed state pool capacity exceeds the supported tensor dimension.");
  }
  if (!impl_->device) {
    throw std::runtime_error(
        "Fixed state pool requires a model state device.");
  }
  if (impl_->device->GetType() != DeviceType::CPU &&
      impl_->device->GetType() != DeviceType::CUDA) {
    throw std::runtime_error(
        "Fixed state pools currently support only CPU and CUDA devices.");
  }

  impl_->owner = this;
  manifest.ValidateSession(impl_->model->session_info_);

  for (const auto& group : manifest.StateGroups()) {
    if (group.kind != StateGroupKind::Fixed) {
      continue;
    }
    for (const int layer_id : group.layer_ids) {
      Impl::TensorSpec spec;
      spec.layer_id = layer_id;
      spec.input_name = ExpandBinding(group.state->input, layer_id);
      spec.output_name = ExpandBinding(group.state->output, layer_id);
      if (group.checkpoint_count > 0) {
        spec.checkpoints_name = ExpandBinding(group.state->checkpoints, layer_id);
        spec.checkpoint_count = static_cast<size_t>(group.checkpoint_count);
        spec.checkpoint_alignment = group.checkpoint_alignment;
      }
      spec.data_type =
          impl_->model->session_info_.GetInputDataType(spec.input_name);
      spec.session_shape =
          impl_->model->session_info_.GetInputShape(spec.input_name);

      const auto output_type =
          impl_->model->session_info_.GetOutputDataType(spec.output_name);
      const auto output_shape =
          impl_->model->session_info_.GetOutputShape(spec.output_name);
      const auto geometry = ValidateFixedStateGeometry(
          spec.input_name, spec.data_type, spec.session_shape,
          spec.output_name, output_type, output_shape);

      if (geometry.fixed_batch_size != 0) {
        if (impl_->fixed_session_batch_size != 0 &&
            impl_->fixed_session_batch_size != geometry.fixed_batch_size) {
          throw std::runtime_error(
              "Fixed state inputs declare inconsistent fixed batch dimensions.");
        }
        impl_->fixed_session_batch_size = geometry.fixed_batch_size;
      }

      spec.row_bytes = CheckedMultiply(
          geometry.row_element_count, Ort::SizeOf(spec.data_type), "row size");
      // Two persistent banks per tensor so publish can flip banks without any device copy.
      impl_->persistent_bytes = CheckedAdd(
          impl_->persistent_bytes,
          CheckedMultiply(CheckedMultiply(capacity, spec.row_bytes, "persistent allocation"),
                          spec.banks.size(), "persistent allocation"),
          "persistent allocation");
      impl_->zeroing_scratch_bytes = CheckedAdd(
          impl_->zeroing_scratch_bytes, spec.row_bytes,
          "zeroing scratch allocation");

      const auto bank_shape = StorageShape(capacity, spec.session_shape);
      const auto zero_shape = StorageShape(1, spec.session_shape);
      for (auto& bank : spec.banks) {
        bank = OrtValue::CreateTensor(
            impl_->device->GetAllocator(), bank_shape, spec.data_type);
      }
      spec.zero_row = OrtValue::CreateTensor(
          impl_->device->GetAllocator(), zero_shape, spec.data_type);
      // Only the reusable zero row is zeroed: it is the gather source for freshly admitted rows. A
      // slot's active bank is written by the commit that publishes it before it is ever gathered,
      // and the inactive bank is only ever written (never read) before it becomes active, so the
      // persistent banks never need construction-time zeroing.
      ByteWrapTensor(*impl_->device, *spec.zero_row).Zero();

      impl_->tensors.push_back(std::move(spec));
    }
  }

  if (impl_->tensors.empty()) {
    throw std::runtime_error(
        "Fixed state pool requires at least one fixed state binding.");
  }
  // A rollback has to move every fixed group's state together, so a partially checkpointed model
  // could never actually roll back. Require all-or-nothing with a shared window.
  const size_t checkpoint_count = impl_->tensors.front().checkpoint_count;
  for (const auto& spec : impl_->tensors) {
    if (spec.checkpoint_count != checkpoint_count) {
      throw std::runtime_error(
          "Fixed state bindings declare inconsistent checkpoint windows.");
    }
  }
  impl_->checkpoint_count = checkpoint_count;
  // A session with a fixed batch dimension can only ever be served with exactly that many rows, so
  // a pool too small to hold a single batch could never satisfy any reservation.
  if (impl_->fixed_session_batch_size != 0 &&
      impl_->fixed_session_batch_size > capacity) {
    throw std::runtime_error(
        "Fixed state pool capacity is smaller than the session's fixed batch dimension.");
  }
  impl_->device->Synchronize();
}

FixedStatePool::~FixedStatePool() {
  // Neutralize a still-live reservation so its accessors stay valid but any further transactional
  // call fails cleanly. Only an in-flight (Reserved/Prepared) reservation is downgraded to Failed;
  // a terminal reservation has already detached itself (see Finish) and must keep its final state.
  if (impl_->active_reservation) {
    impl_->active_reservation->pool_ = nullptr;
    impl_->active_reservation->reservation_id_ = 0;
    if (impl_->active_reservation->state_ == FixedStateReservationState::Reserved ||
        impl_->active_reservation->state_ == FixedStateReservationState::Prepared) {
      impl_->active_reservation->state_ = FixedStateReservationState::Failed;
    }
    impl_->active_reservation = nullptr;
  }
}

size_t FixedStatePool::Capacity() const {
  return impl_->capacity;
}

size_t FixedStatePool::AvailableSlots() const {
  return impl_->FreeSlotCount();
}

size_t FixedStatePool::CommittedSlotCount() const {
  return static_cast<size_t>(std::count_if(
      impl_->slots.begin(), impl_->slots.end(), [](const Impl::Slot& slot) {
        return slot.ownership == FixedStateSlotOwnership::Committed;
      }));
}

size_t FixedStatePool::SessionBatchSize() const {
  return impl_->fixed_session_batch_size;
}

size_t FixedStatePool::PersistentBytes() const {
  return impl_->persistent_bytes;
}

size_t FixedStatePool::ZeroingScratchBytes() const {
  return impl_->zeroing_scratch_bytes;
}

size_t FixedStatePool::ActiveStagingBytes() const {
  return impl_->active_staging_bytes;
}

size_t FixedStatePool::PlannedStagingBytes(size_t row_count, bool capture_checkpoints) const {
  // One gather input and one staged output per fixed tensor, each `row_count` rows wide, plus the
  // checkpoint series when the step captures it. This mirrors the accumulation in Reserve() so the
  // plan and the resulting reservation agree exactly.
  const size_t copies_per_tensor =
      capture_checkpoints ? 2 + impl_->checkpoint_count : 2;
  size_t bytes = 0;
  for (const auto& spec : impl_->tensors) {
    bytes = CheckedAdd(
        bytes,
        CheckedMultiply(
            CheckedMultiply(row_count, spec.row_bytes, "staging allocation"),
            copies_per_tensor, "staging allocation"),
        "staging allocation");
  }
  return bytes;
}

bool FixedStatePool::SupportsCheckpoints() const {
  return impl_->checkpoint_count != 0;
}

size_t FixedStatePool::CheckpointCount() const {
  return impl_->checkpoint_count;
}

FixedStateSlotHandle FixedStatePool::HandleFor(
    const void* request_id) const {
  if (!request_id) {
    throw std::invalid_argument(
        "Fixed state slot requires a request identity.");
  }
  const auto* slot = impl_->FindSlot(request_id);
  if (!slot || slot->ownership != FixedStateSlotOwnership::Committed) {
    throw std::runtime_error(
        "Request does not own a committed fixed state slot.");
  }
  return impl_->MakeHandle(*slot);
}

bool FixedStatePool::OwnsCommittedSlot(const void* request_id) const {
  if (!request_id) {
    return false;
  }
  const auto* slot = impl_->FindSlot(request_id);
  return slot && slot->ownership == FixedStateSlotOwnership::Committed;
}

FixedStateReservation FixedStatePool::Reserve(
    std::span<const FixedStateReservationRequest> requests,
    bool capture_checkpoints) {
  impl_->EnsureHealthy();
  impl_->EnsureIdle();
  if (requests.empty()) {
    throw std::invalid_argument(
        "Fixed state reservation must contain at least one request.");
  }
  if (capture_checkpoints && impl_->checkpoint_count == 0) {
    throw std::runtime_error(
        "This model's fixed state bindings declare no checkpoints output.");
  }
  if (requests.size() > impl_->capacity) {
    throw std::runtime_error(
        "Fixed state reservation exceeds pool capacity.");
  }
  if (impl_->fixed_session_batch_size != 0 &&
      impl_->fixed_session_batch_size != requests.size()) {
    throw std::runtime_error(
        "Fixed state reservation does not match the session's fixed batch dimension.");
  }

  const uint64_t reservation_id = impl_->next_reservation_id;
  if (reservation_id == 0 ||
      reservation_id == std::numeric_limits<uint64_t>::max()) {
    throw std::overflow_error(
        "Fixed state reservation generation is exhausted.");
  }

  // Phase 1: host-only planning. Infer per-request ownership, choose provisional slots, and build
  // handles. No device work is enqueued and no visible slot state is mutated yet, so any exception
  // here leaves the pool exactly as it was.
  struct RowPlan {
    const void* request_id{};
    size_t slot_index{};
    bool provisional{};
    uint64_t handle_generation{};
    uint64_t expected_state_generation{};
    uint64_t target_tokens{};
  };
  std::vector<RowPlan> plan;
  plan.reserve(requests.size());
  std::unordered_set<const void*> request_ids;
  request_ids.reserve(requests.size());
  std::vector<char> slot_taken(impl_->slots.size(), 0);
  for (const auto& slot : impl_->slots) {
    if (slot.ownership != FixedStateSlotOwnership::Free) {
      slot_taken[impl_->SlotIndex(slot)] = 1;
    }
  }

  const auto next_free_slot = [&]() -> size_t {
    for (size_t index = 0; index < impl_->slots.size(); ++index) {
      if (!slot_taken[index]) {
        return index;
      }
    }
    throw std::runtime_error(
        "Not enough free slots for the complete fixed state reservation.");
  };

  for (const auto& request : requests) {
    const void* request_id = request.request_id;
    if (!request_id || !request_ids.insert(request_id).second) {
      throw std::runtime_error(
          "Fixed state reservation contains an invalid or duplicate request.");
    }
    const Impl::Slot* slot = impl_->FindSlot(request_id);
    RowPlan row;
    row.request_id = request_id;
    row.target_tokens = request.target_tokens;
    if (slot && slot->ownership == FixedStateSlotOwnership::Committed) {
      row.slot_index = impl_->SlotIndex(*slot);
      row.provisional = false;
      row.handle_generation = slot->generation;
      row.expected_state_generation = slot->state_generation;
    } else {
      const size_t slot_index = next_free_slot();
      const auto& free_slot = impl_->slots[slot_index];
      if (free_slot.generation == std::numeric_limits<uint64_t>::max()) {
        throw std::overflow_error(
            "Fixed state slot generation is exhausted.");
      }
      slot_taken[slot_index] = 1;
      row.slot_index = slot_index;
      row.provisional = true;
      row.handle_generation = free_slot.generation + 1;
      row.expected_state_generation = 0;
    }
    plan.push_back(row);
  }

  // Phase 2: allocate staging tensors and record binding metadata (host allocations only).
  auto storage = std::make_unique<FixedStateReservation::Storage>();
  storage->model_keepalive = impl_->model;
  storage->handles.resize(requests.size());
  storage->provisional.resize(requests.size());
  storage->expected_state_generations.resize(requests.size());
  storage->target_tokens.resize(requests.size());
  storage->commit_step_tokens.assign(requests.size(), 0);
  storage->commit_kept_tokens.assign(requests.size(), 0);
  storage->batch_rows = requests.size();
  storage->captures_checkpoints = capture_checkpoints;
  storage->gathered_inputs.reserve(impl_->tensors.size());
  storage->staged_outputs.reserve(impl_->tensors.size());
  storage->staged_checkpoints.reserve(capture_checkpoints ? impl_->tensors.size() : 0);
  storage->input_names.reserve(impl_->tensors.size());
  storage->output_names.reserve(impl_->tensors.size());
  storage->checkpoints_names.reserve(capture_checkpoints ? impl_->tensors.size() : 0);
  storage->bindings.reserve(impl_->tensors.size());

  const size_t batch_rows = requests.size();
  for (auto& spec : impl_->tensors) {
    const auto shape = StorageShape(batch_rows, spec.session_shape);
    auto gathered = OrtValue::CreateTensor(
        impl_->device->GetAllocator(), shape, spec.data_type);
    auto staged = OrtValue::CreateTensor(
        impl_->device->GetAllocator(), shape, spec.data_type);
    std::unique_ptr<OrtValue> checkpoints;
    if (capture_checkpoints) {
      auto checkpoint_shape = shape;
      checkpoint_shape.insert(checkpoint_shape.begin(),
                              static_cast<int64_t>(spec.checkpoint_count));
      checkpoints = OrtValue::CreateTensor(
          impl_->device->GetAllocator(), checkpoint_shape, spec.data_type);
    }

    storage->staging_bytes = CheckedAdd(
        storage->staging_bytes,
        CheckedMultiply(
            CheckedMultiply(batch_rows, spec.row_bytes, "staging allocation"),
            capture_checkpoints ? 2 + spec.checkpoint_count : 2, "staging allocation"),
        "staging allocation");
    storage->input_names.push_back(spec.input_name);
    storage->output_names.push_back(spec.output_name);
    if (capture_checkpoints) {
      storage->checkpoints_names.push_back(spec.checkpoints_name);
    }
    storage->bindings.push_back(FixedStateBinding{
        StateGroupKind::Fixed,
        spec.layer_id,
        storage->input_names.back().c_str(),
        gathered.get(),
        storage->output_names.back().c_str(),
        staged.get(),
        capture_checkpoints ? storage->checkpoints_names.back().c_str() : nullptr,
        checkpoints.get(),
    });
    storage->gathered_inputs.push_back(std::move(gathered));
    storage->staged_outputs.push_back(std::move(staged));
    if (capture_checkpoints) {
      storage->staged_checkpoints.push_back(std::move(checkpoints));
    }
  }

  for (size_t row = 0; row < plan.size(); ++row) {
    storage->handles[row] = FixedStateSlotHandle{
        this, plan[row].request_id, plan[row].slot_index,
        plan[row].handle_generation};
    storage->provisional[row] = plan[row].provisional;
    storage->expected_state_generations[row] =
        plan[row].expected_state_generation;
    storage->target_tokens[row] = plan[row].target_tokens;
  }

  // Phase 3: enqueue the gather copies, then synchronize. Once device work is in flight the staging
  // buffers must outlive it, so a failure drains the device before the buffers unwind and marks the
  // pool unhealthy. No visible slot state has changed yet, so there is nothing to roll back.
  try {
    for (size_t tensor_index = 0; tensor_index < impl_->tensors.size();
         ++tensor_index) {
      const auto& spec = impl_->tensors[tensor_index];
      auto& gathered = *storage->gathered_inputs[tensor_index];
      for (size_t row = 0; row < plan.size(); ++row) {
        if (plan[row].provisional) {
          impl_->GatherZeroRow(spec, row, gathered);
        } else {
          impl_->GatherResidentRow(spec, plan[row].slot_index,
                                   impl_->slots[plan[row].slot_index].active_bank,
                                   row, gathered);
        }
      }
    }
    impl_->device->Synchronize();
  } catch (...) {
    // Record the failure before draining: a sticky device error can make the drain itself throw,
    // and the pool must still end up marked unhealthy. The drain is best-effort because the staging
    // buffers are about to unwind and must not still be referenced by in-flight copies.
    impl_->healthy = false;
    try {
      impl_->device->Synchronize();
    } catch (...) {
    }
    throw;
  }

  // Phase 4: publish provisional ownership. This is host bookkeeping only; the gather is complete.
  for (size_t row = 0; row < plan.size(); ++row) {
    if (!plan[row].provisional) {
      continue;
    }
    auto& slot = impl_->slots[plan[row].slot_index];
    slot.request_id = plan[row].request_id;
    ++slot.generation;
    slot.state_generation = 0;
    slot.committed_tokens = 0;
    slot.reservation_id = reservation_id;
    slot.ownership = FixedStateSlotOwnership::Reserved;
  }

  ++impl_->next_reservation_id;
  impl_->active_reservation_id = reservation_id;
  impl_->active_staging_bytes = storage->staging_bytes;
  return FixedStateReservation{*this, reservation_id, std::move(storage)};
}

void FixedStatePool::Release(const FixedStateSlotHandle& handle) {
  // Deliberately not gated on EnsureHealthy(): releasing a committed slot only resets host-side
  // ownership metadata and never reads or writes a persistent bank, so it is safe even after a
  // failed commit left the pool unhealthy. Teardown depends on this -- a fatal device error marks
  // both the pool and the Engine unhealthy, and the Engine must still be able to remove/reap its
  // requests (releasing paged blocks and this fixed slot) without the release itself throwing and
  // masking the recorded fatal error. EnsureIdle() still holds: a live reservation owns the pool.
  impl_->EnsureIdle();
  auto& slot = impl_->ValidateHandle(
      handle, FixedStateSlotOwnership::Committed);
  // No zeroing on release: the slot's persistent banks are never read again until a future
  // admission either gathers the reusable zero row (a new owner) or a commit fully overwrites the
  // bank it publishes.
  slot.request_id = nullptr;
  slot.state_generation = 0;
  slot.committed_tokens = 0;
  slot.reservation_id = 0;
  slot.ownership = FixedStateSlotOwnership::Free;
}

uint64_t FixedStatePool::StateGeneration(
    const FixedStateSlotHandle& handle) const {
  return impl_->ValidateCommittedHandle(handle).state_generation;
}

uint64_t FixedStatePool::CommittedTokens(
    const FixedStateSlotHandle& handle) const {
  return impl_->ValidateCommittedHandle(handle).committed_tokens;
}

FixedStatePoolSnapshot FixedStatePool::Snapshot() const {
  FixedStatePoolSnapshot snapshot;
  snapshot.capacity = impl_->capacity;
  snapshot.persistent_bytes = impl_->persistent_bytes;
  snapshot.zeroing_scratch_bytes = impl_->zeroing_scratch_bytes;
  snapshot.active_staging_bytes = impl_->active_staging_bytes;
  snapshot.healthy = impl_->healthy;
  snapshot.slots.reserve(impl_->slots.size());
  for (size_t index = 0; index < impl_->slots.size(); ++index) {
    const auto& slot = impl_->slots[index];
    switch (slot.ownership) {
      case FixedStateSlotOwnership::Free:
        ++snapshot.free_slots;
        break;
      case FixedStateSlotOwnership::Reserved:
        ++snapshot.reserved_slots;
        break;
      case FixedStateSlotOwnership::Committed:
        ++snapshot.committed_slots;
        break;
    }
    snapshot.slots.push_back(FixedStateSlotSnapshot{
        slot.request_id,
        index,
        slot.generation,
        slot.state_generation,
        slot.committed_tokens,
        slot.ownership,
    });
  }
  return snapshot;
}

void FixedStatePool::ValidateCommit(
    const FixedStateReservation& reservation) const {
  impl_->EnsureHealthy();
  if (impl_->active_reservation_id != reservation.reservation_id_ ||
      !reservation.storage_) {
    throw std::logic_error(
        "Fixed state reservation is not active in this pool.");
  }

  const auto& storage = *reservation.storage_;
  // Prove every checkpointed slot is exactly where the reservation left it, and that publishing it
  // cannot overflow a generation or regress committed tokens. This is the only fallible part of the
  // commit, so a composite transaction can validate every resource before any of them is published.
  for (size_t row = 0; row < storage.handles.size(); ++row) {
    const auto& handle = storage.handles[row];
    if (handle.pool != reservation.pool_ || handle.slot >= impl_->slots.size()) {
      throw std::logic_error(
          "Fixed state changed after the reservation was prepared.");
    }
    const auto& slot = impl_->slots[handle.slot];
    const auto required_ownership =
        storage.provisional[row]
            ? FixedStateSlotOwnership::Reserved
            : FixedStateSlotOwnership::Committed;
    if (slot.ownership != required_ownership ||
        slot.request_id != handle.request_id ||
        slot.generation != handle.generation ||
        slot.state_generation != storage.expected_state_generations[row] ||
        (storage.provisional[row] &&
         slot.reservation_id != reservation.reservation_id_)) {
      throw std::logic_error(
          "Fixed state changed after the reservation was prepared.");
    }
    if (slot.state_generation == std::numeric_limits<uint64_t>::max()) {
      throw std::overflow_error(
          "Fixed state generation is exhausted.");
    }
    if (storage.target_tokens[row] < slot.committed_tokens) {
      throw std::logic_error(
          "Fixed state commit would regress a request's committed token count.");
    }
  }
}

void FixedStatePool::PrepareCommit(FixedStateReservation& reservation) {
  // Re-run validation so PrepareCommit is safe to call on its own, then perform all device work.
  ValidateCommit(reservation);

  auto& storage = *reservation.storage_;
  // Stage every committed row into each slot's inactive bank and synchronize. The active (visible)
  // bank is never touched, so a failure here leaves committed state exactly as it was; we still
  // drain the device before the staging buffers unwind and mark the pool unhealthy because the
  // inactive banks may be left partially written.
  try {
    for (size_t tensor_index = 0; tensor_index < impl_->tensors.size();
         ++tensor_index) {
      const auto& spec = impl_->tensors[tensor_index];
      auto& staged = *storage.staged_outputs[tensor_index];
      for (size_t row = 0; row < storage.handles.size(); ++row) {
        const auto& slot = impl_->slots[storage.handles[row].slot];
        const uint8_t inactive_bank = slot.active_bank ^ 1u;
        const size_t kept_tokens = storage.commit_kept_tokens[row];
        if (kept_tokens == 0 || kept_tokens == storage.commit_step_tokens[row]) {
          impl_->StageRowIntoInactiveBank(
              spec, storage.handles[row].slot, inactive_bank, row, staged);
          continue;
        }
        impl_->StageCheckpointIntoInactiveBank(
            spec, storage.handles[row].slot, inactive_bank, row, storage.batch_rows,
            Impl::CheckpointSlotFor(spec, storage.commit_step_tokens[row], kept_tokens),
            *storage.staged_checkpoints[tensor_index]);
      }
    }
    impl_->device->Synchronize();
  } catch (...) {
    // Record the failure and release this reservation's provisional slots before draining. The
    // active banks were never touched, so committed state is intact; the inactive banks may be
    // partially written, so the pool is marked unhealthy. The drain is best-effort because a sticky
    // device error can make Synchronize itself throw, and the staging buffers are about to unwind.
    impl_->healthy = false;
    reservation.state_ = FixedStateReservationState::Failed;
    ReleaseProvisionalSlots(reservation);
    try {
      impl_->device->Synchronize();
    } catch (...) {
    }
    throw;
  }

  reservation.state_ = FixedStateReservationState::Prepared;
}

void FixedStatePool::PublishCommit(FixedStateReservation& reservation) noexcept {
  auto& storage = *reservation.storage_;
  // Host bookkeeping only: PrepareCommit already wrote and synchronized the inactive banks and
  // ValidateCommit already proved no generation overflow, so every step here is infallible. Flip
  // each slot to its freshly written bank and publish generation, committed tokens, and ownership.
  for (size_t row = 0; row < storage.handles.size(); ++row) {
    auto& slot = impl_->slots[storage.handles[row].slot];
    slot.active_bank ^= 1u;
    ++slot.state_generation;
    slot.committed_tokens = storage.target_tokens[row];
    if (storage.provisional[row]) {
      slot.reservation_id = 0;
      slot.ownership = FixedStateSlotOwnership::Committed;
    }
  }
  reservation.state_ = FixedStateReservationState::Committed;
}

void FixedStatePool::ReleaseProvisionalSlots(
    FixedStateReservation& reservation) noexcept {
  if (!reservation.storage_ || reservation.reservation_id_ == 0 ||
      impl_->active_reservation_id != reservation.reservation_id_) {
    return;
  }
  for (size_t row = 0; row < reservation.storage_->handles.size(); ++row) {
    if (!reservation.storage_->provisional[row]) {
      continue;
    }
    const auto& handle = reservation.storage_->handles[row];
    auto& slot = impl_->slots[handle.slot];
    if (slot.ownership == FixedStateSlotOwnership::Reserved &&
        slot.request_id == handle.request_id &&
        slot.generation == handle.generation &&
        slot.reservation_id == reservation.reservation_id_) {
      slot.request_id = nullptr;
      slot.state_generation = 0;
      slot.committed_tokens = 0;
      slot.reservation_id = 0;
      slot.ownership = FixedStateSlotOwnership::Free;
    }
  }
}

void FixedStatePool::Discard(
    FixedStateReservation& reservation) noexcept {
  // A discarded reservation leaves every resident slot's active state, generation, and committed
  // tokens exactly as they were and returns only its provisional slots to the free pool.
  ReleaseProvisionalSlots(reservation);
  reservation.state_ = FixedStateReservationState::Discarded;
}

void FixedStatePool::Finish(
    FixedStateReservation& reservation) noexcept {
  if (reservation.reservation_id_ != 0 &&
      impl_->active_reservation_id == reservation.reservation_id_) {
    impl_->active_reservation_id = 0;
    impl_->active_staging_bytes = 0;
  }
  if (impl_->active_reservation == &reservation) {
    impl_->active_reservation = nullptr;
  }
  reservation.pool_ = nullptr;
  reservation.reservation_id_ = 0;
}

}  // namespace Generators
