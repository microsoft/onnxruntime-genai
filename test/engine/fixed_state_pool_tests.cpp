// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

#include <gtest/gtest.h>

#include "engine/fixed_state_pool.h"
#include "engine_test_helpers.h"
#include "models/model_state_manifest.h"

namespace Generators {
namespace test {
namespace {

const char kRequestStorageA{};
const char kRequestStorageB{};
const char kRequestStorageC{};
const void* const kRequestA = &kRequestStorageA;
const void* const kRequestB = &kRequestStorageB;
const void* const kRequestC = &kRequestStorageC;

using Request = FixedStateReservationRequest;

// Builds a one-request reservation input in scheduled row order.
std::array<Request, 1> One(const void* request_id, uint64_t target_tokens = 1,
                           size_t capture_count = 0) {
  return {Request{request_id, target_tokens, capture_count}};
}

size_t RowElements(const OrtValue& tensor) {
  const auto shape = tensor.GetTensorTypeAndShapeInfo()->GetShape();
  size_t row_elements = 1;
  for (size_t axis = 1; axis < shape.size(); ++axis) {
    row_elements *= static_cast<size_t>(shape[axis]);
  }
  return row_elements;
}

void FillStagedRow(const FixedStateBinding& binding, size_t row, float value) {
  const auto row_elements = RowElements(*binding.output);
  auto* data = binding.output->GetTensorMutableData<float>();
  std::fill_n(data + row * row_elements, row_elements, value);
}

void FillStagedRows(FixedStateReservation& reservation, size_t row, float value) {
  for (const auto& binding : reservation.Bindings()) {
    FillStagedRow(binding, row, value);
  }
}

void ExpectInputRow(const FixedStateBinding& binding, size_t row, float expected) {
  const auto row_elements = RowElements(*binding.input);
  const auto* data = binding.input->GetTensorData<float>();
  for (size_t index = 0; index < row_elements; ++index) {
    EXPECT_FLOAT_EQ(data[row * row_elements + index], expected);
  }
}

void ExpectInputRow(const FixedStateBinding& binding, size_t row,
                    std::span<const float> expected) {
  const auto row_elements = RowElements(*binding.input);
  ASSERT_EQ(expected.size(), row_elements);
  const auto* data = binding.input->GetTensorData<float>() + row * row_elements;
  for (size_t index = 0; index < row_elements; ++index) {
    EXPECT_FLOAT_EQ(data[index], expected[index]) << "element " << index;
  }
}

void ExpectInputRows(const FixedStateReservation& reservation, size_t row, float expected) {
  for (const auto& binding : reservation.Bindings()) {
    ExpectInputRow(binding, row, expected);
  }
}

// Writes one slot of a binding's checkpoint series, laid out [checkpoint_count, row_count, row...].
void FillCheckpointRow(const FixedStateBinding& binding, size_t checkpoint_slot, size_t row_count,
                       size_t row, float value) {
  const auto shape = binding.checkpoints->GetTensorTypeAndShapeInfo()->GetShape();
  size_t row_elements = 1;
  for (size_t axis = 2; axis < shape.size(); ++axis) {
    row_elements *= static_cast<size_t>(shape[axis]);
  }
  auto* data = binding.checkpoints->GetTensorMutableData<float>();
  std::fill_n(data + (checkpoint_slot * row_count + row) * row_elements, row_elements, value);
}

// Fills every checkpoint slot of every binding so a wrong slot selection is always detectable:
// slot j of a left-aligned binding and slot j of a right-aligned one both get `base + j`.
void FillCheckpointSeries(FixedStateReservation& reservation, size_t row_count, size_t row,
                          float base) {
  for (const auto& binding : reservation.Bindings()) {
    const auto slots =
        static_cast<size_t>(binding.checkpoints->GetTensorTypeAndShapeInfo()->GetShape()[0]);
    for (size_t slot = 0; slot < slots; ++slot) {
      FillCheckpointRow(binding, slot, row_count, row, base + static_cast<float>(slot));
    }
  }
}

void FillConvUpdates(const FixedStateBinding& binding, size_t row,
                     std::span<const float> values) {
  ASSERT_NE(binding.state_update_value, nullptr);
  const auto shape = binding.state_update_value->GetTensorTypeAndShapeInfo()->GetShape();
  const size_t row_elements = static_cast<size_t>(shape[1] * shape[2]);
  ASSERT_EQ(values.size(), row_elements);
  auto* data = binding.state_update_value->GetTensorMutableData<float>();
  std::copy(values.begin(), values.end(), data + row * row_elements);
}

void FillGdnUpdates(const FixedStateBinding& binding, size_t row,
                    std::span<const float> decay,
                    std::span<const float> key,
                    std::span<const float> delta) {
  ASSERT_NE(binding.state_update_decay, nullptr);
  ASSERT_NE(binding.state_update_key, nullptr);
  ASSERT_NE(binding.state_update_delta, nullptr);
  const auto copy_row = [row](OrtValue& tensor, std::span<const float> values) {
    const size_t row_elements = RowElements(tensor);
    ASSERT_EQ(values.size(), row_elements);
    auto* data = tensor.GetTensorMutableData<float>();
    std::copy(values.begin(), values.end(), data + row * row_elements);
  };
  copy_row(*binding.state_update_decay, decay);
  copy_row(*binding.state_update_key, key);
  copy_row(*binding.state_update_delta, delta);
}

// ONNX element type shorthands for the direct geometry tests.
constexpr auto kFloat = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
constexpr auto kDouble = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;

class FixedStatePoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    model_ = LoadSyntheticHybridModel();
    manifest_ = std::make_unique<ModelStateManifest>(model_->config_->model.decoder);
  }

  std::unique_ptr<FixedStatePool> MakePool(size_t capacity = 4) {
    return std::make_unique<FixedStatePool>(model_, *manifest_, capacity);
  }

  // Admits a fresh request and commits it with every state row filled with `value`, returning the
  // now-committed slot handle. This is how the tests create resident state without a standalone
  // Allocate surface.
  FixedStateSlotHandle MakeResident(FixedStatePool& pool, const void* request_id, float value,
                                    uint64_t target_tokens = 1) {
    auto requests = One(request_id, target_tokens);
    auto reservation = pool.Reserve(requests);
    FillStagedRows(reservation, 0, value);
    reservation.Commit();
    return pool.HandleFor(request_id);
  }

  std::shared_ptr<Model> model_;
  std::unique_ptr<ModelStateManifest> manifest_;
};

TEST_F(FixedStatePoolTest, UsesManifestBindingOrderAndSessionGeometry) {
  auto pool = MakePool();
  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);

  ASSERT_EQ(reservation.Bindings().size(), 4u);
  using Kind = Config::Model::Decoder::StateGroupKind;
  for (const auto& binding : reservation.Bindings()) {
    EXPECT_EQ(binding.kind, Kind::Fixed);
  }
  // Convolution group first, then recurrent group, each in layer order.
  EXPECT_EQ(reservation.Bindings()[0].layer_id, 0);
  EXPECT_STREQ(reservation.Bindings()[0].input_name, "past_conv.0");
  EXPECT_STREQ(reservation.Bindings()[0].output_name, "present_conv.0");
  EXPECT_EQ(reservation.Bindings()[1].layer_id, 3);
  EXPECT_STREQ(reservation.Bindings()[2].input_name, "past_recurrent.2");
  EXPECT_STREQ(reservation.Bindings()[3].output_name, "present_recurrent.5");

  for (const auto& binding : reservation.Bindings()) {
    EXPECT_NE(binding.input, binding.output);
    EXPECT_NE(binding.input->GetTensorMutableData<float>(),
              binding.output->GetTensorMutableData<float>());
  }

  EXPECT_EQ(reservation.Bindings()[0].input->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{1, 2, 3}));
  EXPECT_EQ(reservation.Bindings()[2].output->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{1, 2, 2, 2}));
  EXPECT_EQ(reservation.Handles()[0].request_id, kRequestA);
  ASSERT_EQ(reservation.TargetTokens().size(), 1u);
  EXPECT_EQ(reservation.TargetTokens()[0], 1u);
}

TEST_F(FixedStatePoolTest, FreshRowsGatherZeroAndCommitPublishes) {
  auto pool = MakePool(1);
  {
    auto requests = One(kRequestA);
    auto reservation = pool->Reserve(requests);
    for (const auto& binding : reservation.Bindings()) {
      ExpectInputRow(binding, 0, 0.0f);  // Fresh admission gathers the reusable zero row.
    }
    FillStagedRows(reservation, 0, 7.0f);
    reservation.Commit();
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 7.0f);  // Committed state is now gathered for the resident request.
}

TEST_F(FixedStatePoolTest, SlotReuseGathersZeroAfterRelease) {
  auto pool = MakePool(1);
  const auto handle_a = MakeResident(*pool, kRequestA, 5.0f);
  pool->Release(handle_a);
  EXPECT_THROW(pool->Release(handle_a), std::runtime_error);  // Stale handle after release.

  auto requests = One(kRequestB);
  auto reservation = pool->Reserve(requests);
  EXPECT_EQ(reservation.Handles()[0].slot, handle_a.slot);
  EXPECT_GT(reservation.Handles()[0].generation, handle_a.generation);
  ExpectInputRows(reservation, 0, 0.0f);  // Reused slot must not leak the released request's state.
}

TEST_F(FixedStatePoolTest, StagedOutputsBecomeVisibleOnlyAfterCommit) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f);
  {
    auto requests = One(kRequestA);
    auto reservation = pool->Reserve(requests);
    for (const auto& binding : reservation.Bindings()) {
      ExpectInputRow(binding, 0, 4.0f);
      FillStagedRow(binding, 0, 9.0f);
      ExpectInputRow(binding, 0, 4.0f);  // Staging the output does not touch committed state.
    }
    reservation.Discard();
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 4.0f);  // Discard left the committed state unchanged.
}

TEST_F(FixedStatePoolTest, PreparedOutputIsInvisibleUntilPublish) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f, /*target_tokens=*/2);
  {
    auto requests = One(kRequestA, /*target_tokens=*/5);
    auto reservation = pool->Reserve(requests);
    FillStagedRows(reservation, 0, 9.0f);
    reservation.ValidateCommit();
    reservation.PrepareCommit();  // Device copy into the inactive bank only.
    EXPECT_EQ(reservation.State(), FixedStateReservationState::Prepared);
    // Prepare staged into the inactive bank; the visible committed state is untouched.
    EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestA)), 2u);
    reservation.Discard();  // Discarding a prepared reservation preserves active state.
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 4.0f);  // The prepared-but-unpublished 9.0 never became visible.
  EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestA)), 2u);
}

TEST_F(FixedStatePoolTest, ThreePhaseCommitPublishesStateAndTokens) {
  auto pool = MakePool(1);
  {
    auto requests = One(kRequestA, /*target_tokens=*/3);
    auto reservation = pool->Reserve(requests);
    FillStagedRows(reservation, 0, 7.0f);
    reservation.ValidateCommit();
    reservation.PrepareCommit();
    EXPECT_THROW(pool->HandleFor(kRequestA), std::runtime_error);  // Not committed until publish.
    reservation.PublishCommit();
    EXPECT_EQ(reservation.State(), FixedStateReservationState::Committed);
  }
  const auto handle = pool->HandleFor(kRequestA);
  EXPECT_EQ(pool->StateGeneration(handle), 1u);
  EXPECT_EQ(pool->CommittedTokens(handle), 3u);

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 7.0f);
}

TEST_F(FixedStatePoolTest, RepeatedCommitsAdvanceGenerationAndTokens) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 1.0f, /*target_tokens=*/4);
  EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestA)), 4u);

  {
    auto requests = One(kRequestA, /*target_tokens=*/9);
    auto reservation = pool->Reserve(requests);
    ExpectInputRows(reservation, 0, 1.0f);  // Gathered the first committed value.
    FillStagedRows(reservation, 0, 2.0f);
    reservation.Commit();
  }
  const auto handle = pool->HandleFor(kRequestA);
  EXPECT_EQ(pool->StateGeneration(handle), 2u);
  EXPECT_EQ(pool->CommittedTokens(handle), 9u);

  // The second commit must have landed in the other bank; gather now sees the new value.
  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 2.0f);
}

TEST_F(FixedStatePoolTest, RejectsCommitThatRegressesCommittedTokens) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 1.0f, /*target_tokens=*/10);

  auto requests = One(kRequestA, /*target_tokens=*/3);  // 3 < committed 10.
  auto reservation = pool->Reserve(requests);
  FillStagedRows(reservation, 0, 2.0f);
  EXPECT_THROW(reservation.ValidateCommit(), std::logic_error);
  EXPECT_THROW(reservation.PrepareCommit(), std::logic_error);
  reservation.Discard();

  // The rejected commit left the resident state and tokens untouched.
  const auto handle = pool->HandleFor(kRequestA);
  EXPECT_EQ(pool->CommittedTokens(handle), 10u);
  EXPECT_EQ(pool->StateGeneration(handle), 1u);
  EXPECT_TRUE(pool->Snapshot().healthy);
}

TEST_F(FixedStatePoolTest, GathersCommittedSlotsInScheduledRowOrder) {
  auto pool = MakePool(2);
  MakeResident(*pool, kRequestA, 11.0f);
  MakeResident(*pool, kRequestB, 22.0f);

  const std::array<Request, 2> reordered{Request{kRequestB, 1}, Request{kRequestA, 1}};
  auto reservation = pool->Reserve(reordered);
  ASSERT_EQ(reservation.Handles()[0].request_id, kRequestB);
  ASSERT_EQ(reservation.Handles()[1].request_id, kRequestA);
  for (const auto& binding : reservation.Bindings()) {
    ExpectInputRow(binding, 0, 22.0f);
    ExpectInputRow(binding, 1, 11.0f);
  }
}

TEST_F(FixedStatePoolTest, NewSlotOwnershipPublishesAtCommit) {
  auto pool = MakePool(1);
  FixedStateSlotHandle reserved_handle;
  {
    auto requests = One(kRequestA);
    auto reservation = pool->Reserve(requests);
    reserved_handle = reservation.Handles()[0];
    EXPECT_THROW(pool->HandleFor(kRequestA), std::runtime_error);  // Not yet committed.
    EXPECT_EQ(pool->Snapshot().reserved_slots, 1u);
    ExpectInputRows(reservation, 0, 0.0f);
    FillStagedRows(reservation, 0, 5.0f);
    reservation.Commit();
    EXPECT_EQ(pool->HandleFor(kRequestA), reserved_handle);
    EXPECT_EQ(pool->StateGeneration(reserved_handle), 1u);
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 5.0f);
}

TEST_F(FixedStatePoolTest, DiscardReleasesProvisionalSlotWithoutPublishing) {
  auto pool = MakePool(1);
  FixedStateSlotHandle discarded_handle;
  {
    auto requests = One(kRequestA);
    auto reservation = pool->Reserve(requests);
    discarded_handle = reservation.Handles()[0];
    FillStagedRows(reservation, 0, 13.0f);
    reservation.Discard();
  }

  EXPECT_THROW(pool->HandleFor(kRequestA), std::runtime_error);
  EXPECT_EQ(pool->AvailableSlots(), 1u);

  auto requests = One(kRequestB);
  auto replacement = pool->Reserve(requests);
  EXPECT_EQ(replacement.Handles()[0].slot, discarded_handle.slot);
  EXPECT_GT(replacement.Handles()[0].generation, discarded_handle.generation);
}

TEST_F(FixedStatePoolTest, DestructorDiscardsUncommittedReservation) {
  auto pool = MakePool(1);
  {
    auto requests = One(kRequestA);
    auto reservation = pool->Reserve(requests);
    EXPECT_EQ(pool->Snapshot().reserved_slots, 1u);
    // Reservation leaves scope without Commit/Discard: the destructor must free the provisional slot.
  }
  const auto snapshot = pool->Snapshot();
  EXPECT_EQ(snapshot.reserved_slots, 0u);
  EXPECT_EQ(snapshot.free_slots, 1u);
  EXPECT_THROW(pool->HandleFor(kRequestA), std::runtime_error);
}

TEST_F(FixedStatePoolTest, DestructorDiscardsPreparedReservation) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f, /*target_tokens=*/2);
  {
    auto requests = One(kRequestA, /*target_tokens=*/6);
    auto reservation = pool->Reserve(requests);
    FillStagedRows(reservation, 0, 8.0f);
    reservation.PrepareCommit();
    EXPECT_EQ(reservation.State(), FixedStateReservationState::Prepared);
    // Prepared reservation leaves scope without PublishCommit: the destructor discards it, and the
    // active committed state and tokens are preserved.
  }
  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 4.0f);
  EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestA)), 2u);
}

TEST_F(FixedStatePoolTest, MoveTransfersLiveReservationOwnership) {
  auto pool = MakePool(1);
  auto requests = One(kRequestA);
  auto original = pool->Reserve(requests);
  auto moved = std::move(original);

  // The moved-from reservation is inert; the moved-to reservation still governs the live slot.
  EXPECT_EQ(moved.State(), FixedStateReservationState::Reserved);
  ASSERT_EQ(moved.Handles().size(), 1u);
  FillStagedRows(moved, 0, 6.0f);
  moved.Commit();
  EXPECT_EQ(pool->StateGeneration(pool->HandleFor(kRequestA)), 1u);
}

TEST_F(FixedStatePoolTest, ReportsPersistentStagingAndReleaseAccounting) {
  auto pool = MakePool(3);
  constexpr size_t bytes_per_request =
      2 * (2 * 3) * sizeof(float) +     // convolution: 2 layers, row [2, 3]
      2 * (2 * 2 * 2) * sizeof(float);  // recurrent: 2 layers, row [2, 2, 2]
  // Two persistent banks per tensor so publish is a bank flip with no device copy.
  EXPECT_EQ(pool->PersistentBytes(), 2 * 3 * bytes_per_request);
  EXPECT_EQ(pool->ZeroingScratchBytes(), bytes_per_request);
  EXPECT_EQ(pool->ActiveStagingBytes(), 0u);

  const auto handle_a = MakeResident(*pool, kRequestA, 1.0f);
  {
    const std::array<Request, 2> requests{Request{kRequestA, 1}, Request{kRequestB, 1}};  // A resident, B provisional.
    auto reservation = pool->Reserve(requests);
    const size_t expected_staging =
        4 * bytes_per_request + 2 * sizeof(int32_t);  // gather + output + count input
    EXPECT_EQ(reservation.PlannedStagingBytes(), expected_staging);
    EXPECT_EQ(pool->ActiveStagingBytes(), expected_staging);
    const auto snapshot = pool->Snapshot();
    EXPECT_EQ(snapshot.free_slots, 1u);
    EXPECT_EQ(snapshot.reserved_slots, 1u);
    EXPECT_EQ(snapshot.committed_slots, 1u);
    reservation.Discard();
  }
  EXPECT_EQ(pool->ActiveStagingBytes(), 0u);
  EXPECT_EQ(pool->AvailableSlots(), 2u);

  pool->Release(handle_a);
  const auto snapshot = pool->Snapshot();
  EXPECT_EQ(snapshot.free_slots, 3u);
  EXPECT_EQ(snapshot.reserved_slots, 0u);
  EXPECT_EQ(snapshot.committed_slots, 0u);
}

TEST_F(FixedStatePoolTest, CapacityOverflowLeavesPoolUntouched) {
  auto pool = MakePool(1);
  const std::array<Request, 2> requests{Request{kRequestA, 1}, Request{kRequestB, 1}};  // 2 > capacity 1.
  EXPECT_THROW(pool->Reserve(requests), std::runtime_error);

  const auto snapshot = pool->Snapshot();
  EXPECT_TRUE(snapshot.healthy);
  EXPECT_EQ(snapshot.free_slots, 1u);
  EXPECT_EQ(snapshot.reserved_slots, 0u);
  EXPECT_EQ(snapshot.committed_slots, 0u);
  EXPECT_EQ(snapshot.active_staging_bytes, 0u);
  EXPECT_THROW(pool->HandleFor(kRequestA), std::runtime_error);
}

TEST_F(FixedStatePoolTest, RejectsEmptyReservation) {
  auto pool = MakePool(1);
  const std::span<const Request> empty;
  EXPECT_THROW(pool->Reserve(empty), std::invalid_argument);
  EXPECT_EQ(pool->AvailableSlots(), 1u);
}

TEST_F(FixedStatePoolTest, NotEnoughFreeSlotsForNewAdmissions) {
  auto pool = MakePool(2);
  MakeResident(*pool, kRequestA, 1.0f);  // Occupies one of two slots.
  // One resident row plus two brand-new rows needs three slots but only two exist.
  const std::array<Request, 3> requests{Request{kRequestA, 1}, Request{kRequestB, 1},
                                        Request{kRequestC, 1}};
  EXPECT_THROW(pool->Reserve(requests), std::runtime_error);
  EXPECT_TRUE(pool->Snapshot().healthy);
  EXPECT_EQ(pool->AvailableSlots(), 1u);
}

TEST_F(FixedStatePoolTest, RejectsDuplicateRequestsAndConcurrentReservations) {
  auto pool = MakePool(2);
  const std::array<Request, 2> duplicate{Request{kRequestA, 1}, Request{kRequestA, 1}};
  EXPECT_THROW(pool->Reserve(duplicate), std::runtime_error);

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  auto other = One(kRequestB);
  EXPECT_THROW(pool->Reserve(other), std::logic_error);  // Only one live reservation permitted.
}

TEST_F(FixedStatePoolTest, RejectsReleaseWhileReservationIsLive) {
  auto pool = MakePool(2);
  const auto handle_a = MakeResident(*pool, kRequestA, 3.0f);
  auto requests = One(kRequestB);
  auto reservation = pool->Reserve(requests);
  EXPECT_THROW(pool->Release(handle_a), std::logic_error);  // Idle contract during a reservation.
}

TEST_F(FixedStatePoolTest, ReservationHoldsTheLockForItsWholeLifetime) {
  auto pool = MakePool(2);
  {
    auto requests_a = One(kRequestA, /*target_tokens=*/2);
    auto reservation_a = pool->Reserve(requests_a);
    FillStagedRows(reservation_a, 0, 1.0f);
    reservation_a.Commit();
    EXPECT_EQ(reservation_a.State(), FixedStateReservationState::Committed);

    // A committed reservation still holds the single-reservation lock and its staging memory until
    // it is destroyed, so admitting the next batch or releasing a slot is refused while it is alive.
    auto requests_b = One(kRequestB, /*target_tokens=*/2);
    EXPECT_THROW(pool->Reserve(requests_b), std::logic_error);
    EXPECT_THROW(pool->Release(pool->HandleFor(kRequestA)), std::logic_error);
  }

  // The reservation is gone: the lock is free again. A second reservation likewise holds the lock
  // for its own lifetime, so it too must be dropped before the committed slot can be released.
  {
    auto requests_b = One(kRequestB, /*target_tokens=*/2);
    auto reservation_b = pool->Reserve(requests_b);
    reservation_b.Discard();
  }
  pool->Release(pool->HandleFor(kRequestA));
  EXPECT_EQ(pool->AvailableSlots(), 2u);
}

TEST_F(FixedStatePoolTest, DiscardRejectsCommittedButNotTerminalReservations) {
  auto pool = MakePool(1);
  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  FillStagedRows(reservation, 0, 1.0f);
  reservation.Commit();
  EXPECT_THROW(reservation.Discard(), std::logic_error);  // A published commit is irreversible.
  EXPECT_EQ(reservation.State(), FixedStateReservationState::Committed);
}

TEST_F(FixedStatePoolTest, PublishCommitAfterPoolDestructionIsANoOp) {
  std::optional<FixedStateReservation> reservation;
  {
    auto pool = MakePool(1);
    auto requests = One(kRequestA);
    reservation.emplace(pool->Reserve(requests));
    FillStagedRows(*reservation, 0, 1.0f);
    reservation->PrepareCommit();
    ASSERT_EQ(reservation->State(), FixedStateReservationState::Prepared);
  }
  // The pool is gone. PublishCommit must not touch freed pool state; it is a silent no-op and the
  // reservation is left in its pool-neutralized Failed state.
  reservation->PublishCommit();
  EXPECT_EQ(reservation->State(), FixedStateReservationState::Failed);
  reservation.reset();
}

TEST_F(FixedStatePoolTest, RejectsForeignAndStaleHandles) {
  auto pool = MakePool(1);
  auto other_pool = MakePool(1);
  const auto handle_a = MakeResident(*pool, kRequestA, 2.0f);

  // Foreign handle: right shape, wrong pool.
  FixedStateSlotHandle foreign = handle_a;
  foreign.pool = other_pool.get();
  EXPECT_THROW(pool->Release(foreign), std::runtime_error);
  EXPECT_THROW(pool->StateGeneration(foreign), std::runtime_error);
  EXPECT_THROW(pool->CommittedTokens(foreign), std::runtime_error);

  // Stale handle: slot reused by a new request after release.
  pool->Release(handle_a);
  MakeResident(*pool, kRequestB, 8.0f);  // Reuses the slot with a fresh generation.
  EXPECT_THROW(pool->Release(handle_a), std::runtime_error);
  EXPECT_THROW(pool->StateGeneration(handle_a), std::runtime_error);
  EXPECT_THROW(pool->CommittedTokens(handle_a), std::runtime_error);
  EXPECT_THROW(pool->HandleFor(kRequestC), std::runtime_error);  // Unknown request.
}

TEST_F(FixedStatePoolTest, ReservationAccessorsSafeAfterPoolDestruction) {
  std::optional<FixedStateReservation> reservation;
  std::vector<std::string> input_names;
  {
    auto pool = MakePool(1);
    auto requests = One(kRequestA);
    reservation.emplace(pool->Reserve(requests));
    for (const auto& binding : reservation->Bindings()) {
      input_names.emplace_back(binding.input_name);
    }
  }

  // The pool is gone, but the reservation still owns its handles and binding names.
  EXPECT_EQ(reservation->State(), FixedStateReservationState::Failed);
  ASSERT_EQ(reservation->Handles().size(), 1u);
  EXPECT_EQ(reservation->Handles()[0].request_id, kRequestA);
  ASSERT_EQ(reservation->Bindings().size(), input_names.size());
  for (size_t index = 0; index < input_names.size(); ++index) {
    EXPECT_EQ(input_names[index], reservation->Bindings()[index].input_name);
  }
  EXPECT_THROW(reservation->ValidateCommit(), std::logic_error);
  EXPECT_THROW(reservation->PrepareCommit(), std::logic_error);
  reservation.reset();
}

// --- Direct geometry-contract tests (model-independent) ---

TEST(FixedStateGeometryTest, AcceptsDynamicBatchAndDerivesRowElements) {
  const std::array<int64_t, 3> input{-1, 2, 3};
  const std::array<int64_t, 3> output{-1, 2, 3};
  const auto geometry = ValidateFixedStateGeometry(
      "past", kFloat, input, "present", kFloat, output);
  EXPECT_EQ(geometry.row_element_count, 6u);
  EXPECT_EQ(geometry.fixed_batch_size, 0u);
}

TEST(FixedStateGeometryTest, AcceptsFixedBatchAndReportsIt) {
  const std::array<int64_t, 2> input{4, 5};
  const std::array<int64_t, 2> output{4, 5};
  const auto geometry = ValidateFixedStateGeometry(
      "past", kFloat, input, "present", kFloat, output);
  EXPECT_EQ(geometry.row_element_count, 5u);
  EXPECT_EQ(geometry.fixed_batch_size, 4u);
}

TEST(FixedStateGeometryTest, AdoptsFixedOutputBatchWhenInputIsDynamic) {
  const std::array<int64_t, 3> input{-1, 2, 3};
  const std::array<int64_t, 3> output{4, 2, 3};
  const auto geometry = ValidateFixedStateGeometry(
      "past", kFloat, input, "present", kFloat, output);
  EXPECT_EQ(geometry.row_element_count, 6u);
  EXPECT_EQ(geometry.fixed_batch_size, 4u);  // A fixed output batch constrains the whole binding.
}

TEST(FixedStateGeometryTest, AdoptsFixedInputBatchWhenOutputIsDynamic) {
  const std::array<int64_t, 3> input{4, 2, 3};
  const std::array<int64_t, 3> output{-1, 2, 3};
  const auto geometry = ValidateFixedStateGeometry(
      "past", kFloat, input, "present", kFloat, output);
  EXPECT_EQ(geometry.fixed_batch_size, 4u);
}

TEST(FixedStateGeometryTest, RejectsZeroBatch) {
  const std::array<int64_t, 2> input{0, 5};
  const std::array<int64_t, 2> output{0, 5};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kFloat, output),
               std::runtime_error);
}

TEST(FixedStateGeometryTest, RejectsAsymmetricFixedBatch) {
  const std::array<int64_t, 2> input{4, 5};
  const std::array<int64_t, 2> output{2, 5};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kFloat, output),
               std::runtime_error);
}

TEST(FixedStateGeometryTest, RejectsMismatchedDtype) {
  const std::array<int64_t, 2> input{-1, 5};
  const std::array<int64_t, 2> output{-1, 5};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kDouble, output),
               std::runtime_error);
}

TEST(FixedStateGeometryTest, RejectsRankMismatch) {
  const std::array<int64_t, 3> input{-1, 2, 3};
  const std::array<int64_t, 2> output{-1, 6};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kFloat, output),
               std::runtime_error);
}

TEST(FixedStateGeometryTest, RejectsDynamicNonBatchDimension) {
  const std::array<int64_t, 3> input{-1, -1, 3};
  const std::array<int64_t, 3> output{-1, -1, 3};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kFloat, output),
               std::runtime_error);
}

TEST(FixedStateGeometryTest, RejectsIncompatibleNonBatchGeometry) {
  const std::array<int64_t, 3> input{-1, 2, 3};
  const std::array<int64_t, 3> output{-1, 2, 4};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kFloat, output),
               std::runtime_error);
}

TEST(FixedStateGeometryTest, RejectsScalarWithoutBatchAxis) {
  const std::array<int64_t, 0> input{};
  const std::array<int64_t, 0> output{};
  EXPECT_THROW(ValidateFixedStateGeometry("past", kFloat, input, "present", kFloat, output),
               std::runtime_error);
}

TEST_F(FixedStatePoolTest, ReportsCheckpointSupport) {
  auto pool = MakePool();
  EXPECT_TRUE(pool->SupportsCheckpoints());
  EXPECT_EQ(pool->CheckpointCount(), 4u);
}

TEST_F(FixedStatePoolTest, ReportsCompactStateUpdateSupport) {
  auto pool = MakePool();
  EXPECT_TRUE(pool->SupportsStateUpdates());
  EXPECT_EQ(pool->StateUpdateCapacity(), 3u);
}

TEST_F(FixedStatePoolTest, DisabledCompactStateUpdatesBindZeroAndCaptureCheckpoints) {
  for (auto& group : model_->config_->model.decoder.state_groups.value()) {
    if (group.state_update) {
      group.state_update->enabled = false;
    }
  }
  manifest_ = std::make_unique<ModelStateManifest>(model_->config_->model.decoder);
  auto pool = MakePool();
  EXPECT_FALSE(pool->SupportsStateUpdates());
  EXPECT_EQ(pool->StateUpdateCapacity(), 0u);

  const auto requests = One(kRequestA, 4, 3);
  auto reservation = pool->Reserve(requests, true);
  EXPECT_TRUE(reservation.CapturesCheckpoints());
  EXPECT_FALSE(reservation.CapturesStateUpdates());
  for (const auto& binding : reservation.Bindings()) {
    EXPECT_STREQ(binding.state_update_capture_count_name, "state_update_capture_count");
    ASSERT_NE(binding.state_update_capture_count, nullptr);
    EXPECT_EQ(binding.state_update_capture_count->GetTensorData<int32_t>()[0], 0);
    EXPECT_NE(binding.checkpoints, nullptr);
    EXPECT_EQ(binding.state_update_value, nullptr);
    EXPECT_EQ(binding.state_update_decay, nullptr);
    EXPECT_EQ(binding.state_update_key, nullptr);
    EXPECT_EQ(binding.state_update_delta, nullptr);
  }
  reservation.Discard();
}

TEST_F(FixedStatePoolTest, CaptureCountIsAlwaysBoundAndCompactOutputsAreConditional) {
  auto pool = MakePool(2);
  {
    auto requests = One(kRequestA);
    auto reservation = pool->Reserve(requests);
    EXPECT_FALSE(reservation.CapturesStateUpdates());
    OrtValue* shared_count{};
    for (const auto& binding : reservation.Bindings()) {
      EXPECT_STREQ(binding.state_update_capture_count_name, "state_update_capture_count");
      ASSERT_NE(binding.state_update_capture_count, nullptr);
      EXPECT_EQ(binding.state_update_capacity, 3u);
      if (!shared_count) {
        shared_count = binding.state_update_capture_count;
      }
      EXPECT_EQ(binding.state_update_capture_count, shared_count);
      EXPECT_EQ(binding.state_update_capture_count->GetTensorData<int32_t>()[0], 0);
      EXPECT_EQ(binding.state_update_value, nullptr);
      EXPECT_EQ(binding.state_update_decay, nullptr);
      EXPECT_EQ(binding.state_update_key, nullptr);
      EXPECT_EQ(binding.state_update_delta, nullptr);
    }
    reservation.Discard();
  }

  const std::array<Request, 2> requests{
      Request{kRequestA, 4, 3}, Request{kRequestB, 1, 0}};
  auto reservation = pool->Reserve(requests);
  EXPECT_TRUE(reservation.CapturesStateUpdates());
  const auto& conv = reservation.Bindings()[0];
  const auto& recurrent = reservation.Bindings()[2];
  EXPECT_EQ(conv.state_update_capture_count->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{2}));
  EXPECT_EQ(conv.state_update_capture_count->GetTensorData<int32_t>()[0], 3);
  EXPECT_EQ(conv.state_update_capture_count->GetTensorData<int32_t>()[1], 0);
  ASSERT_NE(conv.state_update_value, nullptr);
  EXPECT_EQ(conv.state_update_value->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{2, 3, 2}));
  ASSERT_NE(recurrent.state_update_decay, nullptr);
  ASSERT_NE(recurrent.state_update_key, nullptr);
  ASSERT_NE(recurrent.state_update_delta, nullptr);
  EXPECT_EQ(recurrent.state_update_decay->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{2, 3, 2}));
  EXPECT_EQ(recurrent.state_update_key->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{2, 3, 1, 2}));
  EXPECT_EQ(recurrent.state_update_delta->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{2, 3, 2, 2}));
  EXPECT_EQ(reservation.PlannedStagingBytes(),
            pool->PlannedStagingBytes(2, false, true));
}

TEST_F(FixedStatePoolTest, CompactCaptureAddsExactFactorStagingBytes) {
  auto pool = MakePool();
  const size_t plain = pool->PlannedStagingBytes(2);
  const size_t captured = pool->PlannedStagingBytes(2, false, true);
  constexpr size_t compact_bytes_per_row =
      2 * (3 * 2) * sizeof(float) +                         // two conv update tensors
      2 * (3 * 2 + 3 * 1 * 2 + 3 * 2 * 2) * sizeof(float);  // two GDN factor sets
  EXPECT_EQ(captured, plain + 2 * compact_bytes_per_row);

  const std::array<Request, 2> requests{
      Request{kRequestA, 4, 3}, Request{kRequestB, 1, 0}};
  auto reservation = pool->Reserve(requests);
  EXPECT_EQ(reservation.PlannedStagingBytes(), captured);
}

TEST_F(FixedStatePoolTest, CompactPartialAcceptanceReplaysConvAndGdnFromGatheredInput) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f);
  {
    auto requests = One(kRequestA, /*target_tokens=*/4, /*capture_count=*/3);
    auto reservation = pool->Reserve(requests);
    FillStagedRows(reservation, 0, 99.0f);
    const std::array<float, 6> conv_values{10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
    const std::array<float, 6> decay{0.5f, 0.25f, 1.0f, 0.5f, 1.0f, 1.0f};
    const std::array<float, 6> key{2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 1.0f};
    const std::array<float, 12> delta{
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (const auto& binding : reservation.Bindings()) {
      if (binding.state_update_kind ==
          Config::Model::Decoder::StateUpdateKind::CausalConv) {
        FillConvUpdates(binding, 0, conv_values);
      } else {
        FillGdnUpdates(binding, 0, decay, key, delta);
      }
    }
    reservation.CommitPrefix(0, /*step_tokens=*/4, /*kept_tokens=*/2);
    reservation.Commit();
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  const std::array<float, 6> expected_conv{4.0f, 10.0f, 20.0f, 4.0f, 11.0f, 21.0f};
  const std::array<float, 8> expected_gdn{
      24.0f, 30.0f, 30.0f, 38.0f, 31.5f, 40.0f, 36.5f, 46.5f};
  ExpectInputRow(reservation.Bindings()[0], 0, expected_conv);
  ExpectInputRow(reservation.Bindings()[1], 0, expected_conv);
  ExpectInputRow(reservation.Bindings()[2], 0, expected_gdn);
  ExpectInputRow(reservation.Bindings()[3], 0, expected_gdn);
}

TEST_F(FixedStatePoolTest, CompactAnchorOnlyReplayUsesFirstTransition) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f);
  {
    auto requests = One(kRequestA, /*target_tokens=*/4, /*capture_count=*/3);
    auto reservation = pool->Reserve(requests);
    FillStagedRows(reservation, 0, 99.0f);
    const std::array<float, 6> conv_values{10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
    const std::array<float, 6> decay{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const std::array<float, 6> key{};
    const std::array<float, 12> delta{};
    for (const auto& binding : reservation.Bindings()) {
      if (binding.state_update_value) {
        FillConvUpdates(binding, 0, conv_values);
      } else {
        FillGdnUpdates(binding, 0, decay, key, delta);
      }
    }
    reservation.CommitPrefix(0, /*step_tokens=*/4, /*kept_tokens=*/1);
    reservation.Commit();
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  const std::array<float, 6> expected_conv{4.0f, 4.0f, 10.0f, 4.0f, 4.0f, 11.0f};
  ExpectInputRow(reservation.Bindings()[0], 0, expected_conv);
  ExpectInputRow(reservation.Bindings()[1], 0, expected_conv);
  ExpectInputRow(reservation.Bindings()[2], 0, 4.0f);
  ExpectInputRow(reservation.Bindings()[3], 0, 4.0f);
}

TEST_F(FixedStatePoolTest, CompactFullAcceptanceUsesFinalState) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f);
  {
    auto requests = One(kRequestA, /*target_tokens=*/4, /*capture_count=*/3);
    auto reservation = pool->Reserve(requests);
    FillStagedRows(reservation, 0, 99.0f);
    reservation.CommitPrefix(0, /*step_tokens=*/4, /*kept_tokens=*/4);
    reservation.Commit();
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 99.0f);
}

TEST_F(FixedStatePoolTest, CompactReplayIsInvisibleBeforePublish) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f, /*target_tokens=*/2);
  {
    auto requests = One(kRequestA, /*target_tokens=*/5, /*capture_count=*/3);
    auto reservation = pool->Reserve(requests);
    FillStagedRows(reservation, 0, 99.0f);
    const std::array<float, 6> conv_values{10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
    const std::array<float, 6> decay{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const std::array<float, 6> key{};
    const std::array<float, 12> delta{};
    for (const auto& binding : reservation.Bindings()) {
      if (binding.state_update_value) {
        FillConvUpdates(binding, 0, conv_values);
      } else {
        FillGdnUpdates(binding, 0, decay, key, delta);
      }
    }
    reservation.CommitPrefix(0, /*step_tokens=*/4, /*kept_tokens=*/1);
    reservation.PrepareCommit();
    EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestA)), 2u);
    reservation.Discard();
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 4.0f);
}

TEST_F(FixedStatePoolTest, CompactReplayChangesOnlyTheRequestedRow) {
  auto pool = MakePool(2);
  MakeResident(*pool, kRequestA, 4.0f);
  MakeResident(*pool, kRequestB, 5.0f);
  {
    const std::array<Request, 2> requests{
        Request{kRequestA, 4, 3}, Request{kRequestB, 1, 0}};
    auto reservation = pool->Reserve(requests);
    FillStagedRows(reservation, 0, 99.0f);
    FillStagedRows(reservation, 1, 98.0f);
    const std::array<float, 6> conv_values{10.0f, 11.0f, 20.0f, 21.0f, 30.0f, 31.0f};
    const std::array<float, 6> decay{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const std::array<float, 6> key{};
    const std::array<float, 12> delta{};
    for (const auto& binding : reservation.Bindings()) {
      if (binding.state_update_value) {
        FillConvUpdates(binding, 0, conv_values);
      } else {
        FillGdnUpdates(binding, 0, decay, key, delta);
      }
    }
    reservation.CommitPrefix(0, /*step_tokens=*/4, /*kept_tokens=*/1);
    reservation.Commit();
  }

  const std::array<Request, 2> requests{
      Request{kRequestA, 1, 0}, Request{kRequestB, 1, 0}};
  auto reservation = pool->Reserve(requests);
  const std::array<float, 6> expected_conv{4.0f, 4.0f, 10.0f, 4.0f, 4.0f, 11.0f};
  ExpectInputRow(reservation.Bindings()[0], 0, expected_conv);
  ExpectInputRow(reservation.Bindings()[2], 0, 4.0f);
  ExpectInputRows(reservation, 1, 98.0f);
}

TEST_F(FixedStatePoolTest, RejectsInvalidCompactCaptureRequests) {
  auto pool = MakePool(1);
  auto overflow = One(kRequestA, /*target_tokens=*/5, /*capture_count=*/4);
  EXPECT_THROW(pool->Reserve(overflow), std::runtime_error);

  auto requests = One(kRequestA, /*target_tokens=*/4, /*capture_count=*/3);
  EXPECT_THROW(pool->Reserve(requests, /*capture_checkpoints=*/true), std::runtime_error);
}

TEST_F(FixedStatePoolTest, CompactCommitPrefixRequiresCapturedStepGeometry) {
  auto pool = MakePool(1);
  auto requests = One(kRequestA, /*target_tokens=*/4, /*capture_count=*/3);
  auto reservation = pool->Reserve(requests);
  EXPECT_THROW(
      reservation.CommitPrefix(0, /*step_tokens=*/3, /*kept_tokens=*/2),
      std::runtime_error);
}

TEST_F(FixedStatePoolTest, DenseCheckpointFallbackWorksWithCompactConfigStripped) {
  for (auto& group : *model_->config_->model.decoder.state_groups) {
    group.state_update.reset();
  }
  manifest_ = std::make_unique<ModelStateManifest>(model_->config_->model.decoder);
  auto pool = MakePool(1);
  EXPECT_FALSE(pool->SupportsStateUpdates());
  EXPECT_EQ(pool->StateUpdateCapacity(), 0u);

  auto requests = One(kRequestA, /*target_tokens=*/4, /*capture_count=*/3);
  auto reservation = pool->Reserve(requests, /*capture_checkpoints=*/true);
  EXPECT_TRUE(reservation.CapturesCheckpoints());
  EXPECT_FALSE(reservation.CapturesStateUpdates());
  for (const auto& binding : reservation.Bindings()) {
    EXPECT_EQ(binding.state_update_capture_count, nullptr);
  }
  FillStagedRows(reservation, 0, 99.0f);
  FillCheckpointSeries(reservation, /*row_count=*/1, /*row=*/0, /*base=*/10.0f);
  reservation.CommitPrefix(0, /*step_tokens=*/4, /*kept_tokens=*/1);
  reservation.Commit();
}

TEST_F(FixedStatePoolTest, CheckpointsAreBoundOnlyWhenCaptured) {
  auto pool = MakePool();
  {
    auto requests = One(kRequestA);
    auto reservation = pool->Reserve(requests);
    EXPECT_FALSE(reservation.CapturesCheckpoints());
    for (const auto& binding : reservation.Bindings()) {
      EXPECT_EQ(binding.checkpoints, nullptr);
      EXPECT_EQ(binding.checkpoints_name, nullptr);
    }
    reservation.Discard();
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests, /*capture_checkpoints=*/true);
  EXPECT_TRUE(reservation.CapturesCheckpoints());
  ASSERT_EQ(reservation.Bindings().size(), 4u);
  EXPECT_STREQ(reservation.Bindings()[0].checkpoints_name, "checkpoints_conv.0");
  EXPECT_STREQ(reservation.Bindings()[2].checkpoints_name, "checkpoints_recurrent.2");
  EXPECT_EQ(reservation.Bindings()[0].checkpoints->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{4, 1, 2, 3}));
  EXPECT_EQ(reservation.Bindings()[2].checkpoints->GetTensorTypeAndShapeInfo()->GetShape(),
            (std::vector<int64_t>{4, 1, 2, 2, 2}));
}

TEST_F(FixedStatePoolTest, CapturingCheckpointsAddsWindowStagingBytes) {
  auto pool = MakePool();
  const auto plain = pool->PlannedStagingBytes(2);
  const auto captured = pool->PlannedStagingBytes(2, /*capture_checkpoints=*/true);
  constexpr size_t fixed_row_bytes =
      2 * (2 * 3) * sizeof(float) + 2 * (2 * 2 * 2) * sizeof(float);
  // The shared count input is present in both plans; checkpoint capture adds four tensor rows per
  // scheduled row without duplicating that input.
  EXPECT_EQ(captured, plain + 2 * 4 * fixed_row_bytes);

  auto requests = std::array<Request, 2>{Request{kRequestA, 1}, Request{kRequestB, 1}};
  auto reservation = pool->Reserve(requests, /*capture_checkpoints=*/true);
  EXPECT_EQ(reservation.PlannedStagingBytes(), captured);
}

TEST_F(FixedStatePoolTest, CommitPrefixPublishesTheAlignedCheckpointSlot) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f);
  {
    auto requests = One(kRequestA, /*target_tokens=*/3);
    auto reservation = pool->Reserve(requests, /*capture_checkpoints=*/true);
    FillStagedRows(reservation, 0, 99.0f);  // The step's final state, which must NOT be committed.
    FillCheckpointSeries(reservation, /*row_count=*/1, /*row=*/0, /*base=*/10.0f);
    // A three-token step of which two were accepted: the conv group is left-aligned so the state
    // after token 1 is slot 1, the recurrent group is right-aligned so it is slot 4 - 3 + 1 = 2.
    reservation.CommitPrefix(0, /*step_tokens=*/3, /*kept_tokens=*/2);
    reservation.Commit();
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  const auto bindings = reservation.Bindings();
  ExpectInputRow(bindings[0], 0, 11.0f);  // conv.0, left-aligned slot 1
  ExpectInputRow(bindings[1], 0, 11.0f);  // conv.3
  ExpectInputRow(bindings[2], 0, 12.0f);  // recurrent.2, right-aligned slot 2
  ExpectInputRow(bindings[3], 0, 12.0f);  // recurrent.5
}

TEST_F(FixedStatePoolTest, CommitPrefixOfTheWholeStepUsesTheFinalState) {
  auto pool = MakePool(1);
  MakeResident(*pool, kRequestA, 4.0f);
  {
    auto requests = One(kRequestA, /*target_tokens=*/3);
    auto reservation = pool->Reserve(requests, /*capture_checkpoints=*/true);
    FillStagedRows(reservation, 0, 99.0f);
    FillCheckpointSeries(reservation, /*row_count=*/1, /*row=*/0, /*base=*/10.0f);
    reservation.CommitPrefix(0, /*step_tokens=*/3, /*kept_tokens=*/3);
    reservation.Commit();
  }

  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 99.0f);
}

TEST_F(FixedStatePoolTest, CommitPrefixRollsBackOnlyTheRequestedRow) {
  auto pool = MakePool(2);
  MakeResident(*pool, kRequestA, 4.0f);
  MakeResident(*pool, kRequestB, 5.0f);
  {
    auto requests = std::array<Request, 2>{Request{kRequestA, 3}, Request{kRequestB, 4}};
    auto reservation = pool->Reserve(requests, /*capture_checkpoints=*/true);
    FillStagedRows(reservation, 0, 99.0f);
    FillStagedRows(reservation, 1, 98.0f);
    FillCheckpointSeries(reservation, /*row_count=*/2, /*row=*/0, /*base=*/10.0f);
    FillCheckpointSeries(reservation, /*row_count=*/2, /*row=*/1, /*base=*/20.0f);
    reservation.CommitPrefix(1, /*step_tokens=*/4, /*kept_tokens=*/1);
    reservation.Commit();
  }

  // Rejecting three of the four tokens also moves row 1's committed boundary back to where the
  // published checkpoint actually is.
  EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestA)), 3u);
  EXPECT_EQ(pool->CommittedTokens(pool->HandleFor(kRequestB)), 1u);
  auto requests = std::array<Request, 2>{Request{kRequestA, 3}, Request{kRequestB, 3}};
  auto reservation = pool->Reserve(requests);
  ExpectInputRows(reservation, 0, 99.0f);  // Row 0 kept the whole step.
  const auto bindings = reservation.Bindings();
  ExpectInputRow(bindings[0], 1, 20.0f);  // conv, left-aligned slot 0
  ExpectInputRow(bindings[2], 1, 20.0f);  // recurrent, right-aligned slot 4 - 4 + 0 = 0
}

TEST_F(FixedStatePoolTest, CommitPrefixRequiresACapturingReservation) {
  auto pool = MakePool(1);
  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests);
  EXPECT_THROW(reservation.CommitPrefix(0, 2, 1), std::logic_error);
}

TEST_F(FixedStatePoolTest, CommitPrefixRejectsOutOfContractArguments) {
  auto pool = MakePool(1);
  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests, /*capture_checkpoints=*/true);
  EXPECT_THROW(reservation.CommitPrefix(1, 2, 1), std::out_of_range);
  EXPECT_THROW(reservation.CommitPrefix(0, 2, 0), std::runtime_error);
  EXPECT_THROW(reservation.CommitPrefix(0, 2, 3), std::runtime_error);
  // A step longer than the window has no checkpoint for its earlier tokens.
  EXPECT_THROW(reservation.CommitPrefix(0, 5, 2), std::runtime_error);
}

TEST_F(FixedStatePoolTest, CommitPrefixIsRejectedAfterPrepare) {
  auto pool = MakePool(1);
  auto requests = One(kRequestA);
  auto reservation = pool->Reserve(requests, /*capture_checkpoints=*/true);
  FillStagedRows(reservation, 0, 1.0f);
  reservation.PrepareCommit();
  EXPECT_THROW(reservation.CommitPrefix(0, 2, 1), std::logic_error);
}

TEST_F(FixedStatePoolTest, CommitPrefixIsRejectedTwiceForTheSameRow) {
  auto pool = MakePool(1);
  auto requests = One(kRequestA, /*target_tokens=*/3);
  auto reservation = pool->Reserve(requests, /*capture_checkpoints=*/true);
  reservation.CommitPrefix(0, 3, 2);
  EXPECT_THROW(reservation.CommitPrefix(0, 3, 2), std::logic_error);
}

TEST_F(FixedStatePoolTest, CommitPrefixCannotRejectMoreTokensThanTheStepPlanned) {
  auto pool = MakePool(1);
  auto requests = One(kRequestA, /*target_tokens=*/1);
  auto reservation = pool->Reserve(requests, /*capture_checkpoints=*/true);
  EXPECT_THROW(reservation.CommitPrefix(0, 4, 1), std::runtime_error);
}

}  // namespace
}  // namespace test
}  // namespace Generators
