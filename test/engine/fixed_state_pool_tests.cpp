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
std::array<Request, 1> One(const void* id, uint64_t target_tokens = 1) {
  return {Request{id, target_tokens}};
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

void ExpectInputRows(const FixedStateReservation& reservation, size_t row, float expected) {
  for (const auto& binding : reservation.Bindings()) {
    ExpectInputRow(binding, row, expected);
  }
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
            (std::vector<int64_t>{1, 2, 2}));
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
      2 * (2 * 3) * sizeof(float) +  // convolution: 2 layers, row [2, 3]
      2 * (2 * 2) * sizeof(float);   // recurrent: 2 layers, row [2, 2]
  // Two persistent banks per tensor so publish is a bank flip with no device copy.
  EXPECT_EQ(pool->PersistentBytes(), 2 * 3 * bytes_per_request);
  EXPECT_EQ(pool->ZeroingScratchBytes(), bytes_per_request);
  EXPECT_EQ(pool->ActiveStagingBytes(), 0u);

  const auto handle_a = MakeResident(*pool, kRequestA, 1.0f);
  {
    const std::array<Request, 2> requests{Request{kRequestA, 1}, Request{kRequestB, 1}};  // A resident, B provisional.
    auto reservation = pool->Reserve(requests);
    EXPECT_EQ(reservation.PlannedStagingBytes(), 4 * bytes_per_request);  // gather + staged over 2 rows.
    EXPECT_EQ(pool->ActiveStagingBytes(), 4 * bytes_per_request);
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

}  // namespace
}  // namespace test
}  // namespace Generators
