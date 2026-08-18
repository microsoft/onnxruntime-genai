// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/**
 * @file request_status.h
 * @brief Defines the lifecycle status of an engine Request.
 *
 * Kept in its own lightweight header (free of heavy engine/model includes) so that
 * invariant-checking and state-observing code can depend on the status enum without pulling in the
 * full Request or the rest of the generators runtime.
 */

namespace Generators {

enum class RequestStatus {
  Unassigned,    // Created: initial input may be added before submission to an Engine.
  Assigned,      // Queued: submitted initial work or a resident continuation awaits execution.
  InProgress,    // The current generation turn is executable and owned by the Engine.
  TurnComplete,  // The current turn stopped; output and resident model state remain available.
  Closed,        // Permanently terminal; no scheduler or cache resources remain owned.
};

constexpr bool IsQueued(RequestStatus status) noexcept {
  return status == RequestStatus::Assigned;
}

constexpr bool IsExecuting(RequestStatus status) noexcept {
  return status == RequestStatus::InProgress;
}

constexpr bool IsExecutable(RequestStatus status) noexcept {
  return IsQueued(status) || IsExecuting(status);
}

constexpr bool IsTurnComplete(RequestStatus status) noexcept {
  return status == RequestStatus::TurnComplete;
}

constexpr bool IsClosed(RequestStatus status) noexcept {
  return status == RequestStatus::Closed;
}

}  // namespace Generators
