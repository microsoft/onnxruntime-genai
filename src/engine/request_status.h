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
  Unassigned,  // A request has been created but has not been added to the engine yet.
               // This is the state of a request when it is first created.
  Assigned,    // The request has been added to the engine and is waiting to be scheduled.
  InProgress,  // The request has been scheduled and is currently being processed.
  Completed,   // The request has been completed successfully.
};

}  // namespace Generators
