// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../generators.h"
#include <chrono>

namespace Generators {

enum class RequestStatus {
  Unassigned,  // A request has been created but has not been added to the engine yet.
               // This is the state of a request when it is first created.
  Assigned,    // The request has been added to the engine and is waiting to be scheduled.
  InProgress,  // The request has been scheduled and is currently being processed.
  Completed,   // The request has been completed successfully.
};

struct Request : std::enable_shared_from_this<Request>,
                 LeakChecked<Request>,
                 ExternalRefCounted<Request> {
  Request(std::shared_ptr<GeneratorParams> params);

  void Assign(std::shared_ptr<Engine> engine);

  void Schedule();

  void AddTokens(std::span<const int32_t> tokens);

  void GetNewTokens(std::span<int32_t> tokens);

  void GenerateNextTokens(DeviceSpan<float> logits);

  bool IsDone() const;

  void Remove();

  RequestStatus status_{RequestStatus::Unassigned};
  std::chrono::system_clock::time_point assigned_time_;

 private:
  std::vector<int32_t> unprocessed_input_ids_;
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<Search> search_;
  std::weak_ptr<Engine> engine_;
};

}  // namespace Generators
