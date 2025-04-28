// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "../generators.h"

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

  int32_t UnseenToken();

  DeviceSpan<int32_t> UnprocessedTokens();

  bool HasUnseenTokens() const;

  void GenerateNextTokens(DeviceSpan<float> logits);

  bool IsDone() const;

  void Remove();

  bool IsPrefill() const;

  int64_t CurrentSequenceLength() const;

  RequestStatus status_{RequestStatus::Unassigned};

  std::shared_ptr<GeneratorParams> Params();

 private:
  std::vector<int32_t> prefill_input_ids_;
  int64_t seen_sequence_length_{};
  int64_t processed_sequence_length_{};
  std::shared_ptr<GeneratorParams> params_;
  std::unique_ptr<Search> search_;
  std::weak_ptr<Engine> engine_;
  bool is_prefill_{true};
};

}  // namespace Generators
