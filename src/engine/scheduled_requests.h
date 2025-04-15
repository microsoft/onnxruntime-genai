// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Generators {

struct ScheduledRequests {
  ScheduledRequests(std::vector<std::shared_ptr<Request>> requests);

  std::unique_ptr<OrtRunOptions> RunOptions();

  OrtValue* InputIds();

  OrtValue* PositionIds();

  std::vector<OrtValue*> KeyCaches();

  std::vector<OrtValue*> ValueCaches();

  OrtValue* CumulativeSequenceLengths();

  OrtValue* SequenceLengths();

  OrtValue* MaxQueryLength();

  OrtValue* MaxSequenceLength();

  OrtValue* BlockTable();

  OrtValue* SlotMapping();

  OrtValue* Logits();

  void GenerateNextTokens();

  explicit operator bool() const;

 private:
  std::vector<std::shared_ptr<Request>> requests_;
};

}  // namespace Generators
