// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "tensor.h"
#include "vision_position.h"

#include <memory>
#include <string>
#include <vector>

namespace Generators {

struct Images;
struct Videos;
struct Tokenizer;

struct VisionPreparationInput {
  const std::string& prompt;
  std::span<const int32_t> raw_tokens;
  const Images* images;
  const Videos* videos;
  size_t remaining_rows;
};

struct VisionInputSpec {
  std::string name;
  ONNXTensorElementDataType type;
  std::vector<int64_t> shape;
};

struct VisionEncoderUnit {
  size_t rows;
  std::vector<VisionInputSpec> inputs;
};

struct VisionFeatureSpan {
  size_t start;
  size_t unit;
  size_t rows;
  size_t feature_offset;
  std::shared_ptr<const VisionPosition> positions;
};

struct VisionPlan {
  virtual ~VisionPlan() = default;
  std::vector<int32_t> tokens;
  std::vector<VisionEncoderUnit> units;
  std::vector<VisionFeatureSpan> spans;
};

struct VisionPreparation {
  virtual ~VisionPreparation() = default;
  virtual size_t Estimate(const VisionPreparationInput& input, size_t host_limit) const = 0;
  virtual std::unique_ptr<const VisionPlan> Plan(const Tokenizer& tokenizer,
                                                 const VisionPreparationInput& input, size_t host_limit) const = 0;
  // Donate private owned CPU feeds; materializers must not reuse or mutate their buffers.
  virtual std::vector<NamedTensors> Materialize(const VisionPlan& plan) const = 0;
  virtual void ValidateTextTokens(std::span<const int32_t> tokens) const = 0;
  virtual std::array<int64_t, 2> CoordinateRange() const = 0;
};

}  // namespace Generators
