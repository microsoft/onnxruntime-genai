// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Generators {

struct Embeddings {
  enum struct Mode {
    Input = 0,
    Output
  };

  Embeddings(State& state, Embeddings::Mode mode, const std::string& name);
  Embeddings(const Embeddings&) = delete;
  Embeddings& operator=(const Embeddings&) = delete;

  void Add();

  void UpdateSequenceLength();

  void ReuseEmbeddingsBuffer(const Embeddings& other);

  OrtValue* Get() { return embeddings_.get(); }

  auto& GetShape() const { return shape_; }

 private:
  State& state_;
  const Model& model_{state_.model_};
  std::array<int64_t, 3> shape_{};  // [batch_size, sequence_length, hidden_size]
  ONNXTensorElementDataType type_;
  const Mode mode_{};
  const std::string name_;
  std::unique_ptr<OrtValue> embeddings_;
  size_t index_{};
  StaticBuffer* sb_embeddings_{};
};

}  // namespace Generators
