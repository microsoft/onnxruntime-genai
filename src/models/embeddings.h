// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Generators {

struct Embeddings {
  enum struct Mode {
    Input = 0,
    Output
  };

  Embeddings(const Model& model, State& state, Embeddings::Mode mode, const std::string& name);

  Embeddings(Embeddings&& other, State& state);

  void Add();

  void UpdateSequenceLength();

  Embeddings& operator=(const Embeddings& other);

  OrtValue* Get() { return embeddings_.get(); }

 private:
  const Model& model_;
  State& state_;
  std::array<int64_t, 3> shape_{};  // [batch_size, sequence_length, hidden_size]
  ONNXTensorElementDataType type_;
  const Mode mode_{};
  const std::string name_;
  std::unique_ptr<OrtValue> embeddings_;
  size_t index_{};
};

}  // namespace Generators
