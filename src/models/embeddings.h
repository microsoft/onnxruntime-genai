// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Generators {

struct Model;

struct Embeddings {
  enum struct Mode {
    Input = 0,
    Output
  };

  Embeddings(State& state, Embeddings::Mode mode, const std::string& name, int64_t hidden_size = 0);
  Embeddings(const Embeddings&) = delete;
  Embeddings& operator=(const Embeddings&) = delete;

  void Add();

  void UpdateSequenceLength(size_t new_length);

  void ReuseEmbeddingsBuffer(const Embeddings& other);

  // Prefill chunking support (input mode only): temporarily replaces the input embeddings
  // tensor with a non-owning view of the [offset, offset + length) slice along the sequence
  // dimension, so a long prompt can be fed to the decoder in several smaller runs.
  // Only supported for a batch-beam size of 1, where the slice is contiguous in memory.
  void UseChunkView(size_t offset, size_t length);
  void RestoreFullView();

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
  std::unique_ptr<OrtValue> chunk_view_;  // Non-owning view into embeddings_ used during prefill chunking
  size_t index_{};
};

}  // namespace Generators
