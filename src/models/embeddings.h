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

  Embeddings(State& state, Embeddings::Mode mode, const std::string& name);
  Embeddings(const Embeddings&) = delete;
  Embeddings& operator=(const Embeddings&) = delete;

  void Add();

  void UpdateSequenceLength(size_t new_length);

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
};

struct WindowedEmbeddings : public Embeddings {
  WindowedEmbeddings(State& state, Embeddings::Mode mode, const std::string& name);
  WindowedEmbeddings(const WindowedEmbeddings&) = delete;
  WindowedEmbeddings& operator=(const WindowedEmbeddings&) = delete;

  void Update(Embeddings& embeddings);

 private:
  State& state_;
  const Model& model_{state_.model_};
  std::array<int64_t, 3> shape_{};  // [batch_size, sequence_length, hidden_size]
  ONNXTensorElementDataType type_;
  const Embeddings::Mode mode_{};
  const std::string name_;
  std::unique_ptr<OrtValue> embeddings_;
  size_t index_{};
  size_t input_index_{~0U};
  size_t window_size_{};
  size_t num_windows_{};
  size_t window_index_{};
  std::unique_ptr<OrtValue> total_sequence_length_;
  std::unique_ptr<OrtValue> past_sequence_length_;
  int32_t initial_num_tokens_{};
};

}  // namespace Generators;
