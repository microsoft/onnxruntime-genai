// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace Generators {

struct MultiModalFeatures {
  enum struct Mode {
    Input = 0,
    Output
  };

  MultiModalFeatures(State& state, MultiModalFeatures::Mode mode, const std::string& name, int64_t batch_size, int64_t num_feature_tokens);
  MultiModalFeatures(const MultiModalFeatures&) = delete;
  MultiModalFeatures& operator=(const MultiModalFeatures&) = delete;

  void Add();
  void Update(bool is_prompt);
  void ReuseFeaturesBuffer(MultiModalFeatures& other);

  // Pre-allocate an empty features tensor for Input mode when no source session provides one.
  // Used when the embedding model requires an input (e.g., audio_features) but no corresponding
  // encoder session exists.
  void AllocateEmptyFeatures();

  // Reshape features tensor in-place (e.g., flatten 3D [B, T, H] to 2D [B*T, H])
  void ReshapeFeatures(std::vector<int64_t> new_shape);

  auto& GetShape() const { return shape_; }
  OrtValue* Get() { return features_.get(); }

 private:
  State& state_;
  const Model& model_{state_.model_};

  std::vector<int64_t> shape_;  // [num_feature_tokens, hidden_size]
  ONNXTensorElementDataType type_;

  const Mode mode_{};
  const std::string name_;

  std::unique_ptr<OrtValue> features_;
  size_t index_{~0U};
};

// Scatters image feature rows into an embeddings buffer on the host, at the given token positions,
// consuming the image features in order. It overwrites the text embedding rows that correspond to
// image placeholder tokens with the produced image features.
//
// inputs_embeds:      mutable float embeddings tensor, shape [batch, seq, hidden] or [seq, hidden].
// image_features:     float image features tensor, shape [num_image_tokens, hidden].
// target_token_rows:  flat row indices (into the seq dimension) to overwrite, in order.
//
// Returns the number of image feature rows consumed. Throws if the hidden dimensions mismatch.
size_t MergeImageFeaturesIntoEmbeddings(OrtValue& inputs_embeds,
                                        const OrtValue& image_features,
                                        std::span<const int64_t> target_token_rows);

}  // namespace Generators
