// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "embeddings.h"

namespace Generators {

Embeddings::Embeddings(const Model& model, State& state, Embeddings::Mode mode, const std::string& name)
    : model_{model},
      state_{state},
      shape_{static_cast<int64_t>(state_.params_->batch_size) * state_.params_->search.num_beams,
             state_.params_->sequence_length, model_.config_->model.decoder.hidden_size},
      type_{mode == Embeddings::Mode::Input
                ? model_.session_info_->GetInputDataType(name)
                : model_.session_info_->GetOutputDataType(name)},
      mode_{mode},
      name_{name} {
  // Embeddings are only transient inputs and outputs.
  // They are never the user provided/requested model inputs/outputs
  // So only create the transient output and reuse that ortvalue for subsequent
  // steps in the pipeline.
  if (mode == Embeddings::Mode::Output) {
    if (state_.GetCapturedGraphInfo()) {
      sb_embeddings_ = state_.GetCapturedGraphInfo()->sb_embeddings_.get();
    }

    embeddings_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
  }
}

void Embeddings::Add() {
  if (mode_ == Embeddings::Mode::Input) {
    // In case the embeddings are input to a model, they are added
    // as a nullptr to reserve a slot in the inputs. The embedding
    // input will be overwritten when TransferState is invoked.
    index_ = state_.inputs_.size();
    state_.inputs_.push_back(nullptr);
    state_.input_names_.push_back(name_.c_str());
  } else {
    index_ = state_.outputs_.size();
    state_.outputs_.push_back(embeddings_.get());
    state_.output_names_.push_back(name_.c_str());
  }
}

void Embeddings::UpdateSequenceLength() {
  if (shape_[1] != 1) {
    shape_[1] = 1;

    if (mode_ == Embeddings::Mode::Output) {
      if (!sb_embeddings_) {
        embeddings_ = OrtValue::CreateTensor(*model_.allocator_device_, shape_, type_);
      } else {
        embeddings_ = sb_embeddings_->CreateTensorOnStaticBuffer(shape_, type_);
      }

      state_.outputs_[index_] = embeddings_.get();
    }
  }
}

void Embeddings::ReuseEmbeddingsBuffer(const Embeddings& other) {
  if (mode_ == Embeddings::Mode::Output ||
      other.mode_ == Embeddings::Mode::Input) {
    throw std::runtime_error("Incorrect usage of the embeddings inputs and outputs.");
  }

  // Share the output embeddings OrtValue* from other with the input embedding for this.
  state_.inputs_[index_] = other.state_.outputs_[other.index_];
}

}  // namespace Generators
