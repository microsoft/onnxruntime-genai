// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "model.h"
#include "embeddings.h"

namespace Generators {

Embeddings::Embeddings(State& state, Embeddings::Mode mode, const std::string& name)
    : state_{state},
      shape_{static_cast<int64_t>(state_.params_->search.batch_size) * state_.params_->search.num_beams,
             0, model_.config_->model.decoder.hidden_size},
      type_{mode == Embeddings::Mode::Input
                ? model_.session_info_->GetInputDataType(name)
                : model_.session_info_->GetOutputDataType(name)},
      mode_{mode},
      name_{name} {
  // Embeddings are only transient inputs and outputs.
  // They are never the user provided/requested model inputs/outputs
  // So only create the transient input and reuse that ortvalue for previous
  // steps in the pipeline.
  if (mode == Embeddings::Mode::Input) {
    if (state_.GetCapturedGraphInfo()) {
      sb_embeddings_ = state_.GetCapturedGraphInfo()->sb_embeddings_.get();
    }

    embeddings_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), shape_, type_);
  }
}

void Embeddings::Add() {
  if (mode_ == Embeddings::Mode::Output) {
    // In case the embeddings are output of a model, they are added
    // as a nullptr to reserve a slot in the outputs. The embedding
    // output will be overwritten by the input of the following model
    // when ReuseEmbeddingsBuffer is invoked. For example, if we have
    // a pipeline that looks like EmbeddingModel -> TextModel, we
    // create the embedding tensor in the TextModel as an input and
    // simply reuse it in the EmbeddingModel as an output.
    index_ = state_.outputs_.size();
    state_.outputs_.push_back(nullptr);
    state_.output_names_.push_back(name_.c_str());
  } else {
    index_ = state_.inputs_.size();
    state_.inputs_.push_back(embeddings_.get());
    state_.input_names_.push_back(name_.c_str());
  }
}

void Embeddings::UpdateSequenceLength(size_t new_length) {
  if (static_cast<size_t>(shape_[1]) != new_length) {
    shape_[1] = new_length;

    if (mode_ == Embeddings::Mode::Input) {
      if (!sb_embeddings_) {
        embeddings_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), shape_, type_);
      } else {
        embeddings_ = sb_embeddings_->CreateTensorOnStaticBuffer(shape_, type_);
      }

      state_.inputs_[index_] = embeddings_.get();
    }
  }
}

void Embeddings::ReuseEmbeddingsBuffer(const Embeddings& other) {
  if (mode_ == Embeddings::Mode::Input ||
      other.mode_ == Embeddings::Mode::Output) {
    throw std::runtime_error("Incorrect usage of the embeddings inputs and outputs.");
  }

  // Share the input embeddings OrtValue* from other with the output embedding for this.
  state_.outputs_[index_] = other.state_.inputs_[other.index_];
}

}  // namespace Generators
