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
                ? model_.session_info_.GetInputDataType(name)
                : model_.session_info_.GetOutputDataType(name)},
      mode_{mode},
      name_{name} {
  // Embeddings are only transient inputs and outputs.
  // They are never the user provided/requested model inputs/outputs
  // So only create the transient input and reuse that ortvalue for previous
  // steps in the pipeline.
  if (mode == Embeddings::Mode::Input) {
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
  //if (static_cast<size_t>(shape_[1]) != new_length) {
    shape_[1] = new_length;

    if (mode_ == Embeddings::Mode::Input) {
      embeddings_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), shape_, type_);
      state_.inputs_[index_] = embeddings_.get();
    }
  //}
}

void Embeddings::ReuseEmbeddingsBuffer(const Embeddings& other) {
  if (mode_ == Embeddings::Mode::Input ||
      other.mode_ == Embeddings::Mode::Output) {
    throw std::runtime_error("Incorrect usage of the embeddings inputs and outputs.");
  }

  // Share the input embeddings OrtValue* from other with the output embedding for this.
  state_.outputs_[index_] = other.state_.inputs_[other.index_];
}

WindowedEmbeddings::WindowedEmbeddings(State& state, Embeddings::Mode mode, const std::string& name) 
: Embeddings(state, mode, name),
    state_{state},
      shape_{static_cast<int64_t>(state_.params_->search.batch_size) * state_.params_->search.num_beams,
             0, model_.config_->model.decoder.hidden_size},
      type_{mode == Embeddings::Mode::Input
                ? model_.session_info_.GetInputDataType(name)
                : model_.session_info_.GetOutputDataType(name)},
      mode_{mode},
      name_{name} {

      //name_ = model_.config_->model.decoder.inputs.embeddings.c_str();
      window_size_ = model_.config_->model.decoder.sliding_window->window_size;
      //shape_ = {1, model_.config_->model.decoder.sliding_window->window_size};
      type_ = model_.session_info_.GetInputDataType(name_);
  // Embeddings are only transient inputs and outputs.
  // They are never the user provided/requested model inputs/outputs
  // So only create the transient input and reuse that ortvalue for previous
  // steps in the pipeline.
  if (mode == Embeddings::Mode::Input) {
    embeddings_ = OrtValue::CreateTensor(model_.p_device_->GetAllocator(), shape_, type_);
  }
}




void WindowedEmbeddings::Update(Embeddings& embeddings) {
  const auto& full_embeddings = embeddings.Get();
  const auto& full_shape = embeddings.GetShape();  // [batch_size, sequence_length, hidden_size]

  // Assuming batch_size = 1

  int64_t sequence_length = full_shape[1];
  int64_t hidden_size = full_shape[2];

  const uint16_t* full_data = full_embeddings->GetTensorData<uint16_t>();

  if (window_index_ == 0) {
    num_windows_ = (sequence_length + window_size_ - 1) / window_size_;
    shape_ = {
        static_cast<int64_t>(1),
        static_cast<int64_t>(window_size_),
        static_cast<int64_t>(hidden_size)
    };

    embeddings_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), shape_, type_);
    std::copy_n(
        full_data,
        window_size_ * hidden_size * 2,
        embeddings_->GetTensorMutableData<uint16_t>());

  } else if (window_index_ < num_windows_) {
    shape_ = {
        static_cast<int64_t>(1),
        static_cast<int64_t>(window_size_),
        static_cast<int64_t>(hidden_size)
    };
    embeddings_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), shape_, type_);
    std::copy_n(
        full_data + window_index_ * window_size_ * hidden_size * 2,
        window_size_ * hidden_size * 2,
        embeddings_->GetTensorMutableData<uint16_t>());

  } else {
    // Final token case (e.g., generated token)
    shape_ = {1, 1, hidden_size};
    embeddings_ = OrtValue::CreateTensor(model_.p_device_inputs_->GetAllocator(), shape_, type_);
    std::copy_n(
        full_data + (sequence_length - 1) * hidden_size * 2,
        hidden_size * 2,
        embeddings_->GetTensorMutableData<uint16_t>());

  }

  auto it = std::find(state_.input_names_.begin(), state_.input_names_.end(), name_);
  if (it != state_.input_names_.end()) {
    size_t index = std::distance(state_.input_names_.begin(), it);
    state_.inputs_[index] = embeddings_.get();
  } else {
    std::cerr << "Error: Input name '" << name_ << "' not found in input_names_." << std::endl;
  }
  window_index_++;
}

std::unique_ptr<Embeddings> CreateInputEmbeddings(State& state, Embeddings::Mode mode, const std::string& name) {
  if (state.model_.config_->model.decoder.sliding_window.has_value() && state.model_.config_->model.decoder.sliding_window->slide_inputs) {
    return std::make_unique<WindowedEmbeddings>(state, mode, name);
  } else {
    return std::make_unique<Embeddings>(state, mode, name);
  }
}

}  // namespace Generators
