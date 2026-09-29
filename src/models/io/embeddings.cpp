// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "generator/generators.h"
#include "models/model.h"
#include "models/io/embeddings.h"

namespace Generators {

Embeddings::Embeddings(State& state, Embeddings::Mode mode, const std::string& name, int64_t hidden_size)
    : state_{state},
      shape_{static_cast<int64_t>(state_.params_->search.batch_size) * state_.params_->search.num_beams,
             0, hidden_size > 0 ? hidden_size : model_.config_->model.decoder.hidden_size},
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
    embeddings_ = OrtValue::CreateTensor(state_.p_session_device_inputs_->GetAllocator(), shape_, type_);
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
  RestoreFullView();

  if (static_cast<size_t>(shape_[1]) != new_length) {
    shape_[1] = new_length;

    if (mode_ == Embeddings::Mode::Input) {
      embeddings_ = OrtValue::CreateTensor(state_.p_session_device_inputs_->GetAllocator(), shape_, type_);
      state_.inputs_[index_] = embeddings_.get();
    }
  }
}

void Embeddings::UseChunkView(size_t offset, size_t length) {
  if (mode_ != Embeddings::Mode::Input) {
    throw std::runtime_error("Embeddings::UseChunkView is only valid for input embeddings.");
  }
  if (shape_[0] != 1) {
    throw std::runtime_error("Prefill chunking requires a batch size of 1 for the embeddings input.");
  }
  if (offset + length > static_cast<size_t>(shape_[1])) {
    throw std::runtime_error("Embeddings::UseChunkView - requested chunk exceeds the embeddings sequence length.");
  }

  const size_t element_size = Ort::SizeOf(type_);
  const size_t hidden_size = static_cast<size_t>(shape_[2]);
  auto* raw = static_cast<uint8_t*>(embeddings_->GetTensorMutableRawData());

  std::array<int64_t, 3> chunk_shape{shape_[0], static_cast<int64_t>(length), shape_[2]};
  chunk_view_ = OrtValue::CreateTensor(embeddings_->GetTensorMemoryInfo(),
                                       raw + offset * hidden_size * element_size,
                                       length * hidden_size * element_size,
                                       std::span<const int64_t>(chunk_shape), type_);
  state_.inputs_[index_] = chunk_view_.get();
}

void Embeddings::RestoreFullView() {
  if (!chunk_view_)
    return;

  chunk_view_ = nullptr;
  if (mode_ == Embeddings::Mode::Input) {
    state_.inputs_[index_] = embeddings_.get();
  }
}

void Embeddings::ReuseEmbeddingsBuffer(const Embeddings& other) {
  if (mode_ == Embeddings::Mode::Input ||
      other.mode_ == Embeddings::Mode::Output) {
    throw std::runtime_error("Incorrect usage of the embeddings inputs and outputs.");
  }

  OrtValue* consumer = other.state_.inputs_[other.index_];
  auto& consumer_device = *other.state_.p_session_device_inputs_;

  // This session is about to rewrite the mirror that the last upload reads from. The wait is here
  // rather than after the upload, so that it does not hold back the consumer's run.
  if (upload_pending_) {
    consumer_device_->Synchronize();
    upload_pending_ = false;
  }

  if (SessionCanAccess(*state_.p_session_device_, consumer_device)) {
    // Share the input embeddings OrtValue* from other with the output embedding for this.
    consumer_ = nullptr;
    consumer_device_ = nullptr;
    consumer_bytes_ = {};
    host_view_ = nullptr;
    state_.outputs_[index_] = consumer;
    return;
  }

  // The consumer allocated its input on a device this session has no EP for. Binding it as an
  // output would have ORT write host bytes over that device pointer, so this session writes the
  // buffer's host mirror instead, and CopyToConsumer() uploads it once the session has run.
  if (consumer != consumer_) {
    // The decoder reallocated for a new sequence length: wrap the new buffer once here, so decode
    // steps reuse the wrapper and its mirror instead of allocating a mirror per step.
    auto info = consumer->GetTensorTypeAndShapeInfo();
    auto shape = info->GetShape();
    consumer_bytes_ = ByteWrapTensor(consumer_device, *consumer);
    auto mirror = consumer_bytes_.empty() ? std::span<uint8_t>{} : consumer_bytes_.CpuSpan();
    host_view_ = OrtValue::CreateTensor(*OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
                                        mirror.data(), mirror.size_bytes(), shape, info->GetElementType());
    consumer_ = consumer;
    consumer_device_ = &consumer_device;
  }
  state_.outputs_[index_] = host_view_.get();
}

void Embeddings::CopyToConsumer() {
  if (!consumer_ || consumer_bytes_.empty())
    return;

  // Queued on the consumer device's stream ahead of the consumer's run. ReuseEmbeddingsBuffer waits
  // for it before the mirror is written again.
  consumer_bytes_.CopyCpuToDevice();
  upload_pending_ = true;
}

}  // namespace Generators
