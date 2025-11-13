// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"
#include "oga_utils.h"

namespace OgaPy {

// Forward declarations
struct OgaNamedTensors;
struct OgaStringArray;

struct OgaImages : OgaObject {
  explicit OgaImages(::OgaImages* p) : ptr_(p) {}
  ~OgaImages() override { if (ptr_) OgaDestroyImages(ptr_); }
  ::OgaImages* get() const { return ptr_; }
private:
  ::OgaImages* ptr_;
};

struct OgaAudios : OgaObject {
  explicit OgaAudios(::OgaAudios* p) : ptr_(p) {}
  ~OgaAudios() override { if (ptr_) OgaDestroyAudios(ptr_); }
  ::OgaAudios* get() const { return ptr_; }
private:
  ::OgaAudios* ptr_;
};

struct OgaMultiModalProcessor : OgaObject {
  explicit OgaMultiModalProcessor(::OgaMultiModalProcessor* p) : ptr_(p) {}
  ~OgaMultiModalProcessor() override { if (ptr_) OgaDestroyMultiModalProcessor(ptr_); }
  ::OgaMultiModalProcessor* get() const { return ptr_; }
  
  // Process images with a prompt
  OgaNamedTensors* ProcessImages(const char* prompt, const OgaImages* images) const {
    ::OgaNamedTensors* input_tensors = nullptr;
    OgaCheckResult(OgaProcessorProcessImages(ptr_, prompt, images->get(), &input_tensors));
    return new OgaNamedTensors(input_tensors);
  }
  
  // Process images with multiple prompts
  OgaNamedTensors* ProcessImagesAndPrompts(const OgaStringArray* prompts, const OgaImages* images) const {
    ::OgaNamedTensors* input_tensors = nullptr;
    OgaCheckResult(OgaProcessorProcessImagesAndPrompts(ptr_, prompts->get(), images->get(), &input_tensors));
    return new OgaNamedTensors(input_tensors);
  }
  
  // Process audios with a prompt
  OgaNamedTensors* ProcessAudios(const char* prompt, const OgaAudios* audios) const {
    ::OgaNamedTensors* input_tensors = nullptr;
    OgaCheckResult(OgaProcessorProcessAudios(ptr_, prompt, audios->get(), &input_tensors));
    return new OgaNamedTensors(input_tensors);
  }
  
  // Process audios with multiple prompts
  OgaNamedTensors* ProcessAudiosAndPrompts(const OgaStringArray* prompts, const OgaAudios* audios) const {
    ::OgaNamedTensors* input_tensors = nullptr;
    OgaCheckResult(OgaProcessorProcessAudiosAndPrompts(ptr_, prompts->get(), audios->get(), &input_tensors));
    return new OgaNamedTensors(input_tensors);
  }
  
  // Process images and audios with a prompt
  OgaNamedTensors* ProcessImagesAndAudios(const char* prompt, const OgaImages* images, const OgaAudios* audios) const {
    ::OgaNamedTensors* input_tensors = nullptr;
    OgaCheckResult(OgaProcessorProcessImagesAndAudios(ptr_, prompt, images->get(), audios->get(), &input_tensors));
    return new OgaNamedTensors(input_tensors);
  }
  
  // Process images and audios with multiple prompts
  OgaNamedTensors* ProcessImagesAndAudiosAndPrompts(const OgaStringArray* prompts, const OgaImages* images, const OgaAudios* audios) const {
    ::OgaNamedTensors* input_tensors = nullptr;
    OgaCheckResult(OgaProcessorProcessImagesAndAudiosAndPrompts(ptr_, prompts->get(), images->get(), audios->get(), &input_tensors));
    return new OgaNamedTensors(input_tensors);
  }
  
  // Decode tokens to a string
  const char* Decode(const int32_t* tokens, size_t token_count) const {
    const char* out_string = nullptr;
    OgaCheckResult(OgaProcessorDecode(ptr_, tokens, token_count, &out_string));
    return out_string;
  }
  
private:
  ::OgaMultiModalProcessor* ptr_;
};

} // namespace OgaPy
