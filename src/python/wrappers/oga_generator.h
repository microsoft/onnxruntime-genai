// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"
#include "oga_borrowed_view.h"
#include "oga_utils.h"

namespace OgaPy {

// Forward declaration (full definition needed for SetActiveAdapter)
struct OgaAdapters;

struct OgaGeneratorParams {
  explicit OgaGeneratorParams(::OgaGeneratorParams* p) : ptr_(p) {}
  ~OgaGeneratorParams() { if (ptr_) OgaDestroyGeneratorParams(ptr_); }
  ::OgaGeneratorParams* get() const { return ptr_; }
  
  // Set a search option (numeric value)
  void SetSearchNumber(const char* name, double value) {
    OgaCheckResult(OgaGeneratorParamsSetSearchNumber(ptr_, name, value));
  }
  
  // Set a search option (boolean value)
  void SetSearchBool(const char* name, bool value) {
    OgaCheckResult(OgaGeneratorParamsSetSearchBool(ptr_, name, value));
  }
  
  // Try to enable graph capture with a maximum batch size
  void TryGraphCaptureWithMaxBatchSize(int32_t max_batch_size) {
    OgaCheckResult(OgaGeneratorParamsTryGraphCaptureWithMaxBatchSize(ptr_, max_batch_size));
  }
  
  // Set guidance (for constrained generation)
  void SetGuidance(const char* type, const char* data, bool enable_ff_tokens) {
    OgaCheckResult(OgaGeneratorParamsSetGuidance(ptr_, type, data, enable_ff_tokens));
  }
  
private:
  ::OgaGeneratorParams* ptr_;
};

struct OgaGenerator {
  explicit OgaGenerator(::OgaGenerator* p) : ptr_(p) {}
  ~OgaGenerator() { if (ptr_) OgaDestroyGenerator(ptr_); }
  ::OgaGenerator* get() const { return ptr_; }
  
  // Check if generation is complete
  bool IsDone() const {
    return OgaGenerator_IsDone(ptr_);
  }
  
  // Check if the session has been terminated
  bool IsSessionTerminated() const {
    return OgaGenerator_IsSessionTerminated(ptr_);
  }
  
  // Set a model input tensor
  void SetModelInput(const char* name, OgaTensor* tensor) {
    OgaCheckResult(OgaGenerator_SetModelInput(ptr_, name, tensor));
  }
  
  // Set all model inputs at once
  void SetInputs(const OgaNamedTensors* named_tensors) {
    OgaCheckResult(OgaGenerator_SetInputs(ptr_, named_tensors));
  }
  
  // Append token sequences to the generator
  void AppendTokenSequences(const OgaSequences* sequences) {
    OgaCheckResult(OgaGenerator_AppendTokenSequences(ptr_, sequences->get()));
  }
  
  // Append tokens to the generator
  void AppendTokens(const int32_t* input_ids, size_t input_ids_count) {
    OgaCheckResult(OgaGenerator_AppendTokens(ptr_, input_ids, input_ids_count));
  }
  
  // Generate the next token
  void GenerateNextToken() {
    OgaCheckResult(OgaGenerator_GenerateNextToken(ptr_));
  }
  
  // Get next tokens as a borrowed view (automatically handles reference counting)
  // WARNING: This view is invalidated by the next OgaGenerator call (temporal borrow)
  NextTokensView* GetNextTokens() {
    const int32_t* tokens = nullptr;
    size_t count = 0;
    OgaCheckResult(OgaGenerator_GetNextTokens(ptr_, &tokens, &count));
    return new NextTokensView(this, tokens, count);
  }
  
  // Set a runtime option
  void SetRuntimeOption(const char* key, const char* value) {
    OgaCheckResult(OgaGenerator_SetRuntimeOption(ptr_, key, value));
  }
  
  // Rewind generation to a specific length
  void RewindTo(size_t new_length) {
    OgaCheckResult(OgaGenerator_RewindTo(ptr_, new_length));
  }
  
  // Get an input tensor
  OgaTensor* GetInput(const char* name) {
    OgaTensor* out = nullptr;
    OgaCheckResult(OgaGenerator_GetInput(ptr_, name, &out));
    return out;
  }
  
  // Get an output tensor
  OgaTensor* GetOutput(const char* name) {
    OgaTensor* out = nullptr;
    OgaCheckResult(OgaGenerator_GetOutput(ptr_, name, &out));
    return out;
  }
  
  // Get logits tensor
  OgaTensor* GetLogits() {
    OgaTensor* out = nullptr;
    OgaCheckResult(OgaGenerator_GetLogits(ptr_, &out));
    return out;
  }
  
  // Set logits tensor
  void SetLogits(OgaTensor* tensor) {
    OgaCheckResult(OgaGenerator_SetLogits(ptr_, tensor));
  }
  
  // Get the number of tokens in a sequence
  size_t GetSequenceCount(size_t index) const {
    return OgaGenerator_GetSequenceCount(ptr_, index);
  }
  
  // Get sequence data as a borrowed view (automatically handles reference counting)
  GeneratorSequenceDataView* GetSequenceData(size_t index) {
    const int32_t* data = OgaGenerator_GetSequenceData(ptr_, index);
    size_t count = OgaGenerator_GetSequenceCount(ptr_, index);
    return new GeneratorSequenceDataView(this, data, count);
  }
  
  // Set active adapter for this generator (defined after OgaAdapters)
  void SetActiveAdapter(OgaAdapters* adapters, const char* adapter_name);
  
private:
  ::OgaGenerator* ptr_;
};

} // namespace OgaPy
