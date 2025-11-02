// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "oga_object.h"
#include "oga_borrowed_view.h"
#include "oga_utils.h"

namespace OgaPy {

struct OgaGeneratorParams : OgaObject {
  explicit OgaGeneratorParams(::OgaGeneratorParams* p) : ptr_(p) {}
  ~OgaGeneratorParams() override { if (ptr_) OgaDestroyGeneratorParams(ptr_); }
  ::OgaGeneratorParams* get() const { return ptr_; }
private:
  ::OgaGeneratorParams* ptr_;
};

struct OgaGenerator : OgaObject {
  explicit OgaGenerator(::OgaGenerator* p) : ptr_(p) {}
  ~OgaGenerator() override { if (ptr_) OgaDestroyGenerator(ptr_); }
  ::OgaGenerator* get() const { return ptr_; }
  
  // Get sequence data as a borrowed view (automatically handles reference counting)
  GeneratorSequenceDataView* GetSequenceData(size_t index) {
    const int32_t* data = OgaGenerator_GetSequenceData(ptr_, index);
    size_t count = OgaGenerator_GetSequenceCount(ptr_, index);  // Returns size_t, not OgaResult
    return new GeneratorSequenceDataView(this, data, count);
  }
  
  // Get next tokens as a borrowed view (automatically handles reference counting)
  // WARNING: This view is invalidated by the next OgaGenerator call (temporal borrow)
  NextTokensView* GetNextTokens() {
    const int32_t* tokens = nullptr;
    size_t count = 0;
    OgaCheckResult(OgaGenerator_GetNextTokens(ptr_, &tokens, &count));
    return new NextTokensView(this, tokens, count);
  }
  
private:
  ::OgaGenerator* ptr_;
};

} // namespace OgaPy
