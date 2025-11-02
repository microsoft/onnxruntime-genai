// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "wrappers/oga_wrappers.h"

namespace OgaPy {

// OgaSequences::GetSequenceData implementation
SequenceDataView* OgaSequences::GetSequenceData(size_t index) {
  const int32_t* data = OgaSequencesGetSequenceData(ptr_, index);
  size_t count = OgaSequencesGetSequenceCount(ptr_, index);
  return new SequenceDataView(this, data, count);
}

// OgaTokenizer::GetEosTokenIds implementation
EosTokenIdsView* OgaTokenizer::GetEosTokenIds() {
  const int32_t* token_ids = nullptr;
  size_t count = 0;
  OgaCheckResult(OgaTokenizerGetEosTokenIds(ptr_, &token_ids, &count));
  return new EosTokenIdsView(this, token_ids, count);
}

}  // namespace OgaPy
