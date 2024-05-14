// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "fast_gelu2.h"
#include "paged_attention.h"

FxLoadCustomOpFactory LoadCustomOpClasses_Contrib2 = []() -> CustomOpArray& {
  static OrtOpLoader op_loader(
      []() { return nullptr; }
      ,
      CustomCudaStructV2("FastGelu", FastGelu2<float>),
      CustomCudaStructV2("PagedAttention", PagedAttention<ortc::MFloat16>),
      CustomCudaStructV2("FastGelu", FastGelu2<ortc::MFloat16>),
      CustomCudaStructV2("FastGelu", FastGelu2<ortc::BFloat16>)
  );
  return op_loader.GetCustomOps();
};
