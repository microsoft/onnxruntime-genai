// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "paged_attention.h"

OrtOpLoader genai_op_loader(CustomCudaStructV2("PagedAttention", PagedAttention<ortc::MFloat16>));
