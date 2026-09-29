// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ort_genai_c.h"

extern "C" OgaResult* OgaCreateResultFromError(const char* error);

#define OGA_CAPI_TRY try {
#define OGA_CAPI_CATCH                         \
  }                                            \
  catch (const std::exception& e) {            \
    return OgaCreateResultFromError(e.what()); \
  }
