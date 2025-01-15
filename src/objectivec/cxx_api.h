// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// wrapper for ORT GenAI C/C++ API headers

#if defined(__clang__)
#pragma clang diagnostic push
// ignore clang documentation-related warnings
// instead, we will rely on Doxygen warnings for the C/C++ API headers
#pragma clang diagnostic ignored "-Wdocumentation"
#endif  // defined(__clang__)

#include "ort_genai.h"

#if defined(__clang__)
#pragma clang diagnostic pop
#endif  // defined(__clang__)
