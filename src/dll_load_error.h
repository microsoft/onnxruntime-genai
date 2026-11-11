// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>

std::string DetermineLoadLibraryError(const char* filename);

// Returns true if `filename` (and its native dependencies) can actually be loaded. Used to
// pre-flight an execution provider library before registering it with ONNX Runtime: a provider
// whose native runtime is unavailable (e.g. a removed or disabled GPU) is then rejected cleanly
// instead of leaving ONNX Runtime with a half-registered library that crashes at environment
// teardown. On Windows this fully loads the library (LoadLibraryEx) to verify it and its native
// dependencies resolve; on non-Windows platforms it does the equivalent dlopen(RTLD_NOW) check.
bool CanLoadLibrary(const char* filename);