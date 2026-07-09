// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
std::string DetermineLoadLibraryError(const char* filename);

// Returns true if `filename` (and its native dependencies) can actually be loaded. Used to
// pre-flight an execution provider library before registering it with ONNX Runtime: a provider
// whose native runtime is unavailable (e.g. a removed or disabled GPU) is then rejected cleanly
// instead of leaving ONNX Runtime with a half-registered library that crashes at environment
// teardown. On non-Windows platforms this pre-flight is a no-op that returns true.
bool CanLoadLibrary(const char* filename);