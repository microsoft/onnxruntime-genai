// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Entry point for the dedicated Engine unit-test binary (`engine_unit_tests`).
//
// This binary links the genai OBJECT library (onnxruntime-genai-obj) rather than the public shared
// library so it can exercise internal continuous-batching Engine types (Block, BlockPool, and, as
// the framework grows, the scheduler, cache manager, and Request invariants) directly as white-box
// unit tests. It deliberately stays separate from `unit_tests` (which validates only the public C
// API through the shared library) so that internal test-only fakes and helpers never leak into the
// public-boundary suite and so Engine failures point straight at Engine correctness.
//
// These are Tier 0 pure unit tests: they require no model download, no ONNX session, and no GPU.
// They must run in seconds on every CPU and CUDA native PR build.

#include <gtest/gtest.h>

#include "telemetry_test_environment.h"

int main(int argc, char** argv) {
  // Suppress telemetry before any genai symbol runs, matching the other native test binaries.
  Generators::test::SuppressTelemetryForTests();

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
