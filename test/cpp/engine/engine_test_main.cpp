// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Entry point for the dedicated Engine unit-test binary (`engine_unit_tests`).
//
// This binary links the genai OBJECT library (onnxruntime-genai-obj) rather than the public shared
// library so it can exercise internal continuous-batching Engine types (Block, BlockPool, the
// scheduler, cache manager, model executor, and Request invariants) directly as white-box unit
// tests. It deliberately stays separate from `unit_tests` (which validates only the public C API
// through the shared library) so that internal test-only fakes and helpers never leak into the
// public-boundary suite and so Engine failures point straight at Engine correctness.
//
// It hosts pure unit tests (no model, no ONNX session, no GPU) alongside component
// tests that mint requests from a tiny CPU fixture model. The fixture model only has to load: the
// component tests drive the scheduler and engine with recording doubles that stand in for the cache
// manager and model executor, so the ONNX graph is never executed and the suite still needs no GPU
// and no real inference. Everything runs in seconds on every CPU and CUDA native PR build.

#include <gtest/gtest.h>

#include "generators.h"
#include "telemetry_test_environment.h"

int main(int argc, char** argv) {
  // Suppress telemetry before any genai symbol runs, matching the other native test binaries.
  Generators::test::SuppressTelemetryForTests();

  ::testing::InitGoogleTest(&argc, argv);
  const int result = RUN_ALL_TESTS();
  // Tear genai down while ORT and CUDA are still alive. Left to the process-exit static
  // destructor, releasing the CUDA trivial session faults inside the CUDA EP.
  Generators::Shutdown();
  return result;
}
