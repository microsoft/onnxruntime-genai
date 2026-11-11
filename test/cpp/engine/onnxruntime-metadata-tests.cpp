// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <filesystem>
#include <gtest/gtest.h>

#include "models/onnxruntime_api.h"

namespace Generators::test {
namespace {

struct MetadataAllocationTracker {
  MetadataAllocationTracker() {
    allocations = frees = 0;
    Ort::standard_allocator.Alloc = [](OrtAllocator*, size_t size) -> void* {
      ++allocations;
      return new std::byte[size];
    };
    Ort::standard_allocator.Free = [](OrtAllocator*, void* p) {
      if (p)
        ++frees;
      delete[] static_cast<std::byte*>(p);
    };
  }

  ~MetadataAllocationTracker() {
    Ort::standard_allocator = saved_allocator;
  }

  Ort::StandardAllocator saved_allocator{Ort::standard_allocator};
  inline static size_t allocations{};
  inline static size_t frees{};
};

TEST(OrtMetadataTest, FreesKeysAndPointerArrayThroughAllocator) {
  auto env = OrtEnv::Create(ORT_LOGGING_LEVEL_ERROR, "metadata-test");
  for (bool has_metadata : {false, true}) {
    SCOPED_TRACE(has_metadata);
    const std::filesystem::path path{has_metadata ? MODEL_PATH "longrope-context/model.onnx"
                                                  : MODEL_PATH "longrope-context-without-metadata/model.onnx"};
    auto options = OrtSessionOptions::Create();
    options->SetIntraOpNumThreads(1);
    auto session = OrtSession::Create(*env, path.c_str(), options.get());
    auto metadata = session->GetModelMetadata();
    MetadataAllocationTracker tracker;
    const auto keys = metadata->GetCustomMetadataMapKeysAllocated();
    if (has_metadata) {
      EXPECT_EQ(keys.size(), 2u);
      EXPECT_NE(std::find(keys.begin(), keys.end(), "com.microsoft.genai.rope_context_length_input"), keys.end());
      EXPECT_NE(std::find(keys.begin(), keys.end(), "com.microsoft.genai.original_context_length"), keys.end());
      EXPECT_EQ(tracker.allocations, keys.size() + 1);
    } else {
      EXPECT_TRUE(keys.empty());
    }
    EXPECT_EQ(tracker.frees, tracker.allocations);
  }
}

}  // namespace
}  // namespace Generators::test
