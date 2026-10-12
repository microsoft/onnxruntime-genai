// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>

#include "config.h"
#include "engine_test_doubles.h"
#include "models/runtime_profiles.h"
#include "models/session_options.h"

namespace Generators::test {

TEST(ConfigTest, ParsesStaticBatching) {
  Config config;

  OverlayConfig(config, R"({"engine":{"static_batching":{"max_batch_size":8}}})");

  ASSERT_TRUE(config.engine.static_batching.has_value());
  EXPECT_EQ(config.engine.static_batching->max_batch_size, 8u);
}

TEST(ConfigTest, ParsesSelectedLogitsInput) {
  Config config;
  EXPECT_TRUE(config.model.decoder.inputs.logits_indices.empty());

  OverlayConfig(config, R"({"model":{"decoder":{"inputs":{"logits_indices":"selected_rows"}}}})");

  EXPECT_EQ(config.model.decoder.inputs.logits_indices, "selected_rows");
}

TEST(ConfigTest, RejectsNonPositiveStaticBatchSize) {
  Config config;

  try {
    OverlayConfig(config, R"({"engine":{"static_batching":{"max_batch_size":0}}})");
    FAIL() << "Expected invalid max_batch_size to throw";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("engine:static_batching:max_batch_size:"), std::string::npos) << message;
  }
}

TEST(ConfigTest, ParsesAndAppliesOneMatchingRuntimeProfile) {
  Config config;
  OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{
      "num_blocks":32,"max_batch_size":4,"max_scheduled_tokens":1024}},
    "runtime_profiles":[{
      "id":"larger-gpu",
      "eligibility":{"minimum_total_device_memory_bytes":34359738368},
      "overlay":{
        "model":{"decoder":{"filename":"model_32gib.onnx"}},
        "engine":{"dynamic_batching":{"num_blocks":64,"max_batch_size":8}},
        "search":{"chunk_size":512},
        "speculative":{"max_draft_tokens":6}
      }
    }]
  })");

  ApplyRuntimeProfile(config, 34359738368ULL);

  ASSERT_TRUE(config.engine.dynamic_batching);
  EXPECT_EQ(config.engine.dynamic_batching->num_blocks, 64u);
  EXPECT_EQ(config.engine.dynamic_batching->max_batch_size, 8u);
  EXPECT_EQ(config.engine.dynamic_batching->max_scheduled_tokens, 1024u);
  EXPECT_EQ(config.search.chunk_size, 512u);
  EXPECT_EQ(config.model.decoder.filename, "model_32gib.onnx");
  EXPECT_EQ(config.speculative.max_draft_tokens, 6);
}

TEST(ConfigTest, RuntimeProfileUsesBaseWhenNoRangeMatches) {
  Config config;
  OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{"num_blocks":32}},
    "runtime_profiles":[{
      "id":"larger-gpu",
      "eligibility":{"minimum_total_device_memory_bytes":34359738368},
      "overlay":{"engine":{"dynamic_batching":{"num_blocks":64}}}
    }]
  })");

  ApplyRuntimeProfile(config, 25769803776ULL);

  ASSERT_TRUE(config.engine.dynamic_batching->num_blocks);
  EXPECT_EQ(*config.engine.dynamic_batching->num_blocks, 32u);
}

TEST(ConfigTest, RuntimeProfileUsesDefaultCudaDevice) {
  Config config;
  OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{"num_blocks":32}},
    "runtime_profiles":[{
      "id":"larger-gpu",
      "eligibility":{"minimum_total_device_memory_bytes":100},
      "overlay":{"engine":{"dynamic_batching":{"num_blocks":64}}}
    }]
  })");
  CountingCudaDevice device;
  device.state->total_memory_bytes = 100;

  ApplyRuntimeProfileForSelectedDevice(config, device);

  EXPECT_EQ(device.state->memory_queries, 1u);
  EXPECT_EQ(device.state->integrated_queries, 0u);
  EXPECT_EQ(device.state->device_id_queries, 1u);
  EXPECT_EQ(*config.engine.dynamic_batching->num_blocks, 64u);
  EXPECT_TRUE(config.runtime_profiles.empty());
}

TEST(ConfigTest, IntegratedAndDiscreteProfilesShareMemoryRangeWithoutAmbiguity) {
  Config config;
  OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{"num_blocks":32}},
    "runtime_profiles":[
      {"id":"integrated","eligibility":{"minimum_total_device_memory_bytes":100,
                                        "is_integrated":true},
       "overlay":{"engine":{"dynamic_batching":{"num_blocks":40}}}},
      {"id":"discrete","eligibility":{"minimum_total_device_memory_bytes":100,
                                      "is_integrated":false},
       "overlay":{"engine":{"dynamic_batching":{"num_blocks":80}}}}
    ]
  })");

  for (const auto [integrated, expected_blocks] : {std::pair{true, 40u}, std::pair{false, 80u}}) {
    auto candidate = config;
    CountingCudaDevice device;
    device.state->total_memory_bytes = 100;
    device.state->is_integrated = integrated;
    ApplyRuntimeProfileForSelectedDevice(candidate, device);
    EXPECT_EQ(*candidate.engine.dynamic_batching->num_blocks, expected_blocks);
    EXPECT_EQ(device.state->integrated_queries, 1u);
    EXPECT_TRUE(candidate.runtime_profiles.empty());
  }

  auto legacy_caller = config;
  ApplyRuntimeProfile(legacy_caller, 100);
  EXPECT_EQ(*legacy_caller.engine.dynamic_batching->num_blocks, 32u);
}

TEST(ConfigTest, IntegratedCapabilityFailureUsesBaseAndReportsError) {
  Config config;
  OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{"num_blocks":32}},
    "runtime_profiles":[
      {"id":"generic","eligibility":{"minimum_total_device_memory_bytes":100,
                                      "maximum_total_device_memory_bytes":199},
       "overlay":{"engine":{"dynamic_batching":{"num_blocks":60}}}},
      {"id":"integrated","eligibility":{"minimum_total_device_memory_bytes":200,
                                        "is_integrated":true},
       "overlay":{"engine":{"dynamic_batching":{"num_blocks":40}}}}
    ]
  })");
  CountingCudaDevice device;
  device.state->total_memory_bytes = 100;
  device.state->is_integrated = std::nullopt;
  testing::internal::CaptureStderr();
  ApplyRuntimeProfileForSelectedDevice(config, device);
  const auto warning = testing::internal::GetCapturedStderr();
  EXPECT_NE(warning.find("test integrated-device query failed"), std::string::npos);
  EXPECT_EQ(*config.engine.dynamic_batching->num_blocks, 32u);
  EXPECT_TRUE(config.runtime_profiles.empty());
  EXPECT_EQ(device.state->integrated_queries, 1u);
}

TEST(ConfigTest, IntegratedEligibilityRequiresBooleanAndDisjointRanges) {
  for (const auto* value : {"0", "1", "\"true\"", "null"}) {
    Config config;
    const auto json = std::string{R"({"runtime_profiles":[{
      "id":"invalid","eligibility":{"minimum_total_device_memory_bytes":0,"is_integrated":)"} +
                      value + R"(},"overlay":{"search":{"chunk_size":1}}}]})";
    EXPECT_THROW(OverlayConfig(config, json), std::runtime_error) << value;
  }

  for (const auto* other : {"true", "false"}) {
    Config config;
    const auto json = std::string{R"({"runtime_profiles":[
      {"id":"generic","eligibility":{"minimum_total_device_memory_bytes":1,
        "maximum_total_device_memory_bytes":10},"overlay":{"search":{"chunk_size":1}}},
      {"id":"specific","eligibility":{"minimum_total_device_memory_bytes":10,
        "is_integrated":)"} +
                      other + R"(},"overlay":{"search":{"chunk_size":2}}}
    ]})";
    try {
      OverlayConfig(config, json);
      FAIL() << "Expected wildcard eligibility to overlap: " << other;
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string(error.what()).find("eligibility ranges overlap"), std::string::npos);
    }
  }
}

TEST(ConfigTest, FreeMemoryProfilesMatchInclusiveBoundsAndKeepLegacyCallersSafe) {
  Config config;
  OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{"num_blocks":32}},
    "runtime_profiles":[
      {"id":"less-free","eligibility":{"minimum_total_device_memory_bytes":100,
        "minimum_free_device_memory_bytes":50,"maximum_free_device_memory_bytes":80},
       "overlay":{"engine":{"dynamic_batching":{"num_blocks":40}}}},
      {"id":"more-free","eligibility":{"minimum_total_device_memory_bytes":100,
        "minimum_free_device_memory_bytes":81,"maximum_free_device_memory_bytes":100},
       "overlay":{"engine":{"dynamic_batching":{"num_blocks":80}}}}
    ]
  })");

  for (const auto [free_bytes, expected_blocks] : {
           std::pair{49u, 32u}, std::pair{50u, 40u}, std::pair{80u, 40u},
           std::pair{81u, 80u}, std::pair{100u, 80u}, std::pair{101u, 32u}}) {
    auto candidate = config;
    CountingCudaDevice device;
    device.state->total_memory_bytes = 200;
    device.state->free_memory_bytes = free_bytes;
    ApplyRuntimeProfileForSelectedDevice(candidate, device);
    EXPECT_EQ(*candidate.engine.dynamic_batching->num_blocks, expected_blocks) << free_bytes;
    EXPECT_EQ(device.state->memory_queries, 1u);
    EXPECT_EQ(device.state->integrated_queries, 0u);
  }

  auto legacy_caller = config;
  ApplyRuntimeProfile(legacy_caller, 200);
  EXPECT_EQ(*legacy_caller.engine.dynamic_batching->num_blocks, 32u);
}

TEST(ConfigTest, FreeMemoryEligibilityRespectsPhysicalMemoryDomain) {
  Config config;
  OverlayConfig(config, R"({
    "runtime_profiles":[
      {"id":"first","eligibility":{"minimum_total_device_memory_bytes":100,
        "maximum_total_device_memory_bytes":150,"minimum_free_device_memory_bytes":140,
        "maximum_free_device_memory_bytes":200},"overlay":{"search":{"chunk_size":1}}},
      {"id":"second","eligibility":{"minimum_total_device_memory_bytes":100,
        "maximum_total_device_memory_bytes":200,"minimum_free_device_memory_bytes":160,
        "maximum_free_device_memory_bytes":200},"overlay":{"search":{"chunk_size":2}}}
    ]
  })");
  auto second = config;
  ApplyRuntimeProfile(config, RuntimeProfileDeviceFacts{150, false, 140});
  EXPECT_EQ(config.search.chunk_size, 1u);
  ApplyRuntimeProfile(second, RuntimeProfileDeviceFacts{160, false, 160});
  EXPECT_EQ(second.search.chunk_size, 2u);

  for (const auto* other_free : {"80", "81"}) {
    Config overlapping;
    const auto json = std::string{R"({"runtime_profiles":[
      {"id":"first","eligibility":{"minimum_total_device_memory_bytes":100,
        "maximum_free_device_memory_bytes":80},"overlay":{"search":{"chunk_size":1}}},
      {"id":"second","eligibility":{"minimum_total_device_memory_bytes":100,
        "minimum_free_device_memory_bytes":)"} +
                      other_free + R"(},"overlay":{"search":{"chunk_size":2}}}
    ]})";
    if (std::string_view{other_free} == "80") {
      EXPECT_THROW(OverlayConfig(overlapping, json), std::runtime_error);
    } else {
      EXPECT_NO_THROW(OverlayConfig(overlapping, json));
    }
  }
}

TEST(ConfigTest, FreeMemoryEligibilityRejectsInvalidAndUnreachableRanges) {
  for (const auto* field : {"minimum_free_device_memory_bytes", "maximum_free_device_memory_bytes"}) {
    for (const auto* value : {"-1", "1.5", "true", "null", "\"100\"", "9007199254740992"}) {
      Config config;
      const auto json = std::string{R"({"runtime_profiles":[{
        "id":"invalid","eligibility":{"minimum_total_device_memory_bytes":0,")"} +
                        field + R"(":)" + value + R"(},"overlay":{"search":{"chunk_size":1}}}]})";
      EXPECT_THROW(OverlayConfig(config, json), std::runtime_error) << field << ": " << value;
    }
  }

  for (const auto* eligibility : {
           R"("minimum_free_device_memory_bytes":81,"maximum_free_device_memory_bytes":80)",
           R"("maximum_total_device_memory_bytes":100,"minimum_free_device_memory_bytes":101)"}) {
    Config config;
    const auto json = std::string{R"({"runtime_profiles":[{
      "id":"invalid","eligibility":{"minimum_total_device_memory_bytes":0,)"} +
                      eligibility + R"(},"overlay":{"search":{"chunk_size":1}}}]})";
    EXPECT_THROW(OverlayConfig(config, json), std::runtime_error) << eligibility;
  }

  Config zero_free;
  OverlayConfig(zero_free, R"({"runtime_profiles":[{
    "id":"zero","eligibility":{"minimum_total_device_memory_bytes":100,
      "maximum_free_device_memory_bytes":0},"overlay":{"search":{"chunk_size":1}}}]})");
  ApplyRuntimeProfile(zero_free, RuntimeProfileDeviceFacts{100, false, 0});
  EXPECT_EQ(zero_free.search.chunk_size, 1u);
}

TEST(ConfigTest, FreeMemoryAndIntegratedConditionsMatchTogether) {
  Config config;
  OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{"num_blocks":32}},
    "runtime_profiles":[{
      "id":"shared-with-headroom","eligibility":{
        "minimum_total_device_memory_bytes":100,"maximum_total_device_memory_bytes":200,
        "minimum_free_device_memory_bytes":40,"maximum_free_device_memory_bytes":80,
        "is_integrated":true},
      "overlay":{"engine":{"dynamic_batching":{"num_blocks":64}}}
    }]
  })");

  for (const auto [integrated, free_bytes, expected_blocks] : {
           std::tuple{true, 39u, 32u}, std::tuple{true, 40u, 64u},
           std::tuple{true, 80u, 64u}, std::tuple{true, 81u, 32u},
           std::tuple{false, 40u, 32u}}) {
    auto candidate = config;
    CountingCudaDevice device;
    device.state->total_memory_bytes = 200;
    device.state->free_memory_bytes = free_bytes;
    device.state->is_integrated = integrated;
    ApplyRuntimeProfileForSelectedDevice(candidate, device);
    EXPECT_EQ(*candidate.engine.dynamic_batching->num_blocks, expected_blocks);
    EXPECT_EQ(device.state->integrated_queries, 1u);
  }
}

TEST(ConfigTest, RuntimeProfileLogsSelectedAndBaseWithObservedFacts) {
  Config config;
  OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{"num_blocks":32}},
    "runtime_profiles":[{
      "id":"headroom","eligibility":{"minimum_total_device_memory_bytes":100,
        "minimum_free_device_memory_bytes":40},
      "overlay":{"engine":{"dynamic_batching":{"num_blocks":64}}}
    }]
  })");
  const auto previous_logging = g_log.enabled;
  SetLogBool("enabled", true);

  auto selected = config;
  testing::internal::CaptureStderr();
  ApplyRuntimeProfile(selected, RuntimeProfileDeviceFacts{100, false, 40});
  const auto selected_log = testing::internal::GetCapturedStderr();
  EXPECT_NE(selected_log.find("Runtime profile: headroom"), std::string::npos);
  EXPECT_NE(selected_log.find("total_device_memory_bytes=100"), std::string::npos);
  EXPECT_NE(selected_log.find("free_device_memory_bytes=40"), std::string::npos);
  EXPECT_NE(selected_log.find("is_integrated=false"), std::string::npos);
  EXPECT_EQ(selected_log.find("Runtime profile:"), selected_log.rfind("Runtime profile:"));

  testing::internal::CaptureStderr();
  ApplyRuntimeProfile(config, 100);
  const auto base_log = testing::internal::GetCapturedStderr();
  SetLogBool("enabled", previous_logging);
  EXPECT_NE(base_log.find("Runtime profile: base config"), std::string::npos);
  EXPECT_NE(base_log.find("total_device_memory_bytes=100"), std::string::npos);
  EXPECT_NE(base_log.find("free_device_memory_bytes=unknown"), std::string::npos);
  EXPECT_NE(base_log.find("is_integrated=unknown"), std::string::npos);
  EXPECT_EQ(base_log.find("Runtime profile:"), base_log.rfind("Runtime profile:"));
}

TEST(ConfigTest, RuntimeProfilesRejectAmbiguousCudaDeviceBeforeMemoryQuery) {
  for (const auto& selection : {"provider", "filter", "current"}) {
    SCOPED_TRACE(selection);
    Config config;
    OverlayConfig(config, R"({
      "model":{"decoder":{"session_options":{"provider_options":[{"cuda":{}}]}}},
      "runtime_profiles":[{
        "id":"gpu",
        "eligibility":{"minimum_total_device_memory_bytes":1},
        "overlay":{"search":{"chunk_size":512}}
      }]
    })");
    CountingCudaDevice device;
    auto& provider = config.model.decoder.session_options.provider_options.front();
    if (std::string{selection} == "provider") {
      provider.options.emplace_back("device_id", "1");
    } else if (std::string{selection} == "filter") {
      provider.device_filtering_options.emplace().hardware_device_id = 0;
    } else {
      device.state->device_id = 1;
    }

    EXPECT_THROW(ApplyRuntimeProfileForSelectedDevice(config, device), std::runtime_error);
    EXPECT_EQ(device.state->memory_queries, 0u);
    EXPECT_FALSE(config.search.chunk_size.has_value());
    EXPECT_FALSE(config.runtime_profiles.empty());
  }
}

TEST(ConfigTest, NoRuntimeProfilesSkipDeviceMemoryQuery) {
  Config config;
  CountingCudaDevice device;
  device.state->device_id = 1;

  ApplyRuntimeProfileForSelectedDevice(config, device);

  EXPECT_EQ(device.state->memory_queries, 0u);
  EXPECT_EQ(device.state->device_id_queries, 0u);
}

TEST(ConfigTest, RuntimeProfilePropagatesDeviceMemoryQueryFailure) {
  Config config;
  OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{"num_blocks":32}},
    "runtime_profiles":[{
      "id":"larger-gpu",
      "eligibility":{"minimum_total_device_memory_bytes":100},
      "overlay":{"engine":{"dynamic_batching":{"num_blocks":64}}}
    }]
  })");
  CountingCudaDevice device;
  device.state->fail_memory_query = true;

  try {
    ApplyRuntimeProfileForSelectedDevice(config, device);
    FAIL() << "Expected device memory query failure to propagate";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string(error.what()).find("device memory query failed"),
              std::string::npos)
        << error.what();
  }

  EXPECT_EQ(device.state->memory_queries, 1u);
  EXPECT_EQ(*config.engine.dynamic_batching->num_blocks, 32u);
  EXPECT_FALSE(config.runtime_profiles.empty());
}

TEST(ConfigTest, PrimaryProviderSelectionKeepsFirstDevice) {
  CountingCudaDevice dml_device{DeviceType::DML};
  CountingCudaDevice cuda_device;

  auto* selected = SelectPrimarySessionDevice(nullptr, &dml_device);
  selected = SelectPrimarySessionDevice(selected, &cuda_device);

  EXPECT_EQ(selected, &dml_device);
}

TEST(ConfigTest, PrimaryProviderSelectionKeepsCudaWhenFirst) {
  CountingCudaDevice cuda_device;
  CountingCudaDevice dml_device{DeviceType::DML};

  auto* selected = SelectPrimarySessionDevice(nullptr, &cuda_device);
  selected = SelectPrimarySessionDevice(selected, &dml_device);

  EXPECT_EQ(selected, &cuda_device);
}

TEST(ConfigTest, RuntimeProfilePreservesOmittedSearchFields) {
  Config config;
  config.search.chunk_size = 128;
  config.search.max_length = 4096;
  OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{"num_blocks":32}},
    "runtime_profiles":[{
      "id":"engine-only",
      "eligibility":{"minimum_total_device_memory_bytes":1},
      "overlay":{"engine":{"dynamic_batching":{"num_blocks":64}}}
    }]
  })");

  ApplyRuntimeProfile(config, 1);

  EXPECT_EQ(config.search.chunk_size, 128u);
  EXPECT_EQ(config.search.max_length, 4096);
}

TEST(ConfigTest, AdjacentInclusiveRangesSelectAtTheirBoundaries) {
  Config config;
  config.engine.dynamic_batching = Config::Engine::DynamicBatching{};
  OverlayConfig(config, R"({"runtime_profiles":[
    {"id":"lower","eligibility":{"minimum_total_device_memory_bytes":1,
                                    "maximum_total_device_memory_bytes":10},
     "overlay":{"engine":{"dynamic_batching":{"num_blocks":10}}}},
    {"id":"upper","eligibility":{"minimum_total_device_memory_bytes":11},
     "overlay":{"engine":{"dynamic_batching":{"num_blocks":20}}}}
  ]})");

  auto lower = config;
  ApplyRuntimeProfile(lower, 10);
  EXPECT_EQ(*lower.engine.dynamic_batching->num_blocks, 10u);

  auto upper = config;
  ApplyRuntimeProfile(upper, 11);
  EXPECT_EQ(*upper.engine.dynamic_batching->num_blocks, 20u);
}

TEST(ConfigTest, RejectsDuplicateRuntimeProfileIds) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[
    {"id":"same","eligibility":{"minimum_total_device_memory_bytes":1},
     "overlay":{"engine":{"dynamic_batching":{"num_blocks":1}}}},
    {"id":"same","eligibility":{"minimum_total_device_memory_bytes":2},
     "overlay":{"engine":{"dynamic_batching":{"num_blocks":2}}}}
  ]})"),
               std::runtime_error);
}

TEST(ConfigTest, RejectsInvalidRuntimeProfileRange) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"invalid",
    "eligibility":{"minimum_total_device_memory_bytes":2,
                   "maximum_total_device_memory_bytes":1},
    "overlay":{"engine":{"dynamic_batching":{"num_blocks":1}}}
  }]})"),
               std::runtime_error);
}

TEST(ConfigTest, RejectsMultipleMatchingRuntimeProfiles) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({
    "engine":{"dynamic_batching":{"num_blocks":32}},
    "runtime_profiles":[
      {"id":"first","eligibility":{"minimum_total_device_memory_bytes":1,
                                      "maximum_total_device_memory_bytes":10},
      "overlay":{"engine":{"dynamic_batching":{"num_blocks":64}}}},
      {"id":"second","eligibility":{"minimum_total_device_memory_bytes":5,
                                       "maximum_total_device_memory_bytes":15},
       "overlay":{"engine":{"dynamic_batching":{"num_blocks":96}}}}
    ]
  })"),
               std::runtime_error);
}

TEST(ConfigTest, RejectsUnapprovedSearchOverlayField) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"unapproved-search",
    "eligibility":{"minimum_total_device_memory_bytes":1},
    "overlay":{"search":{"temperature":0.5}}
  }]})"),
               std::runtime_error);
}

TEST(ConfigTest, RejectsUnapprovedModelOverlayField) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"context-length",
    "eligibility":{"minimum_total_device_memory_bytes":1},
    "overlay":{"model":{"context_length":8192}}
  }]})"),
               std::runtime_error);
}

TEST(ConfigTest, RejectsEmptyRuntimeProfileDecoderFilename) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"empty-filename",
    "eligibility":{"minimum_total_device_memory_bytes":1},
    "overlay":{"model":{"decoder":{"filename":""}}}
  }]})"),
               std::runtime_error);
}

TEST(ConfigTest, RejectsRuntimeProfileDecoderFilenameOutsideModelDirectory) {
  for (const char* filename : {"../model.onnx", "variants/../../model.onnx", "/tmp/model.onnx"}) {
    Config config;
    const std::string overlay = std::string{R"({"runtime_profiles":[{
      "id":"escape",
      "eligibility":{"minimum_total_device_memory_bytes":1},
      "overlay":{"model":{"decoder":{"filename":")"} +
                                filename + R"("}}}
    }]})";
    EXPECT_THROW(OverlayConfig(config, overlay), std::runtime_error) << filename;
  }
}

TEST(ConfigTest, RejectsRuntimeProfileMaxDraftTokensAboveSpeculativeLimit) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"too-many-drafts",
    "eligibility":{"minimum_total_device_memory_bytes":1},
    "overlay":{"speculative":{"max_draft_tokens":17}}
  }]})"),
               std::runtime_error);
}

TEST(ConfigTest, AppliesChunkSizeOnlyRuntimeProfileWithoutDynamicBatching) {
  Config config;
  OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"chunk-size-only",
    "eligibility":{"minimum_total_device_memory_bytes":1},
    "overlay":{"search":{"chunk_size":256}}
  }]})");

  ApplyRuntimeProfile(config, 1);

  EXPECT_EQ(config.search.chunk_size, 256u);
}

TEST(ConfigTest, RejectsRuntimeProfileMaxLengthOverride) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"request-limit",
    "eligibility":{"minimum_total_device_memory_bytes":1},
    "overlay":{"search":{"max_length":4096}}
  }]})"),
               std::runtime_error);
}

TEST(ConfigTest, RejectsRuntimeProfileContextLengthChange) {
  Config config;
  config.model.context_length = 4096;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"context-change",
    "eligibility":{"minimum_total_device_memory_bytes":1},
    "overlay":{"model":{"context_length":8192}}
  }]})"),
               std::runtime_error);
}

TEST(ConfigTest, RejectsRecursiveRuntimeProfilesOverlay) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"recursive",
    "eligibility":{"minimum_total_device_memory_bytes":1},
    "overlay":{"runtime_profiles":[]}
  }]})"),
               std::runtime_error);
}

TEST(ConfigTest, ParsesPagedScaleBindings) {
  Config config;
  EXPECT_TRUE(config.model.decoder.inputs.past_key_scale_names.empty());
  EXPECT_TRUE(config.model.dflash2.inputs.past_key_scale_names.empty());
  OverlayConfig(config, R"({"model":{
    "decoder":{"inputs":{"past_key_scale_names":"past.%d.ks","past_value_scale_names":"past.%d.vs"},
               "outputs":{"present_key_scale_names":"present.%d.ks","present_value_scale_names":"present.%d.vs"}},
    "dflash2":{"inputs":{"past_key_scale_names":"draft.%d.ks","past_value_scale_names":"draft.%d.vs"},
               "outputs":{"present_key_scale_names":"out.%d.ks","present_value_scale_names":"out.%d.vs"}}
  }})");
  EXPECT_EQ(config.model.decoder.inputs.past_key_scale_names, "past.%d.ks");
  EXPECT_EQ(config.model.decoder.inputs.past_value_scale_names, "past.%d.vs");
  EXPECT_EQ(config.model.decoder.outputs.present_key_scale_names, "present.%d.ks");
  EXPECT_EQ(config.model.decoder.outputs.present_value_scale_names, "present.%d.vs");
  EXPECT_EQ(config.model.dflash2.inputs.past_key_scale_names, "draft.%d.ks");
  EXPECT_EQ(config.model.dflash2.inputs.past_value_scale_names, "draft.%d.vs");
  EXPECT_EQ(config.model.dflash2.outputs.present_key_scale_names, "out.%d.ks");
  EXPECT_EQ(config.model.dflash2.outputs.present_value_scale_names, "out.%d.vs");
}

// The MTP head is always an unquantized full-attention layer, so it has no scale-name configuration
// surface at all. A config that tries to declare one is rejected rather than silently ignored.
TEST(ConfigTest, RejectsMtpScaleBindings) {
  for (const char* json : {R"({"model":{"mtp":{"inputs":{"past_key_scale_names":"mtp.%d.ks"}}}})",
                           R"({"model":{"mtp":{"inputs":{"past_value_scale_names":"mtp.%d.vs"}}}})",
                           R"({"model":{"mtp":{"outputs":{"present_key_scale_names":"mtp.%d.ks"}}}})",
                           R"({"model":{"mtp":{"outputs":{"present_value_scale_names":"mtp.%d.vs"}}}})"}) {
    Config config;
    EXPECT_THROW(OverlayConfig(config, json), std::runtime_error) << json;
  }
}

template <typename PipelineModels>
void ExpectPipelineOverlayMerged(const PipelineModels& pipeline) {
  ASSERT_EQ(pipeline.size(), 3u);

  const auto& untouched_model = pipeline[0];
  EXPECT_EQ(untouched_model.model_id, "untouched_model");
  EXPECT_EQ(untouched_model.filename, "untouched.onnx");
  ASSERT_TRUE(untouched_model.session_options);
  ASSERT_TRUE(untouched_model.session_options->log_id);
  EXPECT_EQ(*untouched_model.session_options->log_id, "untouched");
  EXPECT_FALSE(untouched_model.session_options->enable_profiling);

  const auto& existing_model = pipeline[1];
  EXPECT_EQ(existing_model.model_id, "existing_model");
  EXPECT_EQ(existing_model.filename, "existing.onnx");
  ASSERT_TRUE(existing_model.session_options);
  ASSERT_TRUE(existing_model.session_options->log_id);
  EXPECT_EQ(*existing_model.session_options->log_id, "original");
  ASSERT_TRUE(existing_model.session_options->enable_profiling);
  EXPECT_EQ(*existing_model.session_options->enable_profiling, "profile");
  ASSERT_EQ(existing_model.session_options->provider_options.size(), 1u);
  EXPECT_EQ(existing_model.session_options->provider_options[0].name, "CPU");

  const auto& new_model = pipeline[2];
  EXPECT_EQ(new_model.model_id, "new_model");
  EXPECT_EQ(new_model.filename, "new.onnx");
}

TEST(ConfigTest, PipelineOverlayMergesByModelId) {
  Config config;
  OverlayConfig(config, R"({
    "model": {
      "decoder": {
        "pipeline": [{
          "untouched_model": {
            "filename": "untouched.onnx",
            "session_options": {"log_id": "untouched"}
          },
          "existing_model": {
            "filename": "existing.onnx",
            "session_options": {
              "log_id": "original",
              "provider_options": [{"CPU": {}}]
            }
          }
        }]
      },
      "vision": {
        "pipeline": [{
          "untouched_model": {
            "filename": "untouched.onnx",
            "session_options": {"log_id": "untouched"}
          },
          "existing_model": {
            "filename": "existing.onnx",
            "session_options": {
              "log_id": "original",
              "provider_options": [{"CPU": {}}]
            }
          }
        }]
      }
    }
  })");

  OverlayConfig(config, R"({
    "model": {
      "decoder": {
        "pipeline": [{
          "existing_model": {
            "session_options": {"enable_profiling": "profile"}
          },
          "new_model": {"filename": "new.onnx"}
        }]
      },
      "vision": {
        "pipeline": [{
          "existing_model": {
            "session_options": {"enable_profiling": "profile"}
          },
          "new_model": {"filename": "new.onnx"}
        }]
      }
    }
  })");

  ExpectPipelineOverlayMerged(config.model.decoder.pipeline);
  ExpectPipelineOverlayMerged(config.model.vision.pipeline);
}

}  // namespace Generators::test
