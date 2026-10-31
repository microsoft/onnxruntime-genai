// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "config.h"
#include "engine_test_doubles.h"
#include "models/runtime_profiles.h"
#include "models/session_options.h"
#include "scoped-environment-variable.h"

namespace Generators::test {

TEST(ConfigTest, ParsesDeterministicCompute) {
  Config config;
  EXPECT_FALSE(config.model.decoder.session_options.use_deterministic_compute.has_value());
  OverlayConfig(config, R"({"model":{"decoder":{"session_options":{"use_deterministic_compute":true}}}})");
  EXPECT_EQ(config.model.decoder.session_options.use_deterministic_compute, true);
  EXPECT_TRUE(config.model.decoder.session_options.config_entries.empty());
  OverlayConfig(config, R"({"model":{"decoder":{"session_options":{"use_deterministic_compute":false}}}})");
  EXPECT_EQ(config.model.decoder.session_options.use_deterministic_compute, false);
  EXPECT_THROW(OverlayConfig(config,
                             R"({"model":{"decoder":{"session_options":{"use_deterministic_compute":"true"}}}})"),
               std::exception);
}

TEST(ConfigTest, ParsesTimestampConfiguration) {
  Config config;

  OverlayConfig(config, R"({"model":{
    "type":"nemotron_speech",
    "timestamp_level":"all",
    "sample_rate":16000,
    "hop_length":160,
    "subsampling_factor":8,
    "segment_separators":[".","!?"] ,
    "segment_gap_threshold_seconds":1.25
  }})");

  EXPECT_EQ(config.model.timestamp_level, Config::TimestampLevel::All);
  EXPECT_EQ(config.model.segment_separators, (std::vector<std::string>{".", "!?"}));
  EXPECT_EQ(config.model.segment_gap_threshold_seconds, 1.25);
}

TEST(ConfigTest, ParsesDisabledTimestampGap) {
  Config config;
  config.model.segment_gap_threshold_seconds = 1.0;

  OverlayConfig(config, R"({"model":{"segment_gap_threshold_seconds":null}})");

  EXPECT_FALSE(config.model.segment_gap_threshold_seconds.has_value());
}

TEST(ConfigTest, RejectsInvalidTimestampConfiguration) {
  for (const char* json : {
           R"({"model":{"timestamp_level":"token"}})",
           R"({"model":{"segment_gap_threshold_frames":12}})",
           R"({"model":{"segment_gap_threshold_seconds":-1}})"}) {
    Config config;
    EXPECT_THROW(OverlayConfig(config, json), std::runtime_error);
  }
}

TEST(ConfigTest, RejectsInvalidTimestampOverlayWithoutChangingConfig) {
  Config config;
  for (const auto& overlay : {
           R"({"model":{"type":"nemotron_speech","timestamp_level":"word"}})",
           R"({"model":{"type":"gpt2","timestamp_level":"word","sample_rate":100,"hop_length":10,"subsampling_factor":1}})",
           R"({"model":{"type":"nemotron_speech","timestamp_level":"segment","sample_rate":16000,"hop_length":160,"subsampling_factor":8,"segment_gap_threshold_seconds":1e20}})"}) {
    EXPECT_THROW(OverlayConfig(config, overlay), std::runtime_error);
    EXPECT_EQ(config.model.timestamp_level, Config::TimestampLevel::Off);
  }
}

TEST(ConfigTest, AcceptsZeroTimestampGap) {
  Config config;
  OverlayConfig(config, R"({"model":{"segment_gap_threshold_seconds":0}})");
  ASSERT_TRUE(config.model.segment_gap_threshold_seconds.has_value());
  EXPECT_EQ(GetSegmentGapThresholdFrames(config.model.segment_gap_threshold_seconds, 100, 10, 1), 0);
}

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

TEST(ConfigTest, ParsesRotaryHistoryAndIndexerCompressionMetadata) {
  Config config;
  OverlayConfig(config, R"({"model":{
    "decoder":{"hidden_size":16,"indexer_compress_ratio":8,"ple_token_pad_id":7,
               "inputs":{"rotary_position_ids":"main_rotary_history"}},
    "mtp":{"filename":"mtp.onnx","num_hidden_layers":1,"num_key_value_heads":1,"head_size":16,
           "inputs":{"rotary_position_ids":"head_rotary_history"}}
  }})");
  EXPECT_EQ(config.model.decoder.inputs.rotary_position_ids, "main_rotary_history");
  EXPECT_EQ(config.model.decoder.indexer_compress_ratio, 8);
  EXPECT_EQ(config.model.decoder.ple_token_pad_id, 7);
  auto head = CreateMtpDecoderConfig(config);
  EXPECT_EQ(head->model.decoder.inputs.rotary_position_ids, "head_rotary_history");
  EXPECT_EQ(head->model.decoder.indexer_compress_ratio, 8);
}

TEST(ConfigTest, RejectsInvalidPlePadIdsAndCompressionRatios) {
  for (const auto* value : {"1.5", "-1", "1e100"}) {
    Config config;
    const std::string overlay = std::string{R"({"model":{"decoder":{"ple_token_pad_id":)"} + value + "}}}";
    EXPECT_THROW(OverlayConfig(config, overlay), std::exception);
  }
  for (const auto* value : {"0", "-1", "1.5", "1e100"}) {
    Config config;
    const std::string overlay = std::string{R"({"model":{"decoder":{"indexer_compress_ratio":)"} + value + "}}}";
    EXPECT_THROW(OverlayConfig(config, overlay), std::exception);
  }
}

TEST(ConfigTest, EngramRequiresExplicitPaddingAndRetainsItsOwnProviders) {
  const auto root = fs::path{MODEL_PATH "engine/dummy-decoder"};
  EXPECT_THROW((Config{root, R"({"model":{"engram":{"filename":"engram.onnx"}}})"}), std::runtime_error);
  EXPECT_THROW((Config{root, R"({"model":{"decoder":{"inputs":{"past_ple_token_names":"past.%d.ple_tokens"}}}})"}),
               std::runtime_error);
  Config config{root, R"({"model":{
    "decoder":{"ple_token_pad_id":7},
    "engram":{"filename":"engram.onnx","session_options":{"provider_options":[{"cuda":{}}]}}
  }})"};
  ASSERT_TRUE(config.model.engram.session_options);
  EXPECT_EQ(config.model.engram.session_options->providers, (std::vector<std::string>{"cuda"}));
}

TEST(ConfigTest, EngramProviderOverridesAreAppliedAndUnknownProvidersFailExplicitly) {
  Config config;
  OverlayConfig(config, R"({"model":{"engram":{"session_options":{"provider_options":[{"cpu":{}}]}}}})");
  EXPECT_EQ(config.model.engram.session_options->providers, (std::vector<std::string>{"CPU"}));
  OverlayConfig(config, R"({"search":{"max_length":8}})");
  EXPECT_EQ(config.model.engram.session_options->providers, (std::vector<std::string>{"CPU"}));
  auto unknown = std::make_unique<Config>(
      fs::path{MODEL_PATH "engine/dummy-decoder"}, R"({"model":{
        "type":"qwen4_exp_text",
        "decoder":{"ple_token_pad_id":7},
        "engram":{"filename":"engram.onnx","session_options":{"provider_options":[{"nonexistent_provider":{}}]}}
      }})");
  try {
    CreateModel(GetOrtEnv(), std::move(unknown));
    FAIL() << "Unknown Engram provider must not silently fall back to CPU";
  } catch (const std::exception& error) {
    EXPECT_NE(std::string{error.what()}.find("nonexistent_provider"), std::string::npos) << error.what();
  }
}

TEST(ConfigTest, ParsesEngramPleTokenNames) {
  Config config;
  OverlayConfig(config, R"({"model":{"engram":{
    "inputs":{"past_ple_token_names":"past.1.ple_tokens"},
    "outputs":{"present_ple_token_names":"present.1.ple_tokens"}
  }}})");

  EXPECT_EQ(config.model.engram.inputs.past_tokens, "past.1.ple_tokens");
  EXPECT_EQ(config.model.engram.outputs.present_tokens, "present.1.ple_tokens");
}

TEST(ConfigTest, ParsesLegacyEngramTokenNames) {
  Config config;
  OverlayConfig(config, R"({"model":{"engram":{
    "inputs":{"past_tokens":"legacy_past"},
    "outputs":{"present_tokens":"legacy_present"}
  }}})");

  EXPECT_EQ(config.model.engram.inputs.past_tokens, "legacy_past");
  EXPECT_EQ(config.model.engram.outputs.present_tokens, "legacy_present");
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

namespace {

constexpr const char* kForcedRuntimeProfileConfig = R"({
  "engine":{"dynamic_batching":{"num_blocks":32}},
  "runtime_profiles":[
    {"id":"consumer-gpu-24g",
     "eligibility":{"minimum_total_device_memory_bytes":0,
                    "maximum_total_device_memory_bytes":25769803776},
     "overlay":{"engine":{"dynamic_batching":{"num_blocks":24}}}},
    {"id":"datacenter-gpu",
     "eligibility":{"minimum_total_device_memory_bytes":85899345920},
     "overlay":{"engine":{"dynamic_batching":{"num_blocks":128}}}}
  ]
})";

constexpr uint64_t kDatacenterTotalDeviceMemoryBytes = 150136356864ULL;

}  // namespace

TEST(ConfigTest, RuntimeProfileEnvOverrideSelectsProfileRegardlessOfEligibility) {
  Config config;
  OverlayConfig(config, kForcedRuntimeProfileConfig);

  ScopedEnvironmentVariable forced{"ORTGENAI_ENABLE_RUNTIME_PROFILE", "consumer-gpu-24g"};
  ApplyRuntimeProfile(config, kDatacenterTotalDeviceMemoryBytes);

  ASSERT_TRUE(config.engine.dynamic_batching);
  EXPECT_EQ(*config.engine.dynamic_batching->num_blocks, 24u);
}

TEST(ConfigTest, RuntimeProfileEnvOverrideRejectsUnknownProfileId) {
  Config config;
  OverlayConfig(config, kForcedRuntimeProfileConfig);

  ScopedEnvironmentVariable forced{"ORTGENAI_ENABLE_RUNTIME_PROFILE", "consumer-gpu-32g"};
  try {
    ApplyRuntimeProfile(config, kDatacenterTotalDeviceMemoryBytes);
    FAIL() << "An unknown forced profile id must not fall back to device eligibility";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("consumer-gpu-32g"), std::string::npos) << error.what();
  }
  EXPECT_EQ(*config.engine.dynamic_batching->num_blocks, 32u);
}

TEST(ConfigTest, RuntimeProfileEnvOverrideIgnoredWhenEmptyOrUnset) {
  for (const std::optional<std::string>& value : {std::optional<std::string>{}, std::optional<std::string>{""}}) {
    Config config;
    OverlayConfig(config, kForcedRuntimeProfileConfig);

    ScopedEnvironmentVariable forced{"ORTGENAI_ENABLE_RUNTIME_PROFILE", value};
    ApplyRuntimeProfile(config, kDatacenterTotalDeviceMemoryBytes);

    EXPECT_EQ(*config.engine.dynamic_batching->num_blocks, 128u);
  }
}

TEST(ConfigTest, Dflash2SnapshotsDefaultToOneAndAllowZeroOrLargerPools) {
  Config defaults;
  OverlayConfig(defaults, R"({"engine":{"dynamic_batching":{"num_blocks":32}}})");
  ASSERT_TRUE(defaults.engine.dynamic_batching);
  EXPECT_EQ(defaults.engine.dynamic_batching->dflash2_max_snapshots, 1u);

  for (const size_t capacity : {size_t{0}, size_t{1}, size_t{3}, size_t{2147483647}}) {
    Config config;
    OverlayConfig(config, R"({"engine":{"dynamic_batching":{"dflash2_max_snapshots":)" +
                              std::to_string(capacity) + "}}}");
    EXPECT_EQ(config.engine.dynamic_batching->dflash2_max_snapshots, capacity);
    OverlayConfig(config, R"({"runtime_profiles":[{
      "id":"snapshot-pool",
      "eligibility":{"minimum_total_device_memory_bytes":100},
      "overlay":{"engine":{"dynamic_batching":{"dflash2_max_snapshots":)" +
                              std::to_string(capacity == 0 ? 3 : 0) + "}}}}]}");
    ApplyRuntimeProfile(config, 99);
    EXPECT_EQ(config.engine.dynamic_batching->dflash2_max_snapshots, capacity);
    ApplyRuntimeProfile(config, 100);
    EXPECT_EQ(config.engine.dynamic_batching->dflash2_max_snapshots, capacity == 0 ? 3u : 0u);
  }
}

TEST(ConfigTest, Dflash2SnapshotLimitRejectsInvalidValuesTransactionally) {
  for (const auto* value : {"-1", "1.5", "2147483648", "true", "\"3\"", "null"}) {
    Config config;
    OverlayConfig(config, R"({"engine":{"dynamic_batching":{"dflash2_max_snapshots":3}}})");
    EXPECT_THROW(OverlayConfig(config, R"({"engine":{"dynamic_batching":{"dflash2_max_snapshots":)" +
                                           std::string{value} + "}}}"),
                 std::runtime_error);
    EXPECT_EQ(config.engine.dynamic_batching->dflash2_max_snapshots, 3u);
    EXPECT_THROW(OverlayConfig(config, R"({"runtime_profiles":[{
      "id":"invalid",
      "eligibility":{"minimum_total_device_memory_bytes":0},
      "overlay":{"engine":{"dynamic_batching":{"dflash2_max_snapshots":)" +
                                           std::string{value} + "}}}}]}"),
                 std::runtime_error);
    EXPECT_TRUE(config.runtime_profiles.empty());
  }
}

TEST(ConfigTest, Dflash2SnapshotOnlyProfileRequiresDynamicBatchingEvenWhenDisabled) {
  Config config;
  OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"disabled",
    "eligibility":{"minimum_total_device_memory_bytes":0},
    "overlay":{"engine":{"dynamic_batching":{"dflash2_max_snapshots":0}}}
  }]})");
  EXPECT_THROW(ApplyRuntimeProfile(config, 0), std::runtime_error);
  EXPECT_FALSE(config.engine.dynamic_batching);
}

TEST(ConfigTest, RuntimeProfileAppliesDecoderSessionOptionsAndHostMemoryEligibility) {
  Config config;
  OverlayConfig(config, R"({
    "model": {
      "decoder": {
        "filename": "model.onnx",
        "session_options": {
          "session.moe.offload.mode": "auto",
          "session.moe.offload.device_expert_bytes": "auto"
        }
      }
    },
    "runtime_profiles": [
      {
        "id": "consumer-gpu-low-ram",
        "eligibility": {
          "minimum_total_device_memory_bytes": 0,
          "maximum_total_device_memory_bytes": 25769803776,
          "maximum_effective_host_memory_bytes": 68719476735
        },
        "overlay": {
          "model": {
            "decoder": {
              "session_options": {
                "session.moe.offload.mode": "required",
                "session.moe.offload.device_expert_bytes": "10351636480",
                "session.moe.offload.host_expert_bytes": "8589934592"
              }
            }
          }
        }
      },
      {
        "id": "consumer-gpu-more-ram",
        "eligibility": {
          "minimum_total_device_memory_bytes": 0,
          "maximum_total_device_memory_bytes": 25769803776,
          "minimum_effective_host_memory_bytes": 68719476736
        },
        "overlay": {
          "model": {
            "decoder": {
              "session_options": {
                "session.moe.offload.mode": "required",
                "session.moe.offload.device_expert_bytes": "10351636480",
                "session.moe.offload.host_expert_bytes": "42949672960"
              }
            }
          }
        }
      }
    ]
  })");

  ApplyRuntimeProfile(config, RuntimeProfileDeviceFacts{
                                  25769803776ULL,
                                  false,
                                  20000000000ULL,
                                  68719476736ULL,
                                  50000000000ULL,
                              });

  std::map<std::string, std::string> entries;
  for (const auto& [k, v] : config.model.decoder.session_options.config_entries) {
    entries[k] = v;
  }
  EXPECT_EQ(entries["session.moe.offload.mode"], "required");
  EXPECT_EQ(entries["session.moe.offload.device_expert_bytes"], "10351636480");
  EXPECT_EQ(entries["session.moe.offload.host_expert_bytes"], "42949672960");
}

TEST(ConfigTest, RuntimeProfileUpdatesEveryDuplicateOfASessionConfigEntry) {
  // Every overlay appends its decoder session config entries, so one key can be present more than once.
  Config config;
  OverlayConfig(config, R"({"model":{"decoder":{"session_options":{"session.moe.offload.mode":"base"}}}})");
  OverlayConfig(config, R"({"model":{"decoder":{"session_options":{"session.moe.offload.mode":"overlay"}}}})");
  ASSERT_EQ(config.model.decoder.session_options.config_entries.size(), 2u);
  OverlayConfig(config, R"({"runtime_profiles":[{"id":"p","eligibility":{"minimum_total_device_memory_bytes":0},
    "overlay":{"model":{"decoder":{"session_options":{
      "session.moe.offload.mode":"required","session.moe.offload.host_expert_bytes":"4096"}}}}}]})");

  ApplyRuntimeProfile(config, 100);

  // The entries are applied in order, so an entry that kept an older value would override the profile's.
  const auto& entries = config.model.decoder.session_options.config_entries;
  ASSERT_EQ(entries.size(), 3u);
  for (const auto& [key, value] : entries) {
    EXPECT_EQ(value, key == "session.moe.offload.mode" ? "required" : "4096") << key;
  }
}

TEST(ConfigTest, RuntimeProfileSessionOptionsRejectTypedSessionOptions) {
  // A profile overlays ORT session config entries. A typed option such as graph_optimization_level would be passed to
  // ORT as a raw config entry, which ignores it, so it is rejected instead of silently having no effect.
  for (const char* name : {"log_id", "enable_profiling", "intra_op_num_threads", "inter_op_num_threads",
                           "log_severity_level", "log_verbosity_level", "enable_cpu_mem_arena", "enable_mem_pattern",
                           "use_deterministic_compute", "graph_optimization_level", "custom_ops_library",
                           "custom_ops_libraries"}) {
    Config config;
    const std::string json = std::string{R"({"runtime_profiles":[{"id":"p",
      "eligibility":{"minimum_total_device_memory_bytes":0},
      "overlay":{"model":{"decoder":{"session_options":{")"} +
                             name + R"(":"1"}}}}}]})";
    try {
      OverlayConfig(config, json);
      ADD_FAILURE() << name << " was accepted";
    } catch (const std::runtime_error& error) {
      EXPECT_NE(std::string{error.what()}.find(name), std::string::npos) << error.what();
    }
  }
}

TEST(ConfigTest, AvailableHostMemoryEligibilityNeedsAKnownValueAtOrAboveTheMinimum) {
  Config config;
  OverlayConfig(config, R"({"runtime_profiles":[{
    "id":"roomy","eligibility":{"minimum_total_device_memory_bytes":0,"minimum_available_host_memory_bytes":100},
    "overlay":{"search":{"chunk_size":1}}}]})");

  auto at_minimum = config;
  ApplyRuntimeProfile(at_minimum, RuntimeProfileDeviceFacts{1, false, std::nullopt, 200, 100});
  EXPECT_EQ(at_minimum.search.chunk_size, 1u);

  auto below_minimum = config;
  ApplyRuntimeProfile(below_minimum, RuntimeProfileDeviceFacts{1, false, std::nullopt, 200, 99});
  EXPECT_FALSE(below_minimum.search.chunk_size.has_value());

  auto unknown = config;
  ApplyRuntimeProfile(unknown, RuntimeProfileDeviceFacts{1, false, std::nullopt, 200, std::nullopt});
  EXPECT_FALSE(unknown.search.chunk_size.has_value());
}

TEST(ConfigTest, AvailableHostMemoryMinimumIsCheckedAgainstTheOtherProfilesEffectiveHostMemoryMaximum) {
  // Available host memory never exceeds the effective host memory, so a profile that needs more available memory than
  // the other profile's effective maximum cannot match the same device.
  for (const uint64_t available_minimum : {100ULL, 101ULL}) {
    Config config;
    const auto json = std::string{R"({"runtime_profiles":[
      {"id":"small-host","eligibility":{"minimum_total_device_memory_bytes":0,
        "maximum_effective_host_memory_bytes":100},"overlay":{"search":{"chunk_size":1}}},
      {"id":"big-available","eligibility":{"minimum_total_device_memory_bytes":0,
        "minimum_available_host_memory_bytes":)"} +
                      std::to_string(available_minimum) + R"(},"overlay":{"search":{"chunk_size":2}}}
    ]})";
    if (available_minimum <= 100) {
      EXPECT_THROW(OverlayConfig(config, json), std::runtime_error) << available_minimum;
    } else {
      EXPECT_NO_THROW(OverlayConfig(config, json)) << available_minimum;
    }
  }
}

TEST(ConfigTest, HostMemoryBoundsAreValidatedWithinOneProfile) {
  const auto load = [](const std::string& eligibility) {
    Config config;
    OverlayConfig(config, R"({"runtime_profiles":[{"id":"p","eligibility":{"minimum_total_device_memory_bytes":0,)" +
                              eligibility + R"(},"overlay":{"search":{"chunk_size":1}}}]})");
  };

  // Like the device memory bounds, the effective host memory maximum cannot be below its minimum.
  EXPECT_NO_THROW(load(R"("minimum_effective_host_memory_bytes":100,"maximum_effective_host_memory_bytes":100)"));
  EXPECT_THROW(load(R"("minimum_effective_host_memory_bytes":101,"maximum_effective_host_memory_bytes":100)"),
               std::runtime_error);

  // The available host memory never exceeds the effective host memory, so a minimum above the effective maximum can
  // never be met.
  EXPECT_NO_THROW(load(R"("minimum_available_host_memory_bytes":100,"maximum_effective_host_memory_bytes":100)"));
  EXPECT_THROW(load(R"("minimum_available_host_memory_bytes":101,"maximum_effective_host_memory_bytes":100)"),
               std::runtime_error);
}

TEST(ConfigTest, ParsesMeminfoValues) {
  const std::string_view meminfo =
      "MemTotal:       65843260 kB\n"
      "MemFree:         1204512 kB\n"
      "MemAvailable:   48211376 kB\n"
      "HugePages_Total:       4\n"
      "Bogus:               abc kB\n"
      "Scale:                12 MB\n";
  EXPECT_EQ(ParseMeminfoBytes(meminfo, "MemAvailable"), std::optional<uint64_t>{48211376ULL * 1024});
  EXPECT_EQ(ParseMeminfoBytes(meminfo, "MemFree"), std::optional<uint64_t>{1204512ULL * 1024});
  EXPECT_EQ(ParseMeminfoBytes(meminfo, "HugePages_Total"), std::optional<uint64_t>{4});
  EXPECT_EQ(ParseMeminfoBytes("MemAvailable: 8 kB", "MemAvailable"), std::optional<uint64_t>{8192});

  EXPECT_FALSE(ParseMeminfoBytes(meminfo, "Mem").has_value());  // a key prefix is not a match
  EXPECT_FALSE(ParseMeminfoBytes(meminfo, "Missing").has_value());
  EXPECT_FALSE(ParseMeminfoBytes(meminfo, "Bogus").has_value());
  EXPECT_FALSE(ParseMeminfoBytes(meminfo, "Scale").has_value());
  EXPECT_FALSE(ParseMeminfoBytes("", "MemAvailable").has_value());
}

TEST(ConfigTest, ResolvesEffectiveAndAvailableHostMemoryAgainstTheCgroup) {
  constexpr uint64_t GiB = 1ULL << 30;
  std::optional<uint64_t> effective;
  std::optional<uint64_t> available;
  const auto resolve = [&](HostMemoryReadings readings) {
    ResolveHostMemoryFacts(readings, effective, available);
  };
  // The tightest limit and headroom over the process's cgroup and its ancestors.
  const auto cgroup = [](std::optional<uint64_t> limit, std::optional<uint64_t> headroom) {
    return CgroupMemoryLimits{CgroupMemoryLimits::Status::kResolved, limit, headroom};
  };

  // Without a cgroup the physical readings are used as they are.
  resolve({128 * GiB, 100 * GiB});
  EXPECT_EQ(effective, std::optional<uint64_t>{128 * GiB});
  EXPECT_EQ(available, std::optional<uint64_t>{100 * GiB});

  // The cgroup limit caps the effective memory, and the cgroup headroom caps the available memory.
  resolve({128 * GiB, 100 * GiB, cgroup(64 * GiB, 54 * GiB)});
  EXPECT_EQ(effective, std::optional<uint64_t>{64 * GiB});
  EXPECT_EQ(available, std::optional<uint64_t>{54 * GiB});

  // MemAvailable is the tighter bound when the cgroup has more headroom than the system.
  resolve({128 * GiB, 20 * GiB, cgroup(64 * GiB, 54 * GiB)});
  EXPECT_EQ(effective, std::optional<uint64_t>{64 * GiB});
  EXPECT_EQ(available, std::optional<uint64_t>{20 * GiB});

  // A cgroup that has used all of its limit leaves no headroom.
  resolve({128 * GiB, 100 * GiB, cgroup(64 * GiB, 0)});
  EXPECT_EQ(available, std::optional<uint64_t>{0});

  // A cgroup limit above the physical memory does not raise the effective memory.
  resolve({64 * GiB, 40 * GiB, cgroup(128 * GiB, 118 * GiB)});
  EXPECT_EQ(effective, std::optional<uint64_t>{64 * GiB});
  EXPECT_EQ(available, std::optional<uint64_t>{40 * GiB});

  // The cgroup headroom does not stand in for a MemAvailable that cannot be read: the host may have less memory free
  // than the cgroup has headroom.
  resolve({128 * GiB, std::nullopt, cgroup(64 * GiB, 54 * GiB)});
  EXPECT_EQ(effective, std::optional<uint64_t>{64 * GiB});
  EXPECT_FALSE(available.has_value());

  // The available memory never exceeds the effective memory.
  resolve({64 * GiB, 100 * GiB});
  EXPECT_EQ(available, std::optional<uint64_t>{64 * GiB});

  // An unreported availability stays unknown instead of being guessed.
  resolve({64 * GiB, std::nullopt});
  EXPECT_EQ(effective, std::optional<uint64_t>{64 * GiB});
  EXPECT_FALSE(available.has_value());

  // A zero cgroup limit is a hard limit of zero bytes, not an unlimited cgroup.
  resolve({64 * GiB, 40 * GiB, cgroup(0, 0)});
  EXPECT_EQ(effective, std::optional<uint64_t>{0});
  EXPECT_EQ(available, std::optional<uint64_t>{0});

  // A cgroup with no limit anywhere leaves the physical readings alone.
  resolve({64 * GiB, 40 * GiB, cgroup(std::nullopt, std::nullopt)});
  EXPECT_EQ(effective, std::optional<uint64_t>{64 * GiB});
  EXPECT_EQ(available, std::optional<uint64_t>{40 * GiB});

  // An unresolved cgroup leaves both facts unknown, because the physical readings would overstate what the process
  // may use.
  const CgroupMemoryLimits unresolved{CgroupMemoryLimits::Status::kUnresolved, std::nullopt, std::nullopt};
  resolve({128 * GiB, 100 * GiB, unresolved});
  EXPECT_FALSE(effective.has_value());
  EXPECT_FALSE(available.has_value());

  resolve({});
  EXPECT_FALSE(effective.has_value());
  EXPECT_FALSE(available.has_value());
}

namespace {

using CgroupStatus = CgroupMemoryLimits::Status;
using FakeHostFiles = std::map<std::string, std::string>;

constexpr uint64_t kGiB = 1ULL << 30;

// A host that mounts the whole cgroup2 hierarchy next to other filesystems.
constexpr std::string_view kHostMountinfo =
    "22 1 8:2 / / rw,relatime shared:1 - ext4 /dev/sda2 rw,errors=remount-ro\n"
    "23 22 0:21 / /proc rw,nosuid,nodev,noexec,relatime shared:12 - proc proc rw\n"
    "24 22 0:22 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:5 - cgroup2 cgroup2 "
    "rw,nsdelegate,memory_recursiveprot\n"
    "25 22 0:25 / /run rw,nosuid,nodev shared:6 - tmpfs tmpfs rw,mode=755\n";

CgroupMemoryLimits ResolveCgroupMemory(const FakeHostFiles& files) {
  return ResolveProcessCgroupMemory([&files](const std::string& path) -> std::optional<std::string> {
    const auto file = files.find(path);
    if (file == files.end()) return std::nullopt;
    return file->second;
  });
}

// The files every cgroup directory has. cgroup.controllers lists the controllers the cgroup has: a cgroup that has the
// memory controller also has memory.max and memory.current (see AddMemoryFiles), except for the hierarchy root.
void AddCgroup(FakeHostFiles& files, const std::string& directory, bool has_memory_controller = true) {
  files[directory + "/cgroup.controllers"] =
      has_memory_controller ? "cpuset cpu io memory pids\n" : "cpuset cpu io pids\n";
}

// The files of a cgroup that has the memory controller. `max` is "max" or a number of bytes.
void AddMemoryFiles(FakeHostFiles& files, const std::string& directory, const std::string& max, uint64_t current) {
  files[directory + "/memory.max"] = max + "\n";
  files[directory + "/memory.current"] = std::to_string(current) + "\n";
}

struct HostFacts {
  std::optional<uint64_t> effective;
  std::optional<uint64_t> available;
};

HostFacts ResolveHostFacts(const CgroupMemoryLimits& cgroup, uint64_t physical_total, uint64_t physical_available) {
  HostMemoryReadings readings;
  readings.physical_total = physical_total;
  readings.physical_available = physical_available;
  readings.cgroup = cgroup;
  HostFacts facts;
  ResolveHostMemoryFacts(readings, facts.effective, facts.available);
  return facts;
}

}  // namespace

TEST(ConfigTest, CgroupMemoryComesFromTheProcessCgroupNotTheMountRoot) {
  // A 128 GiB host whose mounted cgroup root has no limit, and a process in an 8 GiB cgroup that has used 2 GiB.
  FakeHostFiles files;
  const std::string scope = "/sys/fs/cgroup/user.slice/user-1000.slice/session-3.scope";
  files["/proc/self/cgroup"] = "0::/user.slice/user-1000.slice/session-3.scope\n";
  files["/proc/self/mountinfo"] = std::string{kHostMountinfo};
  AddCgroup(files, "/sys/fs/cgroup");
  AddMemoryFiles(files, "/sys/fs/cgroup", "max", 5 * kGiB);
  AddCgroup(files, "/sys/fs/cgroup/user.slice");
  AddMemoryFiles(files, "/sys/fs/cgroup/user.slice", "max", 4 * kGiB);
  AddCgroup(files, "/sys/fs/cgroup/user.slice/user-1000.slice");
  AddMemoryFiles(files, "/sys/fs/cgroup/user.slice/user-1000.slice", "max", 3 * kGiB);
  AddCgroup(files, scope);
  AddMemoryFiles(files, scope, std::to_string(8 * kGiB), 2 * kGiB);

  const CgroupMemoryLimits cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.status, CgroupStatus::kResolved);
  EXPECT_EQ(cgroup.limit, std::optional<uint64_t>{8 * kGiB});
  EXPECT_EQ(cgroup.headroom, std::optional<uint64_t>{6 * kGiB});

  const HostFacts facts = ResolveHostFacts(cgroup, 128 * kGiB, 100 * kGiB);
  EXPECT_EQ(facts.effective, std::optional<uint64_t>{8 * kGiB});
  EXPECT_EQ(facts.available, std::optional<uint64_t>{6 * kGiB});

  // A profile that needs a large host is not eligible inside the small cgroup, but is on the unconstrained host.
  Config config;
  OverlayConfig(config, R"({"runtime_profiles":[{"id":"large-host",
    "eligibility":{"minimum_total_device_memory_bytes":0,"minimum_effective_host_memory_bytes":68719476736},
    "overlay":{"search":{"chunk_size":1}}}]})");
  auto in_cgroup = config;
  ApplyRuntimeProfile(in_cgroup, RuntimeProfileDeviceFacts{1, false, std::nullopt, facts.effective, facts.available});
  EXPECT_FALSE(in_cgroup.search.chunk_size.has_value());
  auto unconstrained = config;
  ApplyRuntimeProfile(unconstrained, RuntimeProfileDeviceFacts{1, false, std::nullopt, 128 * kGiB, 100 * kGiB});
  EXPECT_EQ(unconstrained.search.chunk_size, 1u);
}

TEST(ConfigTest, CgroupMemoryIsReadRelativeToACgroupMountRootThatIsNotTheHierarchyRoot) {
  // A container that sees only its own cgroup: /sys/fs/cgroup shows /kubepods/pod-1/app, which has a 15 GiB limit.
  FakeHostFiles files;
  files["/proc/self/mountinfo"] =
      "30 22 0:26 /kubepods/pod-1/app /sys/fs/cgroup ro,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n";
  AddCgroup(files, "/sys/fs/cgroup");
  AddMemoryFiles(files, "/sys/fs/cgroup", std::to_string(15 * kGiB), 10 * kGiB);
  AddCgroup(files, "/sys/fs/cgroup/sidecar");
  AddMemoryFiles(files, "/sys/fs/cgroup/sidecar", "max", 1 * kGiB);
  // The cgroup path must not be appended to the mount point as a whole.
  AddCgroup(files, "/sys/fs/cgroup/kubepods/pod-1/app/sidecar");
  AddMemoryFiles(files, "/sys/fs/cgroup/kubepods/pod-1/app/sidecar", "1048576", 0);

  // A process in a cgroup below the one that is mounted: the mounted cgroup's own limit still applies.
  files["/proc/self/cgroup"] = "0::/kubepods/pod-1/app/sidecar\n";
  CgroupMemoryLimits cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.status, CgroupStatus::kResolved);
  EXPECT_EQ(cgroup.limit, std::optional<uint64_t>{15 * kGiB});
  EXPECT_EQ(cgroup.headroom, std::optional<uint64_t>{5 * kGiB});

  // A process in the mounted cgroup itself reads the files at the mount point.
  files["/proc/self/cgroup"] = "0::/kubepods/pod-1/app\n";
  cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.status, CgroupStatus::kResolved);
  EXPECT_EQ(cgroup.limit, std::optional<uint64_t>{15 * kGiB});
  EXPECT_EQ(cgroup.headroom, std::optional<uint64_t>{5 * kGiB});

  // In a private cgroup namespace the mount shows the namespace root as "/" and the process path is relative to it.
  files["/proc/self/mountinfo"] = "30 22 0:26 / /sys/fs/cgroup ro,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n";
  files["/proc/self/cgroup"] = "0::/\n";
  cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.status, CgroupStatus::kResolved);
  EXPECT_EQ(cgroup.limit, std::optional<uint64_t>{15 * kGiB});
  EXPECT_EQ(cgroup.headroom, std::optional<uint64_t>{5 * kGiB});

  // A sibling cgroup is not below the mounted root, so its limits cannot be read.
  files["/proc/self/mountinfo"] =
      "30 22 0:26 /kubepods/pod-1/app /sys/fs/cgroup ro,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n";
  files["/proc/self/cgroup"] = "0::/kubepods/pod-1/other\n";
  EXPECT_EQ(ResolveCgroupMemory(files).status, CgroupStatus::kUnresolved);
  // The same name prefix does not make a cgroup a descendant.
  files["/proc/self/cgroup"] = "0::/kubepods/pod-1/app-2\n";
  EXPECT_EQ(ResolveCgroupMemory(files).status, CgroupStatus::kUnresolved);
}

TEST(ConfigTest, CgroupMemoryTakesTheTightestAncestorLimitAndHeadroom) {
  FakeHostFiles files;
  files["/proc/self/cgroup"] = "0::/system.slice/llm.service/worker-1\n";
  files["/proc/self/mountinfo"] = std::string{kHostMountinfo};
  AddCgroup(files, "/sys/fs/cgroup");
  AddCgroup(files, "/sys/fs/cgroup/system.slice");
  AddMemoryFiles(files, "/sys/fs/cgroup/system.slice", "max", 40 * kGiB);
  AddCgroup(files, "/sys/fs/cgroup/system.slice/llm.service");
  AddCgroup(files, "/sys/fs/cgroup/system.slice/llm.service/worker-1");

  // The service has used 15 of its 16 GiB, so it leaves less headroom than the worker's own 8 GiB limit with 2 GiB
  // used.
  AddMemoryFiles(files, "/sys/fs/cgroup/system.slice/llm.service", std::to_string(16 * kGiB), 15 * kGiB);
  AddMemoryFiles(files, "/sys/fs/cgroup/system.slice/llm.service/worker-1", std::to_string(8 * kGiB), 2 * kGiB);
  CgroupMemoryLimits cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.status, CgroupStatus::kResolved);
  EXPECT_EQ(cgroup.limit, std::optional<uint64_t>{8 * kGiB});
  EXPECT_EQ(cgroup.headroom, std::optional<uint64_t>{1 * kGiB});
  HostFacts facts = ResolveHostFacts(cgroup, 128 * kGiB, 100 * kGiB);
  EXPECT_EQ(facts.effective, std::optional<uint64_t>{8 * kGiB});
  EXPECT_EQ(facts.available, std::optional<uint64_t>{1 * kGiB});

  // A parent with a lower limit than the worker caps the effective memory.
  AddMemoryFiles(files, "/sys/fs/cgroup/system.slice/llm.service", std::to_string(4 * kGiB), 1 * kGiB);
  cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.limit, std::optional<uint64_t>{4 * kGiB});
  EXPECT_EQ(cgroup.headroom, std::optional<uint64_t>{3 * kGiB});
  facts = ResolveHostFacts(cgroup, 128 * kGiB, 100 * kGiB);
  EXPECT_EQ(facts.effective, std::optional<uint64_t>{4 * kGiB});
  EXPECT_EQ(facts.available, std::optional<uint64_t>{3 * kGiB});

  // A worker that has used more than its limit has no headroom.
  AddMemoryFiles(files, "/sys/fs/cgroup/system.slice/llm.service", "max", 15 * kGiB);
  AddMemoryFiles(files, "/sys/fs/cgroup/system.slice/llm.service/worker-1", std::to_string(8 * kGiB), 9 * kGiB);
  cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.limit, std::optional<uint64_t>{8 * kGiB});
  EXPECT_EQ(cgroup.headroom, std::optional<uint64_t>{0});

  // The worker does not have the memory controller of its own: the service's limit is the one that applies.
  files.erase("/sys/fs/cgroup/system.slice/llm.service/worker-1/memory.max");
  files.erase("/sys/fs/cgroup/system.slice/llm.service/worker-1/memory.current");
  AddCgroup(files, "/sys/fs/cgroup/system.slice/llm.service/worker-1", /*has_memory_controller=*/false);
  AddMemoryFiles(files, "/sys/fs/cgroup/system.slice/llm.service", std::to_string(16 * kGiB), 15 * kGiB);
  cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.status, CgroupStatus::kResolved);
  EXPECT_EQ(cgroup.limit, std::optional<uint64_t>{16 * kGiB});
  EXPECT_EQ(cgroup.headroom, std::optional<uint64_t>{1 * kGiB});
}

TEST(ConfigTest, CgroupMemoryIsUnboundedWhenEveryLevelIsUnlimited) {
  FakeHostFiles files;
  files["/proc/self/cgroup"] = "0::/system.slice/llm.service\n";
  files["/proc/self/mountinfo"] = std::string{kHostMountinfo};
  AddCgroup(files, "/sys/fs/cgroup");  // the hierarchy root has no memory files
  AddCgroup(files, "/sys/fs/cgroup/system.slice");
  AddMemoryFiles(files, "/sys/fs/cgroup/system.slice", "max", 40 * kGiB);
  AddCgroup(files, "/sys/fs/cgroup/system.slice/llm.service");
  AddMemoryFiles(files, "/sys/fs/cgroup/system.slice/llm.service", "max", 30 * kGiB);

  const CgroupMemoryLimits cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.status, CgroupStatus::kResolved);
  EXPECT_FALSE(cgroup.limit.has_value());
  EXPECT_FALSE(cgroup.headroom.has_value());

  const HostFacts facts = ResolveHostFacts(cgroup, 128 * kGiB, 100 * kGiB);
  EXPECT_EQ(facts.effective, std::optional<uint64_t>{128 * kGiB});
  EXPECT_EQ(facts.available, std::optional<uint64_t>{100 * kGiB});

  // So is a hierarchy without the memory controller anywhere, such as the unified hierarchy of a hybrid cgroup setup.
  files.clear();
  files["/proc/self/cgroup"] =
      "12:memory:/system.slice/llm.service\n1:name=systemd:/system.slice/llm.service\n"
      "0::/system.slice/llm.service\n";
  files["/proc/self/mountinfo"] =
      "26 25 0:23 / /sys/fs/cgroup/unified rw,nosuid,nodev,noexec,relatime shared:9 - "
      "cgroup2 cgroup2 rw,nsdelegate\n"
      "27 25 0:24 / /sys/fs/cgroup/memory rw,nosuid,nodev,noexec,relatime shared:10 - "
      "cgroup cgroup rw,memory\n";
  AddCgroup(files, "/sys/fs/cgroup/unified", /*has_memory_controller=*/false);
  AddCgroup(files, "/sys/fs/cgroup/unified/system.slice", /*has_memory_controller=*/false);
  AddCgroup(files, "/sys/fs/cgroup/unified/system.slice/llm.service", /*has_memory_controller=*/false);
  const CgroupMemoryLimits hybrid = ResolveCgroupMemory(files);
  EXPECT_EQ(hybrid.status, CgroupStatus::kResolved);
  EXPECT_FALSE(hybrid.limit.has_value());
  EXPECT_FALSE(hybrid.headroom.has_value());
}

TEST(ConfigTest, CgroupMemoryMaxMayBeMissingOnlyFromTheHierarchyRootAndCgroupsWithoutTheMemoryController) {
  FakeHostFiles files;
  files["/proc/self/cgroup"] = "0::/app/job\n";
  files["/proc/self/mountinfo"] = std::string{kHostMountinfo};
  AddCgroup(files, "/sys/fs/cgroup");  // the hierarchy root lists the memory controller but has no memory.max
  AddCgroup(files, "/sys/fs/cgroup/app", /*has_memory_controller=*/false);
  AddCgroup(files, "/sys/fs/cgroup/app/job");
  AddMemoryFiles(files, "/sys/fs/cgroup/app/job", std::to_string(8 * kGiB), 2 * kGiB);
  CgroupMemoryLimits cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.status, CgroupStatus::kResolved);
  EXPECT_EQ(cgroup.limit, std::optional<uint64_t>{8 * kGiB});
  EXPECT_EQ(cgroup.headroom, std::optional<uint64_t>{6 * kGiB});

  // memory.max exists wherever the memory controller does, so one that is missing from the process's cgroup, or from
  // an ancestor, is not an unlimited level.
  auto without_leaf_max = files;
  without_leaf_max.erase("/sys/fs/cgroup/app/job/memory.max");
  EXPECT_EQ(ResolveCgroupMemory(without_leaf_max).status, CgroupStatus::kUnresolved);
  auto with_ancestor_controller = files;
  AddCgroup(with_ancestor_controller, "/sys/fs/cgroup/app");
  EXPECT_EQ(ResolveCgroupMemory(with_ancestor_controller).status, CgroupStatus::kUnresolved);

  // A level without memory.max and without a readable cgroup.controllers cannot be classified.
  auto without_controllers = files;
  without_controllers.erase("/sys/fs/cgroup/app/cgroup.controllers");
  EXPECT_EQ(ResolveCgroupMemory(without_controllers).status, CgroupStatus::kUnresolved);

  // A mount rooted at a sub-cgroup does not show the hierarchy root, so its first level needs memory.max as well.
  FakeHostFiles sub_mount;
  sub_mount["/proc/self/cgroup"] = "0::/app/job\n";
  sub_mount["/proc/self/mountinfo"] =
      "30 22 0:26 /app /sys/fs/cgroup ro,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n";
  AddCgroup(sub_mount, "/sys/fs/cgroup");
  AddCgroup(sub_mount, "/sys/fs/cgroup/job");
  AddMemoryFiles(sub_mount, "/sys/fs/cgroup/job", std::to_string(8 * kGiB), 2 * kGiB);
  EXPECT_EQ(ResolveCgroupMemory(sub_mount).status, CgroupStatus::kUnresolved);
  AddMemoryFiles(sub_mount, "/sys/fs/cgroup", std::to_string(16 * kGiB), 4 * kGiB);
  cgroup = ResolveCgroupMemory(sub_mount);
  EXPECT_EQ(cgroup.status, CgroupStatus::kResolved);
  EXPECT_EQ(cgroup.limit, std::optional<uint64_t>{8 * kGiB});
}

TEST(ConfigTest, CgroupMemoryTreatsAZeroMemoryMaxAsAHardLimit) {
  // memory.max is "max" when a cgroup is unlimited, so a number, including zero, is a limit.
  FakeHostFiles files;
  files["/proc/self/cgroup"] = "0::/app/job\n";
  files["/proc/self/mountinfo"] = std::string{kHostMountinfo};
  AddCgroup(files, "/sys/fs/cgroup");
  AddCgroup(files, "/sys/fs/cgroup/app");
  AddMemoryFiles(files, "/sys/fs/cgroup/app", "0", 0);
  AddCgroup(files, "/sys/fs/cgroup/app/job");
  AddMemoryFiles(files, "/sys/fs/cgroup/app/job", std::to_string(8 * kGiB), 2 * kGiB);

  const CgroupMemoryLimits cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.status, CgroupStatus::kResolved);
  EXPECT_EQ(cgroup.limit, std::optional<uint64_t>{0});
  EXPECT_EQ(cgroup.headroom, std::optional<uint64_t>{0});

  const HostFacts facts = ResolveHostFacts(cgroup, 128 * kGiB, 100 * kGiB);
  EXPECT_EQ(facts.effective, std::optional<uint64_t>{0});
  EXPECT_EQ(facts.available, std::optional<uint64_t>{0});

  // The host-wide memory must not make a profile eligible when the cgroup allows none.
  Config config;
  OverlayConfig(config, R"({"runtime_profiles":[{"id":"any-host",
    "eligibility":{"minimum_total_device_memory_bytes":0,"minimum_effective_host_memory_bytes":1},
    "overlay":{"search":{"chunk_size":1}}}]})");
  ApplyRuntimeProfile(config, RuntimeProfileDeviceFacts{1, false, std::nullopt, facts.effective, facts.available});
  EXPECT_FALSE(config.search.chunk_size.has_value());
}

TEST(ConfigTest, CgroupMemoryReadsEscapedMountsAndPicksTheMountThatShowsTheMostAncestors) {
  FakeHostFiles files;
  files["/proc/self/cgroup"] = "0::/a b/leaf\n";
  // The cgroup v1 mount is skipped. The spaces in the paths of the cgroup2 mounts are escaped. The first cgroup2 mount
  // shows the whole hierarchy, the second one starts at "/a b" and the third one at the process's own cgroup.
  const std::string v1_mount =
      "26 25 0:23 / /sys/fs/cgroup/memory rw,nosuid shared:9 master:2 - cgroup cgroup rw,memory\n";
  const std::string whole_mount = "27 25 0:24 / /mnt/cg\\040v2 rw,nosuid shared:5 - cgroup2 cgroup2 rw\n";
  const std::string sub_mount = "28 25 0:24 /a\\040b /mnt/sub rw,nosuid - cgroup2 cgroup2 rw\n";
  const std::string leaf_mount = "29 25 0:24 /a\\040b/leaf /mnt/inner rw,nosuid - cgroup2 cgroup2 rw\n";

  AddCgroup(files, "/mnt/cg v2");
  AddCgroup(files, "/mnt/cg v2/a b");
  AddCgroup(files, "/mnt/cg v2/a b/leaf");
  AddMemoryFiles(files, "/mnt/cg v2/a b", std::to_string(32 * kGiB), 16 * kGiB);
  AddMemoryFiles(files, "/mnt/cg v2/a b/leaf", std::to_string(24 * kGiB), 4 * kGiB);
  AddCgroup(files, "/mnt/sub");
  AddCgroup(files, "/mnt/sub/leaf");
  AddMemoryFiles(files, "/mnt/sub", std::to_string(16 * kGiB), 8 * kGiB);
  AddMemoryFiles(files, "/mnt/sub/leaf", std::to_string(12 * kGiB), 2 * kGiB);
  AddCgroup(files, "/mnt/inner");
  AddMemoryFiles(files, "/mnt/inner", "1048576", 0);

  // The mount of the whole hierarchy shows the most ancestors.
  files["/proc/self/mountinfo"] = v1_mount + sub_mount + whole_mount + leaf_mount;
  CgroupMemoryLimits cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.status, CgroupStatus::kResolved);
  EXPECT_EQ(cgroup.limit, std::optional<uint64_t>{24 * kGiB});
  EXPECT_EQ(cgroup.headroom, std::optional<uint64_t>{16 * kGiB});

  // Without it, the mount rooted at "/a b" does.
  files["/proc/self/mountinfo"] = v1_mount + leaf_mount + sub_mount;
  cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.status, CgroupStatus::kResolved);
  EXPECT_EQ(cgroup.limit, std::optional<uint64_t>{12 * kGiB});
  EXPECT_EQ(cgroup.headroom, std::optional<uint64_t>{8 * kGiB});
}

TEST(ConfigTest, CgroupMemoryFallsBackToThePhysicalReadingsWithoutCgroupV2) {
  FakeHostFiles files;
  // A cgroup v1 system has no unified hierarchy entry, so the mounts are not consulted.
  files["/proc/self/cgroup"] = "12:memory:/docker/abc\n11:cpu,cpuacct:/docker/abc\n1:name=systemd:/docker/abc\n";
  CgroupMemoryLimits cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.status, CgroupStatus::kNotApplicable);
  HostFacts facts = ResolveHostFacts(cgroup, 128 * kGiB, 100 * kGiB);
  EXPECT_EQ(facts.effective, std::optional<uint64_t>{128 * kGiB});
  EXPECT_EQ(facts.available, std::optional<uint64_t>{100 * kGiB});

  // A unified hierarchy entry without a cgroup2 mount has no files to read.
  files["/proc/self/cgroup"] = "0::/docker/abc\n";
  files["/proc/self/mountinfo"] = "22 1 8:2 / / rw,relatime shared:1 - ext4 /dev/sda2 rw\n";
  cgroup = ResolveCgroupMemory(files);
  EXPECT_EQ(cgroup.status, CgroupStatus::kNotApplicable);
  facts = ResolveHostFacts(cgroup, 128 * kGiB, 100 * kGiB);
  EXPECT_EQ(facts.effective, std::optional<uint64_t>{128 * kGiB});
  EXPECT_EQ(facts.available, std::optional<uint64_t>{100 * kGiB});
}

TEST(ConfigTest, UnreadableOrUnparseableCgroupInputsLeaveTheHostMemoryUnknown) {
  const auto valid_files = [] {
    FakeHostFiles files;
    files["/proc/self/cgroup"] = "0::/app/job\n";
    files["/proc/self/mountinfo"] = std::string{kHostMountinfo};
    AddCgroup(files, "/sys/fs/cgroup");
    AddCgroup(files, "/sys/fs/cgroup/app");
    AddMemoryFiles(files, "/sys/fs/cgroup/app", "max", 3 * kGiB);
    AddCgroup(files, "/sys/fs/cgroup/app/job");
    AddMemoryFiles(files, "/sys/fs/cgroup/app/job", std::to_string(8 * kGiB), 2 * kGiB);
    return files;
  };
  ASSERT_EQ(ResolveCgroupMemory(valid_files()).status, CgroupStatus::kResolved);

  const std::string job = "/sys/fs/cgroup/app/job";
  const auto mountinfo = [](std::string text) {
    return [text = std::move(text)](FakeHostFiles& f) { f["/proc/self/mountinfo"] = text; };
  };
  const std::vector<std::pair<std::string, std::function<void(FakeHostFiles&)>>> corruptions = {
      {"missing /proc/self/cgroup", [](FakeHostFiles& f) { f.erase("/proc/self/cgroup"); }},
      {"empty /proc/self/cgroup", [](FakeHostFiles& f) { f["/proc/self/cgroup"] = ""; }},
      {"malformed /proc/self/cgroup", [](FakeHostFiles& f) { f["/proc/self/cgroup"] = "not a cgroup file\n"; }},
      {"non-numeric hierarchy id", [](FakeHostFiles& f) { f["/proc/self/cgroup"] = "x::/app/job\n"; }},
      {"empty cgroup path", [](FakeHostFiles& f) { f["/proc/self/cgroup"] = "0::\n"; }},
      {"relative cgroup path", [](FakeHostFiles& f) { f["/proc/self/cgroup"] = "0::app/job\n"; }},
      {"cgroup path outside the namespace root", [](FakeHostFiles& f) { f["/proc/self/cgroup"] = "0::/../app/job\n"; }},
      {"missing /proc/self/mountinfo", [](FakeHostFiles& f) { f.erase("/proc/self/mountinfo"); }},
      {"empty /proc/self/mountinfo", mountinfo("")},
      {"malformed /proc/self/mountinfo", mountinfo("garbage\n")},
      {"mountinfo without a separator", mountinfo("24 22 0:22 / /sys/fs/cgroup rw shared:5 cgroup2 cgroup2 rw\n")},
      {"bad escape in the mount point", mountinfo("24 22 0:22 / /sys/fs/cg\\9roup rw - cgroup2 cgroup2 rw\n")},
      {"cgroup outside the cgroup2 mount root",
       mountinfo("24 22 0:22 /other /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n")},
      {"cgroup2 mount root outside the namespace root",
       mountinfo("24 22 0:22 /../.. /sys/fs/cgroup rw - cgroup2 cgroup2 rw\n")},
      {"cgroup directory that is not visible", [&](FakeHostFiles& f) { f.erase(job + "/cgroup.controllers"); }},
      {"unparseable memory.max", [&](FakeHostFiles& f) { f[job + "/memory.max"] = "lots\n"; }},
      {"negative memory.max", [&](FakeHostFiles& f) { f[job + "/memory.max"] = "-1\n"; }},
      {"empty memory.max", [&](FakeHostFiles& f) { f[job + "/memory.max"] = ""; }},
      {"memory.max beyond 64 bits", [&](FakeHostFiles& f) { f[job + "/memory.max"] = "18446744073709551616\n"; }},
      {"missing memory.current", [&](FakeHostFiles& f) { f.erase(job + "/memory.current"); }},
      {"unparseable memory.current", [&](FakeHostFiles& f) { f[job + "/memory.current"] = "n/a\n"; }},
      // The cgroup lists the memory controller, so memory.max exists and not being able to read it is not "unlimited".
      {"unreadable memory.max in a cgroup with the memory controller",
       [&](FakeHostFiles& f) { f.erase(job + "/memory.max"); }},
      {"unreadable memory.max in an ancestor with the memory controller",
       [](FakeHostFiles& f) { f.erase("/sys/fs/cgroup/app/memory.max"); }},
      {"unreadable memory.max and cgroup.controllers in an ancestor",
       [](FakeHostFiles& f) {
         f.erase("/sys/fs/cgroup/app/memory.max");
         f.erase("/sys/fs/cgroup/app/cgroup.controllers");
       }},
  };
  for (const auto& [name, corrupt] : corruptions) {
    auto files = valid_files();
    corrupt(files);
    const CgroupMemoryLimits cgroup = ResolveCgroupMemory(files);
    EXPECT_EQ(cgroup.status, CgroupStatus::kUnresolved) << name;
    EXPECT_FALSE(cgroup.limit.has_value()) << name;
    EXPECT_FALSE(cgroup.headroom.has_value()) << name;

    // Neither the host-wide memory nor the unlimited mount root stands in for the process's cgroup.
    const HostFacts facts = ResolveHostFacts(cgroup, 128 * kGiB, 100 * kGiB);
    EXPECT_FALSE(facts.effective.has_value()) << name;
    EXPECT_FALSE(facts.available.has_value()) << name;
  }

  // A profile bounded by the host memory is not eligible when the host memory is unknown.
  Config config;
  OverlayConfig(config, R"({"runtime_profiles":[{"id":"any-host",
    "eligibility":{"minimum_total_device_memory_bytes":0,"minimum_effective_host_memory_bytes":0},
    "overlay":{"search":{"chunk_size":1}}}]})");
  ApplyRuntimeProfile(config, RuntimeProfileDeviceFacts{1, false, std::nullopt, std::nullopt, std::nullopt});
  EXPECT_FALSE(config.search.chunk_size.has_value());
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

namespace {

constexpr const char* kIntegratedRuntimeProfileConfig = R"({
  "engine":{"dynamic_batching":{"num_blocks":32}},
  "runtime_profiles":[
    {"id":"integrated","eligibility":{"minimum_total_device_memory_bytes":0,"is_integrated":true},
     "overlay":{"engine":{"dynamic_batching":{"num_blocks":40}}}},
    {"id":"discrete","eligibility":{"minimum_total_device_memory_bytes":0,"is_integrated":false},
     "overlay":{"engine":{"dynamic_batching":{"num_blocks":80}}}}
  ]
})";

}  // namespace

TEST(ConfigTest, ForcedRuntimeProfileDoesNotNeedTheIntegratedCapabilityQuery) {
  Config config;
  OverlayConfig(config, kIntegratedRuntimeProfileConfig);
  CountingCudaDevice device;
  device.state->total_memory_bytes = 100;
  device.state->is_integrated = std::nullopt;  // the capability query fails

  ScopedEnvironmentVariable forced{"ORTGENAI_ENABLE_RUNTIME_PROFILE", "integrated"};
  testing::internal::CaptureStderr();
  ApplyRuntimeProfileForSelectedDevice(config, device);
  const auto warning = testing::internal::GetCapturedStderr();

  // The forced profile is applied as written, and eligibility, which is what needs the query, is skipped.
  EXPECT_EQ(*config.engine.dynamic_batching->num_blocks, 40u);
  EXPECT_EQ(device.state->integrated_queries, 0u);
  EXPECT_EQ(warning.find("test integrated-device query failed"), std::string::npos) << warning;
  EXPECT_TRUE(config.runtime_profiles.empty());
}

TEST(ConfigTest, UnknownForcedRuntimeProfileIdFailsWhenTheIntegratedCapabilityQueryFails) {
  Config config;
  OverlayConfig(config, kIntegratedRuntimeProfileConfig);
  CountingCudaDevice device;
  device.state->total_memory_bytes = 100;
  device.state->is_integrated = std::nullopt;

  ScopedEnvironmentVariable forced{"ORTGENAI_ENABLE_RUNTIME_PROFILE", "no-such-profile"};
  try {
    ApplyRuntimeProfileForSelectedDevice(config, device);
    FAIL() << "An unknown forced profile id must not fall back to the base settings";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("no-such-profile"), std::string::npos) << error.what();
  }
  EXPECT_EQ(*config.engine.dynamic_batching->num_blocks, 32u);
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
