// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "config.h"
#include "models/model_type.h"
#include "models/qwen38_decode_device.h"
#include "models/qwen38_graph.h"

namespace Generators::test {

namespace {

std::filesystem::path MakeScratchDir(const std::string& suffix) {
  const auto dir = std::filesystem::temp_directory_path() / ("genai_qwen38_graph_" + suffix);
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  return dir;
}

std::filesystem::path WriteFile(const std::filesystem::path& path, const std::string& text) {
  std::ofstream out{path, std::ios::binary};
  out << text;
  return path;
}

struct MockPlaneTracker {
  enum class DeviceAlias { kBiased,
                           kFails,
                           kNull,
                           kIdentity };

  static inline int alloc_calls = 0;
  static inline int fail_on_alloc_call = -1;
  static inline DeviceAlias device_alias = DeviceAlias::kBiased;
  static inline std::vector<void*> live_device_ptrs;
  static inline std::vector<void*> registered_host_ptrs;
  static inline std::vector<void*> unregistered_host_ptrs;
  static constexpr uintptr_t kDevBias = 0x200000u;

  static void Reset(int fail_alloc = -1) {
    for (void* p : live_device_ptrs) std::free(p);
    live_device_ptrs.clear();
    registered_host_ptrs.clear();
    unregistered_host_ptrs.clear();
    alloc_calls = 0;
    fail_on_alloc_call = fail_alloc;
    device_alias = DeviceAlias::kBiased;
  }

  static const char* ErrorString(int) { return "mock cuda error"; }
  static int DeviceAlloc(void** p, size_t bytes) {
    ++alloc_calls;
    if (fail_on_alloc_call > 0 && alloc_calls == fail_on_alloc_call) return 2;  // cudaErrorMemoryAllocation
    *p = std::calloc(1, bytes ? bytes : 16);
    live_device_ptrs.push_back(*p);
    return 0;
  }
  static int DeviceFree(void* p) {
    auto it = std::find(live_device_ptrs.begin(), live_device_ptrs.end(), p);
    if (it != live_device_ptrs.end()) {
      live_device_ptrs.erase(it);
      std::free(p);
    }
    return 0;
  }
  static int HostAlloc(void** p, size_t bytes) {
    *p = std::calloc(1, bytes ? bytes : 16);
    return 0;
  }
  static int HostFree(void* p) {
    std::free(p);
    return 0;
  }
  static int HostRegister(void* p, size_t, unsigned) {
    registered_host_ptrs.push_back(p);
    return 0;
  }
  static int HostUnregister(void* p) {
    unregistered_host_ptrs.push_back(p);
    return 0;
  }
  static int HostGetDevicePointer(void** dp, void* hp, unsigned) {
    switch (device_alias) {
      case DeviceAlias::kFails:
        return 1;  // cudaErrorInvalidValue
      case DeviceAlias::kNull:
        *dp = nullptr;
        return 0;
      case DeviceAlias::kIdentity:
        *dp = hp;
        return 0;
      case DeviceAlias::kBiased:
        break;
    }
    *dp = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(hp) + kDevBias);
    return 0;
  }
  static int DeviceMemset(void* p, int v, size_t bytes, void*) {
    std::memset(p, v, bytes);
    return 0;
  }
  static int MemcpyH2D(void* dst, const void* src, size_t bytes, void*) {
    std::memcpy(dst, src, bytes);
    return 0;
  }
  static int StreamCreate(void** s) {
    *s = reinterpret_cast<void*>(uintptr_t{1});
    return 0;
  }
  static int StreamDestroy(void*) { return 0; }
  static int StreamSync(void*) { return 0; }
};

void InitMockPlane(Qwen38Plane& plane, int fail_alloc = -1) {
  MockPlaneTracker::Reset(fail_alloc);
  plane.error_string = &MockPlaneTracker::ErrorString;
  plane.device_alloc = &MockPlaneTracker::DeviceAlloc;
  plane.device_free = &MockPlaneTracker::DeviceFree;
  plane.host_alloc = &MockPlaneTracker::HostAlloc;
  plane.host_free = &MockPlaneTracker::HostFree;
  plane.host_register = &MockPlaneTracker::HostRegister;
  plane.host_unregister = &MockPlaneTracker::HostUnregister;
  plane.host_get_device_pointer = &MockPlaneTracker::HostGetDevicePointer;
  plane.device_memset = &MockPlaneTracker::DeviceMemset;
  plane.memcpy_h2d = &MockPlaneTracker::MemcpyH2D;
  plane.stream_create = &MockPlaneTracker::StreamCreate;
  plane.stream_destroy = &MockPlaneTracker::StreamDestroy;
  plane.stream_sync = &MockPlaneTracker::StreamSync;
}

}  // namespace

// ---- config ----------------------------------------------------------------------------------

TEST(Qwen38GraphConfigTest, DefaultsAreUnconfigured) {
  Config config;
  EXPECT_FALSE(config.model.qwen38_graph.IsConfigured());
  EXPECT_EQ(config.model.qwen38_graph.kv_capacity, 33028);
  EXPECT_TRUE(config.model.qwen38_graph.warmup);
  EXPECT_TRUE(config.model.qwen38_graph.mtp.enabled);
  EXPECT_TRUE(config.model.qwen38_graph.mtp.prefill_head);
}

TEST(Qwen38GraphConfigTest, ParsesEverySetting) {
  Config config;
  OverlayConfig(config, R"({"model":{"type":"qwen38_flash_graph","qwen38_graph":{
    "plane_library":"libs/libqwen38_plane.so",
    "ple_table":"ple_table.txt",
    "kv_capacity":1024,
    "warmup":false,
    "mtp":{"enabled":false,"prefill_head":false},
    "head":{"filename_s1":"head/s1.onnx","filename_s2":"head/s2.onnx",
            "session_options":{"custom_ops_libraries":["libs/a.so","libs/b.so"],
                               "provider_options":[{"cuda":{"enable_cuda_graph":"1"}}]}}
  }}})");

  const auto& q = config.model.qwen38_graph;
  EXPECT_TRUE(q.IsConfigured());
  EXPECT_EQ(q.plane_library, "libs/libqwen38_plane.so");
  EXPECT_EQ(q.ple_table, "ple_table.txt");
  EXPECT_EQ(q.kv_capacity, 1024);
  EXPECT_FALSE(q.warmup);
  EXPECT_FALSE(q.mtp.enabled);
  EXPECT_FALSE(q.mtp.prefill_head);
  EXPECT_EQ(q.head.filename_s1, "head/s1.onnx");
  EXPECT_EQ(q.head.filename_s2, "head/s2.onnx");
  ASSERT_TRUE(q.head.session_options.has_value());
  EXPECT_EQ(q.head.session_options->custom_ops_libraries, (std::vector<std::string>{"libs/a.so", "libs/b.so"}));
}

TEST(Qwen38GraphConfigTest, FoldAndPrefillDefaultToOff) {
  Config config;
  const auto& q = config.model.qwen38_graph;
  EXPECT_FALSE(q.fold.enabled);
  EXPECT_FALSE(q.prefill.enabled);
  EXPECT_EQ(q.prefill.chunk, 0);
  EXPECT_EQ(q.prefill.warmup_lengths, (std::vector<int>{8192, 2048, 512, 128, 37}));
}

TEST(Qwen38GraphConfigTest, ParsesTheFoldAndPrefillSections) {
  Config config;
  OverlayConfig(config, R"({"model":{"type":"qwen38_flash_graph","qwen38_graph":{
    "plane_library":"libs/libqwen38_plane.so",
    "ple_table":"ple_table.txt",
    "fold":{"enabled":true,
            "head_r2":{"filename":"head_r2/model.onnx",
                       "session_options":{"custom_ops_libraries":["libs/a.so"]}}},
    "prefill":{"enabled":true,"chunk":4096,"weights_pool":"pool.txt","constants":"qsa.txt",
               "dense":{"filename":"dense/model.onnx","session_options":{"provider_options":[{"cuda":{}}]}},
               "head_kv":{"filename":"head_kv/model.onnx"},
               "warmup_lengths":[64,16]}
  }}})");

  const auto& q = config.model.qwen38_graph;
  EXPECT_TRUE(q.IsConfigured());  // no head.filename_s1: the fold head_r2 session is enough
  EXPECT_TRUE(q.fold.enabled);
  EXPECT_EQ(q.fold.head_r2.filename, "head_r2/model.onnx");
  ASSERT_TRUE(q.fold.head_r2.session_options.has_value());
  EXPECT_EQ(q.fold.head_r2.session_options->custom_ops_libraries, (std::vector<std::string>{"libs/a.so"}));
  EXPECT_TRUE(q.prefill.enabled);
  EXPECT_EQ(q.prefill.chunk, 4096);
  EXPECT_EQ(q.prefill.weights_pool, "pool.txt");
  EXPECT_EQ(q.prefill.constants, "qsa.txt");
  EXPECT_EQ(q.prefill.dense.filename, "dense/model.onnx");
  ASSERT_TRUE(q.prefill.dense.session_options.has_value());
  EXPECT_EQ(q.prefill.head_kv.filename, "head_kv/model.onnx");
  EXPECT_FALSE(q.prefill.head_kv.session_options.has_value());
  EXPECT_EQ(q.prefill.warmup_lengths, (std::vector<int>{64, 16}));
}

TEST(Qwen38GraphConfigTest, RejectsUnknownFoldAndPrefillKeys) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"fold":{"unknown":true}}}})"), std::exception);
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"fold":{"head_r2":{"unknown":1}}}}})"), std::exception);
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"prefill":{"unknown":1}}}})"), std::exception);
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"prefill":{"dense":{"unknown":1}}}}})"), std::exception);
}

// Experimental, test-only and derivable settings are not part of the public configuration.
TEST(Qwen38GraphConfigTest, RejectsKeysThatAreNotPublicConfiguration) {
  const char* const overlays[] = {
      R"({"model":{"qwen38_graph":{"mtp":{"skip_refeed":true}}}})",
      R"({"model":{"qwen38_graph":{"fold":{"force_reject":true}}}})",
      R"({"model":{"qwen38_graph":{"fold":{"mtp":{"filename":"mtp.onnx"}}}}})",
      R"({"model":{"qwen38_graph":{"fold":{"drafter":{"filename":"drafter.onnx"}}}}})",
      R"({"model":{"qwen38_graph":{"prefill_chunk_tokens":512}}})",
      R"({"model":{"qwen38_graph":{"target_s1_graph_id":41}}})",
      R"({"model":{"qwen38_graph":{"target_s2_graph_id":51}}})",
      R"({"model":{"qwen38_graph":{"head_s1_graph_id":61}}})",
      R"({"model":{"qwen38_graph":{"head_s2_graph_id":63}}})",
      R"({"model":{"qwen38_graph":{"fold":{"target_graph_id":0}}}})",
      R"({"model":{"qwen38_graph":{"fold":{"head_r2":{"graph_id":0}}}}})",
      R"({"model":{"qwen38_graph":{"fold":{"target_graph_id":71,"head_r2":{"graph_id":71}}}}})",
  };
  for (const char* overlay : overlays) {
    Config config;
    EXPECT_THROW(OverlayConfig(config, overlay), std::exception) << overlay;
  }
}

TEST(Qwen38GraphConfigTest, AcceptsLegacyRc5FoldGraphIdsThroughCompatibilityLayer) {
  Config config;
  EXPECT_NO_THROW(OverlayConfig(
      config, R"({"model":{"qwen38_graph":{"fold":{"enabled":true,"target_graph_id":71,"head_r2":{"filename":"r2.onnx","graph_id":81}}}}})"));
  EXPECT_EQ(config.model.qwen38_graph.fold.legacy_target_graph_id, 71);
  EXPECT_EQ(config.model.qwen38_graph.fold.head_r2.legacy_graph_id, 81);
}

TEST(Qwen38GraphConfigTest, FoldAndPrefillPathsMustStayInsideTheModelFolder) {
  const auto dir = MakeScratchDir("fold_paths");
  const auto write_config = [&](const std::string& qwen_section) {
    WriteFile(dir / "genai_config.json",
              R"({"model":{"type":"qwen38_flash_graph","vocab_size":8,"context_length":16,
                  "decoder":{"filename":"model.onnx","hidden_size":4,"num_attention_heads":1,"num_hidden_layers":1,
                             "num_key_value_heads":1,"head_size":4},
                  "qwen38_graph":)" +
                  qwen_section + R"(},
                  "search":{"max_length":16}})");
  };
  write_config(R"({"plane_library":"p.so","ple_table":"ple.txt","fold":{"head_r2":{"filename":"r2.onnx"}},
                   "prefill":{"weights_pool":"pool.txt","constants":"c.txt","dense":{"filename":"d.onnx"},
                              "head_kv":{"filename":"k.onnx"}}})");
  EXPECT_NO_THROW(Config(fs::path(dir.string()), {}));
  for (const char* section : {R"("fold":{"head_r2":{"filename":"../r2.onnx"}})", R"("prefill":{"weights_pool":"/abs/pool.txt"})",
                              R"("prefill":{"constants":"../c.txt"})", R"("prefill":{"dense":{"filename":"/abs/d.onnx"}})",
                              R"("prefill":{"head_kv":{"filename":"../k.onnx"}})"}) {
    write_config(std::string(R"({"plane_library":"p.so","ple_table":"ple.txt",)") + section + "}");
    EXPECT_THROW(Config(fs::path(dir.string()), {}), std::runtime_error) << section;
  }
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38GraphConfigTest, RejectsUnknownKeys) {
  Config config;
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"unknown":1}}})"), std::exception);
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"mtp":{"unknown":true}}}})"), std::exception);
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"head":{"unknown":"x"}}}})"), std::exception);
}

TEST(Qwen38GraphConfigTest, HeadProviderOptionsBecomeProviders) {
  Config config;
  OverlayConfig(config, R"({"model":{"qwen38_graph":{"head":{"filename_s1":"h.onnx","session_options":{
    "provider_options":[{"cuda":{"enable_cuda_graph":"1"}}]}}}}})");
  // The Config constructor normalises provider_options into providers; an overlay alone must at least keep the options.
  ASSERT_TRUE(config.model.qwen38_graph.head.session_options.has_value());
  ASSERT_EQ(config.model.qwen38_graph.head.session_options->provider_options.size(), 1u);
  EXPECT_EQ(config.model.qwen38_graph.head.session_options->provider_options[0].name, "cuda");
}

TEST(Qwen38GraphConfigTest, TypeIsRegisteredAsALanguageModel) {
  EXPECT_TRUE(ModelType::IsLLM("qwen38_flash_graph"));
}

TEST(Qwen38GraphConfigTest, ModelPathsMustStayInsideTheModelFolder) {
  const auto dir = MakeScratchDir("paths");
  const auto write_config = [&](const std::string& qwen_section) {
    WriteFile(dir / "genai_config.json",
              R"({"model":{"type":"qwen38_flash_graph","vocab_size":8,"context_length":16,
                  "decoder":{"filename":"model.onnx","hidden_size":4,"num_attention_heads":1,"num_hidden_layers":1,
                             "num_key_value_heads":1,"head_size":4},
                  "qwen38_graph":)" +
                  qwen_section + R"(},
                  "search":{"max_length":16}})");
  };
  std::error_code ec;
  write_config(R"({"plane_library":"libs/plane.so","ple_table":"ple.txt","head":{"filename_s1":"h1.onnx"}})");
  EXPECT_NO_THROW(Config(fs::path(dir.string()), {}));
  for (const char* key : {"plane_library", "ple_table"}) {
    write_config(std::string(R"({")") + key + R"(":"../escape","head":{"filename_s1":"h1.onnx"}})");
    EXPECT_THROW(Config(fs::path(dir.string()), {}), std::runtime_error) << key;
    write_config(std::string(R"({")") + key + R"(":"/abs/path","head":{"filename_s1":"h1.onnx"}})");
    EXPECT_THROW(Config(fs::path(dir.string()), {}), std::runtime_error) << key;
  }
  write_config(R"({"head":{"filename_s1":"../h1.onnx"}})");
  EXPECT_THROW(Config(fs::path(dir.string()), {}), std::runtime_error);
  write_config(R"({"head":{"filename_s1":"h1.onnx","filename_s2":"/abs/h2.onnx"}})");
  EXPECT_THROW(Config(fs::path(dir.string()), {}), std::runtime_error);
  std::filesystem::remove_all(dir, ec);
}

// ---- host ledger -----------------------------------------------------------------------------

TEST(Qwen38DecodeLedgerTest, CompactPingPongAlternatesBanksAcrossOneToEightRowSteps) {
  Qwen38DecodeLedger ledger{64};
  ledger.SetCompactPingPong(true, 8);
  EXPECT_EQ(ledger.NumBanks(), 2);
  EXPECT_EQ(ledger.Bank(), 0);
  const size_t steps[] = {8, 1, 5, 7, 3, 8, 2};
  size_t length = 0;
  for (size_t i = 0; i < std::size(steps); ++i) {
    ledger.Advance(steps[i]);
    length += steps[i];
    EXPECT_EQ(ledger.Bank(), static_cast<int>((i + 1) % 2));
    EXPECT_EQ(ledger.Length(), length);
  }
  EXPECT_THROW(ledger.Advance(9), std::logic_error);
}

TEST(Qwen38DecodeLedgerTest, RingBankFollowsTheCommittedRows) {
  Qwen38DecodeLedger ledger{64};
  EXPECT_EQ(ledger.Bank(), 0);
  EXPECT_EQ(ledger.Length(), 0u);

  // Steps of 2, 1, 2, 2, 1, 1, 1: bank = (bank + step) % 3, length = sum.
  const size_t steps[] = {2, 1, 2, 2, 1, 1, 1};
  const int expected_bank[] = {2, 0, 2, 1, 2, 0, 1};
  size_t length = 0;
  for (size_t i = 0; i < std::size(steps); ++i) {
    ledger.Advance(steps[i]);
    length += steps[i];
    EXPECT_EQ(ledger.Bank(), expected_bank[i]) << "step " << i;
    EXPECT_EQ(ledger.Length(), length) << "step " << i;
  }
  ledger.Reset();
  EXPECT_EQ(ledger.Bank(), 0);
  EXPECT_EQ(ledger.Length(), 0u);
}

TEST(Qwen38DecodeLedgerTest, DensePromptRowsAdvanceTheLengthAtBankZero) {
  Qwen38DecodeLedger ledger{64};
  ledger.AdvancePrompt(8);
  ledger.AdvancePrompt(5);
  EXPECT_EQ(ledger.Length(), 13u);
  EXPECT_EQ(ledger.Bank(), 0);
  ledger.Advance(2);  // the decode loop continues from the hand-off bank 0
  EXPECT_EQ(ledger.Length(), 15u);
  EXPECT_EQ(ledger.Bank(), 2);
  EXPECT_THROW(ledger.AdvancePrompt(1), std::logic_error);  // a dense chunk after decoding / S=2 steps is a bug
  EXPECT_EQ(ledger.Length(), 15u);
  EXPECT_NO_THROW(Qwen38DecodeLedger{8}.RequireRoom(8, "x"));
}

TEST(Qwen38DecodeLedgerTest, AStepCommitsOneOrTwoRows) {
  Qwen38DecodeLedger ledger{64};
  EXPECT_THROW(ledger.Advance(0), std::logic_error);
  EXPECT_THROW(ledger.Advance(3), std::logic_error);
  EXPECT_EQ(ledger.Length(), 0u);
}

TEST(Qwen38DecodeLedgerTest, CapacityIncludesTheReservedTail) {
  Qwen38DecodeLedger ledger{10};
  EXPECT_NO_THROW(ledger.RequireRoom(10, "x"));
  EXPECT_THROW(ledger.RequireRoom(11, "x"), std::runtime_error);
  EXPECT_NO_THROW(ledger.RequireRoom(7, "x", 3));
  EXPECT_THROW(ledger.RequireRoom(8, "x", 3), std::runtime_error);
  for (int i = 0; i < 4; ++i) ledger.Advance(2);
  EXPECT_NO_THROW(ledger.RequireRoom(2, "x"));
  EXPECT_THROW(ledger.RequireRoom(3, "x"), std::runtime_error);
}

TEST(Qwen38DecodeLedgerTest, DeviceLengthMustMatchTheHost) {
  Qwen38DecodeLedger ledger{16};
  ledger.Advance(2);
  EXPECT_NO_THROW(ledger.CheckDeviceLength(2));
  EXPECT_THROW(ledger.CheckDeviceLength(3), std::runtime_error);
}

TEST(Qwen38CycleRecordTest, AcceptedCyclesCommitTwoTokens) {
  Qwen38CycleRecord record;
  EXPECT_EQ(record.TokenCount(), 1u);
  record.accepted = true;
  EXPECT_EQ(record.TokenCount(), 2u);
  record.accepted_count = 3;
  EXPECT_EQ(record.TokenCount(), 4u);
}

TEST(Qwen38DecodeLedgerTest, SupportsMultiDraftFiveBankRing) {
  Qwen38DecodeLedger ledger{64};
  ledger.SetNumBanks(5);
  EXPECT_EQ(ledger.NumBanks(), 5);
  ledger.Advance(4);
  EXPECT_EQ(ledger.Bank(), 4);
  EXPECT_EQ(ledger.Length(), 4u);
  ledger.Advance(3);
  EXPECT_EQ(ledger.Bank(), 2);
  EXPECT_EQ(ledger.Length(), 7u);
  EXPECT_THROW(ledger.Advance(5), std::logic_error);
}

TEST(Qwen38SuffixDrafterTest, ProposesMatchingNgramContinuations) {
  Qwen38SuffixDrafter drafter;
  const int32_t prompt[] = {10, 20, 30, 40, 50, 60};
  drafter.AppendSpan(std::span<const int32_t>(prompt, 6));
  drafter.AppendToken(10);
  drafter.AppendToken(20);
  int32_t drafts[3] = {-1, -1, -1};
  const int n = drafter.ProposeDrafts(30, 3, drafts, 3);
  ASSERT_EQ(n, 3);
  EXPECT_EQ(drafts[0], 40);
  EXPECT_EQ(drafts[1], 50);
  EXPECT_EQ(drafts[2], 60);
}

TEST(Qwen38GraphConfigTest, EffectivePrefillChunkIsOneSettingWithAPerPathDefault) {
  Config config;
  EXPECT_EQ(config.model.qwen38_graph.prefill.chunk, 0);
  EXPECT_EQ(EffectiveQwen38PrefillChunkTokens(config.model.qwen38_graph), 2048u);  // decode-graph steps

  OverlayConfig(config, R"({"model":{"qwen38_graph":{"prefill":{"chunk":512}}}})");
  EXPECT_EQ(EffectiveQwen38PrefillChunkTokens(config.model.qwen38_graph), 512u);

  OverlayConfig(config, R"({"model":{"qwen38_graph":{"prefill":{"enabled":true,"chunk":4096}}}})");
  EXPECT_EQ(EffectiveQwen38PrefillChunkTokens(config.model.qwen38_graph), 4096u);

  Config fast;
  OverlayConfig(fast, R"({"model":{"qwen38_graph":{"prefill":{"enabled":true}}}})");
  EXPECT_EQ(EffectiveQwen38PrefillChunkTokens(fast.model.qwen38_graph), 8192u);  // dense prefill

  Config explicit_default;
  OverlayConfig(explicit_default, R"({"model":{"qwen38_graph":{"prefill":{"chunk":0}}}})");
  EXPECT_EQ(EffectiveQwen38PrefillChunkTokens(explicit_default.model.qwen38_graph), 2048u);

  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"prefill":{"chunk":-2}}}})"), std::runtime_error);
}

TEST(Qwen38GraphConfigTest, PrefillChunkValidation) {
  Config config;
  config.model.qwen38_graph.kv_capacity = 4096;
  EXPECT_NO_THROW(ValidateQwen38PrefillChunk(config.model.qwen38_graph));  // default 2048, even, step path

  // Through the decode graphs every chunk but the last must keep the S=2 pairs aligned: even sizes only.
  config.model.qwen38_graph.prefill.chunk = 15;
  EXPECT_THROW(ValidateQwen38PrefillChunk(config.model.qwen38_graph), std::runtime_error);
  config.model.qwen38_graph.prefill.chunk = 16;
  EXPECT_NO_THROW(ValidateQwen38PrefillChunk(config.model.qwen38_graph));
  config.model.qwen38_graph.prefill.chunk = -1;
  EXPECT_THROW(ValidateQwen38PrefillChunk(config.model.qwen38_graph), std::runtime_error);

  // The dense prefill takes any size in [1, kv_capacity].
  config.model.qwen38_graph.prefill.enabled = true;
  config.model.qwen38_graph.prefill.chunk = 15;
  EXPECT_NO_THROW(ValidateQwen38PrefillChunk(config.model.qwen38_graph));
  config.model.qwen38_graph.prefill.chunk = 4096;
  EXPECT_NO_THROW(ValidateQwen38PrefillChunk(config.model.qwen38_graph));
  config.model.qwen38_graph.prefill.chunk = 4097;
  EXPECT_THROW(ValidateQwen38PrefillChunk(config.model.qwen38_graph), std::runtime_error);
  config.model.qwen38_graph.prefill.chunk = 0;  // the 8192 default does not fit a 4096 capacity
  EXPECT_THROW(ValidateQwen38PrefillChunk(config.model.qwen38_graph), std::runtime_error);
}

// The documented slot range is enforced where the key is parsed, with the key and the value in the message, not by a later
// resolution that would map the value to one slot or clamp it.
TEST(Qwen38GraphConfigTest, MaxBatchSizeMustLieInTheDocumentedSlotRange) {
  const auto parse = [](const std::string& value) {
    Config config;
    OverlayConfig(config, R"({"model":{"qwen38_graph":{"max_batch_size":)" + value + "}}}");
    return config.model.qwen38_graph.max_batch_size;
  };
  EXPECT_EQ(Config{}.model.qwen38_graph.max_batch_size, 1);
  EXPECT_EQ(parse("1"), 1);
  EXPECT_EQ(parse("2"), 2);
  EXPECT_EQ(parse("8"), Config::Model::Qwen38Graph::kMaxBatchSize);
  for (const char* value : {"-1", "0", "9", "16", "1000"}) {
    try {
      parse(value);
      ADD_FAILURE() << "max_batch_size " << value << " was accepted";
    } catch (const std::exception& error) {  // the JSON reader reports it with the position in the file
      const std::string message = error.what();
      EXPECT_NE(message.find("max_batch_size"), std::string::npos) << message;
      EXPECT_NE(message.find("[1, 8]"), std::string::npos) << message;
      EXPECT_NE(message.find(std::string{"got "} + value), std::string::npos) << message;
    }
  }
}

TEST(Qwen38GraphConfigTest, TheSlotCountIsResolvedFromItsSettingsAndNeverClampedSilently) {
  Config config;
  EXPECT_EQ(ResolveQwen38MaxBatchSize(config), 1u);
  config.model.qwen38_graph.max_batch_size = 4;
  EXPECT_EQ(ResolveQwen38MaxBatchSize(config), 4u);
  config.model.qwen38_graph.max_batch_size = Config::Model::Qwen38Graph::kMaxBatchSize;
  EXPECT_EQ(ResolveQwen38MaxBatchSize(config), 8u);

  // A programmatically built Config is checked like a parsed one: no value maps to one slot or to the maximum.
  for (const int value : {-1, 0, 9, 64}) {
    config.model.qwen38_graph.max_batch_size = value;
    try {
      ResolveQwen38MaxBatchSize(config);
      ADD_FAILURE() << "max_batch_size " << value << " was resolved";
    } catch (const std::out_of_range& error) {
      EXPECT_NE(std::string{error.what()}.find("model.qwen38_graph.max_batch_size"), std::string::npos) << error.what();
    }
  }

  // The engine-level settings take precedence, and they are held to the same range.
  Config dynamic;
  dynamic.model.qwen38_graph.max_batch_size = 2;
  dynamic.engine.dynamic_batching.emplace();
  dynamic.engine.dynamic_batching->max_batch_size = 3;
  EXPECT_EQ(ResolveQwen38MaxBatchSize(dynamic), 3u);
  dynamic.engine.dynamic_batching->max_batch_size = 16;  // the default of the dynamic engine
  EXPECT_THROW(ResolveQwen38MaxBatchSize(dynamic), std::out_of_range);
  dynamic.engine.dynamic_batching->max_batch_size = 0;
  EXPECT_THROW(ResolveQwen38MaxBatchSize(dynamic), std::out_of_range);

  Config fixed;
  fixed.engine.static_batching.emplace();
  EXPECT_EQ(ResolveQwen38MaxBatchSize(fixed), 4u);
  fixed.engine.static_batching->max_batch_size = 9;
  EXPECT_THROW(ResolveQwen38MaxBatchSize(fixed), std::out_of_range);

  Config search;
  search.search.batch_size = 3;
  EXPECT_EQ(ResolveQwen38MaxBatchSize(search), 3u);
  search.search.batch_size = 9;
  EXPECT_THROW(ResolveQwen38MaxBatchSize(search), std::out_of_range);
}

namespace {

// A folded package: prefill, the dense graph and the slot count are what the tests vary.
Config FoldedPackageConfig(const std::string& qwen38_graph_overlay) {
  Config config;
  OverlayConfig(config,
                R"({"model":{"type":"qwen38_flash_graph","decoder":{"filename":"model.onnx"},"qwen38_graph":)" +
                    qwen38_graph_overlay + "}}");
  return config;
}

void ExpectConfigDiagnostic(const Config& config, const std::string& needle) {
  try {
    ValidateQwen38MultiSlotConfig(config);
    ADD_FAILURE() << "the configuration was accepted, expected a diagnostic with '" << needle << "'";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find(needle), std::string::npos) << error.what();
  }
}

// Hand-encoded protobuf: the framing of the few ONNX fields the output-name reader walks (onnx.ModelProto.graph = 7,
// GraphProto.{node = 1, name = 2, initializer = 5, input = 11, output = 12, value_info = 13}, ValueInfoProto.{name = 1,
// type = 2}).
std::string ProtoVarint(uint64_t value) {
  std::string bytes;
  while (value >= 0x80) {
    bytes.push_back(static_cast<char>((value & 0x7F) | 0x80));
    value >>= 7;
  }
  bytes.push_back(static_cast<char>(value));
  return bytes;
}

std::string ProtoBytes(uint32_t field, const std::string& payload) {
  return ProtoVarint((uint64_t{field} << 3) | 2) + ProtoVarint(payload.size()) + payload;
}

std::string ProtoVarintField(uint32_t field, uint64_t value) {
  return ProtoVarint(uint64_t{field} << 3) + ProtoVarint(value);
}

std::string ProtoValueInfo(const std::string& name) {
  return ProtoBytes(1, name) + ProtoBytes(2, std::string(40, '\x08')) + ProtoBytes(3, "doc");
}

std::string OnnxModelBytes(const std::vector<std::string>& outputs, size_t initializer_bytes = 0) {
  std::string graph;
  graph += ProtoBytes(1, "node payload");
  graph += ProtoBytes(2, "main_graph");
  graph += ProtoBytes(5, std::string(initializer_bytes, '\x7f'));
  graph += ProtoBytes(11, ProtoValueInfo("input_ids"));
  for (const std::string& name : outputs) graph += ProtoBytes(12, ProtoValueInfo(name));
  graph += ProtoBytes(13, ProtoValueInfo("intermediate"));

  std::string model;
  model += ProtoVarintField(1, 9);  // ir_version
  model += ProtoBytes(2, "tests");  // producer_name
  model += ProtoBytes(7, graph);
  model += ProtoBytes(8, ProtoVarintField(2, 21));                // opset_import
  model += ProtoVarint((15u << 3) | 1) + std::string(8, '\x01');  // a fixed64 field
  model += ProtoVarint((16u << 3) | 5) + std::string(4, '\x02');  // a fixed32 field
  return model;
}

}  // namespace

// Several slots are decoded by the batched fold target and prefilled chunk by chunk through it. Everything else is refused when
// the model is created (source validation, no GPU), not at the first decode of an admitted request.
TEST(Qwen38MultiSlotConfigTest, ASingleSlotHasNoRestriction) {
  for (const char* overlay : {R"({"max_batch_size":1})",
                              R"({"max_batch_size":1,"fold":{"enabled":true},"prefill":{"enabled":false}})",
                              R"({"fold":{"enabled":true},"prefill":{"enabled":true,"dense":{"filename":"dense.onnx"}}})"}) {
    EXPECT_NO_THROW(ValidateQwen38MultiSlotConfig(FoldedPackageConfig(overlay))) << overlay;
  }
}

TEST(Qwen38MultiSlotConfigTest, SeveralSlotsNeedTheFoldStack) {
  ExpectConfigDiagnostic(FoldedPackageConfig(R"({"max_batch_size":2})"), "model.qwen38_graph.fold.enabled");
  ExpectConfigDiagnostic(FoldedPackageConfig(R"({"max_batch_size":8,"prefill":{"enabled":true}})"),
                         "model.qwen38_graph.fold.enabled");
}

TEST(Qwen38MultiSlotConfigTest, SeveralSlotsNeedTheUnifiedFastPrefill) {
  // prefill.enabled=false: the prompt would go through the decode graphs, which cannot address a slot.
  ExpectConfigDiagnostic(FoldedPackageConfig(R"({"max_batch_size":2,"fold":{"enabled":true}})"),
                         "unified fast prefill");
  ExpectConfigDiagnostic(FoldedPackageConfig(R"({"max_batch_size":2,"fold":{"enabled":true},"prefill":{"enabled":false}})"),
                         "prefill.enabled");
  // A separate dense prefill graph is not slot-addressed either.
  ExpectConfigDiagnostic(
      FoldedPackageConfig(R"({"max_batch_size":4,"fold":{"enabled":true},
                              "prefill":{"enabled":true,"dense":{"filename":"dense/model.onnx"}}})"),
      "separate prefill.dense.filename");
  EXPECT_TRUE(Qwen38HasSeparatePrefillSessions(
      FoldedPackageConfig(R"({"fold":{"enabled":true},"prefill":{"enabled":true,"dense":{"filename":"dense/model.onnx"}}})")));
}

TEST(Qwen38MultiSlotConfigTest, TheUnifiedFastPrefillOfTheFoldTargetServesEverySlotCount) {
  for (int slots = 2; slots <= Config::Model::Qwen38Graph::kMaxBatchSize; ++slots) {
    const std::string overlay = R"({"max_batch_size":)" + std::to_string(slots) +
                                R"(,"fold":{"enabled":true},"prefill":{"enabled":true}})";
    EXPECT_NO_THROW(ValidateQwen38MultiSlotConfig(FoldedPackageConfig(overlay))) << slots;
  }
  // Naming the decoder itself as the dense graph is still the unified prefill.
  const Config same = FoldedPackageConfig(
      R"({"max_batch_size":2,"fold":{"enabled":true},"prefill":{"enabled":true,"dense":{"filename":"model.onnx"}}})");
  EXPECT_FALSE(Qwen38HasSeparatePrefillSessions(same));
  EXPECT_NO_THROW(ValidateQwen38MultiSlotConfig(same));
}

TEST(Qwen38MultiSlotConfigTest, TheSlotCountOfTheEngineSettingsIsValidatedLikeTheModelKey) {
  Config config = FoldedPackageConfig(R"({"fold":{"enabled":true},"prefill":{"enabled":false}})");
  EXPECT_NO_THROW(ValidateQwen38MultiSlotConfig(config));  // one slot
  config.engine.static_batching.emplace();                 // four slots
  ExpectConfigDiagnostic(config, "unified fast prefill");
  config.engine.static_batching->max_batch_size = 9;
  EXPECT_THROW(ValidateQwen38MultiSlotConfig(config), std::out_of_range);
}

// A ring-bank target keeps one bank per row count for the whole batch, so it cannot restore the accepted prefix of every slot.
// The check names the output and runs before the engine binds anything.
TEST(Qwen38MultiSlotConfigTest, SeveralSlotsNeedACompactStateTarget) {
  const std::vector<std::string> compact = {"logits",
                                            "hidden_states_out",
                                            "next_tokens",
                                            "validation.aggregate",
                                            "aux_hidden_states",
                                            "fold.tail_done",
                                            "present_position_ids",
                                            "present.0.key",
                                            "present.1.recurrent_state",
                                            "state_update.1.conv_value",
                                            "state_update.1.recurrent_value"};
  for (size_t slots = 1; slots <= 8; ++slots) {
    EXPECT_NO_THROW(ValidateQwen38BatchedTargetOutputs(slots, compact)) << slots;
  }

  for (const char* ring_output : {"present_mid.0.key", "present_mid2.1.recurrent_state", "present_mid3.1.conv_state",
                                  "head.present_mid.0.key"}) {
    std::vector<std::string> ring = compact;
    ring.emplace_back(ring_output);
    EXPECT_NO_THROW(ValidateQwen38BatchedTargetOutputs(1, ring)) << ring_output;  // single-slot ring targets stay supported
    for (size_t slots = 2; slots <= 8; ++slots) {
      try {
        ValidateQwen38BatchedTargetOutputs(slots, ring);
        ADD_FAILURE() << slots << " slots accepted " << ring_output;
      } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find(ring_output), std::string::npos) << message;
        EXPECT_NE(message.find(std::to_string(slots) + " request slots"), std::string::npos) << message;
        EXPECT_NE(message.find("state_update"), std::string::npos) << message;
      }
    }
  }

  // No state_update.* outputs and no ring outputs: nothing restores the accepted prefix either.
  const std::vector<std::string> plain = {"logits", "hidden_states_out", "next_tokens", "present.0.key"};
  EXPECT_NO_THROW(ValidateQwen38BatchedTargetOutputs(1, plain));
  EXPECT_THROW(ValidateQwen38BatchedTargetOutputs(2, plain), std::runtime_error);
  EXPECT_THROW(ValidateQwen38BatchedTargetOutputs(2, {}), std::runtime_error);
}

// The same check runs on the model file before any weight is uploaded: the output names come from the protobuf framing of the
// graph, whose initializers are skipped, not parsed.
TEST(Qwen38MultiSlotConfigTest, TheOutputNamesOfTheModelFileAreReadWithoutParsingItsInitializers) {
  const auto dir = MakeScratchDir("onnx_outputs");
  const std::vector<std::string> outputs = {"logits", "present_mid.0.key", "state_update.1.conv_value", "next_tokens"};

  const auto small = WriteFile(dir / "small.onnx", OnnxModelBytes(outputs));
  ASSERT_TRUE(ReadQwen38OnnxOutputNames(small).has_value());
  EXPECT_EQ(*ReadQwen38OnnxOutputNames(small), outputs);  // the input and value_info entries are not outputs

  const auto big = WriteFile(dir / "big.onnx", OnnxModelBytes(outputs, size_t{3} << 20));
  ASSERT_GT(std::filesystem::file_size(big), size_t{3} << 20);
  EXPECT_EQ(ReadQwen38OnnxOutputNames(big), std::optional<std::vector<std::string>>{outputs});

  std::vector<std::string> many;
  for (int i = 0; i < 300; ++i) many.push_back("state_update." + std::to_string(i) + ".conv_value");
  EXPECT_EQ(ReadQwen38OnnxOutputNames(WriteFile(dir / "many.onnx", OnnxModelBytes(many, 1000))),
            std::optional<std::vector<std::string>>{many});

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38MultiSlotConfigTest, ARingBankTargetFileIsRefusedForSeveralSlotsBeforeAnyWeightIsUploaded) {
  const auto dir = MakeScratchDir("onnx_ring_target");
  const std::vector<std::string> compact = {"logits", "next_tokens", "present.0.key", "state_update.1.conv_value"};
  const auto compact_file = WriteFile(dir / "compact.onnx", OnnxModelBytes(compact, 4096));
  for (size_t slots = 1; slots <= 8; ++slots) EXPECT_NO_THROW(ValidateQwen38TargetModelFile(slots, compact_file)) << slots;

  for (const char* ring_output : {"present_mid.0.key", "present_mid2.1.recurrent_state", "present_mid3.1.conv_state"}) {
    std::vector<std::string> ring = compact;
    ring.emplace_back(ring_output);
    const auto ring_file = WriteFile(dir / "ring.onnx", OnnxModelBytes(ring, 4096));
    EXPECT_NO_THROW(ValidateQwen38TargetModelFile(1, ring_file)) << ring_output;
    for (size_t slots = 2; slots <= 8; ++slots) {
      try {
        ValidateQwen38TargetModelFile(slots, ring_file);
        ADD_FAILURE() << slots << " slots accepted " << ring_output;
      } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        EXPECT_NE(message.find(ring_output), std::string::npos) << message;
        EXPECT_NE(message.find(std::to_string(slots) + " request slots"), std::string::npos) << message;
      }
    }
  }

  const auto plain_file = WriteFile(dir / "plain.onnx", OnnxModelBytes({"logits", "present.0.key"}));
  EXPECT_THROW(ValidateQwen38TargetModelFile(2, plain_file), std::runtime_error);  // no state_update.* outputs

  // A single slot has no restriction and reads nothing: not even a file that does not exist.
  EXPECT_NO_THROW(ValidateQwen38TargetModelFile(1, dir / "missing.onnx"));

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// A file the reader cannot make sense of (an ORT-format model, a truncated or damaged one) is not an error of its own: the
// outputs of the session, which ORT loads and checks anyway, stay authoritative.
TEST(Qwen38MultiSlotConfigTest, AFileThatIsNotAnOnnxProtobufIsLeftToTheCheckAgainstTheSession) {
  const auto dir = MakeScratchDir("onnx_malformed");
  const std::string model = OnnxModelBytes({"logits", "present_mid.0.key"}, 256);

  std::map<std::string, std::string> files;
  files["empty"] = "";
  files["text"] = "this is not a model";
  files["ort_format"] = std::string("\x1c\x00\x00\x00ORTM\x00\x00\x00\x00", 12);
  files["truncated_in_the_graph"] = model.substr(0, model.size() / 2);
  files["truncated_in_a_varint"] = ProtoVarint(1u << 3) + ProtoVarint(7) + std::string(1, '\x80');
  files["no_graph"] = ProtoVarintField(1, 9) + ProtoBytes(2, "tests");
  files["a_graph_without_outputs"] = ProtoBytes(7, ProtoBytes(2, "main_graph") + ProtoBytes(11, ProtoValueInfo("x")));
  files["two_graphs"] = model + model;
  files["a_name_that_leaves_its_message"] =
      ProtoBytes(7, ProtoBytes(12, ProtoVarint((1u << 3) | 2) + ProtoVarint(100) + "short"));
  files["an_output_without_a_name"] = ProtoBytes(7, ProtoBytes(12, ProtoBytes(2, "type only")));
  files["a_group"] = ProtoBytes(7, ProtoBytes(12, ProtoValueInfo("logits")) + ProtoVarint((9u << 3) | 3));
  files["an_endless_varint"] = ProtoVarint((7u << 3) | 2) + std::string(12, '\xff');
  files["a_huge_value_info"] = ProtoBytes(7, ProtoBytes(12, ProtoBytes(2, std::string((size_t{1} << 20) + 1, '\x08'))));

  for (const auto& [label, bytes] : files) {
    const auto file = WriteFile(dir / (label + ".onnx"), bytes);
    EXPECT_FALSE(ReadQwen38OnnxOutputNames(file).has_value()) << label;
    EXPECT_NO_THROW(ValidateQwen38TargetModelFile(8, file)) << label;
  }
  EXPECT_FALSE(ReadQwen38OnnxOutputNames(dir / "missing.onnx").has_value());
  EXPECT_NO_THROW(ValidateQwen38TargetModelFile(8, dir / "missing.onnx"));
  EXPECT_FALSE(ReadQwen38OnnxOutputNames(dir).has_value());  // a directory

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

namespace {

// Creates the empty files of a package folder, relative to `root`.
void TouchPackageFiles(const std::filesystem::path& root, const std::vector<std::string>& files) {
  for (const std::string& file : files) {
    std::filesystem::create_directories((root / file).parent_path());
    WriteFile(root / file, "");
  }
}

// What the model constructor asks of the package folder: a regular file under the model folder.
std::function<bool(const std::string&)> FilesOf(const std::filesystem::path& root) {
  return [root](const std::string& file) {
    std::error_code error;
    return std::filesystem::is_regular_file(root / file, error);
  };
}

std::string ResolveTarget(const std::filesystem::path& root, const std::string& decoder, const std::string& context_request,
                          const std::string& mtp_drafts = "") {
  return ResolveQwen38TargetGraphFilename(decoder, context_request, mtp_drafts, FilesOf(root));
}

std::string ResolveTargetError(const std::filesystem::path& root, const std::string& decoder, const std::string& context_request) {
  try {
    const std::string chosen = ResolveTarget(root, decoder, context_request);
    ADD_FAILURE() << "'" << chosen << "' was chosen for the tier request '" << context_request << "'";
  } catch (const std::runtime_error& error) {
    return error.what();
  }
  return {};
}

void RemoveScratchDir(const std::filesystem::path& dir) {
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

}  // namespace

// The tier graphs of a package whose decoder sits in a sub-folder (target/model.onnx) are its siblings, not files of the model
// folder. Filesystem only: no GPU, no Windows.
TEST(Qwen38ContextTierTest, TierGraphsAreResolvedNextToTheConfiguredDecoder) {
  const auto dir = MakeScratchDir("tier_subfolder");
  TouchPackageFiles(dir, {"target/model.onnx", "target/model_8k.onnx", "target/model_16k.onnx", "target/model_dynamic_ctx.onnx",
                          "model_128k.onnx", "a/b/model.onnx", "a/b/model_2k.onnx"});

  EXPECT_EQ(ResolveTarget(dir, "target/model.onnx", "8k"), "target/model_8k.onnx");
  EXPECT_EQ(ResolveTarget(dir, "target/model.onnx", "8192"), "target/model_8k.onnx");
  EXPECT_EQ(ResolveTarget(dir, "target/model.onnx", "16K"), "target/model_16k.onnx");
  EXPECT_EQ(ResolveTarget(dir, "a/b/model.onnx", "2048"), "a/b/model_2k.onnx");
  // No graph of the tier: the dynamic-context graph next to the decoder. The model_128k.onnx of the model folder is not a sibling.
  EXPECT_EQ(ResolveTarget(dir, "target/model.onnx", "64k"), "target/model_dynamic_ctx.onnx");
  EXPECT_EQ(ResolveTarget(dir, "target/model.onnx", "128k"), "target/model_dynamic_ctx.onnx");
  // A decoder in the model folder keeps resolving there.
  EXPECT_EQ(ResolveTarget(dir, "model.onnx", "128k"), "model_128k.onnx");
  // The 32k tier is the package graph.
  EXPECT_EQ(ResolveTarget(dir, "target/model.onnx", "32k"), "target/model.onnx");
  EXPECT_EQ(ResolveTarget(dir, "target/model.onnx", "32768"), "target/model.onnx");

  // The callable decides what exists: a graph supplied from memory is as good as a file.
  const auto from_memory = [](const std::string& file) { return file == "target/model_8k.onnx"; };
  EXPECT_EQ(ResolveQwen38TargetGraphFilename("target/model.onnx", "8k", "", from_memory), "target/model_8k.onnx");
  RemoveScratchDir(dir);
}

// A 32K package need not contain the default graph, and the 32k tier is existence-checked like the others.
TEST(Qwen38ContextTierTest, TheThirtyTwoKTierFollowsTheFallbackChainOfTheOtherTiers) {
  for (const std::string& folder : {std::string{}, std::string{"target/"}}) {
    SCOPED_TRACE("folder '" + folder + "'");
    const std::string decoder = folder + "model.onnx";
    const auto dir = MakeScratchDir("tier_32k");

    // Only model_32k.onnx.
    TouchPackageFiles(dir, {folder + "model_32k.onnx"});
    EXPECT_EQ(ResolveTarget(dir, decoder, "32k"), folder + "model_32k.onnx");
    EXPECT_EQ(ResolveTarget(dir, decoder, "32768"), folder + "model_32k.onnx");
    EXPECT_EQ(ResolveTarget(dir, decoder, "32K"), folder + "model_32k.onnx");

    // Only model_dynamic_ctx.onnx.
    RemoveScratchDir(dir);
    std::filesystem::create_directories(dir);
    TouchPackageFiles(dir, {folder + "model_dynamic_ctx.onnx"});
    EXPECT_EQ(ResolveTarget(dir, decoder, "32k"), folder + "model_dynamic_ctx.onnx");
    EXPECT_EQ(ResolveTarget(dir, decoder, "8k"), folder + "model_dynamic_ctx.onnx");

    // The package graph is the 32K one, so it is preferred to the dynamic-context graph, but not to an explicit model_32k.onnx.
    TouchPackageFiles(dir, {decoder});
    EXPECT_EQ(ResolveTarget(dir, decoder, "32k"), decoder);
    EXPECT_EQ(ResolveTarget(dir, decoder, "8k"), folder + "model_dynamic_ctx.onnx");
    TouchPackageFiles(dir, {folder + "model_32k.onnx"});
    EXPECT_EQ(ResolveTarget(dir, decoder, "32k"), folder + "model_32k.onnx");

    // Only the decoder: the other tiers fall back to it as they always did.
    RemoveScratchDir(dir);
    std::filesystem::create_directories(dir);
    TouchPackageFiles(dir, {decoder});
    EXPECT_EQ(ResolveTarget(dir, decoder, "32k"), decoder);
    EXPECT_EQ(ResolveTarget(dir, decoder, "64k"), decoder);
    RemoveScratchDir(dir);
  }
}

TEST(Qwen38ContextTierTest, ARequestThatMatchesNoGraphIsAClearErrorNamingEveryTriedPath) {
  const auto dir = MakeScratchDir("tier_missing");
  TouchPackageFiles(dir, {"unrelated.onnx", "model_8k.onnx"});  // not next to target/model.onnx

  const std::string tier_8k = ResolveTargetError(dir, "target/model.onnx", "8k");
  for (const char* needle : {"ORT_QWEN38_CONTEXT_LENGTH=8k", "'target/model_8k.onnx'", "'target/model_dynamic_ctx.onnx'", "'target/model.onnx'"}) {
    EXPECT_NE(tier_8k.find(needle), std::string::npos) << needle << " in: " << tier_8k;
  }
  EXPECT_EQ(tier_8k.find("'model_8k.onnx'"), std::string::npos) << tier_8k;

  const std::string tier_32k = ResolveTargetError(dir, "model.onnx", "32768");
  for (const char* needle : {"ORT_QWEN38_CONTEXT_LENGTH=32768", "'32k'", "'model_32k.onnx'", "'model.onnx'", "'model_dynamic_ctx.onnx'"}) {
    EXPECT_NE(tier_32k.find(needle), std::string::npos) << needle << " in: " << tier_32k;
  }
  RemoveScratchDir(dir);
}

// Without a context request the decoder is the configured one: no file is looked up, so a missing decoder is still reported by
// the session creation, and an empty request is no request.
TEST(Qwen38ContextTierTest, WithoutARequestTheDecoderIsKeptAsConfigured) {
  std::vector<std::string> looked_up;
  const auto exists = [&looked_up](const std::string& file) {
    looked_up.push_back(file);
    return false;
  };
  for (const char* drafts : {"", "0", "4", "-1", "x"}) {
    EXPECT_EQ(ResolveQwen38TargetGraphFilename("target/model.onnx", "", drafts, exists), "target/model.onnx") << drafts;
    EXPECT_EQ(ResolveQwen38TargetGraphFilename("model.onnx", "", drafts, exists), "model.onnx") << drafts;
  }
  EXPECT_TRUE(looked_up.empty());
}

TEST(Qwen38ContextTierTest, TheMtpDraftsHookPicksTheGraphOfThatWidthNextToTheDecoderOrUnderTarget) {
  const auto dir = MakeScratchDir("tier_drafts");
  TouchPackageFiles(dir, {"model.onnx", "model_s3.onnx", "target/model_s3.onnx", "target/model_s2.onnx", "target/model.onnx",
                          "target/model_s4.onnx", "model_8k.onnx"});
  EXPECT_EQ(ResolveTarget(dir, "model.onnx", "", "2"), "model_s3.onnx");
  EXPECT_EQ(ResolveTarget(dir, "model.onnx", "", "1"), "target/model_s2.onnx");  // no model_s2.onnx next to the decoder
  EXPECT_EQ(ResolveTarget(dir, "model.onnx", "", "3"), "target/model_s4.onnx");
  EXPECT_EQ(ResolveTarget(dir, "target/model.onnx", "", "2"), "target/model_s3.onnx");
  // No graph of that width: the decoder.
  std::filesystem::remove(dir / "target/model_s4.onnx");
  EXPECT_EQ(ResolveTarget(dir, "model.onnx", "", "3"), "model.onnx");
  EXPECT_EQ(ResolveTarget(dir, "target/model.onnx", "", "3"), "target/model.onnx");
  // A context request takes precedence over the hook.
  EXPECT_EQ(ResolveTarget(dir, "model.onnx", "8k", "2"), "model_8k.onnx");
  EXPECT_EQ(ResolveTarget(dir, "model.onnx", "", "2"), "model_s3.onnx");
  RemoveScratchDir(dir);
}

TEST(Qwen38ContextTierTest, ARequestIsLowerCasedAndTheTierCountsFoldToTheirNames) {
  EXPECT_EQ(NormalizeQwen38ContextTier(""), "");
  const std::vector<std::pair<std::string, std::string>> tiers = {
      {"2k", "2k"}, {"8K", "8k"}, {"128K", "128k"}, {"2048", "2k"}, {"8192", "8k"}, {"16384", "16k"}, {"32768", "32k"}, {"65536", "64k"}, {"131072", "128k"}, {"4096", "4096"}, {"Dynamic", "dynamic"}};
  for (const auto& [request, tier] : tiers) EXPECT_EQ(NormalizeQwen38ContextTier(request), tier) << request;

  // A tier becomes part of a file name: nothing that could leave the folder or name another file is accepted.
  for (const char* request : {"../8k", "8 k", "8k\n", "model_8k.onnx", "8k/", "8k\\", "k\xc3\xa9"}) {
    EXPECT_THROW(NormalizeQwen38ContextTier(request), std::invalid_argument) << request;
  }
  bool looked_up = false;
  const auto exists = [&looked_up](const std::string&) {
    looked_up = true;
    return true;
  };
  EXPECT_THROW(ResolveQwen38TargetGraphFilename("model.onnx", "../8k", "", exists), std::invalid_argument);
  EXPECT_FALSE(looked_up);
}

// The pre-upload reader of the output names must see the graph that CreateSession opens: the constructor resolves the file once
// and hands the one name to both.
TEST(Qwen38ContextTierTest, TheOutputNamesAreReadFromTheGraphThatTheTierSelects) {
  const auto dir = MakeScratchDir("tier_reader");
  const std::vector<std::string> compact = {"logits", "next_tokens", "present.0.key", "state_update.1.conv_value"};
  const std::vector<std::string> ring = {"logits", "next_tokens", "present.0.key", "present_mid.0.key"};
  std::filesystem::create_directories(dir / "target");
  WriteFile(dir / "target/model.onnx", OnnxModelBytes(ring, 64));
  WriteFile(dir / "target/model_8k.onnx", OnnxModelBytes(compact, 64));

  const std::string decoder = ResolveTarget(dir, "target/model.onnx", "");
  EXPECT_EQ(decoder, "target/model.onnx");
  EXPECT_THROW(ValidateQwen38TargetModelFile(2, dir / decoder), std::runtime_error);  // a ring-bank graph cannot serve 2 slots

  const std::string tier = ResolveTarget(dir, "target/model.onnx", "8k");
  EXPECT_EQ(tier, "target/model_8k.onnx");
  EXPECT_NO_THROW(ValidateQwen38TargetModelFile(2, dir / tier));
  ASSERT_TRUE(ReadQwen38OnnxOutputNames(dir / tier).has_value());
  EXPECT_EQ(*ReadQwen38OnnxOutputNames(dir / tier), compact);
  RemoveScratchDir(dir);
}

// Synchronizing the capacity of the target graph is distinct from enlarging the generation budget of a request. Source checks,
// no Engine allocation run: Engine::CreateRequest takes the budget from search.max_length.
TEST(Qwen38ContextSyncTest, WithoutAContextRequestASmallerLimitIsKeptAndALargerOneIsClamped) {
  const int64_t positions = 33032;  // the 32K graph
  const Qwen38GraphContext limited = SyncQwen38GraphContext(positions, 4096, false);
  EXPECT_EQ(limited.kv_capacity, 33032);
  EXPECT_EQ(limited.context_length, 33024);
  EXPECT_EQ(limited.max_length, 4096);  // not 33024

  EXPECT_EQ(SyncQwen38GraphContext(positions, 1, false).max_length, 1);
  EXPECT_EQ(SyncQwen38GraphContext(positions, 33023, false).max_length, 33023);
  EXPECT_EQ(SyncQwen38GraphContext(positions, 33024, false).max_length, 33024);
  for (const int limit : {33025, 33032, 40000, 131072, std::numeric_limits<int>::max()}) {
    const Qwen38GraphContext clamped = SyncQwen38GraphContext(positions, limit, false);
    EXPECT_EQ(clamped.max_length, 33024) << limit;
    EXPECT_EQ(clamped.kv_capacity, 33032) << limit;
    EXPECT_EQ(clamped.context_length, 33024) << limit;
  }
  // A missing limit has nothing to keep: the whole context.
  for (const int limit : {0, -1}) EXPECT_EQ(SyncQwen38GraphContext(positions, limit, false).max_length, 33024) << limit;
}

TEST(Qwen38ContextSyncTest, ARequestedContextTierTakesTheWholeContextAsDocumented) {
  for (const int limit : {0, 1, 4096, 33024, 40000}) {
    const Qwen38GraphContext context = SyncQwen38GraphContext(33032, limit, true);
    EXPECT_EQ(context.kv_capacity, 33032) << limit;
    EXPECT_EQ(context.context_length, 33024) << limit;
    EXPECT_EQ(context.max_length, 33024) << limit;
  }
}

TEST(Qwen38ContextSyncTest, TheCapacityFollowsEveryGraphWhateverTheLimit) {
  for (const int64_t positions : {2056, 8200, 16392, 33032, 65544, 131080}) {
    const int context_length = static_cast<int>(positions) - kQwen38GraphIdMaxRows;
    for (const bool requested : {false, true}) {
      for (const int limit : {0, 4096, 70000}) {
        const Qwen38GraphContext context = SyncQwen38GraphContext(positions, limit, requested);
        EXPECT_EQ(context.kv_capacity, positions) << positions;
        EXPECT_EQ(context.context_length, context_length) << positions;
        EXPECT_LE(context.max_length, context.context_length) << positions << " " << limit << " " << requested;
        const int expected = requested || limit == 0 ? context_length : std::min(limit, context_length);
        EXPECT_EQ(context.max_length, expected) << positions << " " << limit << " " << requested;
      }
    }
  }
  // A graph no wider than the verify rows still leaves one position of context; one that has no positions is no graph.
  for (const int64_t positions : {1, 8, 9}) {
    EXPECT_EQ(SyncQwen38GraphContext(positions, 4096, false).kv_capacity, positions);
    EXPECT_EQ(SyncQwen38GraphContext(positions, 4096, false).context_length, std::max<int64_t>(1, positions - kQwen38GraphIdMaxRows));
  }
  EXPECT_THROW(SyncQwen38GraphContext(0, 4096, false), std::invalid_argument);
  EXPECT_THROW(SyncQwen38GraphContext(-1, 4096, true), std::invalid_argument);
  EXPECT_EQ(SyncQwen38GraphContext(int64_t{1} << 40, 4096, false).kv_capacity, std::numeric_limits<int>::max());
}

// The DFlash drafter's prompt-feature buffer gives every slot Qwen38DflashPromptSlotRows rows. The warm-up of the drafter views
// verify_rows rows of every slot from it, so a prefill.chunk below the verify width (chunk 4 against S=8) must not shrink it.
TEST(Qwen38DflashPromptBufferTest, EverySlotOwnsAtLeastAChunkAndTheVerifyWidthOfRows) {
  EXPECT_EQ(Qwen38DflashPromptSlotRows(4, 8), 8u);
  EXPECT_EQ(Qwen38DflashPromptSlotRows(1, 2), 2u);
  EXPECT_EQ(Qwen38DflashPromptSlotRows(8, 8), 8u);
  EXPECT_EQ(Qwen38DflashPromptSlotRows(9, 8), 9u);
  EXPECT_EQ(Qwen38DflashPromptSlotRows(2048, 8), 2048u);
  for (const int chunk : {1, 2, 3, 4, 7, 8, 9, 16, 512, 2048}) {
    for (int verify_rows = 2; verify_rows <= 8; ++verify_rows) {
      const size_t slot_rows = Qwen38DflashPromptSlotRows(chunk, verify_rows);
      EXPECT_GE(slot_rows, static_cast<size_t>(chunk));
      EXPECT_GE(slot_rows, static_cast<size_t>(verify_rows));
      for (size_t batch = 1; batch <= 8; ++batch) {
        // The rows of the warm-up view (batch x verify_rows) fit the buffer (batch x slot_rows).
        EXPECT_LE(batch * static_cast<size_t>(verify_rows), batch * slot_rows) << chunk << "/" << verify_rows << "/" << batch;
      }
    }
  }
  EXPECT_THROW(Qwen38DflashPromptSlotRows(0, 8), std::invalid_argument);
  EXPECT_THROW(Qwen38DflashPromptSlotRows(4, 0), std::invalid_argument);
  EXPECT_THROW(Qwen38DflashPromptSlotRows(-1, 8), std::invalid_argument);
}

// Reset() and ResetSlot(0) forget the prompt of the request the slot served, so a long-lived engine holds the prompt of the
// request in flight only and its host memory does not grow with the number of tokens served.
TEST(Qwen38PromptTokensTest, ForgettingAPromptDropsItsTokensAndTheBufferOfALongOne) {
  std::vector<int32_t> tokens;
  Qwen38ForgetPromptTokens(tokens);  // nothing to forget
  EXPECT_TRUE(tokens.empty());

  // A request whose prompt arrives in 8-token chunks.
  for (int chunk = 0; chunk < 100; ++chunk) tokens.insert(tokens.end(), 8, chunk);
  ASSERT_EQ(tokens.size(), 800u);
  const size_t short_capacity = tokens.capacity();
  Qwen38ForgetPromptTokens(tokens);
  EXPECT_TRUE(tokens.empty());
  EXPECT_EQ(tokens.capacity(), short_capacity) << "a short prompt keeps its buffer for the next request";

  // The next request starts from an empty prompt, not from the previous one.
  tokens.insert(tokens.end(), {1, 2, 3});
  EXPECT_EQ(tokens, (std::vector<int32_t>{1, 2, 3}));
  Qwen38ForgetPromptTokens(tokens);

  tokens.assign(4 * kQwen38RetainedPromptTokens, 7);
  ASSERT_GT(tokens.capacity(), kQwen38RetainedPromptTokens);
  Qwen38ForgetPromptTokens(tokens);
  EXPECT_TRUE(tokens.empty());
  EXPECT_LE(tokens.capacity(), kQwen38RetainedPromptTokens) << "the buffer of a long prompt is given back";

  // Many requests, none longer than the retained size: the allocation stays where it is.
  for (int request = 0; request < 1000; ++request) {
    tokens.assign(kQwen38RetainedPromptTokens, request);
    Qwen38ForgetPromptTokens(tokens);
    EXPECT_TRUE(tokens.empty());
    EXPECT_LE(tokens.capacity(), kQwen38RetainedPromptTokens);
  }
}

// A device with several slots keeps the validation bits of a decode in the slot that ran it. Its decode status is the status of
// every slot that started a decode, slot 0 or not, and a slot that never started is not even read.
namespace {

struct ScriptedStatusBits {
  std::map<size_t, int64_t> bits;
  std::vector<size_t> read;
  std::function<int64_t(size_t)> Reader() {
    return [this](size_t slot) {
      read.push_back(slot);
      const auto found = bits.find(slot);
      return found == bits.end() ? int64_t{0} : found->second;
    };
  }
};

std::string StatusMessage(std::span<const bool> started, const std::function<int64_t(size_t)>& reader) {
  try {
    CheckQwen38DecodeStatus(started, reader);
  } catch (const std::runtime_error& e) {
    return e.what();
  }
  return {};
}

}  // namespace

TEST(Qwen38DecodeStatusTest, OnlySlotsThatStartedADecodeAreReadAndCleanBitsPass) {
  ScriptedStatusBits device;
  const std::array<bool, 4> started{true, true, false, true};
  const auto reader = device.Reader();
  EXPECT_NO_THROW(CheckQwen38DecodeStatus(started, reader));
  EXPECT_EQ(device.read, (std::vector<size_t>{0, 1, 3}));

  device.read.clear();
  device.bits[2] = 8;  // bits of a slot that never started a decode (stale memory) are not looked at
  EXPECT_NO_THROW(CheckQwen38DecodeStatus(started, reader));
  EXPECT_EQ(device.read, (std::vector<size_t>{0, 1, 3}));

  const std::array<bool, 3> none{false, false, false};
  device.read.clear();
  EXPECT_NO_THROW(CheckQwen38DecodeStatus(none, reader));
  EXPECT_TRUE(device.read.empty());
  EXPECT_NO_THROW(CheckQwen38DecodeStatus(std::span<const bool>{}, reader));
}

TEST(Qwen38DecodeStatusTest, ErrorBitsOfANonZeroSlotAreReportedAndNameTheSlot) {
  const std::array<bool, 4> started{true, true, false, true};
  for (const size_t failing : {size_t{1}, size_t{3}}) {
    ScriptedStatusBits device;
    device.bits[failing] = 4;
    const std::string message = StatusMessage(started, device.Reader());
    EXPECT_NE(message.find("slot " + std::to_string(failing) + "=4"), std::string::npos) << message;
    EXPECT_NE(message.find("device error bits"), std::string::npos) << message;
    EXPECT_EQ(message.find("slot 0"), std::string::npos) << message;
  }
}

TEST(Qwen38DecodeStatusTest, EverySlotWithErrorBitsIsNamedAndEveryStartedSlotIsStillRead) {
  ScriptedStatusBits device;
  device.bits[0] = 1;
  device.bits[3] = 6;
  const std::array<bool, 4> started{true, true, false, true};
  const std::string message = StatusMessage(started, device.Reader());
  EXPECT_NE(message.find("slot 0=1"), std::string::npos) << message;
  EXPECT_NE(message.find("slot 3=6"), std::string::npos) << message;
  EXPECT_EQ(message.find("slot 1"), std::string::npos) << message;
  EXPECT_EQ(device.read, (std::vector<size_t>{0, 1, 3}));
}

TEST(Qwen38DecodeStatusTest, ADeviceWithASingleSlotKeepsItsMessage) {
  ScriptedStatusBits device;
  device.bits[0] = 2;
  const std::array<bool, 1> started{true};
  EXPECT_EQ(StatusMessage(started, device.Reader()), "qwen38_graph: device error bits 2 after decode");

  const std::array<bool, 1> not_started{false};
  device.read.clear();
  EXPECT_EQ(StatusMessage(not_started, device.Reader()), "");  // nothing to check before the decode started
  EXPECT_TRUE(device.read.empty());
}

TEST(Qwen38GraphConfigTest, DflashWarmupAtModelCreationHonorsTheWarmupSetting) {
  // warmup=false: the engine runs no forward while the model is created, neither the drafter's nor the target's.
  Config off;
  OverlayConfig(off, R"({"model":{"type":"qwen38_flash_graph","qwen38_graph":{"warmup":false}}})");
  ASSERT_FALSE(off.model.qwen38_graph.warmup);
  const Qwen38DflashWarmupPlan skipped = PlanQwen38DflashWarmup(off.model.qwen38_graph.warmup, 8, 2);
  EXPECT_FALSE(skipped.drafter);
  EXPECT_EQ(skipped.target_tokens, 0u);

  // The default warms up: twelve S=8 target forwards (every ring bank of the compact replay, enough runs to capture each).
  Config on;
  ASSERT_TRUE(on.model.qwen38_graph.warmup);
  const Qwen38DflashWarmupPlan s8 = PlanQwen38DflashWarmup(on.model.qwen38_graph.warmup, 8, 2);
  EXPECT_TRUE(s8.drafter);
  EXPECT_EQ(s8.target_tokens, 8u * 2u * static_cast<size_t>(kQwen38WarmupRunsPerGraph));
  EXPECT_EQ(s8.target_tokens / 8, 12u);

  const Qwen38DflashWarmupPlan s4 = PlanQwen38DflashWarmup(true, 4, 5);
  EXPECT_TRUE(s4.drafter);
  EXPECT_EQ(s4.target_tokens, 4u * 5u * static_cast<size_t>(kQwen38WarmupRunsPerGraph));

  // A pipeline without verify rows or ring banks has nothing to warm up.
  EXPECT_FALSE(PlanQwen38DflashWarmup(true, 0, 2).drafter);
  EXPECT_EQ(PlanQwen38DflashWarmup(true, 8, 0).target_tokens, 0u);
}

namespace {

using DflashChunk = Qwen38DflashPromptFeatures::Chunk;

struct DflashPromptRun {
  std::vector<DflashChunk> ingested;  // what the drafter is given while the prompt is prefilled, in order
  std::optional<DflashChunk> last;    // what decoding takes when it begins
};

// What Qwen38GraphEngine::FastAppend does with the features of a prompt that arrives in calls of `call_rows` tokens, each
// run as forwards of at most `forward_rows`. `announced` is the prompt length the executor announces up front, 0 when it is
// not: a call then knows only the end of its own tokens.
DflashPromptRun RunDflashPrompt(Qwen38DflashPromptFeatures& features, size_t prompt, size_t call_rows, size_t forward_rows,
                                size_t announced) {
  DflashPromptRun run;
  for (size_t base = 0; base < prompt; base += call_rows) {
    const size_t n = prompt - base < call_rows ? prompt - base : call_rows;
    const size_t total_expected = base + n > announced ? base + n : announced;
    if (const auto saved = features.BeginCall(total_expected)) run.ingested.push_back(*saved);
    for (size_t lo = 0; lo < n;) {
      const size_t m = n - lo < forward_rows ? n - lo : forward_rows;
      if (const auto ingest = features.ChunkDone({base + lo, m}, lo + m == n, total_expected)) run.ingested.push_back(*ingest);
      lo += m;
    }
  }
  run.last = features.TakeSavedChunk();
  return run;
}

// The drafter has the last `window` positions of the prompt without a gap: what it ingested and the chunk decoding takes
// follow one another and end at the end of the prompt, and the first of them starts no later than the window does.
void ExpectDrafterHasTheWindow(const DflashPromptRun& run, size_t prompt, size_t window) {
  ASSERT_TRUE(run.last.has_value());
  EXPECT_EQ(run.last->position + run.last->rows, prompt);
  const size_t first = run.ingested.empty() ? run.last->position : run.ingested.front().position;
  size_t next = first;
  for (const DflashChunk& chunk : run.ingested) {
    EXPECT_EQ(chunk.position, next);
    next = chunk.position + chunk.rows;
  }
  EXPECT_EQ(run.last->position, next);
  EXPECT_LE(first, prompt > window ? prompt - window : size_t{0});
}

}  // namespace

TEST(Qwen38DflashPromptFeaturesTest, WithoutAWindowEveryChunkButTheLastIsIngestedInOrder) {
  Qwen38DflashPromptFeatures features;  // the drafter attends to the whole prompt
  const DflashPromptRun run = RunDflashPrompt(features, 1000, 256, 128, 0);
  EXPECT_EQ(run.ingested.size(), 7u);
  ExpectDrafterHasTheWindow(run, 1000, 1000);
  ASSERT_TRUE(run.last.has_value());
  EXPECT_EQ(run.last->position, 896u);
  EXPECT_EQ(run.last->rows, 104u);

  // A window larger than the prompt is the same.
  Qwen38DflashPromptFeatures wide(5000);
  EXPECT_EQ(RunDflashPrompt(wide, 1000, 256, 128, 1000).ingested.size(), 7u);
}

TEST(Qwen38DflashPromptFeaturesTest, ChunksBeforeTheWindowAreSkippedByAbsolutePosition) {
  // The window of 100 positions at the end of a 1000-token prompt is [900, 1000).
  Qwen38DflashPromptFeatures features(100);
  EXPECT_FALSE(features.ChunkDone({700, 100}, false, 1000).has_value());
  EXPECT_FALSE(features.ChunkDone({800, 100}, false, 1000).has_value());  // [800, 900) ends where the window begins
  const auto reaching = features.ChunkDone({801, 100}, false, 1000);      // [801, 901) holds position 900
  ASSERT_TRUE(reaching.has_value());
  EXPECT_EQ(reaching->position, 801u);
  EXPECT_EQ(reaching->rows, 100u);

  // A prompt length below the end of a chunk (it is never announced so) is not an unsigned wrap to "far before the window".
  Qwen38DflashPromptFeatures unsure(50);
  EXPECT_TRUE(unsure.ChunkDone({0, 500}, false, 100).has_value());
}

TEST(Qwen38DflashPromptFeaturesTest, IngestionStaysContiguousOnceItHasBegun) {
  Qwen38DflashPromptFeatures features(100);
  ASSERT_TRUE(features.ChunkDone({900, 50}, false, 1000).has_value());
  // The prompt turns out longer than it was known to be. Chunks that now fall before the window are still ingested, as the
  // context of the drafter would otherwise have a gap.
  EXPECT_TRUE(features.ChunkDone({950, 50}, false, 1200).has_value());
  EXPECT_TRUE(features.ChunkDone({1000, 50}, false, 1200).has_value());
}

TEST(Qwen38DflashPromptFeaturesTest, ALongPromptIngestsOnlyItsWindowHoweverTheExecutorSplitsIt) {
  // The window is sliding_window 512 plus block 16. The executor calls with prefill chunks (256 tokens here, one forward
  // per call), and every call ends with a chunk saved for the next one: ingesting that unconditionally began the ingestion
  // of the whole history.
  constexpr size_t kPrompt = 4096;
  constexpr size_t kWindow = 528;
  constexpr size_t kForward = 256;
  for (const size_t call_rows : {size_t{256}, size_t{512}, size_t{1024}, size_t{4096}}) {
    SCOPED_TRACE("call_rows=" + std::to_string(call_rows));
    Qwen38DflashPromptFeatures features(kWindow);
    const DflashPromptRun run = RunDflashPrompt(features, kPrompt, call_rows, kForward, kPrompt);
    // [3328, 3584) and [3584, 3840) reach into the window [3568, 4096); the chunk before them ends at 3328, outside it.
    ASSERT_EQ(run.ingested.size(), 2u);
    EXPECT_EQ(run.ingested[0].position, 3328u);
    EXPECT_EQ(run.ingested[1].position, 3584u);
    ASSERT_TRUE(run.last.has_value());
    EXPECT_EQ(run.last->position, 3840u);
    EXPECT_EQ(run.last->rows, 256u);
    ExpectDrafterHasTheWindow(run, kPrompt, kWindow);
  }
}

TEST(Qwen38DflashPromptFeaturesTest, APromptLengthThatIsNotAnnouncedNeverDropsWhatTheWindowNeeds) {
  // Without ExpectPrefillTokens a call knows the end of its own tokens, a lower bound of the prompt length: a chunk it
  // drops is before the window for sure, one it keeps may reach back further than needed.
  constexpr size_t kWindow = 528;
  for (const size_t prompt : {size_t{300}, size_t{1000}, size_t{3000}, size_t{4097}}) {
    for (const size_t call_rows : {size_t{100}, size_t{256}, size_t{700}, size_t{1024}}) {
      for (const size_t forward_rows : {size_t{64}, size_t{256}, size_t{1024}}) {
        SCOPED_TRACE("prompt=" + std::to_string(prompt) + " call_rows=" + std::to_string(call_rows) +
                     " forward_rows=" + std::to_string(forward_rows));
        Qwen38DflashPromptFeatures features(kWindow);
        ExpectDrafterHasTheWindow(RunDflashPrompt(features, prompt, call_rows, forward_rows, 0), prompt, kWindow);
      }
    }
  }

  // Calls at least as large as the window prove with their own length that the earlier chunks are outside it, so only the
  // chunk next to the last one is ingested here.
  Qwen38DflashPromptFeatures features(kWindow);
  const DflashPromptRun run = RunDflashPrompt(features, 3000, 700, 700, 0);
  ASSERT_EQ(run.ingested.size(), 1u);
  EXPECT_EQ(run.ingested[0].position, 2100u);
}

TEST(Qwen38DflashPromptFeaturesTest, TheLastChunkOfACallIsSavedForTheNextCallOrForDecoding) {
  Qwen38DflashPromptFeatures features(100);
  // Decoding takes the last chunk of the prompt whatever the window says, far before it here.
  EXPECT_FALSE(features.ChunkDone({0, 10}, true, 10000).has_value());
  const auto saved = features.TakeSavedChunk();
  ASSERT_TRUE(saved.has_value());
  EXPECT_EQ(saved->position, 0u);
  EXPECT_EQ(saved->rows, 10u);
  EXPECT_FALSE(features.TakeSavedChunk().has_value());
}

TEST(Qwen38DflashPromptFeaturesTest, ASavedChunkIsIngestedOnceByTheNextCallAndDroppedBeforeTheWindow) {
  Qwen38DflashPromptFeatures features(100);
  EXPECT_FALSE(features.ChunkDone({0, 100}, true, 100).has_value());
  EXPECT_FALSE(features.BeginCall(1000).has_value());   // [0, 100) is 900 positions short of the end
  EXPECT_FALSE(features.TakeSavedChunk().has_value());  // and nothing stale is left behind

  EXPECT_FALSE(features.ChunkDone({850, 100}, true, 1000).has_value());
  const auto reaching = features.BeginCall(1000);  // [850, 950) is within the window [900, 1000)
  ASSERT_TRUE(reaching.has_value());
  EXPECT_EQ(reaching->position, 850u);
  EXPECT_EQ(reaching->rows, 100u);
  EXPECT_FALSE(features.TakeSavedChunk().has_value());  // the next call ingests a chunk once
}

TEST(Qwen38DflashPromptFeaturesTest, ResetStartsTheNextTurnFromScratch) {
  Qwen38DflashPromptFeatures features(528);
  const DflashPromptRun first = RunDflashPrompt(features, 4096, 256, 256, 4096);
  ASSERT_EQ(first.ingested.size(), 2u);

  // A turn that ends with a chunk saved and ingestion under way leaves neither behind.
  features.ChunkDone({3840, 256}, true, 4096);
  features.Reset();
  EXPECT_FALSE(features.TakeSavedChunk().has_value());
  const DflashPromptRun second = RunDflashPrompt(features, 4096, 256, 256, 4096);
  ASSERT_EQ(second.ingested.size(), first.ingested.size());
  EXPECT_EQ(second.ingested.front().position, first.ingested.front().position);
}

TEST(Qwen38GraphIdTest, IdsAreDerivedFromTheRunAndAreDistinct) {
  std::set<int> seen;
  const Qwen38GraphRun runs[] = {Qwen38GraphRun::kTarget, Qwen38GraphRun::kHead, Qwen38GraphRun::kHeadR2};
  ASSERT_EQ(std::size(runs), static_cast<size_t>(kQwen38GraphRunCount));
  for (const Qwen38GraphRun run : runs) {
    for (int rows = 1; rows <= kQwen38GraphIdMaxRows; ++rows) {
      for (int phase = 0; phase < kQwen38GraphIdMaxPhases; ++phase) {
        const int id = Qwen38GraphAnnotationId(run, rows, phase);
        EXPECT_GT(id, 0);                                          // 0 is ORT's default annotation, -1 disables capture
        EXPECT_EQ(id, Qwen38GraphAnnotationId(run, rows, phase));  // a pure function of its arguments
        EXPECT_TRUE(seen.insert(id).second) << "duplicate id " << id;
      }
    }
  }
  EXPECT_EQ(seen.size(), static_cast<size_t>(kQwen38GraphRunCount) * kQwen38GraphIdMaxRows * kQwen38GraphIdMaxPhases);
}

TEST(Qwen38GraphIdTest, EveryVerifyWidthUpToSevenDraftsHasAnIdPerRingPhase) {
  // S=8 verification (seven drafts) runs the target with 8 rows; the S=1 .. S=7 widths stay available too.
  EXPECT_GE(kQwen38GraphIdMaxRows, 8);
  std::set<int> target_ids;
  for (int rows = 1; rows <= 8; ++rows) {
    for (int phase = 0; phase < 5; ++phase) {  // the non-compact fold rings up to five banks, the compact one two
      EXPECT_TRUE(target_ids.insert(Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, rows, phase)).second);
    }
  }
  // Rows or phases of one run never alias another run's ids.
  EXPECT_NE(Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, 8, kQwen38GraphIdMaxPhases - 1),
            Qwen38GraphAnnotationId(Qwen38GraphRun::kHead, 1, 0));
  EXPECT_LT(Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, 8, kQwen38GraphIdMaxPhases - 1),
            Qwen38GraphAnnotationId(Qwen38GraphRun::kHead, 1, 0));
}

TEST(Qwen38GraphIdTest, RejectsRunsRowsAndPhasesBeyondTheLimits) {
  EXPECT_THROW(Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, 0, 0), std::out_of_range);
  EXPECT_THROW(Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, kQwen38GraphIdMaxRows + 1, 0), std::out_of_range);
  EXPECT_THROW(Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, 9, 0), std::out_of_range);
  EXPECT_THROW(Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, 1, kQwen38GraphIdMaxPhases), std::out_of_range);
  EXPECT_THROW(Qwen38GraphAnnotationId(Qwen38GraphRun::kHead, 1, -1), std::out_of_range);
  EXPECT_THROW(Qwen38GraphAnnotationId(Qwen38GraphRun::kHeadR2, 2, kQwen38GraphIdMaxPhases), std::out_of_range);
  EXPECT_THROW(Qwen38GraphAnnotationId(static_cast<Qwen38GraphRun>(kQwen38GraphRunCount), 1, 0), std::out_of_range);
  EXPECT_THROW(Qwen38GraphAnnotationId(static_cast<Qwen38GraphRun>(-1), 1, 0), std::out_of_range);
}

// The batched runs of the fold target (B = 2 .. 8 request slots) have ids of their own: none is shared with a single-slot run
// or with a run of another width, and the ids of a single slot do not depend on the batch dimension.
TEST(Qwen38GraphIdTest, BatchedTargetRunsHaveTheirOwnIds) {
  std::set<int> seen;
  for (const Qwen38GraphRun run : {Qwen38GraphRun::kTarget, Qwen38GraphRun::kHead, Qwen38GraphRun::kHeadR2}) {
    for (int rows = 1; rows <= kQwen38GraphIdMaxRows; ++rows) {
      for (int phase = 0; phase < kQwen38GraphIdMaxPhases; ++phase) {
        const int single = Qwen38GraphAnnotationId(run, rows, phase);
        EXPECT_EQ(single, Qwen38GraphAnnotationId(run, rows, phase, 1));
        EXPECT_TRUE(seen.insert(single).second);
      }
    }
  }
  const size_t single_slot_ids = seen.size();
  for (int batch = 2; batch <= kQwen38GraphIdMaxBatch; ++batch) {
    for (int rows = 1; rows <= kQwen38GraphIdMaxRows; ++rows) {
      for (int phase = 0; phase < kQwen38GraphIdMaxPhases; ++phase) {
        const int id = Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, rows, phase, batch);
        EXPECT_GT(id, 0);
        EXPECT_EQ(id, Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, rows, phase, batch));
        EXPECT_TRUE(seen.insert(id).second) << "duplicate id " << id << " for batch " << batch;
      }
    }
  }
  EXPECT_EQ(seen.size() - single_slot_ids,
            static_cast<size_t>(kQwen38GraphIdMaxBatch - 1) * kQwen38GraphIdMaxRows * kQwen38GraphIdMaxPhases);
}

TEST(Qwen38GraphIdTest, OnlyTheTargetRunsBatchedAndTheBatchStaysWithinTheLimits) {
  EXPECT_THROW(Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, 2, 0, 0), std::out_of_range);
  EXPECT_THROW(Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, 2, 0, kQwen38GraphIdMaxBatch + 1), std::out_of_range);
  EXPECT_THROW(Qwen38GraphAnnotationId(Qwen38GraphRun::kTarget, 2, 0, -1), std::out_of_range);
  EXPECT_THROW(Qwen38GraphAnnotationId(Qwen38GraphRun::kHead, 1, 0, 2), std::out_of_range);
  EXPECT_THROW(Qwen38GraphAnnotationId(Qwen38GraphRun::kHeadR2, 2, 0, 2), std::out_of_range);
  EXPECT_NO_THROW(Qwen38GraphAnnotationId(Qwen38GraphRun::kHead, 1, 0, 1));
}

TEST(Qwen38GraphConfigTest, NonInheritedPrefillSessionOptionsRetainCudaProviderWithoutGraphCapture) {
  Config config;
  OverlayConfig(config, R"({"model":{"decoder":{"session_options":{
    "custom_ops_libraries":["libs/a.so"],
    "provider_options":[{"cuda":{"device_id":"1","enable_cuda_graph":"1"}}]
  }}}})");
  config.model.decoder.session_options.providers = {"cuda"};

  // Omitted prefill.head_kv.session_options (inherit=false): retains CUDA + device_id, drops enable_cuda_graph and custom_ops_libraries.
  const auto resolved_omitted =
      ResolveQwen38AuxiliarySessionOptions(config.model.decoder.session_options, std::nullopt, /*inherit=*/false);
  EXPECT_EQ(resolved_omitted.providers, (std::vector<std::string>{"cuda"}));
  EXPECT_TRUE(resolved_omitted.custom_ops_libraries.empty());
  ASSERT_EQ(resolved_omitted.provider_options.size(), 1u);
  EXPECT_EQ(resolved_omitted.provider_options[0].name, "cuda");
  EXPECT_FALSE(IsGraphCaptureEnabled(resolved_omitted));
  bool found_device_id = false;
  for (const auto& [k, v] : resolved_omitted.provider_options[0].options) {
    if (k == "device_id" && v == "1") found_device_id = true;
    EXPECT_NE(k, "enable_cuda_graph");
  }
  EXPECT_TRUE(found_device_id);

  // Inherited head session (inherit=true): keeps custom_ops_libraries and enable_cuda_graph.
  const auto resolved_inherited =
      ResolveQwen38AuxiliarySessionOptions(config.model.decoder.session_options, std::nullopt, /*inherit=*/true);
  EXPECT_EQ(resolved_inherited.custom_ops_libraries, (std::vector<std::string>{"libs/a.so"}));
  EXPECT_TRUE(IsGraphCaptureEnabled(resolved_inherited));
}

TEST(Qwen38CacheShapeTest, AcceptsTheExportedGeometry) {
  const std::vector<Qwen38CacheInput> target = {
      {"input_ids", {-1, 1}},
      {"past_position_ids", {-1, 33028}},
      {"past_key_values.3.key", {-1, 2, 33028, 256}},
      {"past_key_values.3.value", {-1, 2, 33028, 256}},
      {"past_key_values.3.index_key", {-1, 8257, 128}},
      {"past_key_values.3.index_buffer", {-1, 7, 128}},
      {"past_key_values.3.index_lengths", {-1, 2}},
      {"past_key_values.0.conv_state", {-1, 6144, 4}},
      {"past_key_values.0.recurrent_state", {-1, 32, 128, 128}},
  };
  EXPECT_NO_THROW(ValidateQwen38CacheShapes(target, 33028, "target"));
  // The position and index-key buffers may be longer than needed; only the KV cache length must match exactly.
  const std::vector<Qwen38CacheInput> padded = {
      {"past_position_ids", {-1, 33040}},
      {"past_key_values.0.key", {-1, 2, 33028, 256}},
      {"past_key_values.0.index_key", {-1, 9000, 128}},
  };
  EXPECT_NO_THROW(ValidateQwen38CacheShapes(padded, 33028, "target"));
  EXPECT_NO_THROW(ValidateQwen38CacheShapes({}, 33028, "target"));
}

TEST(Qwen38CacheShapeTest, RejectsAnotherKvCapacity) {
  const std::vector<Qwen38CacheInput> inputs = {
      {"past_position_ids", {-1, 33028}},
      {"past_key_values.3.key", {-1, 2, 33028, 256}},
      {"past_key_values.3.index_key", {-1, 8257, 128}},
  };
  const auto message_for = [&](int kv_capacity) {
    try {
      ValidateQwen38CacheShapes(inputs, kv_capacity, "target");
    } catch (const std::runtime_error& e) {
      return std::string(e.what());
    }
    return std::string();
  };

  // A smaller capacity: the engine would stride the 33028-long buffers as 16384-long.
  const std::string smaller = message_for(16384);
  EXPECT_NE(smaller.find("kv_capacity (16384)"), std::string::npos) << smaller;
  EXPECT_NE(smaller.find("target"), std::string::npos) << smaller;
  EXPECT_NE(smaller.find("past_key_values.3.key"), std::string::npos) << smaller;
  EXPECT_NE(smaller.find("[-1, 2, 33028, 256]"), std::string::npos) << smaller;

  // A larger capacity: the position buffer is too short, which is found first.
  const std::string larger = message_for(65536);
  EXPECT_NE(larger.find("kv_capacity (65536)"), std::string::npos) << larger;
  EXPECT_NE(larger.find("past_position_ids"), std::string::npos) << larger;

  // Only the key/value length is exact: with a matching position buffer, a larger capacity fails on the cache.
  const std::vector<Qwen38CacheInput> longer_positions = {
      {"past_position_ids", {-1, 65536}},
      {"past_key_values.3.key", {-1, 2, 33028, 256}},
  };
  EXPECT_THROW(ValidateQwen38CacheShapes(longer_positions, 65536, "target"), std::runtime_error);
}

TEST(Qwen38CacheShapeTest, RejectsUnexpectedCacheDimensions) {
  const auto rejects = [](const Qwen38CacheInput& input, const char* graph = "target") {
    try {
      ValidateQwen38CacheShapes({input}, 33028, graph);
    } catch (const std::runtime_error& e) {
      const std::string message = e.what();
      EXPECT_NE(message.find(input.name), std::string::npos) << message;
      EXPECT_NE(message.find(graph), std::string::npos) << message;
      return;
    }
    ADD_FAILURE() << "accepted " << input.name;
  };

  rejects({"past_position_ids", {-1, 33000}});
  rejects({"past_position_ids", {-1, -1}});  // a dynamic length cannot be checked against the ledger
  rejects({"past_position_ids", {33028}});
  rejects({"past_key_values.0.key", {-1, 2, -1, 256}});
  rejects({"past_key_values.0.value", {-1, 2, 32768, 256}});
  rejects({"past_key_values.0.key", {-1, 2, 33028}});
  rejects({"past_key_values.0.key", {-1, 4, 33028, 256}});
  rejects({"past_key_values.0.value", {-1, 2, 33028, 128}});
  rejects({"past_key_values.0.index_key", {-1, 8000, 128}});
  rejects({"past_key_values.0.index_key", {-1, -1, 128}});
  rejects({"past_key_values.0.index_key", {-1, 8257}});
  // Auxiliary graphs name themselves in the error.
  rejects({"head.past_key_values.0.key", {-1, 2, 16384, 256}}, "fold head");
  rejects({"past_key_values.0.key", {-1, 2, 16384, 256}}, "head (S=2)");
  rejects({"past_key_values.0.value", {-1, 2, 16384, 256}}, "head_r2");
}

TEST(Qwen38CacheShapeTest, IgnoresInputsThatDoNotDependOnTheCapacity) {
  const std::vector<Qwen38CacheInput> inputs = {
      {"hidden_states", {-1, 1, 2048}},
      {"attention_mask", {-1, 5}},
      {"key", {-1, 7}},  // not a cache input
      {"past_key_values.0.conv_state", {-1, 6144, 4}},
      {"past_key_values.0.recurrent_state", {-1, 32, 128, 128}},
      {"past_key_values.0.index_buffer", {-1, 7, 128}},
      {"past_key_values.0.index_lengths", {-1, 2}},
  };
  EXPECT_NO_THROW(ValidateQwen38CacheShapes(inputs, 33028, "target"));
}

TEST(Qwen38QsaConstantsTest, RequiresBothCosAndSinForTargetAndHead) {
  const auto dir = MakeScratchDir("qsa_complete");
  WriteFile(dir / "qsa_consts.bin", std::string(4096, '\0'));

  const auto check_index = [&](const std::string& text) {
    Qwen38Plane plane;
    InitMockPlane(plane);
    const auto idx = WriteFile(dir / "qsa.txt", text);
    Qwen38QsaConstants consts{plane, idx, dir, nullptr};
    (void)consts;
  };

  // Missing target sin
  EXPECT_THROW(check_index("qwen38_qsa_consts 1\nfile qsa_consts.bin\nepsilon 1e-06\nwidth 64\n"
                           "tensor target cos 0 512\n"
                           "tensor target knorm.3 512 256\n"
                           "tensor head cos 768 512\n"
                           "tensor head sin 1280 512\n"
                           "tensor head knorm.0 1792 256\n"),
               std::runtime_error);

  // Missing head sin
  EXPECT_THROW(check_index("qwen38_qsa_consts 1\nfile qsa_consts.bin\nepsilon 1e-06\nwidth 64\n"
                           "tensor target cos 0 512\n"
                           "tensor target sin 512 512\n"
                           "tensor target knorm.3 1024 256\n"
                           "tensor head cos 1280 512\n"
                           "tensor head knorm.0 1792 256\n"),
               std::runtime_error);

  // Complete index succeeds
  EXPECT_NO_THROW(check_index(
      "qwen38_qsa_consts 1\nfile qsa_consts.bin\nepsilon 1e-06\nwidth 64\n"
      "tensor target cos 0 512\n"
      "tensor target sin 512 512\n"
      "tensor target knorm.3 1024 256\n"
      "tensor head cos 1280 512\n"
      "tensor head sin 1792 512\n"
      "tensor head knorm.0 2304 256\n"));

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38QsaConstantsTest, KvScalesMustHoldExactly512Floats) {
  const auto dir = MakeScratchDir("qsa_scales");
  WriteFile(dir / "qsa_consts.bin", std::string(16384, '\0'));

  const std::string base =
      "qwen38_qsa_consts 1\nfile qsa_consts.bin\nepsilon 1e-06\nwidth 64\n"
      "tensor target cos 0 512\n"
      "tensor target sin 512 512\n"
      "tensor target knorm.3 1024 256\n"
      "tensor head cos 1280 512\n"
      "tensor head sin 1792 512\n"
      "tensor head knorm.0 2304 256\n";
  // Returns the error text of a failed construction, empty when it succeeded.
  const auto construct = [&](const std::string& scale_lines) {
    Qwen38Plane plane;
    InitMockPlane(plane);
    const auto idx = WriteFile(dir / "qsa.txt", base + scale_lines);
    try {
      Qwen38QsaConstants consts{plane, idx, dir, nullptr};
      (void)consts;
    } catch (const std::runtime_error& e) {
      return std::string(e.what());
    }
    return std::string();
  };

  EXPECT_EQ(construct(""), "");
  EXPECT_EQ(construct("tensor target kscale.3 4096 2048\ntensor target vscale.3 6144 2048\n"
                      "tensor head kscale.0 8192 2048\ntensor head vscale.0 10240 2048\n"),
            "");

  const auto scale_error = [&](const std::string& line, const std::string& name) {
    const std::string error = construct(line);
    EXPECT_NE(error.find("512 floats"), std::string::npos) << line << " -> " << error;
    EXPECT_NE(error.find(name), std::string::npos) << line << " -> " << error;
  };
  scale_error("tensor target kscale.3 4096 4\n", "kscale.3");  // one float
  scale_error("tensor target vscale.3 4096 4\n", "vscale.3");
  scale_error("tensor target kscale.3 4096 2044\n", "kscale.3");  // 511 floats
  scale_error("tensor target vscale.3 4096 2052\n", "vscale.3");  // 513 floats
  scale_error("tensor head kscale.0 4096 4096\n", "kscale.0");    // 1024 floats
  scale_error("tensor head vscale.0 4096 2046\n", "vscale.0");    // not a whole number of floats
  // A valid scale does not make its invalid partner acceptable.
  scale_error("tensor target kscale.3 4096 2048\ntensor target vscale.3 6144 1024\n", "vscale.3");

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

#if !defined(_WIN32)
TEST(Qwen38WeightPoolTest, CleansUpEarlierAllocationsOnConstructorFailureAndSupportsHostPool) {
  const auto dir = MakeScratchDir("weight_pool");
  WriteFile(dir / "w0.bin", std::string(64, 'a'));
  WriteFile(dir / "w1.bin", std::string(64, 'b'));
  WriteFile(dir / "embed.bin", std::string(128, 'e'));
  const auto manifest_path = WriteFile(dir / "pool.txt",
                                       "qwen38_weights_pool 1\n"
                                       "pool 0 1 0 32 1 8 w0.bin\n"
                                       "pool 1 1 0 32 1 8 w1.bin\n"
                                       "host_pool 2 1 32 64 2 2 8 embed.bin\n"
                                       "session target w0 0\n"
                                       "session target w1 1\n"
                                       "session target embed 2\n");
  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // Fail on the 2nd device_alloc call: the 1st device allocation and the host_pool mapping must be cleaned up.
  {
    Qwen38Plane plane;
    InitMockPlane(plane, /*fail_alloc=*/2);
    EXPECT_THROW((Qwen38WeightPool{plane, manifest_path, dir, *mem_info, 1}), std::runtime_error);
    EXPECT_TRUE(MockPlaneTracker::live_device_ptrs.empty());
    EXPECT_EQ(MockPlaneTracker::unregistered_host_ptrs, MockPlaneTracker::registered_host_ptrs);
  }

  // Succeed and verify device bytes, host_mapped_bytes, and cleanup on destruction.
  {
    Qwen38Plane plane;
    InitMockPlane(plane);
    {
      Qwen38WeightPool pool{plane, manifest_path, dir, *mem_info, 2};
      EXPECT_EQ(pool.bytes, 64u);
      EXPECT_EQ(pool.host_mapped_bytes, 64u);
      EXPECT_EQ(MockPlaneTracker::live_device_ptrs.size(), 2u);
      EXPECT_EQ(MockPlaneTracker::registered_host_ptrs.size(), 1u);
    }
    EXPECT_TRUE(MockPlaneTracker::live_device_ptrs.empty());
    EXPECT_EQ(MockPlaneTracker::unregistered_host_ptrs, MockPlaneTracker::registered_host_ptrs);
  }

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

namespace {

// One device entry (#0), one host-mapped entry (#1) of the same file, and (optionally) one CPU entry (#2).
std::filesystem::path WriteMappedPoolManifest(const std::filesystem::path& dir, bool with_cpu_entry) {
  WriteFile(dir / "w0.bin", std::string(64, 'a'));
  WriteFile(dir / "embed.bin", std::string(128, 'e'));
  std::string manifest =
      "qwen38_weights_pool 1\n"
      "pool 0 1 0 32 1 8 w0.bin\n"
      "host_pool 1 1 32 64 2 2 8 embed.bin\n"
      "session target w0 0\n"
      "session target embed 1\n";
  if (with_cpu_entry) {
    manifest +=
        "pool 2 1 0 32 1 8 w0.bin cpu\n"
        "session dense cpu_weight 2\n";
  }
  return WriteFile(dir / "pool.txt", manifest);
}

}  // namespace

TEST(Qwen38WeightPoolTest, HostMappedEntriesRequireADeviceAlias) {
  const auto dir = MakeScratchDir("weight_pool_alias");
  const auto manifest_path = WriteMappedPoolManifest(dir, /*with_cpu_entry=*/false);
  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  const auto expect_failure = [&](const std::function<void(Qwen38Plane&)>& break_alias) {
    Qwen38Plane plane;
    InitMockPlane(plane);
    break_alias(plane);
    try {
      Qwen38WeightPool pool{plane, manifest_path, dir, *mem_info, 1};
      ADD_FAILURE() << "a host-mapped entry without a device alias must fail the construction";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find("host-mapped weight pool entry 1"), std::string::npos) << e.what();
    }
    // The failure goes through the cleanup path: the device copy is freed and the mapped range is unregistered.
    EXPECT_TRUE(MockPlaneTracker::live_device_ptrs.empty());
    EXPECT_FALSE(MockPlaneTracker::registered_host_ptrs.empty());
    EXPECT_EQ(MockPlaneTracker::unregistered_host_ptrs, MockPlaneTracker::registered_host_ptrs);
  };

  expect_failure([](Qwen38Plane&) { MockPlaneTracker::device_alias = MockPlaneTracker::DeviceAlias::kFails; });
  expect_failure([](Qwen38Plane&) { MockPlaneTracker::device_alias = MockPlaneTracker::DeviceAlias::kNull; });
  expect_failure([](Qwen38Plane& plane) { plane.host_get_device_pointer = nullptr; });

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38WeightPoolTest, HostMappedEntriesAcceptAliasesAndUnifiedAddresses) {
  const auto dir = MakeScratchDir("weight_pool_alias_ok");
  const auto manifest_path = WriteMappedPoolManifest(dir, /*with_cpu_entry=*/false);
  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  for (const auto alias : {MockPlaneTracker::DeviceAlias::kBiased, MockPlaneTracker::DeviceAlias::kIdentity}) {
    Qwen38Plane plane;
    InitMockPlane(plane);
    MockPlaneTracker::device_alias = alias;
    {
      Qwen38WeightPool pool{plane, manifest_path, dir, *mem_info, 1};
      EXPECT_EQ(pool.host_mapped_bytes, 64u);
    }
    EXPECT_TRUE(MockPlaneTracker::live_device_ptrs.empty());
    EXPECT_EQ(MockPlaneTracker::unregistered_host_ptrs, MockPlaneTracker::registered_host_ptrs);
  }

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38WeightPoolTest, CpuEntriesWithoutADeviceAliasStayCpuTensors) {
  const auto dir = MakeScratchDir("weight_pool_cpu");
  const auto manifest_path = WriteMappedPoolManifest(dir, /*with_cpu_entry=*/true);
  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // The CPU-only fallback is separate from the host-mapped requirement: with every lookup failing, the host-mapped
  // entry is what fails the construction, and a pool without one still builds.
  {
    Qwen38Plane plane;
    InitMockPlane(plane);
    MockPlaneTracker::device_alias = MockPlaneTracker::DeviceAlias::kFails;
    EXPECT_THROW((Qwen38WeightPool{plane, manifest_path, dir, *mem_info, 1}), std::runtime_error);
  }

  WriteFile(dir / "pool_cpu_only.txt",
            "qwen38_weights_pool 1\n"
            "pool 0 1 0 32 1 8 w0.bin cpu\n"
            "session dense cpu_weight 0\n");
  for (const auto alias : {MockPlaneTracker::DeviceAlias::kFails, MockPlaneTracker::DeviceAlias::kNull}) {
    Qwen38Plane plane;
    InitMockPlane(plane);
    MockPlaneTracker::device_alias = alias;
    {
      Qwen38WeightPool pool{plane, dir / "pool_cpu_only.txt", dir, *mem_info, 1};
      EXPECT_EQ(pool.cpu_bytes, 32u);
      auto options = OrtSessionOptions::Create();
      EXPECT_EQ(pool.AddTo(*options, "dense"), 1u);
    }
    EXPECT_EQ(MockPlaneTracker::unregistered_host_ptrs, MockPlaneTracker::registered_host_ptrs);
  }

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
#endif

}  // namespace Generators::test
