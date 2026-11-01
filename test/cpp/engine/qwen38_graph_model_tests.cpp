// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "config.h"
#include "models/model_type.h"
#include "models/qwen38_decode_device.h"
#include "models/qwen38_graph.h"
#include "models/qwen38_replay_geometry.h"
#include "scoped-environment-variable.h"
#include "span.h"

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
  static inline std::vector<void*> registered_host_ptrs;  // accepted registrations only
  static inline std::vector<void*> unregistered_host_ptrs;
  static inline std::vector<std::pair<uintptr_t, uintptr_t>> live_host_ranges;
  static inline std::vector<uintptr_t> alias_queries;  // every host pointer a device alias was requested for
  static inline int rejected_host_registrations = 0;
  // CUDA's current device is per thread; the calls that create CUDA state record the device of the thread they ran on.
  static inline thread_local int current_device = 0;
  static inline std::mutex device_uses_mutex;
  static inline std::vector<std::pair<std::string, int>> device_uses;
  static constexpr uintptr_t kDevBias = 0x200000u;
  static constexpr int kHostMemoryAlreadyRegistered = 712;  // cudaErrorHostMemoryAlreadyRegistered

  static void Reset(int fail_alloc = -1) {
    for (void* p : live_device_ptrs) std::free(p);
    live_device_ptrs.clear();
    registered_host_ptrs.clear();
    unregistered_host_ptrs.clear();
    live_host_ranges.clear();
    alias_queries.clear();
    rejected_host_registrations = 0;
    current_device = 0;
    {
      std::lock_guard<std::mutex> lock(device_uses_mutex);
      device_uses.clear();
    }
    alloc_calls = 0;
    fail_on_alloc_call = fail_alloc;
    device_alias = DeviceAlias::kBiased;
  }

  static void RecordDeviceUse(const char* call) {
    std::lock_guard<std::mutex> lock(device_uses_mutex);
    device_uses.emplace_back(call, current_device);
  }

  static const char* ErrorString(int) { return "mock cuda error"; }
  static int SetDevice(int device) {
    current_device = device;
    return 0;
  }
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
    RecordDeviceUse("host_alloc");
    *p = std::calloc(1, bytes ? bytes : 16);
    return 0;
  }
  static int HostFree(void* p) {
    std::free(p);
    return 0;
  }
  // Like cudaHostRegister, refuses a range that overlaps one that is still registered.
  static int HostRegister(void* p, size_t bytes, unsigned) {
    const auto begin = reinterpret_cast<uintptr_t>(p);
    const auto end = begin + bytes;
    for (const auto& [live_begin, live_end] : live_host_ranges) {
      if (begin < live_end && live_begin < end) {
        ++rejected_host_registrations;
        return kHostMemoryAlreadyRegistered;
      }
    }
    live_host_ranges.emplace_back(begin, end);
    registered_host_ptrs.push_back(p);
    return 0;
  }
  static int HostUnregister(void* p) {
    const auto begin = reinterpret_cast<uintptr_t>(p);
    live_host_ranges.erase(std::remove_if(live_host_ranges.begin(), live_host_ranges.end(),
                                          [begin](const auto& range) { return range.first == begin; }),
                           live_host_ranges.end());
    unregistered_host_ptrs.push_back(p);
    return 0;
  }
  static int HostGetDevicePointer(void** dp, void* hp, unsigned) {
    const auto host = reinterpret_cast<uintptr_t>(hp);
    alias_queries.push_back(host);
    switch (device_alias) {
      case DeviceAlias::kFails:
        return 1;  // cudaErrorInvalidValue
      case DeviceAlias::kNull:
        *dp = nullptr;
        return 0;
      case DeviceAlias::kIdentity:
      case DeviceAlias::kBiased:
        break;
    }
    // Like the real call, only memory that lies in a registered range has a device address.
    const bool registered = std::any_of(live_host_ranges.begin(), live_host_ranges.end(), [host](const auto& range) {
      return range.first <= host && host < range.second;
    });
    if (!registered) return 1;
    *dp = device_alias == DeviceAlias::kIdentity ? hp : reinterpret_cast<void*>(host + kDevBias);
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
    RecordDeviceUse("stream_create");
    *s = reinterpret_cast<void*>(uintptr_t{1});
    return 0;
  }
  static int StreamDestroy(void*) { return 0; }
  static int StreamSync(void*) { return 0; }
};

void InitMockPlane(Qwen38Plane& plane, int fail_alloc = -1) {
  MockPlaneTracker::Reset(fail_alloc);
  plane.error_string = &MockPlaneTracker::ErrorString;
  plane.set_device = &MockPlaneTracker::SetDevice;
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

// The refusal names the rows the ledger does commit, whatever the verify width, not the 1 or 2 of the S=2 ring.
TEST(Qwen38DecodeLedgerTest, TheRefusalOfAStepNamesTheRowsTheLedgerCommits) {
  const auto message_of = [](Qwen38DecodeLedger& ledger, size_t step) {
    try {
      ledger.Advance(step);
    } catch (const std::logic_error& e) {
      return std::string{e.what()};
    }
    return std::string{};
  };
  Qwen38DecodeLedger ring{64};
  EXPECT_NE(message_of(ring, 3).find("commits 1 to 2 rows, not 3"), std::string::npos);
  Qwen38DecodeLedger ping_pong{64};
  ping_pong.SetCompactPingPong(true, 8);
  EXPECT_NE(message_of(ping_pong, 9).find("commits 1 to 8 rows, not 9"), std::string::npos);
  EXPECT_NE(message_of(ping_pong, 0).find("commits 1 to 8 rows, not 0"), std::string::npos);
  EXPECT_TRUE(message_of(ping_pong, 8).empty());
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

TEST(Qwen38ReplayCoverageTest, RequiresEveryBankedStateToHaveAStateUpdateOutput) {
  const std::vector<std::string> bank_names = {
      "past_key_values.0.conv_state",
      "past_key_values.0.recurrent_state",
      "past_key_values.1.ple_conv_state",
      "past_key_values.3.index_buffer",
      "past_key_values.3.index_lengths",
  };
  const std::vector<std::string> complete_updates = {
      "state_update.0.conv_value",
      "state_update.0.recurrent_capsule",
      "state_update.1.ple_conv_value",
      "state_update.3.indexer_value",
  };
  EXPECT_NO_THROW(RequireQwen38ReplayCoverage(bank_names, complete_updates));

  // Missing recurrent_capsule for layer 0 must be rejected naming both the missing output and the banked state.
  try {
    RequireQwen38ReplayCoverage(
        bank_names, {"state_update.0.conv_value", "state_update.1.ple_conv_value", "state_update.3.indexer_value"});
    ADD_FAILURE() << "expected missing state_update.0.recurrent_capsule to be rejected";
  } catch (const std::runtime_error& e) {
    const std::string msg = e.what();
    EXPECT_NE(msg.find("state_update.0.recurrent_capsule"), std::string::npos) << msg;
    EXPECT_NE(msg.find("past_key_values.0.recurrent_state"), std::string::npos) << msg;
  }
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

// Warmup() runs the standalone R=2 head the number of times a captured graph needs; the answer has to come from the
// options the head session is created with (the decoder's merged into the configured ones), not from the configured ones
// alone, or a graph is captured inside a live request.
TEST(Qwen38GraphConfigTest, HeadR2CapturesGraphsWhenItsMergedSessionOptionsDo) {
  const auto decoder_options = [](const char* cuda_options) {
    Config config;
    OverlayConfig(config, std::string(R"({"model":{"decoder":{"session_options":{"provider_options":[{"cuda":)") +
                              cuda_options + "}]}}}}");
    config.model.decoder.session_options.providers = {"cuda"};
    return config.model.decoder.session_options;
  };
  // The head_r2 options as the configuration loader leaves them: `providers` names the provider_options sections.
  const auto head_r2_options = [](const char* session_options) {
    Config config;
    OverlayConfig(config, std::string(R"({"model":{"qwen38_graph":{"fold":{"head_r2":{"session_options":)") +
                              session_options + "}}}}}");
    auto options = config.model.qwen38_graph.fold.head_r2.session_options;
    for (const auto& provider : options->provider_options) options->providers.push_back(provider.name);
    return options;
  };

  const auto capturing = decoder_options(R"({"device_id":"0","enable_cuda_graph":"1"})");
  ASSERT_TRUE(IsGraphCaptureEnabled(capturing));
  EXPECT_TRUE(Qwen38HeadR2CapturesGraph(capturing, std::nullopt));

  // Options that name unrelated settings only: the head session still inherits the decoder's CUDA provider and with it
  // the graph capture, although the configured options alone enable none.
  const auto unrelated = head_r2_options(R"({"intra_op_num_threads":2})");
  EXPECT_FALSE(IsGraphCaptureEnabled(*unrelated));
  EXPECT_TRUE(Qwen38HeadR2CapturesGraph(capturing, unrelated));

  // A provider section that does not mention graph capture inherits it from the decoder's section of the same provider.
  const auto device_only = head_r2_options(R"({"provider_options":[{"cuda":{"device_id":"1"}}]})");
  EXPECT_FALSE(IsGraphCaptureEnabled(*device_only));
  EXPECT_TRUE(Qwen38HeadR2CapturesGraph(capturing, device_only));

  // An explicit opt-out of the head session wins over the decoder's setting.
  const auto opted_out = head_r2_options(R"({"provider_options":[{"cuda":{"enable_cuda_graph":"0"}}]})");
  EXPECT_FALSE(Qwen38HeadR2CapturesGraph(capturing, opted_out));

  // A decoder that does not capture does not make its head capture.
  const auto eager = decoder_options(R"({"device_id":"0"})");
  EXPECT_FALSE(Qwen38HeadR2CapturesGraph(eager, std::nullopt));
  EXPECT_FALSE(Qwen38HeadR2CapturesGraph(eager, unrelated));
  EXPECT_FALSE(Qwen38HeadR2CapturesGraph(eager, device_only));
}

TEST(Qwen38DflashFeatureLayoutTest, OnlyTheRowsTheEngineProducesAreAccepted) {
  EXPECT_NO_THROW(ValidateQwen38DflashFeatureLayout(kQwen38DflashFeatureWidth, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16));
  EXPECT_NO_THROW(ValidateQwen38DflashFeatureLayout(12800, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16));
}

// Propose slices the engine's feature buffer with the stride of the drafter's own input, so any other width or element type
// would read outside every row. The message names what the drafter declares and what the engine produces.
TEST(Qwen38DflashFeatureLayoutTest, ADrafterOfAnotherWidthOrTypeIsRefusedAtLoad) {
  const auto message_of = [](size_t width, ONNXTensorElementDataType type) {
    try {
      ValidateQwen38DflashFeatureLayout(width, type);
    } catch (const std::runtime_error& e) {
      return std::string{e.what()};
    }
    return std::string{};
  };
  const std::string wider = message_of(kQwen38DflashFeatureWidth * 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16);
  EXPECT_NE(wider.find("25600"), std::string::npos) << wider;
  EXPECT_NE(wider.find("12800"), std::string::npos) << wider;
  EXPECT_FALSE(message_of(kQwen38DflashFeatureWidth - 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16).empty());
  EXPECT_FALSE(message_of(kQwen38DflashFeatureWidth + 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16).empty());
  EXPECT_FALSE(message_of(0, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16).empty());
  EXPECT_FALSE(message_of(kQwen38DflashFeatureWidth, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16).empty());
  EXPECT_FALSE(message_of(kQwen38DflashFeatureWidth, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT).empty());
  EXPECT_FALSE(message_of(kQwen38DflashFeatureWidth, ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED).empty());
}

TEST(Qwen38PromptTokenTest, AcceptsEveryIdOfTheVocabulary) {
  const std::vector<int32_t> tokens = {0, 1, 1000, 248319};
  EXPECT_NO_THROW(ValidateQwen38PromptTokens(tokens, 248320));
  EXPECT_NO_THROW(ValidateQwen38PromptTokens(std::span<const int32_t>{}, 248320));
}

TEST(Qwen38PromptTokenTest, RejectsIdsOutsideTheVocabularyAndNamesTheOffender) {
  const auto message_of = [](std::vector<int32_t> tokens, int vocab) {
    try {
      ValidateQwen38PromptTokens(tokens, vocab);
    } catch (const std::invalid_argument& e) {
      return std::string{e.what()};
    }
    return std::string{};
  };
  EXPECT_NE(message_of({5, -1}, 10).find("prompt token 1 has id -1"), std::string::npos);
  EXPECT_NE(message_of({3, 4, 10}, 10).find("prompt token 2 has id 10"), std::string::npos);
  EXPECT_NE(message_of({0, std::numeric_limits<int32_t>::max()}, 248320).find("prompt token 1"), std::string::npos);
  EXPECT_NE(message_of({std::numeric_limits<int32_t>::min()}, 248320).find("prompt token 0"), std::string::npos);
  EXPECT_TRUE(message_of({9}, 10).empty());
  EXPECT_FALSE(message_of({0}, 0).empty());  // an empty vocabulary accepts nothing
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
  rejects({"past_key_values.0.index_key", {-1, 8257, 64}});
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

  // Complete index succeeds and records each RoPE table's row count (512 bytes / (64 * 2) = 4 rows).
  {
    Qwen38Plane plane;
    InitMockPlane(plane);
    const auto idx = WriteFile(dir / "qsa.txt",
                               "qwen38_qsa_consts 1\nfile qsa_consts.bin\nepsilon 1e-06\nwidth 64\n"
                               "tensor target cos 0 512\n"
                               "tensor target sin 512 512\n"
                               "tensor target knorm.3 1024 256\n"
                               "tensor head cos 1280 512\n"
                               "tensor head sin 1792 512\n"
                               "tensor head knorm.0 2304 256\n");
    Qwen38QsaConstants consts{plane, idx, dir, nullptr};
    EXPECT_EQ(consts.target.cos_rows, 4u);
    EXPECT_EQ(consts.target.sin_rows, 4u);
    EXPECT_EQ(consts.head.cos_rows, 4u);
    EXPECT_EQ(consts.head.sin_rows, 4u);
    EXPECT_NO_THROW(consts.target.RequireRopePositions(0, "target"));
    EXPECT_NO_THROW(consts.target.RequireRopePositions(1, "target"));  // 4 tokens -> row 0
    EXPECT_THROW(consts.target.RequireRopePositions(2, "target"), std::runtime_error);  // 8 tokens -> row 4
    EXPECT_THROW(consts.head.RequireRopePositions(2, "head"), std::runtime_error);
  }

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

TEST(Qwen38QsaConstantsTest, EveryConstantMustStartOnItsElementBoundary) {
  const auto dir = MakeScratchDir("qsa_alignment");
  WriteFile(dir / "qsa_consts.bin", std::string(16384, '\0'));

  // Returns the error text of a failed construction, empty when it succeeded. The target set's cos, sin and knorm.3 and
  // any extra lines are placed by the caller; the head set's constants sit on aligned offsets.
  const auto construct = [&](size_t cos_offset, size_t sin_offset, size_t knorm_offset, const std::string& extra) {
    Qwen38Plane plane;
    InitMockPlane(plane);
    const auto tensor = [](const char* set, const char* name, size_t offset, size_t bytes) {
      return "tensor " + std::string(set) + " " + name + " " + std::to_string(offset) + " " + std::to_string(bytes) +
             "\n";
    };
    std::string index = "qwen38_qsa_consts 1\nfile qsa_consts.bin\nepsilon 1e-06\nwidth 64\n";
    index += tensor("target", "cos", cos_offset, 512);
    index += tensor("target", "sin", sin_offset, 512);
    index += tensor("target", "knorm.3", knorm_offset, 256);
    index += tensor("head", "cos", 1280, 512);
    index += tensor("head", "sin", 1792, 512);
    index += tensor("head", "knorm.0", 2304, 256);
    const auto idx = WriteFile(dir / "qsa.txt", index + extra);
    try {
      Qwen38QsaConstants consts{plane, idx, dir, nullptr};
      (void)consts;
    } catch (const std::runtime_error& e) {
      return std::string(e.what());
    }
    return std::string();
  };
  const auto expect_misaligned = [](const std::string& error, const std::string& name, size_t offset, size_t alignment) {
    EXPECT_NE(error.find("not aligned to " + std::to_string(alignment) + " bytes"), std::string::npos) << error;
    EXPECT_NE(error.find("constant " + name + " at offset " + std::to_string(offset)), std::string::npos) << error;
  };

  EXPECT_EQ(construct(0, 512, 1024, ""), "");

  // The bf16 constants (cos, sin, knorm) need even offsets.
  expect_misaligned(construct(2561, 512, 1024, ""), "cos", 2561, 2);
  expect_misaligned(construct(0, 2563, 1024, ""), "sin", 2563, 2);
  expect_misaligned(construct(0, 512, 3073, ""), "knorm.3", 3073, 2);

  // The fp32 scales need offsets that are a multiple of 4: an even offset is not enough.
  expect_misaligned(construct(0, 512, 1024, "tensor target kscale.3 4097 2048\n"), "kscale.3", 4097, 4);
  expect_misaligned(construct(0, 512, 1024, "tensor target vscale.3 4098 2048\n"), "vscale.3", 4098, 4);
  expect_misaligned(construct(0, 512, 1024, "tensor head kscale.0 6150 2048\n"), "kscale.0", 6150, 4);
  expect_misaligned(construct(0, 512, 1024, "tensor head vscale.0 8195 2048\n"), "vscale.0", 8195, 4);

  // Each constant is held to its own element size.
  EXPECT_EQ(construct(0, 512, 1026, "tensor target kscale.3 4100 2048\ntensor target vscale.3 6148 2048\n"), "");

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

  // Fail on the 2nd device_alloc call: the 1st device allocation must be released. Host ranges are registered only after
  // every entry is placed, so none exists yet; cleanup after a registration is covered by the tests further down.
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

TEST(Qwen38WeightPoolTest, PinningTheCpuPoolRegistersEachMappedRangeOnce) {
  const auto dir = MakeScratchDir("weight_pool_pin_cpu_pool");
  const auto manifest_path = WriteMappedPoolManifest(dir, /*with_cpu_entry=*/true);
  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  for (const char* pin : {"1", "0"}) {
    ScopedEnvironmentVariable pin_cpu_pool{"ORTGENAI_PIN_CPU_POOL", pin};
    Qwen38Plane plane;
    InitMockPlane(plane);
    {
      // The file mapping must leave the registration to the pool: a whole-file registration made under
      // ORTGENAI_PIN_CPU_POOL=1 would overlap the pool's own range and CUDA rejects that (712).
      Qwen38WeightPool pool{plane, manifest_path, dir, *mem_info, 1};
      EXPECT_EQ(MockPlaneTracker::rejected_host_registrations, 0) << "ORTGENAI_PIN_CPU_POOL=" << pin;
      EXPECT_EQ(MockPlaneTracker::registered_host_ptrs.size(), 2u) << "one range per mapped file";
      EXPECT_EQ(pool.cpu_bytes, 32u);
      EXPECT_EQ(pool.host_mapped_bytes, 64u);
    }
    EXPECT_EQ(MockPlaneTracker::unregistered_host_ptrs, MockPlaneTracker::registered_host_ptrs);
    EXPECT_TRUE(MockPlaneTracker::live_host_ranges.empty());
  }

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38WeightPoolTest, TheBoundedHostCacheCannotBeCombinedWithPinningTheCpuPool) {
  const auto dir = MakeScratchDir("weight_pool_env_switches");
  const auto manifest_path = WriteMappedPoolManifest(dir, /*with_cpu_entry=*/true);
  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  // Returns the error text of a failed construction, empty when it succeeded.
  const auto construct = [&](std::optional<std::string> host_cache_mib, std::optional<std::string> pin_cpu_pool) {
    ScopedEnvironmentVariable host_cache{"ORT_QMOE_HOST_CACHE_MIB", host_cache_mib};
    ScopedEnvironmentVariable pin{"ORTGENAI_PIN_CPU_POOL", pin_cpu_pool};
    Qwen38Plane plane;
    InitMockPlane(plane);
    try {
      Qwen38WeightPool pool{plane, manifest_path, dir, *mem_info, 1};
    } catch (const std::runtime_error& e) {
      return std::string(e.what());
    }
    return std::string();
  };
  const auto expect_refused = [&](const std::string& host_cache_mib, const std::string& pin_cpu_pool) {
    const std::string error = construct(host_cache_mib, pin_cpu_pool);
    EXPECT_NE(error.find("cannot be combined"), std::string::npos)
        << "ORT_QMOE_HOST_CACHE_MIB=" << host_cache_mib << " ORTGENAI_PIN_CPU_POOL=" << pin_cpu_pool << ": " << error;
  };

  // Both switches on is refused, whichever way the boolean is spelled.
  expect_refused("64", "1");
  expect_refused("64", "true");

  // The pin switch is a boolean like the other GenAI switches: "false" and "0" leave it off.
  EXPECT_EQ(construct("64", "false"), "");
  EXPECT_EQ(construct("64", "0"), "");
  EXPECT_EQ(construct("64", std::nullopt), "");
  EXPECT_EQ(construct("64", ""), "");

  // The host cache is off when its variable is unset, empty or "0", so pinning the pool is allowed then.
  EXPECT_EQ(construct(std::nullopt, "1"), "");
  EXPECT_EQ(construct("", "1"), "");
  EXPECT_EQ(construct("0", "true"), "");

  // A value that is not a boolean is an error of its own instead of being read as "on".
  {
    ScopedEnvironmentVariable host_cache{"ORT_QMOE_HOST_CACHE_MIB", "64"};
    ScopedEnvironmentVariable pin{"ORTGENAI_PIN_CPU_POOL", "banana"};
    Qwen38Plane plane;
    InitMockPlane(plane);
    EXPECT_THROW((Qwen38WeightPool{plane, manifest_path, dir, *mem_info, 1}), std::invalid_argument);
  }

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38WeightPoolTest, ACpuEntryLeftOutOfTheRegistrationGetsNoDeviceAlias) {
  const auto dir = MakeScratchDir("weight_pool_unpinned_cpu");
  constexpr uint64_t kLarge = uint64_t{2} << 20;  // CPU entries of 1 MiB and more stay unregistered under the host cache
  // The large entry starts on the page that the small one registers.
  WriteFile(dir / "mixed.bin", std::string(64 + kLarge, 'm'));
  const auto manifest_path = WriteFile(dir / "pool.txt",
                                       "qwen38_weights_pool 1\n"
                                       "pool 0 1 0 64 1 16 mixed.bin cpu\n"
                                       "pool 1 1 64 2097152 1 524288 mixed.bin cpu\n"
                                       "session dense small 0\n"
                                       "session dense large 1\n");
  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
  ScopedEnvironmentVariable pin_cpu_pool{"ORTGENAI_PIN_CPU_POOL", std::nullopt};

  {
    ScopedEnvironmentVariable host_cache{"ORT_QMOE_HOST_CACHE_MIB", "64"};
    Qwen38Plane plane;
    InitMockPlane(plane);
    MockPlaneTracker::device_alias = MockPlaneTracker::DeviceAlias::kIdentity;
    {
      Qwen38WeightPool pool{plane, manifest_path, dir, *mem_info, 1};
      ASSERT_EQ(MockPlaneTracker::live_host_ranges.size(), 1u);
      const auto [begin, end] = MockPlaneTracker::live_host_ranges.front();
      EXPECT_EQ(end - begin, 4096u) << "only the page of the small entry is registered";
      // The page lookup of the large entry would succeed (it starts on that page) although most of the entry is not
      // registered, so the pool must not ask for its alias at all.
      EXPECT_EQ(MockPlaneTracker::alias_queries, std::vector<uintptr_t>{begin});
      EXPECT_EQ(pool.cpu_bytes, 64u + kLarge);
      auto options = OrtSessionOptions::Create();
      EXPECT_EQ(pool.AddTo(*options, "dense"), 2u);
    }
    EXPECT_TRUE(MockPlaneTracker::live_host_ranges.empty());
  }

  // Without the host cache both entries are registered, and both get their alias.
  {
    ScopedEnvironmentVariable host_cache{"ORT_QMOE_HOST_CACHE_MIB", std::nullopt};
    Qwen38Plane plane;
    InitMockPlane(plane);
    MockPlaneTracker::device_alias = MockPlaneTracker::DeviceAlias::kIdentity;
    Qwen38WeightPool pool{plane, manifest_path, dir, *mem_info, 1};
    ASSERT_EQ(MockPlaneTracker::live_host_ranges.size(), 1u);
    const auto [begin, end] = MockPlaneTracker::live_host_ranges.front();
    EXPECT_EQ(end - begin, 64u + kLarge);
    EXPECT_EQ(MockPlaneTracker::alias_queries, (std::vector<uintptr_t>{begin, begin + 64}));
  }

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38WeightPoolTest, UploadWorkersSelectTheDeviceOfTheMemoryInfo) {
  const auto dir = MakeScratchDir("weight_pool_worker_device");
  WriteFile(dir / "w0.bin", std::string(64, 'a'));
  WriteFile(dir / "w1.bin", std::string(64, 'b'));
  const auto manifest_path = WriteFile(dir / "pool.txt",
                                       "qwen38_weights_pool 1\n"
                                       "pool 0 1 0 32 1 8 w0.bin\n"
                                       "pool 1 1 0 32 1 8 w1.bin\n"
                                       "session target w0 0\n"
                                       "session target w1 1\n");
  // The name and the id are all that the pool reads from the memory info; no CUDA runtime is involved.
  auto mem_info = OrtMemoryInfo::Create("Cuda", OrtDeviceAllocator, /*id=*/1, OrtMemTypeDefault);

  {
    Qwen38Plane plane;
    InitMockPlane(plane);
    {
      // One worker per device entry. The thread that builds the pool stays on device 0: the workers must not inherit
      // their device from it (CUDA starts every thread on device 0).
      Qwen38WeightPool pool{plane, manifest_path, dir, *mem_info, 2};
      EXPECT_EQ(pool.bytes, 64u);
    }
    ASSERT_EQ(MockPlaneTracker::device_uses.size(), 4u);  // host_alloc + stream_create of each of the two workers
    for (const auto& [call, device] : MockPlaneTracker::device_uses)
      EXPECT_EQ(device, 1) << call << " ran on device " << device;
    EXPECT_TRUE(MockPlaneTracker::live_device_ptrs.empty());
  }

  // A worker that cannot select the device fails the construction and gives back what was allocated.
  {
    Qwen38Plane plane;
    InitMockPlane(plane);
    plane.set_device = [](int) { return 101; };  // cudaErrorInvalidDevice
    try {
      Qwen38WeightPool pool{plane, manifest_path, dir, *mem_info, 2};
      ADD_FAILURE() << "a failing cudaSetDevice must fail the construction";
    } catch (const std::runtime_error& e) {
      EXPECT_NE(std::string(e.what()).find("cudaSetDevice"), std::string::npos) << e.what();
    }
    EXPECT_TRUE(MockPlaneTracker::device_uses.empty()) << "nothing may be created on a device that was not selected";
    EXPECT_TRUE(MockPlaneTracker::live_device_ptrs.empty());
  }

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38WeightPoolTest, AFailedDeviceCopyUnregistersTheHostRangesAndFreesTheDeviceMemory) {
  const auto dir = MakeScratchDir("weight_pool_short_read");
  WriteFile(dir / "w0.bin", std::string(8, 'a'));  // the entry below declares 32 bytes
  WriteFile(dir / "embed.bin", std::string(128, 'e'));
  const auto manifest_path = WriteFile(dir / "pool.txt",
                                       "qwen38_weights_pool 1\n"
                                       "pool 0 1 0 32 1 8 w0.bin\n"
                                       "host_pool 1 1 32 64 2 2 8 embed.bin\n"
                                       "session target w0 0\n"
                                       "session target embed 1\n");
  auto mem_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

  Qwen38Plane plane;
  InitMockPlane(plane);
  try {
    Qwen38WeightPool pool{plane, manifest_path, dir, *mem_info, 1};
    ADD_FAILURE() << "a weight file shorter than its entry must fail the construction";
  } catch (const std::runtime_error& e) {
    EXPECT_NE(std::string(e.what()).find("short read"), std::string::npos) << e.what();
  }
  // The host range was registered before the upload failed: the failure path must give it back, and the device copy.
  EXPECT_EQ(MockPlaneTracker::registered_host_ptrs.size(), 1u);
  EXPECT_EQ(MockPlaneTracker::unregistered_host_ptrs, MockPlaneTracker::registered_host_ptrs);
  EXPECT_TRUE(MockPlaneTracker::live_host_ranges.empty());
  EXPECT_TRUE(MockPlaneTracker::live_device_ptrs.empty());

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
#endif

}  // namespace Generators::test
