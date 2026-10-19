// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
  static inline int alloc_calls = 0;
  static inline int fail_on_alloc_call = -1;
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
  EXPECT_EQ(config.model.qwen38_graph.target_s1_graph_id, 41);
  EXPECT_EQ(config.model.qwen38_graph.target_s2_graph_id, 51);
  EXPECT_EQ(config.model.qwen38_graph.head_s1_graph_id, 61);
  EXPECT_EQ(config.model.qwen38_graph.head_s2_graph_id, 63);
  EXPECT_EQ(config.model.qwen38_graph.prefill_chunk_tokens, 2048);
  EXPECT_TRUE(config.model.qwen38_graph.warmup);
  EXPECT_TRUE(config.model.qwen38_graph.mtp.enabled);
  EXPECT_TRUE(config.model.qwen38_graph.mtp.prefill_head);
  EXPECT_FALSE(config.model.qwen38_graph.mtp.skip_refeed);
}

TEST(Qwen38GraphConfigTest, ParsesEverySetting) {
  Config config;
  OverlayConfig(config, R"({"model":{"type":"qwen38_flash_graph","qwen38_graph":{
    "plane_library":"libs/libqwen38_plane.so",
    "ple_table":"ple_table.txt",
    "kv_capacity":1024,
    "target_s1_graph_id":10,
    "target_s2_graph_id":20,
    "head_s1_graph_id":30,
    "head_s2_graph_id":40,
    "prefill_chunk_tokens":512,
    "warmup":false,
    "mtp":{"enabled":false,"prefill_head":false,"skip_refeed":true},
    "head":{"filename_s1":"head/s1.onnx","filename_s2":"head/s2.onnx",
            "session_options":{"custom_ops_libraries":["libs/a.so","libs/b.so"],
                               "provider_options":[{"cuda":{"enable_cuda_graph":"1"}}]}}
  }}})");

  const auto& q = config.model.qwen38_graph;
  EXPECT_TRUE(q.IsConfigured());
  EXPECT_EQ(q.plane_library, "libs/libqwen38_plane.so");
  EXPECT_EQ(q.ple_table, "ple_table.txt");
  EXPECT_EQ(q.kv_capacity, 1024);
  EXPECT_EQ(q.target_s1_graph_id, 10);
  EXPECT_EQ(q.target_s2_graph_id, 20);
  EXPECT_EQ(q.head_s1_graph_id, 30);
  EXPECT_EQ(q.head_s2_graph_id, 40);
  EXPECT_EQ(q.prefill_chunk_tokens, 512);
  EXPECT_FALSE(q.warmup);
  EXPECT_FALSE(q.mtp.enabled);
  EXPECT_FALSE(q.mtp.prefill_head);
  EXPECT_TRUE(q.mtp.skip_refeed);
  EXPECT_EQ(q.head.filename_s1, "head/s1.onnx");
  EXPECT_EQ(q.head.filename_s2, "head/s2.onnx");
  ASSERT_TRUE(q.head.session_options.has_value());
  EXPECT_EQ(q.head.session_options->custom_ops_libraries, (std::vector<std::string>{"libs/a.so", "libs/b.so"}));
}

TEST(Qwen38GraphConfigTest, FoldAndPrefillDefaultToOff) {
  Config config;
  const auto& q = config.model.qwen38_graph;
  EXPECT_FALSE(q.fold.enabled);
  EXPECT_FALSE(q.fold.force_reject);
  EXPECT_EQ(q.fold.target_graph_id, 71);
  EXPECT_EQ(q.fold.head_r2.graph_id, 81);
  EXPECT_FALSE(q.prefill.enabled);
  EXPECT_EQ(q.prefill.chunk, 8192);
  EXPECT_EQ(q.prefill.warmup_lengths, (std::vector<int>{8192, 2048, 512, 128, 37}));
}

TEST(Qwen38GraphConfigTest, ParsesTheFoldAndPrefillSections) {
  Config config;
  OverlayConfig(config, R"({"model":{"type":"qwen38_flash_graph","qwen38_graph":{
    "plane_library":"libs/libqwen38_plane.so",
    "ple_table":"ple_table.txt",
    "fold":{"enabled":true,"target_graph_id":90,"force_reject":true,
            "head_r2":{"filename":"head_r2/model.onnx","graph_id":95,
                       "session_options":{"custom_ops_libraries":["libs/a.so"]}}},
    "prefill":{"enabled":true,"chunk":4096,"weights_pool":"pool.txt","constants":"qsa.txt",
               "dense":{"filename":"dense/model.onnx","session_options":{"provider_options":[{"cuda":{}}]}},
               "head_kv":{"filename":"head_kv/model.onnx"},
               "warmup_lengths":[64,16]}
  }}})");

  const auto& q = config.model.qwen38_graph;
  EXPECT_TRUE(q.IsConfigured());  // no head.filename_s1: the fold head_r2 session is enough
  EXPECT_TRUE(q.fold.enabled);
  EXPECT_TRUE(q.fold.force_reject);
  EXPECT_EQ(q.fold.target_graph_id, 90);
  EXPECT_EQ(q.fold.head_r2.filename, "head_r2/model.onnx");
  EXPECT_EQ(q.fold.head_r2.graph_id, 95);
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

TEST(Qwen38GraphConfigTest, FoldAcceptsMtpAndDrafterAliases) {
  Config c1, c2;
  OverlayConfig(c1, R"({"model":{"qwen38_graph":{"fold":{"enabled":true,"mtp":{"filename":"mtp.onnx","graph_id":85}}}}})");
  EXPECT_EQ(c1.model.qwen38_graph.fold.head_r2.filename, "mtp.onnx");
  EXPECT_EQ(c1.model.qwen38_graph.fold.head_r2.graph_id, 85);

  OverlayConfig(c2, R"({"model":{"qwen38_graph":{"fold":{"enabled":true,"drafter":{"filename":"drafter.onnx"}}}}})");
  EXPECT_EQ(c2.model.qwen38_graph.fold.head_r2.filename, "drafter.onnx");
}

TEST(Qwen38GraphConfigTest, EffectivePrefillChunkUsesStepOrFastPrefillSetting) {
  Config config;
  EXPECT_EQ(EffectiveQwen38PrefillChunkTokens(config.model.qwen38_graph), 2048u);

  OverlayConfig(config, R"({"model":{"qwen38_graph":{"prefill_chunk_tokens":512}}})");
  EXPECT_EQ(EffectiveQwen38PrefillChunkTokens(config.model.qwen38_graph), 512u);

  OverlayConfig(config, R"({"model":{"qwen38_graph":{"prefill":{"enabled":true,"chunk":4096}}}})");
  EXPECT_EQ(EffectiveQwen38PrefillChunkTokens(config.model.qwen38_graph), 4096u);

  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"prefill_chunk_tokens":0}}})"), std::runtime_error);
  EXPECT_THROW(OverlayConfig(config, R"({"model":{"qwen38_graph":{"prefill_chunk_tokens":15}}})"), std::runtime_error);
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
  EXPECT_NO_THROW(check_index("qwen38_qsa_consts 1\nfile qsa_consts.bin\nepsilon 1e-06\nwidth 64\n"
                              "tensor target cos 0 512\n"
                              "tensor target sin 512 512\n"
                              "tensor target knorm.3 1024 256\n"
                              "tensor head cos 1280 512\n"
                              "tensor head sin 1792 512\n"
                              "tensor head knorm.0 2304 256\n"));

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
#endif

}  // namespace Generators::test
