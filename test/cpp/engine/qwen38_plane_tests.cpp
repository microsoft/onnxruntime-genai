// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "logging.h"
#include "models/qwen38_graph_plane.h"
#include "scoped-environment-variable.h"

#if !defined(QWEN38_GRAPH_NO_POSIX)
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

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

constexpr const char* kValidPleTable =
    "qwen38_ple_table 1\n"
    "n_heads 2\n"
    "head_dim 4\n"
    "heads_per_ngram 1\n"
    "eos 7\n"
    "scale 0.5\n"
    "mult 1 2 3 0\n"
    "vocab 11 13\n"
    "offset 0 11\n"
    "shard ple/a.safetensors 128 11\n"
    "shard ple/b.safetensors 0 13\n";

// The stubs serve the mapping tests, which run wherever the loader maps files (POSIX and Windows); they are guarded like
// those tests so that no build is left with an unused helper.
#if !defined(QWEN38_GRAPH_NO_POSIX)
struct StubPlaneState {
  static inline std::vector<void*> registered_ptrs;
  static inline std::vector<void*> unregistered_ptrs;
  static inline bool fail_get_device_pointer = false;
  static constexpr uintptr_t kDeviceOffset = 0x100000u;

  static void Reset() {
    registered_ptrs.clear();
    unregistered_ptrs.clear();
    fail_get_device_pointer = false;
  }

  static const char* ErrorString(int) { return "stub error"; }
  static int HostRegister(void* ptr, size_t, unsigned) {
    registered_ptrs.push_back(ptr);
    return 0;
  }
  static int HostUnregister(void* ptr) {
    unregistered_ptrs.push_back(ptr);
    return 0;
  }
  static int HostGetDevicePointer(void** dev_ptr, void* host_ptr, unsigned) {
    if (fail_get_device_pointer) {
      return 1;
    }
    *dev_ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(host_ptr) + kDeviceOffset);
    return 0;
  }
};

void InitStubPlane(Qwen38Plane& plane) {
  StubPlaneState::Reset();
  plane.error_string = &StubPlaneState::ErrorString;
  plane.host_register = &StubPlaneState::HostRegister;
  plane.host_unregister = &StubPlaneState::HostUnregister;
  plane.host_get_device_pointer = &StubPlaneState::HostGetDevicePointer;
}
#endif

}  // namespace

// ---- weight pool manifest and QSA constants index --------------------------------------------

TEST(Qwen38WeightPoolManifestTest, ParsesEntriesAndSessions) {
  const auto dir = MakeScratchDir("pool");
  const auto path = WriteFile(dir / "pool.txt",
                              "qwen38_weights_pool 1\n"
                              "pool 0 16 1000 64 2 4 8 target/w0.safetensors\n"
                              "pool 1 2 0 128 3 2 4 8 target/experts.data\n"
                              "host_pool 2 16 4096 512 2 16 16 target/embed.safetensors\n"
                              "session target model.embed.weight 2\n"
                              "session target experts.w 1\n"
                              "session dense model.embed.weight 2\n"
                              "session mtp head.w 0\n"
                              "session drafter head.w 0\n");
  const auto manifest = ParseQwen38WeightPoolManifest(path);
  ASSERT_EQ(manifest.entries.size(), 3u);
  EXPECT_EQ(manifest.entries[0].dtype, 16);
  EXPECT_EQ(manifest.entries[0].offset, 1000u);
  EXPECT_EQ(manifest.entries[0].length, 64u);
  EXPECT_EQ(manifest.entries[0].shape, (std::vector<int64_t>{4, 8}));
  EXPECT_EQ(manifest.entries[0].file, "target/w0.safetensors");
  EXPECT_EQ(manifest.entries[0].placement, "device");
  EXPECT_FALSE(manifest.entries[0].host_mapped);
  EXPECT_EQ(manifest.entries[1].shape, (std::vector<int64_t>{2, 4, 8}));
  EXPECT_FALSE(manifest.entries[1].host_mapped);
  EXPECT_EQ(manifest.entries[2].file, "target/embed.safetensors");
  EXPECT_EQ(manifest.entries[2].placement, "host_mapped");
  EXPECT_TRUE(manifest.entries[2].host_mapped);
  ASSERT_EQ(manifest.sessions.at("target").size(), 2u);
  EXPECT_EQ(manifest.sessions.at("target")[1], (std::pair<std::string, int>{"experts.w", 1}));
  EXPECT_EQ(manifest.sessions.at("dense").size(), 1u);
  EXPECT_EQ(manifest.sessions.at("mtp").size(), 1u);
  EXPECT_EQ(manifest.sessions.at("drafter").size(), 1u);
  EXPECT_EQ(manifest.sessions.count("head_kv"), 0u);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38WeightPoolManifestTest, RejectsMalformedManifests) {
  const auto dir = MakeScratchDir("pool_bad");
  const auto parse = [&](const std::string& text) { return ParseQwen38WeightPoolManifest(WriteFile(dir / "p.txt", text)); };
  EXPECT_THROW(ParseQwen38WeightPoolManifest(dir / "missing.txt"), std::runtime_error);
  EXPECT_THROW(parse("qwen38_weights_pool 2\n"), std::runtime_error);
  EXPECT_THROW(parse("qwen38_weights_pool 1\npool 1 16 0 8 1 4 a.bin\n"), std::runtime_error);     // ids must start at 0
  EXPECT_THROW(parse("qwen38_weights_pool 1\npool 0 16 0 8 1 4 ../a.bin\n"), std::runtime_error);  // escapes the folder
  EXPECT_THROW(parse("qwen38_weights_pool 1\npool 0 16 0 8 1 4 /abs/a.bin\n"), std::runtime_error);
  EXPECT_THROW(parse("qwen38_weights_pool 1\npool 0 16 0 8 1 4\n"), std::runtime_error);                       // no file
  EXPECT_THROW(parse("qwen38_weights_pool 1\npool 0 16 0 8 1 4 a.bin invalid_place\n"), std::runtime_error);   // bad placement
  EXPECT_THROW(parse("qwen38_weights_pool 1\npool 0 16 0 8 1 4 a.bin\nsession t n 1\n"), std::runtime_error);  // unknown id
  EXPECT_NO_THROW(parse("qwen38_weights_pool 1\n"));
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38QsaConstantsIndexTest, ParsesTheIndex) {
  const auto dir = MakeScratchDir("qsa");
  const auto path = WriteFile(dir / "qsa.txt",
                              "qwen38_qsa_consts 1\n"
                              "file qsa_consts.bin\n"
                              "epsilon 1e-06\n"
                              "width 64\n"
                              "tensor target cos 0 1024\n"
                              "tensor head knorm.0 1024 256\n");
  const auto description = ParseQwen38QsaConstantsIndex(path);
  EXPECT_EQ(description.file, "qsa_consts.bin");
  EXPECT_FLOAT_EQ(description.epsilon, 1e-6f);
  EXPECT_EQ(description.width, 64);
  ASSERT_EQ(description.items.size(), 2u);
  EXPECT_EQ(description.items[1].set, "head");
  EXPECT_EQ(description.items[1].name, "knorm.0");
  EXPECT_EQ(description.items[1].offset, 1024u);
  EXPECT_EQ(description.items[1].bytes, 256u);
  EXPECT_THROW(ParseQwen38QsaConstantsIndex(WriteFile(dir / "bad.txt", "qwen38_qsa_consts 2\n")), std::runtime_error);
  EXPECT_THROW(ParseQwen38QsaConstantsIndex(WriteFile(dir / "bad2.txt", "qwen38_qsa_consts 1\nfile ../x.bin\n")),
               std::runtime_error);
  EXPECT_THROW(ParseQwen38QsaConstantsIndex(WriteFile(dir / "bad3.txt", "qwen38_qsa_consts 1\nfile x.bin\ntensor target cos 0\n")),
               std::runtime_error);
  EXPECT_THROW(ParseQwen38QsaConstantsIndex(
                   WriteFile(dir / "bad_eps1.txt", "qwen38_qsa_consts 1\nfile x.bin\nepsilon bad\nwidth 64\n")),
               std::runtime_error);
  EXPECT_THROW(ParseQwen38QsaConstantsIndex(
                   WriteFile(dir / "bad_eps2.txt", "qwen38_qsa_consts 1\nfile x.bin\nepsilon 0\nwidth 64\n")),
               std::runtime_error);
  EXPECT_THROW(ParseQwen38QsaConstantsIndex(
                   WriteFile(dir / "bad_width1.txt", "qwen38_qsa_consts 1\nfile x.bin\nepsilon 1e-6\nwidth bad\n")),
               std::runtime_error);
  EXPECT_THROW(ParseQwen38QsaConstantsIndex(
                   WriteFile(dir / "bad_width2.txt", "qwen38_qsa_consts 1\nfile x.bin\nepsilon 1e-6\nwidth 0\n")),
               std::runtime_error);
  EXPECT_THROW(ParseQwen38QsaConstantsIndex(WriteFile(dir / "missing_meta.txt", "qwen38_qsa_consts 1\nfile x.bin\n")),
               std::runtime_error);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// ---- PLE table description -------------------------------------------------------------------

TEST(Qwen38PleDescriptionTest, ParsesAValidTable) {
  const auto dir = MakeScratchDir("ple_valid");
  const auto description = ParseQwen38PleDescription(WriteFile(dir / "ple_table.txt", kValidPleTable));

  EXPECT_EQ(description.params.n_heads, 2);
  EXPECT_EQ(description.params.head_dim, 4);
  EXPECT_EQ(description.params.heads_per_ngram, 1);
  EXPECT_EQ(description.params.eos, 7);
  EXPECT_FLOAT_EQ(description.params.scale, 0.5f);
  EXPECT_EQ(description.params.mult[2], 3);
  EXPECT_EQ(description.params.vocab[1], 13);
  EXPECT_EQ(description.params.offset[1], 11);
  EXPECT_EQ(description.params.n_shards, 2);
  ASSERT_EQ(description.shards.size(), 2u);
  EXPECT_EQ(description.shards[0].path, "ple/a.safetensors");
  EXPECT_EQ(description.shards[0].offset, 128u);
  EXPECT_EQ(description.shards[1].rows, 13);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38PleDescriptionTest, RejectsMalformedTables) {
  const auto dir = MakeScratchDir("ple_invalid");
  const auto table = dir / "ple_table.txt";
  const auto parse = [&](std::string text) {
    const std::string header = "qwen38_ple_table 1\n";
    if (text.rfind(header, 0) == 0) {
      if (text.find("\nscale ") == std::string::npos) text.insert(header.size(), "scale 0.5\n");
      if (text.find("\nmult ") == std::string::npos) text.insert(header.size(), "mult 1 2 3 0\n");
    }
    return ParseQwen38PleDescription(WriteFile(table, text));
  };
  const auto parse_raw = [&](const std::string& text) { return ParseQwen38PleDescription(WriteFile(table, text)); };

  EXPECT_THROW(ParseQwen38PleDescription(dir / "missing.txt"), std::runtime_error);
  EXPECT_THROW(parse("qwen38_ple_table 2\n"), std::runtime_error);                         // header
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\n"), std::runtime_error);  // no shards
  EXPECT_THROW(parse("qwen38_ple_table 1\nvocab 1 2\nn_heads 2\n"), std::runtime_error);   // vocab before n_heads
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 33\n"), std::runtime_error);
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nshard ple/a.bin 0\n"), std::runtime_error);  // rows
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nshard ple/a.bin 0 0\n"), std::runtime_error);

  // Missing scale or mult, or zero/non-finite scale.
  EXPECT_THROW(parse_raw("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nheads_per_ngram 1\nmult 1 2 3 0\n"
                         "vocab 4 4\noffset 0 4\nshard ple/a.bin 0 8\n"),
               std::runtime_error);
  EXPECT_THROW(parse_raw("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nheads_per_ngram 1\nscale 0.5\n"
                         "vocab 4 4\noffset 0 4\nshard ple/a.bin 0 8\n"),
               std::runtime_error);
  EXPECT_THROW(parse_raw("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nheads_per_ngram 1\nscale 0\nmult 1 2 3 0\n"
                         "vocab 4 4\noffset 0 4\nshard ple/a.bin 0 8\n"),
               std::runtime_error);

  // Missing vocab/offset, zero/negative vocab, negative/out-of-bounds offset, out-of-range heads_per_ngram, or width > 3072.
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nheads_per_ngram 1\noffset 0 4\nshard ple/a.bin 0 8\n"),
               std::runtime_error);
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nheads_per_ngram 1\nvocab 0 4\noffset 0 4\nshard ple/a.bin 0 8\n"),
               std::runtime_error);
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nheads_per_ngram 1\nvocab 4 -2\noffset 0 4\nshard ple/a.bin 0 8\n"),
               std::runtime_error);
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nheads_per_ngram 1\nvocab 4 4\noffset -1 4\nshard ple/a.bin 0 8\n"),
               std::runtime_error);
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nheads_per_ngram 1\nvocab 4 5\noffset 0 4\nshard ple/a.bin 0 8\n"),
               std::runtime_error);
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nheads_per_ngram -1\nvocab 4 4\noffset 0 4\nshard ple/a.bin 0 8\n"),
               std::runtime_error);
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nheads_per_ngram 3\nvocab 4 4\noffset 0 4\nshard ple/a.bin 0 8\n"),
               std::runtime_error);
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 32\nhead_dim 128\nheads_per_ngram 8\n"
                     "vocab 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n"
                     "offset 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
                     "shard ple/a.bin 0 4\n"),
               std::runtime_error);

  // Shards must stay inside the model folder.
  for (const char* path : {"../a.bin", "ple/../../a.bin", "/etc/a.bin"}) {
    EXPECT_THROW(parse(std::string("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nvocab 2 2\noffset 0 2\nshard ") + path +
                       " 0 4\n"),
                 std::runtime_error)
        << path;
  }

  std::string too_many = "qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nvocab 1 1\noffset 0 1\n";
  for (int i = 0; i < 257; ++i) too_many += "shard ple/a.bin 0 1\n";
  EXPECT_THROW(parse(too_many), std::runtime_error);
  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

#if !defined(QWEN38_GRAPH_NO_POSIX)
TEST(Qwen38PleTableTest, ValidatesShardFileBoundsAndUsesDevicePointer) {
#if defined(_WIN32)
  // Without the pin the Windows table maps its shards without registering them; the assertions below are about registration.
  ScopedEnvironmentVariable pin_ple{"ORT_QMOE_WIN_PIN_PLE", "1"};
#endif
  const auto dir = MakeScratchDir("ple_map");
  std::filesystem::create_directories(dir / "ple");
  const auto table_path = WriteFile(dir / "ple_table.txt", kValidPleTable);

  // Write shard a with 128 + 11*4 = 172 bytes, but shard b truncated to only 20 bytes (needs 13*4 = 52 bytes).
  WriteFile(dir / "ple" / "a.safetensors", std::string(172, 'a'));
  WriteFile(dir / "ple" / "b.safetensors", std::string(20, 'b'));
  {
    Qwen38Plane plane;
    InitStubPlane(plane);
    EXPECT_THROW((Qwen38PleTable{plane, table_path, dir, 64}), std::runtime_error);
    // Shard 0 was registered before shard 1 failed; verify cleanup unregistered shard 0.
    EXPECT_EQ(StubPlaneState::registered_ptrs.size(), 1u);
    EXPECT_EQ(StubPlaneState::unregistered_ptrs, StubPlaneState::registered_ptrs);
  }

  // Write full shard b (52 bytes) and verify device pointers come from host_get_device_pointer while unregister uses host ptrs.
  WriteFile(dir / "ple" / "b.safetensors", std::string(52, 'b'));
  {
    Qwen38Plane plane;
    InitStubPlane(plane);
    {
      Qwen38PleTable ple{plane, table_path, dir, 64};
      ASSERT_EQ(StubPlaneState::registered_ptrs.size(), 2u);
      for (size_t i = 0; i < 2; ++i) {
        const auto expected_dev =
            reinterpret_cast<const unsigned char*>(reinterpret_cast<uintptr_t>(StubPlaneState::registered_ptrs[i]) +
                                                   StubPlaneState::kDeviceOffset);
        EXPECT_EQ(ple.params.shard_ptr[i], expected_dev);
        EXPECT_NE(ple.params.shard_ptr[i], static_cast<const unsigned char*>(StubPlaneState::registered_ptrs[i]));
      }
    }
    EXPECT_EQ(StubPlaneState::unregistered_ptrs, StubPlaneState::registered_ptrs);
  }

  // Also verify Qwen38FileMapping queries host_get_device_pointer and unregisters the host base pointer.
  {
    Qwen38Plane plane;
    InitStubPlane(plane);
    {
      Qwen38FileMapping mapping{plane, dir / "ple" / "a.safetensors", true};
      ASSERT_EQ(StubPlaneState::registered_ptrs.size(), 1u);
      EXPECT_EQ(mapping.data(), static_cast<uint8_t*>(StubPlaneState::registered_ptrs[0]));
      EXPECT_EQ(reinterpret_cast<uintptr_t>(mapping.device_data()),
                reinterpret_cast<uintptr_t>(mapping.data()) + StubPlaneState::kDeviceOffset);
    }
    EXPECT_EQ(StubPlaneState::unregistered_ptrs, StubPlaneState::registered_ptrs);
  }

  // Verify Qwen38FileMapping rolls back registration when host_get_device_pointer fails.
  {
    Qwen38Plane plane;
    InitStubPlane(plane);
    StubPlaneState::fail_get_device_pointer = true;
    EXPECT_THROW((Qwen38FileMapping{plane, dir / "ple" / "a.safetensors", true}), std::runtime_error);
    EXPECT_EQ(StubPlaneState::registered_ptrs.size(), 1u);
    EXPECT_EQ(StubPlaneState::unregistered_ptrs, StubPlaneState::registered_ptrs);
  }

  // A relative shard path that resolves through a symlink outside the model root must be rejected before opening/registering.
  {
    const auto outside_dir = MakeScratchDir("ple_outside");
    const auto outside_file = WriteFile(outside_dir / "external.safetensors", std::string(256, 'x'));
    std::error_code sym_ec;
    std::filesystem::remove(dir / "ple" / "b.safetensors", sym_ec);
    std::filesystem::create_symlink(outside_file, dir / "ple" / "b.safetensors", sym_ec);
    if (!sym_ec) {
      Qwen38Plane plane;
      InitStubPlane(plane);
      EXPECT_THROW((Qwen38PleTable{plane, table_path, dir, 64}), std::runtime_error);
      EXPECT_EQ(StubPlaneState::registered_ptrs.size(), 1u);
      EXPECT_EQ(StubPlaneState::unregistered_ptrs, StubPlaneState::registered_ptrs);
    }
    std::filesystem::remove_all(outside_dir, sym_ec);
  }

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// The weight pool registers the page-aligned ranges it uses. A registration of the whole file would make each of them fail with
// cudaErrorHostMemoryAlreadyRegistered, so a mapping that is not asked to register itself registers nothing, whatever the
// environment says.
TEST(Qwen38FileMappingTest, AMappingThatIsNotRegisteredLeavesTheRegistrationToItsOwner) {
  const auto dir = MakeScratchDir("file_map_unregistered");
  const auto path = WriteFile(dir / "pool.data", std::string(8192, 'p'));
  ScopedEnvironmentVariable pin_cpu_pool{"ORTGENAI_PIN_CPU_POOL", "1"};
  Qwen38Plane plane;
  InitStubPlane(plane);
  {
    Qwen38FileMapping mapping{plane, path, false};
    EXPECT_NE(mapping.data(), nullptr);
    EXPECT_EQ(mapping.size(), 8192u);
    EXPECT_TRUE(StubPlaneState::registered_ptrs.empty());
  }
  EXPECT_TRUE(StubPlaneState::unregistered_ptrs.empty());

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

#if !defined(_WIN32)
namespace {

std::string numa_log;

// Routes the GenAI logger into `numa_log` and puts the host's logging setup back afterwards.
struct ScopedNumaLogCapture {
  ScopedNumaLogCapture() : previous{g_log} {
    numa_log.clear();
    SetLogCallback([](const char* text, size_t length) { numa_log.append(text, length); });
  }
  ~ScopedNumaLogCapture() {
    SetLogCallback(nullptr);
    g_log = previous;
  }
  LogItems previous;
};

// Runs `prepare` and returns what it wrote to stderr.
std::string StderrDuring(const std::function<void()>& prepare) {
  testing::internal::CaptureStderr();
  try {
    prepare();
  } catch (...) {
    testing::internal::GetCapturedStderr();
    throw;
  }
  return testing::internal::GetCapturedStderr();
}

}  // namespace

// ORT_QMOE_NUMA_DIAGNOSTICS=1 asks for the placement report of a mapping. It reaches the host through the GenAI logger: not at
// all while logging is off, and at the destination the host chose (here a callback) while it is on. Nothing goes to stderr.
TEST(Qwen38FileMappingTest, TheNumaReportOfAMappingGoesThroughTheGenAiLogger) {
  const auto dir = MakeScratchDir("file_map_numa_report");
  const auto path = WriteFile(dir / "pool.data", std::string(8192, 'p'));
  // Audit only: the pages are looked at, neither the thread's memory policy nor any page is changed.
  ScopedEnvironmentVariable interleave{"ORT_QMOE_NUMA_INTERLEAVE", "0"};
  ScopedEnvironmentVariable migrate{"ORT_QMOE_NUMA_MIGRATE", "0"};
  ScopedEnvironmentVariable diagnostics{"ORT_QMOE_NUMA_DIAGNOSTICS", "1"};
  {
    Qwen38Plane plane;
    InitStubPlane(plane);
    Qwen38FileMapping mapping{plane, path, false};
    ScopedNumaLogCapture capture;
    const auto prepare = [&] { mapping.PrepareNumaRange(0, mapping.size()); };

    g_log.enabled = false;
    EXPECT_EQ(StderrDuring(prepare), "");
    EXPECT_TRUE(numa_log.empty()) << numa_log;

    g_log.enabled = true;
    EXPECT_EQ(StderrDuring(prepare), "");
    EXPECT_NE(numa_log.find("[QMoE NUMA] file=pool.data offset=0 bytes=8192 "), std::string::npos) << numa_log;
  }

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

// An unaligned range makes the NUMA probe fail without touching the thread's memory policy or any page. The failure is a
// warning that the host can silence or redirect, not a line on stderr that it has no say in.
TEST(Qwen38FileMappingTest, ANumaProbeThatFailsIsAWarningTheGenAiLoggerControls) {
  const auto dir = MakeScratchDir("file_map_numa_failure");
  const auto path = WriteFile(dir / "pool.data", std::string(8192, 'p'));
  ScopedEnvironmentVariable interleave{"ORT_QMOE_NUMA_INTERLEAVE", "1"};
  ScopedEnvironmentVariable diagnostics{"ORT_QMOE_NUMA_DIAGNOSTICS", std::nullopt};
  {
    Qwen38Plane plane;
    InitStubPlane(plane);
    Qwen38FileMapping mapping{plane, path, false};
    ScopedNumaLogCapture capture;
    const auto prepare = [&] { mapping.PrepareNumaRange(1, 1); };

    for (const auto& [enabled, warning] : {std::pair{false, true}, std::pair{true, false}}) {
      g_log.enabled = enabled;
      g_log.warning = warning;
      numa_log.clear();
      EXPECT_EQ(StderrDuring(prepare), "") << "enabled=" << enabled << " warning=" << warning;
      EXPECT_TRUE(numa_log.empty()) << "enabled=" << enabled << " warning=" << warning << ": " << numa_log;
    }

    g_log.enabled = true;
    g_log.warning = true;
    EXPECT_EQ(StderrDuring(prepare), "");
    EXPECT_NE(numa_log.find("  warning  "), std::string::npos) << numa_log;
    EXPECT_NE(numa_log.find("[QMoE NUMA] file=pool.data offset=1 bytes=1 "), std::string::npos) << numa_log;

    // A report the host asked for is still delivered when it silenced the warnings; it is an info line then.
    ScopedEnvironmentVariable requested{"ORT_QMOE_NUMA_DIAGNOSTICS", "1"};
    g_log.warning = false;
    numa_log.clear();
    EXPECT_EQ(StderrDuring(prepare), "");
    EXPECT_NE(numa_log.find("  info  "), std::string::npos) << numa_log;
    EXPECT_EQ(numa_log.find("  warning  "), std::string::npos) << numa_log;
  }

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
#endif
#endif

// ---- plane library loader --------------------------------------------------------------------

TEST(Qwen38PlaneTest, ResolvesWindowsSharedLibraryPathFromSoToDll) {
  const auto dir = MakeScratchDir("resolve_dll");
  const auto so_path = dir / "libqwen38_plane.so";
  EXPECT_EQ(ResolveWindowsSharedLibraryPath(so_path), so_path);
  EXPECT_EQ(ResolveWindowsSharedLibraryPath(dir / "qwen38_plane.dll"), dir / "qwen38_plane.dll");

  // Stripped `lib` prefix (`qwen38_plane.dll`) is found when `libqwen38_plane.dll` is absent.
  WriteFile(dir / "qwen38_plane.dll", "dll");
  EXPECT_EQ(ResolveWindowsSharedLibraryPath(so_path), dir / "qwen38_plane.dll");

  // Direct `.dll` replacement (`libqwen38_plane.dll`) takes precedence when present.
  WriteFile(dir / "libqwen38_plane.dll", "dll");
  EXPECT_EQ(ResolveWindowsSharedLibraryPath(so_path), dir / "libqwen38_plane.dll");

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38PlaneTest, LoadingAMissingLibraryFailsWithItsPath) {
  const auto missing = std::filesystem::path{"definitely"} / "missing" / "libqwen38_plane.so";
  try {
    Qwen38Plane plane{missing};
    FAIL() << "loading a missing library must throw";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("qwen38"), std::string::npos) << message;
  }
}

#if !defined(QWEN38_GRAPH_NO_POSIX)
namespace {

std::filesystem::path GetBuiltPlaneLibraryPath() {
#if defined(ORTGENAI_TEST_QWEN38_PLANE_LIB_PATH)
  return std::filesystem::path{ORTGENAI_TEST_QWEN38_PLANE_LIB_PATH};
#elif defined(_WIN32)
  return std::filesystem::current_path() / "qwen38_plane.dll";
#else
  return std::filesystem::current_path() / "libqwen38_plane.so";
#endif
}

// Whether the library file is mapped in the process, without loading it.
bool IsLibraryMapped(const std::filesystem::path& path) {
#if defined(_WIN32)
  return GetModuleHandleW(ResolveWindowsSharedLibraryPath(path).wstring().c_str()) != nullptr;
#else
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD);
  if (handle == nullptr) return false;
  dlclose(handle);
  return true;
#endif
}

}  // namespace

// The plane library holds process-wide state and starts threads of its own, and unloading a library that has launched CUDA
// kernels and loading it again crashed the CUDA runtime: the destructor of a loader detaches from the library but never unloads
// it, so the loader that follows shares the copy that is still mapped.
TEST(Qwen38PlaneTest, TheLibraryStaysMappedWhenItsLoaderIsDestroyed) {
  const std::filesystem::path built = GetBuiltPlaneLibraryPath();
  if (!std::filesystem::exists(built)) {
    GTEST_SKIP() << built.string() << " not present";
  }
  // A copy of its own, which no other test has mapped.
  const std::filesystem::path scratch = MakeScratchDir("plane_library_stays_mapped");
  const std::filesystem::path copy = scratch / built.filename();
  std::filesystem::copy_file(built, copy);
  ASSERT_FALSE(IsLibraryMapped(copy));

  {
    const Qwen38Plane plane{copy};
    EXPECT_TRUE(IsLibraryMapped(copy));
  }
  EXPECT_TRUE(IsLibraryMapped(copy)) << "the destructor unloaded the plane library";

  const Qwen38Plane again{copy};
  EXPECT_EQ(again.abi_version(), QWEN38_PLANE_ABI_VERSION);

  // The copy stays mapped until the process exits, which on Windows keeps its file from being removed.
  std::error_code ignored;
  std::filesystem::remove_all(scratch, ignored);
}

namespace {

// The bf16 bit pattern of `value`, which has to be exactly representable (or an infinity or a NaN).
uint16_t Bf16Bits(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

float FloatFromBits(uint32_t bits) {
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

}  // namespace

// The INT8 mode of quantize_kv saturates at +-127 whatever the quotient is: a quotient that overflows the INT8 range, an
// infinity from a scale close to zero or from an infinite input included, must not wrap to the other sign.
TEST(Qwen38PlaneTest, QuantizeKvSaturatesToInt8WhateverTheQuotient) {
  const std::filesystem::path plane_lib = GetBuiltPlaneLibraryPath();
  if (!std::filesystem::exists(plane_lib)) {
    GTEST_SKIP() << plane_lib.string() << " not present";
  }
  Qwen38Plane plane{plane_lib};
  if (plane.set_device(0) != 0 || !plane.quantize_kv) {
    GTEST_SKIP() << "No CUDA device or quantize_kv available";
  }
  void* stream = nullptr;
  plane.Check(plane.stream_create(&stream), "stream_create");

  constexpr int kChannels = 256;
  const float kInfinity = std::numeric_limits<float>::infinity();
  const float kNaN = std::numeric_limits<float>::quiet_NaN();
  const float kBf16Max = FloatFromBits(0x7F7F0000u);
  const struct {
    const char* what;
    float x;
    float scale;
    uint8_t expected;
  } cases[] = {
      {"a quotient far above the range", 1.0f, 1e-30f, 0x7F},
      {"a quotient far below the range", -1.0f, 1e-30f, 0x81},
      {"a quotient that overflows to +infinity", kBf16Max, 1e-30f, 0x7F},
      {"a quotient that overflows to -infinity", -kBf16Max, 1e-30f, 0x81},
      {"+infinity", kInfinity, 1.0f, 0x7F},
      {"-infinity", -kInfinity, 1.0f, 0x81},
      {"a NaN", kNaN, 1.0f, 0x00},
      {"a scale of zero", 1.0f, 0.0f, 0x00},
      {"a quotient of 1", 1.0f, 1.0f, 0x01},
      {"a half that rounds to even", -2.5f, 1.0f, 0xFE},
      {"a quotient of 2", 1.0f, 0.5f, 0x02},
      {"a quotient just above the range", 200.0f, 1.0f, 0x7F},
  };
  constexpr size_t kCases = sizeof(cases) / sizeof(cases[0]);

  std::vector<uint16_t> h_src(kChannels, Bf16Bits(0.0f));
  std::vector<float> h_scale(kChannels, 1.0f);
  std::vector<uint8_t> expected(kChannels, 0x00);
  for (size_t c = 0; c < kCases; ++c) {
    h_src[c] = Bf16Bits(cases[c].x);
    h_scale[c] = cases[c].scale;
    expected[c] = cases[c].expected;
  }
  PlaneBuffer d_src{plane, h_src.size() * sizeof(uint16_t), stream};
  PlaneBuffer d_scale{plane, h_scale.size() * sizeof(float), stream};
  PlaneBuffer d_dst{plane, kChannels, stream};
  plane.Check(plane.memcpy_h2d(d_src.data(), h_src.data(), h_src.size() * sizeof(uint16_t), stream), "memcpy_h2d(src)");
  plane.Check(plane.memcpy_h2d(d_scale.data(), h_scale.data(), h_scale.size() * sizeof(float), stream), "memcpy_h2d(scale)");
  plane.Check(plane.device_memset(d_dst.data(), 0xAB, kChannels, stream), "device_memset");

  plane.quantize_kv(stream, d_src.data(), d_dst.data(), d_scale.data(), /*rows=*/1, kChannels, kChannels, /*mode=*/1);
  std::vector<uint8_t> got(kChannels, 0);
  plane.Check(plane.memcpy_d2h(got.data(), d_dst.data(), got.size(), stream), "memcpy_d2h");
  plane.Check(plane.stream_sync(stream), "stream_sync");
  plane.stream_destroy(stream);

  for (size_t c = 0; c < kCases; ++c) {
    EXPECT_EQ(got[c], expected[c]) << cases[c].what;
  }
  for (size_t c = kCases; c < kChannels; ++c) {
    ASSERT_EQ(got[c], expected[c]) << "channel " << c;
  }
}

// embed_rows copies a row of `width` bf16 values wherever it starts: the vector copy is only for a row that starts on a 16-byte
// boundary, and the values after the last whole group of 8 are copied one by one, so no width loses elements or reads or writes
// across a boundary it is not aligned to.
TEST(Qwen38PlaneTest, EmbedRowsCopiesWholeRowsOfAnyWidthAndAlignment) {
  const std::filesystem::path plane_lib = GetBuiltPlaneLibraryPath();
  if (!std::filesystem::exists(plane_lib)) {
    GTEST_SKIP() << plane_lib.string() << " not present";
  }
  Qwen38Plane plane{plane_lib};
  if (plane.set_device(0) != 0 || !plane.embed_rows) {
    GTEST_SKIP() << "No CUDA device or embed_rows available";
  }
  void* stream = nullptr;
  plane.Check(plane.stream_create(&stream), "stream_create");

  constexpr size_t kTableRows = 4;
  const std::vector<int64_t> ids = {3, 0, 2, 2};
  for (const int width : {1, 7, 8, 12, 16, 20, 33}) {
    // The output starts on a 16-byte boundary, or 2 bytes past it (a view into a larger buffer): then no row is aligned.
    for (const size_t skew : {size_t{0}, size_t{1}}) {
      SCOPED_TRACE("width " + std::to_string(width) + ", output skewed by " + std::to_string(skew) + " elements");
      const size_t w = static_cast<size_t>(width);
      std::vector<uint16_t> table(kTableRows * w);
      for (size_t i = 0; i < table.size(); ++i) table[i] = static_cast<uint16_t>(0x3000 + i);
      std::vector<uint16_t> expected(skew + ids.size() * w, 0xABAB);
      for (size_t r = 0; r < ids.size(); ++r) {
        std::copy_n(table.begin() + static_cast<size_t>(ids[r]) * w, w, expected.begin() + skew + r * w);
      }

      PlaneBuffer d_table{plane, table.size() * sizeof(uint16_t), stream};
      PlaneBuffer d_ids{plane, ids.size() * sizeof(int64_t), stream};
      PlaneBuffer d_out{plane, expected.size() * sizeof(uint16_t), stream};
      plane.Check(plane.memcpy_h2d(d_table.data(), table.data(), table.size() * sizeof(uint16_t), stream), "memcpy_h2d(table)");
      plane.Check(plane.memcpy_h2d(d_ids.data(), ids.data(), ids.size() * sizeof(int64_t), stream), "memcpy_h2d(ids)");
      plane.Check(plane.device_memset(d_out.data(), 0xAB, d_out.size(), stream), "device_memset");

      plane.embed_rows(stream, d_table.data(), d_ids.data(), d_out.data() + skew * sizeof(uint16_t),
                       static_cast<int>(ids.size()), width);
      std::vector<uint16_t> got(expected.size(), 0);
      plane.Check(plane.memcpy_d2h(got.data(), d_out.data(), d_out.size(), stream), "memcpy_d2h");
      plane.Check(plane.stream_sync(stream), "stream_sync");
      ASSERT_EQ(got, expected);
    }
  }
  plane.stream_destroy(stream);
}

TEST(Qwen38PlaneTest, PlaneKernelsHandleEmptyAndNonMultipleOf32Dimensions) {
  const std::filesystem::path plane_lib = GetBuiltPlaneLibraryPath();
  if (!std::filesystem::exists(plane_lib)) {
    GTEST_SKIP() << plane_lib.string() << " not present";
  }
  Qwen38Plane plane{plane_lib};
  if (plane.set_device(0) != 0) {
    GTEST_SKIP() << "No CUDA device available";
  }
  void* stream = nullptr;
  plane.Check(plane.stream_create(&stream), "stream_create");

  // Empty batch calls (n <= 0 / batch <= 0) must be no-ops without launching a 0-block grid.
  EXPECT_EQ(plane.ple_rows(stream, nullptr, nullptr, 0, nullptr, nullptr, nullptr), 0);
  EXPECT_EQ(plane.ple_rows(stream, nullptr, nullptr, -1, nullptr, nullptr, nullptr), 0);
  plane.transpose_f32(stream, nullptr, nullptr, 0, 1);
  plane.transpose_f32(stream, nullptr, nullptr, 16, 0);
  if (plane.embed_rows) {
    plane.embed_rows(stream, nullptr, nullptr, nullptr, 0, 128);
    plane.embed_rows(stream, nullptr, nullptr, nullptr, 2, 0);
  }
  plane.Check(plane.stream_sync(stream), "stream_sync(empty launches)");

  // Non-multiple-of-32 transpose_f32 (n = 35, batch = 2).
  constexpr int kN = 35;
  constexpr int kBatch = 2;
  std::vector<float> h_src(kBatch * kN * kN);
  std::vector<float> h_dst(kBatch * kN * kN, -1.0f);
  for (int b = 0; b < kBatch; ++b) {
    for (int r = 0; r < kN; ++r) {
      for (int c = 0; c < kN; ++c) {
        h_src[(b * kN + r) * kN + c] = static_cast<float>(b * 10000 + r * 100 + c);
      }
    }
  }
  PlaneBuffer d_src{plane, h_src.size() * sizeof(float), stream};
  PlaneBuffer d_dst{plane, h_dst.size() * sizeof(float), stream};
  plane.Check(plane.memcpy_h2d(d_src.data(), h_src.data(), h_src.size() * sizeof(float), stream), "memcpy_h2d");
  plane.transpose_f32(stream, d_src.data(), d_dst.data(), kN, kBatch);
  plane.Check(plane.memcpy_d2h(h_dst.data(), d_dst.data(), h_dst.size() * sizeof(float), stream), "memcpy_d2h");
  plane.Check(plane.stream_sync(stream), "stream_sync(transpose)");
  plane.stream_destroy(stream);

  for (int b = 0; b < kBatch; ++b) {
    for (int r = 0; r < kN; ++r) {
      for (int c = 0; c < kN; ++c) {
        EXPECT_FLOAT_EQ(h_dst[(b * kN + c) * kN + r], h_src[(b * kN + r) * kN + c])
            << "b=" << b << " r=" << r << " c=" << c;
      }
    }
  }
}

TEST(Qwen38PlaneTest, StagedPleIsolatesMultipleTablesIgnoresUnrelatedMemcpyAndTearsDown) {
  const std::filesystem::path plane_lib = GetBuiltPlaneLibraryPath();
  if (!std::filesystem::exists(plane_lib)) {
    GTEST_SKIP() << plane_lib.string() << " not present";
  }
  Qwen38Plane plane{plane_lib};
  if (plane.set_device(0) != 0 || !plane.set_ple_host_params) {
    GTEST_SKIP() << "No CUDA device or staged PLE support available";
  }
  void* stream = nullptr;
  plane.Check(plane.stream_create(&stream), "stream_create");

  // Two distinct staged PLE tables (`pad_ = 1`) with 1 head, head_dim = 4, 4 rows.
  // FP8 E4M3 0x38 = 1.0f, 0x40 = 2.0f.
  std::vector<unsigned char> shard1(16, 0x38u);  // 1.0f * scale(1.0f) -> bf16 1.0f (0x3F80)
  std::vector<unsigned char> shard2(16, 0x40u);  // 2.0f * scale(2.0f) -> bf16 4.0f (0x4080)

  auto make_staged_params = [](const unsigned char* shard_ptr, float scale) {
    Qwen38PleParams p{};
    p.n_heads = 1;
    p.head_dim = 4;
    p.heads_per_ngram = 1;
    p.n_shards = 1;
    p.mult[0] = 1;
    p.vocab[0] = 4;
    p.offset[0] = 0;
    p.shard_end[0] = 4;
    p.shard_ptr[0] = shard_ptr;
    p.scale = scale;
    p.pad_ = 1;
    p.eos = 99;
    p.capacity = 16;
    return p;
  };

  Qwen38PleParams p1 = make_staged_params(shard1.data(), 1.0f);
  Qwen38PleParams p2 = make_staged_params(shard2.data(), 2.0f);
  plane.set_ple_host_params(&p1);
  plane.set_ple_host_params(&p2);

  PlaneBuffer d_p1{plane, sizeof(Qwen38PleParams), stream};
  PlaneBuffer d_p2{plane, sizeof(Qwen38PleParams), stream};
  plane.Check(plane.memcpy_h2d(d_p1.data(), &p1, sizeof(Qwen38PleParams), stream), "upload p1");
  plane.Check(plane.memcpy_h2d(d_p2.data(), &p2, sizeof(Qwen38PleParams), stream), "upload p2");

  // Unrelated memcpy_h2d of sizeof(Qwen38PleParams) whose leading ints look like valid n_heads/head_dim/n_shards.
  Qwen38PleParams fake_payload{};
  fake_payload.n_heads = 1;
  fake_payload.head_dim = 4;
  fake_payload.n_shards = 1;
  fake_payload.pad_ = 1;
  fake_payload.shard_ptr[0] = reinterpret_cast<const unsigned char*>(uintptr_t{0xDEADBEEFu});
  PlaneBuffer d_unrelated{plane, sizeof(Qwen38PleParams), stream};
  plane.Check(plane.memcpy_h2d(d_unrelated.data(), &fake_payload, sizeof(Qwen38PleParams), stream),
              "unrelated memcpy_h2d");

  const int64_t h_tokens[2] = {1, 2};
  const int64_t h_ctx_in[2] = {0, 0};
  PlaneBuffer d_tokens{plane, sizeof(h_tokens), stream};
  PlaneBuffer d_ctx_in{plane, sizeof(h_ctx_in), stream};
  PlaneBuffer d_ctx_out{plane, sizeof(h_ctx_in), stream};
  PlaneBuffer d_out1{plane, 2 * 4 * sizeof(uint16_t), stream};
  PlaneBuffer d_out2{plane, 2 * 4 * sizeof(uint16_t), stream};
  plane.Check(plane.memcpy_h2d(d_tokens.data(), h_tokens, sizeof(h_tokens), stream), "tokens");
  plane.Check(plane.memcpy_h2d(d_ctx_in.data(), h_ctx_in, sizeof(h_ctx_in), stream), "ctx_in");

  // Both tables must gather from their own shard and cache independently despite p2 being registered second.
  EXPECT_EQ(plane.ple_rows(stream, d_p1.data(), d_tokens.data(), 2, d_ctx_in.data(), d_ctx_out.data(), d_out1.data()), 0);
  EXPECT_EQ(plane.ple_rows(stream, d_p2.data(), d_tokens.data(), 2, d_ctx_in.data(), d_ctx_out.data(), d_out2.data()), 0);

  std::vector<uint16_t> out1(8, 0), out2(8, 0);
  plane.Check(plane.memcpy_d2h(out1.data(), d_out1.data(), out1.size() * sizeof(uint16_t), stream), "d2h out1");
  plane.Check(plane.memcpy_d2h(out2.data(), d_out2.data(), out2.size() * sizeof(uint16_t), stream), "d2h out2");
  plane.Check(plane.stream_sync(stream), "sync");

  for (uint16_t v : out1) EXPECT_EQ(v, 0x3F80u);  // bf16(1.0f)
  for (uint16_t v : out2) EXPECT_EQ(v, 0x4080u);  // bf16(4.0f)

  // Updating p1 in-place and re-registering must invalidate p1's N-gram row cache.
  std::fill(shard1.begin(), shard1.end(), 0x40u);  // 2.0f * scale(1.0f) -> bf16 2.0f (0x4000)
  plane.set_ple_host_params(&p1);
  EXPECT_EQ(plane.ple_rows(stream, d_p1.data(), d_tokens.data(), 2, d_ctx_in.data(), d_ctx_out.data(), d_out1.data()), 0);
  plane.Check(plane.memcpy_d2h(out1.data(), d_out1.data(), out1.size() * sizeof(uint16_t), stream), "d2h out1 updated");
  plane.Check(plane.stream_sync(stream), "sync updated");
  for (uint16_t v : out1) EXPECT_EQ(v, 0x4000u);  // bf16(2.0f)

  // Unregistering p2 leaves p1 operational and tears down p2's staged state.
  if (plane.unset_ple_host_params) {
    plane.unset_ple_host_params(&p2);
  }
  std::fill(out1.begin(), out1.end(), 0);
  EXPECT_EQ(plane.ple_rows(stream, d_p1.data(), d_tokens.data(), 2, d_ctx_in.data(), d_ctx_out.data(), d_out1.data()), 0);
  plane.Check(plane.memcpy_d2h(out1.data(), d_out1.data(), out1.size() * sizeof(uint16_t), stream), "d2h out1 after unset");
  plane.Check(plane.stream_sync(stream), "sync after unset");
  for (uint16_t v : out1) EXPECT_EQ(v, 0x4000u);

  if (plane.release_ple_workspace) {
    plane.release_ple_workspace();
  }
  plane.stream_destroy(stream);
}

namespace {

constexpr int kCudaErrorInvalidValue = 1;
constexpr int kCudaErrorStreamCaptureUnsupported = 900;

// Keeps the plane library loaded independently of the Qwen38Plane objects that load it, and gives access to the test hooks the
// library exports when it is built with QWEN38_PLANE_TEST_HOOKS (see qwen38_plane_abi.h). The library holds process-wide state,
// the staged PLE tables; holding it open is what lets a test look at that state after the loaders that used it are gone.
class PlaneModuleReference {
 public:
  explicit PlaneModuleReference(const std::filesystem::path& path) {
#if defined(_WIN32)
    handle_ = LoadLibraryW(ResolveWindowsSharedLibraryPath(path).wstring().c_str());
#else
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
    if (handle_ != nullptr) {
      inject_fault_ = Symbol<Qwen38PlaneTestInjectFaultFn>("qwen38_plane_test_inject_staged_fault");
      stats_ = Symbol<Qwen38PlaneTestStatsFn>("qwen38_plane_test_stats");
      peek_last_error_ = Symbol<Qwen38PlaneTestPeekLastErrorFn>("qwen38_plane_test_peek_last_error");
      stream_capture_ = Symbol<Qwen38PlaneTestStreamCaptureFn>("qwen38_plane_test_stream_capture");
    }
  }

  ~PlaneModuleReference() {
    if (handle_ == nullptr) return;
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
  }

  PlaneModuleReference(const PlaneModuleReference&) = delete;
  PlaneModuleReference& operator=(const PlaneModuleReference&) = delete;

  bool HasHooks() const {
    return inject_fault_ != nullptr && stats_ != nullptr && peek_last_error_ != nullptr && stream_capture_ != nullptr;
  }

  // The (skip + 1)-th execution of `site` from now on fails once with `error`.
  void Arm(Qwen38PlaneFaultSite site, int skip, int error) const { EXPECT_EQ(inject_fault_(site, skip, error), 0); }

  void Disarm() const { inject_fault_(QWEN38_PLANE_FAULT_NONE, 0, 0); }

  Qwen38PlaneTestStats Stats() const {
    Qwen38PlaneTestStats stats{};
    EXPECT_EQ(stats_(&stats), 0);
    return stats;
  }

  // The last CUDA error of the calling thread, as the module's CUDA runtime sees it.
  int PeekLastError() const { return peek_last_error_(); }

  // Starts and ends the capture of `stream` into a CUDA graph that is thrown away; both return the CUDA error of the call.
  int BeginCapture(void* stream) const { return stream_capture_(stream, 1); }
  int EndCapture(void* stream) const { return stream_capture_(stream, 0); }

 private:
  template <typename Function>
  Function Symbol(const char* name) const {
#if defined(_WIN32)
    return reinterpret_cast<Function>(GetProcAddress(static_cast<HMODULE>(handle_), name));
#else
    return reinterpret_cast<Function>(dlsym(handle_, name));
#endif
  }

  void* handle_{};
  Qwen38PlaneTestInjectFaultFn inject_fault_{};
  Qwen38PlaneTestStatsFn stats_{};
  Qwen38PlaneTestPeekLastErrorFn peek_last_error_{};
  Qwen38PlaneTestStreamCaptureFn stream_capture_{};
};

// Captures a stream into a CUDA graph for as long as it lives, so that a failing assertion cannot leave the stream capturing.
class StreamCapture {
 public:
  StreamCapture(const PlaneModuleReference& module, void* stream) : module_{module}, stream_{stream} {
    active_ = module_.BeginCapture(stream_) == 0;
  }
  ~StreamCapture() { End(); }
  StreamCapture(const StreamCapture&) = delete;
  StreamCapture& operator=(const StreamCapture&) = delete;

  bool active() const { return active_; }

  // Ends the capture. Returns the CUDA error of ending it, which is not 0 when something invalidated the capture.
  int End() {
    if (!active_) return 0;
    active_ = false;
    return module_.EndCapture(stream_);
  }

 private:
  const PlaneModuleReference& module_;
  void* stream_;
  bool active_{};
};

class PlaneStream {
 public:
  explicit PlaneStream(Qwen38Plane& plane) : plane_{plane} {
    plane_.Check(plane_.stream_create(&stream_), "stream_create");
  }
  ~PlaneStream() { plane_.stream_destroy(stream_); }
  PlaneStream(const PlaneStream&) = delete;
  PlaneStream& operator=(const PlaneStream&) = delete;

  void* get() const { return stream_; }

 private:
  Qwen38Plane& plane_;
  void* stream_{};
};

// FP8 E4M3 codes of 1, 2, 3 and 4 and the bf16 bit patterns they dequantize to at a scale of 1.
constexpr std::array<unsigned char, 4> kAscendingRows = {0x38, 0x40, 0x44, 0x48};
constexpr std::array<unsigned char, 4> kDescendingRows = {0x48, 0x44, 0x40, 0x38};

uint16_t Bf16OfFp8(unsigned char code) {
  switch (code) {
    case 0x38:
      return 0x3F80;
    case 0x40:
      return 0x4000;
    case 0x44:
      return 0x4040;
    default:
      return 0x4080;
  }
}

// A staged PLE table (Qwen38PleParams::pad_ == 1: its FP8 rows stay in host memory and the module gathers them there)
// registered with a plane, and the device buffers to call ple_rows and prep on it. The table has one head and four rows of four
// values, and only the token takes part in the n-gram hash, so token t gathers row t % 4. Row r holds the FP8 code row_codes[r].
class StagedPleRig {
 public:
  static constexpr size_t kRows = 4;
  static constexpr size_t kRowWidth = 4;
  static constexpr size_t kMaxTokens = 4200;
  static constexpr size_t kPositions = 16;  // the table's capacity: how many positions past_pos has room for
  static constexpr size_t kStateWords = 10;

  StagedPleRig(Qwen38Plane& plane, const std::array<unsigned char, kRows>& row_codes)
      : plane_{plane},
        codes_{row_codes},
        stream_{plane},
        d_params_{plane, sizeof(Qwen38PleParams), stream_.get()},
        d_tokens_{plane, kMaxTokens * sizeof(int64_t), stream_.get()},
        d_ctx_in_{plane, 2 * sizeof(int64_t), stream_.get()},
        d_ctx_out_{plane, 2 * sizeof(int64_t), stream_.get()},
        d_out_{plane, kMaxTokens * kRowWidth * sizeof(uint16_t), stream_.get()},
        d_tok_{plane, sizeof(int64_t), stream_.get()},
        d_pos_{plane, sizeof(int64_t), stream_.get()},
        d_state_{plane, kStateWords * sizeof(int64_t), stream_.get()},
        d_input_ids_{plane, sizeof(int64_t), stream_.get()},
        d_position_ids_{plane, sizeof(int64_t), stream_.get()},
        d_committed_{plane, sizeof(int), stream_.get()},
        d_past_pos_{plane, kPositions * sizeof(int64_t), stream_.get()} {
    for (size_t row = 0; row < kRows; ++row) {
      std::fill_n(shard_.begin() + row * kRowWidth, kRowWidth, codes_[row]);
    }
    params_.n_heads = 1;
    params_.head_dim = static_cast<int>(kRowWidth);
    params_.heads_per_ngram = 1;
    params_.n_shards = 1;
    params_.mult[0] = 1;
    params_.vocab[0] = static_cast<long long>(kRows);
    params_.shard_end[0] = static_cast<long long>(kRows);
    params_.shard_ptr[0] = shard_.data();
    params_.scale = 1.0f;
    params_.pad_ = 1;
    params_.eos = 99;
    params_.capacity = static_cast<long long>(kPositions);
    Upload(d_params_, &params_, sizeof(params_));
    plane_.set_ple_host_params(&params_);
  }

  ~StagedPleRig() {
    if (unregister_on_destruction_) plane_.unset_ple_host_params(&params_);
  }

  StagedPleRig(const StagedPleRig&) = delete;
  StagedPleRig& operator=(const StagedPleRig&) = delete;

  const Qwen38PleParams& params() const { return params_; }
  const void* device_params() const { return d_params_.data(); }
  void* stream() const { return stream_.get(); }

  // Leaves the table registered with the module when the rig goes away, as a loader that never unregistered it would.
  void KeepTableRegisteredOnDestruction() { unregister_on_destruction_ = false; }

  // Registers the table afresh, which gives it an empty staged workspace.
  void ResetTable() {
    plane_.unset_ple_host_params(&params_);
    plane_.set_ple_host_params(&params_);
  }

  PlaneBuffer DeviceCopy(const Qwen38PleParams& params) {
    PlaneBuffer copy{plane_, sizeof(params), stream_.get()};
    Upload(copy, &params, sizeof(params));
    return copy;
  }

  // A resident table (pad_ == 0) with the rows of this one: its shard is in device memory, so ple_rows and prep launch the
  // device kernels on it and no staging takes part.
  struct ResidentTable {
    PlaneBuffer shard;
    PlaneBuffer descriptor;
  };

  ResidentTable MakeResidentTable() {
    ResidentTable table;
    table.shard = PlaneBuffer{plane_, shard_.size(), stream_.get()};
    Upload(table.shard, shard_.data(), shard_.size());
    Qwen38PleParams resident = params_;
    resident.pad_ = 0;
    resident.shard_ptr[0] = table.shard.data();
    table.descriptor = DeviceCopy(resident);
    return table;
  }

  int PleRows(const std::vector<int64_t>& tokens) {
    UploadTokens(tokens);
    return PleRowsOn(device_params(), device_tokens(), tokens.size());
  }

  void UploadTokens(const std::vector<int64_t>& tokens) { Upload(d_tokens_, tokens.data(), tokens.size() * sizeof(int64_t)); }
  const void* device_tokens() const { return d_tokens_.data(); }

  int PleRowsOn(const void* params, const void* tokens, size_t count) {
    return plane_.ple_rows(stream_.get(), params, tokens, static_cast<int>(count), d_ctx_in_.data(), d_ctx_out_.data(),
                           d_out_.data());
  }

  // prep with the token and the position in words of their own.
  int PrepGeneric(int64_t token, int64_t position, int pos_add, const void* params = nullptr) {
    UploadPrepWords(token, position);
    return PrepUploaded(pos_add, params);
  }

  // The two halves of PrepGeneric, for a caller that cannot copy while it calls (a stream being captured).
  void UploadPrepWords(int64_t token, int64_t position) {
    Upload(d_tok_, &token, sizeof(token));
    Upload(d_pos_, &position, sizeof(position));
  }

  int PrepUploaded(int pos_add, const void* params = nullptr) {
    return Prep(params, d_tok_.data(), d_pos_.data(), pos_add, d_ctx_in_.data());
  }

  // prep for row 0 of a fold: the token and the position are words 0 and 1 of the ten-word state, which also holds the tokens
  // of the other rows.
  int PrepRowZero(const std::array<int64_t, kStateWords>& state) {
    Upload(d_state_, state.data(), sizeof(state));
    return Prep(nullptr, d_state_.data(), d_state_.data() + sizeof(int64_t), 0, d_ctx_in_.data());
  }

  // prep for row 1..3 of the fold whose row 0 ran last: its context in is the previous row's context out.
  int PrepFoldRow(int pos_add) {
    const size_t token_word = pos_add == 1 ? 2 : (pos_add == 2 ? 8 : 9);
    return Prep(nullptr, d_state_.data() + token_word * sizeof(int64_t), d_state_.data() + sizeof(int64_t), pos_add,
                d_ctx_out_.data());
  }

  void FillOutputs(int byte) {
    for (const PlaneBuffer* buffer : Outputs()) {
      plane_.Check(plane_.device_memset(buffer->data(), byte, buffer->size(), stream_.get()), "device_memset");
    }
    Sync();
  }

  bool OutputsAre(uint8_t byte) {
    for (const PlaneBuffer* buffer : Outputs()) {
      for (uint8_t value : Read<uint8_t>(*buffer, buffer->size())) {
        if (value != byte) return false;
      }
    }
    return true;
  }

  std::vector<uint16_t> ReadPle(size_t tokens) { return Read<uint16_t>(d_out_, tokens * kRowWidth); }

  std::vector<uint16_t> ExpectedPle(const std::vector<int64_t>& tokens) const {
    std::vector<uint16_t> expected;
    for (int64_t token : tokens) {
      expected.insert(expected.end(), kRowWidth, Bf16OfFp8(codes_[static_cast<size_t>(token) % kRows]));
    }
    return expected;
  }

  // What prep has to produce for `token` at `position`: the graph row inputs, its PLE row and the next context, whose first
  // word is `c1`, the token before it.
  void ExpectPrepResult(int64_t token, int64_t position, int64_t c1) {
    EXPECT_EQ(Read<int64_t>(d_input_ids_, 1)[0], token);
    EXPECT_EQ(Read<int64_t>(d_position_ids_, 1)[0], position);
    EXPECT_EQ(Read<int>(d_committed_, 1)[0], position);
    EXPECT_EQ(Read<int64_t>(d_past_pos_, kPositions)[static_cast<size_t>(position)], position);
    EXPECT_EQ(Read<int64_t>(d_ctx_out_, 2), (std::vector<int64_t>{c1, token}));
    EXPECT_EQ(ReadPle(1), ExpectedPle({token}));
  }

 private:
  int Prep(const void* params, const void* tok_ptr, const void* pos_ptr, int pos_add, const void* ctx_in) {
    return plane_.prep(stream_.get(), params != nullptr ? params : device_params(), tok_ptr, pos_ptr, pos_add, ctx_in,
                       d_ctx_out_.data(), d_input_ids_.data(), d_position_ids_.data(), d_committed_.data(),
                       d_past_pos_.data(), d_out_.data(), d_state_.data());
  }

  std::array<const PlaneBuffer*, 6> Outputs() const {
    return {&d_out_, &d_ctx_out_, &d_input_ids_, &d_position_ids_, &d_committed_, &d_past_pos_};
  }

  void Upload(const PlaneBuffer& buffer, const void* host, size_t bytes) {
    plane_.Check(plane_.memcpy_h2d(buffer.data(), host, bytes, stream_.get()), "memcpy_h2d");
  }

  void Sync() { plane_.Check(plane_.stream_sync(stream_.get()), "stream_sync"); }

  template <typename T>
  std::vector<T> Read(const PlaneBuffer& buffer, size_t count) {
    std::vector<T> values(count);
    plane_.Check(plane_.memcpy_d2h(values.data(), buffer.data(), count * sizeof(T), stream_.get()), "memcpy_d2h");
    Sync();
    return values;
  }

  Qwen38Plane& plane_;
  std::array<unsigned char, kRows> codes_;
  PlaneStream stream_;
  std::array<unsigned char, kRows * kRowWidth> shard_{};
  Qwen38PleParams params_{};
  bool unregister_on_destruction_{true};
  PlaneBuffer d_params_, d_tokens_, d_ctx_in_, d_ctx_out_, d_out_;
  PlaneBuffer d_tok_, d_pos_, d_state_, d_input_ids_, d_position_ids_, d_committed_, d_past_pos_;
};

// One CUDA step of the staged path that a test makes fail: the (skip + 1)-th execution of `site` in the call.
struct StagingStep {
  const char* name;
  Qwen38PlaneFaultSite site;
  int skip;
  bool buffers_published;  // the staging buffers already belong to the workspace when this step runs
};

// The CUDA steps of the first call made on a table, in the order the module runs them.
std::vector<StagingStep> PleRowsSteps() {
  return {
      {"read the table back from the device", QWEN38_PLANE_FAULT_COPY, 0, false},
      {"wait for the table read back", QWEN38_PLANE_FAULT_SYNC, 0, false},
      {"allocate the scalar words", QWEN38_PLANE_FAULT_HOST_ALLOC, 0, false},
      {"allocate the prep ring", QWEN38_PLANE_FAULT_HOST_ALLOC, 1, false},
      {"map the prep ring", QWEN38_PLANE_FAULT_DEVICE_ALIAS, 0, false},
      {"allocate the n-gram row cache", QWEN38_PLANE_FAULT_HOST_MALLOC, 0, false},
      {"allocate the token buffer", QWEN38_PLANE_FAULT_HOST_ALLOC, 2, false},
      {"allocate the FP8 row buffer", QWEN38_PLANE_FAULT_HOST_ALLOC, 3, false},
      {"map the FP8 row buffer", QWEN38_PLANE_FAULT_DEVICE_ALIAS, 1, false},
      {"create the completion event", QWEN38_PLANE_FAULT_EVENT_CREATE, 0, true},
      {"copy the tokens", QWEN38_PLANE_FAULT_COPY, 1, true},
      {"copy the context", QWEN38_PLANE_FAULT_COPY, 2, true},
      {"wait for the copies", QWEN38_PLANE_FAULT_SYNC, 1, true},
  };
}

std::vector<StagingStep> PrepGenericSteps() {
  return {
      {"read the table back from the device", QWEN38_PLANE_FAULT_COPY, 0, false},
      {"wait for the table read back", QWEN38_PLANE_FAULT_SYNC, 0, false},
      {"allocate the scalar words", QWEN38_PLANE_FAULT_HOST_ALLOC, 0, false},
      {"allocate the prep ring", QWEN38_PLANE_FAULT_HOST_ALLOC, 1, false},
      {"map the prep ring", QWEN38_PLANE_FAULT_DEVICE_ALIAS, 0, false},
      {"allocate the n-gram row cache", QWEN38_PLANE_FAULT_HOST_MALLOC, 0, false},
      {"create the completion event", QWEN38_PLANE_FAULT_EVENT_CREATE, 0, true},
      {"copy the token", QWEN38_PLANE_FAULT_COPY, 1, true},
      {"copy the position", QWEN38_PLANE_FAULT_COPY, 2, true},
      {"copy the context", QWEN38_PLANE_FAULT_COPY, 3, true},
      {"wait for the copies", QWEN38_PLANE_FAULT_SYNC, 1, true},
  };
}

std::vector<StagingStep> PrepRowZeroSteps() {
  return {
      {"read the table back from the device", QWEN38_PLANE_FAULT_COPY, 0, false},
      {"wait for the table read back", QWEN38_PLANE_FAULT_SYNC, 0, false},
      {"allocate the scalar words", QWEN38_PLANE_FAULT_HOST_ALLOC, 0, false},
      {"allocate the prep ring", QWEN38_PLANE_FAULT_HOST_ALLOC, 1, false},
      {"map the prep ring", QWEN38_PLANE_FAULT_DEVICE_ALIAS, 0, false},
      {"allocate the n-gram row cache", QWEN38_PLANE_FAULT_HOST_MALLOC, 0, false},
      {"create the completion event", QWEN38_PLANE_FAULT_EVENT_CREATE, 0, true},
      {"copy the ctl words", QWEN38_PLANE_FAULT_COPY, 1, true},
      {"copy the context", QWEN38_PLANE_FAULT_COPY, 2, true},
      {"wait for the copies", QWEN38_PLANE_FAULT_SYNC, 1, true},
  };
}

// A call that is made on a table with an empty staged workspace, which has to stage everything from scratch.
struct ColdCall {
  std::function<int()> run;
  std::function<void()> expect_result;  // checks the outputs of a successful run
  long long pinned_buffers;             // pinned staging buffers a successful run leaves in the workspace
  long long capacity_rows;              // prefill capacity it leaves
  long long gathered_rows;
};

class Qwen38PlaneStagedPleTest : public ::testing::Test {
 protected:
  void SetUp() override {
    library_path_ = GetBuiltPlaneLibraryPath();
    if (!std::filesystem::exists(library_path_)) {
      GTEST_SKIP() << library_path_.string() << " not present";
    }
    module_ = std::make_unique<PlaneModuleReference>(library_path_);
    if (!module_->HasHooks()) {
      GTEST_SKIP() << "the plane library was built without QWEN38_PLANE_TEST_HOOKS";
    }
    plane_ = std::make_unique<Qwen38Plane>(library_path_);
    if (plane_->set_device(0) != 0 || !plane_->set_ple_host_params || !plane_->unset_ple_host_params) {
      GTEST_SKIP() << "No CUDA device or staged PLE support available";
    }
  }

  void TearDown() override {
    if (module_ && module_->HasHooks()) module_->Disarm();
    plane_.reset();
  }

  Qwen38PlaneTestStats Stats() const { return module_->Stats(); }
  void Arm(Qwen38PlaneFaultSite site, int skip, int error) const { module_->Arm(site, skip, error); }

  // Makes each of `steps` fail in turn on a first call and checks that the call returns exactly the injected error, runs no
  // gather or kernel and writes no output, that the staged workspace keeps nothing of the steps that had succeeded (the
  // buffers are all allocated or none is, and the capacity is only recorded with them), and that the same call then works.
  void ExpectEveryStagingStepFailsCleanly(StagedPleRig& rig, const std::vector<StagingStep>& steps, const ColdCall& call) {
    int index = 0;
    for (const StagingStep& step : steps) {
      SCOPED_TRACE(step.name);
      // Not a CUDA error code: only the failed step can have produced it.
      const int error = 1000 + index++;
      rig.ResetTable();
      rig.FillOutputs(0xAB);
      const Qwen38PlaneTestStats before = Stats();

      Arm(step.site, step.skip, error);
      EXPECT_EQ(call.run(), error);
      const Qwen38PlaneTestStats failed = Stats();
      EXPECT_EQ(failed.fault_armed, 0) << "the injected fault never fired";
      EXPECT_EQ(failed.gather_rows, before.gather_rows);
      EXPECT_EQ(failed.staged_launches, before.staged_launches);
      EXPECT_TRUE(rig.OutputsAre(0xAB));
      EXPECT_EQ(failed.live_pinned_allocations,
                before.live_pinned_allocations + (step.buffers_published ? call.pinned_buffers : 0));
      EXPECT_EQ(failed.prefill_capacity_rows, step.buffers_published ? call.capacity_rows : 0);

      EXPECT_EQ(call.run(), 0);
      const Qwen38PlaneTestStats retried = Stats();
      EXPECT_EQ(retried.gather_rows, before.gather_rows + call.gathered_rows);
      EXPECT_EQ(retried.staged_launches, before.staged_launches + 1);
      EXPECT_EQ(retried.live_pinned_allocations, before.live_pinned_allocations + call.pinned_buffers);
      EXPECT_EQ(retried.prefill_capacity_rows, call.capacity_rows);
      call.expect_result();
    }
  }

  // Both entry points fail on a capturing stream with cudaErrorStreamCaptureUnsupported, gather and launch nothing, write no
  // output and leave the capture whole, which ending it shows: a call that failed halfway through its own staging steps would
  // have invalidated it.
  void ExpectRefusedUnderCapture(StagedPleRig& rig, const void* descriptor, size_t token_count) {
    rig.FillOutputs(0xAB);
    const Qwen38PlaneTestStats before = Stats();
    {
      StreamCapture capture{*module_, rig.stream()};
      ASSERT_TRUE(capture.active());
      EXPECT_EQ(rig.PleRowsOn(descriptor, rig.device_tokens(), token_count), kCudaErrorStreamCaptureUnsupported);
      EXPECT_EQ(rig.PrepUploaded(0, descriptor), kCudaErrorStreamCaptureUnsupported);
      EXPECT_EQ(capture.End(), 0) << "the refused calls invalidated the capture";
    }
    const Qwen38PlaneTestStats after = Stats();
    EXPECT_EQ(after.gather_rows, before.gather_rows);
    EXPECT_EQ(after.staged_launches, before.staged_launches);
    EXPECT_TRUE(rig.OutputsAre(0xAB));
  }

  // A call whose completion event cannot be recorded cannot tell a later call on another stream when its kernel is done, so it
  // drains its stream instead (the outputs are complete) and still reports the failure. The call that follows has an event to
  // record again, succeeds and drains nothing.
  void ExpectAFailedEventRecordIsDrainedAndReported(StagedPleRig& rig, const std::function<int()>& run,
                                                    const std::function<void()>& expect_result) {
    constexpr int kInjectedError = 5000;
    rig.ResetTable();
    ASSERT_EQ(run(), 0);
    expect_result();
    rig.FillOutputs(0xAB);
    const Qwen38PlaneTestStats before = Stats();

    Arm(QWEN38_PLANE_FAULT_EVENT_RECORD, 0, kInjectedError);
    EXPECT_EQ(run(), kInjectedError);
    const Qwen38PlaneTestStats failed = Stats();
    EXPECT_EQ(failed.fault_armed, 0) << "the injected fault never fired";
    EXPECT_EQ(failed.staged_launches, before.staged_launches + 1);
    EXPECT_EQ(failed.unfenced_drains, before.unfenced_drains + 1) << "the stream was not drained";
    expect_result();

    rig.FillOutputs(0xAB);
    EXPECT_EQ(run(), 0);
    EXPECT_EQ(Stats().unfenced_drains, failed.unfenced_drains);
    expect_result();
  }

  std::filesystem::path library_path_;
  std::unique_ptr<PlaneModuleReference> module_;
  std::unique_ptr<Qwen38Plane> plane_;
};

TEST_F(Qwen38PlaneStagedPleTest, PleRowsReportsEveryFailedStagingStepAndPublishesNothingPartial) {
  StagedPleRig rig{*plane_, kAscendingRows};
  const std::vector<int64_t> tokens = {1, 2};
  ColdCall call;
  call.run = [&] { return rig.PleRows(tokens); };
  call.expect_result = [&] { EXPECT_EQ(rig.ReadPle(tokens.size()), rig.ExpectedPle(tokens)); };
  call.pinned_buffers = 4;
  call.capacity_rows = 4096;
  call.gathered_rows = 2;
  ExpectEveryStagingStepFailsCleanly(rig, PleRowsSteps(), call);
}

TEST_F(Qwen38PlaneStagedPleTest, PrepReportsEveryFailedStagingStepOnTheGenericPath) {
  StagedPleRig rig{*plane_, kAscendingRows};
  ColdCall call;
  call.run = [&] { return rig.PrepGeneric(6, 5, 0); };
  call.expect_result = [&] { rig.ExpectPrepResult(6, 5, 0); };
  call.pinned_buffers = 2;
  call.capacity_rows = 0;
  call.gathered_rows = 1;
  ExpectEveryStagingStepFailsCleanly(rig, PrepGenericSteps(), call);
}

TEST_F(Qwen38PlaneStagedPleTest, PrepReportsEveryFailedStagingStepOnRowZeroOfAFold) {
  StagedPleRig rig{*plane_, kAscendingRows};
  const std::array<int64_t, StagedPleRig::kStateWords> state = {7, 9, 2, 0, 0, 0, 0, 0, 1, 0};
  ColdCall call;
  call.run = [&] { return rig.PrepRowZero(state); };
  call.expect_result = [&] { rig.ExpectPrepResult(7, 9, 0); };
  call.pinned_buffers = 2;
  call.capacity_rows = 0;
  call.gathered_rows = 1;
  ExpectEveryStagingStepFailsCleanly(rig, PrepRowZeroSteps(), call);
}

TEST_F(Qwen38PlaneStagedPleTest, PleRowsDrainsItsStreamWhenTheCompletionEventCannotBeRecorded) {
  StagedPleRig rig{*plane_, kAscendingRows};
  const std::vector<int64_t> tokens = {1, 2};
  ExpectAFailedEventRecordIsDrainedAndReported(
      rig, [&] { return rig.PleRows(tokens); },
      [&] { EXPECT_EQ(rig.ReadPle(tokens.size()), rig.ExpectedPle(tokens)); });
}

TEST_F(Qwen38PlaneStagedPleTest, PrepDrainsItsStreamWhenTheCompletionEventCannotBeRecorded) {
  StagedPleRig rig{*plane_, kAscendingRows};
  ExpectAFailedEventRecordIsDrainedAndReported(
      rig, [&] { return rig.PrepGeneric(6, 5, 0); }, [&] { rig.ExpectPrepResult(6, 5, 0); });
}

// A failed refresh of the token and context words must not leave the previous call's words to be gathered from.
TEST_F(Qwen38PlaneStagedPleTest, PleRowsNeverGathersStaleTokensWhenARefreshFails) {
  StagedPleRig rig{*plane_, kAscendingRows};
  rig.ResetTable();
  ASSERT_EQ(rig.PleRows({1, 2}), 0);

  const std::vector<int64_t> tokens = {0, 3};
  const struct {
    const char* name;
    Qwen38PlaneFaultSite site;
    int skip;
  } failures[] = {
      {"copy the tokens", QWEN38_PLANE_FAULT_COPY, 0},
      {"copy the context", QWEN38_PLANE_FAULT_COPY, 1},
      {"wait for the copies", QWEN38_PLANE_FAULT_SYNC, 0},
  };
  int index = 0;
  for (const auto& failure : failures) {
    SCOPED_TRACE(failure.name);
    const int error = 2000 + index++;
    rig.FillOutputs(0xAB);
    const Qwen38PlaneTestStats before = Stats();
    Arm(failure.site, failure.skip, error);
    EXPECT_EQ(rig.PleRows(tokens), error);
    const Qwen38PlaneTestStats after = Stats();
    EXPECT_EQ(after.fault_armed, 0);
    EXPECT_EQ(after.gather_rows, before.gather_rows);
    EXPECT_EQ(after.staged_launches, before.staged_launches);
    EXPECT_TRUE(rig.OutputsAre(0xAB));

    EXPECT_EQ(rig.PleRows(tokens), 0);
    EXPECT_EQ(rig.ReadPle(tokens.size()), rig.ExpectedPle(tokens));
  }
}

// Growing the prefill buffers replaces working buffers: when it fails the old ones, and their capacity, must stay.
TEST_F(Qwen38PlaneStagedPleTest, PleRowsGrowthIsAllOrNothing) {
  StagedPleRig rig{*plane_, kAscendingRows};
  rig.ResetTable();
  const std::vector<int64_t> short_tokens = {1, 2};
  ASSERT_EQ(rig.PleRows(short_tokens), 0);
  const Qwen38PlaneTestStats after_short = Stats();
  EXPECT_EQ(after_short.prefill_capacity_rows, 4096);

  std::vector<int64_t> long_tokens(4097);
  for (size_t i = 0; i < long_tokens.size(); ++i) long_tokens[i] = static_cast<int64_t>(i);

  const struct {
    const char* name;
    Qwen38PlaneFaultSite site;
  } failures[] = {
      {"allocate the larger token buffer", QWEN38_PLANE_FAULT_HOST_ALLOC},
      {"wait for earlier work before the old buffer is freed", QWEN38_PLANE_FAULT_SYNC},
  };
  int index = 0;
  for (const auto& failure : failures) {
    SCOPED_TRACE(failure.name);
    const int error = 3000 + index++;
    rig.FillOutputs(0xAB);
    const Qwen38PlaneTestStats before = Stats();
    Arm(failure.site, 0, error);
    EXPECT_EQ(rig.PleRows(long_tokens), error);
    const Qwen38PlaneTestStats failed = Stats();
    EXPECT_EQ(failed.fault_armed, 0);
    EXPECT_EQ(failed.prefill_capacity_rows, after_short.prefill_capacity_rows);
    EXPECT_EQ(failed.live_pinned_allocations, after_short.live_pinned_allocations);
    EXPECT_EQ(failed.gather_rows, before.gather_rows);
    EXPECT_EQ(failed.staged_launches, before.staged_launches);
    EXPECT_TRUE(rig.OutputsAre(0xAB));

    EXPECT_EQ(rig.PleRows(short_tokens), 0);
    EXPECT_EQ(rig.ReadPle(short_tokens.size()), rig.ExpectedPle(short_tokens));
  }

  EXPECT_EQ(rig.PleRows(long_tokens), 0);
  const Qwen38PlaneTestStats grown = Stats();
  EXPECT_EQ(grown.prefill_capacity_rows, 4097);
  EXPECT_EQ(grown.live_pinned_allocations, after_short.live_pinned_allocations);
  EXPECT_EQ(rig.ReadPle(long_tokens.size()), rig.ExpectedPle(long_tokens));
}

// Rows 1..3 of a fold reuse what row 0 read back and so run no staging step at all (a CUDA graph capture could not take one);
// a row 0 that failed leaves nothing for them to reuse.
TEST_F(Qwen38PlaneStagedPleTest, FoldRowsReuseTheRowZeroSnapshotOnlyWhenRowZeroSucceeded) {
  StagedPleRig rig{*plane_, kAscendingRows};
  rig.ResetTable();
  const std::array<int64_t, StagedPleRig::kStateWords> state = {7, 9, 2, 0, 0, 0, 0, 0, 1, 0};

  const Qwen38PlaneFaultSite staging_sites[] = {QWEN38_PLANE_FAULT_HOST_ALLOC, QWEN38_PLANE_FAULT_HOST_MALLOC,
                                                QWEN38_PLANE_FAULT_DEVICE_ALIAS, QWEN38_PLANE_FAULT_COPY,
                                                QWEN38_PLANE_FAULT_SYNC};
  for (Qwen38PlaneFaultSite site : staging_sites) {
    SCOPED_TRACE(site);
    module_->Disarm();
    ASSERT_EQ(rig.PrepRowZero(state), 0);
    Arm(site, 0, 4000);
    EXPECT_EQ(rig.PrepFoldRow(1), 0);
    EXPECT_EQ(Stats().fault_armed, 1) << "a fold row ran a staging step";
    rig.ExpectPrepResult(2, 10, 7);
  }

  const struct {
    const char* name;
    Qwen38PlaneFaultSite site;
    int skip;
  } failures[] = {
      {"copy the ctl words", QWEN38_PLANE_FAULT_COPY, 0},
      {"copy the context", QWEN38_PLANE_FAULT_COPY, 1},
      {"wait for the copies", QWEN38_PLANE_FAULT_SYNC, 0},
  };
  int index = 0;
  for (const auto& failure : failures) {
    SCOPED_TRACE(failure.name);
    const int row_zero_error = 4100 + index;
    const int fold_row_error = 4200 + index++;
    module_->Disarm();
    ASSERT_EQ(rig.PrepRowZero(state), 0);

    Arm(failure.site, failure.skip, row_zero_error);
    EXPECT_EQ(rig.PrepRowZero(state), row_zero_error);
    // Row 1 must read its own words (the first copy it makes fails) instead of trusting the snapshot of the earlier row 0.
    Arm(QWEN38_PLANE_FAULT_COPY, 0, fold_row_error);
    EXPECT_EQ(rig.PrepFoldRow(1), fold_row_error);
  }
}

// ple_rows lands its context words in the same pinned words the rows of a fold keep their draft tokens in, so it has to drop the
// snapshot: the row that follows reads its own token and context from the device, not what ple_rows left in the words.
TEST_F(Qwen38PlaneStagedPleTest, PleRowsBetweenTheRowsOfAFoldLeavesNothingForARowToReuse) {
  StagedPleRig rig{*plane_, kAscendingRows};
  rig.ResetTable();
  const std::array<int64_t, StagedPleRig::kStateWords> state = {7, 9, 2, 0, 0, 0, 0, 0, 1, 0};
  ASSERT_EQ(rig.PrepRowZero(state), 0);
  ASSERT_EQ(rig.PleRows({3}), 0);

  // Row 1 is the draft token 2 one position after row 0, behind the context ple_rows left: (0, 3).
  ASSERT_EQ(rig.PrepFoldRow(1), 0);
  rig.ExpectPrepResult(2, 10, 3);
}

// A table that is staged but was never registered has no host rows to gather from: it has to be an error, not a silent no-op.
TEST_F(Qwen38PlaneStagedPleTest, AStagedTableThatWasNeverRegisteredIsAnErrorNotASilentSkip) {
  StagedPleRig rig{*plane_, kAscendingRows};
  std::array<unsigned char, StagedPleRig::kRows * StagedPleRig::kRowWidth> other_shard{};
  Qwen38PleParams unregistered = rig.params();
  unregistered.shard_ptr[0] = other_shard.data();
  const PlaneBuffer d_unregistered = rig.DeviceCopy(unregistered);

  rig.FillOutputs(0xAB);
  const Qwen38PlaneTestStats before = Stats();
  EXPECT_EQ(rig.PleRows({1, 2}), 0);
  const Qwen38PlaneTestStats registered = Stats();
  EXPECT_EQ(registered.gather_rows, before.gather_rows + 2);

  rig.FillOutputs(0xAB);
  EXPECT_EQ(rig.PleRowsOn(d_unregistered.data(), rig.device_params(), 2), kCudaErrorInvalidValue);
  EXPECT_EQ(rig.PrepGeneric(1, 1, 0, d_unregistered.data()), kCudaErrorInvalidValue);
  // Fold rows (pos_add 1..3) with one staged table registered must also validate unknown descriptors instead of
  // assigning them to the single registered staged table.
  EXPECT_EQ(rig.PrepGeneric(1, 1, 1, d_unregistered.data()), kCudaErrorInvalidValue);
  const Qwen38PlaneTestStats after = Stats();
  EXPECT_EQ(after.gather_rows, registered.gather_rows);
  EXPECT_EQ(after.staged_launches, registered.staged_launches);
  EXPECT_TRUE(rig.OutputsAre(0xAB));

  // Even when the staged registry is completely empty, an unregistered staged descriptor (pad_ == 1) must be rejected
  // with cudaErrorInvalidValue rather than launching the resident kernel on CPU-only shard pointers.
  plane_->release_ple_workspace();
  ASSERT_EQ(Stats().staged_tables, 0);
  rig.FillOutputs(0xAB);
  EXPECT_EQ(rig.PleRowsOn(d_unregistered.data(), rig.device_params(), 2), kCudaErrorInvalidValue);
  EXPECT_EQ(rig.PrepGeneric(1, 1, 0, d_unregistered.data()), kCudaErrorInvalidValue);
  EXPECT_TRUE(rig.OutputsAre(0xAB));
}

// Telling a staged table from a resident one takes a read back of its descriptor, which a capturing stream cannot make, so a
// descriptor that was not told apart by a call before the capture is an error there, with staged tables registered or not:
// taken for a resident table, a staged one would reach kernels that dereference host-only shard pointers on every replay.
TEST_F(Qwen38PlaneStagedPleTest, ADescriptorNotToldApartBeforeTheCaptureFailsTheCallAndNotTheCapture) {
  StagedPleRig rig{*plane_, kAscendingRows};
  const std::vector<int64_t> tokens = {1, 2};
  rig.UploadTokens(tokens);
  rig.UploadPrepWords(6, 5);
  const PlaneBuffer descriptor = rig.DeviceCopy(rig.params());

  {
    SCOPED_TRACE("a staged table is registered");
    rig.ResetTable();
    ExpectRefusedUnderCapture(rig, descriptor.data(), tokens.size());
  }
  {
    SCOPED_TRACE("no staged table is registered");
    plane_->release_ple_workspace();
    ASSERT_EQ(Stats().staged_tables, 0);
    ExpectRefusedUnderCapture(rig, descriptor.data(), tokens.size());
  }
}

// What a staged table's kernels read is a row that the host gathered for one call, which a replay would use again: a staged
// table is refused on a capturing stream even once its descriptor is known, the rows of a fold that run no staging step
// included, and the refusal leaves the fold as it was.
TEST_F(Qwen38PlaneStagedPleTest, AStagedTableIsRefusedOnACapturingStreamEvenOnceItsDescriptorIsKnown) {
  StagedPleRig rig{*plane_, kAscendingRows};
  rig.ResetTable();
  const std::vector<int64_t> tokens = {1, 2};
  const std::array<int64_t, StagedPleRig::kStateWords> state = {7, 9, 2, 0, 0, 0, 0, 0, 1, 0};
  ASSERT_EQ(rig.PleRows(tokens), 0);
  ASSERT_EQ(rig.PrepGeneric(6, 5, 0), 0);
  ASSERT_EQ(rig.PrepRowZero(state), 0);

  ExpectRefusedUnderCapture(rig, rig.device_params(), tokens.size());
  {
    StreamCapture capture{*module_, rig.stream()};
    ASSERT_TRUE(capture.active());
    EXPECT_EQ(rig.PrepFoldRow(1), kCudaErrorStreamCaptureUnsupported);
    EXPECT_EQ(capture.End(), 0) << "the refused call invalidated the capture";
  }

  ASSERT_EQ(rig.PrepFoldRow(1), 0);
  rig.ExpectPrepResult(2, 10, 7);
}

// A resident table needs none of that: its kernels read everything from device memory, so once a call made outside the capture
// has told its descriptor from a staged table's they are captured like any other kernel.
TEST_F(Qwen38PlaneStagedPleTest, AResidentTableStaysCapturableOnceItsDescriptorIsKnown) {
  StagedPleRig rig{*plane_, kAscendingRows};
  rig.ResetTable();
  const std::vector<int64_t> tokens = {1, 2};
  rig.UploadTokens(tokens);
  rig.UploadPrepWords(6, 5);
  const StagedPleRig::ResidentTable resident = rig.MakeResidentTable();

  ASSERT_EQ(rig.PleRowsOn(resident.descriptor.data(), rig.device_tokens(), tokens.size()), 0);
  EXPECT_EQ(rig.ReadPle(tokens.size()), rig.ExpectedPle(tokens));
  ASSERT_EQ(rig.PrepUploaded(0, resident.descriptor.data()), 0);
  rig.ExpectPrepResult(6, 5, 0);

  rig.FillOutputs(0xAB);
  const Qwen38PlaneTestStats before = Stats();
  {
    StreamCapture capture{*module_, rig.stream()};
    ASSERT_TRUE(capture.active());
    EXPECT_EQ(rig.PleRowsOn(resident.descriptor.data(), rig.device_tokens(), tokens.size()), 0);
    EXPECT_EQ(rig.PrepUploaded(0, resident.descriptor.data()), 0);
    EXPECT_EQ(capture.End(), 0);
  }
  // The kernels were captured, not run, and no staged table took part.
  EXPECT_TRUE(rig.OutputsAre(0xAB));
  EXPECT_EQ(Stats().staged_launches, before.staged_launches);
}

// Concurrent or back-to-back staged ple_rows calls for the same table across two streams must fence the previous stream's
// kernel before reusing the shared mapped staging buffer.
TEST_F(Qwen38PlaneStagedPleTest, StagedPleRowsAcrossMultipleStreamsDoesNotCorruptInFlightReads) {
  StagedPleRig rig{*plane_, kAscendingRows};
  PlaneStream stream_a{*plane_};
  PlaneStream stream_b{*plane_};

  constexpr size_t kCount = 512;
  std::vector<int64_t> tokens_a(kCount, 0);  // row 0 -> 0x38 -> bf16 1.0f (0x3F80)
  std::vector<int64_t> tokens_b(kCount, 3);  // row 3 -> 0x48 -> bf16 4.0f (0x4080)
  const int64_t h_ctx[2] = {0, 0};

  PlaneBuffer d_tokens_a{*plane_, kCount * sizeof(int64_t), stream_a.get()};
  PlaneBuffer d_tokens_b{*plane_, kCount * sizeof(int64_t), stream_b.get()};
  PlaneBuffer d_ctx_in_a{*plane_, sizeof(h_ctx), stream_a.get()};
  PlaneBuffer d_ctx_in_b{*plane_, sizeof(h_ctx), stream_b.get()};
  PlaneBuffer d_ctx_out_a{*plane_, sizeof(h_ctx), stream_a.get()};
  PlaneBuffer d_ctx_out_b{*plane_, sizeof(h_ctx), stream_b.get()};
  PlaneBuffer d_out_a{*plane_, kCount * StagedPleRig::kRowWidth * sizeof(uint16_t), stream_a.get()};
  PlaneBuffer d_out_b{*plane_, kCount * StagedPleRig::kRowWidth * sizeof(uint16_t), stream_b.get()};

  plane_->Check(plane_->memcpy_h2d(d_tokens_a.data(), tokens_a.data(), kCount * sizeof(int64_t), stream_a.get()), "tok_a");
  plane_->Check(plane_->memcpy_h2d(d_tokens_b.data(), tokens_b.data(), kCount * sizeof(int64_t), stream_b.get()), "tok_b");
  plane_->Check(plane_->memcpy_h2d(d_ctx_in_a.data(), h_ctx, sizeof(h_ctx), stream_a.get()), "ctx_a");
  plane_->Check(plane_->memcpy_h2d(d_ctx_in_b.data(), h_ctx, sizeof(h_ctx), stream_b.get()), "ctx_b");

  ASSERT_EQ(plane_->ple_rows(stream_a.get(), rig.device_params(), d_tokens_a.data(), static_cast<int>(kCount),
                             d_ctx_in_a.data(), d_ctx_out_a.data(), d_out_a.data()),
            0);
  ASSERT_EQ(plane_->ple_rows(stream_b.get(), rig.device_params(), d_tokens_b.data(), static_cast<int>(kCount),
                             d_ctx_in_b.data(), d_ctx_out_b.data(), d_out_b.data()),
            0);

  std::vector<uint16_t> out_a(kCount * StagedPleRig::kRowWidth, 0);
  std::vector<uint16_t> out_b(kCount * StagedPleRig::kRowWidth, 0);
  plane_->Check(plane_->memcpy_d2h(out_a.data(), d_out_a.data(), out_a.size() * sizeof(uint16_t), stream_a.get()), "d2h_a");
  plane_->Check(plane_->memcpy_d2h(out_b.data(), d_out_b.data(), out_b.size() * sizeof(uint16_t), stream_b.get()), "d2h_b");
  plane_->Check(plane_->stream_sync(stream_a.get()), "sync_a");
  plane_->Check(plane_->stream_sync(stream_b.get()), "sync_b");

  for (uint16_t v : out_a) EXPECT_EQ(v, 0x3F80u);
  for (uint16_t v : out_b) EXPECT_EQ(v, 0x4080u);
}

// Errors that CUDA itself reports, not injected ones, are propagated just as exactly.
TEST_F(Qwen38PlaneStagedPleTest, PleRowsReturnsTheCudaErrorOfAFailedCopy) {
  StagedPleRig rig{*plane_, kAscendingRows};
  ASSERT_EQ(rig.PleRows({1, 2}), 0);
  rig.FillOutputs(0xAB);
  const Qwen38PlaneTestStats before = Stats();

  // CUDA rejects a null device source for the copy of the tokens.
  const int error = rig.PleRowsOn(rig.device_params(), nullptr, 2);
  int64_t word = 0;
  EXPECT_NE(error, 0);
  EXPECT_EQ(error, plane_->memcpy_d2h(&word, nullptr, sizeof(word), rig.stream()));
  const Qwen38PlaneTestStats after = Stats();
  EXPECT_EQ(after.gather_rows, before.gather_rows);
  EXPECT_EQ(after.staged_launches, before.staged_launches);
  EXPECT_TRUE(rig.OutputsAre(0xAB));

  EXPECT_EQ(rig.PleRows({3, 0}), 0);
  EXPECT_EQ(rig.ReadPle(2), rig.ExpectedPle({3, 0}));
}

// The staged tables live in the module, which every loader of the library file shares: destroying one loader must leave the
// tables of the others working, and the state is released once, by the last loader.
TEST_F(Qwen38PlaneStagedPleTest, DestroyingOneLoaderKeepsTheStagedTablesOfTheOthers) {
  const Qwen38PlaneTestStats start = Stats();  // loader A, plane_, is attached
  auto plane_b = std::make_unique<Qwen38Plane>(library_path_);
  auto rig_a = std::make_unique<StagedPleRig>(*plane_, kAscendingRows);
  auto rig_b = std::make_unique<StagedPleRig>(*plane_b, kDescendingRows);
  const Qwen38PlaneTestStats both = Stats();
  EXPECT_EQ(both.attached_loaders, start.attached_loaders + 1);
  EXPECT_EQ(both.staged_tables, start.staged_tables + 2);

  const std::vector<int64_t> tokens = {0, 1, 2, 3};
  ASSERT_EQ(rig_b->PleRows(tokens), 0);
  EXPECT_EQ(rig_b->ReadPle(tokens.size()), rig_b->ExpectedPle(tokens));

  rig_a.reset();
  plane_.reset();
  // Checked before loader B does any more work: if its table was dropped, running on it would not be safe.
  const Qwen38PlaneTestStats after_a = Stats();
  ASSERT_EQ(after_a.attached_loaders, start.attached_loaders);
  ASSERT_EQ(after_a.global_cleanups, start.global_cleanups);
  ASSERT_EQ(after_a.staged_tables, start.staged_tables + 1);

  const std::vector<int64_t> more_tokens = {3, 3, 2, 0, 1};
  rig_b->FillOutputs(0xAB);
  ASSERT_EQ(rig_b->PleRows(more_tokens), 0);
  EXPECT_EQ(rig_b->ReadPle(more_tokens.size()), rig_b->ExpectedPle(more_tokens));
  EXPECT_EQ(Stats().staged_launches, after_a.staged_launches + 1);

  rig_b.reset();
  plane_b.reset();
  const Qwen38PlaneTestStats end = Stats();
  EXPECT_EQ(end.attached_loaders, start.attached_loaders - 1);
  EXPECT_EQ(end.global_cleanups, start.global_cleanups + 1);
  EXPECT_EQ(end.staged_tables, start.staged_tables);
  EXPECT_EQ(end.live_pinned_allocations, start.live_pinned_allocations);
}

// Tables that a loader never unregistered are released with the module, when the last loader goes.
TEST_F(Qwen38PlaneStagedPleTest, TheLastLoaderToDetachReleasesTheStagedTablesLeftRegistered) {
  const Qwen38PlaneTestStats start = Stats();
  auto rig = std::make_unique<StagedPleRig>(*plane_, kAscendingRows);
  ASSERT_EQ(rig->PleRows({1, 2}), 0);
  const Qwen38PlaneTestStats used = Stats();
  EXPECT_EQ(used.staged_tables, start.staged_tables + 1);
  EXPECT_GT(used.live_pinned_allocations, start.live_pinned_allocations);

  rig->KeepTableRegisteredOnDestruction();
  rig.reset();
  EXPECT_EQ(Stats().staged_tables, used.staged_tables);

  plane_.reset();
  const Qwen38PlaneTestStats end = Stats();
  EXPECT_EQ(end.attached_loaders, start.attached_loaders - 1);
  EXPECT_EQ(end.global_cleanups, start.global_cleanups + 1);
  EXPECT_EQ(end.staged_tables, 0);
  EXPECT_EQ(end.live_pinned_allocations, 0);
}

// The CUDA runtime records a rejected kernel launch only as the last error of the thread; the stream stays healthy, so a
// plane that returns void from the launch would otherwise leave the outputs stale without any diagnostic.
TEST_F(Qwen38PlaneStagedPleTest, ARejectedKernelLaunchIsReportedByTheNextStreamSyncOnce) {
  PlaneStream stream_a{*plane_};
  PlaneStream stream_b{*plane_};
  ASSERT_EQ(plane_->stream_sync(stream_a.get()), 0);
  ASSERT_EQ(plane_->stream_sync(stream_b.get()), 0);

  // A grid of no blocks is not a valid launch configuration, and nothing runs.
  plane_->argmax_part(stream_a.get(), nullptr, 0, nullptr, nullptr, 0);
  // Synchronizing a healthy stream B must not consume stream A's parked launch error.
  EXPECT_EQ(plane_->stream_sync(stream_b.get()), 0);
  EXPECT_NE(plane_->stream_sync(stream_a.get()), 0);
  EXPECT_EQ(plane_->stream_sync(stream_a.get()), 0);
  EXPECT_EQ(module_->PeekLastError(), 0) << "the failed launch stayed behind as the last error of the thread";
}

// Failures the host tolerates (it falls back to another registration or to the staged path) must not stay behind as the last
// error of the thread, where the next cudaGetLastError() of another CUDA user of the process would report them.
TEST_F(Qwen38PlaneStagedPleTest, FailuresTheHostToleratesAreNotLeftAsTheLastCudaError) {
  std::vector<unsigned char> memory(4096);
  void* device_pointer = nullptr;

  // A registration of no bytes is invalid whatever the flags, so the retry without the read-only flag fails as well.
  EXPECT_NE(plane_->host_register(memory.data(), 0, 8), 0);
  EXPECT_EQ(module_->PeekLastError(), 0);
  EXPECT_NE(plane_->host_register(memory.data(), 0, 1), 0);
  EXPECT_EQ(module_->PeekLastError(), 0);

  // The memory was never registered.
  EXPECT_NE(plane_->host_unregister(memory.data()), 0);
  EXPECT_EQ(module_->PeekLastError(), 0);
  EXPECT_NE(plane_->host_get_device_pointer(&device_pointer, memory.data(), 0), 0);
  EXPECT_EQ(module_->PeekLastError(), 0);
}

}  // namespace
#endif

}  // namespace Generators::test
