// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "config.h"
#include "models/qwen38_graph.h"
#include "models/qwen38_graph_plane.h"
#include "models/qwen38_replay_geometry.h"
#include "process_thread_count.h"
#include "qwen38_plane/qwen38_parallel_pool.h"

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

struct StubPlaneState {
  static inline std::vector<void*> registered_ptrs;
  static inline std::vector<void*> unregistered_ptrs;
  static inline bool fail_get_device_pointer = false;
  static constexpr uintptr_t kDeviceOffset = 0x100000u;

  // The embedding exports record their calls; the gather returns gather_error.
  struct GatherCall {
    const void* table;
    long long table_rows;
    int format;
    int n;
    int width;
  };
  static inline std::vector<GatherCall> gather_calls;
  static inline std::vector<std::pair<int, int>> legacy_gather_calls;  // (rows, width)
  static inline int gather_error = 0;

  static void Reset() {
    registered_ptrs.clear();
    unregistered_ptrs.clear();
    fail_get_device_pointer = false;
    gather_calls.clear();
    legacy_gather_calls.clear();
    gather_error = 0;
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
  static int EmbedGather(void*, const void* table, long long table_rows, int format, const void*, void*, int n, int width) {
    gather_calls.push_back({table, table_rows, format, n, width});
    return gather_error;
  }
  static void LegacyEmbedRows(void*, const void*, const void*, void*, int rows, int width) {
    legacy_gather_calls.emplace_back(rows, width);
  }
};

void InitStubPlane(Qwen38Plane& plane) {
  StubPlaneState::Reset();
  plane.error_string = &StubPlaneState::ErrorString;
  plane.host_register = &StubPlaneState::HostRegister;
  plane.host_unregister = &StubPlaneState::HostUnregister;
  plane.host_get_device_pointer = &StubPlaneState::HostGetDevicePointer;
}

// The stubs of a plane library from before the INT8 formats (only the BF16 embed_rows, no capabilities) and of one built from
// this source (embed_gather and every capability).
void InitLegacyStubPlane(Qwen38Plane& plane) {
  InitStubPlane(plane);
  plane.embed_rows = &StubPlaneState::LegacyEmbedRows;
  plane.library_path = "legacy/libqwen38_plane.so";
}

void InitCurrentStubPlane(Qwen38Plane& plane) {
  InitStubPlane(plane);
  plane.embed_rows = &StubPlaneState::LegacyEmbedRows;
  plane.embed_gather = &StubPlaneState::EmbedGather;
  plane.capabilities = QWEN38_PLANE_ALL_CAPABILITIES;
  plane.library_path = "current/libqwen38_plane.so";
}

// Runs `action`, which has to fail with the diagnostic of a missing plane capability that names the capability, what required
// it and the library.
template <typename Action>
void ExpectMissingCapability(Action&& action, const std::string& capability, const std::string& required_by,
                             const std::string& library) {
  try {
    action();
    ADD_FAILURE() << "expected the missing plane capability " << capability << " to be refused";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("requires the plane capability " + capability), std::string::npos) << message;
    EXPECT_NE(message.find(required_by), std::string::npos) << message;
    EXPECT_NE(message.find(library), std::string::npos) << message;
  }
}

// The PLE table description with INT8 shards (a negative scale) instead of FP8 shards.
std::string Int8PleTable() {
  std::string text = kValidPleTable;
  const std::string fp8_scale = "scale 0.5";
  text.replace(text.find(fp8_scale), fp8_scale.size(), "scale -0.5");
  return text;
}

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
  const auto parse = [&](const std::string& text) { return ParseQwen38PleDescription(WriteFile(table, text)); };

  EXPECT_THROW(ParseQwen38PleDescription(dir / "missing.txt"), std::runtime_error);
  EXPECT_THROW(parse("qwen38_ple_table 2\n"), std::runtime_error);                         // header
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\n"), std::runtime_error);  // no shards
  EXPECT_THROW(parse("qwen38_ple_table 1\nvocab 1 2\nn_heads 2\n"), std::runtime_error);   // vocab before n_heads
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 33\n"), std::runtime_error);
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nshard ple/a.bin 0\n"), std::runtime_error);  // rows
  EXPECT_THROW(parse("qwen38_ple_table 1\nn_heads 2\nhead_dim 4\nshard ple/a.bin 0 0\n"), std::runtime_error);

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

#if !defined(_WIN32)
TEST(Qwen38PleTableTest, ValidatesShardFileBoundsAndUsesDevicePointer) {
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

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

namespace {

// The shard files that kValidPleTable describes: shard a holds 11 rows of 4 bytes from byte 128, shard b 13 rows from byte 0.
void WritePleShards(const std::filesystem::path& dir) {
  std::filesystem::create_directories(dir / "ple");
  WriteFile(dir / "ple" / "a.safetensors", std::string(172, 'a'));
  WriteFile(dir / "ple" / "b.safetensors", std::string(52, 'b'));
}

}  // namespace

// An INT8 PLE package needs a plane that decodes negative-scale shards: a library without that capability would read them as
// FP8, so the table is refused before any shard is mapped, while FP8 packages keep working with it.
TEST(Qwen38PleTableTest, Int8ShardsAreRefusedBeforeAnyShardIsMappedWhenThePlaneLacksTheCapability) {
  const auto dir = MakeScratchDir("ple_int8_map");
  WritePleShards(dir);
  const auto fp8_path = WriteFile(dir / "fp8_table.txt", kValidPleTable);
  const auto int8_path = WriteFile(dir / "int8_table.txt", Int8PleTable());

  Qwen38Plane legacy;
  InitLegacyStubPlane(legacy);
  {
    Qwen38PleTable table{legacy, fp8_path, dir, 64};
    EXPECT_FLOAT_EQ(table.params.scale, 0.5f);
    EXPECT_EQ(StubPlaneState::registered_ptrs.size(), 2u);
  }
  StubPlaneState::Reset();
  ExpectMissingCapability([&] { Qwen38PleTable table{legacy, int8_path, dir, 64}; }, "int8_ple", int8_path.string(),
                          "legacy/libqwen38_plane.so");
  EXPECT_TRUE(StubPlaneState::registered_ptrs.empty());

  Qwen38Plane current;
  InitCurrentStubPlane(current);
  {
    Qwen38PleTable table{current, int8_path, dir, 64};
    EXPECT_FLOAT_EQ(table.params.scale, -0.5f);
    EXPECT_EQ(StubPlaneState::registered_ptrs.size(), 2u);
  }

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}
#endif

// ---- plane capabilities and the embedding table ----------------------------------------------

TEST(Qwen38PlaneCapabilityTest, EveryCapabilityIsADistinctBitWithAName) {
  const std::vector<std::pair<unsigned long long, std::string>> capabilities = {
#define QWEN38_TEST_CAPABILITY(bit, name) {bit, name},
      QWEN38_PLANE_CAPABILITIES(QWEN38_TEST_CAPABILITY)
#undef QWEN38_TEST_CAPABILITY
  };
  unsigned long long seen = 0;
  for (const auto& [bit, name] : capabilities) {
    EXPECT_NE(bit, 0ull) << name;
    EXPECT_EQ(bit & (bit - 1ull), 0ull) << name << " is not a single bit";
    EXPECT_EQ(seen & bit, 0ull) << name << " reuses the bit of another capability";
    seen |= bit;
  }
  EXPECT_EQ(seen, QWEN38_PLANE_ALL_CAPABILITIES);
}

TEST(Qwen38PlaneCapabilityTest, DescribesCapabilitiesByName) {
  EXPECT_EQ(DescribeQwen38PlaneCapabilities(0), "none");
  EXPECT_EQ(DescribeQwen38PlaneCapabilities(QWEN38_PLANE_CAP_INT8_EMBEDDING), "int8_embedding");
  EXPECT_EQ(DescribeQwen38PlaneCapabilities(QWEN38_PLANE_CAP_INT8_PLE), "int8_ple");
  EXPECT_EQ(DescribeQwen38PlaneCapabilities(QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH), "negative_pos_add_batch");
  EXPECT_EQ(DescribeQwen38PlaneCapabilities(QWEN38_PLANE_CAP_BATCHED_PREP), "batched_prep");
  EXPECT_EQ(DescribeQwen38PlaneCapabilities(QWEN38_PLANE_CAP_CONV_HISTORY_FIRST_REPLAY), "conv_history_first_replay");
  EXPECT_EQ(DescribeQwen38PlaneCapabilities(QWEN38_PLANE_ALL_CAPABILITIES),
            "int8_embedding, int8_ple, negative_pos_add_batch, batched_prep, conv_history_first_replay");
  // A bit that this build has no name for is reported, not dropped.
  EXPECT_EQ(DescribeQwen38PlaneCapabilities(QWEN38_PLANE_CAP_INT8_PLE | (1ull << 40)), "int8_ple, unknown bits 0x10000000000");
}

TEST(Qwen38PlaneCapabilityTest, RequireCapabilitiesNamesTheMissingCapabilityWhatRequiredItAndTheLibrary) {
  Qwen38Plane plane;
  plane.library_path = "model/libqwen38_plane.so";
  plane.capabilities = QWEN38_PLANE_CAP_INT8_PLE;

  EXPECT_NO_THROW(plane.RequireCapabilities(0, "a package that needs nothing"));
  EXPECT_NO_THROW(plane.RequireCapabilities(QWEN38_PLANE_CAP_INT8_PLE, "the INT8 PLE table"));

  // Only the capability that is missing is named as missing, and what the library does report is listed.
  try {
    plane.RequireCapabilities(QWEN38_PLANE_CAP_INT8_EMBEDDING | QWEN38_PLANE_CAP_INT8_PLE, "the INT8 package");
    ADD_FAILURE() << "a missing capability must be refused";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("the INT8 package requires the plane capability int8_embedding, which the plane library "
                           "model/libqwen38_plane.so does not provide"),
              std::string::npos)
        << message;
    EXPECT_NE(message.find("it reports: int8_ple"), std::string::npos) << message;
  }

  // Every missing capability is named.
  try {
    plane.RequireCapabilities(QWEN38_PLANE_ALL_CAPABILITIES, "the fold package");
    ADD_FAILURE() << "a missing capability must be refused";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("the fold package requires the plane capability int8_embedding, negative_pos_add_batch, "
                           "batched_prep, conv_history_first_replay, which"),
              std::string::npos)
        << message;
  }

  plane.capabilities = 0;  // a library that predates the capabilities export reports none
  ExpectMissingCapability([&] { plane.RequireCapabilities(QWEN38_PLANE_CAP_INT8_PLE, "the INT8 PLE table"); }, "int8_ple",
                          "the INT8 PLE table", "model/libqwen38_plane.so");
}

// The host validates what it hands to qwen38_plane_prep_batched, whatever the library does with it: the batch size is an
// explicit argument, never packed into another one, and nothing reaches a library that lacks the entry point or the capability.
namespace {

struct BatchedPrepStub {
  struct Call {
    const void* params;
    int rows;
    int batch;
    const void* state;
  };
  static inline std::vector<Call> calls;
  static inline int result = 0;

  static int Run(void*, const void* params, int rows, int batch, const void*, void*, void*, void*, void*, void*, void*,
                 void* state) {
    calls.push_back({params, rows, batch, state});
    return result;
  }
};

}  // namespace

TEST(Qwen38PlaneBatchedPrepTest, TheHostPassesTheRowsAndTheBatchAsExplicitArguments) {
  Qwen38Plane plane;
  InitCurrentStubPlane(plane);
  plane.prep_batched = &BatchedPrepStub::Run;
  BatchedPrepStub::calls.clear();
  BatchedPrepStub::result = 0;
  int params = 0;
  int state = 0;
  for (int rows = 1; rows <= QWEN38_PLANE_MAX_PREP_ROWS; ++rows) {
    for (int batch = 1; batch <= QWEN38_PLANE_MAX_PREP_BATCH; ++batch) {
      plane.PrepBatched(nullptr, &params, rows, batch, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, &state);
      ASSERT_FALSE(BatchedPrepStub::calls.empty());
      EXPECT_EQ(BatchedPrepStub::calls.back().rows, rows);
      EXPECT_EQ(BatchedPrepStub::calls.back().batch, batch);
      EXPECT_EQ(BatchedPrepStub::calls.back().params, &params);
      EXPECT_EQ(BatchedPrepStub::calls.back().state, &state);
    }
  }
  EXPECT_EQ(BatchedPrepStub::calls.size(), static_cast<size_t>(QWEN38_PLANE_MAX_PREP_ROWS) * QWEN38_PLANE_MAX_PREP_BATCH);
}

TEST(Qwen38PlaneBatchedPrepTest, TheHostRejectsARowCountOrBatchOutOfRangeBeforeCallingTheLibrary) {
  Qwen38Plane plane;
  InitCurrentStubPlane(plane);
  plane.prep_batched = &BatchedPrepStub::Run;
  BatchedPrepStub::calls.clear();
  BatchedPrepStub::result = 0;
  const auto prep = [&](int rows, int batch) {
    plane.PrepBatched(nullptr, nullptr, rows, batch, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
  };
  for (const int rows : {-8, -1, 0, QWEN38_PLANE_MAX_PREP_ROWS + 1, 0x108}) {
    EXPECT_THROW(prep(rows, 2), std::invalid_argument) << rows;
  }
  for (const int batch : {-1, 0, QWEN38_PLANE_MAX_PREP_BATCH + 1, 0x200}) {
    EXPECT_THROW(prep(4, batch), std::invalid_argument) << batch;
  }
  EXPECT_TRUE(BatchedPrepStub::calls.empty());
}

TEST(Qwen38PlaneBatchedPrepTest, ALibraryWithoutTheCapabilityOrTheExportIsNeverCalled) {
  Qwen38Plane plane;
  InitCurrentStubPlane(plane);
  plane.prep_batched = &BatchedPrepStub::Run;
  BatchedPrepStub::calls.clear();
  const auto prep = [&] {
    plane.PrepBatched(nullptr, nullptr, 2, 2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
  };

  plane.capabilities = QWEN38_PLANE_ALL_CAPABILITIES & ~QWEN38_PLANE_CAP_BATCHED_PREP;
  ExpectMissingCapability(prep, "batched_prep", "preparing the inputs of several request slots", "current/libqwen38_plane.so");

  plane.capabilities = QWEN38_PLANE_ALL_CAPABILITIES;
  plane.prep_batched = nullptr;  // the bit without the export: a broken library
  EXPECT_THROW(prep(), std::runtime_error);
  EXPECT_TRUE(BatchedPrepStub::calls.empty());
}

TEST(Qwen38PlaneBatchedPrepTest, ACudaErrorOfTheLibraryBecomesAnException) {
  Qwen38Plane plane;
  InitCurrentStubPlane(plane);
  plane.prep_batched = &BatchedPrepStub::Run;
  BatchedPrepStub::calls.clear();
  BatchedPrepStub::result = 1;  // cudaErrorInvalidValue
  try {
    plane.PrepBatched(nullptr, nullptr, 2, 2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    ADD_FAILURE() << "the error of the library was swallowed";
  } catch (const std::runtime_error& error) {
    EXPECT_NE(std::string{error.what()}.find("prep_batched"), std::string::npos) << error.what();
  }
  BatchedPrepStub::result = 0;
}

// ---- the replay descriptor of a convolution state, and the plane capability its layout needs -----------------------------

namespace {

// The replay of the convolution state of one layer as ValidateQwen38ReplayDescriptor sees it: the batch-1 shapes of the slot,
// the state channel-first [channels, history] or history-first [history, channels].
Qwen38ReplayRequest ConvolutionReplayRequest(bool history_first, int64_t channels = 8192, int64_t history = 3,
                                             int capacity = 7) {
  Qwen38ReplayRequest request;
  request.kind = Qwen38ReplayKind::kConvolution;
  request.capacity = capacity;
  request.state = {"past_key_values.5.conv_state",
                   history_first ? std::vector<int64_t>{1, history, channels} : std::vector<int64_t>{1, channels, history},
                   2};
  request.update = {"state_update.5.conv_value", {1, capacity, channels}, 2};
  return request;
}

uint32_t ConvLayoutWord(Qwen38PlaneConvStateLayout layout) { return static_cast<uint32_t>(layout); }

}  // namespace

TEST(Qwen38ConvReplayDescriptorTest, TheLayoutTookTheTailPaddingOfTheDescriptor) {
  // A plane built before the field reads the descriptors at the same stride, and a descriptor that never sets the field is
  // channel-first, which is what such a plane assumes.
  if constexpr (sizeof(void*) == 8) {
    EXPECT_EQ(sizeof(Qwen38StateReplayDesc), 152u);
    EXPECT_EQ(offsetof(Qwen38StateReplayDesc, kind), 144u);
    EXPECT_EQ(offsetof(Qwen38StateReplayDesc, layout), 148u);
  }
  const Qwen38StateReplayDesc zeroed{};
  EXPECT_EQ(zeroed.layout, ConvLayoutWord(QWEN38_PLANE_CONV_STATE_CHANNEL_FIRST));
  EXPECT_EQ(ConvLayoutWord(QWEN38_PLANE_CONV_STATE_CHANNEL_FIRST), 0u);
  EXPECT_EQ(ConvLayoutWord(QWEN38_PLANE_CONV_STATE_HISTORY_FIRST), 1u);
}

TEST(Qwen38ConvReplayDescriptorTest, TheCapabilityHasTheBitAndTheNameOfItsContract) {
  EXPECT_EQ(QWEN38_PLANE_CAP_CONV_HISTORY_FIRST_REPLAY, 0x10ull);
  EXPECT_NE(QWEN38_PLANE_ALL_CAPABILITIES & QWEN38_PLANE_CAP_CONV_HISTORY_FIRST_REPLAY, 0ull);
  EXPECT_EQ(DescribeQwen38PlaneCapabilities(QWEN38_PLANE_CAP_CONV_HISTORY_FIRST_REPLAY), "conv_history_first_replay");
}

TEST(Qwen38ConvReplayDescriptorTest, AChannelFirstStateNeedsNoCapabilityOfAnyLibrary) {
  Qwen38Plane legacy;  // predates the capabilities export
  InitLegacyStubPlane(legacy);
  Qwen38Plane without_the_layout;  // everything but the history-first replay
  InitCurrentStubPlane(without_the_layout);
  without_the_layout.capabilities = QWEN38_PLANE_ALL_CAPABILITIES & ~QWEN38_PLANE_CAP_CONV_HISTORY_FIRST_REPLAY;
  Qwen38Plane current;
  InitCurrentStubPlane(current);

  int source = 0;
  int destination = 0;
  int value = 0;
  for (const Qwen38Plane* plane : {&legacy, &without_the_layout, &current}) {
    SCOPED_TRACE(plane->library_path);
    Qwen38StateReplayDesc descriptor{};
    ASSERT_NO_THROW(descriptor = MakeQwen38ConvReplayDescriptor(*plane, ConvolutionReplayRequest(false), &source,
                                                                &destination, &value));
    EXPECT_EQ(descriptor.layout, ConvLayoutWord(QWEN38_PLANE_CONV_STATE_CHANNEL_FIRST));
    EXPECT_EQ(descriptor.kind, 1u);
    EXPECT_EQ(descriptor.channel_count, 8192u);
    EXPECT_EQ(descriptor.state_width, 3u);
    EXPECT_EQ(descriptor.capacity, 7u);
    EXPECT_EQ(descriptor.kept_count, 1u);
    EXPECT_EQ(descriptor.element_size, 2u);
    EXPECT_EQ(descriptor.source_state, &source);
    EXPECT_EQ(descriptor.destination_state, &destination);
    EXPECT_EQ(descriptor.value, &value);
  }
}

TEST(Qwen38ConvReplayDescriptorTest, AHistoryFirstStateNeedsTheCapabilityAndTheErrorNamesIt) {
  int source = 0;
  int destination = 0;
  int value = 0;
  const auto make = [&](const Qwen38Plane& plane) {
    return MakeQwen38ConvReplayDescriptor(plane, ConvolutionReplayRequest(true), &source, &destination, &value);
  };

  Qwen38Plane without_the_layout;
  InitCurrentStubPlane(without_the_layout);
  without_the_layout.capabilities = QWEN38_PLANE_ALL_CAPABILITIES & ~QWEN38_PLANE_CAP_CONV_HISTORY_FIRST_REPLAY;
  ExpectMissingCapability([&] { make(without_the_layout); }, "conv_history_first_replay",
                          "the history-first conv state past_key_values.5.conv_state", "current/libqwen38_plane.so");

  // Another capability does not stand in for it, and a library that predates the capabilities export reports none.
  Qwen38Plane batched_prep_only;
  InitCurrentStubPlane(batched_prep_only);
  batched_prep_only.capabilities = QWEN38_PLANE_CAP_BATCHED_PREP | QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH;
  ExpectMissingCapability([&] { make(batched_prep_only); }, "conv_history_first_replay", "past_key_values.5.conv_state",
                          "current/libqwen38_plane.so");
  Qwen38Plane legacy;
  InitLegacyStubPlane(legacy);
  ExpectMissingCapability([&] { make(legacy); }, "conv_history_first_replay", "past_key_values.5.conv_state",
                          "legacy/libqwen38_plane.so");

  // A library that has it receives the layout, and every other field is the one of a channel-first descriptor.
  Qwen38Plane current;
  InitCurrentStubPlane(current);
  Qwen38StateReplayDesc descriptor{};
  ASSERT_NO_THROW(descriptor = make(current));
  EXPECT_EQ(descriptor.layout, ConvLayoutWord(QWEN38_PLANE_CONV_STATE_HISTORY_FIRST));
  EXPECT_EQ(descriptor.kind, 1u);
  EXPECT_EQ(descriptor.channel_count, 8192u) << "the channels, not the history";
  EXPECT_EQ(descriptor.state_width, 3u) << "the history, not the channels";
  EXPECT_EQ(descriptor.capacity, 7u);
  EXPECT_EQ(descriptor.kept_count, 1u);
  EXPECT_EQ(descriptor.element_size, 2u);
  EXPECT_EQ(descriptor.source_state, &source);
  EXPECT_EQ(descriptor.destination_state, &destination);
  EXPECT_EQ(descriptor.value, &value);
}

// A state that has as many history steps as channels cannot be told from a channel-first one, so it is replayed as one and
// needs no capability; a history-first state that is not square is told from the channel count of the update.
TEST(Qwen38ConvReplayDescriptorTest, ASquareStateIsChannelFirstAndNeedsNoCapability) {
  Qwen38Plane legacy;
  InitLegacyStubPlane(legacy);
  int source = 0;
  int destination = 0;
  int value = 0;
  Qwen38StateReplayDesc descriptor{};
  ASSERT_NO_THROW(descriptor = MakeQwen38ConvReplayDescriptor(legacy, ConvolutionReplayRequest(true, 16, 16, 7), &source,
                                                              &destination, &value));
  EXPECT_EQ(descriptor.layout, ConvLayoutWord(QWEN38_PLANE_CONV_STATE_CHANNEL_FIRST));
  EXPECT_EQ(descriptor.channel_count, 16u);
  EXPECT_EQ(descriptor.state_width, 16u);
}

TEST(Qwen38ConvReplayDescriptorTest, AStateThatDoesNotMatchTheChannelsOfTheUpdateIsRefusedBeforeTheCapability) {
  Qwen38Plane legacy;
  InitLegacyStubPlane(legacy);
  Qwen38ReplayRequest request = ConvolutionReplayRequest(true);
  request.state.shape = {1, 100, 3};  // neither dimension is the 8192 channels of the update
  try {
    MakeQwen38ConvReplayDescriptor(legacy, request, nullptr, nullptr, nullptr);
    ADD_FAILURE() << "a state of another channel count was accepted";
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    EXPECT_NE(message.find("neither the second nor the third dimension"), std::string::npos) << message;
    EXPECT_EQ(message.find("conv_history_first_replay"), std::string::npos) << message;
  }
}

TEST(Qwen38ConvReplayDescriptorTest, OnlyAConvolutionReplayHasAConvolutionDescriptor) {
  Qwen38Plane current;
  InitCurrentStubPlane(current);
  Qwen38ReplayRequest request = ConvolutionReplayRequest(false);
  request.kind = Qwen38ReplayKind::kIndexer;
  EXPECT_THROW(MakeQwen38ConvReplayDescriptor(current, request, nullptr, nullptr, nullptr), std::invalid_argument);
  request.kind = Qwen38ReplayKind::kGatedDeltaNet;
  EXPECT_THROW(MakeQwen38ConvReplayDescriptor(current, request, nullptr, nullptr, nullptr), std::invalid_argument);
}

TEST(Qwen38PleDescriptionTest, Int8ShardsNeedTheInt8PleCapability) {
  const auto dir = MakeScratchDir("ple_int8_capability");
  const auto fp8_path = WriteFile(dir / "fp8_table.txt", kValidPleTable);
  const auto int8_path = WriteFile(dir / "int8_table.txt", Int8PleTable());
  const auto fp8_table = ParseQwen38PleDescription(fp8_path);
  const auto int8_table = ParseQwen38PleDescription(int8_path);
  EXPECT_FLOAT_EQ(fp8_table.params.scale, 0.5f);
  EXPECT_FLOAT_EQ(int8_table.params.scale, -0.5f);

  Qwen38Plane legacy;  // reports no capability
  legacy.library_path = "legacy/libqwen38_plane.so";
  EXPECT_NO_THROW(RequireQwen38PleTableSupported(legacy, fp8_table, fp8_path));
  ExpectMissingCapability([&] { RequireQwen38PleTableSupported(legacy, int8_table, int8_path); }, "int8_ple",
                          int8_path.string(), "legacy/libqwen38_plane.so");

  // INT8 embedding rows do not make INT8 PLE shards decodable.
  Qwen38Plane embedding_only;
  embedding_only.capabilities = QWEN38_PLANE_CAP_INT8_EMBEDDING;
  EXPECT_THROW(RequireQwen38PleTableSupported(embedding_only, int8_table, int8_path), std::runtime_error);

  Qwen38Plane capable;
  capable.capabilities = QWEN38_PLANE_CAP_INT8_PLE;
  EXPECT_NO_THROW(RequireQwen38PleTableSupported(capable, int8_table, int8_path));
  EXPECT_NO_THROW(RequireQwen38PleTableSupported(capable, fp8_table, fp8_path));

  std::error_code ec;
  std::filesystem::remove_all(dir, ec);
}

TEST(Qwen38EmbeddingTableTest, SizesBothLayoutsFromTheRowCount) {
  EXPECT_EQ(Qwen38EmbeddingTableBytes(Qwen38EmbeddingFormat::kBf16, 1000, 16), 32000u);
  EXPECT_EQ(Qwen38EmbeddingTableBytes(Qwen38EmbeddingFormat::kInt8, 1000, 16), 20000u);  // 1000 * 16 + 1000 * sizeof(float)
  EXPECT_EQ(Qwen38EmbeddingTableBytes(Qwen38EmbeddingFormat::kInt8, 248320, 36), 248320u * 40u);
  EXPECT_EQ(Qwen38EmbeddingTableBytes(Qwen38EmbeddingFormat::kNone, 1000, 16), 0u);
  EXPECT_EQ(Qwen38EmbeddingTableBytes(Qwen38EmbeddingFormat::kBf16, 0, 16), 0u);
  EXPECT_EQ(Qwen38EmbeddingTableBytes(Qwen38EmbeddingFormat::kInt8, 1000, 0), 0u);
}

TEST(Qwen38EmbeddingTableTest, SelectsTheLayoutOfTheFileForAnyVocabularySize) {
  Qwen38Plane plane;
  plane.capabilities = QWEN38_PLANE_ALL_CAPABILITIES;
  const std::filesystem::path file = "embedding.onnx.data";
  const auto select = [&](size_t file_bytes, long long rows, long long width) {
    return SelectQwen38EmbeddingFormat(plane, file, file_bytes, rows, width);
  };

  // A vocabulary of 1000 with a width of 16 is 32000 bytes as bfloat16 and 20000 bytes as INT8 rows followed by float scales.
  EXPECT_EQ(select(32000, 1000, 16), Qwen38EmbeddingFormat::kBf16);
  EXPECT_EQ(select(40000, 1000, 16), Qwen38EmbeddingFormat::kBf16);
  EXPECT_EQ(select(20000, 1000, 16), Qwen38EmbeddingFormat::kInt8);
  EXPECT_EQ(select(19999, 1000, 16), Qwen38EmbeddingFormat::kNone);
  EXPECT_EQ(select(0, 1000, 16), Qwen38EmbeddingFormat::kNone);

  // A vocabulary above the 248320 rows that the plane once assumed.
  const long long large = 248320 + 77;
  const size_t large_bf16 = Qwen38EmbeddingTableBytes(Qwen38EmbeddingFormat::kBf16, large, 8);
  const size_t large_int8 = Qwen38EmbeddingTableBytes(Qwen38EmbeddingFormat::kInt8, large, 8);
  EXPECT_EQ(large_int8, static_cast<size_t>(large) * 12u);
  EXPECT_EQ(select(large_bf16, large, 8), Qwen38EmbeddingFormat::kBf16);
  EXPECT_EQ(select(large_int8, large, 8), Qwen38EmbeddingFormat::kInt8);
  EXPECT_EQ(select(large_int8 - 1, large, 8), Qwen38EmbeddingFormat::kNone);

  // Without a vocabulary size or a hidden size there is no table.
  EXPECT_EQ(select(32000, 0, 16), Qwen38EmbeddingFormat::kNone);
  EXPECT_EQ(select(32000, 1000, 0), Qwen38EmbeddingFormat::kNone);
}

TEST(Qwen38EmbeddingTableTest, AnInt8TableNeedsTheInt8EmbeddingCapability) {
  const std::filesystem::path file = std::filesystem::path{"package"} / "embedding.onnx.data";
  Qwen38Plane legacy;  // reports no capability
  legacy.library_path = "legacy/libqwen38_plane.so";

  // A BF16 table stays usable, and a file that is neither layout is no table at all rather than a capability error.
  EXPECT_EQ(SelectQwen38EmbeddingFormat(legacy, file, 32000, 1000, 16), Qwen38EmbeddingFormat::kBf16);
  EXPECT_EQ(SelectQwen38EmbeddingFormat(legacy, file, 100, 1000, 16), Qwen38EmbeddingFormat::kNone);
  ExpectMissingCapability([&] { SelectQwen38EmbeddingFormat(legacy, file, 20000, 1000, 16); }, "int8_embedding",
                          file.string(), "legacy/libqwen38_plane.so");

  // INT8 PLE shards do not make INT8 embedding rows gatherable.
  Qwen38Plane ple_only;
  ple_only.capabilities = QWEN38_PLANE_CAP_INT8_PLE;
  EXPECT_THROW(SelectQwen38EmbeddingFormat(ple_only, file, 20000, 1000, 16), std::runtime_error);

  Qwen38Plane capable;
  capable.capabilities = QWEN38_PLANE_CAP_INT8_EMBEDDING;
  EXPECT_EQ(SelectQwen38EmbeddingFormat(capable, file, 20000, 1000, 16), Qwen38EmbeddingFormat::kInt8);
}

TEST(Qwen38EmbeddingTableTest, EmbedRowsGathersThroughEmbedGatherWithTheRowCountAndLayoutOfTheTable) {
  Qwen38Plane plane;
  InitCurrentStubPlane(plane);
  int device_table = 0;
  const int64_t ids[2] = {3, 4};
  uint16_t out[2] = {};
  const Qwen38EmbeddingTable bf16{&device_table, 1000, 16, false};
  const Qwen38EmbeddingTable int8{&device_table, 248320 + 77, 8, true};

  plane.EmbedRows(nullptr, bf16, ids, out, 2);
  plane.EmbedRows(nullptr, int8, ids, out, 1);
  ASSERT_EQ(StubPlaneState::gather_calls.size(), 2u);
  EXPECT_EQ(StubPlaneState::gather_calls[0].table, &device_table);
  EXPECT_EQ(StubPlaneState::gather_calls[0].table_rows, 1000);
  EXPECT_EQ(StubPlaneState::gather_calls[0].format, QWEN38_PLANE_EMBEDDING_BF16);
  EXPECT_EQ(StubPlaneState::gather_calls[0].n, 2);
  EXPECT_EQ(StubPlaneState::gather_calls[0].width, 16);
  EXPECT_EQ(StubPlaneState::gather_calls[1].table_rows, 248320 + 77);
  EXPECT_EQ(StubPlaneState::gather_calls[1].format, QWEN38_PLANE_EMBEDDING_INT8_ROW);
  EXPECT_EQ(StubPlaneState::gather_calls[1].n, 1);
  EXPECT_EQ(StubPlaneState::gather_calls[1].width, 8);  // the layout is a format, not a negative width
  EXPECT_TRUE(StubPlaneState::legacy_gather_calls.empty());

  // A CUDA error of the gather is reported.
  StubPlaneState::gather_error = 700;
  EXPECT_THROW(plane.EmbedRows(nullptr, bf16, ids, out, 2), std::runtime_error);
}

TEST(Qwen38EmbeddingTableTest, WithoutEmbedGatherOnlyBf16TablesUseTheLegacyExport) {
  Qwen38Plane legacy;
  InitLegacyStubPlane(legacy);
  int device_table = 0;
  const int64_t ids[1] = {0};
  uint16_t out[1] = {};
  const Qwen38EmbeddingTable bf16{&device_table, 1000, 16, false};
  const Qwen38EmbeddingTable int8{&device_table, 1000, 16, true};

  legacy.EmbedRows(nullptr, bf16, ids, out, 1);
  ASSERT_EQ(StubPlaneState::legacy_gather_calls.size(), 1u);
  EXPECT_EQ(StubPlaneState::legacy_gather_calls[0], std::make_pair(1, 16));

  // The legacy export has no INT8 mode: it must never be handed an INT8 table.
  ExpectMissingCapability([&] { legacy.EmbedRows(nullptr, int8, ids, out, 1); }, "int8_embedding", "an INT8 embedding table",
                          "legacy/libqwen38_plane.so");
  EXPECT_EQ(StubPlaneState::legacy_gather_calls.size(), 1u);

  Qwen38Plane bare;  // neither embedding export
  InitStubPlane(bare);
  EXPECT_THROW(bare.EmbedRows(nullptr, bf16, ids, out, 1), std::runtime_error);
}

namespace {

// A model folder to check against a plane library: genai_config.json, the PLE table description and an embedding table file
// for a vocabulary of 1000 and a hidden size of 16. RequireQwen38PackageSupportedByPlane reads nothing else, so there is no
// ONNX graph and no PLE shard.
class Qwen38TestPackage {
 public:
  static constexpr size_t kBf16TableBytes = 1000 * 16 * 2;
  static constexpr size_t kInt8TableBytes = 1000 * 16 + 1000 * 4;  // INT8 rows, then one float scale per row

  // `fold` makes it a package of the folded MTP stack (model.qwen38_graph.fold.enabled); `max_batch_size` is the number of
  // request slots it configures (model.qwen38_graph.max_batch_size).
  explicit Qwen38TestPackage(bool fold = false, int max_batch_size = 1)
      : dir_{MakeScratchDir(std::string{fold ? "package_support_fold" : "package_support"} +
                            (max_batch_size > 1 ? "_slots" + std::to_string(max_batch_size) : ""))} {
    std::string fold_stack =
        fold ? R"(,"mtp":{"enabled":true},"fold":{"enabled":true,"head_r2":{"filename":"head_r2.onnx"}})" : "";
    if (max_batch_size > 1) fold_stack += R"(,"max_batch_size":)" + std::to_string(max_batch_size);
    WriteFile(dir_ / "genai_config.json",
              R"({"model":{"type":"qwen38_flash_graph","vocab_size":1000,"context_length":64,
                  "decoder":{"filename":"model.onnx","hidden_size":16,"num_attention_heads":1,"num_hidden_layers":1,
                             "num_key_value_heads":1,"head_size":16},
                  "embedding":{"filename":"embedding.onnx"},
                  "qwen38_graph":{"plane_library":"plane.so","ple_table":"ple_table.txt",
                                  "head":{"filename_s1":"head.onnx"})" +
                  fold_stack + R"(}},
                  "search":{"max_length":64}})");
    config_ = std::make_unique<Config>(fs::path{dir_.string()}, std::string_view{});
  }

  ~Qwen38TestPackage() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  Qwen38TestPackage(const Qwen38TestPackage&) = delete;
  Qwen38TestPackage& operator=(const Qwen38TestPackage&) = delete;

  // Sets the PLE table description and the size of the embedding table file.
  void Write(const std::string& ple_description, size_t embedding_table_bytes) {
    WriteFile(ple_path(), ple_description);
    WriteFile(embedding_path(), std::string(embedding_table_bytes, '\0'));
  }

  std::filesystem::path ple_path() const { return dir_ / "ple_table.txt"; }
  std::filesystem::path embedding_path() const { return dir_ / "embedding.onnx.data"; }

  Qwen38EmbeddingFormat Require(const Qwen38Plane& plane) const {
    return RequireQwen38PackageSupportedByPlane(plane, *config_, dir_);
  }

 private:
  std::filesystem::path dir_;
  std::unique_ptr<Config> config_;
};

}  // namespace

// The model constructor runs RequireQwen38PackageSupportedByPlane when it has loaded the plane library: an INT8 package has to
// fail there, before any weight is uploaded or any session is created, and a package of another format has to keep working
// with a library from before the INT8 formats.
TEST(Qwen38PackageSupportTest, AnInt8PleTableNeedsThePleCapabilityAtModelLoad) {
  Qwen38TestPackage package;
  package.Write(Int8PleTable(), Qwen38TestPackage::kBf16TableBytes);
  Qwen38Plane legacy;
  InitLegacyStubPlane(legacy);
  ExpectMissingCapability([&] { package.Require(legacy); }, "int8_ple", package.ple_path().string(),
                          "legacy/libqwen38_plane.so");
  EXPECT_TRUE(StubPlaneState::registered_ptrs.empty());

  Qwen38Plane current;
  InitCurrentStubPlane(current);
  EXPECT_EQ(package.Require(current), Qwen38EmbeddingFormat::kBf16);
}

TEST(Qwen38PackageSupportTest, AnInt8EmbeddingTableNeedsTheEmbeddingCapabilityAtModelLoad) {
  Qwen38TestPackage package;
  package.Write(kValidPleTable, Qwen38TestPackage::kInt8TableBytes);
  Qwen38Plane legacy;
  InitLegacyStubPlane(legacy);
  ExpectMissingCapability([&] { package.Require(legacy); }, "int8_embedding", package.embedding_path().string(),
                          "legacy/libqwen38_plane.so");

  Qwen38Plane current;
  InitCurrentStubPlane(current);
  EXPECT_EQ(package.Require(current), Qwen38EmbeddingFormat::kInt8);
}

TEST(Qwen38PackageSupportTest, EachInt8FormatNeedsItsOwnCapability) {
  Qwen38TestPackage package;
  package.Write(Int8PleTable(), Qwen38TestPackage::kInt8TableBytes);

  Qwen38Plane embedding_only;
  InitCurrentStubPlane(embedding_only);
  embedding_only.capabilities = QWEN38_PLANE_CAP_INT8_EMBEDDING;
  ExpectMissingCapability([&] { package.Require(embedding_only); }, "int8_ple", package.ple_path().string(),
                          "current/libqwen38_plane.so");

  Qwen38Plane ple_only;
  InitCurrentStubPlane(ple_only);
  ple_only.capabilities = QWEN38_PLANE_CAP_INT8_PLE;
  ExpectMissingCapability([&] { package.Require(ple_only); }, "int8_embedding", package.embedding_path().string(),
                          "current/libqwen38_plane.so");

  Qwen38Plane both;
  InitCurrentStubPlane(both);
  EXPECT_EQ(package.Require(both), Qwen38EmbeddingFormat::kInt8);
}

// The folded MTP stack prepares the rows of a verify step with one negative-pos_add call (negative_pos_add_batch). A library
// without it would prepare the first row and leave the others stale, so the model must not load with it.
TEST(Qwen38PackageSupportTest, AFoldedPackageNeedsTheBatchedPrepCapabilityAtModelLoad) {
  Qwen38TestPackage folded{/*fold=*/true};
  folded.Write(kValidPleTable, Qwen38TestPackage::kBf16TableBytes);

  Qwen38Plane legacy;
  InitLegacyStubPlane(legacy);
  ExpectMissingCapability([&] { folded.Require(legacy); }, "negative_pos_add_batch", "model.qwen38_graph.fold",
                          "legacy/libqwen38_plane.so");

  // The INT8 capabilities do not stand in for it.
  Qwen38Plane int8_only;
  InitCurrentStubPlane(int8_only);
  int8_only.capabilities = QWEN38_PLANE_CAP_INT8_EMBEDDING | QWEN38_PLANE_CAP_INT8_PLE;
  ExpectMissingCapability([&] { folded.Require(int8_only); }, "negative_pos_add_batch", "model.qwen38_graph.fold",
                          "current/libqwen38_plane.so");

  Qwen38Plane batch_only;
  InitLegacyStubPlane(batch_only);
  batch_only.capabilities = QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH;
  EXPECT_EQ(folded.Require(batch_only), Qwen38EmbeddingFormat::kBf16);

  // The other capabilities of the package are still required.
  folded.Write(Int8PleTable(), Qwen38TestPackage::kBf16TableBytes);
  ExpectMissingCapability([&] { folded.Require(batch_only); }, "int8_ple", folded.ple_path().string(),
                          "legacy/libqwen38_plane.so");

  Qwen38Plane current;
  InitCurrentStubPlane(current);
  EXPECT_EQ(folded.Require(current), Qwen38EmbeddingFormat::kBf16);
}

// A configuration with several request slots prepares the rows of every slot with qwen38_plane_prep_batched (batched_prep). A
// library without it must not load the model: the rows would otherwise go through the single-slot entry point, which knows
// nothing of the slot count. A single-slot package of the same library loads.
TEST(Qwen38PackageSupportTest, SeveralRequestSlotsNeedTheBatchedPrepCapabilityAtModelLoad) {
  Qwen38TestPackage slots{/*fold=*/true, /*max_batch_size=*/2};
  slots.Write(kValidPleTable, Qwen38TestPackage::kBf16TableBytes);

  Qwen38Plane legacy;
  InitLegacyStubPlane(legacy);
  ExpectMissingCapability([&] { slots.Require(legacy); }, "negative_pos_add_batch", "model.qwen38_graph.fold",
                          "legacy/libqwen38_plane.so");

  // A library of the negative_pos_add_batch generation prepares the rows of one slot only.
  Qwen38Plane one_slot_only;
  InitLegacyStubPlane(one_slot_only);
  one_slot_only.capabilities = QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH | QWEN38_PLANE_CAP_INT8_EMBEDDING | QWEN38_PLANE_CAP_INT8_PLE;
  ExpectMissingCapability([&] { slots.Require(one_slot_only); }, "batched_prep", "model.qwen38_graph.max_batch_size",
                          "legacy/libqwen38_plane.so");

  Qwen38TestPackage single{/*fold=*/true, /*max_batch_size=*/1};
  single.Write(kValidPleTable, Qwen38TestPackage::kBf16TableBytes);
  EXPECT_EQ(single.Require(one_slot_only), Qwen38EmbeddingFormat::kBf16);

  Qwen38Plane with_batched_prep;
  InitLegacyStubPlane(with_batched_prep);
  with_batched_prep.capabilities = QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH | QWEN38_PLANE_CAP_BATCHED_PREP;
  EXPECT_EQ(slots.Require(with_batched_prep), Qwen38EmbeddingFormat::kBf16);

  // Every slot count above one needs it, the other capabilities of the package are still required.
  for (int count = 3; count <= QWEN38_PLANE_MAX_PREP_BATCH; ++count) {
    Qwen38TestPackage package{/*fold=*/true, count};
    package.Write(kValidPleTable, Qwen38TestPackage::kBf16TableBytes);
    ExpectMissingCapability([&] { package.Require(one_slot_only); }, "batched_prep", "max_batch_size",
                            "legacy/libqwen38_plane.so");
    EXPECT_EQ(package.Require(with_batched_prep), Qwen38EmbeddingFormat::kBf16) << count;
  }
  slots.Write(Int8PleTable(), Qwen38TestPackage::kBf16TableBytes);
  ExpectMissingCapability([&] { slots.Require(with_batched_prep); }, "int8_ple", slots.ple_path().string(),
                          "legacy/libqwen38_plane.so");

  Qwen38Plane current;
  InitCurrentStubPlane(current);
  EXPECT_EQ(slots.Require(current), Qwen38EmbeddingFormat::kBf16);
}

// A package without the fold stack never prepares more than one row per call, so it keeps loading with a library from before
// the batch, whatever its formats (the INT8 ones with the capabilities of their own).
TEST(Qwen38PackageSupportTest, PackagesWithoutTheFoldStackNeedNoBatchedPrepCapability) {
  Qwen38TestPackage package;
  package.Write(kValidPleTable, Qwen38TestPackage::kBf16TableBytes);
  Qwen38Plane legacy;
  InitLegacyStubPlane(legacy);
  EXPECT_EQ(package.Require(legacy), Qwen38EmbeddingFormat::kBf16);

  Qwen38Plane before_the_batch;
  InitCurrentStubPlane(before_the_batch);
  before_the_batch.capabilities = QWEN38_PLANE_CAP_INT8_EMBEDDING | QWEN38_PLANE_CAP_INT8_PLE;
  package.Write(Int8PleTable(), Qwen38TestPackage::kInt8TableBytes);
  EXPECT_EQ(package.Require(before_the_batch), Qwen38EmbeddingFormat::kInt8);
}

TEST(Qwen38PackageSupportTest, Bf16EmbeddingAndFp8PleTablesStayUsableWithALegacyLibrary) {
  Qwen38TestPackage package;
  package.Write(kValidPleTable, Qwen38TestPackage::kBf16TableBytes);
  Qwen38Plane legacy;
  InitLegacyStubPlane(legacy);
  EXPECT_EQ(package.Require(legacy), Qwen38EmbeddingFormat::kBf16);
  EXPECT_TRUE(StubPlaneState::registered_ptrs.empty());
}

TEST(Qwen38PackageSupportTest, AMissingOrTooSmallEmbeddingTableMapsNothingAndNeedsNoCapability) {
  Qwen38TestPackage package;
  package.Write(kValidPleTable, Qwen38TestPackage::kInt8TableBytes - 1);
  Qwen38Plane legacy;
  InitLegacyStubPlane(legacy);
  EXPECT_EQ(package.Require(legacy), Qwen38EmbeddingFormat::kNone);

  std::filesystem::remove(package.embedding_path());
  EXPECT_EQ(package.Require(legacy), Qwen38EmbeddingFormat::kNone);

  // The PLE table is still checked when there is no embedding table.
  WriteFile(package.ple_path(), Int8PleTable());
  ExpectMissingCapability([&] { package.Require(legacy); }, "int8_ple", package.ple_path().string(),
                          "legacy/libqwen38_plane.so");
}

TEST(Qwen38PackageSupportTest, AnUnreadablePleDescriptionFailsAtModelLoad) {
  Qwen38TestPackage package;
  package.Write("not a qwen38 PLE table\n", Qwen38TestPackage::kBf16TableBytes);
  Qwen38Plane current;
  InitCurrentStubPlane(current);
  EXPECT_THROW(package.Require(current), std::runtime_error);
}

#if !defined(QWEN38_GRAPH_NO_POSIX)
namespace {

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
#endif

#if defined(ORTGENAI_TEST_QWEN38_LEGACY_PLANE_LIB_PATH) && !defined(QWEN38_GRAPH_NO_POSIX)
// The fixture library stands for a plane library built before the INT8 embedding table and INT8 PLE shards (see
// qwen38_legacy_plane_lib.cpp): the loader has to accept it, since BF16 and FP8 packages work with it, and a package that
// needs a capability it lacks has to fail with a diagnostic when the model is loaded.
class Qwen38LegacyPlaneLibraryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    library_path_ = std::filesystem::path{ORTGENAI_TEST_QWEN38_LEGACY_PLANE_LIB_PATH};
    plane_ = std::make_unique<Qwen38Plane>(library_path_);
  }

  std::filesystem::path library_path_;
  std::unique_ptr<Qwen38Plane> plane_;
};

// An attached plane library is never unloaded: unloading a library that has launched CUDA kernels and loading it again crashed
// the CUDA runtime, so the loader that follows shares the copy that is still mapped.
TEST_F(Qwen38LegacyPlaneLibraryTest, TheLibraryStaysMappedWhenItsLoaderIsDestroyed) {
  ASSERT_TRUE(IsLibraryMapped(library_path_));
  plane_.reset();
  EXPECT_TRUE(IsLibraryMapped(library_path_)) << "the destructor unloaded the plane library";

  const Qwen38Plane again{library_path_};
  EXPECT_EQ(again.abi_version(), QWEN38_PLANE_ABI_VERSION);
}

TEST_F(Qwen38LegacyPlaneLibraryTest, LoadsWithoutTheOptionalExportsAndReportsNoCapabilities) {
  EXPECT_EQ(plane_->abi_version(), QWEN38_PLANE_ABI_VERSION);
  EXPECT_EQ(plane_->capabilities, 0ull);
  EXPECT_TRUE(plane_->embed_gather == nullptr);
  EXPECT_TRUE(plane_->embed_rows != nullptr);
  EXPECT_EQ(std::filesystem::path{plane_->library_path}, library_path_);
}

TEST_F(Qwen38LegacyPlaneLibraryTest, Bf16EmbeddingPackagesStayUsable) {
  std::vector<uint16_t> table(3 * 8);  // a vocabulary of 3 with a width of 8
  for (size_t i = 0; i < table.size(); ++i) table[i] = static_cast<uint16_t>(0x3F00 + i);
  const std::vector<int64_t> ids = {2, 0};
  std::vector<uint16_t> out(2 * 8, 0xDEAD);

  ASSERT_EQ(SelectQwen38EmbeddingFormat(*plane_, "embedding.onnx.data", table.size() * sizeof(uint16_t), 3, 8),
            Qwen38EmbeddingFormat::kBf16);
  const Qwen38EmbeddingTable bf16{table.data(), 3, 8, false};
  plane_->EmbedRows(nullptr, bf16, ids.data(), out.data(), 2);
  EXPECT_TRUE(std::equal(out.begin(), out.begin() + 8, table.begin() + 16));
  EXPECT_TRUE(std::equal(out.begin() + 8, out.end(), table.begin()));
}

TEST_F(Qwen38LegacyPlaneLibraryTest, Int8EmbeddingPackagesAreRefusedAtLoadAndNeverGathered) {
  const std::filesystem::path file = std::filesystem::path{"package"} / "embedding.onnx.data";
  // A vocabulary of 1000 with a width of 16 as INT8 rows and float scales is 20000 bytes; as bfloat16 it would be 32000.
  ExpectMissingCapability([&] { SelectQwen38EmbeddingFormat(*plane_, file, 20000, 1000, 16); }, "int8_embedding",
                          file.string(), library_path_.string());

  // Gathering it anyway would make the legacy export return without writing the rows: it is refused instead.
  std::vector<uint16_t> out(16, 0xDEAD);
  const int64_t id = 5;
  const Qwen38EmbeddingTable int8{out.data(), 1000, 16, true};
  ExpectMissingCapability([&] { plane_->EmbedRows(nullptr, int8, &id, out.data(), 1); }, "int8_embedding",
                          "an INT8 embedding table", library_path_.string());
  EXPECT_EQ(out, std::vector<uint16_t>(16, 0xDEAD));
}

TEST_F(Qwen38LegacyPlaneLibraryTest, ModelPackagesAreCheckedAgainstTheLibraryAtModelLoad) {
  Qwen38TestPackage package;
  package.Write(kValidPleTable, Qwen38TestPackage::kBf16TableBytes);
  EXPECT_EQ(package.Require(*plane_), Qwen38EmbeddingFormat::kBf16);

  package.Write(Int8PleTable(), Qwen38TestPackage::kBf16TableBytes);
  ExpectMissingCapability([&] { package.Require(*plane_); }, "int8_ple", package.ple_path().string(),
                          library_path_.string());

  package.Write(kValidPleTable, Qwen38TestPackage::kInt8TableBytes);
  ExpectMissingCapability([&] { package.Require(*plane_); }, "int8_embedding", package.embedding_path().string(),
                          library_path_.string());
}

// The folded MTP stack needs the batched prep, which a library from before it prepares one row of: the model fails at load
// instead of verifying stale rows, while the same package without the fold stack keeps working.
TEST_F(Qwen38LegacyPlaneLibraryTest, FoldedPackagesAreRefusedAtModelLoad) {
  Qwen38TestPackage folded{/*fold=*/true};
  folded.Write(kValidPleTable, Qwen38TestPackage::kBf16TableBytes);
  ExpectMissingCapability([&] { folded.Require(*plane_); }, "negative_pos_add_batch", "model.qwen38_graph.fold",
                          library_path_.string());

  Qwen38TestPackage plain;
  plain.Write(kValidPleTable, Qwen38TestPackage::kBf16TableBytes);
  EXPECT_EQ(plain.Require(*plane_), Qwen38EmbeddingFormat::kBf16);
}

// A library from before the batched prep exports no qwen38_plane_prep_batched: a package of several request slots is refused
// at model load with a capability diagnostic (never a prep of the first slot only), whatever its other settings, while a
// single-slot package of the same format keeps loading.
TEST_F(Qwen38LegacyPlaneLibraryTest, PackagesOfSeveralRequestSlotsAreRefusedAtModelLoad) {
  EXPECT_TRUE(plane_->prep_batched == nullptr);
  EXPECT_EQ(plane_->capabilities & QWEN38_PLANE_CAP_BATCHED_PREP, 0ull);

  Qwen38TestPackage slots{/*fold=*/false, /*max_batch_size=*/2};
  slots.Write(kValidPleTable, Qwen38TestPackage::kBf16TableBytes);
  ExpectMissingCapability([&] { slots.Require(*plane_); }, "batched_prep", "model.qwen38_graph.max_batch_size",
                          library_path_.string());

  Qwen38TestPackage folded_slots{/*fold=*/true, /*max_batch_size=*/4};
  folded_slots.Write(kValidPleTable, Qwen38TestPackage::kBf16TableBytes);
  ExpectMissingCapability([&] { folded_slots.Require(*plane_); }, "negative_pos_add_batch", "model.qwen38_graph.fold",
                          library_path_.string());

  Qwen38TestPackage single;
  single.Write(kValidPleTable, Qwen38TestPackage::kBf16TableBytes);
  EXPECT_EQ(single.Require(*plane_), Qwen38EmbeddingFormat::kBf16);

  // The entry point itself refuses too: nothing is prepared through the single-slot export in its place.
  EXPECT_THROW(plane_->PrepBatched(nullptr, nullptr, 2, 2, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
               std::runtime_error);
}

#if !defined(_WIN32)
TEST_F(Qwen38LegacyPlaneLibraryTest, Fp8PlePackagesStayUsableAndInt8PlePackagesAreRefused) {
  const auto dir = MakeScratchDir("ple_legacy_library");
  WritePleShards(dir);
  const auto fp8_path = WriteFile(dir / "fp8_table.txt", kValidPleTable);
  const auto int8_path = WriteFile(dir / "int8_table.txt", Int8PleTable());

  {
    Qwen38PleTable table{*plane_, fp8_path, dir, 64};
    EXPECT_FLOAT_EQ(table.params.scale, 0.5f);
    EXPECT_EQ(table.params.n_shards, 2);
  }
  ExpectMissingCapability([&] { Qwen38PleTable table{*plane_, int8_path, dir, 64}; }, "int8_ple", int8_path.string(),
                          library_path_.string());

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
  const std::filesystem::path path{ORTGENAI_TEST_QWEN38_PLANE_LIB_PATH};
#elif defined(_WIN32)
  const std::filesystem::path path = std::filesystem::current_path() / "qwen38_plane.dll";
#else
  const std::filesystem::path path = std::filesystem::current_path() / "libqwen38_plane.so";
#endif
  return path;
}

// The bfloat16 bits of `value` rounded to nearest even, as the plane rounds.
uint16_t Bf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

}  // namespace

TEST(Qwen38PlaneTest, LoadsBuiltInGenaiCudaLibraryWhenPathIsEmpty) {
  std::unique_ptr<Qwen38Plane> plane;
  try {
    plane = std::make_unique<Qwen38Plane>(std::filesystem::path{});
  } catch (const std::runtime_error& error) {
    const std::string message = error.what();
    if (message.find("failed to load the plane library") != std::string::npos ||
        message.find("is not supported") != std::string::npos) {
      GTEST_SKIP() << "onnxruntime-genai-cuda is not available: " << message;
    }
    FAIL() << message;
  }
  EXPECT_EQ(plane->abi_version(), QWEN38_PLANE_ABI_VERSION);
  EXPECT_EQ(plane->sizeof_ple_params(), static_cast<int>(sizeof(Qwen38PleParams)));
  EXPECT_NE(plane->quantize_kv, nullptr);
  EXPECT_NE(plane->embed_rows, nullptr);
  EXPECT_NE(plane->embed_gather, nullptr);
  EXPECT_NE(plane->prep_batched, nullptr);
  EXPECT_NE(plane->set_ple_host_params, nullptr);
  EXPECT_NE(plane->unset_ple_host_params, nullptr);
  EXPECT_NE(plane->dflash_select, nullptr);
  EXPECT_NE(plane->replay_state_updates, nullptr);
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
constexpr int kCudaErrorMemoryAllocation = 2;

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

  bool HasHooks() const { return inject_fault_ != nullptr && stats_ != nullptr; }

  // The (skip + 1)-th execution of `site` from now on fails once with `error`.
  void Arm(Qwen38PlaneFaultSite site, int skip, int error) const { EXPECT_EQ(inject_fault_(site, skip, error), 0); }

  void Disarm() const { inject_fault_(QWEN38_PLANE_FAULT_NONE, 0, 0); }

  Qwen38PlaneTestStats Stats() const {
    Qwen38PlaneTestStats stats{};
    EXPECT_EQ(stats_(&stats), 0);
    return stats;
  }

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

// A PLE table registered with a plane, and the device buffers to call ple_rows and prep on it. The table has one head and four
// rows of four values, and only the token takes part in the n-gram hash, so token t gathers row t % 4. Row r holds the FP8 code
// row_codes[r]. By default the table is staged (Qwen38PleParams::pad_ == 1): its FP8 rows stay in host memory and the module
// gathers them there. A table that is not staged is addressed by the GPU, as the shards are when the CUDA driver maps them: its
// rows are in device memory, nothing is registered with set_ple_host_params and prep gathers them in the kernel.
class StagedPleRig {
 public:
  static constexpr size_t kRows = 4;
  static constexpr size_t kRowWidth = 4;
  static constexpr size_t kMaxTokens = 4200;
  static constexpr size_t kPositions = 16;  // the table's capacity: how many positions past_pos has room for
  static constexpr size_t kStateWords = 14;
  static constexpr size_t kMaxFoldRows = 8;
  // The words of the state that hold the token of row r of a fold step: row 0 reads word 0, row 1 word 2 and rows 2..7 words
  // 8..13.
  static constexpr std::array<size_t, kMaxFoldRows> kFoldTokenWords = {0, 2, 8, 9, 10, 11, 12, 13};

  // `scale` is that of Qwen38PleParams: positive for FP8 shards (row_codes are FP8 E4M3 codes), the negative of the dequantization
  // scale for INT8 shards (row_codes are the int8 values).
  StagedPleRig(Qwen38Plane& plane, const std::array<unsigned char, kRows>& row_codes, float scale = 1.0f,
               bool staged = true)
      : plane_{plane},
        codes_{row_codes},
        staged_{staged},
        stream_{plane},
        d_params_{plane, sizeof(Qwen38PleParams), stream_.get()},
        d_shard_{plane, kRows * kRowWidth, stream_.get()},
        d_tokens_{plane, kMaxTokens * sizeof(int64_t), stream_.get()},
        d_ctx_in_{plane, 2 * sizeof(int64_t), stream_.get()},
        d_ctx_out_{plane, 2 * kMaxFoldRows * sizeof(int64_t), stream_.get()},
        d_out_{plane, kMaxTokens * kRowWidth * sizeof(uint16_t), stream_.get()},
        d_tok_{plane, sizeof(int64_t), stream_.get()},
        d_pos_{plane, sizeof(int64_t), stream_.get()},
        d_state_{plane, kStateWords * sizeof(int64_t), stream_.get()},
        d_input_ids_{plane, kMaxFoldRows * sizeof(int64_t), stream_.get()},
        d_position_ids_{plane, kMaxFoldRows * sizeof(int64_t), stream_.get()},
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
    if (staged_) {
      params_.shard_ptr[0] = shard_.data();
    } else {
      Upload(d_shard_, shard_.data(), shard_.size());
      params_.shard_ptr[0] = d_shard_.data();
    }
    params_.scale = scale;
    params_.pad_ = staged_ ? 1 : 0;
    params_.eos = 99;
    params_.capacity = static_cast<long long>(kPositions);
    Upload(d_params_, &params_, sizeof(params_));
    if (staged_) plane_.set_ple_host_params(&params_);
  }

  ~StagedPleRig() {
    if (staged_ && unregister_on_destruction_) plane_.unset_ple_host_params(&params_);
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

  int PleRows(const std::vector<int64_t>& tokens) {
    Upload(d_tokens_, tokens.data(), tokens.size() * sizeof(int64_t));
    return PleRowsOn(device_params(), d_tokens_.data(), tokens.size());
  }

  int PleRowsOn(const void* params, const void* tokens, size_t count) {
    return plane_.ple_rows(stream_.get(), params, tokens, static_cast<int>(count), d_ctx_in_.data(), d_ctx_out_.data(),
                           d_out_.data());
  }

  // prep with the token and the position in words of their own.
  int PrepGeneric(int64_t token, int64_t position, int pos_add, const void* params = nullptr) {
    Upload(d_tok_, &token, sizeof(token));
    Upload(d_pos_, &position, sizeof(position));
    return Prep(params, d_tok_.data(), d_pos_.data(), pos_add, d_ctx_in_.data());
  }

  // prep for row 0 of a fold: the token and the position are words 0 and 1 of the fourteen-word state, which also holds the
  // tokens of the other rows (word 2 for row 1, words 8..13 for rows 2..7).
  int PrepRowZero(const std::array<int64_t, kStateWords>& state) {
    Upload(d_state_, state.data(), sizeof(state));
    return Prep(nullptr, d_state_.data(), d_state_.data() + sizeof(int64_t), 0, d_ctx_in_.data());
  }

  // prep for row 1..7 of the fold whose row 0 ran last: its context in is the previous row's context out.
  int PrepFoldRow(int pos_add) {
    const size_t token_word = pos_add == 1 ? 2 : static_cast<size_t>(6 + pos_add);
    return Prep(nullptr, d_state_.data() + token_word * sizeof(int64_t), d_state_.data() + sizeof(int64_t), pos_add,
                d_ctx_out_.data());
  }

  // prep for the `rows` rows of a fold step in one call, which a negative pos_add asks for: row r takes its token from the
  // state word of row r (kFoldTokenWords) and runs at position state[1] + r.
  int PrepFoldStep(const std::array<int64_t, kStateWords>& state, int rows, const void* params = nullptr) {
    Upload(d_state_, state.data(), sizeof(state));
    return Prep(params, d_state_.data(), d_state_.data() + sizeof(int64_t), -rows, d_ctx_in_.data());
  }

  // What PrepFoldStep has to leave in the outputs that FillOutputs(0xAB) set: for every row its graph inputs, its slot of
  // past_pos, its PLE row and its context (the token before it, then its own); only row 0 gives the committed length. Whatever
  // lies past the last row, and the state, are untouched.
  void ExpectFoldStepResult(const std::array<int64_t, kStateWords>& state, int rows) {
    const int64_t filled = static_cast<int64_t>(0xABABABABABABABABull);
    const size_t count = static_cast<size_t>(rows);
    const int64_t position = state[1];
    std::vector<int64_t> tokens;
    std::vector<int64_t> input_ids(kMaxFoldRows, filled);
    std::vector<int64_t> position_ids(kMaxFoldRows, filled);
    std::vector<int64_t> contexts(2 * kMaxFoldRows, filled);
    std::vector<int64_t> past_pos(kPositions, filled);
    for (size_t r = 0; r < count; ++r) {
      const int64_t token = state[kFoldTokenWords[r]];
      const int64_t row_position = position + static_cast<int64_t>(r);
      tokens.push_back(token);
      input_ids[r] = token;
      position_ids[r] = row_position;
      contexts[2 * r] = r == 0 ? 0 : tokens[r - 1];  // the context in of the rig is zero
      contexts[2 * r + 1] = token;
      past_pos[static_cast<size_t>(row_position)] = row_position;
    }
    EXPECT_EQ(Read<int64_t>(d_input_ids_, kMaxFoldRows), input_ids);
    EXPECT_EQ(Read<int64_t>(d_position_ids_, kMaxFoldRows), position_ids);
    EXPECT_EQ(Read<int64_t>(d_ctx_out_, 2 * kMaxFoldRows), contexts);
    EXPECT_EQ(Read<int64_t>(d_past_pos_, kPositions), past_pos);
    EXPECT_EQ(Read<int>(d_committed_, 1)[0], position) << "the committed length is that of row 0";
    std::vector<uint16_t> ple = ExpectedPle(tokens);
    ple.resize((count + 1) * kRowWidth, 0xABAB);
    EXPECT_EQ(ReadPle(count + 1), ple);
    EXPECT_EQ(Read<int64_t>(d_state_, kStateWords), std::vector<int64_t>(state.begin(), state.end()));
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
      const unsigned char code = codes_[static_cast<size_t>(token) % kRows];
      const uint16_t value = params_.scale < 0.0f
                                 ? Bf16Bits(static_cast<float>(static_cast<int8_t>(code)) * -params_.scale)
                                 : Bf16OfFp8(code);
      expected.insert(expected.end(), kRowWidth, value);
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
  const bool staged_;
  PlaneStream stream_;
  std::array<unsigned char, kRows * kRowWidth> shard_{};
  Qwen38PleParams params_{};
  bool unregister_on_destruction_{true};
  PlaneBuffer d_params_, d_shard_, d_tokens_, d_ctx_in_, d_ctx_out_, d_out_;
  PlaneBuffer d_tok_, d_pos_, d_state_, d_input_ids_, d_position_ids_, d_committed_, d_past_pos_;
};

// The fold step of every width a verify step has (S=2 .. S=8), prepared in one call and checked row by row.
void ExpectEveryFoldStepWidthIsPreparedInOneCall(Qwen38Plane& plane, const std::array<unsigned char, 4>& codes, float scale,
                                                 bool staged) {
  ASSERT_NE(plane.capabilities & QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH, 0ull);
  StagedPleRig rig{plane, codes, scale, staged};
  const std::array<int64_t, StagedPleRig::kStateWords> state = {7, 5, 2, 0, 0, 0, 0, 0, 1, 6, 9, 4, 3, 10};
  for (int rows = 2; rows <= static_cast<int>(StagedPleRig::kMaxFoldRows); ++rows) {
    SCOPED_TRACE(std::string{staged ? "staged table, " : "device table, "} + std::to_string(rows) + " rows");
    rig.FillOutputs(0xAB);
    ASSERT_EQ(rig.PrepFoldStep(state, rows), 0);
    rig.ExpectFoldStepResult(state, rows);
  }
}

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
  const std::array<int64_t, StagedPleRig::kStateWords> state = {7, 9, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};
  ColdCall call;
  call.run = [&] { return rig.PrepRowZero(state); };
  call.expect_result = [&] { rig.ExpectPrepResult(7, 9, 0); };
  call.pinned_buffers = 2;
  call.capacity_rows = 0;
  call.gathered_rows = 1;
  ExpectEveryStagingStepFailsCleanly(rig, PrepRowZeroSteps(), call);
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

// Rows 1..7 of a fold reuse what row 0 read back and so run no staging step at all (a CUDA graph capture could not take one);
// a row 0 that failed leaves nothing for them to reuse.
TEST_F(Qwen38PlaneStagedPleTest, FoldRowsReuseTheRowZeroSnapshotOnlyWhenRowZeroSucceeded) {
  StagedPleRig rig{*plane_, kAscendingRows};
  rig.ResetTable();
  const std::array<int64_t, StagedPleRig::kStateWords> state = {7, 9, 2, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0};

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
  const std::array<int64_t, StagedPleRig::kStateWords> fold_state = {7, 5, 2, 0, 0, 0, 0, 0, 1, 6, 9, 4, 3, 10};
  EXPECT_EQ(rig.PrepFoldStep(fold_state, 4, d_unregistered.data()), kCudaErrorInvalidValue);
  const Qwen38PlaneTestStats after = Stats();
  EXPECT_EQ(after.gather_rows, registered.gather_rows);
  EXPECT_EQ(after.staged_launches, registered.staged_launches);
  EXPECT_TRUE(rig.OutputsAre(0xAB));
}

// A fold step prepared in one call (a negative pos_add) of a staged table: every row is gathered, from the one copy of the
// control words and the context that the step makes and the one wait for the stream, and the rows are prepared by a single
// kernel launch; only row 0 gets the committed length, and no row is handed anything but device memory.
TEST_F(Qwen38PlaneStagedPleTest, ABatchedFoldStepPreparesEveryRowOfAStagedTable) {
  StagedPleRig rig{*plane_, kAscendingRows};
  rig.ResetTable();
  const std::array<int64_t, StagedPleRig::kStateWords> state = {7, 5, 2, 0, 0, 0, 0, 0, 1, 6, 9, 4, 3, 10};
  for (int rows = 2; rows <= static_cast<int>(StagedPleRig::kMaxFoldRows); ++rows) {
    SCOPED_TRACE(rows);
    rig.FillOutputs(0xAB);
    const Qwen38PlaneTestStats before = Stats();
    ASSERT_EQ(rig.PrepFoldStep(state, rows), 0);
    const Qwen38PlaneTestStats after = Stats();
    EXPECT_EQ(after.gather_rows, before.gather_rows + rows);
    EXPECT_EQ(after.staged_launches, before.staged_launches + 1) << "one launch prepares every row of the step";
    rig.ExpectFoldStepResult(state, rows);
  }

  // The table is warm now: the step makes the two copies of the control words and the context and one wait, no row does.
  const int rows = static_cast<int>(StagedPleRig::kMaxFoldRows);
  rig.FillOutputs(0xAB);
  Arm(QWEN38_PLANE_FAULT_COPY, 2, 5000);
  ASSERT_EQ(rig.PrepFoldStep(state, rows), 0);
  EXPECT_EQ(Stats().fault_armed, 1) << "the step copied more than its control words and its context from the device";
  Arm(QWEN38_PLANE_FAULT_SYNC, 1, 5001);
  ASSERT_EQ(rig.PrepFoldStep(state, rows), 0);
  EXPECT_EQ(Stats().fault_armed, 1) << "the step waited for the stream more than once";
  module_->Disarm();
  rig.ExpectFoldStepResult(state, rows);
}

// A row 0 that fails stops the step: the error is the one of the failed step and no row is prepared.
TEST_F(Qwen38PlaneStagedPleTest, ABatchedFoldStepThatFailsOnRowZeroPreparesNoRow) {
  StagedPleRig rig{*plane_, kAscendingRows};
  rig.ResetTable();
  const std::array<int64_t, StagedPleRig::kStateWords> state = {7, 5, 2, 0, 0, 0, 0, 0, 1, 6, 9, 4, 3, 10};
  ASSERT_EQ(rig.PrepFoldStep(state, 4), 0);

  rig.FillOutputs(0xAB);
  const Qwen38PlaneTestStats before = Stats();
  Arm(QWEN38_PLANE_FAULT_COPY, 0, 5002);
  EXPECT_EQ(rig.PrepFoldStep(state, 4), 5002);
  const Qwen38PlaneTestStats failed = Stats();
  EXPECT_EQ(failed.fault_armed, 0) << "the injected fault never fired";
  EXPECT_EQ(failed.gather_rows, before.gather_rows);
  EXPECT_EQ(failed.staged_launches, before.staged_launches);
  EXPECT_TRUE(rig.OutputsAre(0xAB));

  ASSERT_EQ(rig.PrepFoldStep(state, 4), 0);
  rig.ExpectFoldStepResult(state, 4);
}

TEST_F(Qwen38PlaneStagedPleTest, ABatchedFoldStepPreparesEveryRowOfAnInt8StagedTable) {
  ASSERT_NE(plane_->capabilities & QWEN38_PLANE_CAP_INT8_PLE, 0ull);
  const std::array<unsigned char, StagedPleRig::kRows> codes = {0x38, 0xC0, 0x7F, 0x81};  // 56, -64, 127 and -127
  ExpectEveryFoldStepWidthIsPreparedInOneCall(*plane_, codes, -0.25f, /*staged=*/true);
}

// A table in device memory, which the kernel gathers from itself, next to a staged one that is registered with the module.
TEST_F(Qwen38PlaneStagedPleTest, ABatchedFoldStepOfADeviceTableIsNotMistakenForAStagedOne) {
  StagedPleRig staged{*plane_, kDescendingRows};
  ASSERT_EQ(staged.PleRows({1, 2}), 0);
  ExpectEveryFoldStepWidthIsPreparedInOneCall(*plane_, kAscendingRows, 1.0f, /*staged=*/false);
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

// A negative scale marks INT8 shards: the same bytes are FP8 E4M3 codes at a positive scale (here 1, -2, NaN and a tiny negative
// subnormal) and int8 values at a negative one. A plane library without QWEN38_PLANE_CAP_INT8_PLE decodes them as FP8.
TEST_F(Qwen38PlaneStagedPleTest, Int8ShardsAreDecodedAsInt8ValuesWithTheMagnitudeOfTheScale) {
  ASSERT_NE(plane_->capabilities & QWEN38_PLANE_CAP_INT8_PLE, 0ull);
  const std::array<unsigned char, StagedPleRig::kRows> codes = {0x38, 0xC0, 0x7F, 0x81};  // 56, -64, 127 and -127
  StagedPleRig rig{*plane_, codes, -0.25f};

  const std::vector<int64_t> tokens = {0, 1, 2, 3, 6};  // token t gathers row t % 4
  ASSERT_EQ(rig.PleRows(tokens), 0);
  const std::vector<uint16_t> gathered = rig.ReadPle(tokens.size());
  EXPECT_EQ(gathered, rig.ExpectedPle(tokens));
  const std::array<uint16_t, 4> bf16_of_row = {0x4160, 0xC180, 0x41FE, 0xC1FE};  // 14, -16, 31.75, -31.75
  for (size_t i = 0; i < tokens.size(); ++i) {
    for (size_t c = 0; c < StagedPleRig::kRowWidth; ++c) {
      EXPECT_EQ(gathered[i * StagedPleRig::kRowWidth + c], bf16_of_row[static_cast<size_t>(tokens[i]) % 4])
          << "token=" << tokens[i] << " c=" << c;
    }
  }

  ASSERT_EQ(rig.PrepGeneric(6, 5, 0), 0);
  rig.ExpectPrepResult(6, 5, 0);
}

// The CPU embedding table file of a test: `vocab` rows of `width` values that differ from row to row, as bfloat16 rows or as
// int8 rows followed by one float scale per row (the layouts that SelectQwen38EmbeddingFormat recognizes).
uint32_t EmbeddingValueSeed(size_t row, size_t column) {
  uint32_t h = static_cast<uint32_t>(row) * 2654435761u ^ (static_cast<uint32_t>(column) + 1u) * 40503u;
  h ^= h >> 15;
  h *= 2246822519u;
  h ^= h >> 13;
  return h;
}

std::vector<uint8_t> MakeEmbeddingTable(int64_t vocab, int width, bool int8) {
  const size_t rows = static_cast<size_t>(vocab);
  const size_t columns = static_cast<size_t>(width);
  std::vector<uint8_t> table(int8 ? rows * (columns + sizeof(float)) : rows * columns * sizeof(uint16_t));
  if (int8) {
    for (size_t row = 0; row < rows; ++row) {
      for (size_t c = 0; c < columns; ++c) {
        table[row * columns + c] = static_cast<uint8_t>(static_cast<int8_t>(EmbeddingValueSeed(row, c) % 255u - 127u));
      }
      const float scale = 0.001f * static_cast<float>(1u + EmbeddingValueSeed(row, columns) % 97u);
      std::memcpy(table.data() + rows * columns + row * sizeof(float), &scale, sizeof(scale));
    }
  } else {
    for (size_t row = 0; row < rows; ++row) {
      for (size_t c = 0; c < columns; ++c) {
        const uint16_t value = static_cast<uint16_t>(EmbeddingValueSeed(row, c) >> 8);
        std::memcpy(table.data() + (row * columns + c) * sizeof(uint16_t), &value, sizeof(value));
      }
    }
  }
  return table;
}

// What gathering the row of `id` has to produce: the (dequantized) row, or row 0 for an id outside the table.
std::vector<uint16_t> ExpectedEmbeddingRow(const std::vector<uint8_t>& table, int64_t vocab, int width, bool int8,
                                           int64_t id) {
  const size_t row = (id >= 0 && id < vocab) ? static_cast<size_t>(id) : 0;
  const size_t columns = static_cast<size_t>(width);
  std::vector<uint16_t> expected(columns);
  if (int8) {
    float scale = 0.0f;
    std::memcpy(&scale, table.data() + static_cast<size_t>(vocab) * columns + row * sizeof(float), sizeof(scale));
    for (size_t c = 0; c < columns; ++c) {
      expected[c] = Bf16Bits(static_cast<float>(static_cast<int8_t>(table[row * columns + c])) * scale);
    }
  } else {
    std::memcpy(expected.data(), table.data() + row * columns * sizeof(uint16_t), columns * sizeof(uint16_t));
  }
  return expected;
}

// Uploads `table` and `ids`, calls `gather(stream, device_table, device_ids, device_out)` and returns the gathered rows, which
// start out as 0xABAB.
std::vector<uint16_t> RunEmbeddingGather(
    Qwen38Plane& plane, const std::vector<uint8_t>& table, const std::vector<int64_t>& ids, int width,
    const std::function<void(void* stream, const void* table, const void* ids, void* out)>& gather) {
  PlaneStream stream{plane};
  std::vector<uint16_t> out(ids.size() * static_cast<size_t>(width), 0);
  PlaneBuffer d_table{plane, table.size(), stream.get()};
  PlaneBuffer d_ids{plane, ids.size() * sizeof(int64_t), stream.get()};
  PlaneBuffer d_out{plane, out.size() * sizeof(uint16_t), stream.get()};
  plane.Check(plane.memcpy_h2d(d_table.data(), table.data(), table.size(), stream.get()), "memcpy_h2d(table)");
  plane.Check(plane.memcpy_h2d(d_ids.data(), ids.data(), ids.size() * sizeof(int64_t), stream.get()), "memcpy_h2d(ids)");
  plane.Check(plane.device_memset(d_out.data(), 0xAB, d_out.size(), stream.get()), "device_memset(out)");
  gather(stream.get(), d_table.data(), d_ids.data(), d_out.data());
  plane.Check(plane.memcpy_d2h(out.data(), d_out.data(), out.size() * sizeof(uint16_t), stream.get()), "memcpy_d2h");
  plane.Check(plane.stream_sync(stream.get()), "stream_sync(embedding gather)");
  return out;
}

// Gathers `ids` from a `vocab` x `width` table through qwen38_plane_embed_gather, which is given nothing but the row count and
// the layout, and expects every row to be that of the table; an id outside [0, vocab) reads row 0.
void ExpectEmbedGatherAddressesTheTableByItsRowCount(Qwen38Plane& plane, int64_t vocab, int width, bool int8,
                                                     const std::vector<int64_t>& ids) {
  SCOPED_TRACE(std::string{int8 ? "int8" : "bfloat16"} + " table of " + std::to_string(vocab) + " rows of " +
               std::to_string(width));
  const std::vector<uint8_t> table = MakeEmbeddingTable(vocab, width, int8);
  const std::vector<uint16_t> out = RunEmbeddingGather(
      plane, table, ids, width, [&](void* stream, const void* d_table, const void* d_ids, void* d_out) {
        plane.Check(plane.embed_gather(stream, d_table, vocab, int8 ? QWEN38_PLANE_EMBEDDING_INT8_ROW : QWEN38_PLANE_EMBEDDING_BF16,
                                       d_ids, d_out, static_cast<int>(ids.size()), width),
                    "embed_gather");
      });
  const std::vector<uint16_t> row_zero = ExpectedEmbeddingRow(table, vocab, width, int8, 0);
  for (size_t r = 0; r < ids.size(); ++r) {
    const std::vector<uint16_t> expected = ExpectedEmbeddingRow(table, vocab, width, int8, ids[r]);
    const std::vector<uint16_t> row(out.begin() + static_cast<std::ptrdiff_t>(r * static_cast<size_t>(width)),
                                    out.begin() + static_cast<std::ptrdiff_t>((r + 1) * static_cast<size_t>(width)));
    EXPECT_EQ(row, expected) << "id=" << ids[r];
    const bool in_table = ids[r] > 0 && ids[r] < vocab;
    if (in_table) EXPECT_NE(expected, row_zero) << "the table cannot tell row " << ids[r] << " from row 0";
  }
}

// Loads the built plane library on device 0 for a test of its embedding exports, or skips.
class Qwen38PlaneEmbeddingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::filesystem::path plane_lib = GetBuiltPlaneLibraryPath();
    if (!std::filesystem::exists(plane_lib)) {
      GTEST_SKIP() << plane_lib.string() << " not present";
    }
    plane_ = std::make_unique<Qwen38Plane>(plane_lib);
    if (plane_->set_device(0) != 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
    ASSERT_TRUE(plane_->embed_gather != nullptr) << "the plane library has no embed_gather";
  }

  std::unique_ptr<Qwen38Plane> plane_;
};

}  // namespace

TEST_F(Qwen38PlaneEmbeddingTest, EmbedGatherDequantizesAnInt8TableAndClampsOutOfRangeTokenIds) {
  // The row count of the table the plane once assumed, with an odd-sized hidden dimension.
  constexpr int64_t kVocab = 248320;
  ExpectEmbedGatherAddressesTheTableByItsRowCount(*plane_, kVocab, 36, true, {0, 1, 123456, kVocab - 1, kVocab, -1});
}

TEST_F(Qwen38PlaneEmbeddingTest, EmbedGatherAddressesAnInt8TableSmallerThanTheOldFixedVocabularyByItsOwnRowCount) {
  // 1000 rows of 16 are 16000 bytes of int8 values followed by 1000 float scales: a 20000-byte file whose scales start at
  // byte 16000, not at the 248320 * 16 that a plane with a vocabulary of its own would read them from.
  constexpr int64_t kVocab = 1000;
  constexpr int kWidth = 16;
  ASSERT_EQ(MakeEmbeddingTable(kVocab, kWidth, true).size(), 20000u);
  ExpectEmbedGatherAddressesTheTableByItsRowCount(*plane_, kVocab, kWidth, true,
                                                  {0, 5, 999, 1000, 4000, -1, 248319, 248320, 3, 998});
}

TEST_F(Qwen38PlaneEmbeddingTest, EmbedGatherAddressesAnInt8TableLargerThanTheOldFixedVocabularyByItsOwnRowCount) {
  // Rows from 248320 on are in this table; the ids just past it read row 0.
  constexpr int64_t kVocab = 248320 + 77;
  ExpectEmbedGatherAddressesTheTableByItsRowCount(
      *plane_, kVocab, 8, true, {248319, 248320, 248321, 248396, kVocab, kVocab + 1000, 123456, -1, -248320});
}

TEST_F(Qwen38PlaneEmbeddingTest, EmbedGatherAddressesABf16TableByItsOwnRowCount) {
  ExpectEmbedGatherAddressesTheTableByItsRowCount(*plane_, 1000, 16, false, {0, 5, 999, 1000, 4000, -1, 248319, 248320});
  constexpr int64_t kVocab = 248320 + 77;
  ExpectEmbedGatherAddressesTheTableByItsRowCount(*plane_, kVocab, 8, false,
                                                  {248319, 248320, 248396, kVocab, kVocab + 1000, 0, -7});
}

TEST_F(Qwen38PlaneEmbeddingTest, EmbedGatherRejectsArgumentsThatDoNotDescribeATable) {
  PlaneStream stream{*plane_};
  PlaneBuffer table{*plane_, 4096, stream.get()};
  PlaneBuffer ids{*plane_, 4 * sizeof(int64_t), stream.get()};
  PlaneBuffer out{*plane_, 4 * 64 * sizeof(uint16_t), stream.get()};
  const auto gather = [&](long long table_rows, int format, int n, int width) {
    return plane_->embed_gather(stream.get(), table.data(), table_rows, format, ids.data(), out.data(), n, width);
  };

  EXPECT_EQ(gather(64, QWEN38_PLANE_EMBEDDING_BF16, 2, 16), 0);
  EXPECT_EQ(gather(64, QWEN38_PLANE_EMBEDDING_INT8_ROW, 2, 16), 0);
  // Nothing to gather is not an error, whatever the table.
  EXPECT_EQ(plane_->embed_gather(stream.get(), nullptr, 0, 7, nullptr, nullptr, 0, 0), 0);
  EXPECT_EQ(plane_->embed_gather(stream.get(), nullptr, 0, 7, nullptr, nullptr, -1, 0), 0);

  EXPECT_EQ(gather(0, QWEN38_PLANE_EMBEDDING_BF16, 2, 16), kCudaErrorInvalidValue);
  EXPECT_EQ(gather(-5, QWEN38_PLANE_EMBEDDING_BF16, 2, 16), kCudaErrorInvalidValue);
  EXPECT_EQ(gather(64, QWEN38_PLANE_EMBEDDING_BF16, 2, 0), kCudaErrorInvalidValue);
  // The width of an INT8 table is a layout, no longer a negative width.
  EXPECT_EQ(gather(64, QWEN38_PLANE_EMBEDDING_BF16, 2, -16), kCudaErrorInvalidValue);
  EXPECT_EQ(gather(64, QWEN38_PLANE_EMBEDDING_INT8_ROW, 2, -16), kCudaErrorInvalidValue);
  EXPECT_EQ(gather(64, 7, 2, 16), kCudaErrorInvalidValue);
  EXPECT_EQ(gather(64, -1, 2, 16), kCudaErrorInvalidValue);
  // 3 rows of 6 int8 values are 18 bytes, so the float scales that follow them would not be aligned.
  EXPECT_EQ(gather(3, QWEN38_PLANE_EMBEDDING_INT8_ROW, 2, 6), kCudaErrorInvalidValue);
  plane_->Check(plane_->stream_sync(stream.get()), "stream_sync(embed_gather arguments)");
}

TEST_F(Qwen38PlaneEmbeddingTest, EmbedRowsOfALibraryBeforeEmbedGatherStillGathersBf16RowsAndHasNoInt8Mode) {
  if (!plane_->embed_rows) {
    GTEST_SKIP() << "the plane library has no embed_rows";
  }
  constexpr int64_t kVocab = 64;
  constexpr int kWidth = 16;
  constexpr size_t kColumns = static_cast<size_t>(kWidth);
  const std::vector<uint8_t> table = MakeEmbeddingTable(kVocab, kWidth, false);
  const std::vector<int64_t> ids = {3, 63, 0};

  const std::vector<uint16_t> gathered = RunEmbeddingGather(
      *plane_, table, ids, kWidth, [&](void* stream, const void* d_table, const void* d_ids, void* d_out) {
        plane_->embed_rows(stream, d_table, d_ids, d_out, static_cast<int>(ids.size()), kWidth);
      });
  for (size_t r = 0; r < ids.size(); ++r) {
    const std::vector<uint16_t> row(gathered.begin() + static_cast<std::ptrdiff_t>(r * kColumns),
                                    gathered.begin() + static_cast<std::ptrdiff_t>((r + 1) * kColumns));
    EXPECT_EQ(row, ExpectedEmbeddingRow(table, kVocab, kWidth, false, ids[r])) << "id=" << ids[r];
  }

  // A negative width, once how an INT8 table was requested, now writes nothing instead of reading the table as bfloat16.
  const std::vector<uint16_t> untouched = RunEmbeddingGather(
      *plane_, table, ids, kWidth, [&](void* stream, const void* d_table, const void* d_ids, void* d_out) {
        plane_->embed_rows(stream, d_table, d_ids, d_out, static_cast<int>(ids.size()), -kWidth);
      });
  EXPECT_EQ(untouched, std::vector<uint16_t>(untouched.size(), static_cast<uint16_t>(0xABAB)));
}

TEST(Qwen38PlaneTest, ImplementsEveryCapabilityAndExportsTheCapabilityQuery) {
  const std::filesystem::path plane_lib = GetBuiltPlaneLibraryPath();
  if (!std::filesystem::exists(plane_lib)) {
    GTEST_SKIP() << plane_lib.string() << " not present";
  }
  Qwen38Plane plane{plane_lib};
  EXPECT_EQ(plane.capabilities, static_cast<unsigned long long>(QWEN38_PLANE_ALL_CAPABILITIES));
  EXPECT_NE(plane.capabilities & QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH, 0ull);
  EXPECT_TRUE(plane.embed_gather != nullptr);
  EXPECT_EQ(std::filesystem::path{plane.library_path}.filename(), plane_lib.filename());
  EXPECT_NO_THROW(plane.RequireCapabilities(QWEN38_PLANE_ALL_CAPABILITIES, "an INT8 package"));
}

// A negative pos_add prepares the rows of a fold step in one call; here for a table in device memory, which the kernel gathers
// from by itself (the staged tables are covered by Qwen38PlaneStagedPleTest).
TEST(Qwen38PlaneTest, PreparesEveryRowOfAFoldStepInOneCallForADeviceTable) {
  const std::filesystem::path plane_lib = GetBuiltPlaneLibraryPath();
  if (!std::filesystem::exists(plane_lib)) {
    GTEST_SKIP() << plane_lib.string() << " not present";
  }
  Qwen38Plane plane{plane_lib};
  if (plane.set_device(0) != 0) {
    GTEST_SKIP() << "No CUDA device available";
  }
  ExpectEveryFoldStepWidthIsPreparedInOneCall(plane, kAscendingRows, 1.0f, /*staged=*/false);
  const std::array<unsigned char, 4> int8_codes = {0x38, 0xC0, 0x7F, 0x81};
  if (plane.capabilities & QWEN38_PLANE_CAP_INT8_PLE) {
    ExpectEveryFoldStepWidthIsPreparedInOneCall(plane, int8_codes, -0.25f, /*staged=*/false);
  }
}

// The kernels that decide the committed state of every cycle, against references on the host: the compact replay of the recurrent
// state (k_replay_gdn and k_replay_generic), the fused prep of a fold step (k_prep_fold) and the draft selector (DflashSelect).
// They need a CUDA device and skip without one, like the tests above.

namespace {

// The replay kernels add with __fadd_rn and multiply with __fmul_rn, which round every operation to single precision and never
// contract into an fma. The reference rounds every operation too: the volatile stores keep the compiler from fusing a * b + c * d.
float ExactMultiply(float a, float b) {
  volatile float product = a * b;
  return product;
}

float ExactAdd(float a, float b) {
  volatile float sum = a + b;
  return sum;
}

// A reproducible stream of pseudo-random words (xorshift32): the same on every platform, unlike the standard distributions.
class WordStream {
 public:
  explicit WordStream(uint32_t seed) : state_{seed != 0 ? seed : 1u} {}

  uint32_t Next() {
    state_ ^= state_ << 13;
    state_ ^= state_ >> 17;
    state_ ^= state_ << 5;
    return state_;
  }

  // A float in [low, high).
  float NextFloat(float low, float high) {
    const float unit = static_cast<float>(Next() >> 8) * (1.0f / 16777216.0f);
    return low + (high - low) * unit;
  }

 private:
  uint32_t state_;
};

std::vector<float> RandomFloats(WordStream& words, size_t count, float low, float high) {
  std::vector<float> values(count);
  for (float& value : values) value = words.NextFloat(low, high);
  return values;
}

// `count` elements of `element_size` (2 or 4) bytes as random bit patterns: the replay of a convolution or of an index buffer
// copies elements without looking at their values.
std::vector<uint32_t> RandomElements(WordStream& words, size_t count, size_t element_size) {
  std::vector<uint32_t> elements(count);
  for (uint32_t& element : elements) {
    element = element_size == 2 ? (words.Next() & 0xFFFFu) : words.Next();
  }
  return elements;
}

std::vector<uint8_t> PackElements(const std::vector<uint32_t>& elements, size_t element_size) {
  std::vector<uint8_t> bytes(elements.size() * element_size);
  for (size_t i = 0; i < elements.size(); ++i) {
    if (element_size == 2) {
      const uint16_t element = static_cast<uint16_t>(elements[i]);
      std::memcpy(bytes.data() + i * element_size, &element, element_size);
    } else {
      std::memcpy(bytes.data() + i * element_size, &elements[i], element_size);
    }
  }
  return bytes;
}

std::vector<uint32_t> UnpackElements(const std::vector<uint8_t>& bytes, size_t element_size) {
  std::vector<uint32_t> elements(bytes.size() / element_size);
  for (size_t i = 0; i < elements.size(); ++i) {
    if (element_size == 2) {
      uint16_t element = 0;
      std::memcpy(&element, bytes.data() + i * element_size, element_size);
      elements[i] = element;
    } else {
      std::memcpy(&elements[i], bytes.data() + i * element_size, element_size);
    }
  }
  return elements;
}

// The IEEE bit patterns of `values`: what a bit-exact comparison looks at, a NaN or a negative zero included.
std::vector<uint32_t> BitsOf(const std::vector<float>& values) {
  std::vector<uint32_t> bits(values.size());
  if (!values.empty()) std::memcpy(bits.data(), values.data(), values.size() * sizeof(float));
  return bits;
}

// Compares two tensors element by element and reports where the first difference is instead of printing both.
template <typename T>
void ExpectSameElements(const std::vector<T>& expected, const std::vector<T>& actual, const std::string& what) {
  ASSERT_EQ(expected.size(), actual.size()) << what;
  size_t different = 0;
  size_t first = 0;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (expected[i] == actual[i]) continue;
    if (different++ == 0) first = i;
  }
  EXPECT_EQ(different, 0u) << what << ": " << different << " of " << expected.size()
                           << " elements differ, the first is element " << first << " (expected 0x" << std::hex
                           << static_cast<unsigned long long>(expected[first]) << ", got 0x"
                           << static_cast<unsigned long long>(actual[first]) << ")";
}

// Loads the built plane library on device 0 for a test of its kernels, or skips.
template <typename Base>
class PlaneKernelFixture : public Base {
 protected:
  void SetUp() override {
    const std::filesystem::path plane_lib = GetBuiltPlaneLibraryPath();
    if (!std::filesystem::exists(plane_lib)) {
      GTEST_SKIP() << plane_lib.string() << " not present";
    }
    plane_ = std::make_unique<Qwen38Plane>(plane_lib);
    if (plane_->set_device(0) != 0) {
      GTEST_SKIP() << "No CUDA device available";
    }
  }

  std::unique_ptr<Qwen38Plane> plane_;
};

// A stream of the plane and the transfers the kernel tests make on it.
class KernelTestDevice {
 public:
  explicit KernelTestDevice(Qwen38Plane& plane) : plane_{plane}, stream_{plane} {}

  void* stream() const { return stream_.get(); }

  const Qwen38Plane& plane() const { return plane_; }

  PlaneBuffer UploadBytes(const void* host, size_t bytes) {
    PlaneBuffer buffer{plane_, bytes, stream_.get()};
    plane_.Check(plane_.memcpy_h2d(buffer.data(), host, bytes, stream_.get()), "memcpy_h2d");
    return buffer;
  }

  template <typename T>
  PlaneBuffer Upload(const std::vector<T>& values) {
    return UploadBytes(values.data(), values.size() * sizeof(T));
  }

  // A buffer whose bytes all are `byte`: an output that a kernel has to overwrite, or leave alone.
  PlaneBuffer Filled(size_t bytes, int byte) {
    PlaneBuffer buffer{plane_, bytes, stream_.get()};
    plane_.Check(plane_.device_memset(buffer.data(), byte, bytes, stream_.get()), "device_memset");
    return buffer;
  }

  void Fill(const PlaneBuffer& buffer, int byte) {
    plane_.Check(plane_.device_memset(buffer.data(), byte, buffer.size(), stream_.get()), "device_memset");
  }

  template <typename T>
  std::vector<T> Download(const PlaneBuffer& buffer, size_t count) {
    std::vector<T> values(count);
    plane_.Check(plane_.memcpy_d2h(values.data(), buffer.data(), count * sizeof(T), stream_.get()), "memcpy_d2h");
    plane_.Check(plane_.stream_sync(stream_.get()), "stream_sync");
    return values;
  }

  // Leaves an error in the last-error slot of the calling thread, as a failed call does whose caller never asks for the error:
  // a memset of a null pointer fails, and nothing reads the error afterwards.
  void LeaveAnUnreadError() { EXPECT_NE(plane_.device_memset(nullptr, 0, 16, stream_.get()), 0); }

 private:
  Qwen38Plane& plane_;
  PlaneStream stream_;
};

// How a convolution state of `channels` channels over `history` steps is laid out. The compact replay supports the
// channel-first [channels, history] layout the exporter writes and the history-first [history, channels] one (the plane
// capability conv_history_first_replay, which the descriptor selects with its `layout`). A layout the replay learns adds its
// enumerator here, its case in ConvStateIndex and ConvStateShape, and its value in the instantiation of
// Qwen38PlaneConvReplayTest.
enum class ConvStateLayout { kChannelFirst,
                             kHistoryFirst };

const char* ConvStateLayoutName(ConvStateLayout layout) {
  if (layout == ConvStateLayout::kChannelFirst) return "ChannelFirst";
  if (layout == ConvStateLayout::kHistoryFirst) return "HistoryFirst";
  return "Unknown";
}

void PrintTo(ConvStateLayout layout, std::ostream* os) { *os << ConvStateLayoutName(layout); }

// The index of the element of `channel` at `step` of the history in a state of `layout`.
size_t ConvStateIndex(ConvStateLayout layout, size_t channels, size_t history, size_t channel, size_t step) {
  if (layout == ConvStateLayout::kChannelFirst) return channel * history + step;
  if (layout == ConvStateLayout::kHistoryFirst) return step * channels + channel;
  throw std::logic_error("unsupported convolution state layout");
}

// The batch-1 shape of the state of `layout` that the package declares.
std::vector<int64_t> ConvStateShape(ConvStateLayout layout, size_t channels, size_t history) {
  const int64_t channel_dimension = static_cast<int64_t>(channels);
  const int64_t history_dimension = static_cast<int64_t>(history);
  if (layout == ConvStateLayout::kChannelFirst) return {1, channel_dimension, history_dimension};
  if (layout == ConvStateLayout::kHistoryFirst) return {1, history_dimension, channel_dimension};
  throw std::logic_error("unsupported convolution state layout");
}

// A state the compact replay rewrites: its buffers on the device, the descriptor the engine builds for it, and the state the
// replay has to leave after `kept` accepted transitions, computed on the host.
class ReplayedState {
 public:
  virtual ~ReplayedState() = default;
  // Whether k_replay_gdn replays it (the descriptors of those come first in a launch); the others go to k_replay_generic.
  virtual bool IsFast() const = 0;
  virtual Qwen38StateReplayDesc Descriptor() const = 0;
  // The thread blocks per descriptor that k_replay_gdn needs for it (0 for the others).
  virtual int Blocks() const { return 0; }
  // Fills the destination with 0xAB, so that an element the replay does not write is noticed.
  virtual void ResetDestination() = 0;
  virtual void ExpectReplayedAfter(int kept) = 0;
};

// A gated delta net state of `heads_v` value heads, each a [width_v, width_k] matrix. The capsule holds the `capacity`
// transitions as the state_update output of the model packs them: the decays [capacity, heads_v], then the keys [capacity,
// heads_k, width_k], then the deltas [capacity, heads_v, width_v].
struct GatedDeltaNetShape {
  int heads_v;
  int heads_k;
  int width_v;
  int width_k;
  int capacity;
};

// The state after the first `kept` transitions: s = s * decay + key * delta for every element, in that order of rounding.
// Value head h reads key head h * heads_k / heads_v, so the value heads that share a key head are consecutive.
std::vector<float> ReplayGatedDeltaNetOnTheHost(const GatedDeltaNetShape& shape, const std::vector<float>& state,
                                                const std::vector<float>& capsule, int kept) {
  const size_t capacity = static_cast<size_t>(shape.capacity);
  const size_t heads_v = static_cast<size_t>(shape.heads_v);
  const size_t heads_k = static_cast<size_t>(shape.heads_k);
  const size_t width_v = static_cast<size_t>(shape.width_v);
  const size_t width_k = static_cast<size_t>(shape.width_k);
  const float* decay = capsule.data();
  const float* key = decay + capacity * heads_v;
  const float* delta = key + capacity * heads_k * width_k;
  std::vector<float> next(state.size());
  for (size_t head = 0; head < heads_v; ++head) {
    const size_t key_head = head * heads_k / heads_v;
    for (size_t v = 0; v < width_v; ++v) {
      for (size_t k = 0; k < width_k; ++k) {
        const size_t index = (head * width_v + v) * width_k + k;
        float s = state[index];
        for (size_t t = 0; t < static_cast<size_t>(kept); ++t) {
          const float decayed = ExactMultiply(s, decay[t * heads_v + head]);
          const float written = ExactMultiply(key[(t * heads_k + key_head) * width_k + k], delta[(t * heads_v + head) * width_v + v]);
          s = ExactAdd(decayed, written);
        }
        next[index] = s;
      }
    }
  }
  return next;
}

class GatedDeltaNetState final : public ReplayedState {
 public:
  GatedDeltaNetState(KernelTestDevice& device, const GatedDeltaNetShape& shape, uint32_t seed)
      : device_{device}, shape_{shape} {
    WordStream words{seed};
    const size_t capacity = static_cast<size_t>(shape.capacity);
    const size_t heads_v = static_cast<size_t>(shape.heads_v);
    const size_t heads_k = static_cast<size_t>(shape.heads_k);
    const size_t width_v = static_cast<size_t>(shape.width_v);
    const size_t width_k = static_cast<size_t>(shape.width_k);
    state_ = RandomFloats(words, heads_v * width_v * width_k, -2.0f, 2.0f);
    capsule_ = RandomFloats(words, capacity * heads_v, 0.5f, 1.0f);
    const std::vector<float> keys = RandomFloats(words, capacity * heads_k * width_k, -1.0f, 1.0f);
    const std::vector<float> deltas = RandomFloats(words, capacity * heads_v * width_v, -1.0f, 1.0f);
    capsule_.insert(capsule_.end(), keys.begin(), keys.end());
    capsule_.insert(capsule_.end(), deltas.begin(), deltas.end());
    source_ = device.Upload(state_);
    capsule_device_ = device.Upload(capsule_);
    destination_ = device.Filled(state_.size() * sizeof(float), 0xAB);
  }

  bool IsFast() const override { return true; }

  Qwen38StateReplayDesc Descriptor() const override {
    const uint64_t capacity = static_cast<uint64_t>(shape_.capacity);
    const uint64_t heads_v = static_cast<uint64_t>(shape_.heads_v);
    const uint64_t heads_k = static_cast<uint64_t>(shape_.heads_k);
    const uint64_t width_k = static_cast<uint64_t>(shape_.width_k);
    const float* decay = reinterpret_cast<const float*>(capsule_device_.data());
    const float* key = decay + capacity * heads_v;
    const float* delta = key + capacity * heads_k * width_k;
    Qwen38StateReplayDesc d{};
    d.source_state = source_.data();
    d.destination_state = destination_.data();
    d.decay = decay;
    d.key = key;
    d.delta = delta;
    d.channel_count = heads_v;
    d.state_width = static_cast<uint64_t>(shape_.width_v);
    d.key_width = width_k;
    d.key_head_count = heads_k;
    d.capacity = static_cast<uint32_t>(shape_.capacity);
    d.kept_count = 1;
    d.element_size = 4;
    d.kind = 2;
    return d;
  }

  // The thread blocks the engine budgets for the geometry it derives from the shapes of the package.
  int Blocks() const override {
    const int64_t capacity = shape_.capacity;
    const int64_t per_transition = int64_t{shape_.heads_v} + int64_t{shape_.heads_k} * shape_.width_k +
                                   int64_t{shape_.heads_v} * shape_.width_v;
    Qwen38ReplayRequest request;
    request.kind = Qwen38ReplayKind::kGatedDeltaNet;
    request.capacity = shape_.capacity;
    request.state = {"recurrent_state", {1, shape_.heads_v, shape_.width_v, shape_.width_k}, sizeof(float)};
    request.update = {"recurrent_capsule", {1, capacity * per_transition}, sizeof(float)};
    return ValidateQwen38ReplayDescriptor(request).blocks;
  }

  void ResetDestination() override { device_.Fill(destination_, 0xAB); }

  void ExpectReplayedAfter(int kept) override {
    ExpectSameElements(BitsOf(ReplayGatedDeltaNetOnTheHost(shape_, state_, capsule_, kept)),
                       BitsOf(device_.Download<float>(destination_, state_.size())),
                       "gated delta net state [" + std::to_string(shape_.heads_v) + " value heads of " +
                           std::to_string(shape_.width_v) + " x " + std::to_string(shape_.width_k) + ", " +
                           std::to_string(shape_.heads_k) + " key heads] after " + std::to_string(kept) + " of " +
                           std::to_string(shape_.capacity) + " transitions");
  }

 private:
  KernelTestDevice& device_;
  GatedDeltaNetShape shape_;
  std::vector<float> state_;
  std::vector<float> capsule_;
  PlaneBuffer source_, capsule_device_, destination_;
};

struct ConvolutionShape {
  size_t channels;
  size_t history;
  int capacity;
  size_t element_size;
};

// The state after `kept` steps of `update` ([capacity, channels]: one row per step) were appended to the history: the last
// `history` steps of the old history followed by the kept steps. Written in terms of that concatenation, whatever the layout.
std::vector<uint32_t> ReplayConvolutionOnTheHost(ConvStateLayout layout, const ConvolutionShape& shape,
                                                 const std::vector<uint32_t>& state, const std::vector<uint32_t>& update,
                                                 int kept) {
  std::vector<uint32_t> next(state.size());
  for (size_t channel = 0; channel < shape.channels; ++channel) {
    for (size_t step = 0; step < shape.history; ++step) {
      const size_t source_step = step + static_cast<size_t>(kept);
      next[ConvStateIndex(layout, shape.channels, shape.history, channel, step)] =
          source_step < shape.history
              ? state[ConvStateIndex(layout, shape.channels, shape.history, channel, source_step)]
              : update[(source_step - shape.history) * shape.channels + channel];
    }
  }
  return next;
}

class ConvolutionState final : public ReplayedState {
 public:
  ConvolutionState(KernelTestDevice& device, ConvStateLayout layout, const ConvolutionShape& shape, uint32_t seed)
      : device_{device}, layout_{layout}, shape_{shape} {
    WordStream words{seed};
    state_ = RandomElements(words, shape.channels * shape.history, shape.element_size);
    update_ = RandomElements(words, static_cast<size_t>(shape.capacity) * shape.channels, shape.element_size);
    Upload();
  }

  // The state in the order of `layout` and the update as [capacity, channels], element by element.
  ConvolutionState(KernelTestDevice& device, ConvStateLayout layout, const ConvolutionShape& shape,
                   std::vector<uint32_t> state, std::vector<uint32_t> update)
      : device_{device}, layout_{layout}, shape_{shape}, state_{std::move(state)}, update_{std::move(update)} {
    Upload();
  }

  bool IsFast() const override { return false; }

  // The descriptor of the state, built as the engine builds it (MakeQwen38ConvReplayDescriptor from the batch-1 shapes of the
  // package, so the layout is found from the shapes and the plane has to report the capability a history-first state needs).
  Qwen38StateReplayDesc Descriptor() const override {
    Qwen38ReplayRequest request;
    request.kind = Qwen38ReplayKind::kConvolution;
    request.capacity = shape_.capacity;
    request.state = {"past_key_values.0.conv_state", ConvStateShape(layout_, shape_.channels, shape_.history),
                     shape_.element_size};
    request.update = {"state_update.0.conv_value",
                      {1, static_cast<int64_t>(shape_.capacity), static_cast<int64_t>(shape_.channels)},
                      shape_.element_size};
    return MakeQwen38ConvReplayDescriptor(device_.plane(), request, source_.data(), destination_.data(),
                                          update_device_.data());
  }

  void ResetDestination() override { device_.Fill(destination_, 0xAB); }

  // What the replay wrote to the destination, element by element in the order of the layout.
  std::vector<uint32_t> Replayed() const {
    return UnpackElements(device_.Download<uint8_t>(destination_, destination_.size()), shape_.element_size);
  }

  void ExpectReplayedAfter(int kept) override {
    ExpectSameElements(ReplayConvolutionOnTheHost(layout_, shape_, state_, update_, kept), Replayed(),
                       std::string{ConvStateLayoutName(layout_)} + " convolution state of " + std::to_string(shape_.channels) +
                           " channels over " + std::to_string(shape_.history) + " steps of " +
                           std::to_string(shape_.element_size) + " bytes after " + std::to_string(kept) + " of " +
                           std::to_string(shape_.capacity) + " steps");
  }

 private:
  void Upload() {
    source_ = device_.Upload(PackElements(state_, shape_.element_size));
    update_device_ = device_.Upload(PackElements(update_, shape_.element_size));
    destination_ = device_.Filled(state_.size() * shape_.element_size, 0xAB);
  }

  KernelTestDevice& device_;
  ConvStateLayout layout_;
  ConvolutionShape shape_;
  std::vector<uint32_t> state_;
  std::vector<uint32_t> update_;
  PlaneBuffer source_, update_device_, destination_;
};

// The buffer of the partly filled block of the compressed index keys: `entries` rows of `width` elements, of which the first
// `buffer_length` are in use, and the two lengths words of the layer (the compressed keys so far, the entries in the buffer).
struct IndexerShape {
  size_t entries;
  size_t width;
  int capacity;
  size_t element_size;
  int ratio;
  int key_length;
  int buffer_length;
};

class IndexerState final : public ReplayedState {
 public:
  IndexerState(KernelTestDevice& device, const IndexerShape& shape, uint32_t seed) : device_{device}, shape_{shape} {
    WordStream words{seed};
    buffer_ = RandomElements(words, shape.entries * shape.width, shape.element_size);
    update_ = RandomElements(words, static_cast<size_t>(shape.capacity) * shape.width, shape.element_size);
    source_ = device.Upload(PackElements(buffer_, shape.element_size));
    update_device_ = device.Upload(PackElements(update_, shape.element_size));
    source_lengths_ = device.Upload(std::vector<int32_t>{shape.key_length, shape.buffer_length});
    destination_ = device.Filled(buffer_.size() * shape.element_size, 0xAB);
    destination_lengths_ = device.Filled(2 * sizeof(int32_t), 0xAB);
  }

  bool IsFast() const override { return false; }

  Qwen38StateReplayDesc Descriptor() const override {
    Qwen38StateReplayDesc d{};
    d.value = update_device_.data();
    d.source_aux_state = source_.data();
    d.destination_aux_state = destination_.data();
    d.source_lengths = reinterpret_cast<const int32_t*>(source_lengths_.data());
    d.destination_lengths = reinterpret_cast<int32_t*>(destination_lengths_.data());
    d.state_width = shape_.width;
    d.state_capacity = 0;
    d.aux_capacity = shape_.entries;
    d.capacity = static_cast<uint32_t>(shape_.capacity);
    d.kept_count = 1;
    d.element_size = static_cast<uint32_t>(shape_.element_size);
    d.compress_ratio = static_cast<uint32_t>(shape_.ratio);
    d.kind = 4;
    return d;
  }

  void ResetDestination() override {
    device_.Fill(destination_, 0xAB);
    device_.Fill(destination_lengths_, 0xAB);
  }

  // The buffer and the update make one run of entries [buffer_length old ones, then the kept new ones]. Every `ratio` of them
  // from the start are pooled into one more compressed key; the rest are the entries of the new buffer, whose other rows hold
  // nothing that is read.
  void ExpectReplayedAfter(int kept) override {
    const size_t old_length = static_cast<size_t>(shape_.buffer_length);
    const size_t total = old_length + static_cast<size_t>(kept);
    const size_t ratio = static_cast<size_t>(shape_.ratio);
    const size_t blocks = total / ratio;
    const size_t remainder = total % ratio;
    std::vector<uint32_t> expected;
    for (size_t entry = 0; entry < remainder; ++entry) {
      const size_t position = blocks * ratio + entry;
      for (size_t component = 0; component < shape_.width; ++component) {
        expected.push_back(position < old_length ? buffer_[position * shape_.width + component]
                                                 : update_[(position - old_length) * shape_.width + component]);
      }
    }
    const std::string what = "index buffer of " + std::to_string(shape_.entries) + " entries of " +
                             std::to_string(shape_.width) + " x " + std::to_string(shape_.element_size) +
                             " bytes holding " + std::to_string(shape_.buffer_length) + ", ratio " +
                             std::to_string(shape_.ratio) + ", after " + std::to_string(kept) + " of " +
                             std::to_string(shape_.capacity) + " rows";
    std::vector<uint32_t> actual = UnpackElements(device_.Download<uint8_t>(destination_, destination_.size()), shape_.element_size);
    actual.resize(expected.size());
    ExpectSameElements(expected, actual, what);
    EXPECT_EQ(device_.Download<int32_t>(destination_lengths_, 2),
              (std::vector<int32_t>{shape_.key_length + static_cast<int32_t>(blocks), static_cast<int32_t>(remainder)}))
        << what << ": the lengths";
  }

 private:
  KernelTestDevice& device_;
  IndexerShape shape_;
  std::vector<uint32_t> buffer_;
  std::vector<uint32_t> update_;
  PlaneBuffer source_, update_device_, source_lengths_, destination_, destination_lengths_;
};

// One launch of the replay over `states` with `kept` accepted transitions, laid out as the engine lays the descriptors out (the
// k_replay_gdn ones first), followed by the comparison of every destination with the host reference.
void ReplayAndExpect(Qwen38Plane& plane, KernelTestDevice& device, const std::vector<ReplayedState*>& states, int kept) {
  std::vector<Qwen38StateReplayDesc> descriptors;
  int fast_count = 0;
  int generic_count = 0;
  int fast_blocks = 0;
  for (ReplayedState* state : states) {
    if (!state->IsFast()) continue;
    descriptors.push_back(state->Descriptor());
    ++fast_count;
    fast_blocks = state->Blocks() > fast_blocks ? state->Blocks() : fast_blocks;
  }
  for (ReplayedState* state : states) {
    if (state->IsFast()) continue;
    descriptors.push_back(state->Descriptor());
    ++generic_count;
  }
  for (ReplayedState* state : states) state->ResetDestination();
  const PlaneBuffer device_descriptors = device.Upload(descriptors);
  ASSERT_EQ(plane.replay_state_updates(device.stream(), device_descriptors.data(), fast_count, generic_count, fast_blocks, kept), 0);
  for (ReplayedState* state : states) state->ExpectReplayedAfter(kept);
}

class Qwen38PlaneReplayTest : public PlaneKernelFixture<::testing::Test> {};

class Qwen38PlaneConvReplayTest : public PlaneKernelFixture<::testing::TestWithParam<ConvStateLayout>> {};

}  // namespace

TEST_F(Qwen38PlaneReplayTest, GatedDeltaNetReplaysEveryAcceptanceLengthBitExactly) {
  ASSERT_TRUE(plane_->replay_state_updates != nullptr) << "the plane library has no replay_state_updates";
  KernelTestDevice device{*plane_};
  // Value and key widths that fill neither the 32 rows of a block nor the 128 key elements a warp covers, key heads shared by
  // several value heads, a value width one past a block, the widest key with the most transitions the kernel holds, and the
  // smallest state.
  const GatedDeltaNetShape shapes[] = {
      {4, 2, 100, 132, 7},
      {6, 2, 33, 128, 4},
      {3, 3, 64, QWEN38_PLANE_MAX_REPLAY_KEY_WIDTH, QWEN38_PLANE_MAX_REPLAY_TRANSITIONS},
      {2, 1, 32, 4, 1},
  };
  uint32_t seed = 1;
  for (const GatedDeltaNetShape& shape : shapes) {
    GatedDeltaNetState state{device, shape, seed++};
    // Partial acceptance from one transition up to the full capsule.
    for (int kept = 1; kept <= shape.capacity; ++kept) {
      ReplayAndExpect(*plane_, device, {&state}, kept);
    }
  }
}

TEST_F(Qwen38PlaneReplayTest, IndexerReplaysEveryBufferFillAndAcceptanceLength) {
  ASSERT_TRUE(plane_->replay_state_updates != nullptr) << "the plane library has no replay_state_updates";
  KernelTestDevice device{*plane_};
  constexpr int kCapacity = 7;
  uint32_t seed = 50;
  for (size_t element_size : {size_t{2}, size_t{4}}) {
    for (size_t width : {size_t{6}, size_t{128}}) {
      // Every number of entries the buffer can hold before the update: the block is completed by some accepted rows and
      // overflows into the next one for others.
      for (int buffer_length = 0; buffer_length < kQwen38IndexKeyRatio; ++buffer_length) {
        IndexerState state{device, {7, width, kCapacity, element_size, kQwen38IndexKeyRatio, 10 + buffer_length, buffer_length}, seed++};
        for (int kept = 1; kept <= kCapacity; ++kept) {
          ReplayAndExpect(*plane_, device, {&state}, kept);
        }
      }
    }
  }
}

TEST_F(Qwen38PlaneReplayTest, OneLaunchReplaysEveryKindOfStateOfAModel) {
  ASSERT_TRUE(plane_->replay_state_updates != nullptr) << "the plane library has no replay_state_updates";
  KernelTestDevice device{*plane_};
  constexpr int kCapacity = 7;
  GatedDeltaNetState wide{device, {4, 2, 100, 132, kCapacity}, 11};
  GatedDeltaNetState narrow{device, {6, 3, 33, 64, kCapacity}, 12};
  ConvolutionState conv_bf16{device, ConvStateLayout::kChannelFirst, {5, 3, kCapacity, 2}, 13};
  ConvolutionState conv_fp32{device, ConvStateLayout::kChannelFirst, {300, 4, kCapacity, 4}, 14};
  IndexerState index_bf16{device, {7, 128, kCapacity, 2, kQwen38IndexKeyRatio, 21, 2}, 15};
  IndexerState index_fp32{device, {7, 6, kCapacity, 4, kQwen38IndexKeyRatio, 33, 3}, 16};
  const std::vector<ReplayedState*> states = {&conv_bf16, &wide, &index_fp32, &conv_fp32, &narrow, &index_bf16};
  for (int kept : {1, 3, kCapacity}) {
    SCOPED_TRACE("kept " + std::to_string(kept));
    ReplayAndExpect(*plane_, device, states, kept);
  }
}

TEST_F(Qwen38PlaneReplayTest, RejectsAnAcceptanceLengthTheKernelsCannotReplay) {
  ASSERT_TRUE(plane_->replay_state_updates != nullptr) << "the plane library has no replay_state_updates";
  KernelTestDevice device{*plane_};
  GatedDeltaNetState state{device, {2, 1, 32, 4, 2}, 7};
  const PlaneBuffer descriptors = device.Upload(std::vector<Qwen38StateReplayDesc>{state.Descriptor()});
  void* stream = device.stream();
  for (int kept : {0, -1, QWEN38_PLANE_MAX_REPLAY_TRANSITIONS + 1}) {
    EXPECT_EQ(plane_->replay_state_updates(stream, descriptors.data(), 1, 0, state.Blocks(), kept), kCudaErrorInvalidValue)
        << "kept " << kept;
  }
  EXPECT_EQ(plane_->replay_state_updates(stream, nullptr, 1, 0, state.Blocks(), 1), kCudaErrorInvalidValue);
}

// The error of an earlier call that nobody read (here a memset of a null pointer) is not the error of the launch that follows:
// a replay that reported it would fail a cycle whose state is sound.
TEST_F(Qwen38PlaneReplayTest, AnErrorAnEarlierCallLeftUnreadDoesNotFailTheLaunch) {
  ASSERT_TRUE(plane_->replay_state_updates != nullptr) << "the plane library has no replay_state_updates";
  KernelTestDevice device{*plane_};
  GatedDeltaNetState gdn{device, {2, 1, 32, 4, 3}, 7};
  ConvolutionState conv{device, ConvStateLayout::kChannelFirst, {5, 3, 3, 2}, 8};
  device.LeaveAnUnreadError();
  ReplayAndExpect(*plane_, device, {&gdn}, 2);
  device.LeaveAnUnreadError();
  ReplayAndExpect(*plane_, device, {&conv}, 2);
  device.LeaveAnUnreadError();
  ReplayAndExpect(*plane_, device, {&conv, &gdn}, 3);
}

// The replay of a convolution state in every layout it supports against the host reference; a layout the replay learns is added
// to the instantiation below. The state and the update are random bit patterns, so a channel read at another position, a
// position of another channel or a transposed state shows as a difference.
TEST_P(Qwen38PlaneConvReplayTest, ReplaysEveryAcceptanceLengthBitExactly) {
  ASSERT_TRUE(plane_->replay_state_updates != nullptr) << "the plane library has no replay_state_updates";
  KernelTestDevice device{*plane_};
  // The usual kernel of 4 taps (a history of 3), a history of 1 that the first kept step already pushes out, a history shorter
  // than the acceptance, a history longer than the channels (a state that is not channel-first has its history in the second
  // dimension, so it is the shape that tells the layouts apart), more channels than the 128 blocks of 256 threads cover with
  // one element each, and a history that does not divide the channels. No shape is square: the layouts of a square state
  // cannot be told apart and it is replayed as channel-first (Qwen38ConvReplayDescriptorTest).
  const ConvolutionShape shapes[] = {
      {5, 3, 7, 0},
      {9, 1, 3, 0},
      {64, 2, 7, 0},
      {3, 5, 7, 0},
      {2, 9, 6, 0},
      {9000, 4, 7, 0},
      {1025, 3, 5, 0},
  };
  uint32_t seed = 200;
  for (size_t element_size : {size_t{2}, size_t{4}}) {
    for (ConvolutionShape shape : shapes) {
      shape.element_size = element_size;
      ConvolutionState state{device, GetParam(), shape, seed++};
      for (int kept = 1; kept <= shape.capacity; ++kept) {
        ReplayAndExpect(*plane_, device, {&state}, kept);
      }
    }
  }
}

// The state of three channels over a history of two steps, worked out by hand for both layouts (the example the review of the
// layout gave: a [batch, history = 2, channels = 3] state, where channel 0 at position 1 is element 3 and not element 1).
// The value of the old state at step s of channel c is 10 * (s + 1) + c, that of the update at step s is 100 + 10 * s + c, and
// the new state is the last two steps of [old step 0, old step 1, update step 0, update step 1, update step 2] after the
// first `kept` steps of that sequence are dropped.
TEST_P(Qwen38PlaneConvReplayTest, ReplaysAStateWorkedOutByHandToTheValuesOfItsLayout) {
  ASSERT_TRUE(plane_->replay_state_updates != nullptr) << "the plane library has no replay_state_updates";
  KernelTestDevice device{*plane_};
  const ConvStateLayout layout = GetParam();
  const bool history_first = layout == ConvStateLayout::kHistoryFirst;

  const std::vector<uint32_t> state =
      history_first ? std::vector<uint32_t>{10, 11, 12, 20, 21, 22} : std::vector<uint32_t>{10, 20, 11, 21, 12, 22};
  const std::vector<uint32_t> update = {100, 101, 102, 110, 111, 112, 120, 121, 122};
  const std::vector<std::vector<uint32_t>> expected_after_1_2_3_kept =
      history_first ? std::vector<std::vector<uint32_t>>{{20, 21, 22, 100, 101, 102},
                                                         {100, 101, 102, 110, 111, 112},
                                                         {110, 111, 112, 120, 121, 122}}
                    : std::vector<std::vector<uint32_t>>{{20, 100, 21, 101, 22, 102},
                                                         {100, 110, 101, 111, 102, 112},
                                                         {110, 120, 111, 121, 112, 122}};

  for (size_t element_size : {size_t{2}, size_t{4}}) {
    ConvolutionState replayed{device, layout, {3, 2, 3, element_size}, state, update};
    for (int kept = 1; kept <= 3; ++kept) {
      SCOPED_TRACE(std::string{ConvStateLayoutName(layout)} + " state of " + std::to_string(element_size) +
                   " byte elements after " + std::to_string(kept) + " of 3 steps");
      ReplayAndExpect(*plane_, device, {&replayed}, kept);
      EXPECT_EQ(replayed.Replayed(), expected_after_1_2_3_kept[static_cast<size_t>(kept - 1)]);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(SupportedLayouts, Qwen38PlaneConvReplayTest,
                         ::testing::Values(ConvStateLayout::kChannelFirst, ConvStateLayout::kHistoryFirst),
                         [](const ::testing::TestParamInfo<ConvStateLayout>& layout_info) {
                           return std::string{ConvStateLayoutName(layout_info.param)};
                         });

// The states of several layers of one request slot in one launch, as the engine replays them: convolution states of both layouts
// (the package declares one layout per state, and a model can mix them), different widths and element sizes, among the other
// kinds of state, and every acceptance length. The two history-first states are replayed beside the channel-first ones, so a
// descriptor that carries the layout of its neighbour is found.
TEST_F(Qwen38PlaneReplayTest, OneLaunchReplaysConvolutionStatesOfBothLayoutsAmongTheOtherKinds) {
  ASSERT_TRUE(plane_->replay_state_updates != nullptr) << "the plane library has no replay_state_updates";
  KernelTestDevice device{*plane_};
  constexpr int kCapacity = 7;
  GatedDeltaNetState gdn{device, {4, 2, 100, 132, kCapacity}, 31};
  IndexerState index{device, {7, 128, kCapacity, 2, kQwen38IndexKeyRatio, 21, 2}, 32};
  ConvolutionState channel_first_bf16{device, ConvStateLayout::kChannelFirst, {5, 3, kCapacity, 2}, 33};
  ConvolutionState history_first_bf16{device, ConvStateLayout::kHistoryFirst, {5, 3, kCapacity, 2}, 34};
  ConvolutionState history_first_fp32{device, ConvStateLayout::kHistoryFirst, {300, 4, kCapacity, 4}, 35};
  ConvolutionState channel_first_fp32{device, ConvStateLayout::kChannelFirst, {300, 4, kCapacity, 4}, 36};
  ConvolutionState history_longer_than_channels{device, ConvStateLayout::kHistoryFirst, {3, 5, kCapacity, 4}, 37};
  const std::vector<ReplayedState*> states = {&history_first_bf16, &gdn, &channel_first_fp32,
                                              &history_first_fp32, &index, &channel_first_bf16,
                                              &history_longer_than_channels};
  for (int kept = 1; kept <= kCapacity; ++kept) {
    SCOPED_TRACE("kept " + std::to_string(kept));
    ReplayAndExpect(*plane_, device, states, kept);
  }
}

namespace {

// The value of an FP8 E4M3 code that is not NaN: a sign, four exponent bits (bias 7) and three mantissa bits, where exponent 0
// denotes a subnormal number. Every such value is exact in a float.
float Fp8E4M3Value(uint8_t code) {
  const int exponent = (code >> 3) & 0xF;
  const int mantissa = code & 0x7;
  const float magnitude = exponent == 0 ? std::ldexp(static_cast<float>(mantissa), -9)
                                        : std::ldexp(static_cast<float>(8 + mantissa), exponent - 10);
  return (code & 0x80) != 0 ? -magnitude : magnitude;
}

using FoldState = std::array<int64_t, StagedPleRig::kStateWords>;
using FoldContext = std::array<int64_t, 2>;

// What the preparation of the rows of a fold step leaves in device memory, read back. Every buffer is filled with 0xAB first, so
// that a word the preparation does not write shows.
struct FoldPrepOutputs {
  std::vector<int64_t> input_ids;
  std::vector<int64_t> position_ids;
  std::vector<int64_t> contexts;  // for every row: the token before it, then its own
  std::vector<int64_t> past_pos;
  std::vector<int64_t> state;
  std::vector<uint16_t> ple;  // bfloat16 bits
  int32_t committed{};
};

// A PLE table in which a row depends on the token and on the two tokens before it, as in the models, so that the preparation of
// a fold step shows whether every row sees the right n-gram context. It has four heads of eight values: heads 0 and 1 hash the
// token and the one before it, heads 2 and 3 also hash the one before that. The heads index 61, 53, 47 and 59 rows of a table
// that is split into two shards at row 100, so that head 1 reads from both. A head selects its row as the plane does: the token
// and the contexts times the multipliers, xor-ed, modulo the vocabulary of the head. The rows are FP8 codes, or INT8 values for a
// negative scale. The table is addressed by the GPU or, when `staged`, kept in host memory for the module to gather from.
class HashedPleTable {
 public:
  static constexpr size_t kHeads = 4;
  static constexpr size_t kHeadDim = 8;
  static constexpr size_t kRowValues = kHeads * kHeadDim;
  static constexpr size_t kTableRows = 220;       // the vocabularies of the heads, one after the other
  static constexpr size_t kFirstShardRows = 100;  // the rows after these are in the second shard
  static constexpr size_t kMaxRows = StagedPleRig::kMaxFoldRows;
  static constexpr int64_t kEos = 99;

  HashedPleTable(Qwen38Plane& plane, KernelTestDevice& device, float scale, bool staged)
      : plane_{plane}, device_{device}, staged_{staged}, table_(kTableRows * kHeadDim) {
    WordStream words{12345};
    // Clearing the lowest bit of a code keeps it away from the NaNs of E4M3 (0x7F and 0xFF).
    for (uint8_t& code : table_) code = static_cast<uint8_t>((words.Next() >> 24) & 0xFEu);

    params_.n_heads = static_cast<int>(kHeads);
    params_.head_dim = static_cast<int>(kHeadDim);
    params_.heads_per_ngram = 2;
    params_.n_shards = 2;
    // Large multipliers, so that the hash is often negative as a signed word.
    params_.mult[0] = static_cast<long long>(0x9E3779B97F4A7C15ull);
    params_.mult[1] = static_cast<long long>(0xC2B2AE3D27D4EB4Full);
    params_.mult[2] = static_cast<long long>(0x165667B19E3779F9ull);
    const std::array<long long, kHeads> vocabularies = {61, 53, 47, 59};
    long long offset = 0;
    for (size_t head = 0; head < kHeads; ++head) {
      params_.vocab[head] = vocabularies[head];
      params_.offset[head] = offset;
      offset += vocabularies[head];
    }
    params_.shard_end[0] = static_cast<long long>(kFirstShardRows);
    params_.shard_end[1] = static_cast<long long>(kTableRows);
    const size_t first_shard_bytes = kFirstShardRows * kHeadDim;
    if (staged_) {
      params_.shard_ptr[0] = table_.data();
      params_.shard_ptr[1] = table_.data() + first_shard_bytes;
    } else {
      first_shard_ = device.UploadBytes(table_.data(), first_shard_bytes);
      second_shard_ = device.UploadBytes(table_.data() + first_shard_bytes, table_.size() - first_shard_bytes);
      params_.shard_ptr[0] = first_shard_.data();
      params_.shard_ptr[1] = second_shard_.data();
    }
    params_.scale = scale;
    params_.pad_ = staged_ ? 1 : 0;
    params_.eos = kEos;
    params_.capacity = static_cast<long long>(StagedPleRig::kPositions);
    device_params_ = device.UploadBytes(&params_, sizeof(params_));
    if (staged_) plane_.set_ple_host_params(&params_);

    state_ = device.Filled(StagedPleRig::kStateWords * sizeof(int64_t), 0);
    context_in_ = device.Filled(sizeof(FoldContext), 0);
    input_ids_ = device.Filled(kMaxRows * sizeof(int64_t), 0);
    position_ids_ = device.Filled(kMaxRows * sizeof(int64_t), 0);
    contexts_ = device.Filled(2 * kMaxRows * sizeof(int64_t), 0);
    ple_ = device.Filled(kMaxRows * kRowValues * sizeof(uint16_t), 0);
    past_pos_ = device.Filled(StagedPleRig::kPositions * sizeof(int64_t), 0);
    committed_ = device.Filled(sizeof(int32_t), 0);
    spare_committed_ = device.Filled(sizeof(int32_t), 0);
  }

  ~HashedPleTable() {
    if (staged_) plane_.unset_ple_host_params(&params_);
  }

  HashedPleTable(const HashedPleTable&) = delete;
  HashedPleTable& operator=(const HashedPleTable&) = delete;

  // The PLE row of `token` after `previous` and `before_previous`, as bfloat16 bits: what the host computes from the table.
  std::vector<uint16_t> ExpectedRow(int64_t token, int64_t previous, int64_t before_previous) const {
    // The positions behind an end of sequence token are masked to it.
    const int64_t before = previous == kEos ? kEos : before_previous;
    std::vector<uint16_t> values;
    for (int head = 0; head < static_cast<int>(kHeads); ++head) {
      uint64_t mixed = static_cast<uint64_t>(token) * static_cast<uint64_t>(params_.mult[0]);
      mixed ^= static_cast<uint64_t>(previous) * static_cast<uint64_t>(params_.mult[1]);
      if (head >= params_.heads_per_ngram) mixed ^= static_cast<uint64_t>(before) * static_cast<uint64_t>(params_.mult[2]);
      const long long vocabulary = params_.vocab[head];
      long long remainder = static_cast<long long>(mixed) % vocabulary;
      if (remainder < 0) remainder += vocabulary;
      const size_t row = static_cast<size_t>(remainder + params_.offset[head]);
      for (size_t component = 0; component < kHeadDim; ++component) {
        values.push_back(Dequantized(table_[row * kHeadDim + component]));
      }
    }
    return values;
  }

  // What the preparation of the first `rows` rows of a fold step has to leave in the buffers that Reset filled with 0xAB: for
  // every row its graph inputs, its slot of past_pos, its PLE row and its context; the committed length (of row 0) only when
  // the call was given a place for it; bit 0 of state word 5 when a row lies past the table.
  FoldPrepOutputs Expected(const FoldState& state, const FoldContext& context, int rows, bool with_committed = true) const {
    const int64_t filled = static_cast<int64_t>(0xABABABABABABABABull);
    FoldPrepOutputs expected;
    expected.input_ids.assign(kMaxRows, filled);
    expected.position_ids.assign(kMaxRows, filled);
    expected.contexts.assign(2 * kMaxRows, filled);
    expected.past_pos.assign(StagedPleRig::kPositions, filled);
    expected.state.assign(state.begin(), state.end());
    expected.ple.assign(kMaxRows * kRowValues, 0xABAB);
    expected.committed = static_cast<int32_t>(0xABABABABu);
    std::vector<int64_t> tokens;
    for (size_t r = 0; r < static_cast<size_t>(rows); ++r) {
      const int64_t token = state[StagedPleRig::kFoldTokenWords[r]];
      const int64_t position = state[1] + static_cast<int64_t>(r);
      const int64_t previous = r == 0 ? context[1] : tokens[r - 1];
      const int64_t before_previous = r == 0 ? context[0] : (r == 1 ? context[1] : tokens[r - 2]);
      tokens.push_back(token);
      expected.input_ids[r] = token;
      expected.position_ids[r] = position;
      expected.contexts[2 * r] = previous;
      expected.contexts[2 * r + 1] = token;
      const std::vector<uint16_t> row = ExpectedRow(token, previous, before_previous);
      std::copy(row.begin(), row.end(), expected.ple.begin() + static_cast<std::ptrdiff_t>(r * kRowValues));
      if (position >= 0 && position < static_cast<int64_t>(StagedPleRig::kPositions)) {
        expected.past_pos[static_cast<size_t>(position)] = position;
      } else {
        expected.state[5] |= 1;
      }
      if (r == 0 && with_committed) expected.committed = static_cast<int32_t>(position);
    }
    return expected;
  }

  // The rows prepared one call at a time, as the engine prepared them before the fold step had a call of its own: row r
  // takes its token from its state word, runs at the position word plus r, and starts from the context row r - 1 left.
  FoldPrepOutputs PrepEveryRowSeparately(const FoldState& state, const FoldContext& context, int rows) {
    Reset(state, context);
    for (size_t r = 0; r < static_cast<size_t>(rows); ++r) {
      const uint8_t* token = state_.data() + StagedPleRig::kFoldTokenWords[r] * sizeof(int64_t);
      const uint8_t* context_in = r == 0 ? context_in_.data() : contexts_.data() + 2 * (r - 1) * sizeof(int64_t);
      EXPECT_EQ(Prepare(token, static_cast<int>(r), context_in, contexts_.data() + 2 * r * sizeof(int64_t),
                        input_ids_.data() + r * sizeof(int64_t), position_ids_.data() + r * sizeof(int64_t),
                        r == 0 ? committed_.data() : spare_committed_.data(),
                        ple_.data() + r * kRowValues * sizeof(uint16_t)),
                0)
          << "row " << r;
    }
    return Read();
  }

  // All the rows prepared by the one call that a negative pos_add asks for.
  FoldPrepOutputs PrepInOneCall(const FoldState& state, const FoldContext& context, int rows, bool with_committed = true) {
    Reset(state, context);
    EXPECT_EQ(Prepare(state_.data(), -rows, context_in_.data(), contexts_.data(), input_ids_.data(), position_ids_.data(),
                      with_committed ? committed_.data() : nullptr, ple_.data()),
              0);
    return Read();
  }

  // prep called with `pos_add`, whatever it asks for: the error it returns and what it left in the buffers that Reset filled
  // with 0xAB (which is Expected() of no rows when it wrote nothing).
  std::pair<int, FoldPrepOutputs> PrepWithPosAdd(const FoldState& state, const FoldContext& context, int pos_add) {
    Reset(state, context);
    const int error = Prepare(state_.data(), pos_add, context_in_.data(), contexts_.data(), input_ids_.data(),
                              position_ids_.data(), committed_.data(), ple_.data());
    return {error, Read()};
  }

  static constexpr size_t kCtlWords = QWEN38_PLANE_CTL_WORDS_PER_SLOT;
  static constexpr size_t kCtxWords = QWEN38_PLANE_CTX_WORDS_PER_SLOT;  // of the contexts of one slot, whatever the rows
  static_assert(kCtxWords == 2 * kMaxRows, "a slot has room for the contexts of the most rows a step prepares");
  static constexpr int64_t kSpareWord = 0x5A5A;  // in the control words past the 14 that a fold step uses

  // The first `rows` rows of the fold step of every slot, prepared by one call of prep_batched (`states[b]` are the control
  // words of slot b, `contexts[b]` its n-gram context), read back slot by slot in the layout of Expected(): the rows past `rows`
  // hold the fill value. A word of the control words that the call has no reason to write is checked to be left alone. The
  // contexts of slot b are read at word b * kCtxWords of the buffer, where the rest of the slot holds the fill value too;
  // `prefixes`, when given, receives the whole buffer of batch * kCtxWords words as the call left it.
  std::vector<FoldPrepOutputs> PrepBatched(const std::vector<FoldState>& states, const std::vector<FoldContext>& contexts,
                                           int rows, bool with_committed = true, std::vector<int64_t>* prefixes = nullptr) {
    const size_t batch = states.size();
    const size_t n = static_cast<size_t>(rows);
    std::vector<int64_t> control(batch * kCtlWords, kSpareWord);
    std::vector<int64_t> context_in(batch * 2);
    for (size_t b = 0; b < batch; ++b) {
      std::copy(states[b].begin(), states[b].end(), control.begin() + static_cast<std::ptrdiff_t>(b * kCtlWords));
      std::copy(contexts[b].begin(), contexts[b].end(), context_in.begin() + static_cast<std::ptrdiff_t>(b * 2));
    }
    const PlaneBuffer state = device_.Upload(control);
    const PlaneBuffer ctx_in = device_.Upload(context_in);
    const PlaneBuffer ctx_out = device_.Filled(batch * kCtxWords * sizeof(int64_t), 0xAB);
    const PlaneBuffer input_ids = device_.Filled(batch * n * sizeof(int64_t), 0xAB);
    const PlaneBuffer position_ids = device_.Filled(batch * n * sizeof(int64_t), 0xAB);
    const PlaneBuffer committed = device_.Filled(batch * sizeof(int32_t), 0xAB);
    const PlaneBuffer past_pos = device_.Filled(batch * StagedPleRig::kPositions * sizeof(int64_t), 0xAB);
    const PlaneBuffer ple = device_.Filled(batch * n * kRowValues * sizeof(uint16_t), 0xAB);
    EXPECT_EQ(plane_.prep_batched(device_.stream(), device_params_.data(), rows, static_cast<int>(batch), ctx_in.data(),
                                  ctx_out.data(), input_ids.data(), position_ids.data(),
                                  with_committed ? committed.data() : nullptr, past_pos.data(), ple.data(), state.data()),
              0);

    const auto contexts_out = device_.Download<int64_t>(ctx_out, batch * kCtxWords);
    const auto ids = device_.Download<int64_t>(input_ids, batch * n);
    const auto positions = device_.Download<int64_t>(position_ids, batch * n);
    const auto committed_out = device_.Download<int32_t>(committed, batch);
    const auto past = device_.Download<int64_t>(past_pos, batch * StagedPleRig::kPositions);
    const auto ple_out = device_.Download<uint16_t>(ple, batch * n * kRowValues);
    const auto control_out = device_.Download<int64_t>(state, batch * kCtlWords);
    if (prefixes != nullptr) *prefixes = contexts_out;

    const int64_t filled = static_cast<int64_t>(0xABABABABABABABABull);
    std::vector<FoldPrepOutputs> slots(batch);
    for (size_t b = 0; b < batch; ++b) {
      FoldPrepOutputs& slot = slots[b];
      slot.input_ids.assign(kMaxRows, filled);
      slot.position_ids.assign(kMaxRows, filled);
      slot.contexts.assign(2 * kMaxRows, filled);
      slot.ple.assign(kMaxRows * kRowValues, 0xABAB);
      std::copy_n(ids.begin() + static_cast<std::ptrdiff_t>(b * n), n, slot.input_ids.begin());
      std::copy_n(positions.begin() + static_cast<std::ptrdiff_t>(b * n), n, slot.position_ids.begin());
      std::copy_n(contexts_out.begin() + static_cast<std::ptrdiff_t>(b * kCtxWords), kCtxWords, slot.contexts.begin());
      std::copy_n(ple_out.begin() + static_cast<std::ptrdiff_t>(b * n * kRowValues), n * kRowValues, slot.ple.begin());
      slot.past_pos.assign(past.begin() + static_cast<std::ptrdiff_t>(b * StagedPleRig::kPositions),
                           past.begin() + static_cast<std::ptrdiff_t>((b + 1) * StagedPleRig::kPositions));
      slot.state.assign(control_out.begin() + static_cast<std::ptrdiff_t>(b * kCtlWords),
                        control_out.begin() + static_cast<std::ptrdiff_t>(b * kCtlWords + StagedPleRig::kStateWords));
      for (size_t word = StagedPleRig::kStateWords; word < kCtlWords; ++word) {
        EXPECT_EQ(control_out[b * kCtlWords + word], kSpareWord) << "control word " << word << " of slot " << b;
      }
      slot.committed = with_committed ? committed_out[b] : static_cast<int32_t>(0xABABABABu);
    }
    return slots;
  }

  // What prep_batched has to leave for a muted slot of a table the GPU addresses: idle inputs (token 0 at position r, eos
  // contexts, a zero PLE row), the committed length 0, and the control words as they were.
  FoldPrepOutputs ExpectedIdle(const FoldState& state, int rows, bool with_committed = true) const {
    FoldPrepOutputs expected = Expected(state, FoldContext{{0, 0}}, 0, with_committed);
    for (size_t r = 0; r < static_cast<size_t>(rows); ++r) {
      expected.input_ids[r] = 0;
      expected.position_ids[r] = static_cast<int64_t>(r);
      expected.contexts[2 * r] = kEos;
      expected.contexts[2 * r + 1] = 0;
      std::fill_n(expected.ple.begin() + static_cast<std::ptrdiff_t>(r * kRowValues), kRowValues, uint16_t{0});
      expected.past_pos[r] = static_cast<int64_t>(r);
    }
    if (with_committed) expected.committed = 0;
    return expected;
  }

  // The outcome of prep_batched for arguments that are not valid: the error it returns and whether it wrote anything. Every
  // pointer is valid but the one named by `null_argument`.
  struct RawBatchedPrep {
    int error;
    bool wrote;
  };
  RawBatchedPrep PrepBatchedRaw(int rows, int batch, const std::string& null_argument = std::string{}) {
    constexpr size_t kSlots = QWEN38_PLANE_MAX_PREP_BATCH;
    const PlaneBuffer state = device_.Upload(std::vector<int64_t>(kSlots * kCtlWords, 3));
    const PlaneBuffer ctx_in = device_.Filled(kSlots * 2 * sizeof(int64_t), 0);
    const PlaneBuffer ctx_out = device_.Filled(kSlots * kCtxWords * sizeof(int64_t), 0xAB);
    const PlaneBuffer input_ids = device_.Filled(kSlots * kMaxRows * sizeof(int64_t), 0xAB);
    const PlaneBuffer position_ids = device_.Filled(kSlots * kMaxRows * sizeof(int64_t), 0xAB);
    const PlaneBuffer committed = device_.Filled(kSlots * sizeof(int32_t), 0xAB);
    const PlaneBuffer past_pos = device_.Filled(kSlots * StagedPleRig::kPositions * sizeof(int64_t), 0xAB);
    const PlaneBuffer ple = device_.Filled(kSlots * kMaxRows * kRowValues * sizeof(uint16_t), 0xAB);
    const auto pointer = [&](const char* name, const PlaneBuffer& buffer) -> void* {
      return null_argument == name ? nullptr : buffer.data();
    };
    const int error = plane_.prep_batched(
        device_.stream(), null_argument == "params" ? nullptr : device_params_.data(), rows, batch,
        pointer("ctx_in", ctx_in), pointer("ctx_out", ctx_out), pointer("input_ids", input_ids),
        pointer("position_ids", position_ids), pointer("committed", committed), pointer("past_pos", past_pos),
        pointer("ple", ple), pointer("state", state));
    const auto untouched = [&](const PlaneBuffer& buffer, size_t bytes) {
      const auto values = device_.Download<uint8_t>(buffer, bytes);
      return std::all_of(values.begin(), values.end(), [](uint8_t value) { return value == 0xAB; });
    };
    const bool wrote = !untouched(ctx_out, ctx_out.size()) || !untouched(input_ids, input_ids.size()) ||
                       !untouched(position_ids, position_ids.size()) || !untouched(committed, committed.size()) ||
                       !untouched(past_pos, past_pos.size()) || !untouched(ple, ple.size());
    return {error, wrote};
  }

 private:
  uint16_t Dequantized(uint8_t code) const {
    if (params_.scale < 0.0f) return Bf16Bits(static_cast<float>(static_cast<int8_t>(code)) * -params_.scale);
    return Bf16Bits(Fp8E4M3Value(code) * params_.scale);
  }

  void Reset(const FoldState& state, const FoldContext& context) {
    for (const PlaneBuffer* output : {&input_ids_, &position_ids_, &contexts_, &ple_, &past_pos_, &committed_, &spare_committed_}) {
      device_.Fill(*output, 0xAB);
    }
    plane_.Check(plane_.memcpy_h2d(state_.data(), state.data(), sizeof(state), device_.stream()), "memcpy_h2d");
    plane_.Check(plane_.memcpy_h2d(context_in_.data(), context.data(), sizeof(context), device_.stream()), "memcpy_h2d");
  }

  int Prepare(const void* token, int pos_add, const void* context_in, void* context_out, void* input_ids, void* position_ids,
              void* committed, void* ple) {
    return plane_.prep(device_.stream(), device_params_.data(), token, state_.data() + sizeof(int64_t), pos_add, context_in,
                       context_out, input_ids, position_ids, committed, past_pos_.data(), ple, state_.data());
  }

  FoldPrepOutputs Read() {
    FoldPrepOutputs outputs;
    outputs.input_ids = device_.Download<int64_t>(input_ids_, kMaxRows);
    outputs.position_ids = device_.Download<int64_t>(position_ids_, kMaxRows);
    outputs.contexts = device_.Download<int64_t>(contexts_, 2 * kMaxRows);
    outputs.past_pos = device_.Download<int64_t>(past_pos_, StagedPleRig::kPositions);
    outputs.state = device_.Download<int64_t>(state_, StagedPleRig::kStateWords);
    outputs.ple = device_.Download<uint16_t>(ple_, kMaxRows * kRowValues);
    outputs.committed = device_.Download<int32_t>(committed_, 1)[0];
    return outputs;
  }

  Qwen38Plane& plane_;
  KernelTestDevice& device_;
  const bool staged_;
  std::vector<uint8_t> table_;
  Qwen38PleParams params_{};
  PlaneBuffer first_shard_, second_shard_, device_params_;
  PlaneBuffer state_, context_in_, input_ids_, position_ids_, contexts_, ple_, past_pos_, committed_, spare_committed_;
};

void ExpectSameFoldPrepOutputs(const FoldPrepOutputs& expected, const FoldPrepOutputs& actual, const std::string& what) {
  ExpectSameElements(expected.input_ids, actual.input_ids, what + ": input_ids");
  ExpectSameElements(expected.position_ids, actual.position_ids, what + ": position_ids");
  ExpectSameElements(expected.contexts, actual.contexts, what + ": the contexts");
  ExpectSameElements(expected.past_pos, actual.past_pos, what + ": past_pos");
  ExpectSameElements(expected.state, actual.state, what + ": the state");
  ExpectSameElements(expected.ple, actual.ple, what + ": the PLE rows");
  EXPECT_EQ(expected.committed, actual.committed) << what << ": the committed length";
}

// A fold step to prepare: the control words (word 0, 2 and 8..13 hold the tokens of the rows, word 1 the position of row 0, word
// 5 status bits; the other words are no tokens) and the two tokens before row 0, the older one first.
struct FoldScenario {
  const char* name;
  FoldState state;
  FoldContext context;
};

const std::vector<FoldScenario>& FoldScenarios() {
  constexpr int64_t kEos = HashedPleTable::kEos;
  static const std::vector<FoldScenario> scenarios = {
      {"ordinary", {{7, 5, 2, 701, 702, 0x10, 703, 704, 1, 6, 9, 4, 3, 10}}, {{11, 13}}},
      {"the context ends at the end of sequence", {{7, 5, 2, 701, 702, 0x10, 703, 704, 1, 6, 9, 4, 3, 10}}, {{5, kEos}}},
      {"drafts that are the end of sequence", {{7, 5, kEos, 701, 702, 0x10, 703, 704, 1, 6, kEos, 4, 3, kEos}}, {{11, 13}}},
      {"one token over and over", {{4, 3, 4, 701, 702, 0x10, 703, 704, 4, 4, 4, 4, 4, 4}}, {{4, 4}}},
      {"positions past the table", {{7, 14, 2, 701, 702, 0x10, 703, 704, 1, 6, 9, 4, 3, 10}}, {{11, 13}}},
      {"token ids of a large vocabulary",
       {{151643, 7, 248320, 701, 702, 0x10, 703, 704, 262143, 151935, 100000, 9, 3, 10}},
       {{150000, 152000}}},
      {"zeros", {{0, 0, 0, 701, 702, 0x10, 703, 704, 0, 0, 0, 0, 0, 0}}, {{0, 0}}},
  };
  return scenarios;
}

class Qwen38PlaneFoldPrepTest : public PlaneKernelFixture<::testing::Test> {
 protected:
  // The fused prep (k_prep_fold, or the per-row staging for a staged table) of 2 to 8 rows against the prep of the same rows one
  // call at a time and against the PLE rows, graph inputs and contexts that the host computes from the table.
  void ExpectTheFusedPrepToMatchEveryRowSeparately(bool staged, float scale) {
    ASSERT_NE(plane_->capabilities & QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH, 0ull);
    KernelTestDevice device{*plane_};
    HashedPleTable table{*plane_, device, scale, staged};
    // The rows of the table have to depend on each of the three tokens, or a wrong context could go unnoticed.
    ASSERT_NE(table.ExpectedRow(7, 11, 13), table.ExpectedRow(8, 11, 13)) << "the token";
    ASSERT_NE(table.ExpectedRow(7, 11, 13), table.ExpectedRow(7, 12, 13)) << "the previous token";
    ASSERT_NE(table.ExpectedRow(7, 11, 13), table.ExpectedRow(7, 11, 14)) << "the token before the previous one";
    ASSERT_EQ(table.ExpectedRow(7, HashedPleTable::kEos, 13), table.ExpectedRow(7, HashedPleTable::kEos, 14))
        << "the positions behind an end of sequence are masked";
    for (const FoldScenario& scenario : FoldScenarios()) {
      for (int rows = 2; rows <= static_cast<int>(HashedPleTable::kMaxRows); ++rows) {
        const std::string what = std::string{staged ? "staged table" : "device table"} + ", " +
                                 (scale < 0.0f ? "INT8" : "FP8") + ", " + scenario.name + ", " + std::to_string(rows) + " rows";
        const FoldPrepOutputs expected = table.Expected(scenario.state, scenario.context, rows);
        const FoldPrepOutputs separate = table.PrepEveryRowSeparately(scenario.state, scenario.context, rows);
        const FoldPrepOutputs fused = table.PrepInOneCall(scenario.state, scenario.context, rows);
        ExpectSameFoldPrepOutputs(expected, separate, what + ", one row at a time against the host");
        ExpectSameFoldPrepOutputs(separate, fused, what + ", in one call against one row at a time");
        ExpectSameFoldPrepOutputs(expected, fused, what + ", in one call against the host");
      }
    }
  }
};

TEST_F(Qwen38PlaneFoldPrepTest, FusedPrepMatchesEveryRowSeparatelyForAnFp8DeviceTable) {
  ExpectTheFusedPrepToMatchEveryRowSeparately(/*staged=*/false, 0.3f);
}

TEST_F(Qwen38PlaneFoldPrepTest, FusedPrepMatchesEveryRowSeparatelyForAnFp8StagedTable) {
  ExpectTheFusedPrepToMatchEveryRowSeparately(/*staged=*/true, 0.3f);
}

TEST_F(Qwen38PlaneFoldPrepTest, FusedPrepMatchesEveryRowSeparatelyForAnInt8DeviceTable) {
  if ((plane_->capabilities & QWEN38_PLANE_CAP_INT8_PLE) == 0) GTEST_SKIP() << "the plane library has no INT8 PLE";
  ExpectTheFusedPrepToMatchEveryRowSeparately(/*staged=*/false, -0.0123f);
}

TEST_F(Qwen38PlaneFoldPrepTest, FusedPrepMatchesEveryRowSeparatelyForAnInt8StagedTable) {
  if ((plane_->capabilities & QWEN38_PLANE_CAP_INT8_PLE) == 0) GTEST_SKIP() << "the plane library has no INT8 PLE";
  ExpectTheFusedPrepToMatchEveryRowSeparately(/*staged=*/true, -0.0123f);
}

// The prep of one row skips the committed length when it is given no place for it, and so does the prep of a whole step.
TEST_F(Qwen38PlaneFoldPrepTest, FusedPrepWithoutACommittedLengthWritesEverythingElse) {
  ASSERT_NE(plane_->capabilities & QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH, 0ull);
  const FoldScenario& scenario = FoldScenarios().front();
  for (bool staged : {false, true}) {
    KernelTestDevice device{*plane_};
    HashedPleTable table{*plane_, device, 0.3f, staged};
    for (int rows : {2, 5, 8}) {
      ExpectSameFoldPrepOutputs(table.Expected(scenario.state, scenario.context, rows, /*with_committed=*/false),
                                table.PrepInOneCall(scenario.state, scenario.context, rows, /*with_committed=*/false),
                                std::string{staged ? "staged table, " : "device table, "} + std::to_string(rows) + " rows");
    }
  }
}

// The slots of a batch to prepare: slot b takes the scenario b + `shift`, and its tokens, position and context are told apart
// from those of the other slots by an offset of b (an end of sequence token stays one), so that a slot that reads the words,
// the context or the output rows of another one is noticed. No slot is muted.
struct BatchedFoldSlots {
  std::vector<FoldState> states;
  std::vector<FoldContext> contexts;
};

BatchedFoldSlots MakeBatchedFoldSlots(size_t batch, size_t shift) {
  constexpr int64_t kEos = HashedPleTable::kEos;
  const std::vector<FoldScenario>& scenarios = FoldScenarios();
  BatchedFoldSlots slots;
  for (size_t b = 0; b < batch; ++b) {
    const FoldScenario& scenario = scenarios[(b + shift) % scenarios.size()];
    FoldState state = scenario.state;
    FoldContext context = scenario.context;
    const int64_t offset = static_cast<int64_t>(b);
    for (const size_t word : StagedPleRig::kFoldTokenWords) {
      if (state[word] != kEos) state[word] += 17 * offset;
    }
    state[1] += offset;
    state[3] = 0;  // not muted
    for (int64_t& token : context) {
      if (token != kEos) token += offset;
    }
    slots.states.push_back(state);
    slots.contexts.push_back(context);
  }
  return slots;
}

// The batched prep (qwen38_plane_prep_batched, QWEN38_PLANE_CAP_BATCHED_PREP) of every slot of a batch against the prep of the
// same slot on its own, row by row as the engine prepared a step before the entry point existed, and against the PLE rows,
// graph inputs and contexts that the host computes from the table.
class Qwen38PlaneBatchedPrepKernelTest : public PlaneKernelFixture<::testing::Test> {
 protected:
  void SetUp() override {
    PlaneKernelFixture<::testing::Test>::SetUp();
    if (IsSkipped()) return;
    ASSERT_TRUE(plane_->prep_batched != nullptr) << "the plane library has no qwen38_plane_prep_batched";
    ASSERT_NE(plane_->capabilities & QWEN38_PLANE_CAP_BATCHED_PREP, 0ull);
  }

  static std::string Describe(bool staged, float scale) {
    return std::string{staged ? "staged table, " : "device table, "} + (scale < 0.0f ? "INT8" : "FP8");
  }

  void ExpectEverySlotToMatchItsOwnPrep(bool staged, float scale) {
    KernelTestDevice device{*plane_};
    HashedPleTable table{*plane_, device, scale, staged};
    for (int rows = 1; rows <= static_cast<int>(HashedPleTable::kMaxRows); ++rows) {
      for (size_t batch = 1; batch <= QWEN38_PLANE_MAX_PREP_BATCH; ++batch) {
        const BatchedFoldSlots slots = MakeBatchedFoldSlots(batch, static_cast<size_t>(rows));
        const std::vector<FoldPrepOutputs> prepared = table.PrepBatched(slots.states, slots.contexts, rows);
        ASSERT_EQ(prepared.size(), batch);
        for (size_t b = 0; b < batch; ++b) {
          const std::string what = Describe(staged, scale) + ", " + std::to_string(rows) + " rows of slot " +
                                   std::to_string(b) + " of " + std::to_string(batch);
          ExpectSameFoldPrepOutputs(table.Expected(slots.states[b], slots.contexts[b], rows), prepared[b],
                                    what + ", against the host");
          ExpectSameFoldPrepOutputs(table.PrepEveryRowSeparately(slots.states[b], slots.contexts[b], rows), prepared[b],
                                    what + ", against one row at a time");
        }
      }
    }
  }
};

TEST_F(Qwen38PlaneBatchedPrepKernelTest, EverySlotMatchesItsOwnPrepForAnFp8DeviceTable) {
  ExpectEverySlotToMatchItsOwnPrep(/*staged=*/false, 0.3f);
}

TEST_F(Qwen38PlaneBatchedPrepKernelTest, EverySlotMatchesItsOwnPrepForAnFp8StagedTable) {
  ExpectEverySlotToMatchItsOwnPrep(/*staged=*/true, 0.3f);
}

TEST_F(Qwen38PlaneBatchedPrepKernelTest, EverySlotMatchesItsOwnPrepForAnInt8DeviceTable) {
  if ((plane_->capabilities & QWEN38_PLANE_CAP_INT8_PLE) == 0) GTEST_SKIP() << "the plane library has no INT8 PLE";
  ExpectEverySlotToMatchItsOwnPrep(/*staged=*/false, -0.0123f);
}

TEST_F(Qwen38PlaneBatchedPrepKernelTest, EverySlotMatchesItsOwnPrepForAnInt8StagedTable) {
  if ((plane_->capabilities & QWEN38_PLANE_CAP_INT8_PLE) == 0) GTEST_SKIP() << "the plane library has no INT8 PLE";
  ExpectEverySlotToMatchItsOwnPrep(/*staged=*/true, -0.0123f);
}

// A slot whose control word 3 is nonzero takes no part in the step: a table the GPU addresses prepares idle inputs for it and
// leaves its control words alone, and the slots around it are prepared as if it were live. A staged table prepares every slot
// from its control words, muted or not (the caller ignores the rows of a muted slot either way).
TEST_F(Qwen38PlaneBatchedPrepKernelTest, AMutedSlotIsPreparedAsAnIdleSlotWithoutDisturbingItsNeighbours) {
  const std::vector<std::vector<size_t>> muted_slots = {{1}, {0, 3}, {2, 3, 4}, {0, 1, 2, 3, 4, 5, 6, 7}};
  for (const bool staged : {false, true}) {
    KernelTestDevice device{*plane_};
    HashedPleTable table{*plane_, device, 0.3f, staged};
    for (const int rows : {1, 2, 5, 8}) {
      for (const std::vector<size_t>& muted : muted_slots) {
        const size_t batch = muted.back() == 7 ? 8 : muted.back() + 2;  // at least one live slot after the last muted one
        BatchedFoldSlots slots = MakeBatchedFoldSlots(batch, 1);
        std::vector<bool> is_muted(batch, false);
        for (const size_t b : muted) {
          slots.states[b][3] = 1;
          is_muted[b] = true;
        }
        const std::vector<FoldPrepOutputs> prepared = table.PrepBatched(slots.states, slots.contexts, rows);
        for (size_t b = 0; b < batch; ++b) {
          const std::string what = std::string{staged ? "staged table, " : "device table, "} + std::to_string(rows) +
                                   " rows, slot " + std::to_string(b) + (is_muted[b] ? " (muted)" : "") + " of " +
                                   std::to_string(batch);
          const FoldPrepOutputs expected = is_muted[b] && !staged
                                               ? table.ExpectedIdle(slots.states[b], rows)
                                               : table.Expected(slots.states[b], slots.contexts[b], rows);
          ExpectSameFoldPrepOutputs(expected, prepared[b], what);
        }
      }
    }
  }
}

// The one-slot prep never reads the mute word: a control word 3 that is not zero is no state of the slot there.
TEST_F(Qwen38PlaneBatchedPrepKernelTest, TheSingleSlotPrepIgnoresTheMuteWord) {
  const FoldScenario& scenario = FoldScenarios().front();
  FoldState state = scenario.state;
  state[3] = 1;
  for (const bool staged : {false, true}) {
    KernelTestDevice device{*plane_};
    HashedPleTable table{*plane_, device, 0.3f, staged};
    for (const int rows : {2, 8}) {
      ExpectSameFoldPrepOutputs(table.Expected(state, scenario.context, rows),
                                table.PrepInOneCall(state, scenario.context, rows),
                                std::string{staged ? "staged table, " : "device table, "} + std::to_string(rows) + " rows");
    }
  }
}

TEST_F(Qwen38PlaneBatchedPrepKernelTest, WithoutACommittedLengthEverythingElseIsWritten) {
  for (const bool staged : {false, true}) {
    KernelTestDevice device{*plane_};
    HashedPleTable table{*plane_, device, 0.3f, staged};
    for (const int rows : {1, 3, 8}) {
      for (const size_t batch : {size_t{1}, size_t{4}, size_t{8}}) {
        const BatchedFoldSlots slots = MakeBatchedFoldSlots(batch, 2);
        const std::vector<FoldPrepOutputs> prepared =
            table.PrepBatched(slots.states, slots.contexts, rows, /*with_committed=*/false);
        for (size_t b = 0; b < batch; ++b) {
          ExpectSameFoldPrepOutputs(table.Expected(slots.states[b], slots.contexts[b], rows, /*with_committed=*/false),
                                    prepared[b],
                                    std::string{staged ? "staged table, " : "device table, "} + std::to_string(rows) +
                                        " rows, slot " + std::to_string(b) + " of " + std::to_string(batch));
        }
      }
    }
  }
}

// A row count or a batch outside the supported ranges, or a missing buffer, is an error that is returned before anything is
// launched, not a launch that runs with whatever the numbers make of it.
TEST_F(Qwen38PlaneBatchedPrepKernelTest, RefusesAnInvalidRowCountBatchOrPointerWithoutWritingAnything) {
  constexpr int kInvalidValue = 1;  // cudaErrorInvalidValue
  for (const bool staged : {false, true}) {
    KernelTestDevice device{*plane_};
    HashedPleTable table{*plane_, device, 0.3f, staged};
    const std::string table_kind = staged ? "staged table" : "device table";

    const auto valid = table.PrepBatchedRaw(QWEN38_PLANE_MAX_PREP_ROWS, QWEN38_PLANE_MAX_PREP_BATCH);
    EXPECT_EQ(valid.error, 0) << table_kind;
    EXPECT_TRUE(valid.wrote) << table_kind;

    for (const int rows : {-8, -1, 0, QWEN38_PLANE_MAX_PREP_ROWS + 1, 16, 0x108}) {
      const auto refused = table.PrepBatchedRaw(rows, 2);
      EXPECT_EQ(refused.error, kInvalidValue) << table_kind << ", rows " << rows;
      EXPECT_FALSE(refused.wrote) << table_kind << ", rows " << rows;
    }
    for (const int batch : {-1, 0, QWEN38_PLANE_MAX_PREP_BATCH + 1, 16, 0x200}) {
      const auto refused = table.PrepBatchedRaw(2, batch);
      EXPECT_EQ(refused.error, kInvalidValue) << table_kind << ", batch " << batch;
      EXPECT_FALSE(refused.wrote) << table_kind << ", batch " << batch;
    }
    for (const char* missing : {"params", "ctx_in", "ctx_out", "input_ids", "position_ids", "past_pos", "ple", "state"}) {
      const auto refused = table.PrepBatchedRaw(2, 2, missing);
      EXPECT_EQ(refused.error, kInvalidValue) << table_kind << ", without " << missing;
      EXPECT_FALSE(refused.wrote) << table_kind << ", without " << missing;
    }
    // The committed lengths are optional.
    const auto without_committed = table.PrepBatchedRaw(2, 2, "committed");
    EXPECT_EQ(without_committed.error, 0) << table_kind;
    EXPECT_TRUE(without_committed.wrote) << table_kind;
  }
}

// The single-slot prep used to receive the batch size packed into pos_add (-(rows | batch << 8)). It no longer reads such a
// value as anything: a pos_add below -8 is refused, so a caller that still packs a batch never gets inputs for the wrong rows.
TEST_F(Qwen38PlaneBatchedPrepKernelTest, TheSingleSlotPrepRefusesABatchPackedIntoPosAdd) {
  constexpr int kInvalidValue = 1;  // cudaErrorInvalidValue
  const FoldScenario& scenario = FoldScenarios().front();
  for (const bool staged : {false, true}) {
    KernelTestDevice device{*plane_};
    HashedPleTable table{*plane_, device, 0.3f, staged};
    for (const int pos_add : {-(8 | (2 << 8)), -(2 | (4 << 8)), -(8 | (8 << 8)), -9, -16, -0x100}) {
      const auto [error, outputs] = table.PrepWithPosAdd(scenario.state, scenario.context, pos_add);
      EXPECT_EQ(error, kInvalidValue) << (staged ? "staged table" : "device table") << ", pos_add " << pos_add;
      ExpectSameFoldPrepOutputs(table.Expected(scenario.state, scenario.context, 0), outputs,
                                std::string{"nothing written for pos_add "} + std::to_string(pos_add));
    }
    // The documented range keeps working.
    const auto [error, outputs] = table.PrepWithPosAdd(scenario.state, scenario.context, -8);
    EXPECT_EQ(error, 0);
    ExpectSameFoldPrepOutputs(table.Expected(scenario.state, scenario.context, 8), outputs, "pos_add -8");
  }
}

// SpeculativePrefixCommit reads the n-gram contexts of the fold rows from one `prefixes` input that spans the slots of the batch:
// it divides the size of the input by the number of slots and reads slot s at s times that many words. The engine binds a view of
// batch * 16 words, 2 for each of the 8 rows of the largest verify step, whatever the number of rows of the step, and its
// per-slot views start at word 16 s as well. So every writer of the contexts has to put slot s at word 16 s: a writer that packs
// the slots for the rows of its own call (2 * rows words each) is read at the wrong place by every slot but the first as soon as
// a step has fewer than 8 rows (the verify widths 3 to 7 of the fold stack; 1 and 2 are the plane's own plain and two-row steps).
class Qwen38PlanePrefixCommitStrideTest : public Qwen38PlaneBatchedPrepKernelTest {
 protected:
  static constexpr size_t kViewWordsPerSlot = 2 * 8;  // what the engine binds for each slot of the batched fold.ctx_a view
  static_assert(kViewWordsPerSlot == QWEN38_PLANE_CTX_WORDS_PER_SLOT, "the stride of the view is the stride of the writers");

  // Every token of every row of every slot is a number of its own (none is the end of sequence), and so is the context of
  // each slot, so that the pair of words of a row can be found in one place of one slot only.
  static BatchedFoldSlots MakeSlotsWithUniquePrefixes(size_t batch) {
    BatchedFoldSlots slots = MakeBatchedFoldSlots(batch, 0);
    for (size_t b = 0; b < batch; ++b) {
      const int64_t base = 1000 * static_cast<int64_t>(b + 1);
      for (size_t r = 0; r < StagedPleRig::kMaxFoldRows; ++r) {
        slots.states[b][StagedPleRig::kFoldTokenWords[r]] = base + 10 * static_cast<int64_t>(r) + 1;
      }
      slots.contexts[b] = {{base + 901, base + 902}};
    }
    return slots;
  }

  // The prefixes of `rows` rows of every slot of `batch`, read the way the op reads them out of the buffer that the writer left:
  // the stride is the size over the slots, and the words that no row of the slot owns still hold the fill value.
  void ExpectEverySlotToLandWhereTheViewReadsIt(bool staged, float scale) {
    KernelTestDevice device{*plane_};
    HashedPleTable table{*plane_, device, scale, staged};
    const int64_t filled = static_cast<int64_t>(0xABABABABABABABABull);
    for (const size_t batch : {size_t{2}, size_t{3}, size_t{5}, size_t{8}}) {
      const BatchedFoldSlots slots = MakeSlotsWithUniquePrefixes(batch);
      for (int rows = 1; rows <= static_cast<int>(HashedPleTable::kMaxRows); ++rows) {
        const std::string what =
            Describe(staged, scale) + ", " + std::to_string(rows) + " rows of " + std::to_string(batch) + " slots";
        std::vector<int64_t> prefixes;
        table.PrepBatched(slots.states, slots.contexts, rows, /*with_committed=*/true, &prefixes);
        ASSERT_EQ(prefixes.size(), batch * kViewWordsPerSlot) << what;
        const size_t stride = prefixes.size() / batch;
        EXPECT_EQ(stride, size_t{16}) << what;

        std::vector<std::pair<int64_t, int64_t>> pairs;
        for (size_t s = 0; s < batch; ++s) {
          for (size_t r = 0; r < HashedPleTable::kMaxRows; ++r) {
            const bool written = r < static_cast<size_t>(rows);
            const int64_t previous = r == 0 ? slots.contexts[s][1] : slots.states[s][StagedPleRig::kFoldTokenWords[r - 1]];
            const int64_t token = slots.states[s][StagedPleRig::kFoldTokenWords[r]];
            EXPECT_EQ(prefixes[s * stride + 2 * r], written ? previous : filled)
                << what << ", the previous token of row " << r << " of slot " << s;
            EXPECT_EQ(prefixes[s * stride + 2 * r + 1], written ? token : filled)
                << what << ", the token of row " << r << " of slot " << s;
            if (written) pairs.emplace_back(previous, token);
          }
        }
        std::sort(pairs.begin(), pairs.end());
        EXPECT_EQ(std::adjacent_find(pairs.begin(), pairs.end()), pairs.end()) << what << ": two rows have the same prefix";
        EXPECT_EQ(pairs.size(), batch * static_cast<size_t>(rows)) << what;
      }
    }
  }
};

TEST_F(Qwen38PlanePrefixCommitStrideTest, EverySlotOfAnFp8DeviceTableLandsWhereTheViewReadsIt) {
  ExpectEverySlotToLandWhereTheViewReadsIt(/*staged=*/false, 0.3f);
}

TEST_F(Qwen38PlanePrefixCommitStrideTest, EverySlotOfAnFp8StagedTableLandsWhereTheViewReadsIt) {
  ExpectEverySlotToLandWhereTheViewReadsIt(/*staged=*/true, 0.3f);
}

TEST_F(Qwen38PlanePrefixCommitStrideTest, EverySlotOfAnInt8DeviceTableLandsWhereTheViewReadsIt) {
  if ((plane_->capabilities & QWEN38_PLANE_CAP_INT8_PLE) == 0) GTEST_SKIP() << "the plane library has no INT8 PLE";
  ExpectEverySlotToLandWhereTheViewReadsIt(/*staged=*/false, -0.0123f);
}

TEST_F(Qwen38PlanePrefixCommitStrideTest, EverySlotOfAnInt8StagedTableLandsWhereTheViewReadsIt) {
  if ((plane_->capabilities & QWEN38_PLANE_CAP_INT8_PLE) == 0) GTEST_SKIP() << "the plane library has no INT8 PLE";
  ExpectEverySlotToLandWhereTheViewReadsIt(/*staged=*/true, -0.0123f);
}

// A muted slot of a table the GPU addresses is prepared as an idle slot (eos contexts) by the same kernel that prepares the live
// slots, at the same place; the slots on both sides of it are not disturbed.
TEST_F(Qwen38PlanePrefixCommitStrideTest, AMutedSlotKeepsItsPlaceAndItsNeighboursKeepTheirs) {
  const int64_t filled = static_cast<int64_t>(0xABABABABABABABABull);
  KernelTestDevice device{*plane_};
  HashedPleTable table{*plane_, device, 0.3f, /*staged=*/false};
  constexpr size_t kBatch = 4;
  constexpr size_t kMuted = 1;
  for (const int rows : {3, 4, 5, 6, 7}) {
    BatchedFoldSlots slots = MakeSlotsWithUniquePrefixes(kBatch);
    slots.states[kMuted][3] = 1;
    std::vector<int64_t> prefixes;
    table.PrepBatched(slots.states, slots.contexts, rows, /*with_committed=*/true, &prefixes);
    ASSERT_EQ(prefixes.size(), kBatch * kViewWordsPerSlot);
    const size_t stride = prefixes.size() / kBatch;
    for (size_t s = 0; s < kBatch; ++s) {
      for (size_t r = 0; r < HashedPleTable::kMaxRows; ++r) {
        const bool written = r < static_cast<size_t>(rows);
        int64_t previous = r == 0 ? slots.contexts[s][1] : slots.states[s][StagedPleRig::kFoldTokenWords[r - 1]];
        int64_t token = slots.states[s][StagedPleRig::kFoldTokenWords[r]];
        if (s == kMuted) {
          previous = HashedPleTable::kEos;
          token = 0;
        }
        EXPECT_EQ(prefixes[s * stride + 2 * r], written ? previous : filled) << rows << " rows, row " << r << " of slot " << s;
        EXPECT_EQ(prefixes[s * stride + 2 * r + 1], written ? token : filled) << rows << " rows, row " << r << " of slot " << s;
      }
    }
  }
}

// A table that is staged on the host is prepared slot by slot: the rows of a slot are gathered on the host and prepared by one
// launch (the plain next-token row too), whatever the batch, and every slot gets exactly the rows its own control words ask for.
class Qwen38PlaneStagedBatchedPrepTest : public Qwen38PlaneStagedPleTest {};

TEST_F(Qwen38PlaneStagedBatchedPrepTest, ABatchOfAStagedTableIsGatheredOnTheHostAndLaunchedOncePerSlot) {
  ASSERT_TRUE(plane_->prep_batched != nullptr) << "the plane library has no qwen38_plane_prep_batched";
  KernelTestDevice device{*plane_};
  HashedPleTable table{*plane_, device, 0.3f, /*staged=*/true};
  for (const int rows : {1, 2, 5, 8}) {
    for (const size_t batch : {size_t{1}, size_t{3}, size_t{8}}) {
      const std::string what = std::to_string(rows) + " rows of " + std::to_string(batch) + " slots";
      const BatchedFoldSlots slots = MakeBatchedFoldSlots(batch, static_cast<size_t>(rows));
      const Qwen38PlaneTestStats before = Stats();
      const std::vector<FoldPrepOutputs> prepared = table.PrepBatched(slots.states, slots.contexts, rows);
      const Qwen38PlaneTestStats after = Stats();
      EXPECT_EQ(after.gather_rows, before.gather_rows + static_cast<long long>(batch) * rows) << what;
      EXPECT_EQ(after.staged_launches, before.staged_launches + static_cast<long long>(batch)) << what;
      ASSERT_EQ(prepared.size(), batch);
      for (size_t b = 0; b < batch; ++b) {
        ExpectSameFoldPrepOutputs(table.Expected(slots.states[b], slots.contexts[b], rows), prepared[b],
                                  what + ", slot " + std::to_string(b));
      }
    }
  }
}

// The staged table is gathered on the host by threads of the plane library when a prefill has 32 tokens or more. They run code of
// the library, so they have to be stopped and joined when the last loader of the module goes (a thread that is still alive could
// return into code the host no longer has), and a loader that follows has to start threads of its own. The host never unloads
// an attached plane library, so these tests look at what the module does when its last loader detaches.
class Qwen38PlaneGatherPoolTest : public Qwen38PlaneStagedPleTest {
 protected:
  static constexpr size_t kPoolPrefillTokens = 32;

  void SetUp() override {
    Qwen38PlaneStagedPleTest::SetUp();
    if (IsSkipped()) return;
    if (!plane_->release_ple_workspace) GTEST_SKIP() << "the plane library has no qwen38_plane_release_ple_workspace";
  }

  // Tokens 0, 1, 2, ...: token t gathers row t % 4 of a rig's table.
  static std::vector<int64_t> Tokens(size_t count) {
    std::vector<int64_t> tokens(count);
    for (size_t i = 0; i < count; ++i) tokens[i] = static_cast<int64_t>(i);
    return tokens;
  }

  static int GatherWorkers() {
    return qwen38_plane::DefaultParallelWorkerCount(std::thread::hardware_concurrency());
  }

  // The threads of the process after the plane did its first staged call and its first teardown, so that the threads CUDA starts
  // for itself belong to the baseline. The table is registered again afterwards.
  int ThreadsAtRest(StagedPleRig& rig) {
    EXPECT_EQ(rig.PleRows(Tokens(2)), 0);
    plane_->release_ple_workspace();
    rig.ResetTable();
    return ProcessThreadCount();
  }
};

TEST_F(Qwen38PlaneGatherPoolTest, AShortPrefillIsGatheredOnTheCallerAndThirtyTwoTokensStartTheWorkers) {
  StagedPleRig rig{*plane_, kAscendingRows};
  const std::vector<int64_t> few = Tokens(kPoolPrefillTokens - 1);
  ASSERT_EQ(rig.PleRows(few), 0);
  EXPECT_EQ(Stats().gather_pool_threads, 0) << "a short prefill started threads";
  EXPECT_EQ(rig.ReadPle(few.size()), rig.ExpectedPle(few));

  const int before = ProcessThreadCount();
  const std::vector<int64_t> many = Tokens(kPoolPrefillTokens);
  ASSERT_EQ(rig.PleRows(many), 0);
  EXPECT_EQ(Stats().gather_pool_threads, GatherWorkers());
  if (before >= 0) {
    EXPECT_GE(ProcessThreadCount(), before + GatherWorkers());
  }
  EXPECT_EQ(rig.ReadPle(many.size()), rig.ExpectedPle(many));
}

TEST_F(Qwen38PlaneGatherPoolTest, TheLastLoaderToDetachStopsAndJoinsTheWorkers) {
  ASSERT_EQ(Stats().attached_loaders, 1) << "a loader of an earlier test is still attached";
  int baseline = -1;
  {
    StagedPleRig rig{*plane_, kAscendingRows};
    baseline = ThreadsAtRest(rig);
    const std::vector<int64_t> tokens = Tokens(64);
    ASSERT_EQ(rig.PleRows(tokens), 0);
    ASSERT_GT(Stats().gather_pool_threads, 0);
    EXPECT_EQ(rig.ReadPle(tokens.size()), rig.ExpectedPle(tokens));
  }
  plane_.reset();
  EXPECT_EQ(Stats().attached_loaders, 0);
  EXPECT_EQ(Stats().gather_pool_threads, 0) << "the module was released with a gather thread alive";
  EXPECT_TRUE(ProcessThreadCountSettlesAtOrBelow(baseline)) << "a gather thread of the module is still running";

  // The library is not unloaded with its last loader; the one that follows shares it. The test's own reference goes too, so
  // nothing else holds the file.
  module_.reset();
  EXPECT_TRUE(IsLibraryMapped(library_path_)) << "the plane library was unloaded with its last loader";
}

// A staged prefill of 32 tokens or more, then the destruction of the loader, then a load of the plane again: every load starts
// workers of its own, gathers its own table and leaves no thread behind.
TEST_F(Qwen38PlaneGatherPoolTest, ALoaderThatFollowsTheLastOneStartsItsOwnWorkers) {
  ASSERT_EQ(Stats().attached_loaders, 1) << "a loader of an earlier test is still attached";
  int baseline = -1;
  for (int load = 0; load < 4; ++load) {
    SCOPED_TRACE(load);
    if (load > 0) {
      plane_ = std::make_unique<Qwen38Plane>(library_path_);
      ASSERT_EQ(plane_->set_device(0), 0);
    }
    {
      // The tables differ: a worker that kept the gather of an earlier load would show.
      StagedPleRig rig{*plane_, load % 2 == 0 ? kAscendingRows : kDescendingRows};
      if (load == 0) baseline = ThreadsAtRest(rig);
      EXPECT_EQ(Stats().gather_pool_threads, 0) << "the workers of the earlier load are still alive";

      const std::vector<int64_t> tokens = Tokens(load == 3 ? StagedPleRig::kMaxTokens : 300);
      ASSERT_EQ(rig.PleRows(tokens), 0);
      EXPECT_EQ(Stats().gather_pool_threads, GatherWorkers());
      EXPECT_EQ(rig.ReadPle(tokens.size()), rig.ExpectedPle(tokens));
    }
    plane_.reset();
    EXPECT_EQ(Stats().gather_pool_threads, 0);
    EXPECT_TRUE(ProcessThreadCountSettlesAtOrBelow(baseline));
  }
}

// The module is shared by every loader of the library file: the first of two to go leaves the workers to the one that stays.
TEST_F(Qwen38PlaneGatherPoolTest, TheWorkersOutliveAllButTheLastLoader) {
  ASSERT_EQ(Stats().attached_loaders, 1) << "a loader of an earlier test is still attached";
  auto second = std::make_unique<Qwen38Plane>(library_path_);
  ASSERT_EQ(second->set_device(0), 0);
  ASSERT_EQ(Stats().attached_loaders, 2);
  int baseline = -1;
  {
    StagedPleRig rig{*second, kAscendingRows};
    baseline = ThreadsAtRest(rig);
    ASSERT_EQ(rig.PleRows(Tokens(64)), 0);
    ASSERT_EQ(Stats().gather_pool_threads, GatherWorkers());

    plane_.reset();
    EXPECT_EQ(Stats().attached_loaders, 1);
    EXPECT_EQ(Stats().gather_pool_threads, GatherWorkers()) << "the workers went with a loader that was not the last";
    const std::vector<int64_t> tokens = Tokens(96);
    ASSERT_EQ(rig.PleRows(tokens), 0);
    EXPECT_EQ(rig.ReadPle(tokens.size()), rig.ExpectedPle(tokens));
  }
  second.reset();
  EXPECT_EQ(Stats().attached_loaders, 0);
  EXPECT_EQ(Stats().gather_pool_threads, 0);
  EXPECT_TRUE(ProcessThreadCountSettlesAtOrBelow(baseline));
}

TEST_F(Qwen38PlaneGatherPoolTest, ReleasingThePleWorkspaceStopsTheWorkersAndTheNextPrefillStartsThemAgain) {
  StagedPleRig rig{*plane_, kAscendingRows};
  const int baseline = ThreadsAtRest(rig);
  const std::vector<int64_t> tokens = Tokens(200);
  for (int round = 0; round < 3; ++round) {
    SCOPED_TRACE(round);
    ASSERT_EQ(rig.PleRows(tokens), 0);
    EXPECT_EQ(Stats().gather_pool_threads, GatherWorkers());
    EXPECT_EQ(rig.ReadPle(tokens.size()), rig.ExpectedPle(tokens));

    plane_->release_ple_workspace();
    EXPECT_EQ(Stats().gather_pool_threads, 0);
    EXPECT_TRUE(ProcessThreadCountSettlesAtOrBelow(baseline));
    plane_->release_ple_workspace();  // nothing is left to stop
    EXPECT_EQ(Stats().gather_pool_threads, 0);

    rig.ResetTable();
  }
}

// A thread that cannot be created is an error of the prefill, which gathers nothing, and the workers that did start are joined,
// so that a failure never leaves a pool that waits for threads that do not exist. The next prefill tries again.
TEST_F(Qwen38PlaneGatherPoolTest, AWorkerThatCannotBeStartedIsAnErrorAndLeavesNoThreadBehind) {
  StagedPleRig rig{*plane_, kAscendingRows};
  const int baseline = ThreadsAtRest(rig);
  const int workers = GatherWorkers();
  const std::vector<int64_t> tokens = Tokens(500);
  for (const int failing : {0, workers / 2, workers - 1}) {
    SCOPED_TRACE(failing);
    rig.FillOutputs(0xAB);
    const Qwen38PlaneTestStats before = Stats();
    Arm(QWEN38_PLANE_FAULT_THREAD_START, failing, 5000);
    EXPECT_EQ(rig.PleRows(tokens), kCudaErrorMemoryAllocation);
    const Qwen38PlaneTestStats failed = Stats();
    EXPECT_EQ(failed.fault_armed, 0) << "no thread creation failed";
    EXPECT_EQ(failed.gather_pool_threads, 0) << "the workers that had started were left running";
    EXPECT_EQ(failed.gather_rows, before.gather_rows);
    EXPECT_EQ(failed.staged_launches, before.staged_launches);
    EXPECT_TRUE(rig.OutputsAre(0xAB));
    EXPECT_TRUE(ProcessThreadCountSettlesAtOrBelow(baseline));

    ASSERT_EQ(rig.PleRows(tokens), 0);
    EXPECT_EQ(Stats().gather_pool_threads, workers);
    EXPECT_EQ(rig.ReadPle(tokens.size()), rig.ExpectedPle(tokens));

    plane_->release_ple_workspace();
    rig.ResetTable();
  }
}

// What the draft selector reads and writes: the candidates and scores of up to 7 drafts, and the control words that receive the
// chosen candidates. A sentinel word marks every word of the control state that the selector does not write.
constexpr std::array<size_t, 7> kDflashDraftSlots = {2, 8, 9, 10, 11, 12, 13};
constexpr size_t kDflashControlWords = 16;

std::vector<int64_t> DflashControlSentinels() {
  std::vector<int64_t> control(kDflashControlWords);
  for (size_t i = 0; i < control.size(); ++i) control[i] = -1000 - static_cast<int64_t>(i);
  return control;
}

// The candidates of `drafts` steps and their scores: the scores of step s after candidate p of step s - 1 are
// scores[(s * top_k + p) * top_k ...], one for each candidate of step s (step 0 reads the row of p = 0).
struct DflashLattice {
  int drafts;
  int top_k;
  std::vector<int32_t> candidates;  // [drafts, top_k]
  std::vector<float> scores;        // [drafts, top_k, top_k]
};

// The selection on the host: a greedy walk that takes, at every step, the candidate with the highest score in the row of the
// candidate chosen before it. Of equal scores the first wins (std::max_element), like the selector of the drafter on the host
// that the plane's one replaces (src/dflash2_drafter.cpp).
std::vector<int64_t> SelectOnTheHost(const DflashLattice& lattice) {
  std::vector<int64_t> control = DflashControlSentinels();
  const size_t top_k = static_cast<size_t>(lattice.top_k);
  size_t previous = 0;
  for (size_t step = 0; step < static_cast<size_t>(lattice.drafts); ++step) {
    const float* row = lattice.scores.data() + (step * top_k + previous) * top_k;
    const size_t best = static_cast<size_t>(std::max_element(row, row + top_k) - row);
    control[kDflashDraftSlots[step]] = lattice.candidates[step * top_k + best];
    previous = best;
  }
  return control;
}

class Qwen38PlaneDflashSelectTest : public PlaneKernelFixture<::testing::Test> {
 protected:
  // The control words the selector leaves, the sentinels where it writes nothing.
  std::vector<int64_t> SelectOnTheDevice(KernelTestDevice& device, const DflashLattice& lattice) {
    const PlaneBuffer candidates = device.Upload(lattice.candidates);
    const PlaneBuffer scores = device.Upload(lattice.scores);
    const PlaneBuffer control = device.Upload(DflashControlSentinels());
    EXPECT_EQ(plane_->dflash_select(device.stream(), candidates.data(), scores.data(), control.data(), lattice.drafts,
                                    lattice.top_k),
              0);
    return device.Download<int64_t>(control, kDflashControlWords);
  }
};

// The rule that decides a tie: of several candidates with the highest score the one with the LOWEST index wins, because a
// candidate takes over only when its score is strictly greater than that of the best so far. Signed zeros are equal scores.
TEST_F(Qwen38PlaneDflashSelectTest, TheCandidateWithTheLowestIndexWinsATie) {
  ASSERT_TRUE(plane_->dflash_select != nullptr) << "the plane library has no dflash_select";
  KernelTestDevice device{*plane_};
  const float infinity = std::numeric_limits<float>::infinity();
  struct Case {
    const char* name;
    std::vector<float> scores;
    size_t winner;
  };
  const Case cases[] = {
      {"one maximum in the middle", {1, 5, 3, 2}, 1},
      {"one maximum at the first index", {4, 3, 2, 1}, 0},
      {"one maximum at the last index", {1, 2, 3, 4}, 3},
      {"two equal maxima", {1, 5, 5, 2}, 1},
      {"equal maxima at both ends", {5, 1, 2, 5}, 0},
      {"three equal maxima", {1, 5, 5, 5}, 1},
      {"all scores equal", {3, 3, 3, 3}, 0},
      {"equal negative maxima", {-5, -3, -3, -4}, 1},
      {"equal infinite maxima", {1, infinity, infinity, 2}, 1},
      {"all scores minus infinity", {-infinity, -infinity, -infinity, -infinity}, 0},
      {"a zero and a negative zero", {-1, -0.0f, 0.0f, -1}, 1},
      {"a negative zero and a zero", {-1, 0.0f, -0.0f, -1}, 1},
  };
  for (const Case& test_case : cases) {
    const DflashLattice lattice{1, 4, {10, 20, 30, 40}, test_case.scores};
    const std::vector<int64_t> control = SelectOnTheDevice(device, lattice);
    EXPECT_EQ(control[kDflashDraftSlots[0]], lattice.candidates[test_case.winner]) << test_case.name;
    EXPECT_EQ(control, SelectOnTheHost(lattice)) << test_case.name;
  }
}

// The candidate a step chose selects the row of scores the next step reads. The lattice is built so that every step has a row
// for each candidate chosen before it and only one of them leads to the expected token: the ones that are not read hold a
// maximum of their own, which a selector that ignores the walk would pick.
TEST_F(Qwen38PlaneDflashSelectTest, TheChosenCandidateSelectsTheScoresOfTheNextStep) {
  ASSERT_TRUE(plane_->dflash_select != nullptr) << "the plane library has no dflash_select";
  KernelTestDevice device{*plane_};
  constexpr int kDrafts = 4;
  constexpr int kTopK = 3;
  DflashLattice lattice{kDrafts, kTopK, {}, std::vector<float>(kDrafts * kTopK * kTopK, 0.0f)};
  for (int step = 0; step < kDrafts; ++step) {
    for (int k = 0; k < kTopK; ++k) lattice.candidates.push_back(100 * (step + 1) + k);
  }
  const auto set_row = [&lattice](int step, int previous, std::array<float, kTopK> row) {
    std::copy(row.begin(), row.end(), lattice.scores.begin() + (step * kTopK + previous) * kTopK);
  };
  // The walk: step 0 reads the row of previous = 0 and picks index 1; step 1 reads the row of 1 and picks 2; step 2 reads the
  // row of 2, where all scores are equal, and picks 0; step 3 reads the row of 0, where the maximum is shared by 1 and 2, and
  // picks 1.
  set_row(0, 0, {1, 7, 3});
  set_row(0, 1, {9, 0, 0});
  set_row(0, 2, {0, 9, 0});
  set_row(1, 0, {8, 0, 0});
  set_row(1, 1, {4, 4, 6});
  set_row(1, 2, {0, 0, 0});
  set_row(2, 0, {0, 0, 9});
  set_row(2, 1, {0, 9, 0});
  set_row(2, 2, {5, 5, 5});
  set_row(3, 0, {2, 6, 6});
  set_row(3, 1, {0, 0, 9});
  set_row(3, 2, {9, 0, 0});
  const std::vector<int64_t> control = SelectOnTheDevice(device, lattice);
  EXPECT_EQ(control[kDflashDraftSlots[0]], 101);
  EXPECT_EQ(control[kDflashDraftSlots[1]], 202);
  EXPECT_EQ(control[kDflashDraftSlots[2]], 300);
  EXPECT_EQ(control[kDflashDraftSlots[3]], 401);
  EXPECT_EQ(control, SelectOnTheHost(lattice));
}

TEST_F(Qwen38PlaneDflashSelectTest, MatchesTheHostWalkForRandomLatticesWithManyTies) {
  ASSERT_TRUE(plane_->dflash_select != nullptr) << "the plane library has no dflash_select";
  KernelTestDevice device{*plane_};
  WordStream words{4242};
  const float infinity = std::numeric_limits<float>::infinity();
  for (int top_k : {1, 2, 3, 8, 16}) {
    for (int drafts = 1; drafts <= 7; ++drafts) {
      for (int round = 0; round < 4; ++round) {
        DflashLattice lattice{drafts, top_k, {}, {}};
        for (int i = 0; i < drafts * top_k; ++i) lattice.candidates.push_back(static_cast<int32_t>(words.Next() % 248000u));
        // Scores of four distinct values (and some minus infinities) make equal maxima common.
        for (int i = 0; i < drafts * top_k * top_k; ++i) {
          const uint32_t word = words.Next();
          lattice.scores.push_back(word % 11u == 0 ? -infinity : static_cast<float>(word % 4u));
        }
        const std::string what = std::to_string(drafts) + " drafts of " + std::to_string(top_k) + " candidates, round " +
                                 std::to_string(round);
        ExpectSameElements(SelectOnTheHost(lattice), SelectOnTheDevice(device, lattice), what);
      }
    }
  }
}

// Only the words of the drafts that were asked for are written (the words between them are not the selector's), and a call it
// cannot serve writes none.
TEST_F(Qwen38PlaneDflashSelectTest, WritesOnlyTheWordsOfTheRequestedDraftsAndRejectsWhatItCannotServe) {
  ASSERT_TRUE(plane_->dflash_select != nullptr) << "the plane library has no dflash_select";
  KernelTestDevice device{*plane_};
  const DflashLattice lattice{3, 2, {1, 2, 3, 4, 5, 6}, std::vector<float>(3 * 2 * 2, 1.0f)};
  const std::vector<int64_t> control = SelectOnTheDevice(device, lattice);
  std::vector<int64_t> expected = DflashControlSentinels();
  expected[2] = 1;
  expected[8] = 3;
  expected[9] = 5;
  EXPECT_EQ(control, expected);

  const PlaneBuffer candidates = device.Upload(lattice.candidates);
  const PlaneBuffer scores = device.Upload(lattice.scores);
  const PlaneBuffer words = device.Upload(DflashControlSentinels());
  void* stream = device.stream();
  EXPECT_EQ(plane_->dflash_select(stream, nullptr, scores.data(), words.data(), 3, 2), kCudaErrorInvalidValue);
  EXPECT_EQ(plane_->dflash_select(stream, candidates.data(), nullptr, words.data(), 3, 2), kCudaErrorInvalidValue);
  EXPECT_EQ(plane_->dflash_select(stream, candidates.data(), scores.data(), nullptr, 3, 2), kCudaErrorInvalidValue);
  EXPECT_EQ(plane_->dflash_select(stream, candidates.data(), scores.data(), words.data(), 0, 2), kCudaErrorInvalidValue);
  EXPECT_EQ(plane_->dflash_select(stream, candidates.data(), scores.data(), words.data(), 8, 2), kCudaErrorInvalidValue);
  EXPECT_EQ(plane_->dflash_select(stream, candidates.data(), scores.data(), words.data(), 3, 0), kCudaErrorInvalidValue);
  EXPECT_EQ(device.Download<int64_t>(words, kDflashControlWords), DflashControlSentinels());
}

// The error of an earlier call that nobody read (here a memset of a null pointer) is not the error of the selection that follows.
TEST_F(Qwen38PlaneDflashSelectTest, AnErrorAnEarlierCallLeftUnreadDoesNotFailTheLaunch) {
  ASSERT_TRUE(plane_->dflash_select != nullptr) << "the plane library has no dflash_select";
  KernelTestDevice device{*plane_};
  const DflashLattice lattice{2, 2, {1, 2, 3, 4}, {0, 1, 2, 0, 5, 3, 1, 1}};
  device.LeaveAnUnreadError();
  EXPECT_EQ(SelectOnTheDevice(device, lattice), SelectOnTheHost(lattice));
}

}  // namespace
#endif

}  // namespace Generators::test
