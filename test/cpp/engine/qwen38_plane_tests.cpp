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
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "config.h"
#include "models/qwen38_graph.h"
#include "models/qwen38_graph_plane.h"

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
  EXPECT_EQ(DescribeQwen38PlaneCapabilities(QWEN38_PLANE_ALL_CAPABILITIES),
            "int8_embedding, int8_ple, negative_pos_add_batch");
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
    EXPECT_NE(message.find("the fold package requires the plane capability int8_embedding, negative_pos_add_batch, which"),
              std::string::npos)
        << message;
  }

  plane.capabilities = 0;  // a library that predates the capabilities export reports none
  ExpectMissingCapability([&] { plane.RequireCapabilities(QWEN38_PLANE_CAP_INT8_PLE, "the INT8 PLE table"); }, "int8_ple",
                          "the INT8 PLE table", "model/libqwen38_plane.so");
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

  // `fold` makes it a package of the folded MTP stack (model.qwen38_graph.fold.enabled).
  explicit Qwen38TestPackage(bool fold = false) : dir_{MakeScratchDir(fold ? "package_support_fold" : "package_support")} {
    const std::string fold_stack =
        fold ? R"(,"mtp":{"enabled":true},"fold":{"enabled":true,"head_r2":{"filename":"head_r2.onnx"}})" : "";
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
  return std::filesystem::path{ORTGENAI_TEST_QWEN38_PLANE_LIB_PATH};
#elif defined(_WIN32)
  return std::filesystem::current_path() / "qwen38_plane.dll";
#else
  return std::filesystem::current_path() / "libqwen38_plane.so";
#endif
}

// The bfloat16 bits of `value` rounded to nearest even, as the plane rounds.
uint16_t Bf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

}  // namespace

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

// A fold step prepared in one call (a negative pos_add) of a staged table: every row is prepared, and rows 1 and up reuse what
// row 0 read back, so they copy nothing from the device and wait for no stream; only row 0 gets the committed length, and no row
// is handed anything but device memory.
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
    EXPECT_EQ(after.staged_launches, before.staged_launches + rows);
    rig.ExpectFoldStepResult(state, rows);
  }

  // The table is warm now: row 0 makes the two copies of the control words and the context and one wait, no row after it does.
  const int rows = static_cast<int>(StagedPleRig::kMaxFoldRows);
  rig.FillOutputs(0xAB);
  Arm(QWEN38_PLANE_FAULT_COPY, 2, 5000);
  ASSERT_EQ(rig.PrepFoldStep(state, rows), 0);
  EXPECT_EQ(Stats().fault_armed, 1) << "a row after row 0 copied from the device";
  Arm(QWEN38_PLANE_FAULT_SYNC, 1, 5001);
  ASSERT_EQ(rig.PrepFoldStep(state, rows), 0);
  EXPECT_EQ(Stats().fault_armed, 1) << "a row after row 0 waited for the stream";
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
#endif

}  // namespace Generators::test
