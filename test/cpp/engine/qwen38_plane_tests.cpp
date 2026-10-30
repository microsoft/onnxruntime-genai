// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "models/qwen38_graph_plane.h"

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
  plane.ple_rows(stream, nullptr, nullptr, 0, nullptr, nullptr, nullptr);
  plane.ple_rows(stream, nullptr, nullptr, -1, nullptr, nullptr, nullptr);
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
  plane.ple_rows(stream, d_p1.data(), d_tokens.data(), 2, d_ctx_in.data(), d_ctx_out.data(), d_out1.data());
  plane.ple_rows(stream, d_p2.data(), d_tokens.data(), 2, d_ctx_in.data(), d_ctx_out.data(), d_out2.data());

  std::vector<uint16_t> out1(8, 0), out2(8, 0);
  plane.Check(plane.memcpy_d2h(out1.data(), d_out1.data(), out1.size() * sizeof(uint16_t), stream), "d2h out1");
  plane.Check(plane.memcpy_d2h(out2.data(), d_out2.data(), out2.size() * sizeof(uint16_t), stream), "d2h out2");
  plane.Check(plane.stream_sync(stream), "sync");

  for (uint16_t v : out1) EXPECT_EQ(v, 0x3F80u);  // bf16(1.0f)
  for (uint16_t v : out2) EXPECT_EQ(v, 0x4080u);  // bf16(4.0f)

  // Updating p1 in-place and re-registering must invalidate p1's N-gram row cache.
  std::fill(shard1.begin(), shard1.end(), 0x40u);  // 2.0f * scale(1.0f) -> bf16 2.0f (0x4000)
  plane.set_ple_host_params(&p1);
  plane.ple_rows(stream, d_p1.data(), d_tokens.data(), 2, d_ctx_in.data(), d_ctx_out.data(), d_out1.data());
  plane.Check(plane.memcpy_d2h(out1.data(), d_out1.data(), out1.size() * sizeof(uint16_t), stream), "d2h out1 updated");
  plane.Check(plane.stream_sync(stream), "sync updated");
  for (uint16_t v : out1) EXPECT_EQ(v, 0x4000u);  // bf16(2.0f)

  // Unregistering p2 leaves p1 operational and tears down p2's staged state.
  if (plane.unset_ple_host_params) {
    plane.unset_ple_host_params(&p2);
  }
  std::fill(out1.begin(), out1.end(), 0);
  plane.ple_rows(stream, d_p1.data(), d_tokens.data(), 2, d_ctx_in.data(), d_ctx_out.data(), d_out1.data());
  plane.Check(plane.memcpy_d2h(out1.data(), d_out1.data(), out1.size() * sizeof(uint16_t), stream), "d2h out1 after unset");
  plane.Check(plane.stream_sync(stream), "sync after unset");
  for (uint16_t v : out1) EXPECT_EQ(v, 0x4000u);

  if (plane.release_ple_workspace) {
    plane.release_ple_workspace();
  }
  plane.stream_destroy(stream);
}
#endif

}  // namespace Generators::test
