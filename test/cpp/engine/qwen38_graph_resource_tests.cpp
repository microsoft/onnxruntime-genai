// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(__linux__)
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sys/mman.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include "models/qwen38_graph.h"

namespace Generators::test {
namespace {

class Qwen38GraphResources : public ::testing::Test {
 protected:
  void SetUp() override {
    active = this;
    root = std::filesystem::temp_directory_path() /
           ("ortgenai_qwen38_resources_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root);
    plane.error_string = [](int) { return "injected failure"; };
    plane.set_device = [](int) { return 0; };
    plane.malloc_ = [](void** ptr, size_t bytes) {
      if (++active->allocation_attempts == active->fail_allocation) return 1;
      *ptr = std::malloc(bytes);
      if (!*ptr) return 1;
      ++active->allocations;
      return 0;
    };
    plane.free_ = [](void* ptr) {
      std::free(ptr);
      ++active->frees;
      return 0;
    };
    plane.memset_ = [](void* ptr, int value, size_t bytes, void*) {
      if (active->fail_memset) return 1;
      std::memset(ptr, value, bytes);
      return 0;
    };
    plane.memcpy_h2d = [](void* dst, const void* src, size_t bytes, void*) {
      std::memcpy(dst, src, bytes);
      return 0;
    };
    plane.stream_sync = [](void*) { return 0; };
    plane.host_register = [](void* ptr, size_t, unsigned) {
      active->mappings.push_back(ptr);
      return active->mappings.size() == active->fail_registration ? 1 : 0;
    };
    plane.host_unregister = [](void*) {
      ++active->unregistrations;
      return 0;
    };
    plane.host_alloc = [](void** ptr, size_t bytes) {
      *ptr = std::malloc(bytes);
      return *ptr ? 0 : 1;
    };
    plane.host_free = [](void* ptr) {
      std::free(ptr);
      return 0;
    };
    plane.stream_create = [](void** stream) {
      *stream = active;
      return 0;
    };
    plane.stream_destroy = [](void*) { return 0; };
  }

  void TearDown() override {
    EXPECT_EQ(allocations, frees);
    for (void* ptr : mappings) {
      unsigned char residency{};
      errno = 0;
      EXPECT_EQ(mincore(ptr, static_cast<size_t>(sysconf(_SC_PAGESIZE)), &residency), -1);
      EXPECT_EQ(errno, ENOMEM);
    }
    std::filesystem::remove_all(root);
    active = nullptr;
  }

  void Write(const std::string& name, const std::string& contents) {
    std::ofstream file(root / name, std::ios::binary);
    file << contents;
  }

  void LoadPle() {
    Qwen38PleTable table(plane, (root / "ple.txt").string(), root.string(), 32);
  }

  void LoadWeights() {
    (void)GetOrtEnv();
    auto memory = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Qwen38WeightPool pool(plane, (root / "weights.txt").string(), root.string(), *memory, 1);
  }

  void WriteConstants(const std::string& omitted = "") {
    std::string index = "qwen38_qsa_consts 1\nfile constants.bin\nwidth 1\nepsilon 0.000001\n";
    size_t offset = 0;
    for (const char* set : {"target", "head"}) {
      for (const char* name : {"cos", "sin", "knorm.0"}) {
        const std::string key = std::string(set) + "." + name;
        if (key != omitted)
          index += "tensor " + std::string(set) + " " + name + " " + std::to_string(offset) + " 2\n";
        offset += 2;
      }
    }
    Write("constants.txt", index);
    Write("constants.bin", std::string(offset, '\0'));
  }

  void LoadConstants() {
    Qwen38QsaConstants constants(plane, (root / "constants.txt").string(), root.string(), nullptr);
  }

  inline static Qwen38GraphResources* active{};
  Qwen38Plane plane;
  std::filesystem::path root;
  size_t allocations{}, frees{}, allocation_attempts{}, fail_allocation{};
  size_t fail_registration{}, unregistrations{};
  bool fail_memset{};
  std::vector<void*> mappings;
};

TEST_F(Qwen38GraphResources, RejectsHeadCountsBeforeArrayAccess) {
  for (const char* count : {"33", "0", "-1", "not-an-integer"}) {
    Write("ple.txt", std::string("qwen38_ple_table 1\nn_heads ") + count + "\nvocab 1\noffset 0\n");
    EXPECT_THROW(LoadPle(), std::runtime_error);
  }
  EXPECT_TRUE(mappings.empty());
}

TEST(Qwen38GraphPlaneInterface, RejectsIncompleteFunctionTable) {
  EXPECT_THROW((Qwen38Plane{Qwen38PlaneFunctions{}}), std::runtime_error);
  Qwen38Plane test_double;
  EXPECT_EQ(test_double.malloc_, nullptr);
  EXPECT_EQ(test_double.prep, nullptr);
}

TEST_F(Qwen38GraphResources, UnmapsCurrentAndEarlierShardsOnRegistrationFailure) {
  Write("shard.bin", "abcd");
  Write("ple.txt",
        "qwen38_ple_table 1\nn_heads 1\nhead_dim 1\nvocab 4\noffset 0\n"
        "shard shard.bin 0 4\nshard shard.bin 0 4\n");
  fail_registration = 2;
  EXPECT_THROW(LoadPle(), std::runtime_error);
  EXPECT_EQ(mappings.size(), 2U);
  EXPECT_EQ(unregistrations, 1U);
}

TEST_F(Qwen38GraphResources, UnmapsRegisteredShardWhenLaterFileIsMissing) {
  Write("shard.bin", "abcd");
  Write("ple.txt",
        "qwen38_ple_table 1\nn_heads 1\nhead_dim 1\nvocab 4\noffset 0\n"
        "shard shard.bin 0 4\nshard missing.bin 0 4\n");
  EXPECT_THROW(LoadPle(), std::runtime_error);
  EXPECT_EQ(unregistrations, 1U);
}

TEST_F(Qwen38GraphResources, ReleasesAllocationIfInitializationFails) {
  fail_memset = true;
  EXPECT_THROW((PlaneBuffer{plane, 16, nullptr}), std::runtime_error);
  EXPECT_EQ(allocations, 1U);
  EXPECT_EQ(frees, 1U);
}

TEST_F(Qwen38GraphResources, ReleasesWeightsWhenAllocationOrTensorCreationFails) {
  Write("weights.txt", "qwen38_weights_pool 1\npool 0 2 0 4 1 4 weights.bin\npool 1 2 4 4 1 4 weights.bin\n");
  fail_allocation = 2;
  EXPECT_THROW(LoadWeights(), std::runtime_error);
  EXPECT_EQ(frees, 1U);
  fail_allocation = 0;
  Write("weights.txt", "qwen38_weights_pool 1\npool 0 2 0 4 1 4 weights.bin\npool 1 1 4 4 1 4 weights.bin\n");
  EXPECT_THROW(LoadWeights(), std::exception);
  EXPECT_EQ(allocations, frees);
}

TEST_F(Qwen38GraphResources, ReleasesWeightsWhenFileIsMissingOrTruncated) {
  Write("weights.txt", "qwen38_weights_pool 1\npool 0 2 0 4 1 4 weights.bin\npool 1 2 4 4 1 4 weights.bin\n");
  EXPECT_THROW(LoadWeights(), std::runtime_error);
  EXPECT_EQ(allocations, frees);
  Write("weights.bin", "abc");
  EXPECT_THROW(LoadWeights(), std::runtime_error);
  EXPECT_EQ(allocations, frees);
}

TEST_F(Qwen38GraphResources, RejectsConstantRangesBeforeAllocation) {
  for (const auto& range : {"13 2", "11 2", "0 0"}) {
    WriteConstants();
    std::ofstream file(root / "constants.txt", std::ios::app);
    file << "tensor target cos " << range << "\n";
    file.close();
    EXPECT_THROW(LoadConstants(), std::runtime_error);
  }
  WriteConstants();
  std::ofstream file(root / "constants.txt", std::ios::app);
  file << "tensor target cos 1 " << std::numeric_limits<size_t>::max() << "\n";
  file.close();
  EXPECT_THROW(LoadConstants(), std::runtime_error);
  EXPECT_EQ(allocations, 0U);
}

TEST_F(Qwen38GraphResources, RequiresBothSineTablesAndAcceptsCompleteConstants) {
  for (const char* omitted : {"target.sin", "head.sin"}) {
    WriteConstants(omitted);
    EXPECT_THROW(LoadConstants(), std::runtime_error);
    EXPECT_EQ(allocations, frees);
  }
  WriteConstants();
  EXPECT_NO_THROW(LoadConstants());
}

TEST_F(Qwen38GraphResources, RejectsTruncatedConstantsBeforeAllocation) {
  WriteConstants();
  Write("constants.bin", "abc");
  EXPECT_THROW(LoadConstants(), std::runtime_error);
  EXPECT_EQ(allocations, 0U);
}

}  // namespace
}  // namespace Generators::test
#endif
