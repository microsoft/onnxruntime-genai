// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <filesystem>
#include <gtest/gtest.h>

#include "models/qwen38_numa.h"

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>

namespace Generators::test {

namespace {
struct FileFixture {
  FileFixture() {
    std::string pattern = (std::filesystem::temp_directory_path() / "qwen38_numa_XXXXXX").string();
    fd = mkstemp(pattern.data());
    if (fd < 0) throw std::runtime_error("mkstemp failed");
    unlink(pattern.c_str());
    std::vector<uint8_t> content(bytes, 0x5a);
    if (write(fd, content.data(), content.size()) != static_cast<ssize_t>(bytes))
      throw std::runtime_error("weight fixture write failed");
    fsync(fd);
    data = mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
    if (data == MAP_FAILED) throw std::runtime_error("weight fixture mmap failed");
  }
  ~FileFixture() {
    if (data != MAP_FAILED) munmap(data, bytes);
    if (fd >= 0) close(fd);
  }
  static constexpr size_t bytes = 16 << 20;
  int fd = -1;
  void* data = MAP_FAILED;
};
}  // namespace

TEST(Qwen38NumaTest, SharedFilePlacementAndPolicyRestoration) {
  qwen38_numa::NodeMask allowed{}, before{}, after{};
  const auto nodes = qwen38_numa::AllowedNodes(allowed);
  if (nodes.size() < 2) GTEST_SKIP() << "Requires two allowed NUMA memory nodes";
  int mode_before = 0, mode_after = 0;
  ASSERT_EQ(syscall(SYS_get_mempolicy, &mode_before, before.data(), qwen38_numa::kMaxNodes, nullptr, 0), 0);
  FileFixture file;
  const auto report = qwen38_numa::PrepareFileRange(file.data, file.bytes, 0, true, true);
  ASSERT_EQ(syscall(SYS_get_mempolicy, &mode_after, after.data(), qwen38_numa::kMaxNodes, nullptr, 0), 0);
  EXPECT_EQ(mode_after, mode_before);
  EXPECT_EQ(after, before);
  EXPECT_EQ(report.restore_error, 0);
  if (report.policy_error || report.query_error || report.move_error || report.move_failures)
    GTEST_SKIP() << "NUMA operations unavailable: policy=" << report.policy_error << " query=" << report.query_error
                << " move=" << report.move_error << " per-page failures=" << report.move_failures;
  ASSERT_TRUE(report.policy_applied);
  ASSERT_GT(report.pages, 0u);
  EXPECT_EQ(report.unknown_pages, 0u);
  EXPECT_EQ(report.misplaced_after, 0u);
  for (size_t pages : report.pages_after) EXPECT_NEAR(pages, report.pages / nodes.size(), 1);
  const auto* data = static_cast<const uint8_t*>(file.data);
  for (size_t i = 0; i < file.bytes; ++i) ASSERT_EQ(data[i], 0x5a);
  const auto repeat = qwen38_numa::PrepareFileRange(file.data, file.bytes, 0, true, true);
  EXPECT_EQ(repeat.move_attempts, 0u);
  EXPECT_EQ(repeat.pages_after, report.pages_after);
}

TEST(Qwen38NumaTest, AuditDoesNotMigrateAndRejectsUnalignedRanges) {
  FileFixture file;
  const auto report = qwen38_numa::PrepareFileRange(file.data, file.bytes, 0, false, false);
  EXPECT_FALSE(report.policy_applied);
  EXPECT_EQ(report.move_attempts, 0u);
  EXPECT_EQ(report.pages_before, report.pages_after);
  const auto unaligned = qwen38_numa::PrepareFileRange(static_cast<uint8_t*>(file.data) + 1, 1, 0, true, true);
  if (unaligned.nodes.empty()) GTEST_SKIP() << "NUMA query unavailable";
  EXPECT_EQ(unaligned.policy_error, EINVAL);
  EXPECT_EQ(unaligned.move_attempts, 0u);
}

}  // namespace Generators::test
#endif
