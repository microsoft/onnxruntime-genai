// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <stdlib.h>

#include <gtest/gtest.h>

#include "telemetry/device_info.h"
#include "telemetry/sha256.h"
#include "telemetry/telemetry_string.h"
#include "telemetry_test_environment.h"
#include "scoped-environment-variable.h"

namespace {

namespace fs = std::filesystem;

using Generators::test::ScopedEnvironmentVariable;

class ScopedTestDirectory {
 public:
  explicit ScopedTestDirectory(std::string_view name)
      : path_(fs::temp_directory_path() /
              (std::string{"ortgenai_device_id_"} + std::string{name} + "_" +
               std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
    fs::create_directories(path_);
  }

  ~ScopedTestDirectory() {
    std::error_code error;
    fs::remove_all(path_, error);
  }

  const fs::path& Path() const { return path_; }

 private:
  fs::path path_;
};

#if !defined(__APPLE__)

TEST(TelemetryDeviceInfoTest, UsesAbsoluteXdgCacheHomeWithoutHome) {
  ScopedTestDirectory test_dir{"absolute_xdg"};
  const fs::path cache_home = test_dir.Path() / "cache";
  ScopedEnvironmentVariable home{"HOME", std::nullopt};
  ScopedEnvironmentVariable xdg_cache_home{"XDG_CACHE_HOME", cache_home.string()};

  EXPECT_EQ(Generators::GetTelemetryStorageDir(),
            cache_home / "Microsoft" / "DeveloperTools" / ".onnxruntime");
}

TEST(TelemetryDeviceInfoTest, IgnoresRelativeXdgCacheHome) {
  ScopedTestDirectory test_dir{"relative_xdg"};
  const fs::path home_path = test_dir.Path() / "home";
  ScopedEnvironmentVariable home{"HOME", home_path.string()};
  ScopedEnvironmentVariable xdg_cache_home{"XDG_CACHE_HOME", "relative-cache"};

  EXPECT_EQ(Generators::GetTelemetryStorageDir(),
            home_path / ".cache" / "Microsoft" / "DeveloperTools" / ".onnxruntime");
}

TEST(TelemetryDeviceInfoTest, RejectsOversizedXdgCacheHomeWithoutTruncating) {
  ScopedTestDirectory test_dir{"oversized_xdg"};
  const fs::path home_path = test_dir.Path() / "home";
  ScopedEnvironmentVariable home{"HOME", home_path.string()};
  ScopedEnvironmentVariable xdg_cache_home{
      "XDG_CACHE_HOME", "/" + std::string(Generators::kMaxTelemetryInputBytes, 'x')};
  EXPECT_EQ(Generators::GetTelemetryStorageDir(),
            home_path / ".cache" / "Microsoft" / "DeveloperTools" / ".onnxruntime");
}

#endif

TEST(TelemetryDeviceInfoTest, DeviceIdHashMatchesSharedProtocol) {
  EXPECT_EQ(Generators::telemetry_internal::Sha256::HashStringHex(
                "00000000-0000-4000-8000-000000000000"),
            "db8055e0e0307d5a016bec4dc338d69875eb0fb7e614a8b125b08fb082095d98");
}

TEST(TelemetryDeviceInfoDeathTest, RejectsSymlinkedOwnedDirectoryBeforeReading) {
  ScopedTestDirectory test_dir{"symlink_leaf"};
  const fs::path home_path = test_dir.Path() / "home";
  ScopedEnvironmentVariable home{"HOME", home_path.string()};
  ScopedEnvironmentVariable xdg_cache_home{"XDG_CACHE_HOME", std::nullopt};

#if defined(__APPLE__)
  const fs::path storage_dir =
      home_path / "Library" / "Application Support" / "Microsoft" / "DeveloperTools" / ".onnxruntime";
#else
  const fs::path storage_dir =
      home_path / ".cache" / "Microsoft" / "DeveloperTools" / ".onnxruntime";
#endif
  const fs::path redirected_dir = test_dir.Path() / "redirected";
  fs::create_directories(storage_dir.parent_path());
  fs::create_directories(redirected_dir);
  std::ofstream(redirected_dir / "deviceid") << "11111111-2222-4333-8444-555555555555";
  fs::create_directory_symlink(redirected_dir, storage_dir);

  EXPECT_EXIT(
      {
        std::_Exit(Generators::GetDeviceInfo().device_id_status == "Failed"
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST(TelemetryDeviceInfoDeathTest, RepairsCorruptedFile) {
  ScopedTestDirectory test_dir{"corrupted"};
  const fs::path home_path = test_dir.Path() / "home";
  ScopedEnvironmentVariable home{"HOME", home_path.string()};
  ScopedEnvironmentVariable xdg_cache_home{"XDG_CACHE_HOME", std::nullopt};

#if defined(__APPLE__)
  const fs::path storage_dir =
      home_path / "Library" / "Application Support" / "Microsoft" / "DeveloperTools" / ".onnxruntime";
#else
  const fs::path storage_dir =
      home_path / ".cache" / "Microsoft" / "DeveloperTools" / ".onnxruntime";
#endif
  fs::create_directories(storage_dir);
  std::ofstream(storage_dir / "deviceid") << "corrupted";

  EXPECT_EXIT(
      {
        std::_Exit(Generators::GetDeviceInfo().device_id_status == "Corrupted"
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");

  std::ifstream input(storage_dir / "deviceid");
  std::string persisted;
  input >> persisted;
  EXPECT_EQ(persisted.size(), 36u);
  EXPECT_NE(persisted, "corrupted");
}

TEST(TelemetryDeviceInfoDeathTest, PreservesWhitespaceWrappedDeviceId) {
  ScopedTestDirectory test_dir{"whitespace"};
  const fs::path home_path = test_dir.Path() / "home";
  ScopedEnvironmentVariable home{"HOME", home_path.string()};
  ScopedEnvironmentVariable xdg_cache_home{"XDG_CACHE_HOME", std::nullopt};

#if defined(__APPLE__)
  const fs::path storage_dir =
      home_path / "Library" / "Application Support" / "Microsoft" / "DeveloperTools" / ".onnxruntime";
#else
  const fs::path storage_dir =
      home_path / ".cache" / "Microsoft" / "DeveloperTools" / ".onnxruntime";
#endif
  fs::create_directories(storage_dir);
  const std::string stored = " \t11111111-2222-4333-8444-555555555555\r\n ";
  std::ofstream(storage_dir / "deviceid") << stored;
  EXPECT_EXIT(
      {
        std::_Exit(Generators::GetDeviceInfo().device_id_status == "Existing"
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");

  std::ifstream input(storage_dir / "deviceid", std::ios::binary);
  const std::string persisted{std::istreambuf_iterator<char>{input}, std::istreambuf_iterator<char>{}};
  EXPECT_EQ(persisted, stored);
}

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
