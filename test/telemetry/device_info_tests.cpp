// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <stdlib.h>

#include <gtest/gtest.h>

#include "telemetry/device_info.h"
#include "telemetry/sha256.h"
#include "telemetry_test_environment.h"

namespace {

namespace fs = std::filesystem;

class ScopedEnvironmentVariable {
 public:
  ScopedEnvironmentVariable(const char* name, std::optional<std::string> value) : name_(name) {
    if (const char* existing = std::getenv(name); existing != nullptr) {
      original_value_ = existing;
    }
    Set(value);
  }

  ~ScopedEnvironmentVariable() { Set(original_value_); }

 private:
  void Set(const std::optional<std::string>& value) const {
    if (value.has_value()) {
      setenv(name_.c_str(), value->c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  std::string name_;
  std::optional<std::string> original_value_;
};

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

#endif

TEST(TelemetryDeviceInfoTest, SharedDeviceIdUsesUnsaltedSha256) {
  EXPECT_EQ(Generators::TelemetryInternal::Sha256::HashStringHex(""),
            "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855");
  EXPECT_EQ(Generators::TelemetryInternal::Sha256::HashStringHex("abc"),
            "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD");
  EXPECT_EQ(Generators::TelemetryInternal::Sha256::HashStringHex(
                "01234567-89ab-4def-8123-456789abcdef"),
            "6225BD190D6CCF87766A49C9986D174DEF3391FE175A61525E49A1D2334D6A43");
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

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
