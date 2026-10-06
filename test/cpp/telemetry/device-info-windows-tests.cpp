// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <Windows.h>

#include <array>
#include <cctype>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

#include <gtest/gtest.h>

#include "telemetry/device_info.h"
#include "telemetry_test_environment.h"

namespace {

constexpr char kDeviceIdRegistryKey[] = "SOFTWARE\\Microsoft\\DeveloperTools\\.onnxruntime";

class ScopedRegistryOverride {
 public:
  ScopedRegistryOverride()
      : path_("SOFTWARE\\Microsoft\\onnxruntime-genai-tests\\device-id-" +
              std::to_string(GetCurrentProcessId())) {
    if (RegCreateKeyExA(HKEY_CURRENT_USER, path_.c_str(), 0, nullptr, REG_OPTION_NON_VOLATILE,
                        KEY_ALL_ACCESS, nullptr, &key_, nullptr) != ERROR_SUCCESS) {
      throw std::runtime_error("Cannot create isolated device-ID registry key");
    }
    if (RegOverridePredefKey(HKEY_CURRENT_USER, key_) != ERROR_SUCCESS) {
      RegCloseKey(key_);
      RegDeleteTreeA(HKEY_CURRENT_USER, path_.c_str());
      throw std::runtime_error("Cannot override device-ID registry root");
    }
  }

  ~ScopedRegistryOverride() {
    RestoreReads();
    EXPECT_EQ(RegOverridePredefKey(HKEY_CURRENT_USER, nullptr), ERROR_SUCCESS);
    EXPECT_EQ(RegCloseKey(key_), ERROR_SUCCESS);
    EXPECT_EQ(RegDeleteTreeA(HKEY_CURRENT_USER, path_.c_str()), ERROR_SUCCESS);
  }

  ScopedRegistryOverride(const ScopedRegistryOverride&) = delete;
  ScopedRegistryOverride& operator=(const ScopedRegistryOverride&) = delete;

  void DenyReads() {
    ASSERT_EQ(RegOpenKeyExA(HKEY_CURRENT_USER, kDeviceIdRegistryKey, 0, KEY_ALL_ACCESS, &protected_key_),
              ERROR_SUCCESS);
    ACL acl{};
    SECURITY_DESCRIPTOR descriptor{};
    ASSERT_TRUE(InitializeAcl(&acl, sizeof(acl), ACL_REVISION));
    ASSERT_TRUE(InitializeSecurityDescriptor(&descriptor, SECURITY_DESCRIPTOR_REVISION));
    ASSERT_TRUE(SetSecurityDescriptorDacl(&descriptor, TRUE, &acl, FALSE));
    ASSERT_EQ(RegSetKeySecurity(protected_key_, DACL_SECURITY_INFORMATION, &descriptor), ERROR_SUCCESS);
  }

  void RestoreReads() {
    if (protected_key_ == nullptr) return;
    SECURITY_DESCRIPTOR descriptor{};
    ASSERT_TRUE(InitializeSecurityDescriptor(&descriptor, SECURITY_DESCRIPTOR_REVISION));
    ASSERT_TRUE(SetSecurityDescriptorDacl(&descriptor, TRUE, nullptr, FALSE));
    EXPECT_EQ(RegSetKeySecurity(protected_key_, DACL_SECURITY_INFORMATION, &descriptor), ERROR_SUCCESS);
    EXPECT_EQ(RegCloseKey(protected_key_), ERROR_SUCCESS);
    protected_key_ = nullptr;
  }

  void Write(DWORD type, std::string_view value) {
    HKEY key{};
    ASSERT_EQ(RegCreateKeyExA(HKEY_CURRENT_USER, kDeviceIdRegistryKey, 0, nullptr, REG_OPTION_NON_VOLATILE,
                              KEY_ALL_ACCESS, nullptr, &key, nullptr),
              ERROR_SUCCESS);
    const LSTATUS result = RegSetValueExA(key, "deviceid", 0, type,
                                          reinterpret_cast<const BYTE*>(value.data()),
                                          static_cast<DWORD>(value.size()));
    EXPECT_EQ(RegCloseKey(key), ERROR_SUCCESS);
    ASSERT_EQ(result, ERROR_SUCCESS);
  }

  std::string Read() {
    HKEY key{};
    if (RegOpenKeyExA(HKEY_CURRENT_USER, kDeviceIdRegistryKey, 0, KEY_READ, &key) != ERROR_SUCCESS) {
      throw std::runtime_error("Cannot open persisted device-ID registry value");
    }
    std::array<char, 256> buffer{};
    DWORD type = 0;
    DWORD size = static_cast<DWORD>(buffer.size());
    const LSTATUS result = RegQueryValueExA(key, "deviceid", nullptr, &type,
                                            reinterpret_cast<BYTE*>(buffer.data()), &size);
    EXPECT_EQ(RegCloseKey(key), ERROR_SUCCESS);
    if (result != ERROR_SUCCESS || type != REG_SZ || size == 0 || size > buffer.size()) {
      throw std::runtime_error("Cannot read persisted device-ID registry value");
    }
    return {buffer.data(), size - (buffer[size - 1] == '\0' ? 1 : 0)};
  }

 private:
  std::string path_;
  HKEY key_{};
  HKEY protected_key_{};
};

bool IsUuid(std::string_view value) {
  if (value.size() != 36) return false;
  for (size_t i = 0; i < value.size(); ++i) {
    const bool separator = i == 8 || i == 13 || i == 18 || i == 23;
    if (separator ? value[i] != '-' : !std::isxdigit(static_cast<unsigned char>(value[i]))) return false;
  }
  return true;
}

bool CheckDeviceId(DWORD type, std::string_view stored, std::string_view expected_status,
                   bool write_value = true) {
  bool passed = false;
  {
    ScopedRegistryOverride registry;
    if (write_value) registry.Write(type, stored);
    const auto& info = Generators::GetDeviceInfo();
    const std::string persisted = registry.Read();
    passed = info.device_id_status == expected_status && IsUuid(persisted) &&
             (expected_status != "Existing" || persisted == stored);
  }
  return passed && !::testing::Test::HasFailure();
}

TEST(TelemetryDeviceInfoWindowsDeathTest, CreatesMissingRegistryValue) {
  EXPECT_EXIT(std::_Exit(CheckDeviceId(REG_SZ, {}, "New", false) ? EXIT_SUCCESS : EXIT_FAILURE),
              ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST(TelemetryDeviceInfoWindowsDeathTest, PreservesExistingIdWithoutTerminatingNull) {
  EXPECT_EXIT(
      std::_Exit(CheckDeviceId(REG_SZ, "11111111-2222-4333-8444-555555555555", "Existing")
                     ? EXIT_SUCCESS
                     : EXIT_FAILURE),
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST(TelemetryDeviceInfoWindowsDeathTest, PreservesWhitespaceWrappedId) {
  EXPECT_EXIT(
      {
        bool passed = false;
        {
          ScopedRegistryOverride registry;
          const std::string stored = " \t11111111-2222-4333-8444-555555555555\r\n ";
          registry.Write(REG_SZ, stored);
          const auto& info = Generators::GetDeviceInfo();
          passed = !::testing::Test::HasFailure() && info.device_id_status == "Existing" &&
                   registry.Read() == stored;
        }
        std::_Exit(passed && !::testing::Test::HasFailure() ? EXIT_SUCCESS : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST(TelemetryDeviceInfoWindowsDeathTest, RepairsMalformedRegistryValues) {
  for (const auto& value : {std::string("corrupted"), std::string(512, 'x'),
                            std::string("11111111-2222-4333-8444-555555555555\0junk", 41)}) {
    EXPECT_EXIT(std::_Exit(CheckDeviceId(REG_SZ, value, "Corrupted") ? EXIT_SUCCESS : EXIT_FAILURE),
                ::testing::ExitedWithCode(EXIT_SUCCESS), "");
  }
  EXPECT_EXIT(std::_Exit(CheckDeviceId(REG_BINARY, "invalid type", "Corrupted") ? EXIT_SUCCESS : EXIT_FAILURE),
              ::testing::ExitedWithCode(EXIT_SUCCESS), "");
  EXPECT_EXIT(std::_Exit(CheckDeviceId(REG_SZ, {}, "Corrupted") ? EXIT_SUCCESS : EXIT_FAILURE),
              ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST(TelemetryDeviceInfoWindowsDeathTest, DoesNotOverwriteUnreadableRegistryValue) {
  EXPECT_EXIT(
      {
        bool passed = false;
        {
          ScopedRegistryOverride registry;
          constexpr std::string_view stored = "11111111-2222-4333-8444-555555555555";
          registry.Write(REG_SZ, stored);
          registry.DenyReads();
          const auto& info = Generators::GetDeviceInfo();
          registry.RestoreReads();
          passed = !::testing::Test::HasFailure() && info.device_id_status == "Failed" &&
                   registry.Read() == stored;
        }
        std::_Exit(passed && !::testing::Test::HasFailure() ? EXIT_SUCCESS : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

}  // namespace

int main(int argc, char** argv) {
  Generators::test::SuppressTelemetryForTests();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
