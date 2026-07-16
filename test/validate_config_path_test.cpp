// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "models/validate_config_path.h"

#include <gtest/gtest.h>

namespace Generators::test {

TEST(ValidateConfigPathTest, AcceptsSimpleFilename) {
  EXPECT_NO_THROW(ValidateConfigPath("model.onnx"));
}

TEST(ValidateConfigPathTest, AcceptsRelativeSubdirectory) {
  EXPECT_NO_THROW(ValidateConfigPath("subdir/model.onnx"));
}

TEST(ValidateConfigPathTest, RejectsAbsolutePathUnix) {
  EXPECT_THROW(ValidateConfigPath("/etc/passwd"), std::runtime_error);
}

#if defined(_WIN32)
TEST(ValidateConfigPathTest, RejectsAbsolutePathWindows) {
  EXPECT_THROW(ValidateConfigPath("C:\\Windows\\System32\\evil.dll"), std::runtime_error);
}

TEST(ValidateConfigPathTest, RejectsDriveRelativePathWindows) {
  EXPECT_THROW(ValidateConfigPath("C:Windows\\System32\\evil.dll"), std::runtime_error);
}
#endif

TEST(ValidateConfigPathTest, RejectsParentTraversal) {
  EXPECT_THROW(ValidateConfigPath("../../../etc/passwd"), std::runtime_error);
}

TEST(ValidateConfigPathTest, RejectsEmbeddedParentTraversal) {
  EXPECT_THROW(ValidateConfigPath("subdir/../../etc/passwd"), std::runtime_error);
}

TEST(ValidateConfigPathTest, RejectsSingleDotDot) {
  EXPECT_THROW(ValidateConfigPath(".."), std::runtime_error);
}

TEST(ValidateConfigPathTest, AcceptsSingleDot) {
  EXPECT_NO_THROW(ValidateConfigPath("./model.onnx"));
}

}  // namespace Generators::test
