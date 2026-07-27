// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "config.h"

#include <gtest/gtest.h>

namespace Generators::test {

TEST(ValidateConfigPathTest, AcceptsSimpleFilename) {
  EXPECT_NO_THROW(Config::ValidatePath("model.onnx"));
}

TEST(ValidateConfigPathTest, AcceptsRelativeSubdirectory) {
  EXPECT_NO_THROW(Config::ValidatePath("subdir/model.onnx"));
}

TEST(ValidateConfigPathTest, RejectsAbsolutePathUnix) {
  EXPECT_THROW(Config::ValidatePath("/etc/passwd"), std::runtime_error);
}

#if defined(_WIN32)
TEST(ValidateConfigPathTest, RejectsAbsolutePathWindows) {
  EXPECT_THROW(Config::ValidatePath("C:\\Windows\\System32\\evil.dll"), std::runtime_error);
}

TEST(ValidateConfigPathTest, RejectsDriveRelativePathWindows) {
  EXPECT_THROW(Config::ValidatePath("C:Windows\\System32\\evil.dll"), std::runtime_error);
}
#endif

TEST(ValidateConfigPathTest, RejectsParentTraversal) {
  EXPECT_THROW(Config::ValidatePath("../../../etc/passwd"), std::runtime_error);
}

TEST(ValidateConfigPathTest, RejectsEmbeddedParentTraversal) {
  EXPECT_THROW(Config::ValidatePath("subdir/../../etc/passwd"), std::runtime_error);
}

TEST(ValidateConfigPathTest, RejectsSingleDotDot) {
  EXPECT_THROW(Config::ValidatePath(".."), std::runtime_error);
}

TEST(ValidateConfigPathTest, AcceptsSingleDot) {
  EXPECT_NO_THROW(Config::ValidatePath("./model.onnx"));
}

}  // namespace Generators::test
