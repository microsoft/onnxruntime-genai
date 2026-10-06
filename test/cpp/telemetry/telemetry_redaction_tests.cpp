// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telemetry/telemetry_redaction.h"

#include <string>
#include <utility>
#include <gtest/gtest.h>

namespace Generators::test {

TEST(TelemetryRedactionTest, PreservesNonPathText) {
  for (const char* input : {"", "no path here", "error code 13", "models/foo.onnx",
                            "ratio 3/4 and and/or", "domain\\user", "read\\write access"}) {
    EXPECT_EQ(ScrubStringForTelemetry(input), input);
  }
}

TEST(TelemetryRedactionTest, RedactsPathAnchorsToEndOfMessage) {
  const std::pair<const char*, const char*> cases[] = {
      {"/home/alice/model.onnx", "[path]"},
      {"/home/alice/proj/rn/model.onnx", "[path]"},
      {"/data/models/rn/model.onnx", "[path]"},
      {"/data/models/secret/", "[path]"},
      {"~/.config/app/x", "[path]"},
      {"C:\\Users\\bob\\model.onnx", "[path]"},
      {"C:/Users\\bob\\model.onnx", "[path]"},
      {"\\\\server\\share\\dir\\weights.bin", "[path]"},
      {"Load model from /home/alice/models/foo.onnx failed", "Load model from [path]"},
      {"Load C:\\proj\\bin\\m.onnx failed", "Load [path]"},
      {"open D:/data/secret/model.onnx", "open [path]"},
      {"from \\\\server\\share\\dir\\weights.bin done", "from [path]"},
      {"Load C:\\Users\\First Last\\model.onnx failed", "Load [path]"},
      {"C:\\UsErS\\alice\\model.onnx", "[path]"},
      {"C:\\Users/alice\\model.onnx", "[path]"},
      {"/UsErS/alice/model.onnx", "[path]"},
      {"C:\\Users/alice/proj\\model.onnx", "[path]"},
      {"/home//alice/model.onnx", "[path]"},
      {"C:\\Users\\\\alice\\model.onnx", "[path]"},
      {"/home/./alice/model.onnx", "[path]"},
      {"Users\\alice\\model.onnx", "[path]"},
      {"at proj\\alice\\weights\\m.onnx", "at [path]"},
      {"alice/models/phi3.onnx", "[path]"},
      {"at alice/models/phi3.onnx", "at [path]"},
      {"a/b/c", "[path]"},
      {"x/y/z/", "[path]"},
      {"a\\b\\c", "[path]"},
      {"alice\\models\\phi3.onnx", "[path]"},
      {"Load Users\\bob\\m.onnx failed", "Load [path]"},
      {"Users\\First Last\\model.onnx", "[path]"},
      {"Load Users\\First Last\\m.onnx failed", "Load [path]"}};
  for (const auto& [input, expected] : cases) {
    EXPECT_EQ(ScrubStringForTelemetry(input), expected) << input;
  }
}

TEST(TelemetryRedactionTest, RedactsUsernamesEmbeddedInProtocolTokens) {
  for (const char* input : {"input:/home/alice/secret/m.onnx", "file:///home/alice/secret/model.onnx"}) {
    const auto result = ScrubStringForTelemetry(input);
    EXPECT_EQ(result.find("alice"), std::string::npos) << input;
    EXPECT_NE(result.find("[path]"), std::string::npos) << input;
  }
}

TEST(TelemetryRedactionTest, BoundsOutputWithoutSplittingUtf8) {
  EXPECT_EQ(ScrubStringForTelemetry(std::string(kMaxTelemetryStringLength + 1, 'x')),
            std::string(kMaxTelemetryStringLength, 'x'));
  const std::string euro = "\xE2\x82\xAC";
  const std::string prefix(kMaxTelemetryStringLength - 1, 'x');
  EXPECT_EQ(ScrubStringForTelemetry(prefix + euro), prefix);
  const std::string exact = std::string(kMaxTelemetryStringLength - euro.size(), 'x') + euro;
  EXPECT_EQ(ScrubStringForTelemetry(exact), exact);
}

TEST(TelemetryRedactionTest, RedactsAnchorsAcrossTheOutputBoundary) {
  for (const char* path : {"C:\\Users\\First Last\\model", "\\\\server\\share",
                           "~/alice/model", "/home/alice/model", "alice/models/weights",
                           "alice\\models\\weights"}) {
    for (size_t offset = kMaxTelemetryStringLength - 6; offset <= kMaxTelemetryStringLength + 1; ++offset) {
      const std::string prefix = std::string(offset - 1, 'x') + " ";
      const auto result = ScrubStringForTelemetry(prefix + path);
      EXPECT_LE(result.size(), kMaxTelemetryStringLength);
      EXPECT_EQ(result, BoundTelemetryString(prefix + "[path]"));
    }
  }
}

TEST(TelemetryRedactionTest, UnseenSuffixCannotCompleteAnExposedPath) {
  const std::string huge(kMaxTelemetryInputBytes, 'a');
  for (const std::string message : {
           "error alice" + huge + "/models/weights",
           "error alice\\" + huge + "\\weights",
           "error /alice/" + huge + "/weights",
           "error alice\\First Last " + huge + "\\weights",
           "error C" + huge + ":\\Users\\First Last\\weights"}) {
    EXPECT_EQ(ScrubStringForTelemetry(message), "error [path]");
    EXPECT_EQ(ScrubStringForTelemetry(BoundedTelemetryCString(message.c_str())), "error [path]");
  }
  EXPECT_EQ(ScrubStringForTelemetry("error " + huge), "error [path]");
}

}  // namespace Generators::test
