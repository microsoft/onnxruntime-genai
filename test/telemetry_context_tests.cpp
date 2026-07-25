// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "telemetry/telemetry_context.h"

#include <map>
#include <string>

#include <gtest/gtest.h>

namespace Generators::test {
namespace {

class RecordingSemanticContext {
 public:
  void SetCommonField(const std::string& name, const std::string& value) {
    fields_[name] = value;
  }

  const std::map<std::string, std::string>& Fields() const {
    return fields_;
  }

 private:
  std::map<std::string, std::string> fields_;
};

TEST(TelemetryContextTests, SuppressesUnneededCommonContext) {
  RecordingSemanticContext context;
  TelemetryInternal::SuppressUnneededCommonContext(context);

  ASSERT_EQ(context.Fields().size(), TelemetryInternal::kSuppressedCommonContextFields.size());
  for (const char* field : TelemetryInternal::kSuppressedCommonContextFields) {
    EXPECT_EQ(context.Fields().at(field), "");
  }
}

TEST(TelemetryContextTests, SuppressesNetworkContextAfterProcessInfo) {
  RecordingSemanticContext context;
  TelemetryInternal::SuppressNetworkContext(context);

  ASSERT_EQ(context.Fields().size(), TelemetryInternal::kProcessInfoOnlyNetworkContextFields.size());
  for (const char* field : TelemetryInternal::kProcessInfoOnlyNetworkContextFields) {
    EXPECT_EQ(context.Fields().at(field), "");
  }
}

}  // namespace
}  // namespace Generators::test
