// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(__ANDROID__)

#include "android_log_sink.h"

#include <android/log.h>

namespace Generators {

namespace {

constexpr const char* const kLogTag = "onnxruntime-genai";

int LogSeverityToAndroidLogPriority(LogSeverity severity) {
  switch (severity) {
    case Verbose:
      return ANDROID_LOG_VERBOSE;
    case Info:
      return ANDROID_LOG_INFO;
    case Warning:
      return ANDROID_LOG_WARN;
    case Error:
      return ANDROID_LOG_ERROR;
    case Fatal:
      return ANDROID_LOG_FATAL;
    default:
      return ANDROID_LOG_UNKNOWN;
  }
}

class AndroidLogSink : public LogSink {
  void Send(const LogCapture& capture) override {
    const auto priority = LogSeverityToAndroidLogPriority(capture.Severity());
    __android_log_print(priority, kLogTag,
                        "[%s] %s", capture.Label().c_str(), capture.Message().c_str());
  }
};

}  // namespace

std::unique_ptr<LogSink> MakeAndroidLogSink() {
  return std::make_unique<AndroidLogSink>();
}

}  // namespace Generators

#endif  // defined(__ANDROID__)
