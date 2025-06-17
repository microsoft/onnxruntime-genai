// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tracing.h"

#include <chrono>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>

#include "models/env_utils.h"

namespace Generators {

#if defined(ORTGENAI_ENABLE_TRACING)

namespace {

// Writes trace events to a file in Chrome tracing format.
// See more details about the format here:
// https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
class FileTraceSink : public TraceSink {
 public:
  FileTraceSink(std::string_view file_path)
      : ostream_{std::ofstream{file_path.data()}},
        start_{Clock::now()},
        insert_event_delimiter_{false} {
    ostream_ << "[";
  }

  ~FileTraceSink() {
    ostream_ << "]\n";
  }

  void BeginDuration(std::string_view label) {
    LogEvent("B", label);
  }

  void EndDuration() {
    LogEvent("E");
  }

 private:
  using Clock = std::chrono::steady_clock;

  void LogEvent(std::string_view phase_type, std::optional<std::string_view> label = std::nullopt) {
    const auto thread_id = std::this_thread::get_id();
    const auto ts = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - start_);

    std::ostringstream event{};

    event << "{";

    if (label.has_value()) {
      event << "\"name\": \"" << *label << "\", ";
    }

    event << "\"cat\": \"perf\", "
          << "\"ph\": \"" << phase_type << "\", "
          << "\"pid\": 0, "
          << "\"tid\": " << thread_id << ", "
          << "\"ts\": " << ts.count()
          << "}";

    {
      std::scoped_lock g{output_mutex_};

      // add the delimiter only after writing the first event
      if (insert_event_delimiter_) {
        ostream_ << ",\n";
      } else {
        insert_event_delimiter_ = true;
      }

      ostream_ << event.str();
    }
  }

  std::ofstream ostream_;
  const Clock::time_point start_;
  bool insert_event_delimiter_;

  std::mutex output_mutex_;
};

std::string GetTraceFileName() {
  constexpr const char* kTraceFileEnvironmentVariableName = "ORTGENAI_TRACE_FILE_PATH";
  auto trace_file_name = GetEnv(kTraceFileEnvironmentVariableName);
  if (trace_file_name.empty()) {
    trace_file_name = "ortgenai_trace.log";
  }
  return trace_file_name;
}

}  // namespace

#endif  // defined(ORTGENAI_ENABLE_TRACING)

Tracer::Tracer() {
#if defined(ORTGENAI_ENABLE_TRACING)
  const auto trace_file_name = GetTraceFileName();
  sink_ = std::make_unique<FileTraceSink>(trace_file_name);
#endif
}

void Tracer::BeginDuration(std::string_view label) {
#if defined(ORTGENAI_ENABLE_TRACING)
  sink_->BeginDuration(label);
#else
  static_cast<void>(label);
#endif
}

void Tracer::EndDuration() {
#if defined(ORTGENAI_ENABLE_TRACING)
  sink_->EndDuration();
#endif
}

Tracer& DefaultTracerInstance() {
  static auto tracer = Tracer{};
  return tracer;
}

}  // namespace Generators
