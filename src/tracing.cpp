// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tracing.h"

#include <chrono>
#include <fstream>
#include <optional>
#include <sstream>
#include <thread>

namespace Generators {

namespace {
using TracingClock = std::chrono::steady_clock;

class FileTracer : public Tracer {
 public:
  FileTracer(std::string_view file_path)
      : ostream_{std::ofstream{file_path.data()}}, start_{TracingClock::now()} {
    ostream_ << "[";
  }

  ~FileTracer() {
    ostream_ << "]\n";
  }

  void BeginDuration(std::string_view label) {
    LogEvent("B", label);
  }

  void EndDuration() {
    LogEvent("E");
  }

 private:
  void LogEvent(std::string_view phase_type, std::optional<std::string_view> label = std::nullopt) {
    const auto thread_id = std::this_thread::get_id();
    const auto ts = std::chrono::duration_cast<std::chrono::microseconds>(TracingClock::now() - start_);

    std::ostringstream os{};

    if (event_count_ > 0) {
      os << ",\n";
    }

    os << "{";

    if (label.has_value()) {
      os << "\"name\": \"" << *label << "\", ";
    }

    os << "\"cat\": \"perf\", "
       << "\"ph\": \"" << phase_type << "\", "
       << "\"pid\": 0, "
       << "\"tid\": " << thread_id << ", "
       << "\"ts\": " << ts.count()
       << "}";

    ostream_ << os.str();

    ++event_count_;
  }

  std::ofstream ostream_;
  TracingClock::time_point start_;
  size_t event_count_{};
};

}  // namespace

Tracer& DefaultTracerInstance() {
  static auto tracer = FileTracer{"trace.log"};
  return tracer;
}

}  // namespace Generators
