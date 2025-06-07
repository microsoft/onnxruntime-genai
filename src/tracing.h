// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Build with CMake option ENABLE_TRACING=ON to enable tracing.
// To avoid performance overhead, tracing is not enabled by default.

// When tracing is enabled, the trace data will be recorded to a file.
// The trace file path can be specified with the environment variable ORTGENAI_TRACE_FILE_PATH.
// The trace file can be viewed with Perfetto UI (https://ui.perfetto.dev/).

#pragma once

#include <memory>
#include <string_view>

namespace Generators {

// Trace consumer interface.
class TraceSink {
 public:
  virtual void BeginDuration(std::string_view label) = 0;
  virtual void EndDuration() = 0;
  virtual ~TraceSink() = default;
};

// Main tracing class.
class Tracer {
 public:
  Tracer();

  // Begins a traced duration with the given label.
  void BeginDuration(std::string_view label);

  // Ends the traced duration from the most recent call to BeginDuration() in the same thread.
  void EndDuration();

 private:
  Tracer(const Tracer&) = delete;
  Tracer& operator=(const Tracer&) = delete;
  Tracer(Tracer&&) = delete;
  Tracer& operator=(Tracer&&) = delete;

#if defined(ORTGENAI_ENABLE_TRACING)
  std::unique_ptr<TraceSink> sink_;
#endif
};

// Gets the default tracer instance.
Tracer& DefaultTracerInstance();

// Records a traced duration while in scope.
class DurationTrace {
 public:
  [[nodiscard]] DurationTrace(std::string_view label)
      : DurationTrace{DefaultTracerInstance(), label} {
  }

  [[nodiscard]] DurationTrace(Tracer& tracer, std::string_view label)
      : tracer_{tracer} {
    tracer_.BeginDuration(label);
  }

  ~DurationTrace() {
    tracer_.EndDuration();
  }

 private:
  DurationTrace(const DurationTrace&) = delete;
  DurationTrace& operator=(const DurationTrace&) = delete;
  DurationTrace(DurationTrace&&) = delete;
  DurationTrace& operator=(DurationTrace&&) = delete;

  Tracer& tracer_;
};

}  // namespace Generators
