// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string_view>

namespace Generators {

class Tracer {
 public:
  virtual void BeginDuration(std::string_view label) = 0;
  virtual void EndDuration() = 0;
};

Tracer& DefaultTracerInstance();

class DurationTrace {
 public:
  DurationTrace(std::string_view label)
      : DurationTrace{DefaultTracerInstance(), label} {
  }

  DurationTrace(Tracer& tracer, std::string_view label)
      : tracer_{tracer} {
    tracer_.BeginDuration(label);
  }

  ~DurationTrace() {
    tracer_.EndDuration();
  }

 private:
  Tracer& tracer_;
};

}  // namespace Generators
