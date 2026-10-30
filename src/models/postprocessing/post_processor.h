#pragma once

#include "generator/generators.h"

namespace Generators {

class PostProcessor {
 public:
  virtual ~PostProcessor() = default;
  virtual void ProcessNext() = 0;
  virtual void Finalize() = 0;
  virtual void Reset() = 0;

 protected:
  explicit PostProcessor(const Generator& generator) : generator_{generator} {}

  const Generator& generator_;
};

}  // namespace Generators