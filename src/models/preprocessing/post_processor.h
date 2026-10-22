#pragma once

#include "generator/generators.h"

namespace Generators {

class PostProcessor {
 public:
  virtual ~PostProcessor() = default;

 protected:
  explicit PostProcessor(const Generator& generator) : generator_{generator} {}

  const Generator& generator_;
};

}  // namespace Generators