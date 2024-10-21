#include "runtime_settings.h"

namespace Generators {

std::unique_ptr<RuntimeSettings> CreateRuntimeSettings() {
  return std::make_unique<RuntimeSettings>();
}

}  // namespace Generators