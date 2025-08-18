#include "runtime_settings.h"

namespace Generators {

std::unique_ptr<RuntimeSettings> CreateRuntimeSettings() {
  return std::make_unique<RuntimeSettings>();
}

std::string RuntimeSettings::GenerateConfigOverlay() const {
  constexpr std::string_view webgpu_overlay_pre = R"({
  "model": {
    "decoder": {
      "session_options": {
        "provider_options": [
          {
            "WebGPU": {
              "dawnProcTable": ")";
  constexpr std::string_view webgpu_overlay_post = R"("
            }
          }
        ]
      }
    }
  }
}
)";

  auto it = handles_.find("dawnProcTable");
  if (it != handles_.end()) {
    void* dawn_proc_table_handle = it->second;
    std::string overlay;
    overlay.reserve(webgpu_overlay_pre.size() + webgpu_overlay_post.size() + 20);  // Optional small optimization of buffer size
    overlay += webgpu_overlay_pre;
    overlay += std::to_string((size_t)(dawn_proc_table_handle));
    overlay += webgpu_overlay_post;
    return overlay;
  }

  return {};
}

}  // namespace Generators
