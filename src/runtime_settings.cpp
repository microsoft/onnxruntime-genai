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

  constexpr std::string_view batch_size_overlay = R"({
  "search": {
  "batch_size": )";

  // Handle WebGPU dawnProcTable
  auto webgpu_it = handles_.find("dawnProcTable");
  if (webgpu_it != handles_.end()) {
    void* dawn_proc_table_handle = webgpu_it->second;
    std::string overlay;
    overlay.reserve(webgpu_overlay_pre.size() + webgpu_overlay_post.size() + 20);
    overlay += webgpu_overlay_pre;
    overlay += std::to_string((size_t)(dawn_proc_table_handle));
    overlay += webgpu_overlay_post;
    return overlay;
  }

  // Handle batch_size
  auto batch_size_it = handles_.find("batch_size");
  if (batch_size_it != handles_.end()) {
    int* batch_size_ptr = static_cast<int*>(batch_size_it->second);
    std::string overlay;
    overlay.reserve(batch_size_overlay.size() + 50);
    overlay += batch_size_overlay;
    overlay += std::to_string(*batch_size_ptr);
    overlay += R"(
  }
}
)";
    return overlay;
  }

  return {};
}

}  // namespace Generators