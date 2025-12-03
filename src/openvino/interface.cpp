// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "../generators.h"
#include "../search.h"
#include "interface.h"
#include "../cpu/interface.h"
#include "../models/model.h"

namespace Generators {
namespace OpenVINO {

struct InterfaceImpl : DeviceInterface {
  InterfaceImpl() {
  }

  DeviceType GetType() const override { return DeviceType::OpenVINO; }

  void InitOrt(const OrtApi& /*api*/, Ort::Allocator& allocator) override {
    // since we use the CPU interface for allocation (right now), InitOrt should not be getting called.
    assert(false);
  }

  Ort::Allocator& GetAllocator() override {
    return GetCpuInterface()->GetAllocator();
  }

  std::shared_ptr<DeviceBuffer> AllocateBase(size_t size) override {
    return GetCpuInterface()->AllocateBase(size);
  }

  std::shared_ptr<DeviceBuffer> WrapMemoryBase(void* p, size_t size) override {
    return GetCpuInterface()->WrapMemoryBase(p, size);
  }

  std::unique_ptr<Search> CreateGreedy(const GeneratorParams& params) override { return std::make_unique<GreedySearch_Cpu>(params); }
  std::unique_ptr<Search> CreateBeam(const GeneratorParams& params) override { return std::make_unique<BeamSearch_Cpu>(params); }

  void Synchronize() override {}  // Nothing to do
};

}  // namespace OpenVINO

DeviceInterface* GetOpenVINOInterface() {
  static std::unique_ptr<DeviceInterface> g_device = std::make_unique<OpenVINO::InterfaceImpl>();
  return g_device.get();
}

bool IsOpenVINOStatefulModel(const Model& model) {
  if (model.p_device_->GetType() == DeviceType::OpenVINO) {
    const auto& provider_options = model.config_->model.decoder.session_options.provider_options;
    for (auto& po : provider_options) {
      if (po.name == "OpenVINO") {
        const auto& openvino_options = po.options;
        for (auto& option : openvino_options) {
          // For OpenVINO, if session option 'enable_causallm' is set, the session will encapsulate
          // a stateful model, so KVCache will be managed internally.
          if (option.first == "enable_causallm" && option.second == "True") {
            return true;
          }
        }
      }
    }
  }

  return false;
}

static inline std::string GetOVDeviceStringFromOrtDevice(const OrtEpDevice* device_ptr) {
  const OrtKeyValuePairs* keyvals = Ort::api->EpDevice_EpMetadata(device_ptr);
  size_t num_entries;
  const char* const* keys = nullptr;
  const char* const* values = nullptr;
  Ort::api->GetKeyValuePairs(keyvals, &keys, &values, &num_entries);
  for (size_t kvi = 0; kvi < num_entries; kvi++) {
    const std::string key = keys[kvi];
    const std::string val = values[kvi];
    if (key == "ov_device") {
      return val;
    }
  }

  throw std::runtime_error("OrtEpDevice doesn't have ov_device meta field.");
}

static inline const OrtEpDevice* SelectEpDeviceFromProviderOptions(const Generators::Config::ProviderOptions& provider_options) {
  // Get device filtering config
  Config::DeviceFilteringOptions resolved_device_filtering;
  if (provider_options.device_filtering_options.has_value()) {
    resolved_device_filtering = provider_options.device_filtering_options.value();
  }

  std::optional<uint32_t> config_device_id = resolved_device_filtering.hardware_device_id;
  std::optional<uint32_t> config_vendor_id = resolved_device_filtering.hardware_vendor_id;
  std::optional<OrtHardwareDeviceType> config_device_type_enum = resolved_device_filtering.hardware_device_type;

  // Use "device_type" in provider_options exclusively if it's provided
  std::optional<std::string> config_ov_device_type = std::nullopt;

  for (auto& option : provider_options.options) {
    if (option.first == "device_type") {
      config_ov_device_type = option.second;
    }
  }

  // if 'device_type' provider option has been set, this will take precedence over device_id/vendor_id/device_type_enum
  if (config_ov_device_type.has_value()) {
    config_device_id = std::nullopt;
    config_vendor_id = std::nullopt;
    config_device_type_enum = std::nullopt;
  }

  // If config_ov_device_type isn't set, but there also hasn't been set any device_id/vendor_id/device_type_enum.
  // In this case, default ov_device_type to "CPU".
  if (!config_ov_device_type.has_value() &&
      !(config_device_id.has_value() || config_vendor_id.has_value() || config_device_type_enum.has_value())) {
    config_ov_device_type = "CPU";
  }

  size_t num_devices = 0;
  const OrtEpDevice* const* device_ptrs = nullptr;
  Ort::GetEpDevices(&GetOrtEnv(), &device_ptrs, &num_devices);

  const std::string ep_name = "OpenVINOExecutionProvider";
  std::string chosen_ov_device;
  for (size_t i = 0; i < num_devices; ++i) {
    // skip this device if it's not an OpenVINO device.
    if (Ort::api->EpDevice_EpName(device_ptrs[i]) != ep_name)
      continue;

    const OrtHardwareDevice* hardware_device = Ort::api->EpDevice_Device(device_ptrs[i]);
    const uint32_t hardware_device_id = Ort::api->HardwareDevice_DeviceId(hardware_device);
    const uint32_t hardware_vendor_id = Ort::api->HardwareDevice_VendorId(hardware_device);
    const OrtHardwareDeviceType hardware_device_type = Ort::api->HardwareDevice_Type(hardware_device);

    auto check_ov_device_type = [&config_ov_device_type, &provider_options](const OrtEpDevice* device_ptr) -> bool {
      if (!config_ov_device_type.has_value()) {
        return true;
      } else {
        auto meta_ov_device = GetOVDeviceStringFromOrtDevice(device_ptr);
        if (meta_ov_device.find(*config_ov_device_type) != std::string::npos) {
          return true;
        }
        return false;
      }
    };

    bool hardware_device_id_matched = (!config_device_id.has_value()) || config_device_id.value() == hardware_device_id;
    bool hardware_vendor_id_matched = (!config_vendor_id.has_value()) || config_vendor_id.value() == hardware_vendor_id;
    bool hardware_device_type_matched = (!config_device_type_enum.has_value()) ||
                                        config_device_type_enum.value() == hardware_device_type;
    bool hardware_ov_device_type_matched = check_ov_device_type(device_ptrs[i]);
    // Append matched EP device
    if (hardware_device_id_matched &&
        hardware_vendor_id_matched &&
        hardware_device_type_matched &&
        hardware_ov_device_type_matched) {
      return device_ptrs[i];
    }
  }

  return nullptr;
}

static inline void EscapeBackslashes(std::string& s) {
  size_t pos = 0;
  while ((pos = s.find("\\", pos)) != std::string::npos) {
    s.replace(pos, 1, "\\\\");
    pos += 2;
  }
}

static inline std::string MakeCacheDirAbsolute(std::string cache_dir, fs::path config_path) {
  fs::path cache_dir_path(cache_dir);

  // if cache_dir is a relative path, then make it absolute.
  if (cache_dir_path.is_relative()) {
    fs::path abs_cache_dir = config_path / cache_dir_path;
    std::string abs_cache_dir_str = abs_cache_dir.string();
    // convert '\' to '\\'
    EscapeBackslashes(abs_cache_dir_str);
    return abs_cache_dir_str;
  }

  EscapeBackslashes(cache_dir);
  return cache_dir;
}

static inline void ReplaceCommaBrace(std::string& s) {
  const std::string from = ",}";
  const std::string to = "}";

  size_t pos = 0;
  while ((pos = s.find(from, pos)) != std::string::npos) {
    s.replace(pos, from.length(), to);
    // No need to advance pos because replacement is shorter
  }
}

static inline void RemoveAllWhitespace(std::string& s) {
  s.erase(std::remove_if(s.begin(), s.end(), [](unsigned char c) { return std::isspace(c); }), s.end());
}

static inline bool StartsWith(const std::string& str, const std::string& prefix) {
  return str.size() >= prefix.size() &&
         str.compare(0, prefix.size(), prefix) == 0;
}

static inline bool EndsWith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

static inline std::optional<std::string> AddCacheDirToLoadConfig(const std::string& cache_dir,
                                                                 std::optional<std::string> load_config_option,
                                                                 const std::string& ov_device) {
  // convert raw cache_dir path into OpenVINO key/value pair
  std::string cache_dir_option = "\"CACHE_DIR\":\"" + cache_dir + "\"";

  // if load_config was set..
  if (load_config_option.has_value()) {
    // load_config is set. We need to add the cache_dir OV option to the existing load_config.
    auto& load_config_raw = *load_config_option;

    // few sanity checks..
    if (EndsWith(load_config_raw, ".json")) {
      if (g_log.enabled)
        Log("warning", "Unable to merge cache_dir into load_config when it references a .json file");
      return load_config_option;
    }

    if (load_config_raw.find("CACHE_DIR") != std::string::npos) {
      if (g_log.enabled)
        Log("warning", "Unable to merge cache_dir into load_config, as it already defines CACHE_DIR");
      return load_config_option;
    }

    // let's go ahead and try to merge in our load_config
    // First, strip all whitespace to aid in any future pattern matching.
    RemoveAllWhitespace(load_config_raw);

    if (!StartsWith(load_config_raw, "{")) {
      if (g_log.enabled)
        Log("warning", "Expected load_config to begin with '{'");
      return load_config_option;
    }

    // first try to find the start of our device entry. For example, for CPU device, it would look like:
    // "CPU":{
    std::string search_str = "\"" + ov_device + "\":{";
    size_t device_config_pos = load_config_raw.find(search_str);
    if (device_config_pos != std::string::npos) {
      // if it's found, we just want to insert our new CACHE_DIR option at the start of that
      std::string replacement_string = search_str + cache_dir_option + ",";
      load_config_raw.replace(device_config_pos, search_str.length(), replacement_string);
    } else {
      // there doesn't seem to be an entry for this device in the config. So, we'll just add one.
      // Here, we'll find the first occurrence of '{' and replace it with '{"CPU":{"CACHE_DIR":"<cache_path>"},'
      size_t brace_pos = load_config_raw.find("{");
      if (brace_pos != std::string::npos) {
        std::string replacement_string = "{" + search_str + cache_dir_option + "},";
        load_config_raw.replace(brace_pos, 1, replacement_string);
      }
    }

    // last step. In rare cases, it's possible that we added a ',' where we shouldn't have -- resulting in ',}'
    // So replace ',}' with '}'
    ReplaceCommaBrace(load_config_raw);
    return load_config_raw;
  } else {
    // In this case, load_config hasn't been set. So it's pretty easy -- we just create one using
    // the ov_device & cache_dir_option
    load_config_option = "{\"" + ov_device + "\":{" + cache_dir_option + "}}";
  }

  return load_config_option;
}

void OpenVINO_AppendProviderOptions(OrtSessionOptions& session_options,
                                    const Generators::Config& config,
                                    const Generators::Config::ProviderOptions& provider_options) {
  if (provider_options.name != "OpenVINO") {
    throw std::runtime_error("OpenVINO_AppendProviderOptions called with provider_options.name = " + provider_options.name);
  }

#if USE_WINML
  // from the given provider options, select the right OVEP OrtDevice to use.
  auto openvino_ep_device = SelectEpDeviceFromProviderOptions(provider_options);
  if (!openvino_ep_device) {
    throw std::runtime_error("OpenVINO_AppendProviderOptions: Unable to find suitable OpenVINOExecutionProvider OrtEpDevice");
  }

  // get the OpenVINO device string, from the selected device (e.g. "CPU", "GPU", "NPU", etc.)
  auto selected_ov_device = GetOVDeviceStringFromOrtDevice(openvino_ep_device);

  std::vector<const char*> keys, values;
  std::optional<std::string> cache_dir_option;
  std::optional<std::string> load_config_option;
  for (auto& option : provider_options.options) {
    // device type isn't a supported provider option when using SessionOptionsAppendExecutionProvider_V2
    // (It's set via the OrtDevice ptr which we selected above)
    if (option.first == "device_type") {
      continue;
    }

    // For load_config we won't add to keys/vals just yet..
    if (option.first == "load_config") {
      load_config_option = option.second;
      continue;
    }

    // For cache_dir, we will perform some manipulation and pack into load_config,
    // so don't set it either.
    if (option.first == "cache_dir") {
      cache_dir_option = option.second;
      continue;
    }

    keys.emplace_back(option.first.c_str());
    values.emplace_back(option.second.c_str());
  }

  // if cache_dir option is set
  if (cache_dir_option) {
    // make it absolute
    cache_dir_option = MakeCacheDirAbsolute(*cache_dir_option, config.config_path);

    // for SessionOptionsAppendExecutionProvider_V2, cache_dir isn't supported as a provider option,
    // so add it to load_config.
    load_config_option = AddCacheDirToLoadConfig(*cache_dir_option, load_config_option, selected_ov_device);
  }

  if (load_config_option.has_value()) {
    keys.emplace_back("load_config");
    values.emplace_back((*load_config_option).c_str());
  }

  std::vector<const OrtEpDevice*> ep_devices_ptrs = {openvino_ep_device};
  Ort::api->SessionOptionsAppendExecutionProvider_V2(
      &session_options,
      &GetOrtEnv(),
      ep_devices_ptrs.data(), ep_devices_ptrs.size(),
      keys.data(), values.data(), keys.size());

#else
  std::vector<const char*> keys, values;
  std::optional<std::string> cache_dir_option;
  for (auto& option : provider_options.options) {
    // For cache_dir, we will perform some manipulation before setting.
    if (option.first == "cache_dir") {
      cache_dir_option = option.second;
      continue;
    }
    keys.emplace_back(option.first.c_str());
    values.emplace_back(option.second.c_str());
  }

  if (cache_dir_option) {
    cache_dir_option = MakeCacheDirAbsolute(*cache_dir_option, config.config_path);
    keys.emplace_back("cache_dir");
    values.emplace_back((*cache_dir_option).c_str());
  }
  session_options.AppendExecutionProvider(provider_options.name.c_str(), keys.data(), values.data(), keys.size());
#endif
}

}  // namespace Generators
