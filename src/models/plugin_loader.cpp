// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Portions of this file consist of AI generated content.

#include "plugin_loader.h"

#include <stdexcept>
#include <string>

#include "model.h"
#include "plugin_api.h"
#include "controller_host.h"

#if USE_GENAI_PLUGINS
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

namespace Generators {

#if USE_GENAI_PLUGINS

namespace {

// RAII for the loaded plugin module. The handle must outlive every model the plugin produced (the
// model's code, vtable and destructor all live inside this module), so the wrapping shared_ptr<Model>
// deleter captures a shared_ptr<PluginLibrary> to keep it loaded until the last model is freed.
struct PluginLibrary {
#ifdef _WIN32
  explicit PluginLibrary(HMODULE handle) : handle_{handle} {}
  ~PluginLibrary() {
    if (handle_) FreeLibrary(handle_);
  }
  void* GetSymbol(const char* name) { return reinterpret_cast<void*>(::GetProcAddress(handle_, name)); }
  HMODULE handle_{};
#else
  explicit PluginLibrary(void* handle) : handle_{handle} {}
  ~PluginLibrary() {
    if (handle_) dlclose(handle_);
  }
  void* GetSymbol(const char* name) { return ::dlsym(handle_, name); }
  void* handle_{};
#endif
};

std::shared_ptr<PluginLibrary> OpenLibrary(const std::string& library) {
#ifdef _WIN32
  HMODULE handle = LoadLibraryA(library.c_str());
  if (!handle)
    throw std::runtime_error("Failed to load pipeline plugin library '" + library +
                             "' (LoadLibraryA error " + std::to_string(GetLastError()) + ").");
  return std::make_shared<PluginLibrary>(handle);
#else
  void* handle = dlopen(library.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    const char* err = dlerror();
    throw std::runtime_error("Failed to load pipeline plugin library '" + library +
                             "': " + (err ? err : "unknown dlopen error") + ".");
  }
  return std::make_shared<PluginLibrary>(handle);
#endif
}

}  // namespace

std::shared_ptr<Model> LoadPluginPipeline(const Config::Pipeline::Plugin& plugin,
                                          std::unique_ptr<Config> config, OrtEnv& ort_env) {
  if (plugin.library.empty())
    throw std::runtime_error("pipeline.plugin.library is empty; a plugin library path is required.");
  if (plugin.entry_point.empty())
    throw std::runtime_error("pipeline.plugin.entry_point is empty; a plugin entry-point symbol name is required.");

  auto library = OpenLibrary(plugin.library);

  auto entry = reinterpret_cast<OgaCreatePipelineFn>(library->GetSymbol(plugin.entry_point.c_str()));
  if (!entry)
    throw std::runtime_error("Pipeline plugin '" + plugin.library + "' does not export entry point '" +
                             plugin.entry_point + "'.");

  // Ownership of the Config transfers to the plugin on success (the ABI is C, so we hand over a raw
  // pointer). On failure we reclaim it via a unique_ptr so it is freed.
  Config* config_raw = config.release();

  OgaPipelinePluginModel* plugin_model = nullptr;
  OgaPipelinePluginModelDestroyFn destroy = nullptr;
  const int status = entry(&ort_env, config_raw, &plugin_model, &destroy);

  if (status != OgaPipelinePluginStatus_Ok) {
    std::unique_ptr<Config> reclaimed{config_raw};  // plugin did not take ownership on failure
    throw std::runtime_error("Pipeline plugin entry point '" + plugin.entry_point + "' in '" +
                             plugin.library + "' failed with status " + std::to_string(status) + ".");
  }
  if (!plugin_model || !destroy) {
    std::unique_ptr<Config> reclaimed{config_raw};  // success-but-null: ownership is undefined, so reclaim to avoid a leak
    throw std::runtime_error("Pipeline plugin entry point '" + plugin.entry_point + "' in '" +
                             plugin.library + "' reported success but returned a null model or destructor.");
  }

  // Adapt the opaque handle back into a Model on the runtime side. The plugin built this object
  // against this runtime, so the handle is a Generators::Model*; we never reinterpret a foreign type.
  // The custom deleter releases via the plugin and keeps the library loaded for the model's lifetime.
  Model* model = reinterpret_cast<Model*>(plugin_model);
  return std::shared_ptr<Model>(model, [destroy, library](Model* m) {
    destroy(reinterpret_cast<OgaPipelinePluginModel*>(m));
    (void)library;  // captured to keep the plugin module loaded until the model is destroyed
  });
}

std::unique_ptr<ControllerHook> LoadDecodeController(const Config::Pipeline::Controller& controller) {
  if (controller.library.empty())
    throw std::runtime_error("pipeline.controller.library is empty; a controller plugin library path is required.");
  if (controller.entry_point.empty())
    throw std::runtime_error("pipeline.controller.entry_point is empty; a controller entry-point symbol name is required.");

  auto library = OpenLibrary(controller.library);

  auto entry = reinterpret_cast<OgaCreateDecodeControllerFn>(library->GetSymbol(controller.entry_point.c_str()));
  if (!entry)
    throw std::runtime_error("Controller plugin '" + controller.library + "' does not export entry point '" +
                             controller.entry_point + "'.");

  // The library handle must outlive the controller (its code/vtable/destructor live inside the
  // module), so it is captured as the controller hook's keepalive.
  return CreateControllerHook(entry, controller.config, library);
}

#else  // USE_GENAI_PLUGINS

std::shared_ptr<Model> LoadPluginPipeline(const Config::Pipeline::Plugin& plugin,
                                          std::unique_ptr<Config> config, OrtEnv& ort_env) {
  (void)config;
  (void)ort_env;
  throw std::runtime_error(
      "Pipeline plugin support is not enabled in this build (rebuild with USE_GENAI_PLUGINS=ON). "
      "Requested plugin library: '" +
      plugin.library + "'.");
}

std::unique_ptr<ControllerHook> LoadDecodeController(const Config::Pipeline::Controller& controller) {
  throw std::runtime_error(
      "Controller plugin support is not enabled in this build (rebuild with USE_GENAI_PLUGINS=ON). "
      "Requested controller library: '" +
      controller.library + "'.");
}

#endif  // USE_GENAI_PLUGINS

}  // namespace Generators
