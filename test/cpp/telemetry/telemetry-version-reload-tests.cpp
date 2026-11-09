// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <bit>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>

#if defined(_WIN32)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

namespace {
struct CloseModule {
  void operator()(void* module) const {
#if defined(_WIN32)
    FreeLibrary(static_cast<HMODULE>(module));
#else
    dlclose(module);
#endif
  }
};

bool RunCycle(const std::string& path) {
#if defined(_WIN32)
  std::unique_ptr<void, CloseModule> module{LoadLibraryA(path.c_str())};
#else
  std::unique_ptr<void, CloseModule> module{dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)};
#endif
  if (!module) {
    std::fputs("FAIL: could not load SDK version fixture\n", stderr);
    return false;
  }

  using Getter = const char* (*)();
#if defined(_WIN32)
  const auto getter = std::bit_cast<Getter>(GetProcAddress(static_cast<HMODULE>(module.get()), "TelemetryReloadSdkVersion"));
#else
  const auto getter = std::bit_cast<Getter>(dlsym(module.get(), "TelemetryReloadSdkVersion"));
#endif
  if (!getter) {
    return false;
  }

  const std::string_view version{getter()};
  if (!version.starts_with("EVT-") || version.find("3.10.309.1") == std::string_view::npos ||
      version.data() != getter()) {
    std::fputs("FAIL: unexpected SDK version value or unstable storage\n", stderr);
    return false;
  }

  module.reset();
#if defined(_WIN32)
  const bool unloaded = GetModuleHandleA(path.c_str()) == nullptr;
#else
  std::unique_ptr<void, CloseModule> retained{dlopen(path.c_str(), RTLD_NOW | RTLD_NOLOAD)};
  const bool unloaded = !retained;
#endif
  if (!unloaded) {
    std::fputs("FAIL: SDK version fixture remained loaded\n", stderr);
  }

  return unloaded;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    return 1;
  }

  const std::string path{argv[1]};
  if (!RunCycle(path) || !RunCycle(path)) {
    return 1;
  }

  std::fputs("PASS: two real SDK version unload/reload cycles\n", stderr);
  return 0;
}
