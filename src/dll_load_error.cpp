// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#if defined(_WIN32)

#include <Windows.h>
#include <dbghelp.h>
#include <memory>
#include <string>
#pragma comment(lib, "dbghelp.lib")

struct HMODULE_Deleter {
  typedef HMODULE pointer;
  void operator()(HMODULE h) { FreeLibrary(h); }
};

using ModulePtr = std::unique_ptr<HMODULE, HMODULE_Deleter>;

std::string DetermineLoadLibraryError(const char* filename) {
  std::string error("Error loading");

  ModulePtr hModule;  // Here so that filename is valid until the next iteration
  while (filename) {
    error += std::string(" \"") + filename + "\"";

    // We use DONT_RESOLVE_DLL_REFERENCES instead of LOAD_LIBRARY_AS_DATAFILE because the latter will not process the import table
    // and will result in the IMAGE_IMPORT_DESCRIPTOR table names being uninitialized.
    hModule = ModulePtr{LoadLibraryEx(filename, NULL, DONT_RESOLVE_DLL_REFERENCES)};
    if (!hModule) {
      error += " which is missing. (Error " + std::to_string(GetLastError()) + ')';
      return error;
    }

    // Get the address of the Import Directory.
    // NOTE: MappedAsImage must be TRUE here: LoadLibraryEx maps the module as an image (section
    // alignment, not file alignment), so the import descriptors and their Name RVAs are relative to
    // the image base. Passing FALSE makes ImageDirectoryEntryToData apply file-offset math to an
    // image-mapped module and return a bogus pointer, which then reads garbage as the dependency
    // name. (This matches ONNX Runtime's core/platform/windows/dll_load_error.cc.)
    ULONG size{};
    IMAGE_IMPORT_DESCRIPTOR* import_desc = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(ImageDirectoryEntryToData(hModule.get(), TRUE, IMAGE_DIRECTORY_ENTRY_IMPORT, &size));
    if (!import_desc) {
      error += " No import directory found.";  // This is unexpected, and I'm not sure how it could happen but we handle it just in case.
      return error;
    }

    // Iterate through the import descriptors to see which dependent DLL can't load
    filename = nullptr;
    for (; import_desc->Characteristics; import_desc++) {
      char* dll_name = (char*)((BYTE*)(hModule.get()) + import_desc->Name);
      // Try to load the dependent DLL, and if it fails, we loop again with this as the DLL and we'll be one step closer to the missing file.
      ModulePtr hDepModule{LoadLibrary(dll_name)};
      if (!hDepModule) {
        filename = dll_name;
        error += " which depends on";
        break;
      }
    }
  }
  error += " But no dependency issue could be determined.";
  return error;
}

bool CanLoadLibrary(const char* filename) {
  // Fully load the library (runs its DllMain and resolves imports), with the library's own
  // directory searched first for co-located native dependencies (LOAD_WITH_ALTERED_SEARCH_PATH),
  // matching how ONNX Runtime resolves an EP's private dependencies. If the provider or any native
  // dependency fails to initialize (e.g. an NVIDIA runtime on a machine whose GPU was removed), the
  // load fails here and the caller can skip the EP instead of handing it to ORT.
  HMODULE hModule = LoadLibraryExA(filename, nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!hModule)
    return false;
  FreeLibrary(hModule);
  return true;
}

#else

// Non-Windows (Linux / macOS / Android): ONNX Runtime has no import-table dependency walk on these
// platforms -- its loader (core/platform/posix/env.cc LoadDynamicLibrary) just uses dlopen() /
// dlerror(), and the dynamic linker's error already names the missing dependency (for example
// "libcuda.so.1: cannot open shared object file: No such file or directory"). We reuse that same
// mechanism here so the pre-flight and its error message stay consistent with ORT.
#include <dlfcn.h>
#include <string>

std::string DetermineLoadLibraryError(const char* filename) {
  dlerror();  // clear any stale error
  void* handle = dlopen(filename, RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    const char* err = dlerror();
    return std::string("Error loading \"") + (filename ? filename : "") + "\": " +
           (err ? err : "unknown error");
  }
  dlclose(handle);  // Loaded successfully on this attempt (not expected on the error path).
  return std::string("Error loading \"") + (filename ? filename : "") + "\".";
}

bool CanLoadLibrary(const char* filename) {
  // Pre-flight the load the same way ORT's POSIX loader does: RTLD_NOW resolves all symbols, so a
  // missing native dependency (e.g. an absent CUDA runtime) fails here. Note: on Linux a hardware
  // EP whose GPU was removed may still dlopen successfully and only fail at first use; that case is
  // surfaced at model-load time rather than here.
  dlerror();
  void* handle = dlopen(filename, RTLD_NOW | RTLD_LOCAL);
  if (!handle)
    return false;
  dlclose(handle);
  return true;
}

#endif
