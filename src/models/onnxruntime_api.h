// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Summary: The Ort C++ API is a header only wrapper around the Ort C API.
//
// This header turns the C API opaque types into C++ objects with methods and destructors.
// For the methods, values are returned directly and exceptions are thrown in case of errors. Standard C++ practices
// like std::unique_ptr are used to show ownership.
//
// As this just gives a definition of the C opaque types, it is simple to interop with the C API as the types match.
// Interop can be useful for advanced usage like a custom OrtAllocator.

/*
Technical explanation on how it works:

The C header onnxruntime_c_api.h declares a type and some functions:
struct OrtSession;

This type is not defined, only declared, as it is opaque.

This C++ header (this file) defines that same type:

struct OrtSession {

  static std::unique_ptr<OrtSession> Create(...); // Calls C API CreateSession(...) to construct a new OrtSession

  size_t GetInputCount() const {
    size_t out;
    Ort::ThrowOnError(Ort::api->SessionGetInputCount(this, &out)); // Calls C API here, and throws a const OrtStatus* on error
    return out;
  }

  // This enables delete to work as expected, so an OrtSession* can be held in a std::unique_ptr or deleted directly.
  static void operator delete(void* p) { Ort::api->ReleaseSession(reinterpret_cast<OrtSession*>(p)); }

  Ort::Abstract make_abstract; // Ensures this type cannot be directly constructed or copied by value
};

It is still an opaque type, there is no way to construct it by value, but it now acts like a C++ type in useful ways.
The methods on it wrap the equivalent C API methods for, and deleting a pointer to this type will release the data
behind it (equivalent to calling the OrtRelease* method).

Example usage:

NOTE: The std::unique_ptr<t>s could just be 'auto', it's expanded for clarity

Ort::InitApi();
std::unique_ptr<OrtEnv> p_env=OrtEnv::Create(ORT_LOGGING_LEVEL_WARNING, "test");

std::unique_ptr<OrtMemoryInfo> p_memory_info = OrtMemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
std::unique_ptr<OrtValue> p_input_tensor = OrtValue::CreateTensor<float>(*p_memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
std::unique_ptr<OrtValue> p_output_tensor = OrtValue::CreateTensor<float>(*p_memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());

std::unique_ptr<OrtSession> p_session_ = OrtSession::Create(*p_env, L"model.onnx", nullptr)};

const char *input_names[] = { "Example Input Name" };
const char *output_names[] = { "Example Output Name" };

OrtValue* inputs[] = { p_input_tensor.get() };
OrtValue* outputs[] = { p_output_tensor.get() };

p_session_->Run(nullptr, input_names, inputs, std::size(inputs), output_names, outputs, std::size(outputs));

*/

#pragma once
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <array>

#include "onnxruntime_c_api.h"
#include "../span.h"
#include "../logging.h"
#include "env_utils.h"

#if defined(__linux__)
#include <dlfcn.h>

#if defined(__ANDROID__)
#include <android/log.h>
#define TAG "GenAI"

#define LOG_DEBUG(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOG_INFO(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOG_WARN(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOG_FATAL(...) __android_log_print(ANDROID_LOG_FATAL, TAG, __VA_ARGS__)
#endif

#elif defined(__APPLE__)
#include "TargetConditionals.h"
#if TARGET_OS_OSX && _ORT_GENAI_USE_DLOPEN
#define MACOS_USE_DLOPEN
#include <dlfcn.h>
#endif
#endif

#ifndef PATH_MAX
#define PATH_MAX (4096)
#endif

#if !defined(__ANDROID__)
#define LOG_WHEN_ENABLED(LOG_FUNC) \
  if (Generators::g_log.enabled && Generators::g_log.ort_lib) LOG_FUNC

#define LOG_DEBUG(...) LOG_WHEN_ENABLED(Generators::Log("debug", __VA_ARGS__))
#define LOG_INFO(...) LOG_WHEN_ENABLED(Generators::Log("info", __VA_ARGS__))
#define LOG_WARN(...) LOG_WHEN_ENABLED(Generators::Log("warning", __VA_ARGS__))
#define LOG_ERROR(...) LOG_WHEN_ENABLED(Generators::Log("error", __VA_ARGS__))
#define LOG_FATAL(...) LOG_WHEN_ENABLED(Generators::Log("fatal", __VA_ARGS__))
#endif

/** \brief Free functions and a few helpers are defined inside this namespace. Otherwise all types are the C API types
 *
 */
namespace Ort {

using OrtApiBaseFn = const OrtApiBase* (*)(void);

/// Before using this C++ wrapper API, you MUST call Ort::InitApi to set the below 'api' variable
inline const OrtApi* api{};

#if defined(__linux__) || defined(MACOS_USE_DLOPEN)
inline std::string GetCurrentModuleDir() {
  Dl_info dl_info;
  dladdr((void*)GetCurrentModuleDir, &dl_info);
  std::string module_name(dl_info.dli_fname);
  std::string module_directory{};

  const size_t last_slash_idx = module_name.rfind('/');
  if (std::string::npos != last_slash_idx) {
    module_directory = module_name.substr(0, last_slash_idx);
  }
  return module_directory;
}

inline void* LoadDynamicLibraryIfExists(const std::string& path) {
  LOG_INFO("Attempting to dlopen %s", path.c_str());
  void* ort_lib_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (ort_lib_handle == nullptr) {
    char* err = dlerror();
    LOG_WARN("Error while dlopen: %s", (err != nullptr ? err : "Unknown"));
    if (path.front() != '/') {
      // If not absolute path, try search for current dir
      std::string current_module_dir = GetCurrentModuleDir();
      std::string local_path{current_module_dir + "/" + path};
      LOG_INFO("Attempting to dlopen %s", local_path.c_str());
      ort_lib_handle = dlopen(local_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    }
  }
  if (ort_lib_handle) {
#if !defined(__ANDROID__) && !defined(__APPLE__)  // RTLD_DI_ORIGIN not available on Android & Darwin
    char pathname[PATH_MAX];
    dlinfo((void*)ort_lib_handle, RTLD_DI_ORIGIN, &pathname);
    LOG_INFO("Loaded native library at %s", pathname);
#endif
  } else {
    char* err = dlerror();
    LOG_WARN("Error while dlopen: %s", (err != nullptr ? err : "Unknown"));
  }
  return ort_lib_handle;
}

inline void InitApiWithDynamicFn(OrtApiBaseFn ort_api_base_fn) {
  if (ort_api_base_fn == nullptr) {
    throw std::runtime_error("OrtGetApiBase not found");
  }

  const OrtApiBase* ort_api_base = ort_api_base_fn();
  if (ort_api_base == nullptr) {
    throw std::runtime_error("OrtGetApiBase() returned nullptr");
  }

  // loop from the ORT version GenAI was built with, down to the minimum ORT version we require.
  // as long as the libonnxruntime.so we loaded supports one of those we're good.
  constexpr int genai_min_ort_api_version = 18;  // GenAI was first released around the time of ORT 1.18 so use that
  for (int i = ORT_API_VERSION; i >= genai_min_ort_api_version; --i) {
    api = ort_api_base->GetApi(i);
    if (api) {
      LOG_INFO("ORT API Version %d was found.", i);
      break;
    }
  }

  if (!api) {
    LOG_WARN("The loaded library did not have an ORT API version between %d and %d.",
             ORT_API_VERSION, genai_min_ort_api_version);
    throw std::runtime_error("Failed to load onnxruntime. Please make sure you installed the correct version");
  }
}
#endif

inline void InitApi() {
  if (api) {
    // api was already set.
    return;
  }

  bool ort_lib = false;
  Generators::GetEnvironmentVariable("ORTGENAI_LOG_ORT_LIB", ort_lib);
  if (ort_lib) {
    Generators::SetLogBool("enabled", true);
    Generators::SetLogBool("ort_lib", true);
  }

#if defined(__linux__) || defined(MACOS_USE_DLOPEN)
  // If the GenAI library links against the onnxruntime library, it will have a dependency on a specific
  // version of OrtGetApiBase.
  //
  // e.g. nm -D libonnxruntime-genai.so built this way shows it requires version 1.19.0 of OrtGetApiBase:
  //  U OrtGetApiBase @VERS_1.19.0
  //
  //
  // If you referenced those two packages in an app, and libonnxruntime-genai.so had a dependency on libonnxruntime.so
  // for OrtGetApiBase, and the ORT versions don't match, you will get a cryptic runtime error like:
  //   java.lang.UnsatisfiedLinkError: dlopen failed: cannot locate symbol "OrtGetApiBase" referenced by
  //                                   "/data/app/<appname>/base.apk!/lib/x86_64/libonnxruntime-genai.so"
  //
  // In order to be flexible we need to add complexity here by:
  //   - don't call OrtGetApiBase directly so libonnxruntime-genai.so does not have a hard dependency
  //     on libonnxruntime.so for the symbol.
  //   - use dlopen/dysm to manually load the onnxruntime library and get the OrtGetApiBase function pointer.
  //     - requires the Android app has referenced both the GenAI and ORT Android packages so the library is available.
  //   - iterate between the current ORT version we're aware of, and a minimum required version, so that we work with
  //     any libonnxruntime.so that supports one of those versions.
  //

  void* ort_lib_handle = nullptr;
  const char* ort_lib_path = std::getenv("ORT_LIB_PATH");
  if (ort_lib_path) {
    ort_lib_handle = LoadDynamicLibraryIfExists(ort_lib_path);
  }

#if defined(__linux__)
  if (ort_lib_handle == nullptr) {
    // For Android and NuGet Linux package, the file name is libonnxruntime.so
    // "libonnxruntime4j_jni.so" is also an option on Android if we have issues
    ort_lib_handle = LoadDynamicLibraryIfExists("libonnxruntime.so");
  }

  if (ort_lib_handle == nullptr) {
    // On Linux it can also be `libonnxruntime.so.1`. See: https://github.com/microsoft/onnxruntime/pull/21339
    ort_lib_handle = LoadDynamicLibraryIfExists("libonnxruntime.so.1");
  }
#endif

#if defined(MACOS_USE_DLOPEN)
  if (ort_lib_handle == nullptr) {
    ort_lib_handle = LoadDynamicLibraryIfExists("libonnxruntime.dylib");
  }
#endif

  if (ort_lib_handle == nullptr) {
    throw std::runtime_error(std::string("Failed to load onnxruntime. Set ORTGENAI_LOG_ORT_LIB envvar to enable detailed logging."));
  }

  OrtApiBaseFn ort_api_base_fn = (OrtApiBaseFn)dlsym(ort_lib_handle, "OrtGetApiBase");
  if (ort_api_base_fn == nullptr) {
    char* err = dlerror();
    throw std::runtime_error(std::string("Failed to load symbol OrtGetApiBase: ") + (err != nullptr ? err : "Unknown"));
  }

  InitApiWithDynamicFn(ort_api_base_fn);
#else   // defined(__linux__) || defined(MACOS_USE_DLOPEN)
  api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (!api)
    throw std::runtime_error("Onnxruntime is installed but is too old, please install a newer version");
#endif  // defined(__linux__) || defined(MACOS_USE_DLOPEN)
}

/** \brief All C++ methods that can fail will throw an exception of this type
 *
 * If <tt>ORT_NO_EXCEPTIONS</tt> is defined, then any error will result in a call to abort()
 */
struct Exception : std::exception {
  Exception(const Exception& e);
  Exception(std::unique_ptr<OrtStatus>&& ort_status) : ort_status_{std::move(ort_status)} {}

  OrtErrorCode GetOrtErrorCode() const;
  const char* what() const noexcept override;

 private:
  std::unique_ptr<OrtStatus> ort_status_;
};

/// This is a C++ wrapper for OrtApi::GetAvailableProviders() and returns a vector of strings representing the available execution providers.
std::vector<std::string> GetAvailableProviders();

inline void SetCurrentGpuDeviceId(int device_id);
inline int GetCurrentGpuDeviceId();

/** \brief IEEE 754 half-precision floating point data type
 * \details It is necessary for type dispatching to make use of C++ API
 * The type is implicitly convertible to/from uint16_t.
 * The size of the structure should align with uint16_t and one can freely cast
 * uint16_t buffers to/from Ort::Float16_t to feed and retrieve data.
 *
 * Generally, you can feed any of your types as float16/blfoat16 data to create a tensor
 * on top of it, providing it can form a continuous buffer with 16-bit elements with no padding.
 * And you can also feed a array of uint16_t elements directly. For example,
 *
 * \code{.unparsed}
 * uint16_t values[] = { 15360, 16384, 16896, 17408, 17664};
 * constexpr size_t values_length = sizeof(values) / sizeof(values[0]);
 * std::vector<int64_t> dims = {values_length};  // one dimensional example
 * Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
 * // Note we are passing bytes count in this api, not number of elements -> sizeof(values)
 * auto float16_tensor = Ort::Value::CreateTensor(info, values, sizeof(values),
 *                                                dims.data(), dims.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16);
 * \endcode
 *
 * Here is another example, a little bit more elaborate. Let's assume that you use your own float16 type and you want to use
 * a templated version of the API above so the type is automatically set based on your type. You will need to supply an extra
 * template specialization.
 *
 * \code{.unparsed}
 * namespace yours { struct half {}; } // assume this is your type, define this:
 * namespace Ort {
 * template<>
 * struct TypeToTensorType<yours::half> { static constexpr ONNXTensorElementDataType type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16; };
 * } //namespace Ort
 *
 * std::vector<yours::half> values;
 * std::vector<int64_t> dims = {values.size()}; // one dimensional example
 * Ort::MemoryInfo info("Cpu", OrtDeviceAllocator, 0, OrtMemTypeDefault);
 * // Here we are passing element count -> values.size()
 * auto float16_tensor = Ort::Value::CreateTensor<yours::half>(info, values.data(), values.size(), dims.data(), dims.size());
 *
 *  \endcode
 */
struct Float16_t {
  uint16_t value{};
  constexpr Float16_t() = default;
  constexpr Float16_t(uint16_t v) noexcept : value(v) {}
  constexpr operator uint16_t() const noexcept { return value; }
  constexpr bool operator==(const Float16_t& rhs) const noexcept { return value == rhs.value; };
  constexpr bool operator!=(const Float16_t& rhs) const noexcept { return value != rhs.value; };
};

static_assert(sizeof(Float16_t) == sizeof(uint16_t), "Sizes must match");

/** \brief bfloat16 (Brain Floating Point) data type
 * \details It is necessary for type dispatching to make use of C++ API
 * The type is implicitly convertible to/from uint16_t.
 * The size of the structure should align with uint16_t and one can freely cast
 * uint16_t buffers to/from Ort::BFloat16_t to feed and retrieve data.
 *
 * See also code examples for Float16_t above.
 */
struct BFloat16_t {
  uint16_t value{};
  constexpr BFloat16_t() = default;
  constexpr BFloat16_t(uint16_t v) noexcept : value(v) {}
  constexpr operator uint16_t() const noexcept { return value; }
  constexpr bool operator==(const BFloat16_t& rhs) const noexcept { return value == rhs.value; };
  constexpr bool operator!=(const BFloat16_t& rhs) const noexcept { return value != rhs.value; };
};

static_assert(sizeof(BFloat16_t) == sizeof(uint16_t), "Sizes must match");

// This is added as a member variable in the wrapped types to prevent accidental construction/copying
// Since the wrapped types are never instantiated by value, this member doesn't really exist. The types are still opaque.
struct Abstract {
  Abstract() = delete;
  Abstract(const Abstract&) = delete;
  void operator=(const Abstract&) = delete;
};

/** \brief Wrapper around ::OrtAllocator
 *
 * Defined inside the Ort namespace because 'OrtAllocator' is already defined as a struct in the C header (just without methods)
 */
struct Allocator : OrtAllocator {
  static Allocator& GetWithDefaultOptions();  ///< ::OrtAllocator default instance that is owned by Onnxruntime
  static std::unique_ptr<Allocator> Create(const OrtSession& session, const OrtMemoryInfo& memory_info);

  void* Alloc(size_t size);
  void Free(void* p);
  const OrtMemoryInfo& GetInfo() const;

  static void operator delete(void* p) { Ort::api->ReleaseAllocator(reinterpret_cast<OrtAllocator*>(p)); }
  Ort::Abstract make_abstract;
};

struct AllocatorDeleter {
  AllocatorDeleter() = default;
  explicit AllocatorDeleter(OrtAllocator* allocator)
      : allocator_(allocator) {}

  void operator()(void* p) const {
    if (allocator_)
      allocator_->Free(allocator_, p);
  }

 private:
  OrtAllocator* allocator_{};
};

template <typename T>
using IAllocatorUniquePtr = std::unique_ptr<T, AllocatorDeleter>;

template <typename TAlloc>
std::span<TAlloc> Allocate(OrtAllocator& allocator,
                           size_t size,
                           IAllocatorUniquePtr<TAlloc>& unique_ptr) {
  assert(!unique_ptr.get());  // Ensure pointer is empty, to avoid accidentally passing the wrong pointer and overwriting things
  unique_ptr = IAllocatorUniquePtr<TAlloc>(reinterpret_cast<TAlloc*>(allocator.Alloc(&allocator, size * sizeof(TAlloc))), AllocatorDeleter(&allocator));
  return std::span(unique_ptr.get(), size);
}

}  // namespace Ort

/** \brief The Status that holds ownership of OrtStatus received from C API
 *  Use it to safely destroy OrtStatus* returned from the C API. Use appropriate
 *  constructors to construct an instance of a Status object from exceptions.
 */
struct OrtStatus {
  static std::unique_ptr<OrtStatus> Create(OrtErrorCode code, const std::string& what);

  std::string GetErrorMessage() const;
  OrtErrorCode GetErrorCode() const;

  static void operator delete(void* p) { Ort::api->ReleaseStatus(reinterpret_cast<OrtStatus*>(p)); }
  Ort::Abstract make_abstract;
};

/** \brief The Env (Environment)
 *
 * The Env holds the logging state used by all other objects.
 * <b>Note:</b> One Env must be created before using any other Onnxruntime functionality
 */
struct OrtEnv {
  /// \brief Wraps OrtApi::CreateEnv
  static std::unique_ptr<OrtEnv> Create(OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING, _In_ const char* logid = "");

  /// \brief Wraps OrtApi::CreateEnvWithCustomLogger
  static std::unique_ptr<OrtEnv> Create(OrtLoggingLevel logging_level, const char* logid, OrtLoggingFunction logging_function, void* logger_param);

  /// \brief Wraps OrtApi::CreateEnvWithGlobalThreadPools
  static std::unique_ptr<OrtEnv> Create(const OrtThreadingOptions* tp_options, OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING, _In_ const char* logid = "");

  /// \brief Wraps OrtApi::CreateEnvWithCustomLoggerAndGlobalThreadPools
  static std::unique_ptr<OrtEnv> Create(const OrtThreadingOptions* tp_options, OrtLoggingFunction logging_function, void* logger_param,
                                        OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING, _In_ const char* logid = "");

  OrtEnv& EnableTelemetryEvents();   ///< Wraps OrtApi::EnableTelemetryEvents
  OrtEnv& DisableTelemetryEvents();  ///< Wraps OrtApi::DisableTelemetryEvents

  OrtEnv& CreateAndRegisterAllocator(const OrtMemoryInfo& mem_info, const OrtArenaCfg& arena_cfg);  ///< Wraps OrtApi::CreateAndRegisterAllocator

  static void operator delete(void* p) { Ort::api->ReleaseEnv(reinterpret_cast<OrtEnv*>(p)); }
  Ort::Abstract make_abstract;
};

/** \brief Custom Op Domain
 *
 */
struct OrtThreadingOptions {
  /// \brief Wraps OrtApi::CreateThreadingOptions
  static std::unique_ptr<OrtThreadingOptions> Create();

  void SetGlobalIntraOpNumThreads(int intra_op_num_threads = 0 /* 0 = default thread count */);
  void SetGlobalInterOpNumThreads(int inter_op_num_threads = 0 /* 0 = default thread count */);
  void SetGlobalSpinControl(bool allow_spinning);
  void SetGlobalDenormalAsZero();

  void SetGlobalCustomCreateThreadFn(OrtCustomCreateThreadFn ort_custom_create_thread_fn);
  void SetGlobalCustomThreadCreationOptions(void* ort_custom_thread_creation_options);
  void SetGlobalCustomJoinThreadFn(OrtCustomJoinThreadFn ort_custom_join_thread_fn);

  static void operator delete(void* p) { Ort::api->ReleaseThreadingOptions(reinterpret_cast<OrtThreadingOptions*>(p)); }
  Ort::Abstract make_abstract;
};

/** \brief Custom Op Domain
 *
 */
struct OrtCustomOpDomain {
  /// \brief Wraps OrtApi::CreateCustomOpDomain
  static std::unique_ptr<OrtCustomOpDomain> Create(const char* domain);

  // This does not take ownership of the op, simply registers it.
  void Add(const OrtCustomOp& op);  ///< Wraps CustomOpDomain_Add

  static void operator delete(void* p) { Ort::api->ReleaseCustomOpDomain(reinterpret_cast<OrtCustomOpDomain*>(p)); }
  Ort::Abstract make_abstract;
};

/** \brief RunOptions
 *
 */
struct OrtRunOptions {
  static std::unique_ptr<OrtRunOptions> Create();  ///< Wraps OrtApi::CreateRunOptions

  OrtRunOptions& SetRunLogVerbosityLevel(int);  ///< Wraps OrtApi::RunOptionsSetRunLogVerbosityLevel
  int GetRunLogVerbosityLevel() const;          ///< Wraps OrtApi::RunOptionsGetRunLogVerbosityLevel

  OrtRunOptions& SetRunLogSeverityLevel(int);  ///< Wraps OrtApi::RunOptionsSetRunLogSeverityLevel
  int GetRunLogSeverityLevel() const;          ///< Wraps OrtApi::RunOptionsGetRunLogSeverityLevel

  OrtRunOptions& SetRunTag(const char* run_tag);  ///< wraps OrtApi::RunOptionsSetRunTag
  const char* GetRunTag() const;                  ///< Wraps OrtApi::RunOptionsGetRunTag

  OrtRunOptions& AddConfigEntry(const char* config_key, const char* config_value);  ///< Wraps OrtApi::AddRunConfigEntry

  /** \brief Terminates all currently executing Session::Run calls that were made using this RunOptions instance
   *
   * If a currently executing session needs to be force terminated, this can be called from another thread to force it to fail with an error
   * Wraps OrtApi::RunOptionsSetTerminate
   */
  OrtRunOptions& SetTerminate();

  /** \brief Clears the terminate flag so this RunOptions instance can be used in a new Session::Run call without it instantly terminating
   *
   * Wraps OrtApi::RunOptionsUnsetTerminate
   */
  OrtRunOptions& UnsetTerminate();

  OrtRunOptions& AddActiveLoraAdapter(const OrtLoraAdapter& adapter);  ///< Wraps OrtApi::RunOptionsSetActiveLoraAdapter

  static void operator delete(void* p) { Ort::api->ReleaseRunOptions(reinterpret_cast<OrtRunOptions*>(p)); }
  Ort::Abstract make_abstract;
};

struct OrtCUDAProviderOptionsV2 {
  static std::unique_ptr<OrtCUDAProviderOptionsV2> Create();

  void Update(const char* const* keys, const char* const* values, size_t count);
  void UpdateValue(const char* key, void* value);

  static void operator delete(void* p) { Ort::api->ReleaseCUDAProviderOptions(reinterpret_cast<OrtCUDAProviderOptionsV2*>(p)); }
  Ort::Abstract make_abstract;
};

/** \brief Options object used when creating a new Session object
 *
 * Wraps ::OrtSessionOptions object and methods
 */
struct OrtSessionOptions {
  static std::unique_ptr<OrtSessionOptions> Create();  ///< Creates a new OrtSessionOptions. Wraps OrtApi::CreateSessionOptions
  std::unique_ptr<OrtSessionOptions> Clone() const;    ///< Creates and returns a copy of this SessionOptions object. Wraps OrtApi::CloneSessionOptions

  OrtSessionOptions& SetIntraOpNumThreads(int intra_op_num_threads);                              ///< Wraps OrtApi::SetIntraOpNumThreads
  OrtSessionOptions& SetInterOpNumThreads(int inter_op_num_threads);                              ///< Wraps OrtApi::SetInterOpNumThreads
  OrtSessionOptions& SetGraphOptimizationLevel(GraphOptimizationLevel graph_optimization_level);  ///< Wraps OrtApi::SetSessionGraphOptimizationLevel

  OrtSessionOptions& EnableCpuMemArena();   ///< Wraps OrtApi::EnableCpuMemArena
  OrtSessionOptions& DisableCpuMemArena();  ///< Wraps OrtApi::DisableCpuMemArena

  OrtSessionOptions& EnableCpuEpFallback();
  OrtSessionOptions& DisableCpuEpFallback();

  OrtSessionOptions& EnableQuantQdq();
  OrtSessionOptions& DisableQuantQdq();

  OrtSessionOptions& EnableQuantQdqCleanup();
  OrtSessionOptions& DisableQuantQdqCleanup();

  OrtSessionOptions& SetEpContextEnable();
  OrtSessionOptions& SetEpContextEmbedMode(const char* mode);
  OrtSessionOptions& SetEpContextFilePath(const char* file_path);

  OrtSessionOptions& SetOptimizedModelFilePath(const ORTCHAR_T* optimized_model_file);  ///< Wraps OrtApi::SetOptimizedModelFilePath

  OrtSessionOptions& EnableProfiling(const ORTCHAR_T* profile_file_prefix);  ///< Wraps OrtApi::EnableProfiling
  OrtSessionOptions& DisableProfiling();                                     ///< Wraps OrtApi::DisableProfiling

  OrtSessionOptions& EnableOrtCustomOps();  ///< Wraps OrtApi::EnableOrtCustomOps

  OrtSessionOptions& EnableMemPattern();   ///< Wraps OrtApi::EnableMemPattern
  OrtSessionOptions& DisableMemPattern();  ///< Wraps OrtApi::DisableMemPattern

  OrtSessionOptions& SetExecutionMode(ExecutionMode execution_mode);  ///< Wraps OrtApi::SetSessionExecutionMode

  OrtSessionOptions& SetLogId(const char* logid);     ///< Wraps OrtApi::SetSessionLogId
  OrtSessionOptions& SetLogSeverityLevel(int level);  ///< Wraps OrtApi::SetSessionLogSeverityLevel

  OrtSessionOptions& Add(OrtCustomOpDomain& custom_op_domain);  ///< Wraps OrtApi::AddCustomOpDomain

  OrtSessionOptions& DisablePerSessionThreads();  ///< Wraps OrtApi::DisablePerSessionThreads

  OrtSessionOptions& AddConfigEntry(const char* config_key, const char* config_value);                                                          ///< Wraps OrtApi::AddSessionConfigEntry
  OrtSessionOptions& AddInitializer(const char* name, const OrtValue& ort_val);                                                                 ///< Wraps OrtApi::AddInitializer
  OrtSessionOptions& AddExternalInitializers(const std::vector<std::string>& names, const std::vector<std::unique_ptr<OrtValue>>& ort_values);  ///< Wraps OrtApi::AddExternalInitializers

  OrtSessionOptions& AppendExecutionProvider_CUDA(const OrtCUDAProviderOptions& provider_options);               ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_CUDA
  OrtSessionOptions& AppendExecutionProvider_CUDA_V2(const OrtCUDAProviderOptionsV2& provider_options);          ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_CUDA_V2
  OrtSessionOptions& AppendExecutionProvider_ROCM(const OrtROCMProviderOptions& provider_options);               ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_ROCM
  OrtSessionOptions& AppendExecutionProvider_OpenVINO(const OrtOpenVINOProviderOptions& provider_options);       ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_OpenVINO
  OrtSessionOptions& AppendExecutionProvider_TensorRT(const OrtTensorRTProviderOptions& provider_options);       ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_TensorRT
  OrtSessionOptions& AppendExecutionProvider_TensorRT_V2(const OrtTensorRTProviderOptionsV2& provider_options);  ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_TensorRT
  OrtSessionOptions& AppendExecutionProvider_MIGraphX(const OrtMIGraphXProviderOptions& provider_options);       ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_MIGraphX
  ///< Wraps OrtApi::SessionOptionsAppendExecutionProvider_CANN
  OrtSessionOptions& AppendExecutionProvider_CANN(const OrtCANNProviderOptions& provider_options);
  /// Wraps OrtApi::SessionOptionsAppendExecutionProvider. Currently supports SNPE and XNNPACK.
  OrtSessionOptions& AppendExecutionProvider(const std::string& provider_name,
                                             const std::unordered_map<std::string, std::string>& provider_options = {});

  OrtSessionOptions& SetCustomCreateThreadFn(OrtCustomCreateThreadFn ort_custom_create_thread_fn);  ///< Wraps OrtApi::SessionOptionsSetCustomCreateThreadFn
  OrtSessionOptions& SetCustomThreadCreationOptions(void* ort_custom_thread_creation_options);      ///< Wraps OrtApi::SessionOptionsSetCustomThreadCreationOptions
  OrtSessionOptions& SetCustomJoinThreadFn(OrtCustomJoinThreadFn ort_custom_join_thread_fn);        ///< Wraps OrtApi::SessionOptionsSetCustomJoinThreadFn

  static void operator delete(void* p) { Ort::api->ReleaseSessionOptions(reinterpret_cast<OrtSessionOptions*>(p)); }
  Ort::Abstract make_abstract;
};

/** \brief Wrapper around ::OrtModelMetadata
 *
 */
struct OrtModelMetadata {
  /** \brief Returns a copy of the producer name.
   */
  std::string GetProducerName() const;  ///< Wraps OrtApi::ModelMetadataGetProducerName

  /** \brief Returns a copy of the graph name.
   */
  std::string GetGraphName() const;  ///< Wraps OrtApi::ModelMetadataGetGraphName

  /** \brief Returns a copy of the domain name.
   */
  std::string GetDomain() const;  ///< Wraps OrtApi::ModelMetadataGetDomain

  /** \brief Returns a copy of the description.
   */
  std::string GetDescription() const;  ///< Wraps OrtApi::ModelMetadataGetDescription

  /** \brief Returns a copy of the graph description.
   */
  std::string GetGraphDescription() const;  ///< Wraps OrtApi::ModelMetadataGetGraphDescription

  /** \brief Returns a vector of copies of the custom metadata keys.
   */
  std::vector<std::string> GetCustomMetadataMapKeysAllocated() const;  ///< Wraps OrtApi::ModelMetadataGetCustomMetadataMapKeys

  /** \brief Looks up a value by a key in the Custom Metadata map
   *
   * \param key zero terminated string key to lookup
   * \return Looked up value, may be an empty string if key is not found.
   */
  std::string LookupCustomMetadataMap(const char* key) const;  ///< Wraps OrtApi::ModelMetadataLookupCustomMetadataMap

  int64_t GetVersion() const;  ///< Wraps OrtApi::ModelMetadataGetVersion

  static void operator delete(void* p) { Ort::api->ReleaseModelMetadata(reinterpret_cast<OrtModelMetadata*>(p)); }
  Ort::Abstract make_abstract;
};

/** \brief Wrapper around ::OrtSession
 *
 */
struct OrtSession {
  static std::unique_ptr<OrtSession> Create(OrtEnv& env, const ORTCHAR_T* model_path, _In_opt_ const OrtSessionOptions* options);                                                                                  ///< Wraps OrtApi::CreateSession
  static std::unique_ptr<OrtSession> Create(OrtEnv& env, const ORTCHAR_T* model_path, _In_opt_ const OrtSessionOptions* options, OrtPrepackedWeightsContainer& prepacked_weights_container);                       ///< Wraps OrtApi::CreateSessionWithPrepackedWeightsContainer
  static std::unique_ptr<OrtSession> Create(OrtEnv& env, const void* model_data, size_t model_data_length, _In_opt_ const OrtSessionOptions* options);                                                             ///< Wraps OrtApi::CreateSessionFromArray
  static std::unique_ptr<OrtSession> Create(OrtEnv& env, const void* model_data, size_t model_data_length, _In_opt_ const OrtSessionOptions* options, OrtPrepackedWeightsContainer& prepacked_weights_container);  ///< Wraps OrtApi::CreateSessionFromArrayWithPrepackedWeightsContainer

  size_t GetInputCount() const;                   ///< Returns the number of model inputs
  size_t GetOutputCount() const;                  ///< Returns the number of model outputs
  size_t GetOverridableInitializerCount() const;  ///< Returns the number of inputs that have defaults that can be overridden

  /** \brief Returns a copy of input name at the specified index.
   *
   * \param index must less than the value returned by GetInputCount()
   */
  std::string GetInputName(size_t index) const;

  /** \brief Get all input names
   */
  std::vector<std::string> GetInputNames() const;

  /** \brief Returns a copy of output name at then specified index.
   *
   * \param index must less than the value returned by GetOutputCount()
   */
  std::string GetOutputName(size_t index) const;

  /** \brief Get all output names
   */
  std::vector<std::string> GetOutputNames() const;

  /** \brief Returns a copy of the overridable initializer name at then specified index.
   *
   * \param index must less than the value returned by GetOverridableInitializerCount()
   */
  std::string GetOverridableInitializerName(size_t index) const;  ///< Wraps OrtApi::SessionGetOverridableInitializerName

  /** \brief Get all overridable initializer names
   */
  std::vector<std::string> GetOverridableInitializerNames() const;

  /** \brief Returns a copy of the profiling file name.
   */
  std::string EndProfiling();                                  ///< Wraps OrtApi::SessionEndProfiling
  uint64_t GetProfilingStartTimeNs() const;                    ///< Wraps OrtApi::SessionGetProfilingStartTimeNs
  std::unique_ptr<OrtModelMetadata> GetModelMetadata() const;  ///< Wraps OrtApi::SessionGetModelMetadata

  std::unique_ptr<OrtTypeInfo> GetInputTypeInfo(size_t index) const;                   ///< Wraps OrtApi::SessionGetInputTypeInfo
  std::unique_ptr<OrtTypeInfo> GetOutputTypeInfo(size_t index) const;                  ///< Wraps OrtApi::SessionGetOutputTypeInfo
  std::unique_ptr<OrtTypeInfo> GetOverridableInitializerTypeInfo(size_t index) const;  ///< Wraps OrtApi::SessionGetOverridableInitializerTypeInfo

  /** \brief Run the model returning results in an Ort allocated vector.
   *
   * Wraps OrtApi::Run
   *
   * The caller provides a list of inputs and a list of the desired outputs to return.
   *
   * See the output logs for more information on warnings/errors that occur while processing the model.
   * Common errors are.. (TODO)
   *
   * \param[in] run_options
   * \param[in] input_names Array of null terminated strings of length input_count that is the list of input names
   * \param[in] input_values Array of Value objects of length input_count that is the list of input values
   * \param[in] input_count Number of inputs (the size of the input_names & input_values arrays)
   * \param[in] output_names Array of C style strings of length output_count that is the list of output names
   * \param[in] output_count Number of outputs (the size of the output_names array)
   * \return A std::vector of Value objects that directly maps to the output_names array (eg. output_name[0] is the first entry of the returned vector)
   */
  std::vector<std::unique_ptr<OrtValue>> Run(_In_opt_ const OrtRunOptions* run_options, const char* const* input_names, const OrtValue* const* input_values, size_t input_count,
                                             const char* const* output_names, size_t output_count);

  /** \brief Run the model returning results in user provided outputs
   * Same as Run(const RunOptions&, const char* const*, const Value*, size_t,const char* const*, size_t)
   */
  void Run(_In_opt_ const OrtRunOptions* run_options, const char* const* input_names, const OrtValue* const* input_values, size_t input_count,
           const char* const* output_names, OrtValue** output_values, size_t output_count);

  void Run(_In_opt_ const OrtRunOptions* run_options, const OrtIoBinding&);  ///< Wraps OrtApi::RunWithBinding

  static void operator delete(void* p) { Ort::api->ReleaseSession(reinterpret_cast<OrtSession*>(p)); }
  Ort::Abstract make_abstract;
};

/** \brief Wrapper around ::OrtMemoryInfo
 *
 */
struct OrtMemoryInfo {
  static std::unique_ptr<OrtMemoryInfo> CreateCpu(OrtAllocatorType type, OrtMemType mem_type1);
  static std::unique_ptr<OrtMemoryInfo> Create(const char* name, OrtAllocatorType type, int id, OrtMemType mem_type);

  std::string GetAllocatorName() const;
  OrtAllocatorType GetAllocatorType() const;
  int GetDeviceId() const;
  OrtMemoryInfoDeviceType GetDeviceType() const;
  OrtMemType GetMemoryType() const;

  bool operator==(const OrtMemoryInfo& o) const;

  static void operator delete(void* p) { Ort::api->ReleaseMemoryInfo(reinterpret_cast<OrtMemoryInfo*>(p)); }
  Ort::Abstract make_abstract;
};

/** \brief Wrapper around ::OrtTensorTypeAndShapeInfo
 *
 */
struct OrtTensorTypeAndShapeInfo {
  ONNXTensorElementDataType GetElementType() const;  ///< Wraps OrtApi::GetTensorElementType
  size_t GetElementCount() const;                    ///< Wraps OrtApi::GetTensorShapeElementCount

  std::vector<int64_t> GetShape() const;  ///< Uses GetDimensionsCount & GetDimensions to return a std::vector of the shape

  std::vector<const char*> GetSymbolicDimensions() const;  ///< Wraps OrtApi::GetSymbolicDimensions

  static void operator delete(void* p) { Ort::api->ReleaseTensorTypeAndShapeInfo(reinterpret_cast<OrtTensorTypeAndShapeInfo*>(p)); }
  Ort::Abstract make_abstract;
};

/** \brief Wrapper around ::OrtSequenceTypeInfo
 *
 */
struct OrtSequenceTypeInfo {
  std::unique_ptr<OrtTypeInfo> GetSequenceElementType() const;  ///< Wraps OrtApi::GetSequenceElementType

  static void operator delete(void* p) { Ort::api->ReleaseSequenceTypeInfo(reinterpret_cast<OrtSequenceTypeInfo*>(p)); }
  Ort::Abstract make_abstract;
};

/** \brief Wrapper around ::OrtMapTypeInfo
 *
 */
struct OrtMapTypeInfo {
  ONNXTensorElementDataType GetMapKeyType() const;       ///< Wraps OrtApi::GetMapKeyType
  std::unique_ptr<OrtTypeInfo> GetMapValueType() const;  ///< Wraps OrtApi::GetMapValueType

  static void operator delete(void* p) { Ort::api->ReleaseMapTypeInfo(reinterpret_cast<OrtMapTypeInfo*>(p)); }
  Ort::Abstract make_abstract;
};

/// <summary>
/// Type information that may contain either TensorTypeAndShapeInfo or
/// the information about contained sequence or map depending on the ONNXType.
/// </summary>
struct OrtTypeInfo {
  const OrtTensorTypeAndShapeInfo& GetTensorTypeAndShapeInfo() const;  ///< Wraps OrtApi::CastTypeInfoToTensorInfo
  const OrtSequenceTypeInfo& GetSequenceTypeInfo() const;              ///< Wraps OrtApi::CastTypeInfoToSequenceTypeInfo
  const OrtMapTypeInfo& GetMapTypeInfo() const;                        ///< Wraps OrtApi::CastTypeInfoToMapTypeInfo

  ONNXType GetONNXType() const;

  static void operator delete(void* p) { Ort::api->ReleaseTypeInfo(reinterpret_cast<OrtTypeInfo*>(p)); }
  Ort::Abstract make_abstract;
};

// This structure is used to feed  sparse tensor values
// information for use with FillSparseTensor<Format>() API
// if the data type for the sparse tensor values is numeric
// use data.p_data, otherwise, use data.str pointer to feed
// values. data.str is an array of const char* that are zero terminated.
// number of strings in the array must match shape size.
// For fully sparse tensors use shape {0} and set p_data/str
// to nullptr.
struct OrtSparseValuesParam {
  const int64_t* values_shape;
  size_t values_shape_len;
  union {
    const void* p_data;
    const char** str;
  } data;
};

// Provides a way to pass shape in a single
// argument
struct OrtShape {
  const int64_t* shape;
  size_t shape_len;
};

/** \brief Wrapper around ::OrtValue
 *
 */
struct OrtValue {
  /** \brief Creates a tensor with a user supplied buffer. Wraps OrtApi::CreateTensorWithDataAsOrtValue.
   * \tparam T The numeric datatype. This API is not suitable for strings.
   * \param info Memory description of where the p_data buffer resides (CPU vs GPU etc).
   * \param p_data Pointer to the data buffer.
   * \param p_data_element_count The number of elements in the data buffer.
   * \param shape Pointer to the tensor shape dimensions.
   * \param shape_len The number of tensor shape dimensions.
   */
  template <typename T>
  static std::unique_ptr<OrtValue> CreateTensor(const OrtMemoryInfo& info, std::span<T> p_data, std::span<const int64_t> shape);

  /** \brief Creates a tensor with a user supplied buffer. Wraps OrtApi::CreateTensorWithDataAsOrtValue.
   * \param info Memory description of where the p_data buffer resides (CPU vs GPU etc).
   * \param p_data Pointer to the data buffer.
   * \param p_data_byte_count The number of bytes in the data buffer.
   * \param shape Pointer to the tensor shape dimensions.
   * \param shape_len The number of tensor shape dimensions.
   * \param type The data type.
   */
  static std::unique_ptr<OrtValue> CreateTensor(const OrtMemoryInfo& info, void* p_data, size_t p_data_byte_count, std::span<const int64_t> shape,
                                                ONNXTensorElementDataType type);

  /** \brief Creates a tensor using a supplied OrtAllocator. Wraps OrtApi::CreateTensorAsOrtValue.
   * \tparam T The numeric datatype. This API is not suitable for strings.
   * \param allocator The allocator to use.
   * \param shape Pointer to the tensor shape dimensions.
   * \param shape_len The number of tensor shape dimensions.
   */
  template <typename T>
  static std::unique_ptr<OrtValue> CreateTensor(OrtAllocator& allocator, std::span<const int64_t> shape);

  /** \brief Creates a tensor using a supplied OrtAllocator. Wraps OrtApi::CreateTensorAsOrtValue.
   * \param allocator The allocator to use.
   * \param shape Pointer to the tensor shape dimensions.
   * \param shape_len The number of tensor shape dimensions.
   * \param type The data type.
   */
  static std::unique_ptr<OrtValue> CreateTensor(OrtAllocator& allocator, std::span<const int64_t> shape, ONNXTensorElementDataType type);

  static std::unique_ptr<OrtValue> CreateMap(OrtValue& keys, OrtValue& values);                  ///< Wraps OrtApi::CreateValue
  static std::unique_ptr<OrtValue> CreateSequence(const OrtValue* const* values, size_t count);  ///< Wraps OrtApi::CreateValue

  template <typename T>
  static std::unique_ptr<OrtValue> CreateOpaque(const char* domain, const char* type_name, const T&);  ///< Wraps OrtApi::CreateOpaqueValue

  /// <summary>
  /// Obtains a pointer to a user defined data for experimental purposes
  /// </summary>
  template <typename T>
  void GetOpaqueData(const char* domain, const char* type_name, T&) const;  ///< Wraps OrtApi::GetOpaqueValue

  bool IsTensor() const;  ///< Returns true if Value is a tensor, false for other types like map/sequence/etc
  bool HasValue() const;  /// < Return true if OrtValue contains data and returns false if the OrtValue is a None

  size_t GetCount() const;  // If a non tensor, returns 2 for map and N for sequence, where N is the number of elements
  std::unique_ptr<OrtValue> GetValue(int index) const;

  /// <summary>
  /// This API returns a full length of string data contained within either a tensor or a sparse Tensor.
  /// For sparse tensor it returns a full length of stored non-empty strings (values). The API is useful
  /// for allocating necessary memory and calling GetStringTensorContent().
  /// </summary>
  /// <returns>total length of UTF-8 encoded bytes contained. No zero terminators counted.</returns>
  size_t GetStringTensorDataLength() const;

  /// <summary>
  /// The API copies all of the UTF-8 encoded string data contained within a tensor or a sparse tensor
  /// into a supplied buffer. Use GetStringTensorDataLength() to find out the length of the buffer to allocate.
  /// The user must also allocate offsets buffer with the number of entries equal to that of the contained
  /// strings.
  ///
  /// Strings are always assumed to be on CPU, no X-device copy.
  /// </summary>
  /// <param name="buffer">user allocated buffer</param>
  /// <param name="buffer_length">length in bytes of the allocated buffer</param>
  /// <param name="offsets">a pointer to the offsets user allocated buffer</param>
  /// <param name="offsets_count">count of offsets, must be equal to the number of strings contained.
  ///   that can be obtained from the shape of the tensor or from GetSparseTensorValuesTypeAndShapeInfo()
  ///   for sparse tensors</param>
  void GetStringTensorContent(void* buffer, size_t buffer_length, size_t* offsets, size_t offsets_count) const;

  /// <summary>
  /// Returns a const typed pointer to the tensor contained data.
  /// No type checking is performed, the caller must ensure the type matches the tensor type.
  /// </summary>
  /// <typeparam name="T"></typeparam>
  /// <returns>const pointer to data, no copies made</returns>
  template <typename T>
  const T* GetTensorData() const;  ///< Wraps OrtApi::GetTensorMutableData   /// <summary>

  /// <summary>
  /// Returns a non-typed pointer to a tensor contained data.
  /// </summary>
  /// <returns>const pointer to data, no copies made</returns>
  const void* GetTensorRawData() const;

  /// <summary>
  /// The API returns type information for data contained in a tensor. For sparse
  /// tensors it returns type information for contained non-zero values.
  /// It returns dense shape for sparse tensors.
  /// </summary>
  /// <returns>TypeInfo</returns>
  std::unique_ptr<OrtTypeInfo> GetTypeInfo() const;

  /// <summary>
  /// The API returns type information for data contained in a tensor. For sparse
  /// tensors it returns type information for contained non-zero values.
  /// It returns dense shape for sparse tensors.
  /// </summary>
  /// <returns>TensorTypeAndShapeInfo</returns>
  std::unique_ptr<OrtTensorTypeAndShapeInfo> GetTensorTypeAndShapeInfo() const;

  /// <summary>
  /// This API returns information about the memory allocation used to hold data.
  /// </summary>
  /// <returns>Non owning instance of MemoryInfo</returns>
  const OrtMemoryInfo& GetTensorMemoryInfo() const;

  /// <summary>
  /// The API copies UTF-8 encoded bytes for the requested string element
  /// contained within a tensor or a sparse tensor into a provided buffer.
  /// Use GetStringTensorElementLength() to obtain the length of the buffer to allocate.
  /// </summary>
  /// <param name="buffer_length"></param>
  /// <param name="element_index"></param>
  /// <param name="buffer"></param>
  void GetStringTensorElement(size_t buffer_length, size_t element_index, void* buffer) const;

  /// <summary>
  /// The API returns a byte length of UTF-8 encoded string element
  /// contained in either a tensor or a spare tensor values.
  /// </summary>
  /// <param name="element_index"></param>
  /// <returns>byte length for the specified string element</returns>
  size_t GetStringTensorElementLength(size_t element_index) const;

  /// <summary>
  /// Returns a non-const typed pointer to an OrtValue/Tensor contained buffer
  /// No type checking is performed, the caller must ensure the type matches the tensor type.
  /// </summary>
  /// <returns>non-const pointer to data, no copies made</returns>
  template <typename T>
  T* GetTensorMutableData();

  /// <summary>
  /// Returns a non-typed non-const pointer to a tensor contained data.
  /// </summary>
  /// <returns>pointer to data, no copies made</returns>
  void* GetTensorMutableRawData();

  /// <summary>
  /// Obtain a reference to an element of data at the location specified
  /// by the vector of dims.
  /// </summary>
  template <typename T>
  T& At(const std::vector<int64_t>& location);

  /// <summary>
  /// Set all strings at once in a string tensor
  /// </summary>
  /// <param>[in] s An array of strings. Each string in this array must be null terminated.</param>
  /// <param>s_len Count of strings in s (Must match the size of \p value's tensor shape)</param>
  void FillStringTensor(const char* const* s, size_t s_len);

  /// <summary>
  ///  Set a single string in a string tensor
  /// </summary>
  /// <param>s A null terminated UTF-8 encoded string</param>
  /// <param>index Index of the string in the tensor to set</param>
  void FillStringTensorElement(const char* s, size_t index);

#if !defined(DISABLE_SPARSE_TENSORS)
  /// <summary>
  /// The API returns the sparse data format this OrtValue holds in a sparse tensor.
  /// If the sparse tensor was not fully constructed, i.e. Use*() or Fill*() API were not used
  /// the value returned is ORT_SPARSE_UNDEFINED.
  /// </summary>
  /// <returns>Format enum</returns>
  OrtSparseFormat GetSparseFormat() const;

  /// <summary>
  /// The API returns type and shape information for stored non-zero values of the
  /// sparse tensor. Use GetSparseTensorValues() to obtain values buffer pointer.
  /// </summary>
  /// <returns>TensorTypeAndShapeInfo values information</returns>
  std::unique_ptr<OrtTensorTypeAndShapeInfo> GetSparseTensorValuesTypeAndShapeInfo() const;

  /// <summary>
  /// The API returns type and shape information for the specified indices. Each supported
  /// indices have their own enum values even if a give format has more than one kind of indices.
  /// Use GetSparseTensorIndicesData() to obtain pointer to indices buffer.
  /// </summary>
  /// <param name="format">enum requested</param>
  /// <returns>type and shape information</returns>
  std::unique_ptr<OrtTensorTypeAndShapeInfo> GetSparseTensorIndicesTypeShapeInfo(OrtSparseIndicesFormat format) const;

  /// <summary>
  /// The API retrieves a pointer to the internal indices buffer. The API merely performs
  /// a convenience data type casting on the return type pointer. Make sure you are requesting
  /// the right type, use GetSparseTensorIndicesTypeShapeInfo();
  /// </summary>
  /// <typeparam name="T">type to cast to</typeparam>
  /// <param name="indices_format">requested indices kind</param>
  /// <param name="num_indices">number of indices entries</param>
  /// <returns>Pinter to the internal sparse tensor buffer containing indices. Do not free this pointer.</returns>
  template <typename T>
  const T* GetSparseTensorIndicesData(OrtSparseIndicesFormat indices_format, size_t& num_indices) const;

  /// <summary>
  /// Returns true if the OrtValue contains a sparse tensor
  /// </summary>
  /// <returns></returns>
  bool IsSparseTensor() const;

  /// <summary>
  /// The API returns a pointer to an internal buffer of the sparse tensor
  /// containing non-zero values. The API merely does casting. Make sure you
  /// are requesting the right data type by calling GetSparseTensorValuesTypeAndShapeInfo()
  /// first.
  /// </summary>
  /// <typeparam name="T">numeric data types only. Use GetStringTensor*() to retrieve strings.</typeparam>
  /// <returns>a pointer to the internal values buffer. Do not free this pointer.</returns>
  template <typename T>
  const T* GetSparseTensorValues() const;

  /// <summary>
  /// Supplies COO format specific indices and marks the contained sparse tensor as being a COO format tensor.
  /// Values are supplied with a CreateSparseTensor() API. The supplied indices are not copied and the user
  /// allocated buffers lifespan must eclipse that of the OrtValue.
  /// The location of the indices is assumed to be the same as specified by OrtMemoryInfo argument at the creation time.
  /// </summary>
  /// <param name="indices_data">pointer to the user allocated buffer with indices. Use nullptr for fully sparse tensors.</param>
  /// <param name="indices_num">number of indices entries. Use 0 for fully sparse tensors</param>
  void UseCooIndices(int64_t* indices_data, size_t indices_num);

  /// <summary>
  /// Supplies CSR format specific indices and marks the contained sparse tensor as being a CSR format tensor.
  /// Values are supplied with a CreateSparseTensor() API. The supplied indices are not copied and the user
  /// allocated buffers lifespan must eclipse that of the OrtValue.
  /// The location of the indices is assumed to be the same as specified by OrtMemoryInfo argument at the creation time.
  /// </summary>
  /// <param name="inner_data">pointer to the user allocated buffer with inner indices or nullptr for fully sparse tensors</param>
  /// <param name="inner_num">number of csr inner indices or 0 for fully sparse tensors</param>
  /// <param name="outer_data">pointer to the user allocated buffer with outer indices or nullptr for fully sparse tensors</param>
  /// <param name="outer_num">number of csr outer indices or 0 for fully sparse tensors</param>
  void UseCsrIndices(int64_t* inner_data, size_t inner_num, int64_t* outer_data, size_t outer_num);

  /// <summary>
  /// Supplies BlockSparse format specific indices and marks the contained sparse tensor as being a BlockSparse format tensor.
  /// Values are supplied with a CreateSparseTensor() API. The supplied indices are not copied and the user
  /// allocated buffers lifespan must eclipse that of the OrtValue.
  /// The location of the indices is assumed to be the same as specified by OrtMemoryInfo argument at the creation time.
  /// </summary>
  /// <param name="indices_shape">indices shape or a {0} for fully sparse</param>
  /// <param name="indices_data">user allocated buffer with indices or nullptr for fully spare tensors</param>
  void UseBlockSparseIndices(const OrtShape& indices_shape, int32_t* indices_data);

  /// <summary>
  /// The API will allocate memory using the allocator instance supplied to the CreateSparseTensor() API
  /// and copy the values and COO indices into it. If data_mem_info specifies that the data is located
  /// at difference device than the allocator, a X-device copy will be performed if possible.
  /// </summary>
  /// <param name="data_mem_info">specified buffer memory description</param>
  /// <param name="values_param">values buffer information.</param>
  /// <param name="indices_data">coo indices buffer or nullptr for fully sparse data</param>
  /// <param name="indices_num">number of COO indices or 0 for fully sparse data</param>
  void FillSparseTensorCoo(const OrtMemoryInfo& data_mem_info, const OrtSparseValuesParam& values_param,
                           const int64_t* indices_data, size_t indices_num);

  /// <summary>
  /// The API will allocate memory using the allocator instance supplied to the CreateSparseTensor() API
  /// and copy the values and CSR indices into it. If data_mem_info specifies that the data is located
  /// at difference device than the allocator, a X-device copy will be performed if possible.
  /// </summary>
  /// <param name="data_mem_info">specified buffer memory description</param>
  /// <param name="values">values buffer information</param>
  /// <param name="inner_indices_data">csr inner indices pointer or nullptr for fully sparse tensors</param>
  /// <param name="inner_indices_num">number of csr inner indices or 0 for fully sparse tensors</param>
  /// <param name="outer_indices_data">pointer to csr indices data or nullptr for fully sparse tensors</param>
  /// <param name="outer_indices_num">number of csr outer indices or 0</param>
  void FillSparseTensorCsr(const OrtMemoryInfo& data_mem_info,
                           const OrtSparseValuesParam& values,
                           const int64_t* inner_indices_data, size_t inner_indices_num,
                           const int64_t* outer_indices_data, size_t outer_indices_num);

  /// <summary>
  /// The API will allocate memory using the allocator instance supplied to the CreateSparseTensor() API
  /// and copy the values and BlockSparse indices into it. If data_mem_info specifies that the data is located
  /// at difference device than the allocator, a X-device copy will be performed if possible.
  /// </summary>
  /// <param name="data_mem_info">specified buffer memory description</param>
  /// <param name="values">values buffer information</param>
  /// <param name="indices_shape">indices shape. use {0} for fully sparse tensors</param>
  /// <param name="indices_data">pointer to indices data or nullptr for fully sparse tensors</param>
  void FillSparseTensorBlockSparse(const OrtMemoryInfo& data_mem_info,
                                   const OrtSparseValuesParam& values,
                                   const OrtShape& indices_shape,
                                   const int32_t* indices_data);

  /// <summary>
  /// This is a simple forwarding method to the other overload that helps deducing
  /// data type enum value from the type of the buffer.
  /// </summary>
  /// <typeparam name="T">numeric datatype. This API is not suitable for strings.</typeparam>
  /// <param name="info">Memory description where the user buffers reside (CPU vs GPU etc)</param>
  /// <param name="p_data">pointer to the user supplied buffer, use nullptr for fully sparse tensors</param>
  /// <param name="dense_shape">a would be dense shape of the tensor</param>
  /// <param name="values_shape">non zero values shape. Use a single 0 shape for fully sparse tensors.</param>
  /// <returns></returns>
  template <typename T>
  static std::unique_ptr<OrtValue> CreateSparseTensor(const OrtMemoryInfo& info, T* p_data, const OrtShape& dense_shape,
                                                      const OrtShape& values_shape);

  /// <summary>
  /// Creates an OrtValue instance containing SparseTensor. This constructs
  /// a sparse tensor that makes use of user allocated buffers. It does not make copies
  /// of the user provided data and does not modify it. The lifespan of user provided buffers should
  /// eclipse the life span of the resulting OrtValue. This call constructs an instance that only contain
  /// a pointer to non-zero values. To fully populate the sparse tensor call Use<Format>Indices() API below
  /// to supply a sparse format specific indices.
  /// This API is not suitable for string data. Use CreateSparseTensor() with allocator specified so strings
  /// can be properly copied into the allocated buffer.
  /// </summary>
  /// <param name="info">Memory description where the user buffers reside (CPU vs GPU etc)</param>
  /// <param name="p_data">pointer to the user supplied buffer, use nullptr for fully sparse tensors</param>
  /// <param name="dense_shape">a would be dense shape of the tensor</param>
  /// <param name="values_shape">non zero values shape. Use a single 0 shape for fully sparse tensors.</param>
  /// <param name="type">data type</param>
  /// <returns>Ort::Value instance containing SparseTensor</returns>
  static std::unique_ptr<OrtValue> CreateSparseTensor(const OrtMemoryInfo& info, void* p_data, const OrtShape& dense_shape,
                                                      const OrtShape& values_shape, ONNXTensorElementDataType type);

  /// <summary>
  /// This is a simple forwarding method to the below CreateSparseTensor.
  /// This helps to specify data type enum in terms of C++ data type.
  /// Use CreateSparseTensor<T>
  /// </summary>
  /// <typeparam name="T">numeric data type only. String data enum must be specified explicitly.</typeparam>
  /// <param name="allocator">allocator to use</param>
  /// <param name="dense_shape">a would be dense shape of the tensor</param>
  /// <returns>Ort::Value</returns>
  template <typename T>
  static std::unique_ptr<OrtValue> CreateSparseTensor(OrtAllocator* allocator, const OrtShape& dense_shape);

  /// <summary>
  /// Creates an instance of OrtValue containing sparse tensor. The created instance has no data.
  /// The data must be supplied by on of the FillSparseTensor<Format>() methods that take both non-zero values
  /// and indices. The data will be copied into a buffer that would be allocated using the supplied allocator.
  /// Use this API to create OrtValues that contain sparse tensors with all supported data types including
  /// strings.
  /// </summary>
  /// <param name="allocator">allocator to use. The allocator lifespan must eclipse that of the resulting OrtValue</param>
  /// <param name="dense_shape">a would be dense shape of the tensor</param>
  /// <param name="type">data type</param>
  /// <returns>an instance of Ort::Value</returns>
  static std::unique_ptr<OrtValue> CreateSparseTensor(OrtAllocator* allocator, const OrtShape& dense_shape, ONNXTensorElementDataType type);

#endif  // !defined(DISABLE_SPARSE_TENSORS)

  static void operator delete(void* p) { Ort::api->ReleaseValue(reinterpret_cast<OrtValue*>(p)); }
  Ort::Abstract make_abstract;
};

/** \brief Wrapper around ::OrtIoBinding
 *
 */
struct OrtIoBinding {
  static std::unique_ptr<OrtIoBinding> Create(OrtSession& session);

  std::vector<std::string> GetOutputNames() const;
  std::vector<std::unique_ptr<OrtValue>> GetOutputValues() const;

  void BindInput(const char* name, const OrtValue&);
  void BindOutput(const char* name, const OrtValue&);
  void BindOutput(const char* name, const OrtMemoryInfo&);
  void ClearBoundInputs();
  void ClearBoundOutputs();
  void SynchronizeInputs();
  void SynchronizeOutputs();

  static void operator delete(void* p) { Ort::api->ReleaseIoBinding(reinterpret_cast<OrtIoBinding*>(p)); }
  Ort::Abstract make_abstract;
};

/*! \struct Ort::ArenaCfg
 * \brief it is a structure that represents the configuration of an arena based allocator
 * \details Please see docs/C_API.md for details
 */
struct OrtArenaCfg {
  /**
   * Wraps OrtApi::CreateArenaCfg
   * \param max_mem - use 0 to allow ORT to choose the default
   * \param arena_extend_strategy -  use -1 to allow ORT to choose the default, 0 = kNextPowerOfTwo, 1 = kSameAsRequested
   * \param initial_chunk_size_bytes - use -1 to allow ORT to choose the default
   * \param max_dead_bytes_per_chunk - use -1 to allow ORT to choose the default
   * See docs/C_API.md for details on what the following parameters mean and how to choose these values
   */
  static std::unique_ptr<OrtArenaCfg> Create(size_t max_mem, int arena_extend_strategy, int initial_chunk_size_bytes, int max_dead_bytes_per_chunk);

  static void operator delete(void* p) { Ort::api->ReleaseArenaCfg(reinterpret_cast<OrtArenaCfg*>(p)); }
  Ort::Abstract make_abstract;
};

//
// Custom OPs (only needed to implement custom OPs)
//

/// <summary>
/// This struct provides life time management for custom op attribute
/// </summary>
struct OrtOpAttr {
  static std::unique_ptr<OrtOpAttr> Create(const char* name, const void* data, int len, OrtOpAttrType type);

  static void operator delete(void* p) { Ort::api->ReleaseOpAttr(reinterpret_cast<OrtOpAttr*>(p)); }
  Ort::Abstract make_abstract;
};

/// <summary>
/// This class wraps a raw pointer OrtKernelContext* that is being passed
/// to the custom kernel Compute() method. Use it to safely access context
/// attributes, input and output parameters with exception safety guarantees.
/// See usage example in onnxruntime/test/testdata/custom_op_library/custom_op_library.cc
/// </summary>
struct OrtKernelContext {
  size_t GetInputCount() const;
  size_t GetOutputCount() const;
  const OrtValue* GetInput(size_t index) const;
  OrtValue* GetOutput(size_t index, const int64_t* dim_values, size_t dim_count);
  OrtValue* GetOutput(size_t index, const std::vector<int64_t>& dims);
  void* GetGPUComputeStream() const;

  static void operator delete(void* p) = delete;
  Ort::Abstract make_abstract;
};

struct OrtKernelInfo {
  std::unique_ptr<OrtKernelInfo> Clone() const;

  template <typename T>  // T is only implemented for float, int64_t, and string
  T GetAttribute(const char* name) const {
    T val;
    GetAttr(name, val);
    return val;
  }

  template <typename T>  // T is only implemented for std::vector<float>, std::vector<int64_t>
  std::vector<T> GetAttributes(const char* name) const {
    std::vector<T> result;
    GetAttrs(name, result);
    return result;
  }

  void GetAttr(const char* name, float&);
  void GetAttr(const char* name, int64_t&);
  void GetAttr(const char* name, std::string&);
  void GetAttrs(const char* name, std::vector<float>&);
  void GetAttrs(const char* name, std::vector<int64_t>&);

  static void operator delete(void* p) { Ort::api->ReleaseKernelInfo(reinterpret_cast<OrtKernelInfo*>(p)); }
  Ort::Abstract make_abstract;
};

/// <summary>
/// Create and own custom defined operation.
/// </summary>
struct OrtOp {
  static std::unique_ptr<OrtOp> Create(const OrtKernelInfo* info, const char* op_name, const char* domain,
                                       int version, const char** type_constraint_names,
                                       const ONNXTensorElementDataType* type_constraint_values,
                                       size_t type_constraint_count,
                                       const OrtOpAttr* const* attr_values,
                                       size_t attr_count,
                                       size_t input_count, size_t output_count);

  void Invoke(const OrtKernelContext* context,
              const OrtValue* const* input_values,
              size_t input_count,
              OrtValue* const* output_values,
              size_t output_count);
};

/** \brief LoraAdapter
 *
 */
struct OrtLoraAdapter {
  static std::unique_ptr<OrtLoraAdapter> Create(const ORTCHAR_T* adapter_file_path, OrtAllocator& allocator);  ///< Wraps OrtApi::CreateOrtLoraAdapter

  static void operator delete(void* p) { Ort::api->ReleaseLoraAdapter(reinterpret_cast<OrtLoraAdapter*>(p)); }
  Ort::Abstract make_abstract;
};

#include "onnxruntime_inline.h"
