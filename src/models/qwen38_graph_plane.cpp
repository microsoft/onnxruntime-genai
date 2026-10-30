// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "qwen38_graph_plane.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include "qwen38_numa.h"

// QWEN38_GRAPH_NO_POSIX forces the unsupported-platform stub variant on any platform (used to compile-check it).
#if defined(_WIN32) && !defined(QWEN38_GRAPH_NO_POSIX)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif !defined(QWEN38_GRAPH_NO_POSIX)
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace Generators {

namespace {

constexpr int kMaxPleHeads = 32;
constexpr int kMaxPleLaunchElems = 3 * 1024;
constexpr size_t kMaxPleShards = 256;

#if defined(QWEN38_GRAPH_NO_POSIX)
[[noreturn]] void ThrowUnsupportedPlatform() {
  throw std::runtime_error("The qwen38_flash_graph model type is not supported when QWEN38_GRAPH_NO_POSIX is set.");
}
#endif

#if defined(_WIN32) && !defined(QWEN38_GRAPH_NO_POSIX)
bool ReadExactWin32(HANDLE hFile, uint64_t offset, void* dst, size_t bytes) {
  LARGE_INTEGER li{};
  li.QuadPart = static_cast<LONGLONG>(offset);
  if (!SetFilePointerEx(hFile, li, nullptr, FILE_BEGIN)) return false;
  uint8_t* out = static_cast<uint8_t*>(dst);
  size_t remaining = bytes;
  while (remaining > 0) {
    const DWORD chunk = static_cast<DWORD>(std::min<size_t>(remaining, 64u * 1024u * 1024u));
    DWORD read_bytes = 0;
    if (!ReadFile(hFile, out, chunk, &read_bytes, nullptr) || read_bytes == 0) return false;
    out += read_bytes;
    remaining -= read_bytes;
  }
  return true;
}
#endif

}  // namespace

// ------------------------------------------------------------------------------------------------
// Plane library
// ------------------------------------------------------------------------------------------------
std::filesystem::path ResolveWindowsSharedLibraryPath(const std::filesystem::path& path) {
  if (path.extension() != ".so") {
    return path;
  }
  std::error_code ec;
  auto dll_path = path;
  dll_path.replace_extension(".dll");
  if (std::filesystem::exists(dll_path, ec)) {
    return dll_path;
  }
  const std::string stem = path.stem().string();
  if (stem.rfind("lib", 0) == 0 && stem.size() > 3) {
    auto stripped = path.parent_path() / (stem.substr(3) + ".dll");
    if (std::filesystem::exists(stripped, ec)) {
      return stripped;
    }
  }
  return path;
}

#if defined(QWEN38_GRAPH_NO_POSIX)

Qwen38Plane::Qwen38Plane(const std::filesystem::path&) { ThrowUnsupportedPlatform(); }
Qwen38Plane::~Qwen38Plane() = default;

#elif defined(_WIN32)

Qwen38Plane::Qwen38Plane(const std::filesystem::path& path) {
  HMODULE mod = nullptr;
  std::filesystem::path actual_path = path;
  if (actual_path.empty()) {
    actual_path = "onnxruntime-genai-cuda.dll";
    HMODULE self_mod = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&ParseQwen38PleDescription), &self_mod) &&
        self_mod) {
      wchar_t self_buf[MAX_PATH]{};
      if (GetModuleFileNameW(self_mod, self_buf, MAX_PATH) > 0) {
        auto candidate = std::filesystem::path(self_buf).parent_path() / L"onnxruntime-genai-cuda.dll";
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec)) {
          actual_path = candidate;
        }
      }
    }
    mod = LoadLibraryW(actual_path.wstring().c_str());
    if (!mod) {
      mod = LoadLibraryW(L"onnxruntime-genai-cuda.dll");
    }
  } else {
    actual_path = ResolveWindowsSharedLibraryPath(actual_path);
    mod = LoadLibraryW(actual_path.wstring().c_str());
  }
  if (!mod) {
    const DWORD err = GetLastError();
    throw std::runtime_error("qwen38_graph: failed to load the plane library " + actual_path.string() +
                             " (Win32 error " + std::to_string(err) + ")");
  }
  handle_ = static_cast<void*>(mod);
  library_path = actual_path.string();
  auto load = [mod](auto& function, const char* symbol_name) {
    FARPROC symbol = GetProcAddress(mod, symbol_name);
    if (!symbol) {
      throw std::runtime_error(std::string("qwen38_graph: the plane library lacks the symbol ") + symbol_name);
    }
    function = reinterpret_cast<std::remove_reference_t<decltype(function)> >(symbol);
  };
  try {
    // The version is checked before the other symbols are resolved so that a stale library is reported as such rather than
    // as lacking a symbol that a newer ABI added.
    load(abi_version, "qwen38_plane_abi_version");
    if (abi_version() != QWEN38_PLANE_ABI_VERSION) {
      throw std::runtime_error("qwen38_graph: the plane library has ABI version " + std::to_string(abi_version()) +
                               ", expected " + std::to_string(QWEN38_PLANE_ABI_VERSION));
    }
#define QWEN38_PLANE_LOAD(ret, name, args) load(name, "qwen38_plane_" #name);
    QWEN38_PLANE_FUNCTIONS(QWEN38_PLANE_LOAD)
#undef QWEN38_PLANE_LOAD
    quantize_kv = reinterpret_cast<decltype(quantize_kv)>(GetProcAddress(mod, "qwen38_plane_quantize_kv"));
    embed_rows = reinterpret_cast<decltype(embed_rows)>(GetProcAddress(mod, "qwen38_plane_embed_rows"));
    embed_gather = reinterpret_cast<decltype(embed_gather)>(GetProcAddress(mod, "qwen38_plane_embed_gather"));
    prep_batched = reinterpret_cast<decltype(prep_batched)>(GetProcAddress(mod, "qwen38_plane_prep_batched"));
    set_ple_host_params =
        reinterpret_cast<decltype(set_ple_host_params)>(GetProcAddress(mod, "qwen38_plane_set_ple_host_params"));
    unset_ple_host_params =
        reinterpret_cast<decltype(unset_ple_host_params)>(GetProcAddress(mod, "qwen38_plane_unset_ple_host_params"));
    release_ple_workspace =
        reinterpret_cast<decltype(release_ple_workspace)>(GetProcAddress(mod, "qwen38_plane_release_ple_workspace"));
    dflash_select = reinterpret_cast<decltype(dflash_select)>(GetProcAddress(mod, "qwen38_plane_dflash_select"));
    replay_state_updates =
        reinterpret_cast<decltype(replay_state_updates)>(GetProcAddress(mod, "qwen38_plane_replay_state_updates"));
    // A library that predates the capabilities export has none: it still loads, for the packages that need none.
    const auto capabilities_export =
        reinterpret_cast<Qwen38PlaneCapabilitiesFn>(GetProcAddress(mod, "qwen38_plane_capabilities"));
    capabilities = capabilities_export ? capabilities_export() : 0ull;
    if (sizeof_ple_params() != static_cast<int>(sizeof(Qwen38PleParams))) {
      throw std::runtime_error("qwen38_graph: the plane library's PLE parameter layout differs from the host definition");
    }
    attach();
    attached_ = true;
  } catch (...) {
    FreeLibrary(mod);
    handle_ = nullptr;
    throw;
  }
}

Qwen38Plane::~Qwen38Plane() {
  // detach() stops and joins the threads the module started. The module stays loaded: see the comment of the class.
  if (attached_) detach();
}

#else

Qwen38Plane::Qwen38Plane(const std::filesystem::path& path) {
  std::string tried_name = path.string();
  if (path.empty()) {
    const char* default_lib =
#if defined(__APPLE__)
        "libonnxruntime-genai-cuda.dylib";
#else
        "libonnxruntime-genai-cuda.so";
#endif
    tried_name = default_lib;
    // The CUDA library is normally already loaded by the CUDA execution provider. Otherwise look next to GenAI.
    handle_ = dlopen(default_lib, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    if (!handle_) {
      Dl_info info{};
      if (dladdr(reinterpret_cast<const void*>(&ParseQwen38PleDescription), &info) != 0 && info.dli_fname) {
        const auto candidate = std::filesystem::path(info.dli_fname).parent_path() / default_lib;
        tried_name = candidate.string();
        handle_ = dlopen(candidate.c_str(), RTLD_NOW | RTLD_LOCAL);
      }
    }
    if (!handle_) {
      handle_ = dlopen(default_lib, RTLD_NOW | RTLD_LOCAL);
    }
  } else {
    handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  }
  if (!handle_) {
    const char* reason = dlerror();
    throw std::runtime_error("qwen38_graph: failed to load the plane library " + tried_name + ": " +
                             (reason ? reason : "unknown error"));
  }
  library_path = path.string();
  auto load = [this](auto& function, const char* symbol_name) {
    void* symbol = dlsym(handle_, symbol_name);
    if (!symbol) {
      throw std::runtime_error(std::string("qwen38_graph: the plane library lacks the symbol ") + symbol_name);
    }
    function = reinterpret_cast<std::remove_reference_t<decltype(function)> >(symbol);
  };
  try {
    // The version is checked before the other symbols are resolved so that a stale library is reported as such rather than
    // as lacking a symbol that a newer ABI added.
    load(abi_version, "qwen38_plane_abi_version");
    if (abi_version() != QWEN38_PLANE_ABI_VERSION) {
      throw std::runtime_error("qwen38_graph: the plane library has ABI version " + std::to_string(abi_version()) +
                               ", expected " + std::to_string(QWEN38_PLANE_ABI_VERSION));
    }
#define QWEN38_PLANE_LOAD(ret, name, args) load(name, "qwen38_plane_" #name);
    QWEN38_PLANE_FUNCTIONS(QWEN38_PLANE_LOAD)
#undef QWEN38_PLANE_LOAD
    quantize_kv = reinterpret_cast<decltype(quantize_kv)>(dlsym(handle_, "qwen38_plane_quantize_kv"));
    embed_rows = reinterpret_cast<decltype(embed_rows)>(dlsym(handle_, "qwen38_plane_embed_rows"));
    embed_gather = reinterpret_cast<decltype(embed_gather)>(dlsym(handle_, "qwen38_plane_embed_gather"));
    prep_batched = reinterpret_cast<decltype(prep_batched)>(dlsym(handle_, "qwen38_plane_prep_batched"));
    set_ple_host_params =
        reinterpret_cast<decltype(set_ple_host_params)>(dlsym(handle_, "qwen38_plane_set_ple_host_params"));
    unset_ple_host_params =
        reinterpret_cast<decltype(unset_ple_host_params)>(dlsym(handle_, "qwen38_plane_unset_ple_host_params"));
    release_ple_workspace =
        reinterpret_cast<decltype(release_ple_workspace)>(dlsym(handle_, "qwen38_plane_release_ple_workspace"));
    dflash_select = reinterpret_cast<decltype(dflash_select)>(dlsym(handle_, "qwen38_plane_dflash_select"));
    replay_state_updates =
        reinterpret_cast<decltype(replay_state_updates)>(dlsym(handle_, "qwen38_plane_replay_state_updates"));
    // A library that predates the capabilities export has none: it still loads, for the packages that need none.
    const auto capabilities_export =
        reinterpret_cast<Qwen38PlaneCapabilitiesFn>(dlsym(handle_, "qwen38_plane_capabilities"));
    capabilities = capabilities_export ? capabilities_export() : 0ull;
    if (sizeof_ple_params() != static_cast<int>(sizeof(Qwen38PleParams))) {
      throw std::runtime_error("qwen38_graph: the plane library's PLE parameter layout differs from the host definition");
    }
    attach();
    attached_ = true;
  } catch (...) {
    dlclose(handle_);
    handle_ = nullptr;
    throw;
  }
}

Qwen38Plane::~Qwen38Plane() {
  // detach() stops and joins the threads the module started. The module stays loaded: see the comment of the class.
  if (attached_) detach();
}

#endif

void Qwen38Plane::Check(int error, const char* what) const {
  if (error != 0) {
    throw std::runtime_error(std::string("qwen38_graph: CUDA error in ") + what + ": " + error_string(error));
  }
}

namespace {
struct Qwen38PlaneCapabilityName {
  unsigned long long bit;
  const char* name;
};

#define QWEN38_PLANE_CAPABILITY_NAME(bit, name) {bit, name},
constexpr Qwen38PlaneCapabilityName kQwen38PlaneCapabilityNames[] = {
    QWEN38_PLANE_CAPABILITIES(QWEN38_PLANE_CAPABILITY_NAME)};
#undef QWEN38_PLANE_CAPABILITY_NAME
}  // namespace

std::string DescribeQwen38PlaneCapabilities(unsigned long long mask) {
  std::string description;
  unsigned long long named = 0;
  for (const auto& capability : kQwen38PlaneCapabilityNames) {
    named |= capability.bit;
    if ((mask & capability.bit) == 0) continue;
    if (!description.empty()) description += ", ";
    description += capability.name;
  }
  if ((mask & ~named) != 0) {
    char unknown[40];
    std::snprintf(unknown, sizeof(unknown), "unknown bits 0x%llx", mask & ~named);
    if (!description.empty()) description += ", ";
    description += unknown;
  }
  return description.empty() ? "none" : description;
}

void Qwen38Plane::RequireCapabilities(unsigned long long required, const std::string& required_by) const {
  const unsigned long long missing = required & ~capabilities;
  if (missing == 0) return;
  throw std::runtime_error(
      "qwen38_graph: " + required_by + " requires the plane capability " + DescribeQwen38PlaneCapabilities(missing) +
      ", which the plane library " + library_path + " does not provide (it reports: " +
      DescribeQwen38PlaneCapabilities(capabilities) +
      "). Use a plane library built from the same source as this model package.");
}

Qwen38StateReplayDesc MakeQwen38ConvReplayDescriptor(const Qwen38Plane& plane, const Qwen38ReplayRequest& request,
                                                     const void* source, void* destination, const void* value) {
  if (request.kind != Qwen38ReplayKind::kConvolution) {
    throw std::invalid_argument("qwen38_graph: " + request.state.name + " is not the state of a convolution replay");
  }
  const Qwen38ReplayGeometry geometry = ValidateQwen38ReplayDescriptor(request);
  if (geometry.layout == QWEN38_PLANE_CONV_STATE_HISTORY_FIRST) {
    plane.RequireCapabilities(QWEN38_PLANE_CAP_CONV_HISTORY_FIRST_REPLAY,
                              "the history-first conv state " + request.state.name);
  }
  Qwen38StateReplayDesc descriptor{};
  descriptor.source_state = source;
  descriptor.destination_state = destination;
  descriptor.value = value;
  descriptor.channel_count = geometry.channel_count;
  descriptor.state_width = geometry.state_width;
  descriptor.capacity = static_cast<uint32_t>(request.capacity);
  descriptor.kept_count = 1;
  descriptor.element_size = geometry.element_size;
  descriptor.kind = 1;
  descriptor.layout = static_cast<uint32_t>(geometry.layout);
  return descriptor;
}

void Qwen38Plane::EmbedRows(void* stream, const Qwen38EmbeddingTable& table, const void* ids, void* out, int n) const {
  if (embed_gather) {
    const int format = table.int8 ? QWEN38_PLANE_EMBEDDING_INT8_ROW : QWEN38_PLANE_EMBEDDING_BF16;
    Check(embed_gather(stream, table.data, table.rows, format, ids, out, n, table.width), "embed_gather");
    return;
  }
  if (table.int8) {
    // The legacy embed_rows has no INT8 mode: it returns without writing anything.
    RequireCapabilities(QWEN38_PLANE_CAP_INT8_EMBEDDING, "an INT8 embedding table");
  } else if (embed_rows) {
    embed_rows(stream, table.data, ids, out, n, table.width);
    return;
  }
  throw std::runtime_error("qwen38_graph: the plane library " + library_path +
                           " lacks the qwen38_plane_embed_gather export needed to gather embedding rows");
}

void Qwen38Plane::PrepBatched(void* stream, const void* params, int rows, int batch, const void* ctx_in, void* ctx_out,
                              void* input_ids, void* position_ids, void* committed, void* past_pos, void* ple_out,
                              void* state) const {
  if (rows < 1 || rows > QWEN38_PLANE_MAX_PREP_ROWS) {
    throw std::invalid_argument("qwen38_graph: batched prep needs 1.." + std::to_string(QWEN38_PLANE_MAX_PREP_ROWS) +
                                " rows per slot, got " + std::to_string(rows));
  }
  if (batch < 1 || batch > QWEN38_PLANE_MAX_PREP_BATCH) {
    throw std::invalid_argument("qwen38_graph: batched prep needs 1.." + std::to_string(QWEN38_PLANE_MAX_PREP_BATCH) +
                                " slots, got " + std::to_string(batch));
  }
  RequireCapabilities(QWEN38_PLANE_CAP_BATCHED_PREP, "preparing the inputs of several request slots in one call");
  if (!prep_batched) {
    throw std::runtime_error("qwen38_graph: the plane library " + library_path +
                             " lacks the qwen38_plane_prep_batched export needed to prepare several request slots");
  }
  Check(prep_batched(stream, params, rows, batch, ctx_in, ctx_out, input_ids, position_ids, committed, past_pos, ple_out,
                     state),
        "prep_batched");
}

size_t Qwen38EmbeddingTableBytes(Qwen38EmbeddingFormat format, long long rows, long long width) {
  if (rows <= 0 || width <= 0) return 0;
  const size_t payload = static_cast<size_t>(rows) * static_cast<size_t>(width);
  if (format == Qwen38EmbeddingFormat::kBf16) return payload * sizeof(uint16_t);
  if (format == Qwen38EmbeddingFormat::kInt8) return payload + static_cast<size_t>(rows) * sizeof(float);
  return 0;
}

Qwen38EmbeddingFormat SelectQwen38EmbeddingFormat(const Qwen38Plane& plane, const std::filesystem::path& table_path,
                                                  size_t file_bytes, long long rows, long long width) {
  if (rows <= 0 || width <= 0) return Qwen38EmbeddingFormat::kNone;
  if (file_bytes >= Qwen38EmbeddingTableBytes(Qwen38EmbeddingFormat::kBf16, rows, width)) {
    return Qwen38EmbeddingFormat::kBf16;
  }
  if (file_bytes >= Qwen38EmbeddingTableBytes(Qwen38EmbeddingFormat::kInt8, rows, width)) {
    plane.RequireCapabilities(QWEN38_PLANE_CAP_INT8_EMBEDDING, "the INT8 embedding table " + table_path.string());
    return Qwen38EmbeddingFormat::kInt8;
  }
  return Qwen38EmbeddingFormat::kNone;
}

PlaneBuffer::PlaneBuffer(Qwen38Plane& plane, size_t bytes, void* stream) : plane_{&plane}, bytes_{bytes} {
  void* p = nullptr;
  plane.Check(plane.device_alloc(&p, std::max<size_t>(bytes, 16)), "cudaMalloc");
  ptr_ = static_cast<uint8_t*>(p);
  try {
    plane.Check(plane.device_memset(ptr_, 0, bytes_, stream), "cudaMemsetAsync");
  } catch (...) {
    plane.device_free(ptr_);
    ptr_ = nullptr;
    throw;
  }
}

PlaneBuffer::~PlaneBuffer() {
  if (ptr_ && plane_) plane_->device_free(ptr_);
}

PlaneBuffer::PlaneBuffer(PlaneBuffer&& other) noexcept : plane_{other.plane_}, ptr_{other.ptr_}, bytes_{other.bytes_} {
  other.ptr_ = nullptr;
  other.bytes_ = 0;
}

PlaneBuffer& PlaneBuffer::operator=(PlaneBuffer&& other) noexcept {
  if (this != &other) {
    if (ptr_ && plane_) plane_->device_free(ptr_);
    plane_ = other.plane_;
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

PlaneHostBuffer::PlaneHostBuffer(Qwen38Plane& plane, size_t bytes) : plane_{&plane} {
  void* p = nullptr;
  plane.Check(plane.host_alloc(&p, bytes), "cudaHostAlloc");
  ptr_ = static_cast<int64_t*>(p);
  std::fill_n(reinterpret_cast<uint8_t*>(ptr_), bytes, uint8_t{0});
}

PlaneHostBuffer::~PlaneHostBuffer() {
  if (ptr_ && plane_) plane_->host_free(ptr_);
}

PlaneHostBuffer::PlaneHostBuffer(PlaneHostBuffer&& other) noexcept : plane_{other.plane_}, ptr_{other.ptr_} {
  other.ptr_ = nullptr;
}

PlaneHostBuffer& PlaneHostBuffer::operator=(PlaneHostBuffer&& other) noexcept {
  if (this != &other) {
    if (ptr_ && plane_) plane_->host_free(ptr_);
    plane_ = other.plane_;
    ptr_ = other.ptr_;
    other.ptr_ = nullptr;
  }
  return *this;
}

// ------------------------------------------------------------------------------------------------
// PLE table
// ------------------------------------------------------------------------------------------------
Qwen38PleDescription ParseQwen38PleDescription(const std::filesystem::path& table_path) {
  std::ifstream in(table_path);
  if (!in) throw std::runtime_error("qwen38_graph: cannot open the PLE table " + table_path.string());
  std::string line;
  std::getline(in, line);
  if (line != "qwen38_ple_table 1") throw std::runtime_error("qwen38_graph: unsupported PLE table header: " + line);

  Qwen38PleDescription description;
  Qwen38PleParams& params = description.params;
  bool has_vocab = false;
  bool has_offset = false;
  while (std::getline(in, line)) {
    std::istringstream fields(line);
    std::string key;
    if (!(fields >> key)) continue;
    if (key == "n_heads") {
      fields >> params.n_heads;
      if (fields.fail()) throw std::runtime_error("qwen38_graph: malformed PLE n_heads line: " + line);
    } else if (key == "head_dim") {
      fields >> params.head_dim;
      if (fields.fail()) throw std::runtime_error("qwen38_graph: malformed PLE head_dim line: " + line);
    } else if (key == "heads_per_ngram") {
      fields >> params.heads_per_ngram;
      if (fields.fail()) throw std::runtime_error("qwen38_graph: malformed PLE heads_per_ngram line: " + line);
    } else if (key == "eos") {
      fields >> params.eos;
      if (fields.fail()) throw std::runtime_error("qwen38_graph: malformed PLE eos line: " + line);
    } else if (key == "scale") {
      fields >> params.scale;
      if (fields.fail()) throw std::runtime_error("qwen38_graph: malformed PLE scale line: " + line);
    } else if (key == "mult") {
      for (long long& m : params.mult) fields >> m;
      if (fields.fail()) throw std::runtime_error("qwen38_graph: malformed PLE mult line: " + line);
    } else if (key == "vocab" || key == "offset") {
      if (params.n_heads <= 0 || params.n_heads > kMaxPleHeads)
        throw std::runtime_error("qwen38_graph: the PLE table must declare n_heads (1.." + std::to_string(kMaxPleHeads) +
                                 ") before " + key);
      long long* values = key == "vocab" ? params.vocab : params.offset;
      for (int h = 0; h < params.n_heads; ++h) fields >> values[h];
      if (fields.fail()) throw std::runtime_error("qwen38_graph: malformed PLE " + key + " line: " + line);
      if (key == "vocab")
        has_vocab = true;
      else
        has_offset = true;
    } else if (key == "shard") {
      Qwen38PleDescription::Shard shard;
      fields >> shard.path >> shard.offset >> shard.rows;
      if (fields.fail() || shard.rows <= 0)
        throw std::runtime_error("qwen38_graph: malformed PLE shard line: " + line);
      const std::filesystem::path shard_path{shard.path};
      // has_root_path(), not is_absolute(): on Windows "/a.bin" and "C:a.bin" are not absolute but still leave the folder.
      if (shard_path.has_root_path() || shard.path.find("..") != std::string::npos)
        throw std::runtime_error(
            "qwen38_graph: a PLE shard path must be relative to the model folder and must not contain "
            "'..': " +
            shard.path);
      description.shards.push_back(std::move(shard));
    }
  }
  if (params.n_heads <= 0 || params.n_heads > kMaxPleHeads || params.head_dim <= 0 ||
      params.n_heads > kMaxPleLaunchElems / params.head_dim || params.heads_per_ngram < 0 ||
      params.heads_per_ngram > params.n_heads || !has_vocab || !has_offset || description.shards.empty() ||
      description.shards.size() > kMaxPleShards)
    throw std::runtime_error("qwen38_graph: malformed PLE table " + table_path.string());
  long long total_rows = 0;
  for (const auto& shard : description.shards) {
    if (shard.rows <= 0 || shard.rows > std::numeric_limits<long long>::max() - total_rows)
      throw std::runtime_error("qwen38_graph: malformed PLE table " + table_path.string());
    total_rows += shard.rows;
  }
  for (int h = 0; h < params.n_heads; ++h) {
    if (params.vocab[h] <= 0 || params.offset[h] < 0 || params.vocab[h] > total_rows - params.offset[h])
      throw std::runtime_error("qwen38_graph: PLE head " + std::to_string(h) +
                               " vocab/offset out of shard row bounds in " + table_path.string());
  }
  params.n_shards = static_cast<int>(description.shards.size());
  return description;
}

void RequireQwen38PleTableSupported(const Qwen38Plane& plane, const Qwen38PleDescription& description,
                                    const std::filesystem::path& table_path) {
  // A negative scale marks INT8 shards; a library without the capability would decode them as FP8.
  if (description.params.scale < 0.0f) {
    plane.RequireCapabilities(QWEN38_PLANE_CAP_INT8_PLE, "the INT8 PLE table " + table_path.string());
  }
}

// ------------------------------------------------------------------------------------------------
// Weight pool manifest / QSA constants index
// ------------------------------------------------------------------------------------------------
namespace {

void RequireInsideModelFolder(const std::string& path, const char* what) {
  // has_root_path(), not is_absolute(): on Windows "/a.bin" and "C:a.bin" are not absolute but still leave the folder.
  if (path.empty() || std::filesystem::path{path}.has_root_path() || path.find("..") != std::string::npos)
    throw std::runtime_error(std::string("qwen38_graph: ") + what + " must be relative to the model folder and must not contain '..': " +
                             path);
}

}  // namespace

Qwen38WeightPoolManifest ParseQwen38WeightPoolManifest(const std::filesystem::path& manifest_path) {
  std::ifstream in(manifest_path);
  if (!in) throw std::runtime_error("qwen38_graph: cannot open the weight pool manifest " + manifest_path.string());
  std::string line;
  std::getline(in, line);
  if (line != "qwen38_weights_pool 1")
    throw std::runtime_error("qwen38_graph: unsupported weight pool manifest header: " + line);
  Qwen38WeightPoolManifest manifest;
  while (std::getline(in, line)) {
    std::istringstream fields(line);
    std::string key;
    fields >> key;
    if (key == "pool" || key == "host_pool") {
      size_t id, rank;
      Qwen38WeightPoolManifest::Entry entry;
      if (key == "host_pool") {
        entry.placement = "host_mapped";
        entry.host_mapped = true;
      }
      fields >> id >> entry.dtype >> entry.offset >> entry.length >> rank;
      if (fields.fail() || rank > 8) throw std::runtime_error("qwen38_graph: malformed weight pool line: " + line);
      entry.shape.resize(rank);
      for (auto& d : entry.shape) fields >> d;
      fields >> entry.file;
      if (fields.fail()) throw std::runtime_error("qwen38_graph: malformed weight pool line: " + line);
      std::string placement;
      if (fields >> placement) {
        if (placement != "device" && placement != "cpu" && placement != "host_mapped") {
          throw std::runtime_error("qwen38_graph: invalid weight pool placement '" + placement + "' in line: " + line);
        }
        entry.placement = std::move(placement);
        entry.host_mapped = (entry.placement == "host_mapped");
      }
      if (id != manifest.entries.size()) throw std::runtime_error("qwen38_graph: weight pool ids must be consecutive");
      RequireInsideModelFolder(entry.file, "a weight pool file");
      manifest.entries.push_back(std::move(entry));
    } else if (key == "session") {
      std::string label, name;
      int id = -1;
      fields >> label >> name >> id;
      if (fields.fail() || id < 0 || static_cast<size_t>(id) >= manifest.entries.size())
        throw std::runtime_error("qwen38_graph: weight pool session entry references an unknown pool id: " + line);
      manifest.sessions[label].emplace_back(name, id);
    }
  }
  return manifest;
}

Qwen38QsaConstantsDescription ParseQwen38QsaConstantsIndex(const std::filesystem::path& index_path) {
  std::ifstream in(index_path);
  if (!in) throw std::runtime_error("qwen38_graph: cannot open the QSA constants index " + index_path.string());
  std::string line;
  std::getline(in, line);
  if (line != "qwen38_qsa_consts 1") throw std::runtime_error("qwen38_graph: unsupported QSA constants header: " + line);
  Qwen38QsaConstantsDescription description;
  while (std::getline(in, line)) {
    std::istringstream fields(line);
    std::string key;
    fields >> key;
    if (key == "file") {
      fields >> description.file;
    } else if (key == "epsilon") {
      fields >> description.epsilon;
    } else if (key == "width") {
      fields >> description.width;
    } else if (key == "tensor") {
      Qwen38QsaConstantsDescription::Item item;
      fields >> item.set >> item.name >> item.offset >> item.bytes;
      if (fields.fail()) throw std::runtime_error("qwen38_graph: malformed QSA constants line: " + line);
      description.items.push_back(std::move(item));
    }
  }
  RequireInsideModelFolder(description.file, "the QSA constants file");
  return description;
}

#if defined(QWEN38_GRAPH_NO_POSIX)

Qwen38FileMapping::Qwen38FileMapping(Qwen38Plane& plane, const std::filesystem::path&, bool) : plane_{plane} {
  ThrowUnsupportedPlatform();
}
Qwen38FileMapping::~Qwen38FileMapping() = default;
void Qwen38FileMapping::PrepareNumaRange(size_t, size_t) { ThrowUnsupportedPlatform(); }

Qwen38PleTable::Qwen38PleTable(Qwen38Plane& plane, const std::filesystem::path&, const std::filesystem::path&, int)
    : plane_{plane} {
  ThrowUnsupportedPlatform();
}
Qwen38PleTable::~Qwen38PleTable() = default;

#elif defined(_WIN32)

Qwen38FileMapping::Qwen38FileMapping(Qwen38Plane& plane, const std::filesystem::path& full_path,
                                     bool register_cuda_mapped)
    : plane_{plane}, path_{full_path} {
  HANDLE hFile = CreateFileW(full_path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (hFile == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("qwen38_graph: cannot open weight pool file " + full_path.string());
  }
  LARGE_INTEGER file_size{};
  if (!GetFileSizeEx(hFile, &file_size) || file_size.QuadPart <= 0) {
    CloseHandle(hFile);
    throw std::runtime_error("qwen38_graph: empty or invalid weight pool file " + full_path.string());
  }
  length_ = static_cast<size_t>(file_size.QuadPart);
  HANDLE hMap = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!hMap) {
    CloseHandle(hFile);
    throw std::runtime_error("qwen38_graph: CreateFileMappingW failed for " + full_path.string());
  }
  void* base = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
  if (!base) {
    CloseHandle(hMap);
    CloseHandle(hFile);
    throw std::runtime_error("qwen38_graph: MapViewOfFile failed for " + full_path.string());
  }
  file_handle_ = hFile;
  map_handle_ = hMap;
  base_ = base;
  dev_ptr_ = base;
  if (register_cuda_mapped) {
    int err = plane_.host_register(base_, length_, 2 | 8);
    if (err == 0) {
      registered_ = base_;
      try {
        void* dev_ptr = nullptr;
        plane_.Check(plane_.host_get_device_pointer(&dev_ptr, base_, 0), "cudaHostGetDevicePointer(weight pool file)");
        dev_ptr_ = dev_ptr;
      } catch (...) {
        plane_.host_unregister(registered_);
        UnmapViewOfFile(base_);
        CloseHandle(hMap);
        CloseHandle(hFile);
        registered_ = nullptr;
        base_ = nullptr;
        dev_ptr_ = nullptr;
        map_handle_ = nullptr;
        file_handle_ = nullptr;
        length_ = 0;
        throw;
      }
    } else {
      UnmapViewOfFile(base_);
      CloseHandle(hMap);
      map_handle_ = nullptr;
      base_ = nullptr;
      void* vbuf = VirtualAlloc(nullptr, length_, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
      if (!vbuf || !ReadExactWin32(hFile, 0, vbuf, length_)) {
        if (vbuf) VirtualFree(vbuf, 0, MEM_RELEASE);
        CloseHandle(hFile);
        file_handle_ = nullptr;
        throw std::runtime_error("qwen38_graph: failed to read weight pool file into VirtualAlloc buffer: " +
                                 full_path.string());
      }
      CloseHandle(hFile);
      file_handle_ = nullptr;
      base_ = vbuf;
      dev_ptr_ = vbuf;
      virtual_alloc_ = true;
      try {
        plane_.Check(plane_.host_register(base_, length_, 2), "cudaHostRegister(weight pool file)");
        registered_ = base_;
        void* dev_ptr = nullptr;
        plane_.Check(plane_.host_get_device_pointer(&dev_ptr, base_, 0), "cudaHostGetDevicePointer(weight pool file)");
        dev_ptr_ = dev_ptr;
      } catch (...) {
        if (registered_) plane_.host_unregister(registered_);
        VirtualFree(base_, 0, MEM_RELEASE);
        base_ = nullptr;
        dev_ptr_ = nullptr;
        registered_ = nullptr;
        length_ = 0;
        throw;
      }
    }
  } else if (const char* pin_env = std::getenv("ORTGENAI_PIN_CPU_POOL");
             pin_env != nullptr && std::string(pin_env) == "1") {
    if (plane_.host_register(base_, length_, 1 | 8) == 0) {
      registered_ = base_;
    }
  }
}

void Qwen38FileMapping::PrepareNumaRange(size_t offset, size_t bytes) {
  if (offset > length_ || bytes > length_ - offset)
    throw std::runtime_error("qwen38_graph: NUMA range exceeds weight file");
}

Qwen38FileMapping::~Qwen38FileMapping() {
  if (registered_) plane_.host_unregister(registered_);
  if (base_) {
    if (virtual_alloc_) {
      VirtualFree(base_, 0, MEM_RELEASE);
    } else {
      UnmapViewOfFile(base_);
    }
  }
  if (map_handle_) CloseHandle(static_cast<HANDLE>(map_handle_));
  if (file_handle_) CloseHandle(static_cast<HANDLE>(file_handle_));
}

Qwen38PleTable::Qwen38PleTable(Qwen38Plane& plane, const std::filesystem::path& table_path,
                               const std::filesystem::path& root, int capacity)
    : plane_{plane} {
  Qwen38PleDescription description = ParseQwen38PleDescription(table_path);
  RequireQwen38PleTableSupported(plane, description, table_path);
  params = description.params;
  params.capacity = capacity;
  const auto start = std::chrono::steady_clock::now();
  SYSTEM_INFO si{};
  GetSystemInfo(&si);
  const size_t page = si.dwAllocationGranularity ? static_cast<size_t>(si.dwAllocationGranularity) : 65536u;
  const char* pin_ple_env = std::getenv("ORT_QMOE_WIN_PIN_PLE");
  const bool pin_ple = (pin_ple_env != nullptr && std::string(pin_ple_env) == "1");
  params.pad_ = pin_ple ? 0 : 1;
  long long first_row = 0;
  try {
    for (size_t i = 0; i < description.shards.size(); ++i) {
      const auto& shard = description.shards[i];
      const std::filesystem::path full = root / shard.path;
      HANDLE hFile = CreateFileW(full.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr,
                                 OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
      if (hFile == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("qwen38_graph: cannot open the PLE shard file " + full.string());
      }
      LARGE_INTEGER st_size{};
      if (!GetFileSizeEx(hFile, &st_size) || st_size.QuadPart <= 0) {
        CloseHandle(hFile);
        throw std::runtime_error("qwen38_graph: cannot stat the PLE shard file " + full.string());
      }
      const uint64_t file_size = static_cast<uint64_t>(st_size.QuadPart);
      const uint64_t rows_u64 = static_cast<uint64_t>(shard.rows);
      const uint64_t hd_u64 = static_cast<uint64_t>(params.head_dim);
      if (hd_u64 == 0 || rows_u64 > std::numeric_limits<size_t>::max() / hd_u64) {
        CloseHandle(hFile);
        throw std::runtime_error("qwen38_graph: PLE shard byte size overflows for " + full.string());
      }
      const size_t bytes = static_cast<size_t>(rows_u64 * hd_u64);
      if (shard.offset > file_size || static_cast<uint64_t>(bytes) > file_size - shard.offset) {
        CloseHandle(hFile);
        throw std::runtime_error("qwen38_graph: PLE shard range exceeds file size for " + full.string());
      }
      const uint64_t aligned = (shard.offset / page) * page;
      const size_t delta = static_cast<size_t>(shard.offset - aligned);
      if (bytes > std::numeric_limits<size_t>::max() - delta) {
        CloseHandle(hFile);
        throw std::runtime_error("qwen38_graph: PLE shard mapped length overflows for " + full.string());
      }
      const size_t map_len = bytes + delta;
      HANDLE hMap = CreateFileMappingW(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
      void* base = nullptr;
      if (hMap) {
        base = MapViewOfFile(hMap, FILE_MAP_READ, static_cast<DWORD>(aligned >> 32),
                             static_cast<DWORD>(aligned & 0xFFFFFFFFu), map_len);
      }
      if (base && !pin_ple) {
        auto* rows = static_cast<unsigned char*>(base) + delta;
        mappings_.push_back(Mapping{base, map_len, nullptr, hFile, hMap, false});
        params.shard_ptr[i] = rows;
        first_row += shard.rows;
        params.shard_end[i] = first_row;
        continue;
      }
      if (base) {
        auto* rows = static_cast<unsigned char*>(base) + delta;
        if (plane.host_register(rows, bytes, 2 | 8) == 0) {
          mappings_.push_back(Mapping{base, map_len, rows, hFile, hMap, false});
          void* dev_rows = nullptr;
          plane.Check(plane.host_get_device_pointer(&dev_rows, rows, 0), "cudaHostGetDevicePointer(PLE shard)");
          params.shard_ptr[i] = static_cast<const unsigned char*>(dev_rows);
          first_row += shard.rows;
          params.shard_end[i] = first_row;
          continue;
        }
        UnmapViewOfFile(base);
      }
      if (hMap) CloseHandle(hMap);
      void* vbuf = VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
      if (!vbuf || !ReadExactWin32(hFile, shard.offset, vbuf, bytes)) {
        if (vbuf) VirtualFree(vbuf, 0, MEM_RELEASE);
        CloseHandle(hFile);
        throw std::runtime_error("qwen38_graph: failed to read PLE shard into VirtualAlloc buffer: " + full.string());
      }
      CloseHandle(hFile);
      mappings_.push_back(Mapping{vbuf, bytes, nullptr, nullptr, nullptr, true});
      if (pin_ple) {
        plane.Check(plane.host_register(vbuf, bytes, 2), "cudaHostRegister(PLE shard)");
        mappings_.back().registered = vbuf;
        void* dev_vbuf = nullptr;
        plane.Check(plane.host_get_device_pointer(&dev_vbuf, vbuf, 0), "cudaHostGetDevicePointer(PLE shard)");
        params.shard_ptr[i] = static_cast<const unsigned char*>(dev_vbuf);
      } else {
        params.shard_ptr[i] = static_cast<const unsigned char*>(vbuf);
      }
      first_row += shard.rows;
      params.shard_end[i] = first_row;
    }
  } catch (...) {
    for (auto& mapping : mappings_) {
      if (mapping.registered) plane_.host_unregister(mapping.registered);
      if (mapping.base) {
        if (mapping.virtual_alloc) {
          VirtualFree(mapping.base, 0, MEM_RELEASE);
        } else {
          UnmapViewOfFile(mapping.base);
        }
      }
      if (mapping.map_handle) CloseHandle(static_cast<HANDLE>(mapping.map_handle));
      if (mapping.file_handle) CloseHandle(static_cast<HANDLE>(mapping.file_handle));
    }
    mappings_.clear();
    throw;
  }
  if (plane.set_ple_host_params) plane.set_ple_host_params(&params);
  register_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

Qwen38PleTable::~Qwen38PleTable() {
  if (plane_.unset_ple_host_params) plane_.unset_ple_host_params(&params);
  for (auto& mapping : mappings_) {
    if (mapping.registered) plane_.host_unregister(mapping.registered);
    if (mapping.base) {
      if (mapping.virtual_alloc) {
        VirtualFree(mapping.base, 0, MEM_RELEASE);
      } else {
        UnmapViewOfFile(mapping.base);
      }
    }
    if (mapping.map_handle) CloseHandle(static_cast<HANDLE>(mapping.map_handle));
    if (mapping.file_handle) CloseHandle(static_cast<HANDLE>(mapping.file_handle));
  }
}

#else

Qwen38FileMapping::Qwen38FileMapping(Qwen38Plane& plane, const std::filesystem::path& full_path,
                                     bool register_cuda_mapped)
    : plane_{plane}, path_{full_path} {
  const int fd = open(full_path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("qwen38_graph: cannot open weight pool file " + full_path.string());
  struct stat st{};
  if (fstat(fd, &st) != 0 || st.st_size <= 0) {
    close(fd);
    throw std::runtime_error("qwen38_graph: empty or invalid weight pool file " + full_path.string());
  }
  length_ = static_cast<size_t>(st.st_size);
  void* base = mmap(nullptr, length_, PROT_READ, MAP_SHARED, fd, 0);
  close(fd);
  if (base == MAP_FAILED) throw std::runtime_error("qwen38_graph: mmap failed for " + full_path.string());
  base_ = base;
  if (register_cuda_mapped) {
    try {
      PrepareNumaRange(0, length_);
      plane_.Check(plane_.host_register(base_, length_, 2 | 8), "cudaHostRegister(weight pool file)");
      registered_ = base_;
      plane_.Check(plane_.host_get_device_pointer(&dev_ptr_, base_, 0), "cudaHostGetDevicePointer(weight pool file)");
    } catch (...) {
      if (registered_) plane_.host_unregister(registered_);
      munmap(base_, length_);
      base_ = nullptr;
      length_ = 0;
      registered_ = nullptr;
      dev_ptr_ = nullptr;
      throw;
    }
  } else if (const char* pin_env = std::getenv("ORTGENAI_PIN_CPU_POOL");
             pin_env != nullptr && std::string(pin_env) == "1") {
    try {
      PrepareNumaRange(0, length_);
    } catch (...) {
      munmap(base_, length_);
      base_ = nullptr;
      length_ = 0;
      throw;
    }
    if (plane_.host_register(base_, length_, 1 | 8) == 0) {
      registered_ = base_;
    }
  }
}

void Qwen38FileMapping::PrepareNumaRange(size_t offset, size_t bytes) {
  if (offset > length_ || bytes > length_ - offset)
    throw std::runtime_error("qwen38_graph: NUMA range exceeds weight file");
  const bool interleave = qwen38_numa::Enabled("ORT_QMOE_NUMA_INTERLEAVE", true);
  const bool diagnostics = qwen38_numa::Enabled("ORT_QMOE_NUMA_DIAGNOSTICS", false);
  if ((!interleave && !diagnostics) || bytes == 0 || registered_) return;
  const auto report = qwen38_numa::PrepareFileRange(data() + offset, bytes, offset, interleave,
                                                    qwen38_numa::Enabled("ORT_QMOE_NUMA_MIGRATE", true));
  if (diagnostics || report.policy_error || report.query_error || report.move_failures || report.restore_error) {
    std::ostringstream line;
    line << "[QMoE NUMA] file=" << path_.filename().string() << " offset=" << offset << " bytes=" << bytes
         << " policy_applied=" << report.policy_applied << " pages=" << report.pages;
    for (size_t i = 0; i < report.nodes.size(); ++i)
      line << " N" << report.nodes[i] << "=" << report.pages_before[i] << "->" << report.pages_after[i];
    line << " misplaced=" << report.misplaced_before << "->" << report.misplaced_after
         << " move_attempts=" << report.move_attempts << " move_failures=" << report.move_failures
         << " unknown=" << report.unknown_pages << " policy_errno=" << report.policy_error
         << " query_errno=" << report.query_error << " move_errno=" << report.move_error
         << " restore_errno=" << report.restore_error;
    std::fprintf(stderr, "%s\n", line.str().c_str());
  }
  if (report.restore_error)
    throw std::runtime_error("qwen38_graph: failed to restore the loader thread NUMA policy");
}

Qwen38FileMapping::~Qwen38FileMapping() {
  if (registered_) plane_.host_unregister(registered_);
  if (base_) munmap(base_, length_);
}

Qwen38PleTable::Qwen38PleTable(Qwen38Plane& plane, const std::filesystem::path& table_path,
                               const std::filesystem::path& root, int capacity)
    : plane_{plane} {
  Qwen38PleDescription description = ParseQwen38PleDescription(table_path);
  RequireQwen38PleTableSupported(plane, description, table_path);
  params = description.params;
  params.capacity = capacity;
  const auto start = std::chrono::steady_clock::now();
  const long page = sysconf(_SC_PAGESIZE);
  const size_t page_sz = page > 0 ? static_cast<size_t>(page) : size_t{4096};
  long long first_row = 0;
  try {
    for (size_t i = 0; i < description.shards.size(); ++i) {
      const auto& shard = description.shards[i];
      const std::filesystem::path full = root / shard.path;
      const int fd = open(full.c_str(), O_RDONLY);
      if (fd < 0) throw std::runtime_error("qwen38_graph: cannot open the PLE shard file " + full.string());
      struct stat st{};
      if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        throw std::runtime_error("qwen38_graph: cannot stat the PLE shard file " + full.string());
      }
      const uint64_t file_size = static_cast<uint64_t>(st.st_size);
      const uint64_t rows_u64 = static_cast<uint64_t>(shard.rows);
      const uint64_t hd_u64 = static_cast<uint64_t>(params.head_dim);
      if (hd_u64 == 0 || rows_u64 > std::numeric_limits<size_t>::max() / hd_u64) {
        close(fd);
        throw std::runtime_error("qwen38_graph: PLE shard byte size overflows for " + full.string());
      }
      const size_t bytes = static_cast<size_t>(rows_u64 * hd_u64);
      if (shard.offset > file_size || static_cast<uint64_t>(bytes) > file_size - shard.offset) {
        close(fd);
        throw std::runtime_error("qwen38_graph: PLE shard range exceeds file size for " + full.string());
      }
      const size_t aligned = static_cast<size_t>(shard.offset / page_sz * page_sz);
      const size_t delta = static_cast<size_t>(shard.offset) - aligned;
      if (bytes > std::numeric_limits<size_t>::max() - delta) {
        close(fd);
        throw std::runtime_error("qwen38_graph: PLE shard mapped length overflows for " + full.string());
      }
      const size_t map_len = bytes + delta;
      void* base = mmap(nullptr, map_len, PROT_READ, MAP_SHARED, fd, static_cast<off_t>(aligned));
      close(fd);
      if (base == MAP_FAILED) throw std::runtime_error("qwen38_graph: mmap failed for " + full.string());
      (void)madvise(base, map_len, MADV_WILLNEED);
      mappings_.push_back(Mapping{base, map_len, nullptr});
      auto* rows = static_cast<unsigned char*>(base) + delta;
      // cudaHostRegisterMapped | cudaHostRegisterReadOnly: the gather kernel reads the table zero-copy over PCIe.
      plane.Check(plane.host_register(rows, bytes, 2 | 8), "cudaHostRegister(PLE shard)");
      mappings_.back().registered = rows;
      void* dev_rows = nullptr;
      plane.Check(plane.host_get_device_pointer(&dev_rows, rows, 0), "cudaHostGetDevicePointer(PLE shard)");
      params.shard_ptr[i] = static_cast<const unsigned char*>(dev_rows);
      first_row += shard.rows;
      params.shard_end[i] = first_row;
    }
  } catch (...) {
    for (auto& mapping : mappings_) {
      if (mapping.registered) plane_.host_unregister(mapping.registered);
      if (mapping.base) munmap(mapping.base, mapping.length);
    }
    mappings_.clear();
    throw;
  }
  if (plane.set_ple_host_params) plane.set_ple_host_params(&params);
  register_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

Qwen38PleTable::~Qwen38PleTable() {
  if (plane_.unset_ple_host_params) plane_.unset_ple_host_params(&params);
  for (auto& mapping : mappings_) {
    if (mapping.registered) plane_.host_unregister(mapping.registered);
    if (mapping.base) munmap(mapping.base, mapping.length);
  }
}

#endif

}  // namespace Generators
