// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "qwen38_graph_plane.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>

// The plane loader and the PLE mapping need POSIX (dlopen, mmap). Elsewhere the same entry points report that the model type
// is unsupported; QWEN38_GRAPH_NO_POSIX forces that variant on any platform (used to compile-check it).
#if defined(_WIN32) && !defined(QWEN38_GRAPH_NO_POSIX)
#define QWEN38_GRAPH_NO_POSIX
#endif

#if !defined(QWEN38_GRAPH_NO_POSIX)
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
  throw std::runtime_error("The qwen38_flash_graph model type is only supported on Linux with the CUDA execution provider.");
}
#endif

}  // namespace

// ------------------------------------------------------------------------------------------------
// Plane library
// ------------------------------------------------------------------------------------------------
#if defined(QWEN38_GRAPH_NO_POSIX)

Qwen38Plane::Qwen38Plane(const std::filesystem::path&) { ThrowUnsupportedPlatform(); }
Qwen38Plane::~Qwen38Plane() = default;

#else

Qwen38Plane::Qwen38Plane(const std::filesystem::path& path) {
  handle_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle_) {
    const char* reason = dlerror();
    throw std::runtime_error("qwen38_graph: failed to load the plane library " + path.string() + ": " +
                             (reason ? reason : "unknown error"));
  }
  auto load = [this](auto& function, const char* symbol_name) {
    void* symbol = dlsym(handle_, symbol_name);
    if (!symbol) {
      throw std::runtime_error(std::string("qwen38_graph: the plane library lacks the symbol ") + symbol_name);
    }
    function = reinterpret_cast<std::remove_reference_t<decltype(function)> >(symbol);
  };
  try {
#define QWEN38_PLANE_LOAD(ret, name, args) load(name, "qwen38_plane_" #name);
    QWEN38_PLANE_FUNCTIONS(QWEN38_PLANE_LOAD)
#undef QWEN38_PLANE_LOAD
    quantize_kv = reinterpret_cast<decltype(quantize_kv)>(dlsym(handle_, "qwen38_plane_quantize_kv"));
    if (abi_version() != QWEN38_PLANE_ABI_VERSION) {
      throw std::runtime_error("qwen38_graph: the plane library has ABI version " + std::to_string(abi_version()) +
                               ", expected " + std::to_string(QWEN38_PLANE_ABI_VERSION));
    }
    if (sizeof_ple_params() != static_cast<int>(sizeof(Qwen38PleParams))) {
      throw std::runtime_error("qwen38_graph: the plane library's PLE parameter layout differs from the host definition");
    }
  } catch (...) {
    dlclose(handle_);
    handle_ = nullptr;
    throw;
  }
}

Qwen38Plane::~Qwen38Plane() {
  if (handle_) dlclose(handle_);
}

#endif

void Qwen38Plane::Check(int error, const char* what) const {
  if (error != 0) {
    throw std::runtime_error(std::string("qwen38_graph: CUDA error in ") + what + ": " + error_string(error));
  }
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

Qwen38PleTable::Qwen38PleTable(Qwen38Plane& plane, const std::filesystem::path&, const std::filesystem::path&, int)
    : plane_{plane} {
  ThrowUnsupportedPlatform();
}
Qwen38PleTable::~Qwen38PleTable() = default;

#else

Qwen38FileMapping::Qwen38FileMapping(Qwen38Plane& plane, const std::filesystem::path& full_path,
                                     bool register_cuda_mapped)
    : plane_{plane} {
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
  }
}

Qwen38FileMapping::~Qwen38FileMapping() {
  if (registered_) plane_.host_unregister(registered_);
  if (base_) munmap(base_, length_);
}

Qwen38PleTable::Qwen38PleTable(Qwen38Plane& plane, const std::filesystem::path& table_path,
                               const std::filesystem::path& root, int capacity)
    : plane_{plane} {
  Qwen38PleDescription description = ParseQwen38PleDescription(table_path);
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
  register_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

Qwen38PleTable::~Qwen38PleTable() {
  for (auto& mapping : mappings_) {
    if (mapping.registered) plane_.host_unregister(mapping.registered);
    if (mapping.base) munmap(mapping.base, mapping.length);
  }
}

#endif

}  // namespace Generators
