// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Host side of the Qwen3.8-Flash CUDA data plane (libqwen38_plane, see src/qwen38_plane): the dynamic loader, RAII device
// buffers and the host-registered PLE table. Everything that needs POSIX (dlopen, mmap) or a CUDA runtime lives behind the
// plane's C ABI, so this file compiles on every platform; on Windows the loader and the table report that the
// qwen38_flash_graph model type is not supported there.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "qwen38_plane/qwen38_plane_abi.h"

namespace Generators {

// Resolves a POSIX-named shared library path (`libfoo.so` / `foo.so`) to an existing Windows `.dll` sibling
// (`libfoo.dll` or `foo.dll`) when present alongside `path`; otherwise returns `path` unchanged.
std::filesystem::path ResolveWindowsSharedLibraryPath(const std::filesystem::path& path);

// The loaded plane library: one function pointer per entry of QWEN38_PLANE_FUNCTIONS, named like the entry.
//
// The destructor detaches from the module, which stops and joins the threads the module started and releases the staged
// tables once the last loader is gone, but it does not unload the module: a plane library that was loaded and attached stays
// mapped until the process exits, and a loader of the same file that follows (the next model, or the same model loaded again)
// shares that copy. Unloading a library that has launched CUDA kernels and loading it again made the first kernel launch of the
// later copy crash inside the CUDA runtime (a SIGSEGV in libcudart 13.0, in its lazy loading of the module of the kernel), so the
// host never takes that path. A library that is rejected while it loads (a stale ABI, a missing symbol) has launched nothing and
// is released.
struct Qwen38Plane {
  Qwen38Plane() = default;
  // Loads `path`, resolves every symbol and checks the ABI version and structure layout.
  explicit Qwen38Plane(const std::filesystem::path& path);
  ~Qwen38Plane();
  Qwen38Plane(const Qwen38Plane&) = delete;
  Qwen38Plane& operator=(const Qwen38Plane&) = delete;

  // Throws std::runtime_error naming `what` when `error` (a cudaError_t) is not 0.
  void Check(int error, const char* what) const;

#define QWEN38_PLANE_DECLARE(ret, name, args) ret(*name) args{};
  QWEN38_PLANE_FUNCTIONS(QWEN38_PLANE_DECLARE)
#undef QWEN38_PLANE_DECLARE
  void (*quantize_kv)(void* stream, const void* src, void* dst, const void* scale, int rows, int src_pitch_elems,
                      int dst_pitch_elems, int mode){};
  void (*embed_rows)(void* stream, const void* table, const void* ids, void* out, int rows, int width){};
  void (*set_ple_host_params)(const void* params){};
  void (*unset_ple_host_params)(const void* params){};
  // Drops the staged PLE tables of the whole module, i.e. of every loader of the same library file, not just this one.
  // The destructor does not call it: it detaches from the module, which releases that state when the last loader goes.
  void (*release_ple_workspace)(void){};

 private:
  void* handle_{};   // the module; never released once the library is attached (see above)
  bool attached_{};  // attach() has been called and detach() has not
};

// Zero-initialised device allocation owned by RAII.
struct PlaneBuffer {
  PlaneBuffer() = default;
  PlaneBuffer(Qwen38Plane& plane, size_t bytes, void* stream);
  ~PlaneBuffer();
  PlaneBuffer(PlaneBuffer&& other) noexcept;
  PlaneBuffer& operator=(PlaneBuffer&& other) noexcept;
  PlaneBuffer(const PlaneBuffer&) = delete;
  PlaneBuffer& operator=(const PlaneBuffer&) = delete;

  uint8_t* data() const { return ptr_; }
  size_t size() const { return bytes_; }

 private:
  Qwen38Plane* plane_{};
  uint8_t* ptr_{};
  size_t bytes_{};
};

// Pinned host allocation owned by RAII (the result ring and staging words).
struct PlaneHostBuffer {
  PlaneHostBuffer() = default;
  PlaneHostBuffer(Qwen38Plane& plane, size_t bytes);
  ~PlaneHostBuffer();
  PlaneHostBuffer(PlaneHostBuffer&& other) noexcept;
  PlaneHostBuffer& operator=(PlaneHostBuffer&& other) noexcept;
  PlaneHostBuffer(const PlaneHostBuffer&) = delete;
  PlaneHostBuffer& operator=(const PlaneHostBuffer&) = delete;

  int64_t* data() const { return ptr_; }

 private:
  Qwen38Plane* plane_{};
  int64_t* ptr_{};
};

// Parsed PLE table description: the hash/gather parameters and the shard files in row order (the platform-independent half
// of Qwen38PleTable, kept separate so it can be validated without mapping any file).
struct Qwen38PleDescription {
  struct Shard {
    std::string path;  // relative to the model folder
    uint64_t offset{};
    long long rows{};
  };
  Qwen38PleParams params{};  // shard_ptr / shard_end / capacity are filled when the shards are mapped
  std::vector<Shard> shards;
};

// Reads a "qwen38_ple_table 1" description. Throws std::runtime_error on a wrong header, a malformed table, or a shard
// path that is absolute or contains "..": shards must live inside the model folder.
Qwen38PleDescription ParseQwen38PleDescription(const std::filesystem::path& table_path);

// Parsed "qwen38_weights_pool 1" manifest: the initializers that several sessions of the stack share, as tensors stored in
// files of the model folder, plus the (session label -> initializer name, pool id) lists. Kept apart from the weight pool
// that uploads them so that it can be validated without a GPU.
struct Qwen38WeightPoolManifest {
  struct Entry {
    int dtype{};  // ONNX tensor element type
    uint64_t offset{};
    uint64_t length{};
    std::vector<int64_t> shape;
    std::string file;                 // relative to the model folder
    std::string placement{"device"};  // "device", "cpu", or "host_mapped"
    bool host_mapped{false};
  };
  std::vector<Entry> entries;
  std::map<std::string, std::vector<std::pair<std::string, int>>> sessions;
};

// Read-only memory-mapped weight file (optionally registered with CUDA UVA mapped memory).
struct Qwen38FileMapping {
  Qwen38FileMapping(Qwen38Plane& plane, const std::filesystem::path& full_path, bool register_cuda_mapped);
  ~Qwen38FileMapping();
  Qwen38FileMapping(const Qwen38FileMapping&) = delete;
  Qwen38FileMapping& operator=(const Qwen38FileMapping&) = delete;

  uint8_t* data() const { return static_cast<uint8_t*>(base_); }
  uint8_t* device_data() const { return static_cast<uint8_t*>(dev_ptr_ ? dev_ptr_ : base_); }
  size_t size() const { return length_; }
  void PrepareNumaRange(size_t offset, size_t bytes);

 private:
  Qwen38Plane& plane_;
  std::filesystem::path path_;
  void* base_{};
  size_t length_{};
  void* registered_{};
  void* dev_ptr_{};
  void* file_handle_{};
  void* map_handle_{};
  bool virtual_alloc_{};
};

// Throws std::runtime_error on a wrong header, non-consecutive pool ids, an unknown pool id in a session line, or a file
// path that is absolute or contains "..": the files must live inside the model folder.
Qwen38WeightPoolManifest ParseQwen38WeightPoolManifest(const std::filesystem::path& manifest_path);

// Parsed "qwen38_qsa_consts 1" index of the dense -> packed hand-off constants (key-norm weights and RoPE tables).
struct Qwen38QsaConstantsDescription {
  struct Item {
    std::string set;   // "target" or "head"
    std::string name;  // "cos", "sin" or "knorm.<layer>"
    size_t offset{};
    size_t bytes{};
  };
  std::string file;  // relative to the model folder
  float epsilon{};
  int width{};
  std::vector<Item> items;
};

Qwen38QsaConstantsDescription ParseQwen38QsaConstantsIndex(const std::filesystem::path& index_path);

// Host-registered FP8 PLE shards. The table file (written by tools/qwen38_graph/build_package.py) lists the shard files;
// each is mapped read-only and registered with the CUDA driver so that the gather kernel reads it zero-copy over PCIe.
struct Qwen38PleTable {
  // `table_path` is the description file, `root` the model folder that relative shard paths resolve against (and may not
  // leave), `capacity` the KV capacity (positions the kernels may write).
  Qwen38PleTable(Qwen38Plane& plane, const std::filesystem::path& table_path, const std::filesystem::path& root,
                 int capacity);
  ~Qwen38PleTable();
  Qwen38PleTable(const Qwen38PleTable&) = delete;
  Qwen38PleTable& operator=(const Qwen38PleTable&) = delete;

  Qwen38PleParams params{};
  double register_seconds{};

 private:
  struct Mapping {
    void* base{};
    size_t length{};
    void* registered{};
    void* file_handle{};
    void* map_handle{};
    bool virtual_alloc{};
  };
  Qwen38Plane& plane_;
  std::vector<Mapping> mappings_;
};

}  // namespace Generators
