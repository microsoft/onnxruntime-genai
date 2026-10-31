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
#include "qwen38_replay_geometry.h"

namespace Generators {

// Resolves a POSIX-named shared library path (`libfoo.so` / `foo.so`) to an existing Windows `.dll` sibling
// (`libfoo.dll` or `foo.dll`) when present alongside `path`; otherwise returns `path` unchanged.
std::filesystem::path ResolveWindowsSharedLibraryPath(const std::filesystem::path& path);

// True in a Windows build: the host that translates a POSIX library name in a package to the `.dll` the package ships.
#if defined(_WIN32)
inline constexpr bool kQwen38WindowsHost = true;
#else
inline constexpr bool kQwen38WindowsHost = false;
#endif

// Resolves model.qwen38_graph.plane_library (`configured`, relative to `model_root`) to the standalone plane library to
// load, or to an empty path when the plane built into onnxruntime-genai-cuda is to be used: nothing is configured, or the
// configured library is not in the package. On a Windows host a configured `libfoo.so` / `foo.so` stands for the
// `libfoo.dll` / `foo.dll` the package ships, so that alias is looked up (ResolveWindowsSharedLibraryPath, the translation
// the loader applies) before the library is judged missing. The library, and the alias, must resolve inside `model_root`,
// symlinks included (std::runtime_error otherwise). `windows_host` is a parameter so that the decision of a Windows host
// can be tested on every platform; the model passes kQwen38WindowsHost.
std::filesystem::path ResolveQwen38PlaneLibraryPath(const std::filesystem::path& model_root,
                                                    const std::string& configured, bool windows_host);

// A CPU embedding table that the plane gathers rows from (the model's embed_tokens, mapped from <embedding>.data).
struct Qwen38EmbeddingTable {
  const void* data{};  // device-addressable base of the table
  long long rows{};    // rows in the table: the model's vocabulary size
  int width{};         // elements per row: the model's hidden size
  bool int8{};         // per-row INT8 payload followed by one float scale per row; bfloat16 rows otherwise
};

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
  // An empty plane with no library loaded; the function pointers are filled in by the caller (tests).
  Qwen38Plane() = default;
  // Loads `path` (or the built-in onnxruntime-genai-cuda library when `path` is empty), resolves every symbol and checks
  // the ABI version and structure layout.
  explicit Qwen38Plane(const std::filesystem::path& path);
  ~Qwen38Plane();
  Qwen38Plane(const Qwen38Plane&) = delete;
  Qwen38Plane& operator=(const Qwen38Plane&) = delete;

  // Throws std::runtime_error naming `what` when `error` (a cudaError_t) is not 0.
  void Check(int error, const char* what) const;

  // Throws std::runtime_error when the library lacks any of the QWEN38_PLANE_CAP_* bits in `required`. The message reads
  // "<required_by> requires the plane capability <names>, which the plane library <path> does not provide (it reports: ...)",
  // with `required_by` naming what the package selected (e.g. "the INT8 PLE table <path>"), so a package that needs a newer
  // library fails when the model is loaded instead of running on a library that would misread it.
  void RequireCapabilities(unsigned long long required, const std::string& required_by) const;

  // Gathers `n` rows of `table` for the int64 token ids at `ids` into the bfloat16 `out` ([n, table.width]) on `stream`. Uses
  // embed_gather, which takes the row count of the table; a library without it only has the legacy BF16 embed_rows, and
  // throws (naming the int8_embedding capability) for an INT8 table.
  void EmbedRows(void* stream, const Qwen38EmbeddingTable& table, const void* ids, void* out, int n) const;

  // prep for the `rows` rows of one folded verify step of each of `batch` request slots in one call (the optional export
  // qwen38_plane_prep_batched; the layout of the arrays is documented in the ABI header). Throws std::invalid_argument for a
  // `rows` or `batch` out of range, std::runtime_error naming the batched_prep or ctx_slot_stride capability for a library
  // without it, and std::runtime_error with the CUDA error the plane returns.
  void PrepBatched(void* stream, const void* params, int rows, int batch, const void* ctx_in, void* ctx_out, void* input_ids,
                   void* position_ids, void* committed, void* past_pos, void* ple_out, void* state) const;

#define QWEN38_PLANE_DECLARE(ret, name, args) ret(*name) args{};
  QWEN38_PLANE_FUNCTIONS(QWEN38_PLANE_DECLARE)
#undef QWEN38_PLANE_DECLARE
  // Optional exports: null in a library that predates them (see "Optional exports and capabilities" in the ABI header).
  void (*quantize_kv)(void* stream, const void* src, void* dst, const void* scale, int rows, int src_pitch_elems,
                      int dst_pitch_elems, int mode){};
  void (*embed_rows)(void* stream, const void* table, const void* ids, void* out, int rows, int width){};
  Qwen38PlaneEmbedGatherFn embed_gather{};
  Qwen38PlanePrepBatchedFn prep_batched{};
  void (*set_ple_host_params)(const void* params){};
  void (*unset_ple_host_params)(const void* params){};
  // Drops the staged PLE tables of the whole module, i.e. of every loader of the same library file, not just this one.
  // The destructor does not call it: it detaches from the module, which releases that state when the last loader goes.
  void (*release_ple_workspace)(void){};
  Qwen38PlaneDflashSelectFn dflash_select{};
  Qwen38PlaneReplayStateUpdatesFn replay_state_updates{};

  // The QWEN38_PLANE_CAP_* bits the library reports; 0 for a library that predates qwen38_plane_capabilities.
  unsigned long long capabilities{};
  // The file the library was loaded from (named by the capability diagnostics).
  std::string library_path;

 private:
  void* handle_{};   // the module; never released once the library is attached (see above)
  bool attached_{};  // attach() has been called and detach() has not
};

// The names of the QWEN38_PLANE_CAP_* bits set in `mask`, comma separated ("none" for 0); bits this build has no name for
// are listed as one hexadecimal value.
std::string DescribeQwen38PlaneCapabilities(unsigned long long mask);

// The descriptor of the compact replay of one CausalConvWithState state (past_key_values.<layer>.conv_state | ple_conv_state)
// of a request slot: `request` (kind kConvolution, the batch-1 shapes of the slot) is validated by
// ValidateQwen38ReplayDescriptor, and `source` (the old state), `destination` (the new state) and `value` (the slot's
// state_update output) are the device pointers of the slot. The descriptor carries the layout of the state. A history-first
// state [batch, history, channels] needs the conv_history_first_replay capability of `plane`: a library without it reads every
// conv state as channel-first and would corrupt that one on a partial acceptance, so this throws a std::runtime_error naming
// the capability, the state and the library (Qwen38Plane::RequireCapabilities) instead of handing it the descriptor. A
// channel-first state needs no capability, so such a package keeps loading with a library that predates it.
Qwen38StateReplayDesc MakeQwen38ConvReplayDescriptor(const Qwen38Plane& plane, const Qwen38ReplayRequest& request,
                                                     const void* source, void* destination, const void* value);

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

// Reads a "qwen38_ple_table 1" description. Throws std::runtime_error on a wrong header, a malformed table, a missing, zero or
// non-finite scale (positive for FP8 shards, negative for INT8 shards), or a shard path that is absolute or contains "..":
// shards must live inside the model folder.
Qwen38PleDescription ParseQwen38PleDescription(const std::filesystem::path& table_path);

// Throws std::runtime_error, naming the missing int8_ple capability and the library, when the shards of the table are INT8
// (a negative scale) and `plane` would decode them as FP8. Checked before any shard is mapped.
void RequireQwen38PleTableSupported(const Qwen38Plane& plane, const Qwen38PleDescription& description,
                                    const std::filesystem::path& table_path);

// The layout of a CPU embedding table file (<model.embedding.filename>.data).
enum class Qwen38EmbeddingFormat {
  kNone,  // neither layout below: nothing may gather rows from the file
  kBf16,  // rows x width bfloat16
  kInt8,  // rows x width int8, followed by one float scale per row (per-row symmetric)
};

// The size in bytes of a `rows` x `width` table in `format` (0 for kNone).
size_t Qwen38EmbeddingTableBytes(Qwen38EmbeddingFormat format, long long rows, long long width);

// The most that a file may be larger than the INT8 layout and still be read as that layout: the 4096-byte alignment that
// external data can be padded to. (Two layouts that differ by no more than this cannot be told apart by size.)
constexpr size_t kQwen38Int8EmbeddingSlackBytes = 4096;

// Decides the layout of the embedding table file `table_path` of `file_bytes` bytes for a vocabulary of `rows` and a hidden
// size of `width`: bfloat16 when the file holds at least that many bytes, INT8 when it is the size of the INT8 layout plus
// at most kQwen38Int8EmbeddingSlackBytes, otherwise kNone. The size is all that tells the layouts apart, so a file between
// the two sizes (a truncated bfloat16 table, or a vocabulary that is not the one of the file) is kNone, not INT8, which
// would gather garbage from it.
// Throws std::runtime_error when the plane cannot gather the table: when it is INT8 and `plane` lacks the int8_embedding
// capability (the message names it and the library), when a bfloat16 table has a width that is not a multiple of 8 (rows
// are copied in 16-byte vectors), or when the float scales that follow the int8 rows would not be 4-byte aligned
// (rows * width is not a multiple of 4). The model constructor calls it when the table is mapped, i.e. at model load.
Qwen38EmbeddingFormat SelectQwen38EmbeddingFormat(const Qwen38Plane& plane, const std::filesystem::path& table_path,
                                                  size_t file_bytes, long long rows, long long width);

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
