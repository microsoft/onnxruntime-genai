// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// C ABI of libqwen38_plane (the CUDA data plane of the Qwen3.8-Flash CUDA-graph stack). The library is loaded at run time
// by the core GenAI library, which therefore never links the CUDA runtime. This header is included by both sides: the
// CUDA translation unit (qwen38_plane.cu) and the host loader (src/models/qwen38_graph_plane.h), so the structure layout
// and the function list cannot drift apart. It must stay plain C-compatible (no GenAI or CUDA includes).
#pragma once

#include <stddef.h>
#include <stdint.h>

// 3: prep and ple_rows return the CUDA error of the staged PLE path (they returned void); attach/detach were added.
#define QWEN38_PLANE_ABI_VERSION 3

// Host-registered PLE (per-layer embedding) table description, copied to the device once.
typedef struct Qwen38PleParams {
  int n_heads, head_dim, heads_per_ngram, n_shards;
  long long mult[4];
  long long vocab[32];
  long long offset[32];
  long long shard_end[256];
  const unsigned char* shard_ptr[256];
  // Dequantisation scale of the shard bytes, which are FP8 E4M3 when it is positive. A negative value marks INT8 shards
  // whose scale is |scale| (QWEN38_PLANE_CAP_INT8_PLE); a library without that capability would decode them as FP8.
  float scale;
  int pad_;
  long long eos;
  long long capacity;
} Qwen38PleParams;

// Device descriptor for compact state-update replay (GatedDeltaNet, CausalConvWithState, PackedSparseAttentionIndexer).
typedef struct Qwen38StateReplayDesc {
  const void* source_state;
  void* destination_state;
  const void* value;
  const float* decay;
  const float* key;
  const float* delta;
  const void* source_aux_state;
  void* destination_aux_state;
  const int32_t* source_lengths;
  int32_t* destination_lengths;
  uint64_t channel_count;
  uint64_t state_width;
  uint64_t key_width;
  uint64_t key_head_count;
  uint64_t state_capacity;
  uint64_t aux_capacity;
  uint32_t capacity;
  uint32_t kept_count;
  uint32_t element_size;
  uint32_t compress_ratio;
  uint32_t kind;
} Qwen38StateReplayDesc;

// Limits of the compact state-update replay kernels (k_replay_gdn, k_replay_generic). The plane sizes its shared tiles and its
// grid from them and the host checks every replay descriptor against them while it builds it (ValidateQwen38ReplayDescriptor),
// so a geometry the kernels cannot hold fails the model load instead of silently corrupting the recurrent state.
#define QWEN38_PLANE_MAX_REPLAY_TRANSITIONS 8  // transitions one replay keeps: the capacity of the state_update outputs
#define QWEN38_PLANE_MAX_REPLAY_KEY_WIDTH 256  // GatedDeltaNet key width: a row of the key tile (a multiple of 4: float4 loads)
#define QWEN38_PLANE_REPLAY_ROWS_PER_BLOCK 32  // value rows of a GatedDeltaNet state one thread block of k_replay_gdn replays

// Limits of the folded multi-row prep (prep with a negative pos_add, prep_batched).
#define QWEN38_PLANE_MAX_PREP_ROWS 8        // rows of one folded verify step
#define QWEN38_PLANE_MAX_PREP_BATCH 8       // request slots prep_batched prepares in one call
#define QWEN38_PLANE_CTL_WORDS_PER_SLOT 16  // int64 control words of one request slot (the `state` of prep)

// X(return type, name, parameter list). The exported symbol is "qwen38_plane_" + name. `stream` is a cudaStream_t.
// CUDA runtime helpers return the cudaError_t as int (0 = success).
//
// prep and ple_rows return 0 on success. A PLE table that is not GPU-addressable (Qwen38PleParams::pad_ == 1) is "staged":
// the token and context words are read back to the host and the FP8 rows are gathered there (a gather of 32 or more rows by
// worker threads that the module starts when it first needs them). Every step of that staging (pinned allocation, device
// alias, copy, synchronisation, starting the worker threads, which fails with cudaErrorMemoryAllocation) is checked; when one
// fails, or when the table was never registered with set_ple_host_params, the function returns the cudaError_t
// (cudaErrorInvalidValue for the unregistered table) before it gathers or launches anything, so the outputs are untouched.
// Callers must check the result.
//
// prep prepares the graph inputs of one row: the token is read from tok_ptr and the position from pos_ptr plus pos_add. With
// QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH a pos_add in [-8, -2] instead prepares the rows = -pos_add rows of one folded verify
// step in a single call, from the control words `state` (int64): row r takes its token from word {0, 2, 8, 9, 10, 11, 12, 13}[r]
// and runs at position state[1] + r, and its n-gram context follows the tokens of the rows before it (row 0 starts from
// ctx_in). For row r it writes input_ids[r] and position_ids[r] (int64), past_pos[position], ple_out + r * n_heads * head_dim
// (bfloat16) and ctx_out[2r .. 2r + 1] (the token before it, then its own). Only row 0 writes committed[0] (int32), and a null
// committed pointer skips it, as for one row; a position outside the table sets bit 0 of state[5], as for one row. A pos_add
// below -8 is not a row count: prep returns cudaErrorInvalidValue before it launches anything. Several request slots in one
// call are prepared by prep_batched (QWEN38_PLANE_CAP_BATCHED_PREP), never by an encoding inside pos_add.
//
// The module keeps process-wide state (the staged tables registered through set_ple_host_params, and the worker threads of the
// staged gather) that is shared by every loader of the same library file. A loader calls attach once the module is loaded and
// validated and detach once before it unloads it; the module drops that state, and stops and joins its threads, only when the
// last attached loader detaches. A host must therefore detach before it releases the library (FreeLibrary / dlclose). The
// reference host, Qwen38Plane, detaches and then keeps the library mapped until the process exits (see qwen38_graph_plane.h).
#define QWEN38_PLANE_FUNCTIONS(X)                                                                                  \
  X(int, abi_version, (void))                                                                                      \
  X(int, sizeof_ple_params, (void))                                                                                \
  X(int, set_device, (int device))                                                                                 \
  X(const char*, error_string, (int error))                                                                        \
  X(int, device_alloc, (void** ptr, size_t bytes))                                                                 \
  X(int, device_free, (void* ptr))                                                                                 \
  X(int, host_alloc, (void** ptr, size_t bytes))                                                                   \
  X(int, host_free, (void* ptr))                                                                                   \
  X(int, host_register, (void* ptr, size_t bytes, unsigned flags))                                                 \
  X(int, host_unregister, (void* ptr))                                                                             \
  X(int, host_get_device_pointer, (void** device_ptr, void* host_ptr, unsigned flags))                             \
  X(int, device_memset, (void* ptr, int value, size_t bytes, void* stream))                                        \
  X(int, memcpy_h2d, (void* dst, const void* src, size_t bytes, void* stream))                                     \
  X(int, memcpy_d2h, (void* dst, const void* src, size_t bytes, void* stream))                                     \
  X(int, memcpy_d2d, (void* dst, const void* src, size_t bytes, void* stream))                                     \
  X(int, stream_sync, (void* stream))                                                                              \
  X(int, memcpy2d_d2d,                                                                                             \
    (void* dst, size_t dst_pitch, const void* src, size_t src_pitch, size_t width, size_t height, void* stream))   \
  X(int, stream_create, (void** stream))                                                                           \
  X(int, stream_destroy, (void* stream))                                                                           \
  X(int, prep,                                                                                                     \
    (void* stream, const void* params, const void* tok_ptr, const void* pos_ptr, int pos_add, const void* ctx_in,  \
     void* ctx_out, void* input_ids, void* position_ids, void* committed, void* past_pos, void* ple_out,           \
     void* state))                                                                                                 \
  X(void, argmax_part, (void* stream, const void* logits, int n, void* out_values, void* out_indices, int blocks)) \
  X(void, post_tok,                                                                                                \
    (void* stream, const void* tokens01, void* state, void* ctx_commit, const void* ctx_a, const void* ctx_b,      \
     const void* status, int status_count, void* result, int ring))                                                \
  X(void, post1,                                                                                                   \
    (void* stream, const void* values, const void* indices, int blocks, void* state, void* ctx_commit,             \
     const void* ctx_a, const void* status, int status_count, void* result, int ring))                             \
  X(void, head_prep,                                                                                               \
    (void* stream, const void* state, int rows, int head_length, int hidden_width, const void* hidden_src,         \
     void* hidden_dst, void* tokens, void* positions, void* committed))                                            \
  X(void, head_post, (void* stream, const void* next_token, void* state))                                          \
  X(void, or_status, (void* stream, const void* status, int status_count, void* state))                            \
  X(int, ple_rows,                                                                                                 \
    (void* stream, const void* params, const void* tokens, int n, const void* ctx_in, void* ctx_out,               \
     void* ple_out))                                                                                               \
  X(void, pool_keys,                                                                                               \
    (void* stream, const void* raw, int blocks, const void* key_norm, const void* cos, const void* sin, int width, \
     float epsilon, int first_block, void* out))                                                                   \
  X(void, transpose_f32, (void* stream, const void* src, void* dst, int n, int batch))                             \
  X(void, attach, (void))                                                                                          \
  X(void, detach, (void))

// ---- Optional exports and capabilities ---------------------------------------------------------------------------------
// A feature added after ABI version 3 does not bump the version: a library that predates it has to stay loadable for the
// packages that do not use it. Instead the library reports the features it was built with through the optional export
// "qwen38_plane_capabilities" (Qwen38PlaneCapabilitiesFn) as the OR of the QWEN38_PLANE_CAP_* bits below
// (QWEN38_PLANE_ALL_CAPABILITIES for a library built from this header). The loader resolves it without requiring it, so a
// library that predates the export reports 0, and keeps the mask in Qwen38Plane::capabilities. Wherever a package selects a
// feature, the loader calls Qwen38Plane::RequireCapabilities(bits, what), which fails the model load with a diagnostic naming
// the missing capability, what required it and the library path, instead of letting the package run on a library that would
// misread it.
//
// To add a capability: define the next free bit, add it to QWEN38_PLANE_CAPABILITIES (the name is what the diagnostic prints;
// the library reports every listed bit with no further change), implement it, call RequireCapabilities where the loader
// selects it, and cover a library with and a library without the bit in the qwen38_plane tests.
#define QWEN38_PLANE_CAP_INT8_EMBEDDING 0x1ull  // embed_gather: per-row INT8 embedding tables (any row count)
#define QWEN38_PLANE_CAP_INT8_PLE 0x2ull        // a negative Qwen38PleParams::scale marks INT8 PLE shards
// prep: a pos_add in [-8, -2] prepares the -pos_add rows of one folded verify step in a single call (see the notes on prep
// above). A library without it reads that pos_add as an offset of one row and leaves the other rows of the step stale.
#define QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH 0x4ull
// prep_batched: the optional export "qwen38_plane_prep_batched" (below), the folded prep of several request slots in one call.
// Only a model with model.qwen38_graph.max_batch_size > 1 needs it, so a library without it still loads every other package.
#define QWEN38_PLANE_CAP_BATCHED_PREP 0x8ull

// X(bit, name), in bit order.
#define QWEN38_PLANE_CAPABILITIES(X)                                   \
  X(QWEN38_PLANE_CAP_INT8_EMBEDDING, "int8_embedding")                 \
  X(QWEN38_PLANE_CAP_INT8_PLE, "int8_ple")                             \
  X(QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH, "negative_pos_add_batch") \
  X(QWEN38_PLANE_CAP_BATCHED_PREP, "batched_prep")

#define QWEN38_PLANE_CAPABILITY_BIT_(bit, name) | (bit)
#define QWEN38_PLANE_ALL_CAPABILITIES (0ull QWEN38_PLANE_CAPABILITIES(QWEN38_PLANE_CAPABILITY_BIT_))

typedef unsigned long long (*Qwen38PlaneCapabilitiesFn)(void);

// Layout of the embedding table that embed_gather reads.
typedef enum Qwen38PlaneEmbeddingFormat {
  QWEN38_PLANE_EMBEDDING_BF16 = 0,      // table_rows x width bfloat16
  QWEN38_PLANE_EMBEDDING_INT8_ROW = 1,  // table_rows x width int8, followed by table_rows float scales (one per row)
} Qwen38PlaneEmbeddingFormat;

// Optional export "qwen38_plane_embed_gather" (QWEN38_PLANE_CAP_INT8_EMBEDDING): enqueues the gather of n rows of an
// embedding table into `out` ([n, width] bfloat16; INT8 rows are multiplied by their row scale). `table` is device
// addressable and holds table_rows rows of `width` elements in `format` (a Qwen38PlaneEmbeddingFormat); `ids` holds n int64
// token ids on the device, and an id outside [0, table_rows) reads row 0. Both that bound and the start of the INT8 row
// scales (table + table_rows * width bytes) come from table_rows, the vocabulary size of the model, so the plane has no
// vocabulary size of its own. Returns 0, including for n <= 0 (nothing is launched), or cudaErrorInvalidValue before it
// launches anything for table_rows <= 0, width <= 0, an unknown format, or an INT8 table whose scales would not be 4-byte
// aligned (table_rows * width not a multiple of 4).
typedef int (*Qwen38PlaneEmbedGatherFn)(void* stream, const void* table, long long table_rows, int format, const void* ids,
                                        void* out, int n, int width);

// Optional export "qwen38_plane_embed_rows" (void; stream, table, ids, out, rows, width): the gather of libraries that predate
// embed_gather. It reads a BF16 table of an unknown row count, so it applies no upper bound to the ids, and it does nothing
// for a width <= 0 (it has no INT8 mode). The loader uses it only for a BF16 table when the library has no embed_gather.

// Optional export "qwen38_plane_prep_batched" (QWEN38_PLANE_CAP_BATCHED_PREP): prep of the `rows` rows of one folded verify
// step (see prep; rows == 1 is the plain next-token row) for each of `batch` request slots, in one call. `rows` is in
// [1, QWEN38_PLANE_MAX_PREP_ROWS] and `batch` in [1, QWEN38_PLANE_MAX_PREP_BATCH]; the pointers are those of prep except
// that every one is the base of an array over the slots, which are numbered 0 .. batch - 1:
//   state         int64, QWEN38_PLANE_CTL_WORDS_PER_SLOT words per slot, the control words of the slot (prep's `state`)
//   ctx_in        int64[batch][2]                       the n-gram context in front of row 0
//   ctx_out       int64[batch][rows][2]                 as for prep, slot after slot
//   input_ids, position_ids   int64[batch][rows]
//   committed     int32[batch], or null to skip         row 0 of the slot writes its committed length
//   past_pos      int64[batch][Qwen38PleParams::capacity]
//   ple_out       bfloat16[batch][rows][n_heads * head_dim]
// A slot whose control word state[3] is nonzero is muted: its rows are token 0 at position r with eos contexts and a zero PLE
// row, so a slot that takes no part in the step still feeds the graph valid inputs, and its control words are left alone. A
// table that is staged on the host prepares the rows of every slot from its control words, muted or not (those of a muted
// slot hold token 0 at position 0 after the engine mutes it, plus the draft tokens it kept, all valid inputs); the caller
// ignores the rows of a muted slot either way.
// Returns 0, or the cudaError_t: cudaErrorInvalidValue before anything is launched for a `rows` or `batch` out of range or a
// null pointer other than `committed`, and, for a PLE table staged on the host, the errors named for prep.
typedef int (*Qwen38PlanePrepBatchedFn)(void* stream, const void* params, int rows, int batch, const void* ctx_in, void* ctx_out,
                                        void* input_ids, void* position_ids, void* committed, void* past_pos, void* ple_out,
                                        void* state);

// ---- Test hooks -------------------------------------------------------------------------------------------------------
// Compiled into the library only when it is built with QWEN38_PLANE_TEST_HOOKS (the CMake option ENABLE_TESTS). They are not
// part of QWEN38_PLANE_FUNCTIONS and the host loader never resolves them: the unit tests look them up by name
// ("qwen38_plane_test_inject_staged_fault", "qwen38_plane_test_stats") to fail one CUDA step of the staged PLE path and to
// observe what the module did.

// The step of the staged PLE path that an injected fault replaces with an error.
typedef enum Qwen38PlaneFaultSite {
  QWEN38_PLANE_FAULT_NONE = 0,
  QWEN38_PLANE_FAULT_HOST_ALLOC = 1,    // pinned cudaHostAlloc of a staging buffer
  QWEN38_PLANE_FAULT_HOST_MALLOC = 2,   // malloc of the host n-gram row cache
  QWEN38_PLANE_FAULT_DEVICE_ALIAS = 3,  // cudaHostGetDevicePointer of a mapped staging buffer
  QWEN38_PLANE_FAULT_COPY = 4,          // device-to-host cudaMemcpyAsync
  QWEN38_PLANE_FAULT_SYNC = 5,          // cudaStreamSynchronize / cudaDeviceSynchronize
  QWEN38_PLANE_FAULT_THREAD_START = 6,  // creation of a gather worker thread (std::thread throws std::system_error)
} Qwen38PlaneFaultSite;

typedef struct Qwen38PlaneTestStats {
  long long gather_rows;              // host gathers of a PLE row
  long long staged_launches;          // staged kernels launched
  long long global_cleanups;          // releases of the whole staged state of the module
  long long live_pinned_allocations;  // pinned staging buffers currently allocated
  long long prefill_capacity_rows;    // prefill row capacity summed over the staged workspaces
  long long staged_tables;            // registered staged tables
  long long attached_loaders;         // loaders currently attached to the module
  long long fault_armed;              // 1 while an injected fault has not fired yet
  long long gather_pool_threads;      // live worker threads of the staged PLE gather
} Qwen38PlaneTestStats;

// Arms one fault: the (skip + 1)-th execution of `site` from now on fails with `error` instead of calling CUDA, once. A site
// of QWEN38_PLANE_FAULT_NONE disarms. Returns 0, or cudaErrorInvalidValue for an unknown site or an `error` of 0.
typedef int (*Qwen38PlaneTestInjectFaultFn)(int site, int skip, int error);
// Fills `stats`. Returns 0, or cudaErrorInvalidValue for a null `stats`.
typedef int (*Qwen38PlaneTestStatsFn)(Qwen38PlaneTestStats* stats);
