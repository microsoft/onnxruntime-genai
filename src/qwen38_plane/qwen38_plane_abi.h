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

// Host-registered FP8 PLE (per-layer embedding) table description, copied to the device once.
typedef struct Qwen38PleParams {
  int n_heads, head_dim, heads_per_ngram, n_shards;
  long long mult[4];
  long long vocab[32];
  long long offset[32];
  long long shard_end[256];
  const unsigned char* shard_ptr[256];
  float scale;
  int pad_;
  long long eos;
  long long capacity;
} Qwen38PleParams;

// X(return type, name, parameter list). The exported symbol is "qwen38_plane_" + name. `stream` is a cudaStream_t.
// CUDA runtime helpers return the cudaError_t as int (0 = success).
//
// prep and ple_rows return 0 on success. A PLE table that is not GPU-addressable (Qwen38PleParams::pad_ == 1) is "staged":
// the token and context words are read back to the host and the FP8 rows are gathered there. Every step of that staging
// (pinned allocation, device alias, copy, synchronisation) is checked; when one fails, or when the table was never
// registered with set_ple_host_params, the function returns the cudaError_t (cudaErrorInvalidValue for the unregistered
// table) before it gathers or launches anything, so the outputs are untouched. Callers must check the result.
//
// The module keeps process-wide state (the staged tables registered through set_ple_host_params) that is shared by every
// loader of the same library file. A loader calls attach once the module is loaded and validated and detach once before it
// unloads it; the module drops that state only when the last attached loader detaches.
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
} Qwen38PlaneTestStats;

// Arms one fault: the (skip + 1)-th execution of `site` from now on fails with `error` instead of calling CUDA, once. A site
// of QWEN38_PLANE_FAULT_NONE disarms. Returns 0, or cudaErrorInvalidValue for an unknown site or an `error` of 0.
typedef int (*Qwen38PlaneTestInjectFaultFn)(int site, int skip, int error);
// Fills `stats`. Returns 0, or cudaErrorInvalidValue for a null `stats`.
typedef int (*Qwen38PlaneTestStatsFn)(Qwen38PlaneTestStats* stats);
