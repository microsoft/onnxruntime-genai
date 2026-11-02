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

// The number of int64 words the `state` buffer of prep (the `ctl` buffer of a fold) must have room for; a larger buffer is fine.
// The kernels of the plane keep their words in 0..6, and words 8 and 9 hold the draft tokens of rows 2 and 3 of a fold (the
// layout is in qwen38_plane.cu). With a staged PLE table (see prep below) row 0 of a fold (pos_add == 0, tok_ptr == state,
// pos_ptr == state + 1) copies all of these words to the host in one go, and rows 1..3 (pos_add 1..3, tok_ptr == state + 2,
// state + 8, state + 9) are then prepared from that copy. The library is given the buffer as a pointer and cannot check its
// size: a smaller buffer is an out-of-bounds read of the device allocation.
#define QWEN38_PLANE_STATE_MIN_WORDS 10

// Host-registered FP8 PLE (per-layer embedding) table description, copied to the device once. The library trusts the loader
// for its invariants and does not validate them (ParseQwen38PleDescription enforces them): 1 <= n_heads <= 32,
// n_heads * head_dim <= 3072 (one 1024-thread block gathers a row, and a staged row must fit an 8192-byte ring slot),
// 0 <= heads_per_ngram <= n_heads, vocab[h] > 0 with the rows of every head inside the shards, 1 <= n_shards <= 256 with
// ascending shard_end, and a non-null shard_ptr for every shard.
// pad_ is not padding: 1 marks a "staged" table, one the GPU cannot address, whose rows the host gathers (see below).
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

#ifdef __cplusplus
// The structure crosses the module boundary by layout, so a change to it must bump QWEN38_PLANE_ABI_VERSION.
static_assert(sizeof(void*) != 8 || sizeof(Qwen38PleParams) == 4680, "Qwen38PleParams layout changed");
static_assert(sizeof(void*) != 8 || offsetof(Qwen38PleParams, shard_ptr) == 2608, "Qwen38PleParams layout changed");
static_assert(sizeof(void*) != 8 || offsetof(Qwen38PleParams, pad_) == 4660, "Qwen38PleParams layout changed");
static_assert(sizeof(void*) != 8 || offsetof(Qwen38PleParams, eos) == 4664, "Qwen38PleParams layout changed");
static_assert(sizeof(void*) != 8 || offsetof(Qwen38PleParams, capacity) == 4672, "Qwen38PleParams layout changed");
#endif

// X(return type, name, parameter list). The exported symbol is "qwen38_plane_" + name. `stream` is a cudaStream_t.
// CUDA runtime helpers return the cudaError_t as int (0 = success).
//
// prep and ple_rows return 0 on success. A PLE table that is not GPU-addressable (Qwen38PleParams::pad_ == 1) is "staged":
// the token and context words are read back to the host and the FP8 rows are gathered there. Every step of that staging
// (pinned allocation, device alias, creation of the completion event, copy, synchronisation) is checked; when one fails, or
// when the table was never registered with set_ple_host_params, the function returns the cudaError_t (cudaErrorInvalidValue
// for the unregistered table) before it gathers or launches anything, so the outputs are untouched. Both also return the
// error of their own kernel launch. The one step that can fail after the launch is recording the completion event, which
// fences the staged buffers against a call on another stream: the function then waits for its stream before it returns the
// error, so the kernel has finished and the outputs are complete. Callers must check the result. ple_rows reads ctx_in while
// its last block writes ctx_out, so the two must not alias.
//
// A staged table is bound to one GPU. The first call that stages anything for it creates its pinned buffers and its completion
// events with the device that is current in the calling thread, and every later call that uses the table has to be made with
// that device current: an event can only be recorded on a stream of its own device, and these events are what fence the
// buffers against a call on another stream. A call made with another device current returns cudaErrorInvalidDevice before it
// stages, gathers or launches anything, so its outputs are untouched and the table keeps working on its own GPU. Using one
// table on several GPUs is not supported. The binding ends when the staged state of the table is dropped (by
// unset_ple_host_params, by release_ple_workspace or by the detach of the last loader), after which registering the table
// again with set_ple_host_params lets it bind to another GPU; registering a table that is still registered keeps the binding.
// A resident table (pad_ != 1) is not bound to a GPU.
//
// A staged table is not usable while the stream is being captured into a CUDA graph: the host gathers its rows when the call
// is made, which a replay would not repeat, and the read backs and waits of the call are not allowed on a capturing stream.
// The call returns cudaErrorStreamCaptureUnsupported before it copies, gathers or launches anything. Telling a staged table
// from a resident one takes a read back of the descriptor, which a capturing stream cannot make either, so the first call
// that uses a descriptor, and the first one after set_ple_host_params (it forgets what was learnt about every descriptor),
// has to be made outside a capture; a capture that meets a descriptor not yet told apart fails the same way.
//
// The other kernel exports return void. A launch the CUDA runtime rejects (bad configuration, no kernel image for the
// device, a failed lazy module load) leaves the stream healthy, so the failure is kept and reported, once, by the next
// stream_sync.
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
// ("qwen38_plane_test_inject_staged_fault", "qwen38_plane_test_stats", "qwen38_plane_test_peek_last_error",
// "qwen38_plane_test_stream_capture", "qwen38_plane_test_device_count") to fail one CUDA step of the staged PLE path, to observe
// what the module did, to see the last CUDA error it left on the calling thread, to call it while a stream is being captured
// and to count the GPUs.

// The step of the staged PLE path that an injected fault replaces with an error.
typedef enum Qwen38PlaneFaultSite {
  QWEN38_PLANE_FAULT_NONE = 0,
  QWEN38_PLANE_FAULT_HOST_ALLOC = 1,    // pinned cudaHostAlloc of a staging buffer
  QWEN38_PLANE_FAULT_HOST_MALLOC = 2,   // malloc of the host n-gram row cache
  QWEN38_PLANE_FAULT_DEVICE_ALIAS = 3,  // cudaHostGetDevicePointer of a mapped staging buffer
  QWEN38_PLANE_FAULT_COPY = 4,          // device-to-host cudaMemcpyAsync
  QWEN38_PLANE_FAULT_SYNC = 5,          // cudaStreamSynchronize / cudaDeviceSynchronize
  QWEN38_PLANE_FAULT_EVENT_CREATE = 6,  // cudaEventCreateWithFlags of a completion event (only when it is created)
  QWEN38_PLANE_FAULT_EVENT_RECORD = 7,  // cudaEventRecord of a completion event, after the kernel was launched
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
  long long unfenced_drains;          // waits for a stream because its completion event could not be recorded
} Qwen38PlaneTestStats;

// Arms one fault: the (skip + 1)-th execution of `site` from now on fails with `error` instead of calling CUDA, once. A site
// of QWEN38_PLANE_FAULT_NONE disarms. Returns 0, or cudaErrorInvalidValue for an unknown site or an `error` of 0.
typedef int (*Qwen38PlaneTestInjectFaultFn)(int site, int skip, int error);
// Fills `stats`. Returns 0, or cudaErrorInvalidValue for a null `stats`.
typedef int (*Qwen38PlaneTestStatsFn)(Qwen38PlaneTestStats* stats);
// Returns the last CUDA error of the calling thread without clearing it (cudaPeekAtLastError).
typedef int (*Qwen38PlaneTestPeekLastErrorFn)(void);
// Starts (`begin` != 0) or ends the capture of `stream` into a CUDA graph that is thrown away, in the thread-local mode so
// that only the calling thread is affected. Returns the CUDA error of the call, 0 on success.
typedef int (*Qwen38PlaneTestStreamCaptureFn)(void* stream, int begin);
// Returns the number of CUDA devices, 0 when there is none. Unlike set_device, a failure leaves no error behind as the last
// error of the calling thread, so a test can look before it commits to a device.
typedef int (*Qwen38PlaneTestDeviceCountFn)(void);
