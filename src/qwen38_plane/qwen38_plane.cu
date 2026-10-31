// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
/* Qwen3.8-Flash CUDA data plane for GenAI (model.type "qwen38_flash_graph"). The kernels are the research data plane of
 * CG/mtp_runner/dataplane/mtp_plane.cu (unchanged arithmetic and launch shapes); the device_* / host_* / memcpy_* functions at
 * the end are the CUDA runtime helpers the GenAI model code needs (allocation, host registration, copies, sync) so that the
 * core GenAI library does not link cudart. The exported C ABI is described in qwen38_plane_abi.h.
 *
 * Device data plane for the MTP(1) loop.  All kernels take device pointers only (no host scalars that change per cycle),
 * so every sequence below can be captured into a CUDA graph.
 *
 *  k_prep         : write graph-row inputs for one token: input_ids, position_ids, committed_length, the in-place
 *                   past_position_ids slot, the PLE embedding row (n-gram hash + gather from the host-registered FP8
 *                   table + dequantisation; bit-identical to Qwen4ExpCPUPLELookup) and the next PLE context.
 *  k_argmax_part  : stage 1 of a deterministic (smallest index on ties) argmax over bf16 logits.
 *  k_post_tok     : verify epilogue: accept test, token/state outputs, status OR, publish a result record to pinned
 *                   memory (the host spin-waits on result[0]).
 *  k_post1        : plain S=1 step epilogue (argmax merge, commit one token, publish).
 *  k_head_prep/post: MTP head launch preparation and draft publication.
 *  k_or_status    : accumulate validation status words (deferred check during prefill).
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <new>
#include <system_error>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <memoryapi.h>
#endif

#include "qwen38_parallel_pool.h"
#include "qwen38_plane_abi.h"

typedef long long i64;
typedef Qwen38PleParams PleParams;

// Every CUDA step of the staged PLE path (pinned allocation, device aliasing, device-to-host copy, synchronisation) goes
// through the helpers below, which report the cudaError_t and clear CUDA's last-error state on failure. Built with
// QWEN38_PLANE_TEST_HOOKS (see qwen38_plane_abi.h) they can also be made to fail on demand and are counted; otherwise they
// forward to CUDA. The hook state is guarded by g_staged_ple_mu like the staged state itself.
#ifdef QWEN38_PLANE_TEST_HOOKS
struct StagedTestHooks {
  int fault_site = QWEN38_PLANE_FAULT_NONE;
  int fault_skip = 0;
  int fault_error = 0;
  long long gather_rows = 0;
  long long staged_launches = 0;
  long long global_cleanups = 0;
  long long live_pinned = 0;
  long long unfenced_drains = 0;
};
static StagedTestHooks g_test_hooks;

static inline cudaError_t hook_fault(int site) {
  if (g_test_hooks.fault_site != site) return cudaSuccess;
  if (g_test_hooks.fault_skip > 0) {
    --g_test_hooks.fault_skip;
    return cudaSuccess;
  }
  g_test_hooks.fault_site = QWEN38_PLANE_FAULT_NONE;
  return (cudaError_t)g_test_hooks.fault_error;
}
static inline void hook_gather(long long rows = 1) { g_test_hooks.gather_rows += rows; }
static inline void hook_launch() { ++g_test_hooks.staged_launches; }
static inline void hook_cleanup() { ++g_test_hooks.global_cleanups; }
static inline void hook_pinned(int delta) { g_test_hooks.live_pinned += delta; }
static inline void hook_unfenced_drain() { ++g_test_hooks.unfenced_drains; }
#else
static inline cudaError_t hook_fault(int) { return cudaSuccess; }
static inline void hook_gather(long long = 1) {}
static inline void hook_launch() {}
static inline void hook_cleanup() {}
static inline void hook_pinned(int) {}
static inline void hook_unfenced_drain() {}
#endif

template <typename T>
static inline cudaError_t staged_host_alloc(T** ptr, size_t bytes, unsigned int flags) {
  void* raw = nullptr;
  cudaError_t err = hook_fault(QWEN38_PLANE_FAULT_HOST_ALLOC);
  if (err == cudaSuccess) err = cudaHostAlloc(&raw, bytes, flags);
  if (err != cudaSuccess) {
    cudaGetLastError();
    *ptr = nullptr;
    return err;
  }
  *ptr = static_cast<T*>(raw);
  hook_pinned(1);
  return cudaSuccess;
}

static inline void staged_host_free(void* ptr) {
  if (ptr == nullptr) return;
  if (cudaFreeHost(ptr) != cudaSuccess) cudaGetLastError();
  hook_pinned(-1);
}

static inline cudaError_t staged_malloc(unsigned char** ptr, size_t bytes) {
  *ptr = nullptr;
  cudaError_t err = hook_fault(QWEN38_PLANE_FAULT_HOST_MALLOC);
  if (err == cudaSuccess) {
    *ptr = static_cast<unsigned char*>(malloc(bytes));
    if (*ptr == nullptr) err = cudaErrorMemoryAllocation;
  }
  return err;
}

// Device address of a mapped pinned buffer. The host pointer is never substituted for it: a failure, or a null result, is an
// error.
static inline cudaError_t staged_device_alias(unsigned char** device_ptr, void* host_ptr) {
  void* raw = nullptr;
  cudaError_t err = hook_fault(QWEN38_PLANE_FAULT_DEVICE_ALIAS);
  if (err == cudaSuccess) err = cudaHostGetDevicePointer(&raw, host_ptr, 0);
  if (err == cudaSuccess && raw == nullptr) err = cudaErrorInvalidValue;
  if (err != cudaSuccess) {
    cudaGetLastError();
    return err;
  }
  *device_ptr = static_cast<unsigned char*>(raw);
  return cudaSuccess;
}

static inline cudaError_t staged_copy_to_host(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind,
                                              cudaStream_t stream) {
  cudaError_t err = hook_fault(QWEN38_PLANE_FAULT_COPY);
  if (err == cudaSuccess) err = cudaMemcpyAsync(dst, src, bytes, kind, stream);
  if (err != cudaSuccess) cudaGetLastError();
  return err;
}

static inline cudaError_t staged_stream_sync(cudaStream_t stream) {
  cudaError_t err = hook_fault(QWEN38_PLANE_FAULT_SYNC);
  if (err == cudaSuccess) err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) cudaGetLastError();
  return err;
}

static inline cudaError_t staged_device_sync() {
  cudaError_t err = hook_fault(QWEN38_PLANE_FAULT_SYNC);
  if (err == cudaSuccess) err = cudaDeviceSynchronize();
  if (err != cudaSuccess) cudaGetLastError();
  return err;
}

// Whether `stream` is being captured into a CUDA graph. A failed query counts as not capturing: whatever the call does next on
// the stream then reports the same failure.
static inline bool staged_stream_is_capturing(cudaStream_t stream) {
  cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
  if (cudaStreamIsCapturing(stream, &status) != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return status != cudaStreamCaptureStatusNone;
}

// Creates the completion event of a staged buffer once. A call that cannot get one fails before it touches anything: without
// the event a call on another stream would not know to wait for the kernel that reads the buffer.
static inline cudaError_t staged_ensure_done_event(cudaEvent_t* event) {
  if (*event != nullptr) return cudaSuccess;
  cudaError_t err = hook_fault(QWEN38_PLANE_FAULT_EVENT_CREATE);
  if (err == cudaSuccess) err = cudaEventCreateWithFlags(event, cudaEventDisableTiming);
  if (err != cudaSuccess) {
    cudaGetLastError();
    *event = nullptr;
  }
  return err;
}

// Records the completion event after the kernel that reads a staged buffer was launched on `stream`, and marks the buffer in
// flight. When the event cannot be recorded the kernel cannot be waited for through it later, so the stream is drained here
// instead, which leaves nothing in flight, and the error is returned: the outputs are complete but the call has failed.
static inline cudaError_t staged_record_done(cudaEvent_t event, cudaStream_t stream, cudaStream_t* last_stream,
                                             bool* in_flight) {
  cudaError_t err = hook_fault(QWEN38_PLANE_FAULT_EVENT_RECORD);
  if (err == cudaSuccess) err = cudaEventRecord(event, stream);
  if (err == cudaSuccess) {
    *last_stream = stream;
    *in_flight = true;
    return cudaSuccess;
  }
  cudaGetLastError();
  *in_flight = false;
  hook_unfenced_drain();
  static_cast<void>(staged_stream_sync(stream));
  return err;
}

struct StagedPleWorkspace {
  PleParams params{};
  i64* scalars = nullptr;                  // pinned [64]: tok/ctl words + ctx_in
  unsigned char* prep_ring = nullptr;      // pinned mapped [kPrepRingRows * kPrepRingRowBytes]
  unsigned char* prep_ring_dev = nullptr;  // device alias for prep_ring
  unsigned int ring_idx = 0;
  i64* prefill_tokens = nullptr;             // pinned [cap_rows]
  unsigned char* prefill_fp8 = nullptr;      // pinned mapped [cap_rows * max_total]
  unsigned char* prefill_fp8_dev = nullptr;  // device alias for prefill_fp8
  size_t prefill_cap_rows = 0;
  size_t prefill_cap_bytes = 0;
  const void* cached_state_ptr = nullptr;
  const void* cached_ctx_out_ptr = nullptr;
  i64 cached_c0 = 0;
  i64 cached_c1 = 0;
  cudaEvent_t prep_done_event = nullptr;
  cudaStream_t last_prep_stream = nullptr;
  bool prep_in_flight = false;
  cudaEvent_t prefill_done_event = nullptr;
  cudaStream_t last_prefill_stream = nullptr;
  bool prefill_in_flight = false;
  static constexpr size_t kCacheSlots = 16384;
  static constexpr size_t kMaxRowBytes = 3072;
  static constexpr size_t kPrepRingRows = 64;        // rows of the prep ring (a power of two)
  static constexpr size_t kPrepRingRowBytes = 8192;  // bytes of one row of the prep ring
  unsigned char* ngram_cache_data = nullptr;
  i64 ngram_cache_keys[kCacheSlots][3]{};
  unsigned char ngram_cache_valid[kCacheSlots]{};

  // The words of the fourteen-word `ctl` snapshot that rows 1..7 of a fold read from `scalars` instead of the device.
  // Whatever else writes `scalars` has to drop the snapshot first, or the next fold row would be prepared from words
  // that are not its own.
  void invalidate_fold_snapshot() {
    cached_state_ptr = nullptr;
    cached_ctx_out_ptr = nullptr;
  }

  void reset_cache() {
    invalidate_fold_snapshot();
    cached_c0 = 0;
    cached_c1 = 0;
    memset(ngram_cache_valid, 0, sizeof(ngram_cache_valid));
  }

  bool has_cuda_allocations() const {
    return scalars != nullptr || prep_ring != nullptr || prefill_tokens != nullptr || prefill_fp8 != nullptr ||
           prep_done_event != nullptr || prefill_done_event != nullptr;
  }

  void release(bool sync_device) {
    if (sync_device && has_cuda_allocations()) {
      if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
      }
    }
    if (prep_done_event) {
      if (cudaEventDestroy(prep_done_event) != cudaSuccess) {
        cudaGetLastError();
      }
      prep_done_event = nullptr;
    }
    last_prep_stream = nullptr;
    prep_in_flight = false;
    if (prefill_done_event) {
      if (cudaEventDestroy(prefill_done_event) != cudaSuccess) {
        cudaGetLastError();
      }
      prefill_done_event = nullptr;
    }
    last_prefill_stream = nullptr;
    prefill_in_flight = false;
    if (scalars) {
      staged_host_free(scalars);
      scalars = nullptr;
    }
    if (prep_ring) {
      staged_host_free(prep_ring);
      prep_ring = nullptr;
      prep_ring_dev = nullptr;
    }
    if (prefill_tokens) {
      staged_host_free(prefill_tokens);
      prefill_tokens = nullptr;
    }
    if (prefill_fp8) {
      staged_host_free(prefill_fp8);
      prefill_fp8 = nullptr;
      prefill_fp8_dev = nullptr;
    }
    if (ngram_cache_data) {
      free(ngram_cache_data);
      ngram_cache_data = nullptr;
    }
    ring_idx = 0;
    prefill_cap_rows = 0;
    prefill_cap_bytes = 0;
    reset_cache();
  }

  ~StagedPleWorkspace() { release(false); }
};

static std::mutex g_staged_ple_mu;
static std::unordered_map<const void*, std::unique_ptr<StagedPleWorkspace>> g_host_ple_workspaces;
static std::unordered_map<const void*, const void*> g_dev_to_host_ple;
static const void* g_last_staged_host_key = nullptr;
// Loaders (Qwen38Plane objects, possibly several per process) that attach()-ed to this module and have not detached yet.
static int g_attached_loaders = 0;

// Pinned buffers that ensure_staged_buffers has allocated but the workspace does not own yet.
struct PendingStagedBuffers {
  i64* scalars = nullptr;
  unsigned char* prep_ring = nullptr;
  unsigned char* prep_ring_dev = nullptr;
  unsigned char* ngram_cache_data = nullptr;
  i64* prefill_tokens = nullptr;
  unsigned char* prefill_fp8 = nullptr;
  unsigned char* prefill_fp8_dev = nullptr;

  void discard() {
    staged_host_free(scalars);
    staged_host_free(prep_ring);
    free(ngram_cache_data);
    staged_host_free(prefill_tokens);
    staged_host_free(prefill_fp8);
    *this = PendingStagedBuffers{};
  }
};

// Gives the workspace the pinned buffers one call needs: the scalar words and the prep ring and, for rows > 0, a token buffer
// and a mapped FP8 buffer for `rows` rows of `row_bytes` bytes. All or nothing: every new buffer is allocated, and its
// device alias verified, before any of them is published, so on failure what was allocated is freed and the workspace,
// including its recorded capacities, is exactly as it was.
static cudaError_t ensure_staged_buffers(StagedPleWorkspace& ws, size_t rows, size_t row_bytes) {
  constexpr size_t kMinPrefillRows = 4096;
  constexpr size_t kMinPrefillBytes = 4096ULL * 4096ULL;
  if (rows != 0 && row_bytes != 0 && rows > SIZE_MAX / row_bytes) {
    return cudaErrorInvalidValue;
  }
  const size_t need_bytes = rows * row_bytes;
  PendingStagedBuffers fresh;
  size_t new_rows = ws.prefill_cap_rows;
  size_t new_bytes = ws.prefill_cap_bytes;
  bool replaces_buffer = false;
  cudaError_t err = cudaSuccess;
  if (!ws.scalars) {
    err = staged_host_alloc(&fresh.scalars, 64 * sizeof(i64), cudaHostAllocPortable);
  }
  if (err == cudaSuccess && !ws.prep_ring) {
    err = staged_host_alloc(&fresh.prep_ring, StagedPleWorkspace::kPrepRingRows * StagedPleWorkspace::kPrepRingRowBytes,
                            cudaHostAllocMapped | cudaHostAllocPortable);
    if (err == cudaSuccess) err = staged_device_alias(&fresh.prep_ring_dev, fresh.prep_ring);
  }
  if (err == cudaSuccess && !ws.ngram_cache_data) {
    err = staged_malloc(&fresh.ngram_cache_data, StagedPleWorkspace::kCacheSlots * StagedPleWorkspace::kMaxRowBytes);
  }
  if (err == cudaSuccess && rows != 0 && (!ws.prefill_tokens || rows > ws.prefill_cap_rows)) {
    new_rows = rows < kMinPrefillRows ? kMinPrefillRows : rows;
    err = staged_host_alloc(&fresh.prefill_tokens, new_rows * sizeof(i64), cudaHostAllocPortable);
    replaces_buffer = replaces_buffer || ws.prefill_tokens != nullptr;
  }
  if (err == cudaSuccess && rows != 0 && (!ws.prefill_fp8 || need_bytes > ws.prefill_cap_bytes)) {
    new_bytes = need_bytes < kMinPrefillBytes ? kMinPrefillBytes : need_bytes;
    err = staged_host_alloc(&fresh.prefill_fp8, new_bytes, cudaHostAllocMapped | cudaHostAllocPortable);
    if (err == cudaSuccess) err = staged_device_alias(&fresh.prefill_fp8_dev, fresh.prefill_fp8);
    replaces_buffer = replaces_buffer || ws.prefill_fp8 != nullptr;
  }
  // The buffers being replaced may still be read by kernels of earlier calls: wait for them before they are freed.
  if (err == cudaSuccess && replaces_buffer) {
    err = staged_device_sync();
    if (err == cudaSuccess) {
      ws.prep_in_flight = false;
      ws.prefill_in_flight = false;
    }
  }
  if (err != cudaSuccess) {
    fresh.discard();
    return err;
  }

  if (fresh.scalars) ws.scalars = fresh.scalars;
  if (fresh.prep_ring) {
    ws.prep_ring = fresh.prep_ring;
    ws.prep_ring_dev = fresh.prep_ring_dev;
  }
  if (fresh.ngram_cache_data) {
    ws.ngram_cache_data = fresh.ngram_cache_data;
    memset(ws.ngram_cache_valid, 0, sizeof(ws.ngram_cache_valid));
  }
  if (fresh.prefill_tokens) {
    staged_host_free(ws.prefill_tokens);
    ws.prefill_tokens = fresh.prefill_tokens;
    ws.prefill_cap_rows = new_rows;
  }
  if (fresh.prefill_fp8) {
    staged_host_free(ws.prefill_fp8);
    ws.prefill_fp8 = fresh.prefill_fp8;
    ws.prefill_fp8_dev = fresh.prefill_fp8_dev;
    ws.prefill_cap_bytes = new_bytes;
  }
  return cudaSuccess;
}

// Where head `h` of the PLE row for the n-gram (c0, c1, tok) sits in the host table.
static inline const unsigned char* resolve_ple_head_ptr(const PleParams& HP, int h, i64 tok, i64 c0, i64 c1) {
  unsigned long long mixed = (unsigned long long)tok * (unsigned long long)HP.mult[0];
  mixed ^= (unsigned long long)c1 * (unsigned long long)HP.mult[1];
  if (h >= HP.heads_per_ngram) mixed ^= (unsigned long long)c0 * (unsigned long long)HP.mult[2];
  const i64 v = HP.vocab[h];
  i64 r = (i64)mixed % v;
  if (r < 0) r += v;
  const i64 id = r + HP.offset[h];
  int lo = 0, hi = HP.n_shards - 1;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (HP.shard_end[mid] > id)
      hi = mid;
    else
      lo = mid + 1;
  }
  const i64 first = lo == 0 ? 0 : HP.shard_end[lo - 1];
  const i64 row_idx = id - first;
  return HP.shard_ptr[lo] + (size_t)row_idx * (size_t)HP.head_dim;
}

static inline size_t ple_ngram_cache_slot(i64 tok, i64 c0, i64 c1) {
  const unsigned long long h =
      ((unsigned long long)tok * 11400714819323198485ULL) ^
      ((unsigned long long)c1 * 14029467366897019727ULL) ^
      ((unsigned long long)c0 * 1609587929392839161ULL);
  return (size_t)(h & (StagedPleWorkspace::kCacheSlots - 1));
}

static inline void host_gather_ple_row(StagedPleWorkspace& ws, i64 tok, i64 s_c0, i64 s_c1, unsigned char* dst_fp8) {
  hook_gather();
  const PleParams& HP = ws.params;
  const i64 c1 = s_c1;
  const i64 c0 = (c1 == HP.eos) ? HP.eos : s_c0;
  const int hd = HP.head_dim;
  const size_t total_bytes = (size_t)HP.n_heads * (size_t)hd;
  size_t cache_slot = 0;
  const bool use_cache = (ws.ngram_cache_data != nullptr && total_bytes <= StagedPleWorkspace::kMaxRowBytes);
  if (use_cache) {
    cache_slot = ple_ngram_cache_slot(tok, c0, c1);
    if (ws.ngram_cache_valid[cache_slot] &&
        ws.ngram_cache_keys[cache_slot][0] == tok &&
        ws.ngram_cache_keys[cache_slot][1] == c0 &&
        ws.ngram_cache_keys[cache_slot][2] == c1) {
      memcpy(dst_fp8, ws.ngram_cache_data + cache_slot * StagedPleWorkspace::kMaxRowBytes, total_bytes);
      return;
    }
  }
  for (int h = 0; h < HP.n_heads; ++h) {
    memcpy(dst_fp8 + (size_t)h * (size_t)hd, resolve_ple_head_ptr(HP, h, tok, c0, c1), (size_t)hd);
  }
  if (use_cache) {
    memcpy(ws.ngram_cache_data + cache_slot * StagedPleWorkspace::kMaxRowBytes, dst_fp8, total_bytes);
    ws.ngram_cache_keys[cache_slot][0] = tok;
    ws.ngram_cache_keys[cache_slot][1] = c0;
    ws.ngram_cache_keys[cache_slot][2] = c1;
    ws.ngram_cache_valid[cache_slot] = 1;
  }
}

struct PleGatherTask {
  StagedPleWorkspace* ws = nullptr;
  const i64* h_toks = nullptr;
  i64 ctx0 = 0;
  i64 ctx1 = 0;
  int n = 0;
  size_t row_stride = 0;
  unsigned char* dst_base = nullptr;
  std::atomic<int> next_row{0};
};

// Creates a gather worker. A test build can make the creation fail (QWEN38_PLANE_FAULT_THREAD_START), as std::thread does when
// the system is out of threads.
static std::thread start_gather_thread(std::function<void()> entry) {
  if (hook_fault(QWEN38_PLANE_FAULT_THREAD_START) != cudaSuccess) {
    throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again));
  }
  return std::thread(std::move(entry));
}

// The worker threads of the staged PLE gather. They run code of this module, so release_all_staged_state_locked stops and
// joins them before the last loader detaches, i.e. before the host unloads the library.
struct PleGatherPool {
  qwen38_plane::ParallelPool pool{qwen38_plane::DefaultParallelWorkerCount(std::thread::hardware_concurrency()),
                                  start_gather_thread};

  static void process_task(PleGatherTask* t) {
    StagedPleWorkspace& ws = *t->ws;
    const PleParams& HP = ws.params;
    const int hd = HP.head_dim;
    const int nh = HP.n_heads;
    const size_t total_bytes = (size_t)nh * (size_t)hd;
    const bool use_cache = (ws.ngram_cache_data != nullptr && total_bytes <= StagedPleWorkspace::kMaxRowBytes);
    // Locals, not thread_local: the caller of the pool is a host thread, and a thread_local with a destructor would leave that
    // destructor, which is code of this module, to run when the host thread exits, perhaps after the module was unloaded.
#ifdef _WIN32
    std::vector<WIN32_MEMORY_RANGE_ENTRY> ranges;
#endif
    std::vector<const unsigned char*> miss_srcs;
    std::vector<unsigned char*> miss_dsts;
    constexpr int kSpan = 32;

    while (true) {
      const int i0 = t->next_row.fetch_add(kSpan, std::memory_order_relaxed);
      if (i0 >= t->n) break;
      const int i1 = (i0 + kSpan < t->n) ? (i0 + kSpan) : t->n;
#ifdef _WIN32
      ranges.clear();
      ranges.reserve((size_t)(i1 - i0) * (size_t)nh);
#endif
      miss_srcs.clear();
      miss_dsts.clear();
      miss_srcs.reserve((size_t)(i1 - i0) * (size_t)nh);
      miss_dsts.reserve((size_t)(i1 - i0) * (size_t)nh);

      for (int i = i0; i < i1; ++i) {
        const i64 tok = t->h_toks[i];
        const i64 c1 = i >= 1 ? t->h_toks[i - 1] : t->ctx1;
        const i64 c0raw = i >= 2 ? t->h_toks[i - 2] : (i == 1 ? t->ctx1 : t->ctx0);
        const i64 c0 = (c1 == HP.eos) ? HP.eos : c0raw;
        unsigned char* dst_fp8 = t->dst_base + (size_t)i * t->row_stride;
        if (use_cache) {
          const size_t cache_slot = ple_ngram_cache_slot(tok, c0, c1);
          if (ws.ngram_cache_valid[cache_slot] &&
              ws.ngram_cache_keys[cache_slot][0] == tok &&
              ws.ngram_cache_keys[cache_slot][1] == c0 &&
              ws.ngram_cache_keys[cache_slot][2] == c1) {
            memcpy(dst_fp8, ws.ngram_cache_data + cache_slot * StagedPleWorkspace::kMaxRowBytes, total_bytes);
            continue;
          }
        }
        for (int h = 0; h < nh; ++h) {
          const unsigned char* src = resolve_ple_head_ptr(HP, h, tok, c0, c1);
          miss_srcs.push_back(src);
          miss_dsts.push_back(dst_fp8 + (size_t)h * (size_t)hd);
#ifdef _WIN32
          ranges.push_back({const_cast<unsigned char*>(src), (SIZE_T)hd});
#endif
        }
      }

      for (size_t k = 0; k < miss_srcs.size(); ++k) {
        memcpy(miss_dsts[k], miss_srcs[k], (size_t)hd);
      }
#ifdef _WIN32
      for (const auto& r : ranges) {
        (void)VirtualUnlock(r.VirtualAddress, r.NumberOfBytes);
      }
#endif
    }
  }

  // Gathers the PLE rows of the `n` token ids at `h_toks` into `dst_base`, `row_stride` bytes apart, on every worker and on the
  // calling thread. Returns cudaErrorMemoryAllocation, with nothing gathered, when the worker threads cannot be started.
  cudaError_t run(StagedPleWorkspace& ws, const i64* h_toks, i64 ctx0, i64 ctx1, int n, unsigned char* dst_base,
                  size_t row_stride) {
    PleGatherTask t;
    t.ws = &ws;
    t.h_toks = h_toks;
    t.ctx0 = ctx0;
    t.ctx1 = ctx1;
    t.n = n;
    t.row_stride = row_stride;
    t.dst_base = dst_base;
    t.next_row.store(0, std::memory_order_relaxed);

    if (pool.Run([&t] { process_task(&t); })) return cudaErrorMemoryAllocation;

    const PleParams& HP = ws.params;
    const size_t total_bytes = (size_t)HP.n_heads * (size_t)HP.head_dim;
    if (ws.ngram_cache_data != nullptr && total_bytes <= StagedPleWorkspace::kMaxRowBytes) {
      const int start_i = (n > 2048) ? (n - 2048) : 0;
      for (int i = start_i; i < n; ++i) {
        const i64 tok = h_toks[i];
        const i64 c1 = i >= 1 ? h_toks[i - 1] : ctx1;
        const i64 c0raw = i >= 2 ? h_toks[i - 2] : (i == 1 ? ctx1 : ctx0);
        const i64 c0 = (c1 == HP.eos) ? HP.eos : c0raw;
        const size_t cache_slot = ple_ngram_cache_slot(tok, c0, c1);
        memcpy(ws.ngram_cache_data + cache_slot * StagedPleWorkspace::kMaxRowBytes, dst_base + (size_t)i * row_stride,
               total_bytes);
        ws.ngram_cache_keys[cache_slot][0] = tok;
        ws.ngram_cache_keys[cache_slot][1] = c0;
        ws.ngram_cache_keys[cache_slot][2] = c1;
        ws.ngram_cache_valid[cache_slot] = 1;
      }
    }
    return cudaSuccess;
  }
};

// The pool is created by the first large staged gather and destroyed, its workers stopped and joined, by
// release_all_staged_state_locked. Guarded by g_staged_ple_mu. It is a plain pointer on purpose: nothing may run (and join
// threads) when the module is unloaded or the process exits, so a pool is only ever stopped at that explicit teardown.
static PleGatherPool* g_ple_gather_pool = nullptr;

// Returns nullptr when the pool cannot be allocated. Caller holds g_staged_ple_mu.
static PleGatherPool* get_ple_gather_pool_locked() {
  if (g_ple_gather_pool == nullptr) {
    try {
      g_ple_gather_pool = new PleGatherPool();
    } catch (const std::bad_alloc&) {
      return nullptr;
    }
  }
  return g_ple_gather_pool;
}

// Stops and joins the gather workers. Caller holds g_staged_ple_mu, which every gather holds for its whole duration, so none is
// in flight.
static void stop_ple_gather_pool_locked() {
  delete g_ple_gather_pool;  // ~ParallelPool() stops and joins the workers
  g_ple_gather_pool = nullptr;
}

// Finds the staged workspace of the table `P` (a registered host key or the device copy of its parameters). Returns nullptr
// for a table that is not staged; *out_unregistered_staged is set when `P` is staged but was never registered, and *out_error
// when the device parameters could not be read back (or, on a capturing stream, could not be told from a resident table's),
// in which case nothing may be launched for `P`.
static StagedPleWorkspace* resolve_staged_ple_workspace(const void* P, cudaStream_t stream, bool allow_sync,
                                                        bool* out_unregistered_staged, cudaError_t* out_error) {
  *out_unregistered_staged = false;
  *out_error = cudaSuccess;
  if (P == nullptr) {
    if (g_last_staged_host_key != nullptr) {
      auto it = g_host_ple_workspaces.find(g_last_staged_host_key);
      if (it != g_host_ple_workspaces.end()) return it->second.get();
    }
    *out_unregistered_staged = true;
    return nullptr;
  }
  auto host_it = g_host_ple_workspaces.find(P);
  if (host_it != g_host_ple_workspaces.end()) {
    return host_it->second.get();
  }
  auto dev_it = g_dev_to_host_ple.find(P);
  if (dev_it != g_dev_to_host_ple.end()) {
    if (dev_it->second == nullptr) {
      return nullptr;
    }
    if (!allow_sync || g_host_ple_workspaces.size() == 1) {
      auto ws_it = g_host_ple_workspaces.find(dev_it->second);
      if (ws_it != g_host_ple_workspaces.end()) {
        return ws_it->second.get();
      }
      g_dev_to_host_ple.erase(dev_it);
    }
  }
  // Telling a staged table from a resident one takes a read back of the descriptor, which a capturing stream cannot make. A
  // descriptor that has not been told apart is an error there, not a resident table: a staged table taken for one would reach
  // kernels that dereference host-only shard pointers on every replay. An eager call has to use a descriptor first.
  if (staged_stream_is_capturing(stream)) {
    *out_error = cudaErrorStreamCaptureUnsupported;
    return nullptr;
  }
  PleParams dev_params{};
  cudaError_t err = staged_copy_to_host(&dev_params, P, sizeof(PleParams), cudaMemcpyDefault, stream);
  if (err == cudaSuccess) {
    err = staged_stream_sync(stream);
  }
  if (err != cudaSuccess) {
    *out_error = err;
    return nullptr;
  }
  if (dev_params.pad_ == 1) {
    for (auto& kv : g_host_ple_workspaces) {
      if (memcmp(&kv.second->params, &dev_params, sizeof(PleParams)) == 0) {
        g_dev_to_host_ple[P] = kv.first;
        return kv.second.get();
      }
    }
    if (dev_params.n_shards > 0 && dev_params.n_shards <= 256) {
      for (auto& kv : g_host_ple_workspaces) {
        const PleParams& hp = kv.second->params;
        if (hp.n_shards == dev_params.n_shards &&
            memcmp(hp.shard_ptr, dev_params.shard_ptr, (size_t)dev_params.n_shards * sizeof(void*)) == 0) {
          g_dev_to_host_ple[P] = kv.first;
          return kv.second.get();
        }
      }
    }
    *out_unregistered_staged = true;
    return nullptr;
  }
  if (dev_params.n_heads == 0 && dev_params.n_shards == 0 && g_host_ple_workspaces.size() == 1) {
    auto it = g_host_ple_workspaces.begin();
    g_dev_to_host_ple[P] = it->first;
    return it->second.get();
  }
  g_dev_to_host_ple[P] = nullptr;
  return nullptr;
}

__global__ void k_prep_staged(i64 capacity, int n_heads, int head_dim, float scale, i64 tok, i64 pos, i64 s_c1,
                              const unsigned char* fp8_row, i64* ctx_out, i64* input_ids, i64* position_ids,
                              int* committed, i64* past_pos, __nv_bfloat16* ple_out, i64* state) {
  const int tid = threadIdx.x;
  if (tid == 0) {
    input_ids[0] = tok;
    position_ids[0] = pos;
    if (committed != nullptr) committed[0] = (int)pos;
    if (pos >= 0 && pos < capacity)
      past_pos[pos] = pos;
    else
      state[5] |= 1;
  }
  const int total = n_heads * head_dim;
  unsigned char b[3];
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    b[u] = (j < total) ? fp8_row[j] : 0;
  }
  const bool is_int8 = (scale < 0.0f);
  const float abs_scale = is_int8 ? -scale : scale;
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    if (j < total) {
      float f;
      if (is_int8) {
        f = static_cast<float>(static_cast<signed char>(b[u]));
      } else {
        const __half_raw hr = __nv_cvt_fp8_to_halfraw(b[u], __NV_E4M3);
        f = __half2float(*reinterpret_cast<const __half*>(&hr));
      }
      ple_out[j] = __float2bfloat16_rn(f * abs_scale);
    }
  }
  if (tid == 0) {
    ctx_out[0] = s_c1;
    ctx_out[1] = tok;
  }
}

// The tokens of the `rows` rows of one folded step and the token in front of each of them (the c1 of its n-gram context).
// They are passed to the kernel by value, so the kernel reads nothing from host memory but the gathered rows.
struct StagedFoldRows {
  i64 tok[QWEN38_PLANE_MAX_PREP_ROWS];
  i64 c1[QWEN38_PLANE_MAX_PREP_ROWS];
};

// One block per row of a folded step of a staged table: the graph inputs, the context and the PLE row of row `blockIdx.x`.
// Only row 0 writes the committed length, and not at all for a null `committed`.
__global__ void k_prep_fold_staged(i64 capacity, int n_heads, int head_dim, float scale, StagedFoldRows step, i64 base_pos,
                                   const unsigned char* fp8_rows, i64* ctx_out, i64* input_ids, i64* position_ids,
                                   int* committed, i64* past_pos, __nv_bfloat16* ple_out, i64* state) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const i64 tok = step.tok[row];
  const i64 pos = base_pos + row;
  if (tid == 0) {
    input_ids[row] = tok;
    position_ids[row] = pos;
    if (row == 0 && committed != nullptr) committed[0] = (int)pos;
    if (pos >= 0 && pos < capacity)
      past_pos[pos] = pos;
    else
      atomicOr((unsigned long long*)&state[5], 1ULL);
    ctx_out[2 * row + 0] = step.c1[row];
    ctx_out[2 * row + 1] = tok;
  }
  const int total = n_heads * head_dim;
  const unsigned char* fp8_row = fp8_rows + (size_t)row * StagedPleWorkspace::kPrepRingRowBytes;
  __nv_bfloat16* out_row = ple_out + (size_t)row * total;
  unsigned char b[3];
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    b[u] = (j < total) ? fp8_row[j] : 0;
  }
  const bool is_int8 = (scale < 0.0f);
  const float abs_scale = is_int8 ? -scale : scale;
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    if (j < total) {
      float f;
      if (is_int8) {
        f = static_cast<float>(static_cast<signed char>(b[u]));
      } else {
        const __half_raw hr = __nv_cvt_fp8_to_halfraw(b[u], __NV_E4M3);
        f = __half2float(*reinterpret_cast<const __half*>(&hr));
      }
      out_row[j] = __float2bfloat16_rn(f * abs_scale);
    }
  }
}

__global__ void k_ple_rows_staged(int n_heads, int head_dim, float scale, const unsigned char* fp8_rows, int n,
                                  i64 last_c1, i64 last_tok, i64* ctx_out, __nv_bfloat16* ple_out) {
  const int tid = threadIdx.x;
  const int i = blockIdx.x;
  const int total = n_heads * head_dim;
  const unsigned char* src = fp8_rows + (size_t)i * total;
  __nv_bfloat16* out = ple_out + (size_t)i * total;
  unsigned char b[3];
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    b[u] = (j < total) ? src[j] : 0;
  }
  const bool is_int8 = (scale < 0.0f);
  const float abs_scale = is_int8 ? -scale : scale;
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    if (j < total) {
      float f;
      if (is_int8) {
        f = static_cast<float>(static_cast<signed char>(b[u]));
      } else {
        const __half_raw hr = __nv_cvt_fp8_to_halfraw(b[u], __NV_E4M3);
        f = __half2float(*reinterpret_cast<const __half*>(&hr));
      }
      out[j] = __float2bfloat16_rn(f * abs_scale);
    }
  }
  if (i == n - 1 && tid == 0) {
    ctx_out[0] = last_c1;
    ctx_out[1] = last_tok;
  }
}

__global__ void k_prep(const PleParams* P, const i64* tok_ptr, const i64* pos_ptr, int pos_add, const i64* ctx_in,
                       i64* ctx_out, i64* input_ids, i64* position_ids, int* committed, i64* past_pos,
                       __nv_bfloat16* ple_out, i64* state) {
  __shared__ i64 s_row[32];
  __shared__ const unsigned char* s_src[32];
  __shared__ i64 s_tok, s_pos, s_c0, s_c1;
  const int tid = threadIdx.x;
  if (tid == 0) {
    s_tok = tok_ptr[0];
    s_pos = pos_ptr[0] + pos_add;
    s_c0 = ctx_in[0];
    s_c1 = ctx_in[1];
  }
  __syncthreads();
  const i64 tok = s_tok, pos = s_pos, c1 = s_c1;
  const i64 c0 = (c1 == P->eos) ? P->eos : s_c0;  // n-gram positions behind an EOS are masked to EOS
  if (tid < P->n_heads) {
    const int h = tid;
    unsigned long long mixed = (unsigned long long)tok * (unsigned long long)P->mult[0];
    mixed ^= (unsigned long long)c1 * (unsigned long long)P->mult[1];
    if (h >= P->heads_per_ngram) mixed ^= (unsigned long long)c0 * (unsigned long long)P->mult[2];
    const i64 v = P->vocab[h];
    i64 r = (i64)mixed % v;
    if (r < 0) r += v;
    const i64 id = r + P->offset[h];
    int lo = 0, hi = P->n_shards - 1;  // first shard whose end > id
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (P->shard_end[mid] > id)
        hi = mid;
      else
        lo = mid + 1;
    }
    const i64 first = lo == 0 ? 0 : P->shard_end[lo - 1];
    s_row[h] = id - first;
    s_src[h] = P->shard_ptr[lo];
  }
  if (tid == 0) {
    input_ids[0] = tok;
    position_ids[0] = pos;
    if (committed != nullptr) committed[0] = (int)pos;
    if (pos >= 0 && pos < P->capacity)
      past_pos[pos] = pos;
    else
      state[5] |= 1;
  }
  __syncthreads();
  const int hd = P->head_dim;
  const int total = P->n_heads * hd;
  // all (<= 3) host-memory byte loads of this thread are issued before any use: one PCIe round trip
  unsigned char b[3];
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    b[u] = 0;
    if (j < total) {
      const int h = j / hd, c = j - h * hd;
      b[u] = s_src[h][s_row[h] * hd + c];
    }
  }
  const float scale = P->scale;
  const bool is_int8 = (scale < 0.0f);
  const float abs_scale = is_int8 ? -scale : scale;
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    if (j < total) {
      float f;
      if (is_int8) {
        f = static_cast<float>(static_cast<signed char>(b[u]));
      } else {
        const __half_raw hr = __nv_cvt_fp8_to_halfraw(b[u], __NV_E4M3);
        f = __half2float(*reinterpret_cast<const __half*>(&hr));
      }
      ple_out[j] = __float2bfloat16_rn(f * abs_scale);
    }
  }
  if (tid == 0) {
    ctx_out[0] = c1;  // history[-2:] = (c1, tok); uses the raw (unmasked) previous token
    ctx_out[1] = tok;
  }
}

// One block per (row, slot). `honor_mute` is set by prep_batched only: a slot whose control word state[3] is nonzero prepares
// idle inputs there, while prep (one slot, any control words the caller hands over) never reads that word.
__global__ void __launch_bounds__(1024) k_prep_fold(const PleParams* P, const i64* ctx_in, i64* ctx_a, i64* input_ids,
                                                    i64* position_ids, int* committed, i64* past_pos,
                                                    __nv_bfloat16* ple_out, i64* state, int honor_mute) {
  __shared__ i64 s_row[32];
  __shared__ const unsigned char* s_src[32];
  __shared__ i64 s_tok, s_pos, s_c0, s_c1, s_muted;
  const int row = blockIdx.x;
  const int rows = gridDim.x;
  const int b = blockIdx.y;
  const int tid = threadIdx.x;
  i64* state_b = state + (size_t)b * 16;
  const i64* ctx_in_b = ctx_in + (size_t)b * 2;
  i64* ctx_a_b = ctx_a + (size_t)b * 2 * rows;
  i64* input_ids_b = input_ids + (size_t)b * rows;
  i64* position_ids_b = position_ids + (size_t)b * rows;
  int* committed_b = committed != nullptr ? committed + b : nullptr;  // null: the caller keeps no committed length
  i64* past_pos_b = past_pos + (size_t)b * P->capacity;
  const int hd = P->head_dim;
  const int total = P->n_heads * hd;
  __nv_bfloat16* row_ple = ple_out + ((size_t)b * rows + row) * total;

  if (tid == 0) {
    s_muted = honor_mute != 0 ? state_b[3] : 0;
    if (s_muted != 0) {
      s_tok = 0;
      s_pos = row;
      s_c1 = P->eos;
      s_c0 = P->eos;
    } else {
      constexpr int kSlots[8] = {0, 2, 8, 9, 10, 11, 12, 13};
      s_tok = state_b[kSlots[row]];
      s_pos = state_b[1] + row;
      s_c1 = (row == 0) ? ctx_in_b[1] : state_b[kSlots[row - 1]];
      s_c0 = (row == 0) ? ctx_in_b[0] : ((row == 1) ? ctx_in_b[1] : state_b[kSlots[row - 2]]);
    }
  }
  __syncthreads();
  if (s_muted != 0) {
    if (tid == 0) {
      input_ids_b[row] = 0;
      position_ids_b[row] = row;
      if (row == 0 && committed_b != nullptr) committed_b[0] = 0;
      if (row < P->capacity) past_pos_b[row] = row;
      ctx_a_b[2 * row + 0] = P->eos;
      ctx_a_b[2 * row + 1] = 0;
    }
    for (int j = tid; j < total; j += blockDim.x) {
      row_ple[j] = __float2bfloat16_rn(0.0f);
    }
    return;
  }
  const i64 tok = s_tok, pos = s_pos, c1 = s_c1;
  const i64 c0 = (c1 == P->eos) ? P->eos : s_c0;
  if (tid < P->n_heads) {
    const int h = tid;
    unsigned long long mixed = (unsigned long long)tok * (unsigned long long)P->mult[0];
    mixed ^= (unsigned long long)c1 * (unsigned long long)P->mult[1];
    if (h >= P->heads_per_ngram) mixed ^= (unsigned long long)c0 * (unsigned long long)P->mult[2];
    const i64 v = P->vocab[h];
    i64 r = (i64)mixed % v;
    if (r < 0) r += v;
    const i64 id = r + P->offset[h];
    int lo = 0, hi = P->n_shards - 1;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (P->shard_end[mid] > id)
        hi = mid;
      else
        lo = mid + 1;
    }
    const i64 first = lo == 0 ? 0 : P->shard_end[lo - 1];
    s_row[h] = id - first;
    s_src[h] = P->shard_ptr[lo];
  }
  if (tid == 0) {
    input_ids_b[row] = tok;
    position_ids_b[row] = pos;
    if (row == 0 && committed_b != nullptr) committed_b[0] = (int)pos;
    if (pos >= 0 && pos < P->capacity)
      past_pos_b[pos] = pos;
    else
      atomicOr((unsigned long long*)&state_b[5], 1ULL);
  }
  __syncthreads();
  unsigned char b_vals[3];
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    b_vals[u] = 0;
    if (j < total) {
      const int h = j / hd, c = j - h * hd;
      b_vals[u] = s_src[h][s_row[h] * hd + c];
    }
  }
  const float scale = P->scale;
  const bool is_int8 = (scale < 0.0f);
  const float abs_scale = is_int8 ? -scale : scale;
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    if (j < total) {
      float f;
      if (is_int8) {
        f = static_cast<float>(static_cast<signed char>(b_vals[u]));
      } else {
        const __half_raw hr = __nv_cvt_fp8_to_halfraw(b_vals[u], __NV_E4M3);
        f = __half2float(*reinterpret_cast<const __half*>(&hr));
      }
      row_ple[j] = __float2bfloat16_rn(f * abs_scale);
    }
  }
  if (tid == 0) {
    ctx_a_b[2 * row + 0] = c1;
    ctx_a_b[2 * row + 1] = tok;
  }
}

__global__ void k_argmax_part(const __nv_bfloat16* logits, int n, float* outv, int* outi) {
  __shared__ float sv[256];
  __shared__ int si[256];
  const int per = (n + gridDim.x - 1) / gridDim.x;
  const int lo = blockIdx.x * per, hi = min(n, lo + per);
  float best = -INFINITY;
  int bi = 0x7fffffff;
  for (int i = lo + threadIdx.x; i < hi; i += blockDim.x) {
    const float v = __bfloat162float(logits[i]);
    if (v > best || (v == best && i < bi)) {
      best = v;
      bi = i;
    }
  }
  sv[threadIdx.x] = best;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float v = sv[threadIdx.x + s];
      const int i = si[threadIdx.x + s];
      if (v > sv[threadIdx.x] || (v == sv[threadIdx.x] && i < si[threadIdx.x])) {
        sv[threadIdx.x] = v;
        si[threadIdx.x] = i;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    outv[blockIdx.x] = sv[0];
    outi[blockIdx.x] = si[0];
  }
}

__device__ int merge_argmax(const float* v, const int* idx, int nb) {
  float best = -INFINITY;
  int bi = 0x7fffffff;
  for (int b = 0; b < nb; ++b)
    if (v[b] > best || (v[b] == best && idx[b] < bi)) {
      best = v[b];
      bi = idx[b];
    }
  return bi;
}

/* state (int64[8]): [0]=pending token, [1]=committed length L, [2]=draft token, [4]=sequence counter,
 * [5]=error bits (1 = position capacity, 2 = nonzero validation status), [6]=last accept flag
 * result ring (pinned, 8 int64 per record, `ring` records, slot = seq % ring):
 *   [0]=seq (written last), [1]=pred0, [2]=pred1, [3]=accept, [4]=status OR, [5]=new pending, [6]=new L
 * accept==1 -> commit row 1 (2 tokens), else commit row 0 (1 token). */
/* verify epilogue for graphs that output argmax tokens (qwen.fused_gemm ArgmaxRows -> int64[2]) */
__global__ void k_post_tok(const i64* tok01, i64* state, i64* ctx_commit, const i64* ctx_a, const i64* ctx_b,
                           const int* st0, int nst, i64* result, int ring) {
  if (threadIdx.x != 0) return;
  const i64 pred0 = tok01[0];
  const i64 pred1 = tok01[1];
  const bool accept = pred0 == state[2];
  i64 status = 0;
  for (int k = 0; k < nst; ++k) status |= (i64)st0[k];
  const i64 new_pending = accept ? pred1 : pred0;
  const i64 new_len = state[1] + (accept ? 2 : 1);
  const i64* src = accept ? ctx_b : ctx_a;
  ctx_commit[0] = src[0];
  ctx_commit[1] = src[1];
  volatile i64* r = result + 8 * (state[4] % ring);
  state[0] = new_pending;
  state[1] = new_len;
  state[4] += 1;
  state[5] |= status ? 2 : 0;
  state[6] = accept ? 1 : 0;
  r[1] = pred0;
  r[2] = pred1;
  r[3] = accept ? 1 : 0;
  r[4] = status;
  r[5] = new_pending;
  r[6] = new_len;
  __threadfence_system();
  r[0] = state[4];
}

/* plain S=1 step epilogue: argmax merge, commit one token, publish */
__global__ void k_post1(const float* v0, const int* i0, int nb, i64* state, i64* ctx_commit, const i64* ctx_a,
                        const int* st0, int nst, i64* result, int ring) {
  if (threadIdx.x != 0) return;
  const i64 pred0 = merge_argmax(v0, i0, nb);
  i64 status = 0;
  for (int k = 0; k < nst; ++k) status |= (i64)st0[k];
  ctx_commit[0] = ctx_a[0];
  ctx_commit[1] = ctx_a[1];
  volatile i64* r = result + 8 * (state[4] % ring);
  state[0] = pred0;
  state[1] += 1;
  state[4] += 1;
  state[5] |= status ? 2 : 0;
  r[1] = pred0;
  r[4] = status;
  r[5] = pred0;
  r[6] = state[1];
  __threadfence_system();
  r[0] = state[4];
}

/* WP-D head launch preparation (device-pointer driver).  ctl = verifier state: [0]=pending token (already the NEXT pending after post),
 * [1]=committed length L' (position of pending), [2]=verified draft.  rows==2 (accepted cycle): head rows [(draft, h0) @ L'-1,
 * (pending, h1) @ L'], rows==1: [(pending, h0) @ L'].  Copies rows x hidden_width bf16 hidden from src (verify hidden output rows). */
__global__ void k_head_prep(const i64* ctl, int rows, int hlen, int hidden_width, const uint4* src, uint4* dst, i64* tokens,
                            i64* positions, int* committed) {
  if (threadIdx.x == 0) {
    const i64 pend = ctl[0], L = ctl[1], draft = ctl[2];
    if (rows == 2) {
      tokens[0] = draft;
      tokens[1] = pend;
      positions[0] = L - 1;
      positions[1] = L;
      committed[0] = hlen;
      committed[1] = hlen + 1;
    } else {
      tokens[0] = pend;
      positions[0] = L;
      committed[0] = hlen;
    }
  }
  const int n16 = rows * hidden_width * 2 / 16;
  for (int i = threadIdx.x; i < n16; i += blockDim.x) dst[i] = src[i];
}

__global__ void k_head_post(const i64* next_token, i64* ctl) {
  if (threadIdx.x == 0) ctl[2] = next_token[0];
}

__global__ void k_or_status(const int* st, int nst, i64* state) {
  if (threadIdx.x != 0) return;
  i64 status = 0;
  for (int k = 0; k < nst; ++k) status |= (i64)st[k];
  if (status) state[5] |= 2;
}

/* ---- fast prefill support (v12): batched PLE rows for the dense prompt chunks, dense -> packed state hand-off ---- */

/* PLE embedding rows of n consecutive tokens (one block per row; same arithmetic as k_prep). history before the first
 * token = ctx_in = (older, last); ctx_out = history[-2:] after the last token. */
__global__ void k_ple_rows(const PleParams* P, const i64* tokens, int n, const i64* ctx_in, i64* ctx_out,
                           __nv_bfloat16* ple_out) {
  __shared__ i64 s_row[32];
  __shared__ const unsigned char* s_src[32];
  const int tid = threadIdx.x;
  const int i = blockIdx.x;
  const i64 tok = tokens[i];
  const i64 c1 = i >= 1 ? tokens[i - 1] : ctx_in[1];
  const i64 c0raw = i >= 2 ? tokens[i - 2] : (i == 1 ? ctx_in[1] : ctx_in[0]);
  const i64 c0 = (c1 == P->eos) ? P->eos : c0raw;
  if (tid < P->n_heads) {
    const int h = tid;
    unsigned long long mixed = (unsigned long long)tok * (unsigned long long)P->mult[0];
    mixed ^= (unsigned long long)c1 * (unsigned long long)P->mult[1];
    if (h >= P->heads_per_ngram) mixed ^= (unsigned long long)c0 * (unsigned long long)P->mult[2];
    const i64 v = P->vocab[h];
    i64 r = (i64)mixed % v;
    if (r < 0) r += v;
    const i64 id = r + P->offset[h];
    int lo = 0, hi = P->n_shards - 1;
    while (lo < hi) {
      const int mid = (lo + hi) >> 1;
      if (P->shard_end[mid] > id)
        hi = mid;
      else
        lo = mid + 1;
    }
    const i64 first = lo == 0 ? 0 : P->shard_end[lo - 1];
    s_row[h] = id - first;
    s_src[h] = P->shard_ptr[lo];
  }
  __syncthreads();
  const int hd = P->head_dim;
  const int total = P->n_heads * hd;
  __nv_bfloat16* out = ple_out + (size_t)i * total;
  unsigned char b[3];
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    b[u] = 0;
    if (j < total) {
      const int h = j / hd, c = j - h * hd;
      b[u] = s_src[h][s_row[h] * hd + c];
    }
  }
  const float scale = P->scale;
  const bool is_int8 = (scale < 0.0f);
  const float abs_scale = is_int8 ? -scale : scale;
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    if (j < total) {
      float f;
      if (is_int8) {
        f = static_cast<float>(static_cast<signed char>(b[u]));
      } else {
        const __half_raw hr = __nv_cvt_fp8_to_halfraw(b[u], __NV_E4M3);
        f = __half2float(*reinterpret_cast<const __half*>(&hr));
      }
      out[j] = __float2bfloat16_rn(f * abs_scale);
    }
  }
  if (i == n - 1 && tid == 0) {
    ctx_out[0] = c1;
    ctx_out[1] = tok;
  }
}

/* Mean-pooled, RMS-normalised, leading-half-RoPE compressed index keys of complete blocks of 4 raw keys
 * (arithmetic order of handoff_lib.pooled_block_keys; no FMA contraction, the python reference rounds every elementwise op).
 * raw bf16 [nb*4, 128] -> out fp32 [nb, 128]; block b has position 4 * (b + first_block). */
__global__ void k_pool_keys(const __nv_bfloat16* raw, const __nv_bfloat16* knorm, const __nv_bfloat16* cosv,
                            const __nv_bfloat16* sinv, int width, float eps, int first_block, float* out) {
  __shared__ float s_red[4];
  __shared__ float s_norm[128];
  const int b = blockIdx.x, c = threadIdx.x;
  const __nv_bfloat16* r = raw + (size_t)b * 4 * 128 + c;
  const float k0 = __bfloat162float(r[0]), k1 = __bfloat162float(r[128]), k2 = __bfloat162float(r[256]),
              k3 = __bfloat162float(r[384]);
  float pooled = __fadd_rn(__fadd_rn(__fadd_rn(k0, k1), k2), k3);
  pooled = __fmul_rn(pooled, 0.25f);
  float sq = __fmul_rn(pooled, pooled);
  for (int o = 16; o > 0; o >>= 1) sq = __fadd_rn(sq, __shfl_xor_sync(0xffffffffu, sq, o));
  if ((c & 31) == 0) s_red[c >> 5] = sq;
  __syncthreads();
  const float total = __fadd_rn(__fadd_rn(s_red[0], s_red[1]), __fadd_rn(s_red[2], s_red[3]));
  const float inv_rms = rsqrtf(__fadd_rn(__fmul_rn(total, 0.0078125f), eps));
  const float normed = __fmul_rn(__fmul_rn(pooled, inv_rms), __bfloat162float(knorm[c]));
  s_norm[c] = normed;
  __syncthreads();
  const int half = width / 2;
  const size_t position = (size_t)(b + first_block) * 4;
  float result = normed;
  if (c < half) {
    const float cs = __bfloat162float(cosv[position * width + c]), sn = __bfloat162float(sinv[position * width + c]);
    result = __fadd_rn(__fmul_rn(normed, cs), __fmul_rn(-s_norm[c + half], sn));
  } else if (c < width) {
    const float cs = __bfloat162float(cosv[position * width + c]), sn = __bfloat162float(sinv[position * width + c]);
    result = __fadd_rn(__fmul_rn(normed, cs), __fmul_rn(s_norm[c - half], sn));
  }
  out[(size_t)b * 128 + c] = result;
}

/* dst[b][k][v] = src[b][v][k] for batches of n x n fp32 matrices (GDN recurrent state: dense [V, K] -> packed [K, V]) */
__global__ void k_transpose_f32(const float* src, float* dst, int n) {
  __shared__ float tile[32][33];
  const size_t base = (size_t)blockIdx.z * n * n;
  const int x = blockIdx.x * 32 + threadIdx.x;
  for (int j = 0; j < 32; j += 8) {
    const int y = blockIdx.y * 32 + threadIdx.y + j;
    if (x < n && y < n) {
      tile[threadIdx.y + j][threadIdx.x] = src[base + (size_t)y * n + x];
    }
  }
  __syncthreads();
  const int tx = blockIdx.y * 32 + threadIdx.x;
  for (int j = 0; j < 32; j += 8) {
    const int ty = blockIdx.x * 32 + threadIdx.y + j;
    if (tx < n && ty < n) {
      dst[base + (size_t)ty * n + tx] = tile[threadIdx.x][threadIdx.y + j];
    }
  }
}

/* Quantize BF16 KV rows into 1-byte per-channel INT8 (mode=1) or FP8 E4M3FN (mode=2).
 * Each row has 256 elements (1 KV head of dim 256); `scale` points to the 256 float scales of that head.
 * `src` has stride `src_pitch_elems` BF16 elements between rows; `dst` has stride `dst_pitch_elems` bytes between rows. */
__global__ void k_quantize_kv(const __nv_bfloat16* src, uint8_t* dst, const float* scale, int rows, int src_pitch_elems,
                              int dst_pitch_elems, int mode) {
  const int r = blockIdx.x;
  const int c = threadIdx.x;
  if (r >= rows || c >= 256) return;
  const float x = __bfloat162float(src[(size_t)r * src_pitch_elems + c]);
  const float s = scale[c];
  const float q = s > 0.0f ? (x / s) : 0.0f;
  uint8_t out_byte = 0;
  if (mode == 1) {
    // Clamped in floating point before the conversion, so that it never sees a value outside the INT8 range (an overflow to
    // infinity included) whatever it does with one; a NaN is 0.
    const float qc = q != q ? 0.0f : fminf(fmaxf(q, -127.0f), 127.0f);
    out_byte = static_cast<uint8_t>(static_cast<int8_t>(__float2int_rn(qc)));
  } else {
    float qc = fminf(fmaxf(q, -448.0f), 448.0f);
    out_byte = static_cast<uint8_t>(__nv_cvt_float_to_fp8(qc, __NV_SATFINITE, __NV_E4M3));
  }
  dst[(size_t)r * dst_pitch_elems + c] = out_byte;
}

// The embedding gathers read row `tok` of a table of table_rows rows (the model's vocabulary size, passed by the loader); an
// id outside [0, table_rows) reads row 0.
__global__ void k_embed_rows(const __nv_bfloat16* table, i64 table_rows, const i64* ids, __nv_bfloat16* out, int rows,
                             int width) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const i64 raw_tok = ids[r];
  const i64 tok = (raw_tok >= 0 && raw_tok < table_rows) ? raw_tok : 0;
  const __nv_bfloat16* src = table + (size_t)tok * width;
  __nv_bfloat16* dst = out + (size_t)r * width;
  // A row is 2 * width bytes, so the rows are 16-byte aligned for the vector copy only when width is a multiple of 8 (and the
  // buffers are aligned). Otherwise, and for the elements past the last whole group of 8, the row is copied element by element.
  const bool aligned = ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15u) == 0;
  const int n8 = aligned ? width / 8 : 0;
  const uint4* src4 = reinterpret_cast<const uint4*>(src);
  uint4* dst4 = reinterpret_cast<uint4*>(dst);
  for (int i = threadIdx.x; i < n8; i += blockDim.x) dst4[i] = src4[i];
  for (int i = n8 * 8 + threadIdx.x; i < width; i += blockDim.x) dst[i] = src[i];
}

__global__ void k_embed_rows_int8(const int8_t* table, const float* row_scales, i64 table_rows, const i64* ids,
                                  __nv_bfloat16* out, int rows, int width) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const i64 raw_tok = ids[r];
  const i64 tok = (raw_tok >= 0 && raw_tok < table_rows) ? raw_tok : 0;
  const int8_t* src = table + (size_t)tok * width;
  const float sc = row_scales[tok];
  __nv_bfloat16* dst = out + (size_t)r * width;
  if ((width & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(src) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(dst) & 15) == 0) {
    const int n16 = width >> 4;
    const uint4* src16 = reinterpret_cast<const uint4*>(src);
    uint4* dst8 = reinterpret_cast<uint4*>(dst);
    for (int i = threadIdx.x; i < n16; i += blockDim.x) {
      const uint4 q = src16[i];
      const int8_t* q8 = reinterpret_cast<const int8_t*>(&q);
      uint4 lo_u4, hi_u4;
      __nv_bfloat16* lo_bf = reinterpret_cast<__nv_bfloat16*>(&lo_u4);
      __nv_bfloat16* hi_bf = reinterpret_cast<__nv_bfloat16*>(&hi_u4);
#pragma unroll
      for (int m = 0; m < 8; ++m) {
        lo_bf[m] = __float2bfloat16_rn(static_cast<float>(q8[m]) * sc);
        hi_bf[m] = __float2bfloat16_rn(static_cast<float>(q8[m + 8]) * sc);
      }
      dst8[2 * i] = lo_u4;
      dst8[2 * i + 1] = hi_u4;
    }
    return;
  }
  for (int i = threadIdx.x; i < width; i += blockDim.x) {
    dst[i] = __float2bfloat16_rn(static_cast<float>(src[i]) * sc);
  }
}

#ifdef _WIN32
#define QWEN38_PLANE_EXPORT __declspec(dllexport)
#else
#define QWEN38_PLANE_EXPORT
#endif

// Drops every staged table of the module, whichever loader registered it, and stops the gather workers. Caller holds
// g_staged_ple_mu.
static void release_all_staged_state_locked() {
  stop_ple_gather_pool_locked();
  bool need_sync = false;
  for (const auto& kv : g_host_ple_workspaces) {
    if (kv.second && kv.second->has_cuda_allocations()) {
      need_sync = true;
      break;
    }
  }
  if (need_sync) {
    if (cudaDeviceSynchronize() != cudaSuccess) {
      cudaGetLastError();
    }
  }
  for (auto& kv : g_host_ple_workspaces) {
    if (kv.second) kv.second->release(false);
  }
  g_host_ple_workspaces.clear();
  g_dev_to_host_ple.clear();
  g_last_staged_host_key = nullptr;
  hook_cleanup();
}

// A kernel launch that is rejected (bad launch configuration, no kernel image for the device, a failed lazy module load) is
// recorded only in the calling thread's last-error slot: the stream stays healthy, so cudaStreamSynchronize still succeeds
// and the outputs keep whatever they held. The launch exports that return void therefore park the failure keyed by its
// originating (device, stream) and qwen38_plane_stream_sync reports and clears only that stream's entry; prep and ple_rows
// return the launch status directly.
struct ParkedStreamKey {
  int device = 0;
  cudaStream_t stream = nullptr;
  bool operator==(const ParkedStreamKey& other) const {
    return device == other.device && stream == other.stream;
  }
};

struct ParkedStreamKeyHash {
  size_t operator()(const ParkedStreamKey& k) const {
    return std::hash<int>{}(k.device) ^ (std::hash<void*>{}(reinterpret_cast<void*>(k.stream)) << 1);
  }
};

static std::mutex g_parked_launch_mu;
static std::unordered_map<ParkedStreamKey, int, ParkedStreamKeyHash> g_parked_launch_errors;

static inline ParkedStreamKey CurrentParkedStreamKey(cudaStream_t stream) {
  int dev = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) {
    cudaGetLastError();
    dev = 0;
  }
  return ParkedStreamKey{dev, stream};
}

// BeginLaunch drops an error that the thread may still hold from an earlier call that nobody read, so that LaunchStatus
// reports only the launches since then. An error that sticks (a faulted context) is not dropped and still fails the launch.
static inline void BeginLaunch() { (void)cudaGetLastError(); }
static inline int LaunchStatus() { return static_cast<int>(cudaGetLastError()); }
static inline void ParkLaunchStatus(cudaStream_t stream) {
  if (const int err = LaunchStatus()) {
    const ParkedStreamKey key = CurrentParkedStreamKey(stream);
    std::lock_guard<std::mutex> lock(g_parked_launch_mu);
    auto it = g_parked_launch_errors.find(key);
    if (it == g_parked_launch_errors.end() || it->second == 0) {
      g_parked_launch_errors[key] = err;
    }
  }
}

static inline int TakeParkedLaunchStatus(cudaStream_t stream) {
  const ParkedStreamKey key = CurrentParkedStreamKey(stream);
  std::lock_guard<std::mutex> lock(g_parked_launch_mu);
  auto it = g_parked_launch_errors.find(key);
  if (it == g_parked_launch_errors.end()) return 0;
  const int err = it->second;
  g_parked_launch_errors.erase(it);
  return err;
}

static inline void ClearParkedLaunchStatusForStream(cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(g_parked_launch_mu);
  for (auto it = g_parked_launch_errors.begin(); it != g_parked_launch_errors.end();) {
    if (it->first.stream == stream) {
      it = g_parked_launch_errors.erase(it);
    } else {
      ++it;
    }
  }
}

// Geometry of the staged (host-gathered) PLE table that `P` selects: its row width (n_heads * head_dim) and its position
// capacity, both 0 when the table is read by the GPU. A staged table that was never registered, or whose parameters cannot
// be read back, is an error and nothing may be launched for it.
static cudaError_t resolve_staged_ple_geometry(const void* P, cudaStream_t stream, size_t* row_width, size_t* capacity) {
  *row_width = 0;
  *capacity = 0;
  std::lock_guard<std::mutex> lock(g_staged_ple_mu);
  bool unregistered_staged = false;
  cudaError_t err = cudaSuccess;
  StagedPleWorkspace* ws = resolve_staged_ple_workspace(P, stream, /*allow_sync=*/true, &unregistered_staged, &err);
  if (err != cudaSuccess) return err;
  if (unregistered_staged) return cudaErrorInvalidValue;
  if (ws != nullptr) {
    *row_width = static_cast<size_t>(ws->params.n_heads) * static_cast<size_t>(ws->params.head_dim);
    *capacity = static_cast<size_t>(ws->params.capacity);
  }
  return cudaSuccess;
}

// Prepares the `rows` (2 .. QWEN38_PLANE_MAX_PREP_ROWS) rows of one folded step of one request slot of a staged table with a
// single launch. The pointers are those of prep for the slot; `committed` is null or the slot's own int. The control words and
// the context in are read back once, as row 0 of the one-row form does; the PLE rows are gathered on the host and one block per
// row prepares the rest, only row 0 getting the committed length. A step that fails returns its cudaError_t before it gathers
// or launches anything.
static int prep_staged_fold_rows(cudaStream_t stream, const void* P, int rows, const void* ctx_in, void* ctx_out,
                                 void* input_ids, void* position_ids, void* committed, void* past_pos, void* ple_out,
                                 void* state) {
  if (rows < 2 || rows > QWEN38_PLANE_MAX_PREP_ROWS) return (int)cudaErrorInvalidValue;
  constexpr int kSlots[QWEN38_PLANE_MAX_PREP_ROWS] = {0, 2, 8, 9, 10, 11, 12, 13};
  constexpr size_t kRowBytes = StagedPleWorkspace::kPrepRingRowBytes;
  std::lock_guard<std::mutex> lock(g_staged_ple_mu);
  bool unregistered_staged = false;
  cudaError_t err = cudaSuccess;
  StagedPleWorkspace* ws = resolve_staged_ple_workspace(P, stream, /*allow_sync=*/true, &unregistered_staged, &err);
  if (err != cudaSuccess) return (int)err;
  if (unregistered_staged || ws == nullptr) return (int)cudaErrorInvalidValue;
  err = ensure_staged_buffers(*ws, 0, 0);
  if (err != cudaSuccess) return (int)err;

  // The words that the one-row form keeps a snapshot of are overwritten here, so that snapshot, a half-written one included,
  // must not be reused by a row that follows.
  ws->cached_state_ptr = nullptr;
  ws->cached_ctx_out_ptr = nullptr;
  err = staged_copy_to_host(&ws->scalars[0], state, 14 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
  if (err == cudaSuccess) {
    err = staged_copy_to_host(&ws->scalars[14], ctx_in, 2 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
  }
  if (err == cudaSuccess) err = staged_stream_sync(stream);
  if (err != cudaSuccess) return (int)err;

  // The rows of one step are contiguous in the ring. Writing over rows of earlier steps is safe: the wait above has retired
  // every kernel that read them.
  size_t first = ws->ring_idx % StagedPleWorkspace::kPrepRingRows;
  if (first + static_cast<size_t>(rows) > StagedPleWorkspace::kPrepRingRows) first = 0;
  ws->ring_idx = static_cast<unsigned int>(first + static_cast<size_t>(rows));
  unsigned char* host_rows = ws->prep_ring + first * kRowBytes;
  const unsigned char* device_rows = ws->prep_ring_dev + first * kRowBytes;

  StagedFoldRows step{};
  i64 c0 = ws->scalars[14];
  i64 c1 = ws->scalars[15];
  for (int r = 0; r < rows; ++r) {
    const i64 tok = ws->scalars[kSlots[r]];
    step.tok[r] = tok;
    step.c1[r] = c1;
    host_gather_ple_row(*ws, tok, c0, c1, host_rows + static_cast<size_t>(r) * kRowBytes);
    c0 = c1;
    c1 = tok;
  }
  hook_launch();
  k_prep_fold_staged<<<rows, 1024, 0, stream>>>(ws->params.capacity, ws->params.n_heads, ws->params.head_dim,
                                                ws->params.scale, step, ws->scalars[1], device_rows, (i64*)ctx_out,
                                                (i64*)input_ids, (i64*)position_ids, (int*)committed, (i64*)past_pos,
                                                (__nv_bfloat16*)ple_out, (i64*)state);
  return 0;
}

extern "C" {
QWEN38_PLANE_EXPORT int qwen38_plane_abi_version(void) { return QWEN38_PLANE_ABI_VERSION; }
QWEN38_PLANE_EXPORT int qwen38_plane_sizeof_ple_params(void) { return (int)sizeof(PleParams); }
// Optional export: what this build implements beyond the mandatory symbols (see the capabilities in qwen38_plane_abi.h).
QWEN38_PLANE_EXPORT unsigned long long qwen38_plane_capabilities(void) { return QWEN38_PLANE_ALL_CAPABILITIES; }

// The staged tables live in this module and the module is shared by every loader of the library file, so the state is
// released by the last loader to detach, not by whichever loader is destroyed first.
QWEN38_PLANE_EXPORT void qwen38_plane_attach(void) {
  std::lock_guard<std::mutex> lock(g_staged_ple_mu);
  ++g_attached_loaders;
}

QWEN38_PLANE_EXPORT void qwen38_plane_detach(void) {
  std::lock_guard<std::mutex> lock(g_staged_ple_mu);
  if (g_attached_loaders > 0 && --g_attached_loaders == 0) {
    release_all_staged_state_locked();
  }
}

// Explicit teardown of all staged tables of the module (every loader's); loaders do not call it from their destructors.
QWEN38_PLANE_EXPORT void qwen38_plane_release_ple_workspace(void) {
  std::lock_guard<std::mutex> lock(g_staged_ple_mu);
  release_all_staged_state_locked();
}

QWEN38_PLANE_EXPORT void qwen38_plane_unset_ple_host_params(const void* params) {
  if (params == nullptr) return;
  std::lock_guard<std::mutex> lock(g_staged_ple_mu);
  for (auto it = g_dev_to_host_ple.begin(); it != g_dev_to_host_ple.end();) {
    if (it->first == params || it->second == params) {
      it = g_dev_to_host_ple.erase(it);
    } else {
      ++it;
    }
  }
  auto it = g_host_ple_workspaces.find(params);
  if (it != g_host_ple_workspaces.end()) {
    if (it->second) it->second->release(true);
    g_host_ple_workspaces.erase(it);
  }
  if (g_last_staged_host_key == params) {
    g_last_staged_host_key = g_host_ple_workspaces.empty() ? nullptr : g_host_ple_workspaces.begin()->first;
  }
}

QWEN38_PLANE_EXPORT void qwen38_plane_set_ple_host_params(const void* params) {
  if (params == nullptr) {
    qwen38_plane_release_ple_workspace();
    return;
  }
  const PleParams* hp = static_cast<const PleParams*>(params);
  std::lock_guard<std::mutex> lock(g_staged_ple_mu);
  g_dev_to_host_ple.clear();
  if (hp->pad_ == 1) {
    auto& ws = g_host_ple_workspaces[params];
    if (!ws) {
      ws = std::make_unique<StagedPleWorkspace>();
    }
    ws->params = *hp;
    ws->reset_cache();
    g_last_staged_host_key = params;
  } else {
    auto it = g_host_ple_workspaces.find(params);
    if (it != g_host_ple_workspaces.end()) {
      if (it->second) it->second->release(true);
      g_host_ple_workspaces.erase(it);
    }
    if (g_last_staged_host_key == params) {
      g_last_staged_host_key = g_host_ple_workspaces.empty() ? nullptr : g_host_ple_workspaces.begin()->first;
    }
  }
}

QWEN38_PLANE_EXPORT int qwen38_plane_prep(void* s, const void* P, const void* tok_ptr, const void* pos_ptr, int pos_add, const void* ctx_in,
                                          void* ctx_out, void* input_ids, void* position_ids, void* committed, void* past_pos,
                                          void* ple_out, void* state) {
  // Only [-8, -2] selects the folded multi-row form; a value below it can only come from a caller that packed something else
  // into pos_add (the batched form has its own entry, qwen38_plane_prep_batched), so it fails instead of acting as a position
  // offset.
  if (pos_add < -8) return static_cast<int>(cudaErrorInvalidValue);
  if (pos_add <= -2) {
    // Prepares the `rows` = -pos_add rows of one fold/verify step in a single call.
    const int rows = -pos_add;
    cudaStream_t stream = (cudaStream_t)s;
    size_t staged_ple_total = 0;
    size_t staged_capacity = 0;
    const cudaError_t resolved = resolve_staged_ple_geometry(P, stream, &staged_ple_total, &staged_capacity);
    if (resolved != cudaSuccess) return (int)resolved;
    if (staged_ple_total != 0) {
      // The host-table gather is per row, so the staged path gathers every row on the host and prepares them all with one
      // launch (the mutex is not held here). Row 0 owns the committed length: no other row is handed a committed pointer.
      return prep_staged_fold_rows(stream, P, rows, ctx_in, ctx_out, input_ids, position_ids, committed, past_pos, ple_out,
                                   state);
    }
    BeginLaunch();
    k_prep_fold<<<rows, 1024, 0, stream>>>((const PleParams*)P, (const i64*)ctx_in, (i64*)ctx_out, (i64*)input_ids,
                                           (i64*)position_ids, (int*)committed, (i64*)past_pos,
                                           (__nv_bfloat16*)ple_out, (i64*)state, /*honor_mute=*/0);
    return LaunchStatus();
  }
  {
    std::lock_guard<std::mutex> lock(g_staged_ple_mu);
    cudaStream_t stream = (cudaStream_t)s;
    const bool allow_sync = !(pos_add >= 1 && pos_add <= 7 && state != nullptr);
    bool unregistered_staged = false;
    cudaError_t err = cudaSuccess;
    StagedPleWorkspace* ws = resolve_staged_ple_workspace(P, stream, allow_sync, &unregistered_staged, &err);
    if (err != cudaSuccess) {
      return (int)err;
    }
    if (unregistered_staged) {
      return (int)cudaErrorInvalidValue;
    }
    if (ws != nullptr) {
      if (staged_stream_is_capturing(stream)) {
        return (int)cudaErrorStreamCaptureUnsupported;
      }
      err = ensure_staged_buffers(*ws, 0, 0);
      if (err == cudaSuccess) {
        err = staged_ensure_done_event(&ws->prep_done_event);
      }
      if (err != cudaSuccess) {
        return (int)err;
      }
      if (ws->prep_in_flight && ws->last_prep_stream != stream) {
        err = cudaEventSynchronize(ws->prep_done_event);
        if (err != cudaSuccess) {
          cudaGetLastError();
          return (int)err;
        }
        ws->prep_in_flight = false;
      }
      const i64* state_i64 = (const i64*)state;
      i64 tok = 0, pos = 0, s_c0 = 0, s_c1 = 0;
      const i64* expected_fold_tok_ptr =
          (pos_add == 1) ? (state_i64 + 2) : ((pos_add >= 2 && pos_add <= 7) ? (state_i64 + 6 + pos_add) : nullptr);
      if (pos_add >= 1 && pos_add <= 7 && state != nullptr &&
          state == ws->cached_state_ptr &&
          pos_ptr == (state_i64 + 1) &&
          tok_ptr == expected_fold_tok_ptr &&
          ctx_in == ws->cached_ctx_out_ptr) {
        // Fast path for folded multi-draft rows 1..7: reuse the 14-word `ctl` snapshot copied on row 0
        // with zero additional D2H copies and zero cudaStreamSynchronize calls!
        const int ctl_idx = (pos_add == 1) ? 2 : (6 + pos_add);
        tok = ws->scalars[ctl_idx];
        pos = ws->scalars[1] + pos_add;
        s_c0 = ws->cached_c0;
        s_c1 = ws->cached_c1;
        ws->cached_ctx_out_ptr = ctx_out;
        ws->cached_c0 = s_c1;
        ws->cached_c1 = tok;
      } else if (pos_add == 0 && state != nullptr && tok_ptr == state_i64 && pos_ptr == (state_i64 + 1)) {
        // Row 0 of EnqueueFold / EnqueueVerify / EnqueueStep: copy all 14 words of `ctl` (`ctl[0..13]`)
        // plus `ctx_in[0..1]` in one batch so subsequent draft rows 1..7 need no stream synchronization.
        err = staged_copy_to_host(&ws->scalars[0], state_i64, 14 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
        if (err == cudaSuccess) {
          err = staged_copy_to_host(&ws->scalars[14], ctx_in, 2 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
        }
        if (err == cudaSuccess) {
          err = staged_stream_sync(stream);
        }
        if (err != cudaSuccess) {
          // The snapshot may be half written: rows 1..7 must not reuse it.
          ws->invalidate_fold_snapshot();
          return (int)err;
        }
        ws->prep_in_flight = false;
        tok = ws->scalars[0];
        pos = ws->scalars[1];
        s_c0 = ws->scalars[14];
        s_c1 = ws->scalars[15];
        ws->cached_state_ptr = state;
        ws->cached_ctx_out_ptr = ctx_out;
        ws->cached_c0 = s_c1;
        ws->cached_c1 = tok;
      } else {
        ws->invalidate_fold_snapshot();
        err = staged_copy_to_host(&ws->scalars[0], tok_ptr, sizeof(i64), cudaMemcpyDeviceToHost, stream);
        if (err == cudaSuccess) {
          err = staged_copy_to_host(&ws->scalars[1], pos_ptr, sizeof(i64), cudaMemcpyDeviceToHost, stream);
        }
        if (err == cudaSuccess) {
          err = staged_copy_to_host(&ws->scalars[2], ctx_in, 2 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
        }
        if (err == cudaSuccess) {
          err = staged_stream_sync(stream);
        }
        if (err != cudaSuccess) {
          return (int)err;
        }
        ws->prep_in_flight = false;
        tok = ws->scalars[0];
        pos = ws->scalars[1] + pos_add;
        s_c0 = ws->scalars[2];
        s_c1 = ws->scalars[3];
      }
      const size_t slot_offset = (size_t)(ws->ring_idx++ % StagedPleWorkspace::kPrepRingRows) *
                                 StagedPleWorkspace::kPrepRingRowBytes;
      unsigned char* slot_host = ws->prep_ring + slot_offset;
      const unsigned char* slot_dev = ws->prep_ring_dev + slot_offset;
      host_gather_ple_row(*ws, tok, s_c0, s_c1, slot_host);
      hook_launch();
      BeginLaunch();
      k_prep_staged<<<1, 1024, 0, stream>>>(ws->params.capacity, ws->params.n_heads, ws->params.head_dim,
                                            ws->params.scale, tok, pos, s_c1, slot_dev, (i64*)ctx_out,
                                            (i64*)input_ids, (i64*)position_ids, (int*)committed, (i64*)past_pos,
                                            (__nv_bfloat16*)ple_out, (i64*)state);
      const int launch_err = LaunchStatus();
      if (launch_err != 0) {
        return launch_err;
      }
      return (int)staged_record_done(ws->prep_done_event, stream, &ws->last_prep_stream, &ws->prep_in_flight);
    }
  }
  BeginLaunch();
  k_prep<<<1, 1024, 0, (cudaStream_t)s>>>((const PleParams*)P, (const i64*)tok_ptr, (const i64*)pos_ptr, pos_add,
                                          (const i64*)ctx_in, (i64*)ctx_out, (i64*)input_ids, (i64*)position_ids,
                                          (int*)committed, (i64*)past_pos, (__nv_bfloat16*)ple_out, (i64*)state);
  return LaunchStatus();
}

QWEN38_PLANE_EXPORT void qwen38_plane_argmax_part(void* s, const void* logits, int n, void* outv, void* outi, int nblocks) {
  BeginLaunch();
  k_argmax_part<<<nblocks, 256, 0, (cudaStream_t)s>>>((const __nv_bfloat16*)logits, n, (float*)outv, (int*)outi);
  ParkLaunchStatus((cudaStream_t)s);
}

QWEN38_PLANE_EXPORT void qwen38_plane_post1(void* s, const void* v0, const void* i0, int nb, void* state, void* ctx_commit, const void* ctx_a,
                                            const void* st0, int nst, void* result, int ring) {
  BeginLaunch();
  k_post1<<<1, 32, 0, (cudaStream_t)s>>>((const float*)v0, (const int*)i0, nb, (i64*)state, (i64*)ctx_commit,
                                         (const i64*)ctx_a, (const int*)st0, nst, (i64*)result, ring);
  ParkLaunchStatus((cudaStream_t)s);
}

QWEN38_PLANE_EXPORT void qwen38_plane_post_tok(void* s, const void* tok01, void* state, void* ctx_commit, const void* ctx_a, const void* ctx_b,
                                               const void* st0, int nst, void* result, int ring) {
  BeginLaunch();
  k_post_tok<<<1, 32, 0, (cudaStream_t)s>>>((const i64*)tok01, (i64*)state, (i64*)ctx_commit, (const i64*)ctx_a,
                                            (const i64*)ctx_b, (const int*)st0, nst, (i64*)result, ring);
  ParkLaunchStatus((cudaStream_t)s);
}

QWEN38_PLANE_EXPORT void qwen38_plane_head_prep(void* s, const void* ctl, int rows, int hlen, int hidden_width, const void* src, void* dst,
                                                void* tokens, void* positions, void* committed) {
  BeginLaunch();
  k_head_prep<<<1, 1024, 0, (cudaStream_t)s>>>((const i64*)ctl, rows, hlen, hidden_width, (const uint4*)src, (uint4*)dst,
                                               (i64*)tokens, (i64*)positions, (int*)committed);
  ParkLaunchStatus((cudaStream_t)s);
}

QWEN38_PLANE_EXPORT void qwen38_plane_head_post(void* s, const void* next_token, void* ctl) {
  BeginLaunch();
  k_head_post<<<1, 32, 0, (cudaStream_t)s>>>((const i64*)next_token, (i64*)ctl);
  ParkLaunchStatus((cudaStream_t)s);
}

QWEN38_PLANE_EXPORT void qwen38_plane_or_status(void* s, const void* st, int nst, void* state) {
  BeginLaunch();
  k_or_status<<<1, 32, 0, (cudaStream_t)s>>>((const int*)st, nst, (i64*)state);
  ParkLaunchStatus((cudaStream_t)s);
}

/* ---- fast prefill kernels: batched PLE rows, dense -> packed state hand-off ---- */
QWEN38_PLANE_EXPORT int qwen38_plane_ple_rows(void* s, const void* P, const void* tokens, int n, const void* ctx_in, void* ctx_out,
                                              void* ple_out) {
  if (n <= 0) return 0;
  {
    std::lock_guard<std::mutex> lock(g_staged_ple_mu);
    cudaStream_t stream = (cudaStream_t)s;
    bool unregistered_staged = false;
    cudaError_t err = cudaSuccess;
    StagedPleWorkspace* ws = resolve_staged_ple_workspace(P, stream, /*allow_sync=*/true, &unregistered_staged, &err);
    if (err != cudaSuccess) {
      return (int)err;
    }
    if (unregistered_staged) {
      return (int)cudaErrorInvalidValue;
    }
    if (ws != nullptr) {
      if (staged_stream_is_capturing(stream)) {
        return (int)cudaErrorStreamCaptureUnsupported;
      }
      const size_t total = (size_t)ws->params.n_heads * (size_t)ws->params.head_dim;
      err = ensure_staged_buffers(*ws, (size_t)n, total);
      if (err == cudaSuccess) {
        err = staged_ensure_done_event(&ws->prefill_done_event);
      }
      if (err != cudaSuccess) {
        return (int)err;
      }
      if (ws->prefill_in_flight && ws->last_prefill_stream != stream) {
        err = cudaEventSynchronize(ws->prefill_done_event);
        if (err != cudaSuccess) {
          cudaGetLastError();
          return (int)err;
        }
        ws->prefill_in_flight = false;
      }
      // The context words land in scalars[2..3], which the snapshot of a fold in progress keeps its draft token in.
      ws->invalidate_fold_snapshot();
      err = staged_copy_to_host(ws->prefill_tokens, tokens, (size_t)n * sizeof(i64), cudaMemcpyDeviceToHost, stream);
      if (err == cudaSuccess) {
        err = staged_copy_to_host(&ws->scalars[2], ctx_in, 2 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
      }
      if (err == cudaSuccess) {
        err = staged_stream_sync(stream);
      }
      if (err != cudaSuccess) {
        return (int)err;
      }
      ws->prefill_in_flight = false;
      const i64* h_toks = ws->prefill_tokens;
      const i64 ctx0 = ws->scalars[2];
      const i64 ctx1 = ws->scalars[3];
      if (n >= 32) {
        PleGatherPool* gather_pool = get_ple_gather_pool_locked();
        err = gather_pool != nullptr ? gather_pool->run(*ws, h_toks, ctx0, ctx1, n, ws->prefill_fp8, total)
                                     : cudaErrorMemoryAllocation;
        if (err != cudaSuccess) {
          return (int)err;
        }
        hook_gather(n);
      } else {
        for (int i = 0; i < n; ++i) {
          const i64 tok = h_toks[i];
          const i64 c1 = i >= 1 ? h_toks[i - 1] : ctx1;
          const i64 c0raw = i >= 2 ? h_toks[i - 2] : (i == 1 ? ctx1 : ctx0);
          host_gather_ple_row(*ws, tok, c0raw, c1, ws->prefill_fp8 + (size_t)i * total);
        }
      }
      const i64 last_tok = h_toks[n - 1];
      const i64 last_c1 = n >= 2 ? h_toks[n - 2] : ctx1;
      hook_launch();
      BeginLaunch();
      k_ple_rows_staged<<<n, 1024, 0, stream>>>(ws->params.n_heads, ws->params.head_dim, ws->params.scale,
                                                ws->prefill_fp8_dev, n, last_c1, last_tok, (i64*)ctx_out,
                                                (__nv_bfloat16*)ple_out);
      const int launch_err = LaunchStatus();
      if (launch_err != 0) {
        return launch_err;
      }
      return (int)staged_record_done(ws->prefill_done_event, stream, &ws->last_prefill_stream, &ws->prefill_in_flight);
    }
  }
  BeginLaunch();
  k_ple_rows<<<n, 1024, 0, (cudaStream_t)s>>>((const PleParams*)P, (const i64*)tokens, n, (const i64*)ctx_in, (i64*)ctx_out,
                                              (__nv_bfloat16*)ple_out);
  return LaunchStatus();
}

QWEN38_PLANE_EXPORT void qwen38_plane_pool_keys(void* s, const void* raw, int nb, const void* knorm, const void* cosv, const void* sinv,
                                                int width, float eps, int first_block, void* out) {
  if (nb <= 0) return;
  BeginLaunch();
  k_pool_keys<<<nb, 128, 0, (cudaStream_t)s>>>((const __nv_bfloat16*)raw, (const __nv_bfloat16*)knorm,
                                               (const __nv_bfloat16*)cosv, (const __nv_bfloat16*)sinv, width, eps,
                                               first_block, (float*)out);
  ParkLaunchStatus((cudaStream_t)s);
}

QWEN38_PLANE_EXPORT void qwen38_plane_transpose_f32(void* s, const void* src, void* dst, int n, int batch) {
  if (n <= 0 || batch <= 0) return;
  const unsigned int blocks = static_cast<unsigned int>((n + 31) / 32);
  dim3 grid(blocks, blocks, static_cast<unsigned int>(batch)), block(32, 8);
  BeginLaunch();
  k_transpose_f32<<<grid, block, 0, (cudaStream_t)s>>>((const float*)src, (float*)dst, n);
  ParkLaunchStatus((cudaStream_t)s);
}

QWEN38_PLANE_EXPORT void qwen38_plane_quantize_kv(void* s, const void* src, void* dst, const void* scale, int rows, int src_pitch_elems,
                                                  int dst_pitch_elems, int mode) {
  if (rows <= 0) return;
  BeginLaunch();
  k_quantize_kv<<<rows, 256, 0, (cudaStream_t)s>>>((const __nv_bfloat16*)src, (uint8_t*)dst, (const float*)scale, rows,
                                                   src_pitch_elems, dst_pitch_elems, mode);
  ParkLaunchStatus((cudaStream_t)s);
}

QWEN38_PLANE_EXPORT void qwen38_plane_embed_rows(void* s, const void* table, const void* ids, void* out, int rows, int width) {
  if (rows <= 0 || width <= 0) return;
  // Kept for loaders that predate embed_gather: the row count of the table is unknown here, so no id is too large.
  BeginLaunch();
  k_embed_rows<<<rows, 256, 0, (cudaStream_t)s>>>((const __nv_bfloat16*)table, LLONG_MAX, (const i64*)ids, (__nv_bfloat16*)out, rows,
                                                  width);
  ParkLaunchStatus((cudaStream_t)s);
}

QWEN38_PLANE_EXPORT int qwen38_plane_embed_gather(void* s, const void* table, long long table_rows, int format, const void* ids,
                                                  void* out, int n, int width) {
  if (n <= 0) return 0;
  if (table_rows <= 0 || width <= 0) return (int)cudaErrorInvalidValue;
  if (format == QWEN38_PLANE_EMBEDDING_INT8_ROW) {
    // The row scales follow the int8 payload, whose size comes from the row count of the table.
    const size_t payload_bytes = (size_t)table_rows * (size_t)width;
    if (payload_bytes % sizeof(float) != 0) return (int)cudaErrorInvalidValue;
    const int8_t* payload = static_cast<const int8_t*>(table);
    const float* row_scales = reinterpret_cast<const float*>(payload + payload_bytes);
    BeginLaunch();
    k_embed_rows_int8<<<n, 256, 0, (cudaStream_t)s>>>(payload, row_scales, table_rows, (const i64*)ids, (__nv_bfloat16*)out, n,
                                                      width);
    return LaunchStatus();
  }
  if (format != QWEN38_PLANE_EMBEDDING_BF16) return (int)cudaErrorInvalidValue;
  // A BF16 row is copied in 16-byte vectors: another width would leave the tail of each row unwritten and misalign every
  // row after the first.
  if (width % 8 != 0) return (int)cudaErrorInvalidValue;
  BeginLaunch();
  k_embed_rows<<<n, 256, 0, (cudaStream_t)s>>>((const __nv_bfloat16*)table, table_rows, (const i64*)ids, (__nv_bfloat16*)out, n,
                                               width);
  return LaunchStatus();
}

static_assert(std::is_same<decltype(&qwen38_plane_capabilities), Qwen38PlaneCapabilitiesFn>::value,
              "qwen38_plane_capabilities must match Qwen38PlaneCapabilitiesFn");
static_assert(std::is_same<decltype(&qwen38_plane_embed_gather), Qwen38PlaneEmbedGatherFn>::value,
              "qwen38_plane_embed_gather must match Qwen38PlaneEmbedGatherFn");

/* ---- CUDA runtime helpers (all return cudaError_t as int, 0 = success) ---- */
QWEN38_PLANE_EXPORT int qwen38_plane_set_device(int device) { return (int)cudaSetDevice(device); }
QWEN38_PLANE_EXPORT const char* qwen38_plane_error_string(int err) { return cudaGetErrorString((cudaError_t)err); }
QWEN38_PLANE_EXPORT int qwen38_plane_device_alloc(void** p, size_t bytes) { return (int)cudaMalloc(p, bytes); }
QWEN38_PLANE_EXPORT int qwen38_plane_device_free(void* p) {
  {
    std::lock_guard<std::mutex> lock(g_staged_ple_mu);
    if (!g_dev_to_host_ple.empty()) {
      g_dev_to_host_ple.erase(p);
    }
  }
  return (int)cudaFree(p);
}
QWEN38_PLANE_EXPORT int qwen38_plane_host_alloc(void** p, size_t bytes) { return (int)cudaHostAlloc(p, bytes, cudaHostAllocPortable); }
QWEN38_PLANE_EXPORT int qwen38_plane_host_free(void* p) { return (int)cudaFreeHost(p); }
QWEN38_PLANE_EXPORT int qwen38_plane_host_register(void* p, size_t bytes, unsigned flags) {
  cudaError_t err = cudaHostRegister(p, bytes, flags);
  if (err != cudaSuccess && (flags & 8u) != 0) {
    cudaGetLastError();
    err = cudaHostRegister(p, bytes, flags & ~8u);
  }
  // The caller gets the error as the return value; a failure the host tolerates must not stay behind as the last error of
  // the thread, where the next cudaGetLastError() of another CUDA user of the process would report it.
  if (err != cudaSuccess) cudaGetLastError();
  return (int)err;
}
QWEN38_PLANE_EXPORT int qwen38_plane_host_unregister(void* p) {
  cudaError_t err = cudaHostUnregister(p);
  if (err != cudaSuccess) cudaGetLastError();
  return (int)err;
}
QWEN38_PLANE_EXPORT int qwen38_plane_host_get_device_pointer(void** dp, void* hp, unsigned flags) {
  cudaError_t err = cudaHostGetDevicePointer(dp, hp, flags);
  if (err != cudaSuccess) cudaGetLastError();
  return (int)err;
}
QWEN38_PLANE_EXPORT int qwen38_plane_device_memset(void* p, int value, size_t bytes, void* s) {
  return (int)cudaMemsetAsync(p, value, bytes, (cudaStream_t)s);
}
QWEN38_PLANE_EXPORT int qwen38_plane_memcpy_h2d(void* dst, const void* src, size_t bytes, void* s) {
  return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, (cudaStream_t)s);
}
QWEN38_PLANE_EXPORT int qwen38_plane_memcpy_d2h(void* dst, const void* src, size_t bytes, void* s) {
  return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, (cudaStream_t)s);
}
QWEN38_PLANE_EXPORT int qwen38_plane_memcpy_d2d(void* dst, const void* src, size_t bytes, void* s) {
  return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, (cudaStream_t)s);
}
QWEN38_PLANE_EXPORT int qwen38_plane_memcpy2d_d2d(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height,
                                                  void* s) {
  return (int)cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDeviceToDevice, (cudaStream_t)s);
}
QWEN38_PLANE_EXPORT int qwen38_plane_stream_create(void** s) { return (int)cudaStreamCreateWithFlags((cudaStream_t*)s, cudaStreamNonBlocking); }
QWEN38_PLANE_EXPORT int qwen38_plane_stream_destroy(void* s) {
  ClearParkedLaunchStatusForStream((cudaStream_t)s);
  return (int)cudaStreamDestroy((cudaStream_t)s);
}
// Besides the stream's own status this reports a launch that an earlier void export could not enqueue on this stream
// (see g_parked_launch_errors), once.
QWEN38_PLANE_EXPORT int qwen38_plane_stream_sync(void* s) {
  const cudaStream_t stream = (cudaStream_t)s;
  const int parked = TakeParkedLaunchStatus(stream);
  const cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) return (int)err;
  return parked;
}

#ifdef QWEN38_PLANE_TEST_HOOKS
QWEN38_PLANE_EXPORT int qwen38_plane_test_inject_staged_fault(int site, int skip, int error) {
  const bool disarm = site == QWEN38_PLANE_FAULT_NONE;
  if (site < QWEN38_PLANE_FAULT_NONE || site > QWEN38_PLANE_FAULT_THREAD_START || (!disarm && error == 0)) {
    return (int)cudaErrorInvalidValue;
  }
  std::lock_guard<std::mutex> lock(g_staged_ple_mu);
  g_test_hooks.fault_site = site;
  g_test_hooks.fault_skip = skip < 0 ? 0 : skip;
  g_test_hooks.fault_error = error;
  return 0;
}

QWEN38_PLANE_EXPORT int qwen38_plane_test_stats(Qwen38PlaneTestStats* stats) {
  if (stats == nullptr) return (int)cudaErrorInvalidValue;
  std::lock_guard<std::mutex> lock(g_staged_ple_mu);
  long long capacity_rows = 0;
  for (const auto& kv : g_host_ple_workspaces) {
    if (kv.second) capacity_rows += (long long)kv.second->prefill_cap_rows;
  }
  stats->gather_rows = g_test_hooks.gather_rows;
  stats->staged_launches = g_test_hooks.staged_launches;
  stats->global_cleanups = g_test_hooks.global_cleanups;
  stats->live_pinned_allocations = g_test_hooks.live_pinned;
  stats->prefill_capacity_rows = capacity_rows;
  stats->staged_tables = (long long)g_host_ple_workspaces.size();
  stats->attached_loaders = g_attached_loaders;
  stats->fault_armed = g_test_hooks.fault_site != QWEN38_PLANE_FAULT_NONE ? 1 : 0;
  stats->unfenced_drains = g_test_hooks.unfenced_drains;
  stats->gather_pool_threads = g_ple_gather_pool != nullptr ? g_ple_gather_pool->pool.ThreadCount() : 0;
  return 0;
}

QWEN38_PLANE_EXPORT int qwen38_plane_test_peek_last_error(void) { return (int)cudaPeekAtLastError(); }

// The graph of the capture is thrown away: tests only need the stream to be capturing while they call the module.
QWEN38_PLANE_EXPORT int qwen38_plane_test_stream_capture(void* stream, int begin) {
  const cudaStream_t s = (cudaStream_t)stream;
  if (begin != 0) return (int)cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
  cudaGraph_t graph = nullptr;
  const cudaError_t err = cudaStreamEndCapture(s, &graph);
  if (graph != nullptr) cudaGraphDestroy(graph);
  return (int)err;
}
#endif
}

// The graph inputs of the `rows` rows of one folded verify step for each of `batch` request slots (QWEN38_PLANE_CAP_BATCHED_PREP;
// the contract is in qwen38_plane_abi.h). Both counts are checked before anything is launched.
extern "C" QWEN38_PLANE_EXPORT int qwen38_plane_prep_batched(void* s, const void* P, int rows, int batch, const void* ctx_in,
                                                             void* ctx_out, void* input_ids, void* position_ids,
                                                             void* committed, void* past_pos, void* ple_out, void* state) {
  if (rows < 1 || rows > QWEN38_PLANE_MAX_PREP_ROWS || batch < 1 || batch > QWEN38_PLANE_MAX_PREP_BATCH || P == nullptr ||
      ctx_in == nullptr || ctx_out == nullptr || input_ids == nullptr || position_ids == nullptr || past_pos == nullptr ||
      ple_out == nullptr || state == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaStream_t stream = static_cast<cudaStream_t>(s);
  size_t staged_row_width = 0;
  size_t staged_capacity = 0;
  const cudaError_t resolved = resolve_staged_ple_geometry(P, stream, &staged_row_width, &staged_capacity);
  if (resolved != cudaSuccess) return static_cast<int>(resolved);
  if (staged_row_width != 0) {
    // The host-table gather is per row, so the staged path gathers the rows of one slot on the host and prepares them with one
    // launch (the plain next-token row, rows == 1, is the one-row prep), slot after slot. Row 0 owns the committed length of
    // its slot: no other row is handed a committed pointer, so nothing but device memory reaches a kernel.
    for (int b = 0; b < batch; ++b) {
      i64* state_b = static_cast<i64*>(state) + static_cast<size_t>(b) * QWEN38_PLANE_CTL_WORDS_PER_SLOT;
      const i64* ctx_in_b = static_cast<const i64*>(ctx_in) + static_cast<size_t>(b) * 2;
      i64* ctx_out_b = static_cast<i64*>(ctx_out) + static_cast<size_t>(b) * 2 * static_cast<size_t>(rows);
      i64* input_ids_b = static_cast<i64*>(input_ids) + static_cast<size_t>(b) * rows;
      i64* position_ids_b = static_cast<i64*>(position_ids) + static_cast<size_t>(b) * rows;
      int* committed_b = committed != nullptr ? static_cast<int*>(committed) + b : nullptr;
      i64* past_pos_b = static_cast<i64*>(past_pos) + static_cast<size_t>(b) * staged_capacity;
      __nv_bfloat16* ple_out_b =
          static_cast<__nv_bfloat16*>(ple_out) + static_cast<size_t>(b) * static_cast<size_t>(rows) * staged_row_width;
      const int rc = rows >= 2 ? prep_staged_fold_rows(stream, P, rows, ctx_in_b, ctx_out_b, input_ids_b, position_ids_b,
                                                       committed_b, past_pos_b, ple_out_b, state_b)
                               : qwen38_plane_prep(s, P, state_b, state_b + 1, 0, ctx_in_b, ctx_out_b, input_ids_b,
                                                   position_ids_b, committed_b, past_pos_b, ple_out_b, state_b);
      if (rc != 0) return rc;
    }
    return 0;
  }
  BeginLaunch();
  k_prep_fold<<<dim3(static_cast<unsigned>(rows), static_cast<unsigned>(batch)), 1024, 0, stream>>>(
      static_cast<const PleParams*>(P), static_cast<const i64*>(ctx_in), static_cast<i64*>(ctx_out),
      static_cast<i64*>(input_ids), static_cast<i64*>(position_ids), static_cast<int*>(committed),
      static_cast<i64*>(past_pos), static_cast<__nv_bfloat16*>(ple_out), static_cast<i64*>(state), /*honor_mute=*/1);
  return LaunchStatus();
}

__global__ void DflashSelect(const int32_t* candidates, const float* scores, i64* control,
                             int drafts, int top_k) {
  const int slots[] = {2, 8, 9, 10, 11, 12, 13};
  int previous = 0;
  for (int step = 0; step < drafts; ++step) {
    const float* row = scores + (step * top_k + previous) * top_k;
    int best = 0;
    for (int index = 1; index < top_k; ++index) {
      if (row[best] < row[index]) best = index;
    }
    control[slots[step]] = candidates[step * top_k + best];
    previous = best;
  }
}

extern "C" QWEN38_PLANE_EXPORT int qwen38_plane_dflash_select(
    void* stream, const void* candidates, const void* scores,
    void* control, int drafts, int top_k) {
  if (!candidates || !scores || !control || drafts < 1 || drafts > 7 || top_k < 1)
    return static_cast<int>(cudaErrorInvalidValue);
  BeginLaunch();
  DflashSelect<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const int32_t*>(candidates), static_cast<const float*>(scores),
      static_cast<i64*>(control), drafts, top_k);
  return LaunchStatus();
}
static_assert(std::is_same<decltype(&qwen38_plane_dflash_select), Qwen38PlaneDflashSelectFn>::value,
              "qwen38_plane_dflash_select must match Qwen38PlaneDflashSelectFn");

namespace {

constexpr int kPlaneReplayThreads = 256;
constexpr int kPlaneReplayRows = QWEN38_PLANE_REPLAY_ROWS_PER_BLOCK;
constexpr int kPlaneMaxReplayTransitions = QWEN38_PLANE_MAX_REPLAY_TRANSITIONS;
constexpr int kPlaneMaxReplayKeyWidth = QWEN38_PLANE_MAX_REPLAY_KEY_WIDTH;
static_assert(kPlaneMaxReplayKeyWidth % 4 == 0, "k_replay_gdn loads float4 along the key width");
constexpr int kPlaneGenericBlocksPerDesc = 128;

__global__ void __launch_bounds__(kPlaneReplayThreads) k_replay_gdn(
    const Qwen38StateReplayDesc* __restrict__ descs, int kept_override) {
  __shared__ float key_tile[kPlaneMaxReplayTransitions * kPlaneMaxReplayKeyWidth];
  __shared__ float delta_tile[kPlaneMaxReplayTransitions * kPlaneReplayRows];
  __shared__ float decay_tile[kPlaneMaxReplayTransitions];

  const Qwen38StateReplayDesc d = descs[blockIdx.y];
  const int value_width = static_cast<int>(d.state_width);
  const int key_width = static_cast<int>(d.key_width);
  const int heads = static_cast<int>(d.channel_count);
  const int key_heads = static_cast<int>(d.key_head_count);
  const int kept = kept_override > 0 ? kept_override : static_cast<int>(d.kept_count);
  const int chunks = QWEN38_PLANE_REPLAY_BLOCKS_PER_HEAD(value_width);
  const int head = blockIdx.x / chunks;
  if (head >= heads) return;
  const int first_row = (blockIdx.x - head * chunks) * kPlaneReplayRows;
  const int key_head = head / (heads / key_heads);

  for (int i = threadIdx.x; i < kept * key_width; i += blockDim.x) {
    const int t = i / key_width;
    const int k = i - t * key_width;
    key_tile[t * key_width + k] =
        d.key[(static_cast<size_t>(t) * key_heads + key_head) * key_width + k];
  }
  for (int i = threadIdx.x; i < kept * kPlaneReplayRows; i += blockDim.x) {
    const int t = i / kPlaneReplayRows;
    const int r = i - t * kPlaneReplayRows;
    const int v = first_row + r;
    delta_tile[i] = v < value_width
                        ? d.delta[(static_cast<size_t>(t) * heads + head) * value_width + v]
                        : 0.0f;
  }
  if (static_cast<int>(threadIdx.x) < kept) {
    decay_tile[threadIdx.x] = d.decay[static_cast<size_t>(threadIdx.x) * heads + head];
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const auto* source = static_cast<const float*>(d.source_state);
  auto* destination = static_cast<float*>(d.destination_state);
  for (int r = warp; r < kPlaneReplayRows; r += kPlaneReplayThreads / 32) {
    const int v = first_row + r;
    if (v >= value_width) break;
    const size_t row = (static_cast<size_t>(head) * value_width + v) * key_width;
    for (int k = lane * 4; k < key_width; k += 128) {
      float4 s = *reinterpret_cast<const float4*>(source + row + k);
      for (int t = 0; t < kept; ++t) {
        const float decay = decay_tile[t];
        const float delta = delta_tile[t * kPlaneReplayRows + r];
        const float* key = key_tile + t * key_width + k;
        s.x = __fmaf_rn(key[0], delta, __fmul_rn(s.x, decay));
        s.y = __fmaf_rn(key[1], delta, __fmul_rn(s.y, decay));
        s.z = __fmaf_rn(key[2], delta, __fmul_rn(s.z, decay));
        s.w = __fmaf_rn(key[3], delta, __fmul_rn(s.w, decay));
      }
      *reinterpret_cast<float4*>(destination + row + k) = s;
    }
  }
}

__global__ void k_replay_generic(const Qwen38StateReplayDesc* __restrict__ descs, int kept_override) {
  const Qwen38StateReplayDesc descriptor = descs[blockIdx.y];
  const uint32_t kept_count = kept_override > 0 ? static_cast<uint32_t>(kept_override) : descriptor.kept_count;
  const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
  const uint64_t start = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (descriptor.kind == 4) {
    const int32_t old_key_length = descriptor.source_lengths[0];
    const int32_t old_buffer_length = descriptor.source_lengths[1];
    const uint64_t new_block_count =
        (static_cast<uint64_t>(old_buffer_length) + kept_count) / descriptor.compress_ratio;
    const uint64_t new_buffer_length =
        (static_cast<uint64_t>(old_buffer_length) + kept_count) % descriptor.compress_ratio;
    for (uint64_t index = start; index < descriptor.aux_capacity * descriptor.state_width; index += stride) {
      const uint64_t entry = index / descriptor.state_width;
      const uint64_t component = index % descriptor.state_width;
      uint64_t source_index = index;
      const void* source = descriptor.source_aux_state;
      if (entry < new_buffer_length) {
        const uint64_t virtual_position = new_block_count * descriptor.compress_ratio + entry;
        const bool from_old_buffer = virtual_position < static_cast<uint64_t>(old_buffer_length);
        source_index =
            (from_old_buffer ? virtual_position : virtual_position - old_buffer_length) *
                descriptor.state_width +
            component;
        source = from_old_buffer ? descriptor.source_aux_state : descriptor.value;
      }
      if (descriptor.element_size == 2) {
        static_cast<uint16_t*>(descriptor.destination_aux_state)[index] =
            static_cast<const uint16_t*>(source)[source_index];
      } else {
        static_cast<uint32_t*>(descriptor.destination_aux_state)[index] =
            static_cast<const uint32_t*>(source)[source_index];
      }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      descriptor.destination_lengths[0] = old_key_length + static_cast<int32_t>(new_block_count);
      descriptor.destination_lengths[1] = static_cast<int32_t>(new_buffer_length);
    }
    return;
  }

  if (descriptor.kind == 1) {
    const uint64_t state_elements = descriptor.channel_count * descriptor.state_width;
    for (uint64_t state_index = start; state_index < state_elements; state_index += stride) {
      const uint64_t channel = state_index / descriptor.state_width;
      const uint64_t position = state_index - channel * descriptor.state_width;
      const uint64_t shifted_position = position + kept_count;
      const uint64_t source_index = shifted_position < descriptor.state_width
                                        ? channel * descriptor.state_width + shifted_position
                                        : (shifted_position - descriptor.state_width) *
                                                  descriptor.channel_count +
                                              channel;
      if (descriptor.element_size == 2) {
        const auto* source = shifted_position < descriptor.state_width
                                 ? static_cast<const uint16_t*>(descriptor.source_state)
                                 : static_cast<const uint16_t*>(descriptor.value);
        static_cast<uint16_t*>(descriptor.destination_state)[state_index] = source[source_index];
      } else if (descriptor.element_size == 4) {
        const auto* source = shifted_position < descriptor.state_width
                                 ? static_cast<const uint32_t*>(descriptor.source_state)
                                 : static_cast<const uint32_t*>(descriptor.value);
        static_cast<uint32_t*>(descriptor.destination_state)[state_index] = source[source_index];
      }
    }
  }
}

}  // namespace

extern "C" QWEN38_PLANE_EXPORT int qwen38_plane_replay_state_updates(
    void* stream, const void* descs, int fast_count, int generic_count,
    int fast_blocks_per_descriptor, int kept_count) {
  // Counts and block budgets that would skip a launch without a word leave the destination bank stale: they are refused.
  if (!descs || kept_count < 1 || kept_count > kPlaneMaxReplayTransitions || fast_count < 0 || generic_count < 0 ||
      (fast_count > 0 && fast_blocks_per_descriptor < 1)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto* typed = static_cast<const Qwen38StateReplayDesc*>(descs);
  cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
  BeginLaunch();
  if (fast_count > 0) {
    const dim3 grid(static_cast<unsigned>(fast_blocks_per_descriptor), static_cast<unsigned>(fast_count));
    k_replay_gdn<<<grid, kPlaneReplayThreads, 0, cu_stream>>>(typed, kept_count);
    if (const int status = LaunchStatus()) return status;
  }
  if (generic_count > 0) {
    const dim3 grid(kPlaneGenericBlocksPerDesc, static_cast<unsigned>(generic_count));
    k_replay_generic<<<grid, kPlaneReplayThreads, 0, cu_stream>>>(typed + fast_count, kept_count);
    if (const int status = LaunchStatus()) return status;
  }
  return 0;
}
static_assert(std::is_same<decltype(&qwen38_plane_replay_state_updates), Qwen38PlaneReplayStateUpdatesFn>::value,
              "qwen38_plane_replay_state_updates must match Qwen38PlaneReplayStateUpdatesFn");
