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
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <memory>
#include <mutex>
#include <unordered_map>

#include "qwen38_plane_abi.h"

typedef long long i64;
typedef Qwen38PleParams PleParams;

struct StagedPleWorkspace {
  PleParams params{};
  i64* scalars = nullptr;                   // pinned [64]: tok/ctl words + ctx_in
  unsigned char* prep_ring = nullptr;       // pinned mapped [64 * 8192]
  unsigned char* prep_ring_dev = nullptr;   // device alias for prep_ring
  unsigned int ring_idx = 0;
  i64* prefill_tokens = nullptr;            // pinned [cap_rows]
  unsigned char* prefill_fp8 = nullptr;     // pinned mapped [cap_rows * max_total]
  unsigned char* prefill_fp8_dev = nullptr; // device alias for prefill_fp8
  size_t prefill_cap_rows = 0;
  size_t prefill_cap_bytes = 0;
  const void* cached_state_ptr = nullptr;
  const void* cached_ctx_out_ptr = nullptr;
  i64 cached_c0 = 0;
  i64 cached_c1 = 0;
  static constexpr size_t kCacheSlots = 2048;
  static constexpr size_t kMaxRowBytes = 8192;
  unsigned char* ngram_cache_data = nullptr;
  i64 ngram_cache_keys[kCacheSlots][3]{};
  unsigned char ngram_cache_valid[kCacheSlots]{};

  void reset_cache() {
    cached_state_ptr = nullptr;
    cached_ctx_out_ptr = nullptr;
    cached_c0 = 0;
    cached_c1 = 0;
    memset(ngram_cache_valid, 0, sizeof(ngram_cache_valid));
  }

  bool has_cuda_allocations() const {
    return scalars != nullptr || prep_ring != nullptr || prefill_tokens != nullptr || prefill_fp8 != nullptr;
  }

  void release(bool sync_device) {
    if (sync_device && has_cuda_allocations()) {
      if (cudaDeviceSynchronize() != cudaSuccess) {
        cudaGetLastError();
      }
    }
    if (scalars) {
      cudaFreeHost(scalars);
      scalars = nullptr;
    }
    if (prep_ring) {
      cudaFreeHost(prep_ring);
      prep_ring = nullptr;
      prep_ring_dev = nullptr;
    }
    if (prefill_tokens) {
      cudaFreeHost(prefill_tokens);
      prefill_tokens = nullptr;
    }
    if (prefill_fp8) {
      cudaFreeHost(prefill_fp8);
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
static bool g_ever_had_staged_ple = false;

static inline void ensure_prep_workspace(StagedPleWorkspace& ws) {
  if (!ws.scalars) {
    cudaHostAlloc((void**)&ws.scalars, 64 * sizeof(i64), cudaHostAllocPortable);
  }
  if (!ws.prep_ring) {
    cudaHostAlloc((void**)&ws.prep_ring, 64 * 8192, cudaHostAllocMapped | cudaHostAllocPortable);
    void* dev_ptr = nullptr;
    if (cudaHostGetDevicePointer(&dev_ptr, ws.prep_ring, 0) == cudaSuccess && dev_ptr != nullptr) {
      ws.prep_ring_dev = static_cast<unsigned char*>(dev_ptr);
    } else {
      cudaGetLastError();
      ws.prep_ring_dev = ws.prep_ring;
    }
  }
  if (!ws.ngram_cache_data) {
    ws.ngram_cache_data =
        (unsigned char*)malloc(StagedPleWorkspace::kCacheSlots * StagedPleWorkspace::kMaxRowBytes);
    memset(ws.ngram_cache_valid, 0, sizeof(ws.ngram_cache_valid));
  }
}

static inline void ensure_prefill_workspace(StagedPleWorkspace& ws, size_t rows, size_t row_bytes) {
  ensure_prep_workspace(ws);
  if (rows > ws.prefill_cap_rows) {
    if (ws.prefill_tokens) cudaFreeHost(ws.prefill_tokens);
    size_t new_rows = rows < 4096 ? 4096 : rows;
    cudaHostAlloc((void**)&ws.prefill_tokens, new_rows * sizeof(i64), cudaHostAllocPortable);
    ws.prefill_cap_rows = new_rows;
  }
  size_t need_bytes = rows * row_bytes;
  if (need_bytes > ws.prefill_cap_bytes) {
    if (ws.prefill_fp8) cudaFreeHost(ws.prefill_fp8);
    size_t new_bytes = need_bytes < (4096ULL * 4096ULL) ? (4096ULL * 4096ULL) : need_bytes;
    cudaHostAlloc((void**)&ws.prefill_fp8, new_bytes, cudaHostAllocMapped | cudaHostAllocPortable);
    void* dev_ptr = nullptr;
    if (cudaHostGetDevicePointer(&dev_ptr, ws.prefill_fp8, 0) == cudaSuccess && dev_ptr != nullptr) {
      ws.prefill_fp8_dev = static_cast<unsigned char*>(dev_ptr);
    } else {
      cudaGetLastError();
      ws.prefill_fp8_dev = ws.prefill_fp8;
    }
    ws.prefill_cap_bytes = new_bytes;
  }
}

static inline void host_gather_ple_row(StagedPleWorkspace& ws, i64 tok, i64 s_c0, i64 s_c1, unsigned char* dst_fp8) {
  const PleParams& HP = ws.params;
  const i64 c1 = s_c1;
  const i64 c0 = (c1 == HP.eos) ? HP.eos : s_c0;
  const int hd = HP.head_dim;
  const size_t total_bytes = (size_t)HP.n_heads * (size_t)hd;
  size_t cache_slot = 0;
  const bool use_cache = (ws.ngram_cache_data != nullptr && total_bytes <= StagedPleWorkspace::kMaxRowBytes);
  if (use_cache) {
    const unsigned long long h =
        ((unsigned long long)tok * 11400714819323198485ULL) ^
        ((unsigned long long)c1 * 14029467366897019727ULL) ^
        ((unsigned long long)c0 * 1609587929392839161ULL);
    cache_slot = (size_t)(h & (StagedPleWorkspace::kCacheSlots - 1));
    if (ws.ngram_cache_valid[cache_slot] &&
        ws.ngram_cache_keys[cache_slot][0] == tok &&
        ws.ngram_cache_keys[cache_slot][1] == c0 &&
        ws.ngram_cache_keys[cache_slot][2] == c1) {
      memcpy(dst_fp8, ws.ngram_cache_data + cache_slot * StagedPleWorkspace::kMaxRowBytes, total_bytes);
      return;
    }
  }
  for (int h = 0; h < HP.n_heads; ++h) {
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
    const unsigned char* src = HP.shard_ptr[lo] + (size_t)row_idx * (size_t)hd;
    memcpy(dst_fp8 + (size_t)h * (size_t)hd, src, (size_t)hd);
  }
  if (use_cache) {
    memcpy(ws.ngram_cache_data + cache_slot * StagedPleWorkspace::kMaxRowBytes, dst_fp8, total_bytes);
    ws.ngram_cache_keys[cache_slot][0] = tok;
    ws.ngram_cache_keys[cache_slot][1] = c0;
    ws.ngram_cache_keys[cache_slot][2] = c1;
    ws.ngram_cache_valid[cache_slot] = 1;
  }
}

static StagedPleWorkspace* resolve_staged_ple_workspace(const void* P, cudaStream_t stream, bool allow_sync,
                                                        bool* out_unregistered_staged) {
  *out_unregistered_staged = false;
  if (g_host_ple_workspaces.empty() && !g_ever_had_staged_ple) {
    return nullptr;
  }
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
  if (!allow_sync && g_host_ple_workspaces.size() == 1) {
    return g_host_ple_workspaces.begin()->second.get();
  }
  PleParams dev_params{};
  cudaError_t err = cudaMemcpyAsync(&dev_params, P, sizeof(PleParams), cudaMemcpyDefault, stream);
  if (err == cudaSuccess) {
    err = cudaStreamSynchronize(stream);
  }
  if (err != cudaSuccess) {
    cudaGetLastError();
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
    committed[0] = (int)pos;
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
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    if (j < total) {
      const __half_raw hr = __nv_cvt_fp8_to_halfraw(b[u], __NV_E4M3);
      const float f = __half2float(*reinterpret_cast<const __half*>(&hr));
      ple_out[j] = __float2bfloat16_rn(f * scale);
    }
  }
  if (tid == 0) {
    ctx_out[0] = s_c1;
    ctx_out[1] = tok;
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
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    if (j < total) {
      const __half_raw hr = __nv_cvt_fp8_to_halfraw(b[u], __NV_E4M3);
      const float f = __half2float(*reinterpret_cast<const __half*>(&hr));
      out[j] = __float2bfloat16_rn(f * scale);
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
    committed[0] = (int)pos;
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
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    if (j < total) {
      const __half_raw hr = __nv_cvt_fp8_to_halfraw(b[u], __NV_E4M3);
      const float f = __half2float(*reinterpret_cast<const __half*>(&hr));
      ple_out[j] = __float2bfloat16_rn(f * P->scale);
    }
  }
  if (tid == 0) {
    ctx_out[0] = c1;  // history[-2:] = (c1, tok); uses the raw (unmasked) previous token
    ctx_out[1] = tok;
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
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    if (j < total) {
      const __half_raw hr = __nv_cvt_fp8_to_halfraw(b[u], __NV_E4M3);
      const float f = __half2float(*reinterpret_cast<const __half*>(&hr));
      out[j] = __float2bfloat16_rn(f * P->scale);
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
    int qi = __float2int_rn(q);
    if (qi < -127) qi = -127;
    if (qi > 127) qi = 127;
    out_byte = static_cast<uint8_t>(static_cast<int8_t>(qi));
  } else {
    float qc = fminf(fmaxf(q, -448.0f), 448.0f);
    out_byte = static_cast<uint8_t>(__nv_cvt_float_to_fp8(qc, __NV_SATFINITE, __NV_E4M3));
  }
  dst[(size_t)r * dst_pitch_elems + c] = out_byte;
}

__global__ void k_embed_rows(const __nv_bfloat16* table, const i64* ids, __nv_bfloat16* out, int rows, int width) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const i64 tok = ids[r];
  const uint4* src = reinterpret_cast<const uint4*>(table + (size_t)tok * width);
  uint4* dst = reinterpret_cast<uint4*>(out + (size_t)r * width);
  const int n8 = width / 8;
  for (int i = threadIdx.x; i < n8; i += blockDim.x) dst[i] = src[i];
}

#ifdef _WIN32
#define QWEN38_PLANE_EXPORT __declspec(dllexport)
#else
#define QWEN38_PLANE_EXPORT
#endif

extern "C" {
QWEN38_PLANE_EXPORT int qwen38_plane_abi_version(void) { return QWEN38_PLANE_ABI_VERSION; }
QWEN38_PLANE_EXPORT int qwen38_plane_sizeof_ple_params(void) { return (int)sizeof(PleParams); }

QWEN38_PLANE_EXPORT void qwen38_plane_release_ple_workspace(void) {
  std::lock_guard<std::mutex> lock(g_staged_ple_mu);
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
  g_ever_had_staged_ple = false;
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
    g_ever_had_staged_ple = true;
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

QWEN38_PLANE_EXPORT void qwen38_plane_prep(void* s, const void* P, const void* tok_ptr, const void* pos_ptr, int pos_add, const void* ctx_in,
                       void* ctx_out, void* input_ids, void* position_ids, void* committed, void* past_pos,
                       void* ple_out, void* state) {
  {
    std::lock_guard<std::mutex> lock(g_staged_ple_mu);
    if (!g_host_ple_workspaces.empty() || g_ever_had_staged_ple) {
      cudaStream_t stream = (cudaStream_t)s;
      const bool allow_sync = !(pos_add >= 1 && pos_add <= 3 && state != nullptr);
      bool unregistered_staged = false;
      StagedPleWorkspace* ws = resolve_staged_ple_workspace(P, stream, allow_sync, &unregistered_staged);
      if (unregistered_staged) {
        return;
      }
      if (ws != nullptr) {
        ensure_prep_workspace(*ws);
        const i64* state_i64 = (const i64*)state;
        i64 tok = 0, pos = 0, s_c0 = 0, s_c1 = 0;
        const i64* expected_fold_tok_ptr =
            (pos_add == 1) ? (state_i64 + 2)
                           : ((pos_add == 2) ? (state_i64 + 8) : ((pos_add == 3) ? (state_i64 + 9) : nullptr));
        if (pos_add >= 1 && pos_add <= 3 && state != nullptr &&
            state == ws->cached_state_ptr &&
            pos_ptr == (state_i64 + 1) &&
            tok_ptr == expected_fold_tok_ptr &&
            ctx_in == ws->cached_ctx_out_ptr) {
          // Fast path for folded multi-draft rows 1..3: reuse the 10-word `ctl` snapshot copied on row 0
          // with zero additional D2H copies and zero cudaStreamSynchronize calls!
          const int ctl_idx = (pos_add == 1) ? 2 : ((pos_add == 2) ? 8 : 9);
          tok = ws->scalars[ctl_idx];
          pos = ws->scalars[1] + pos_add;
          s_c0 = ws->cached_c0;
          s_c1 = ws->cached_c1;
          ws->cached_ctx_out_ptr = ctx_out;
          ws->cached_c0 = s_c1;
          ws->cached_c1 = tok;
        } else if (pos_add == 0 && state != nullptr && tok_ptr == state_i64 && pos_ptr == (state_i64 + 1)) {
          // Row 0 of EnqueueFold / EnqueueVerify / EnqueueStep: copy all 10 words of `ctl` (`ctl[0..9]`)
          // plus `ctx_in[0..1]` in one batch so subsequent draft rows 1..3 need no stream synchronization.
          cudaMemcpyAsync(&ws->scalars[0], state_i64, 10 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
          cudaMemcpyAsync(&ws->scalars[10], ctx_in, 2 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
          cudaStreamSynchronize(stream);
          tok = ws->scalars[0];
          pos = ws->scalars[1];
          s_c0 = ws->scalars[10];
          s_c1 = ws->scalars[11];
          ws->cached_state_ptr = state;
          ws->cached_ctx_out_ptr = ctx_out;
          ws->cached_c0 = s_c1;
          ws->cached_c1 = tok;
        } else {
          ws->cached_state_ptr = nullptr;
          ws->cached_ctx_out_ptr = nullptr;
          cudaMemcpyAsync(&ws->scalars[0], tok_ptr, sizeof(i64), cudaMemcpyDeviceToHost, stream);
          cudaMemcpyAsync(&ws->scalars[1], pos_ptr, sizeof(i64), cudaMemcpyDeviceToHost, stream);
          cudaMemcpyAsync(&ws->scalars[2], ctx_in, 2 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
          cudaStreamSynchronize(stream);
          tok = ws->scalars[0];
          pos = ws->scalars[1] + pos_add;
          s_c0 = ws->scalars[2];
          s_c1 = ws->scalars[3];
        }
        const size_t slot_offset = (size_t)(ws->ring_idx++ & 63u) * 8192u;
        unsigned char* slot_host = ws->prep_ring + slot_offset;
        const unsigned char* slot_dev = (ws->prep_ring_dev ? ws->prep_ring_dev : ws->prep_ring) + slot_offset;
        host_gather_ple_row(*ws, tok, s_c0, s_c1, slot_host);
        k_prep_staged<<<1, 1024, 0, stream>>>(ws->params.capacity, ws->params.n_heads, ws->params.head_dim,
                                              ws->params.scale, tok, pos, s_c1, slot_dev, (i64*)ctx_out,
                                              (i64*)input_ids, (i64*)position_ids, (int*)committed, (i64*)past_pos,
                                              (__nv_bfloat16*)ple_out, (i64*)state);
        return;
      }
    }
  }
  k_prep<<<1, 1024, 0, (cudaStream_t)s>>>((const PleParams*)P, (const i64*)tok_ptr, (const i64*)pos_ptr, pos_add,
                                          (const i64*)ctx_in, (i64*)ctx_out, (i64*)input_ids, (i64*)position_ids,
                                          (int*)committed, (i64*)past_pos, (__nv_bfloat16*)ple_out, (i64*)state);
}

QWEN38_PLANE_EXPORT void qwen38_plane_argmax_part(void* s, const void* logits, int n, void* outv, void* outi, int nblocks) {
  k_argmax_part<<<nblocks, 256, 0, (cudaStream_t)s>>>((const __nv_bfloat16*)logits, n, (float*)outv, (int*)outi);
}

QWEN38_PLANE_EXPORT void qwen38_plane_post1(void* s, const void* v0, const void* i0, int nb, void* state, void* ctx_commit, const void* ctx_a,
                        const void* st0, int nst, void* result, int ring) {
  k_post1<<<1, 32, 0, (cudaStream_t)s>>>((const float*)v0, (const int*)i0, nb, (i64*)state, (i64*)ctx_commit,
                                         (const i64*)ctx_a, (const int*)st0, nst, (i64*)result, ring);
}

QWEN38_PLANE_EXPORT void qwen38_plane_post_tok(void* s, const void* tok01, void* state, void* ctx_commit, const void* ctx_a, const void* ctx_b,
                           const void* st0, int nst, void* result, int ring) {
  k_post_tok<<<1, 32, 0, (cudaStream_t)s>>>((const i64*)tok01, (i64*)state, (i64*)ctx_commit, (const i64*)ctx_a,
                                            (const i64*)ctx_b, (const int*)st0, nst, (i64*)result, ring);
}

QWEN38_PLANE_EXPORT void qwen38_plane_head_prep(void* s, const void* ctl, int rows, int hlen, int hidden_width, const void* src, void* dst,
                            void* tokens, void* positions, void* committed) {
  k_head_prep<<<1, 1024, 0, (cudaStream_t)s>>>((const i64*)ctl, rows, hlen, hidden_width, (const uint4*)src, (uint4*)dst,
                                               (i64*)tokens, (i64*)positions, (int*)committed);
}

QWEN38_PLANE_EXPORT void qwen38_plane_head_post(void* s, const void* next_token, void* ctl) {
  k_head_post<<<1, 32, 0, (cudaStream_t)s>>>((const i64*)next_token, (i64*)ctl);
}

QWEN38_PLANE_EXPORT void qwen38_plane_or_status(void* s, const void* st, int nst, void* state) {
  k_or_status<<<1, 32, 0, (cudaStream_t)s>>>((const int*)st, nst, (i64*)state);
}

/* ---- fast prefill kernels: batched PLE rows, dense -> packed state hand-off ---- */
QWEN38_PLANE_EXPORT void qwen38_plane_ple_rows(void* s, const void* P, const void* tokens, int n, const void* ctx_in, void* ctx_out,
                           void* ple_out) {
  if (n <= 0) return;
  {
    std::lock_guard<std::mutex> lock(g_staged_ple_mu);
    if (!g_host_ple_workspaces.empty() || g_ever_had_staged_ple) {
      cudaStream_t stream = (cudaStream_t)s;
      bool unregistered_staged = false;
      StagedPleWorkspace* ws = resolve_staged_ple_workspace(P, stream, /*allow_sync=*/true, &unregistered_staged);
      if (unregistered_staged) {
        return;
      }
      if (ws != nullptr) {
        const size_t total = (size_t)ws->params.n_heads * (size_t)ws->params.head_dim;
        ensure_prefill_workspace(*ws, (size_t)n, total);
        cudaMemcpyAsync(ws->prefill_tokens, tokens, (size_t)n * sizeof(i64), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(&ws->scalars[2], ctx_in, 2 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        const i64* h_toks = ws->prefill_tokens;
        const i64 ctx0 = ws->scalars[2];
        const i64 ctx1 = ws->scalars[3];
        for (int i = 0; i < n; ++i) {
          const i64 tok = h_toks[i];
          const i64 c1 = i >= 1 ? h_toks[i - 1] : ctx1;
          const i64 c0raw = i >= 2 ? h_toks[i - 2] : (i == 1 ? ctx1 : ctx0);
          host_gather_ple_row(*ws, tok, c0raw, c1, ws->prefill_fp8 + (size_t)i * total);
        }
        const i64 last_tok = h_toks[n - 1];
        const i64 last_c1 = n >= 2 ? h_toks[n - 2] : ctx1;
        const unsigned char* fp8_dev = ws->prefill_fp8_dev ? ws->prefill_fp8_dev : ws->prefill_fp8;
        k_ple_rows_staged<<<n, 1024, 0, stream>>>(ws->params.n_heads, ws->params.head_dim, ws->params.scale, fp8_dev, n,
                                                  last_c1, last_tok, (i64*)ctx_out, (__nv_bfloat16*)ple_out);
        return;
      }
    }
  }
  k_ple_rows<<<n, 1024, 0, (cudaStream_t)s>>>((const PleParams*)P, (const i64*)tokens, n, (const i64*)ctx_in, (i64*)ctx_out,
                                              (__nv_bfloat16*)ple_out);
}

QWEN38_PLANE_EXPORT void qwen38_plane_pool_keys(void* s, const void* raw, int nb, const void* knorm, const void* cosv, const void* sinv,
                            int width, float eps, int first_block, void* out) {
  if (nb <= 0) return;
  k_pool_keys<<<nb, 128, 0, (cudaStream_t)s>>>((const __nv_bfloat16*)raw, (const __nv_bfloat16*)knorm,
                                               (const __nv_bfloat16*)cosv, (const __nv_bfloat16*)sinv, width, eps,
                                               first_block, (float*)out);
}

QWEN38_PLANE_EXPORT void qwen38_plane_transpose_f32(void* s, const void* src, void* dst, int n, int batch) {
  if (n <= 0 || batch <= 0) return;
  const unsigned int blocks = static_cast<unsigned int>((n + 31) / 32);
  dim3 grid(blocks, blocks, static_cast<unsigned int>(batch)), block(32, 8);
  k_transpose_f32<<<grid, block, 0, (cudaStream_t)s>>>((const float*)src, (float*)dst, n);
}

QWEN38_PLANE_EXPORT void qwen38_plane_quantize_kv(void* s, const void* src, void* dst, const void* scale, int rows, int src_pitch_elems,
                              int dst_pitch_elems, int mode) {
  if (rows <= 0) return;
  k_quantize_kv<<<rows, 256, 0, (cudaStream_t)s>>>((const __nv_bfloat16*)src, (uint8_t*)dst, (const float*)scale, rows,
                                                   src_pitch_elems, dst_pitch_elems, mode);
}

QWEN38_PLANE_EXPORT void qwen38_plane_embed_rows(void* s, const void* table, const void* ids, void* out, int rows, int width) {
  if (rows <= 0 || width <= 0) return;
  k_embed_rows<<<rows, 256, 0, (cudaStream_t)s>>>((const __nv_bfloat16*)table, (const i64*)ids, (__nv_bfloat16*)out, rows, width);
}

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
  return (int)err;
}
QWEN38_PLANE_EXPORT int qwen38_plane_host_unregister(void* p) { return (int)cudaHostUnregister(p); }
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
QWEN38_PLANE_EXPORT int qwen38_plane_stream_destroy(void* s) { return (int)cudaStreamDestroy((cudaStream_t)s); }
QWEN38_PLANE_EXPORT int qwen38_plane_stream_sync(void* s) { return (int)cudaStreamSynchronize((cudaStream_t)s); }
}
