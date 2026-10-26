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
#include <string.h>

#include "qwen38_plane_abi.h"

typedef long long i64;
typedef Qwen38PleParams PleParams;

static PleParams g_host_ple_params{};
static bool g_has_host_ple_params = false;

struct StagedPleWorkspace {
  i64* scalars = nullptr;            // pinned [64]: tok/ctl words + ctx_in
  unsigned char* prep_ring = nullptr; // pinned mapped [64 * 8192]
  unsigned int ring_idx = 0;
  i64* prefill_tokens = nullptr;     // pinned [cap_rows]
  unsigned char* prefill_fp8 = nullptr; // pinned mapped [cap_rows * max_total]
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
};

static StagedPleWorkspace g_staged_ple{};

static inline void ensure_prep_workspace() {
  if (!g_staged_ple.scalars) {
    cudaHostAlloc((void**)&g_staged_ple.scalars, 256 * sizeof(i64), cudaHostAllocMapped | cudaHostAllocPortable);
  }
  if (!g_staged_ple.prep_ring) {
    cudaHostAlloc((void**)&g_staged_ple.prep_ring, 64 * 8192, cudaHostAllocMapped | cudaHostAllocPortable);
  }
  if (!g_staged_ple.ngram_cache_data) {
    g_staged_ple.ngram_cache_data =
        (unsigned char*)malloc(StagedPleWorkspace::kCacheSlots * StagedPleWorkspace::kMaxRowBytes);
    memset(g_staged_ple.ngram_cache_valid, 0, sizeof(g_staged_ple.ngram_cache_valid));
  }
}

static inline void ensure_prefill_workspace(size_t rows, size_t row_bytes) {
  ensure_prep_workspace();
  if (rows > g_staged_ple.prefill_cap_rows) {
    if (g_staged_ple.prefill_tokens) cudaFreeHost(g_staged_ple.prefill_tokens);
    size_t new_rows = rows < 4096 ? 4096 : rows;
    cudaHostAlloc((void**)&g_staged_ple.prefill_tokens, new_rows * sizeof(i64), cudaHostAllocPortable);
    g_staged_ple.prefill_cap_rows = new_rows;
  }
  size_t need_bytes = rows * row_bytes;
  if (need_bytes > g_staged_ple.prefill_cap_bytes) {
    if (g_staged_ple.prefill_fp8) cudaFreeHost(g_staged_ple.prefill_fp8);
    size_t new_bytes = need_bytes < (4096ULL * 4096ULL) ? (4096ULL * 4096ULL) : need_bytes;
    cudaHostAlloc((void**)&g_staged_ple.prefill_fp8, new_bytes, cudaHostAllocMapped | cudaHostAllocPortable);
    g_staged_ple.prefill_cap_bytes = new_bytes;
  }
}

static inline void host_gather_ple_row(const PleParams& HP, i64 tok, i64 s_c0, i64 s_c1, unsigned char* dst_fp8) {
  const i64 c1 = s_c1;
  const i64 c0 = (c1 == HP.eos) ? HP.eos : s_c0;
  const int hd = HP.head_dim;
  const size_t total_bytes = (size_t)HP.n_heads * (size_t)hd;
  size_t cache_slot = 0;
  const bool use_cache = (g_staged_ple.ngram_cache_data != nullptr && total_bytes <= StagedPleWorkspace::kMaxRowBytes);
  if (use_cache) {
    const unsigned long long h =
        ((unsigned long long)tok * 11400714819323198485ULL) ^
        ((unsigned long long)c1 * 14029467366897019727ULL) ^
        ((unsigned long long)c0 * 1609587929392839161ULL);
    cache_slot = (size_t)(h & (StagedPleWorkspace::kCacheSlots - 1));
    if (g_staged_ple.ngram_cache_valid[cache_slot] &&
        g_staged_ple.ngram_cache_keys[cache_slot][0] == tok &&
        g_staged_ple.ngram_cache_keys[cache_slot][1] == c0 &&
        g_staged_ple.ngram_cache_keys[cache_slot][2] == c1) {
      memcpy(dst_fp8, g_staged_ple.ngram_cache_data + cache_slot * StagedPleWorkspace::kMaxRowBytes, total_bytes);
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
    memcpy(g_staged_ple.ngram_cache_data + cache_slot * StagedPleWorkspace::kMaxRowBytes, dst_fp8, total_bytes);
    g_staged_ple.ngram_cache_keys[cache_slot][0] = tok;
    g_staged_ple.ngram_cache_keys[cache_slot][1] = c0;
    g_staged_ple.ngram_cache_keys[cache_slot][2] = c1;
    g_staged_ple.ngram_cache_valid[cache_slot] = 1;
  }
}

__global__ void k_prep_staged(const PleParams* P, i64 tok, i64 pos, i64 s_c1, const unsigned char* fp8_row,
                              i64* ctx_out, i64* input_ids, i64* position_ids, int* committed, i64* past_pos,
                              __nv_bfloat16* ple_out, i64* state) {
  const int tid = threadIdx.x;
  if (tid == 0) {
    input_ids[0] = tok;
    position_ids[0] = pos;
    if (committed != nullptr) committed[0] = (int)pos;
    if (pos >= 0 && pos < P->capacity)
      past_pos[pos] = pos;
    else
      state[5] |= 1;
  }
  const int hd = P->head_dim;
  const int total = P->n_heads * hd;
  unsigned char b[3];
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    b[u] = (j < total) ? fp8_row[j] : 0;
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
    ctx_out[0] = s_c1;
    ctx_out[1] = tok;
  }
}

__global__ void k_prep_fold_staged(const PleParams* P, const i64* row_toks, i64 base_pos, const i64* row_c1s,
                                   const unsigned char* fp8_rows, i64* ctx_out, i64* input_ids, i64* position_ids,
                                   int* committed, i64* past_pos, __nv_bfloat16* ple_out, i64* state) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const i64 tok = row_toks[row];
  const i64 pos = base_pos + row;
  const i64 s_c1 = row_c1s[row];
  if (tid == 0) {
    input_ids[row] = tok;
    position_ids[row] = pos;
    if (row == 0 && committed != nullptr) committed[0] = (int)pos;
    if (pos >= 0 && pos < P->capacity)
      past_pos[pos] = pos;
    else
      atomicOr((unsigned long long*)&state[5], 1ULL);
    ctx_out[2 * row + 0] = s_c1;
    ctx_out[2 * row + 1] = tok;
  }
  const int hd = P->head_dim;
  const int total = P->n_heads * hd;
  const unsigned char* fp8_row = fp8_rows + (size_t)row * 8192u;
  __nv_bfloat16* out_row = ple_out + (size_t)row * total;
  unsigned char b[3];
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    b[u] = (j < total) ? fp8_row[j] : 0;
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
      out_row[j] = __float2bfloat16_rn(f * abs_scale);
    }
  }
}

__global__ void k_ple_rows_staged(const PleParams* P, const unsigned char* fp8_rows, int n, i64 last_c1, i64 last_tok,
                                  i64* ctx_out, __nv_bfloat16* ple_out) {
  const int tid = threadIdx.x;
  const int i = blockIdx.x;
  const int hd = P->head_dim;
  const int total = P->n_heads * hd;
  const unsigned char* src = fp8_rows + (size_t)i * total;
  __nv_bfloat16* out = ple_out + (size_t)i * total;
  unsigned char b[3];
#pragma unroll
  for (int u = 0; u < 3; ++u) {
    const int j = tid + u * blockDim.x;
    b[u] = (j < total) ? src[j] : 0;
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

__global__ void k_prep_fold(const PleParams* P, const i64* ctx_in, i64* ctx_a, i64* input_ids, i64* position_ids,
                            int* committed, i64* past_pos, __nv_bfloat16* ple_out, i64* state) {
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
  int* committed_b = committed + b;
  i64* past_pos_b = past_pos + (size_t)b * P->capacity;
  const int hd = P->head_dim;
  const int total = P->n_heads * hd;
  __nv_bfloat16* row_ple = ple_out + ((size_t)b * rows + row) * total;

  if (tid == 0) {
    s_muted = state_b[3];
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
      if (row == 0) committed_b[0] = 0;
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
    if (row == 0) committed_b[0] = (int)pos;
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
    tile[threadIdx.y + j][threadIdx.x] = src[base + (size_t)y * n + x];
  }
  __syncthreads();
  const int tx = blockIdx.y * 32 + threadIdx.x;
  for (int j = 0; j < 32; j += 8) {
    const int ty = blockIdx.x * 32 + threadIdx.y + j;
    dst[base + (size_t)ty * n + tx] = tile[threadIdx.x][threadIdx.y + j];
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
  const i64 raw_tok = ids[r];
  const i64 tok = (raw_tok >= 0 && raw_tok < 248320) ? raw_tok : 0;
  const uint4* src = reinterpret_cast<const uint4*>(table + (size_t)tok * width);
  uint4* dst = reinterpret_cast<uint4*>(out + (size_t)r * width);
  const int n8 = width / 8;
  for (int i = threadIdx.x; i < n8; i += blockDim.x) dst[i] = src[i];
}

__global__ void k_embed_rows_int8(const int8_t* table, const float* row_scales, const i64* ids, __nv_bfloat16* out,
                                  int rows, int width) {
  const int r = blockIdx.x;
  if (r >= rows) return;
  const i64 raw_tok = ids[r];
  const i64 tok = (raw_tok >= 0 && raw_tok < 248320) ? raw_tok : 0;
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

extern "C" {
QWEN38_PLANE_EXPORT int qwen38_plane_abi_version(void) { return QWEN38_PLANE_ABI_VERSION; }
QWEN38_PLANE_EXPORT int qwen38_plane_sizeof_ple_params(void) { return (int)sizeof(PleParams); }

QWEN38_PLANE_EXPORT void qwen38_plane_prep(void* s, const void* P, const void* tok_ptr, const void* pos_ptr, int pos_add, const void* ctx_in,
                       void* ctx_out, void* input_ids, void* position_ids, void* committed, void* past_pos,
                       void* ple_out, void* state) {
  if (pos_add <= -2) {
    const int raw = -pos_add;
    const int rows = raw & 0xFF;
    const int batch = (raw >> 8) > 0 ? (raw >> 8) : 1;
    if (rows >= 2 && rows <= 8) {
      if (g_has_host_ple_params && g_host_ple_params.pad_ == 1) {
        constexpr int kSlots[8] = {0, 2, 8, 9, 10, 11, 12, 13};
        const int ple_total = g_host_ple_params.n_heads * g_host_ple_params.head_dim;
        for (int b = 0; b < batch; ++b) {
          i64* state_b = (i64*)state + (size_t)b * 16;
          const i64* ctx_in_b = (const i64*)ctx_in + (size_t)b * 2;
          i64* ctx_out_b = (i64*)ctx_out + (size_t)b * 2 * rows;
          i64* input_ids_b = (i64*)input_ids + (size_t)b * rows;
          i64* position_ids_b = (i64*)position_ids + (size_t)b * rows;
          int* committed_b = (int*)committed + b;
          i64* past_pos_b = (i64*)past_pos + (size_t)b * g_host_ple_params.capacity;
          __nv_bfloat16* ple_out_b = (__nv_bfloat16*)ple_out + (size_t)b * rows * ple_total;
          if (batch == 1) {
            ensure_prep_workspace();
            cudaStream_t stream = (cudaStream_t)s;
            cudaMemcpyAsync(&g_staged_ple.scalars[0], state_b, 14 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&g_staged_ple.scalars[14], ctx_in_b, 2 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
            const i64 base_pos = g_staged_ple.scalars[1];
            i64 cur_c0 = g_staged_ple.scalars[14];
            i64 cur_c1 = g_staged_ple.scalars[15];
            const unsigned int base_ring = (g_staged_ple.ring_idx * 8u) & 56u;
            g_staged_ple.ring_idx = (g_staged_ple.ring_idx + 1u) & 7u;
            unsigned char* fp8_base = g_staged_ple.prep_ring + base_ring * 8192u;
            i64* d_toks = g_staged_ple.scalars + 16 + base_ring * 2;
            i64* d_c1s = d_toks + 8;
            for (int r = 0; r < rows; ++r) {
              const i64 tok = g_staged_ple.scalars[kSlots[r]];
              d_toks[r] = tok;
              d_c1s[r] = cur_c1;
              host_gather_ple_row(g_host_ple_params, tok, cur_c0, cur_c1, fp8_base + (size_t)r * 8192u);
              cur_c0 = cur_c1;
              cur_c1 = tok;
            }
            k_prep_fold_staged<<<rows, 1024, 0, stream>>>((const PleParams*)P, d_toks, base_pos, d_c1s, fp8_base,
                                                          ctx_out_b, input_ids_b, position_ids_b,
                                                          committed_b, past_pos_b, ple_out_b, state_b);
          } else {
            for (int r = 0; r < rows; ++r) {
              const void* r_ctx_in = (r == 0) ? ctx_in_b : (ctx_out_b + 2 * (r - 1));
              void* r_ctx_out = ctx_out_b + 2 * r;
              qwen38_plane_prep(s, P, state_b + kSlots[r], state_b + 1, r, r_ctx_in, r_ctx_out,
                                input_ids_b + r, position_ids_b + r, r == 0 ? committed_b : nullptr,
                                past_pos_b, ple_out_b + (size_t)r * ple_total, state_b);
            }
          }
        }
        return;
      }
      k_prep_fold<<<dim3(rows, batch), 1024, 0, (cudaStream_t)s>>>((const PleParams*)P, (const i64*)ctx_in, (i64*)ctx_out,
                                                                   (i64*)input_ids, (i64*)position_ids, (int*)committed,
                                                                   (i64*)past_pos, (__nv_bfloat16*)ple_out, (i64*)state);
      return;
    }
  }
  if (g_has_host_ple_params && g_host_ple_params.pad_ == 1) {
    ensure_prep_workspace();
    cudaStream_t stream = (cudaStream_t)s;
    const i64* state_i64 = (const i64*)state;
    i64 tok = 0, pos = 0, s_c0 = 0, s_c1 = 0;
    const i64* expected_fold_tok_ptr =
        (pos_add == 1) ? (state_i64 + 2) : ((pos_add >= 2 && pos_add <= 7) ? (state_i64 + 6 + pos_add) : nullptr);
    if (pos_add >= 1 && pos_add <= 7 && state != nullptr &&
        state == g_staged_ple.cached_state_ptr &&
        pos_ptr == (state_i64 + 1) &&
        tok_ptr == expected_fold_tok_ptr &&
        ctx_in == g_staged_ple.cached_ctx_out_ptr) {
      // Fast path for folded multi-draft rows 1..7: reuse the 14-word `ctl` snapshot copied on row 0
      // with zero additional D2H copies and zero cudaStreamSynchronize calls!
      const int ctl_idx = (pos_add == 1) ? 2 : (6 + pos_add);
      tok = g_staged_ple.scalars[ctl_idx];
      pos = g_staged_ple.scalars[1] + pos_add;
      s_c0 = g_staged_ple.cached_c0;
      s_c1 = g_staged_ple.cached_c1;
      g_staged_ple.cached_ctx_out_ptr = ctx_out;
      g_staged_ple.cached_c0 = s_c1;
      g_staged_ple.cached_c1 = tok;
    } else if (pos_add == 0 && state != nullptr && tok_ptr == state_i64 && pos_ptr == (state_i64 + 1)) {
      // Row 0 of EnqueueFold / EnqueueVerify / EnqueueStep: copy all 14 words of `ctl` (`ctl[0..13]`)
      // plus `ctx_in[0..1]` in one batch so subsequent draft rows 1..7 need no stream synchronization.
      cudaMemcpyAsync(&g_staged_ple.scalars[0], state_i64, 14 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(&g_staged_ple.scalars[14], ctx_in, 2 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      tok = g_staged_ple.scalars[0];
      pos = g_staged_ple.scalars[1];
      s_c0 = g_staged_ple.scalars[14];
      s_c1 = g_staged_ple.scalars[15];
      g_staged_ple.cached_state_ptr = state;
      g_staged_ple.cached_ctx_out_ptr = ctx_out;
      g_staged_ple.cached_c0 = s_c1;
      g_staged_ple.cached_c1 = tok;
    } else {
      g_staged_ple.cached_state_ptr = nullptr;
      g_staged_ple.cached_ctx_out_ptr = nullptr;
      cudaMemcpyAsync(&g_staged_ple.scalars[0], tok_ptr, sizeof(i64), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(&g_staged_ple.scalars[1], pos_ptr, sizeof(i64), cudaMemcpyDeviceToHost, stream);
      cudaMemcpyAsync(&g_staged_ple.scalars[2], ctx_in, 2 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      tok = g_staged_ple.scalars[0];
      pos = g_staged_ple.scalars[1] + pos_add;
      s_c0 = g_staged_ple.scalars[2];
      s_c1 = g_staged_ple.scalars[3];
    }
    unsigned char* slot = g_staged_ple.prep_ring + ((g_staged_ple.ring_idx++ & 63u) * 8192u);
    host_gather_ple_row(g_host_ple_params, tok, s_c0, s_c1, slot);
    k_prep_staged<<<1, 1024, 0, stream>>>((const PleParams*)P, tok, pos, s_c1, slot, (i64*)ctx_out, (i64*)input_ids,
                                          (i64*)position_ids, (int*)committed, (i64*)past_pos,
                                          (__nv_bfloat16*)ple_out, (i64*)state);
    return;
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
  if (g_has_host_ple_params && g_host_ple_params.pad_ == 1) {
    const size_t total = (size_t)g_host_ple_params.n_heads * (size_t)g_host_ple_params.head_dim;
    ensure_prefill_workspace((size_t)n, total);
    cudaStream_t stream = (cudaStream_t)s;
    cudaMemcpyAsync(g_staged_ple.prefill_tokens, tokens, (size_t)n * sizeof(i64), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(&g_staged_ple.scalars[2], ctx_in, 2 * sizeof(i64), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const i64* h_toks = g_staged_ple.prefill_tokens;
    const i64 ctx0 = g_staged_ple.scalars[2];
    const i64 ctx1 = g_staged_ple.scalars[3];
    for (int i = 0; i < n; ++i) {
      const i64 tok = h_toks[i];
      const i64 c1 = i >= 1 ? h_toks[i - 1] : ctx1;
      const i64 c0raw = i >= 2 ? h_toks[i - 2] : (i == 1 ? ctx1 : ctx0);
      host_gather_ple_row(g_host_ple_params, tok, c0raw, c1, g_staged_ple.prefill_fp8 + (size_t)i * total);
    }
    const i64 last_tok = h_toks[n - 1];
    const i64 last_c1 = n >= 2 ? h_toks[n - 2] : ctx1;
    k_ple_rows_staged<<<n, 1024, 0, stream>>>((const PleParams*)P, g_staged_ple.prefill_fp8, n, last_c1, last_tok,
                                              (i64*)ctx_out, (__nv_bfloat16*)ple_out);
    return;
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
  dim3 grid(n / 32, n / 32, batch), block(32, 8);
  k_transpose_f32<<<grid, block, 0, (cudaStream_t)s>>>((const float*)src, (float*)dst, n);
}

QWEN38_PLANE_EXPORT void qwen38_plane_quantize_kv(void* s, const void* src, void* dst, const void* scale, int rows, int src_pitch_elems,
                              int dst_pitch_elems, int mode) {
  if (rows <= 0) return;
  k_quantize_kv<<<rows, 256, 0, (cudaStream_t)s>>>((const __nv_bfloat16*)src, (uint8_t*)dst, (const float*)scale, rows,
                                                   src_pitch_elems, dst_pitch_elems, mode);
}

QWEN38_PLANE_EXPORT void qwen38_plane_embed_rows(void* s, const void* table, const void* ids, void* out, int rows, int width) {
  if (rows <= 0 || width == 0) return;
  if (width < 0) {
    const int actual_width = -width;
    const int8_t* table_i8 = static_cast<const int8_t*>(table);
    const float* scales_f32 = reinterpret_cast<const float*>(table_i8 + 248320ULL * static_cast<size_t>(actual_width));
    k_embed_rows_int8<<<rows, 256, 0, (cudaStream_t)s>>>(table_i8, scales_f32, (const i64*)ids, (__nv_bfloat16*)out, rows, actual_width);
    return;
  }
  k_embed_rows<<<rows, 256, 0, (cudaStream_t)s>>>((const __nv_bfloat16*)table, (const i64*)ids, (__nv_bfloat16*)out, rows, width);
}

/* ---- CUDA runtime helpers (all return cudaError_t as int, 0 = success) ---- */
QWEN38_PLANE_EXPORT int qwen38_plane_set_device(int device) { return (int)cudaSetDevice(device); }
QWEN38_PLANE_EXPORT const char* qwen38_plane_error_string(int err) { return cudaGetErrorString((cudaError_t)err); }
QWEN38_PLANE_EXPORT int qwen38_plane_device_alloc(void** p, size_t bytes) { return (int)cudaMalloc(p, bytes); }
QWEN38_PLANE_EXPORT int qwen38_plane_device_free(void* p) { return (int)cudaFree(p); }
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
QWEN38_PLANE_EXPORT int qwen38_plane_host_get_device_pointer(void** dev_ptr, void* host_ptr, unsigned flags) {
  cudaError_t err = cudaHostGetDevicePointer(dev_ptr, host_ptr, flags);
  if (err != cudaSuccess) cudaGetLastError();
  return (int)err;
}
QWEN38_PLANE_EXPORT int qwen38_plane_device_memset(void* p, int value, size_t bytes, void* s) {
  return (int)cudaMemsetAsync(p, value, bytes, (cudaStream_t)s);
}
QWEN38_PLANE_EXPORT int qwen38_plane_memcpy_h2d(void* dst, const void* src, size_t bytes, void* s) {
  if (bytes == sizeof(PleParams) && src != nullptr) {
    const PleParams* cand = (const PleParams*)src;
    if (cand->n_heads > 0 && cand->n_heads <= 32 && cand->head_dim > 0 && cand->n_shards > 0 && cand->n_shards <= 256) {
      g_host_ple_params = *cand;
      g_has_host_ple_params = true;
    }
  }
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
  DflashSelect<<<1, 1, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const int32_t*>(candidates), static_cast<const float*>(scores),
      static_cast<i64*>(control), drafts, top_k);
  return static_cast<int>(cudaGetLastError());
}

namespace {

constexpr int kPlaneReplayThreads = 256;
constexpr int kPlaneReplayRows = 32;
constexpr int kPlaneMaxReplayTransitions = 8;
constexpr int kPlaneMaxReplayKeyWidth = 256;
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
  const int chunks = (value_width + kPlaneReplayRows - 1) / kPlaneReplayRows;
  const int head = blockIdx.x / chunks;
  if (head >= heads) return;
  const int first_row = (blockIdx.x - head * chunks) * kPlaneReplayRows;
  const int key_head = head * key_heads / heads;

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
        s.x = __fadd_rn(__fmul_rn(s.x, decay), __fmul_rn(key[0], delta));
        s.y = __fadd_rn(__fmul_rn(s.y, decay), __fmul_rn(key[1], delta));
        s.z = __fadd_rn(__fmul_rn(s.z, decay), __fmul_rn(key[2], delta));
        s.w = __fadd_rn(__fmul_rn(s.w, decay), __fmul_rn(key[3], delta));
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
    if (kept_override == 0) {
      if (blockIdx.x == 0 && threadIdx.x == 0) {
        const_cast<int32_t*>(descriptor.source_lengths)[0] = 0;
        const_cast<int32_t*>(descriptor.source_lengths)[1] = 0;
        descriptor.destination_lengths[0] = 0;
        descriptor.destination_lengths[1] = 0;
      }
      return;
    }
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
  if (!descs || kept_count < 0 || kept_count > kPlaneMaxReplayTransitions) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto* typed = static_cast<const Qwen38StateReplayDesc*>(descs);
  cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);
  if (kept_count == 0) {
    if (generic_count > 0) {
      const dim3 grid(1, static_cast<unsigned>(generic_count));
      k_replay_generic<<<grid, 32, 0, cu_stream>>>(typed + fast_count, 0);
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) return static_cast<int>(err);
    }
    return 0;
  }
  if (fast_count > 0 && fast_blocks_per_descriptor > 0) {
    const dim3 grid(static_cast<unsigned>(fast_blocks_per_descriptor), static_cast<unsigned>(fast_count));
    k_replay_gdn<<<grid, kPlaneReplayThreads, 0, cu_stream>>>(typed, kept_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int>(err);
  }
  if (generic_count > 0) {
    const dim3 grid(kPlaneGenericBlocksPerDesc, static_cast<unsigned>(generic_count));
    k_replay_generic<<<grid, kPlaneReplayThreads, 0, cu_stream>>>(typed + fast_count, kept_count);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return static_cast<int>(err);
  }
  return 0;
}
