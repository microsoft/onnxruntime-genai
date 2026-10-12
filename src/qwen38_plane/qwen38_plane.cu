// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
/* Qwen3.8-Flash CUDA data plane for GenAI (model.type "qwen38_flash_graph"). The kernels are the research data plane of
 * CG/mtp_runner/dataplane/mtp_plane.cu (unchanged arithmetic and launch shapes); the qg_* functions at the end are the CUDA
 * runtime helpers the GenAI model code needs (allocation, host registration, copies, sync) so that the core GenAI library
 * does not link cudart.
 *
 * Device data plane for the MTP(1) loop (WP-E).  All kernels take device pointers only (no host scalars that
 * change per cycle), so every sequence below can be captured into a CUDA graph.
 *
 *  k_prep         : write graph-row inputs for one token: input_ids, position_ids, committed_length, the in-place
 *                   past_position_ids slot, the PLE embedding row (n-gram hash + gather from the host-registered FP8
 *                   table + dequantisation; bit-identical to Qwen4ExpCPUPLELookup) and the next PLE context.
 *  k_argmax_part  : stage 1 of a deterministic (smallest index on ties) argmax over bf16 logits.
 *  k_post         : merge argmax partials, accept test, token/state outputs, status OR, publish a result record to
 *                   pinned memory (host spin-waits on result[0]).
 *  k_or_status    : accumulate validation status words (deferred check during prefill).
 *  k_select_copy  : commit-copy variant: dst = flag ? src1 : src0 (flat bank arenas).
 */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

typedef long long i64;

struct PleParams {
  int n_heads, head_dim, heads_per_ngram, n_shards;
  i64 mult[4];
  i64 vocab[32];
  i64 offset[32];
  i64 shard_end[256];
  const unsigned char* shard_ptr[256];
  float scale;
  int pad_;
  i64 eos;
  i64 capacity;
};

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
      if (P->shard_end[mid] > id) hi = mid; else lo = mid + 1;
    }
    const i64 first = lo == 0 ? 0 : P->shard_end[lo - 1];
    s_row[h] = id - first;
    s_src[h] = P->shard_ptr[lo];
  }
  if (tid == 0) {
    input_ids[0] = tok;
    position_ids[0] = pos;
    committed[0] = (int)pos;
    if (pos >= 0 && pos < P->capacity) past_pos[pos] = pos; else state[5] |= 1;
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
    if (v > best || (v == best && i < bi)) { best = v; bi = i; }
  }
  sv[threadIdx.x] = best;
  si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float v = sv[threadIdx.x + s];
      const int i = si[threadIdx.x + s];
      if (v > sv[threadIdx.x] || (v == sv[threadIdx.x] && i < si[threadIdx.x])) { sv[threadIdx.x] = v; si[threadIdx.x] = i; }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { outv[blockIdx.x] = sv[0]; outi[blockIdx.x] = si[0]; }
}

__device__ int merge_argmax(const float* v, const int* idx, int nb) {
  float best = -INFINITY;
  int bi = 0x7fffffff;
  for (int b = 0; b < nb; ++b)
    if (v[b] > best || (v[b] == best && idx[b] < bi)) { best = v[b]; bi = idx[b]; }
  return bi;
}

/* state (int64[8]): [0]=pending token, [1]=committed length L, [2]=draft token, [4]=sequence counter,
 * [5]=error bits (1 = position capacity, 2 = nonzero validation status), [6]=last accept flag
 * result ring (pinned, 8 int64 per record, `ring` records, slot = seq % ring):
 *   [0]=seq (written last), [1]=pred0, [2]=pred1, [3]=accept, [4]=status OR, [5]=new pending, [6]=new L
 * accept==1 -> commit row 1 (2 tokens), else commit row 0 (1 token). */
__global__ void k_post(const float* v0, const int* i0, const float* v1, const int* i1, int nb, i64* state,
                       i64* ctx_commit, const i64* ctx_a, const i64* ctx_b, const int* st0, const int* st1, int nst,
                       i64* result, int ring) {
  if (threadIdx.x != 0) return;
  const i64 pred0 = merge_argmax(v0, i0, nb);
  const i64 pred1 = merge_argmax(v1, i1, nb);
  const bool accept = pred0 == state[2];
  i64 status = 0;
  for (int k = 0; k < nst; ++k) status |= (i64)st0[k] | (i64)st1[k];
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

/* variant of k_post for graphs that already output argmax tokens (qwen.fused_gemm ArgmaxRows -> int64[2]) */
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
 * (pending, h1) @ L'], rows==1: [(pending, h0) @ L'].  Copies rows x 10240 bf16 hidden from src (verify hidden output rows). */
__global__ void k_head_prep(const i64* ctl, int rows, int hlen, const uint4* src, uint4* dst, i64* tokens,
                            i64* positions, int* committed) {
  if (threadIdx.x == 0) {
    const i64 pend = ctl[0], L = ctl[1], draft = ctl[2];
    if (rows == 2) {
      tokens[0] = draft; tokens[1] = pend; positions[0] = L - 1; positions[1] = L;
      committed[0] = hlen; committed[1] = hlen + 1;
    } else {
      tokens[0] = pend; positions[0] = L; committed[0] = hlen;
    }
  }
  const int n16 = rows * 10240 * 2 / 16;
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

__global__ void k_select_copy(uint4* dst, const uint4* src0, const uint4* src1, size_t n16, const i64* flag) {
  const uint4* src = flag[0] ? src1 : src0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x)
    dst[i] = src[i];
}

extern "C" {
int mtp_sizeof_params() { return (int)sizeof(PleParams); }

void mtp_prep(cudaStream_t s, const PleParams* P, const i64* tok_ptr, const i64* pos_ptr, int pos_add,
              const i64* ctx_in, i64* ctx_out, i64* input_ids, i64* position_ids, int* committed, i64* past_pos,
              __nv_bfloat16* ple_out, i64* state) {
  k_prep<<<1, 1024, 0, s>>>(P, tok_ptr, pos_ptr, pos_add, ctx_in, ctx_out, input_ids, position_ids, committed,
                           past_pos, ple_out, state);
}

void mtp_argmax_part(cudaStream_t s, const __nv_bfloat16* logits, int n, float* outv, int* outi, int nblocks) {
  k_argmax_part<<<nblocks, 256, 0, s>>>(logits, n, outv, outi);
}

void mtp_post(cudaStream_t s, const float* v0, const int* i0, const float* v1, const int* i1, int nb, i64* state,
              i64* ctx_commit, const i64* ctx_a, const i64* ctx_b, const int* st0, const int* st1, int nst,
              i64* result, int ring) {
  k_post<<<1, 32, 0, s>>>(v0, i0, v1, i1, nb, state, ctx_commit, ctx_a, ctx_b, st0, st1, nst, result, ring);
}

void mtp_post1(cudaStream_t s, const float* v0, const int* i0, int nb, i64* state, i64* ctx_commit,
               const i64* ctx_a, const int* st0, int nst, i64* result, int ring) {
  k_post1<<<1, 32, 0, s>>>(v0, i0, nb, state, ctx_commit, ctx_a, st0, nst, result, ring);
}

void mtp_post_tok(cudaStream_t s, const i64* tok01, i64* state, i64* ctx_commit, const i64* ctx_a,
                  const i64* ctx_b, const int* st0, int nst, i64* result, int ring) {
  k_post_tok<<<1, 32, 0, s>>>(tok01, state, ctx_commit, ctx_a, ctx_b, st0, nst, result, ring);
}

void mtp_head_prep(cudaStream_t s, const i64* ctl, int rows, int hlen, const void* src, void* dst, i64* tokens,
                   i64* positions, int* committed) {
  k_head_prep<<<1, 1024, 0, s>>>(ctl, rows, hlen, (const uint4*)src, (uint4*)dst, tokens, positions, committed);
}

void mtp_head_post(cudaStream_t s, const i64* next_token, i64* ctl) { k_head_post<<<1, 32, 0, s>>>(next_token, ctl); }

void mtp_or_status(cudaStream_t s, const int* st, int nst, i64* state) { k_or_status<<<1, 32, 0, s>>>(st, nst, state); }

void mtp_select_copy(cudaStream_t s, void* dst, const void* src0, const void* src1, size_t bytes, const i64* flag,
                     int blocks) {
  k_select_copy<<<blocks, 512, 0, s>>>((uint4*)dst, (const uint4*)src0, (const uint4*)src1, bytes / 16, flag);
}

/* ---- CUDA runtime helpers for the GenAI model code (all return cudaError_t as int, 0 = success) ---- */
int qg_set_device(int device) { return (int)cudaSetDevice(device); }
const char* qg_error_string(int err) { return cudaGetErrorString((cudaError_t)err); }
int qg_malloc(void** p, size_t bytes) { return (int)cudaMalloc(p, bytes); }
int qg_free(void* p) { return (int)cudaFree(p); }
int qg_host_alloc(void** p, size_t bytes) { return (int)cudaHostAlloc(p, bytes, cudaHostAllocPortable); }
int qg_host_free(void* p) { return (int)cudaFreeHost(p); }
int qg_host_register(void* p, size_t bytes, unsigned flags) { return (int)cudaHostRegister(p, bytes, flags); }
int qg_host_unregister(void* p) { return (int)cudaHostUnregister(p); }
int qg_memset(void* p, int value, size_t bytes, cudaStream_t s) { return (int)cudaMemsetAsync(p, value, bytes, s); }
int qg_memcpy_h2d(void* dst, const void* src, size_t bytes, cudaStream_t s) {
  return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s);
}
int qg_memcpy_d2h(void* dst, const void* src, size_t bytes, cudaStream_t s) {
  return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, s);
}
int qg_memcpy_d2d(void* dst, const void* src, size_t bytes, cudaStream_t s) {
  return (int)cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
}
int qg_stream_sync(cudaStream_t s) { return (int)cudaStreamSynchronize(s); }
int qg_device_sync() { return (int)cudaDeviceSynchronize(); }
int qg_mem_info(size_t* free_bytes, size_t* total_bytes) { return (int)cudaMemGetInfo(free_bytes, total_bytes); }
}
