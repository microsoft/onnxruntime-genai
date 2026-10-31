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

// Host-registered PLE (per-layer embedding) table description, copied to the device once. The library trusts the loader
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
  // Dequantisation scale of the shard bytes, which are FP8 E4M3 when it is positive. A negative value marks INT8 shards
  // whose scale is |scale| (QWEN38_PLANE_CAP_INT8_PLE); a library without that capability would decode them as FP8.
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

// Physical layout of the state of a CausalConvWithState replay (Qwen38StateReplayDesc::kind 1; `layout` is ignored by the other
// kinds). The state of one slot holds channel_count channels of state_width positions in either order. Zero is the
// channel-first layout, which is what every plane that predates the field assumes (the field was tail padding that the host
// zeroed), so a descriptor that never sets it keeps its meaning. The history-first layout needs a plane that reports
// QWEN38_PLANE_CAP_CONV_HISTORY_FIRST_REPLAY; the plane reads the `value` input as [capacity, channel_count] in both layouts.
typedef enum Qwen38PlaneConvStateLayout {
  // [channel_count, state_width]: (channel, position) is at channel * state_width + position
  QWEN38_PLANE_CONV_STATE_CHANNEL_FIRST = 0,
  // [state_width, channel_count]: (channel, position) is at position * channel_count + channel
  QWEN38_PLANE_CONV_STATE_HISTORY_FIRST = 1,
} Qwen38PlaneConvStateLayout;

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
  uint32_t layout;  // a Qwen38PlaneConvStateLayout, for kind 1 (it took the place of the 4 bytes of tail padding)
} Qwen38StateReplayDesc;

// The new field must not have changed the size of the descriptor that a plane built before it reads: every LP64 and LLP64
// target lays the 10 pointers, the 6 uint64_t and the 6 uint32_t (this one included; without it there were 5 of them and 4
// bytes of tail padding) out in 152 bytes, with `layout` in the last 4 bytes. This header is compiled into both the plane and
// the host loader, so neither side can drift from the other.
#ifdef __cplusplus
static_assert(sizeof(void*) != 8 || sizeof(Qwen38StateReplayDesc) == 152, "Qwen38StateReplayDesc must keep its 152 bytes");
static_assert(sizeof(void*) != 8 || offsetof(Qwen38StateReplayDesc, kind) == 144, "kind must stay where the planes read it");
static_assert(sizeof(void*) != 8 || offsetof(Qwen38StateReplayDesc, layout) == 148, "layout must take the former tail padding");
#else
_Static_assert(sizeof(void*) != 8 || sizeof(Qwen38StateReplayDesc) == 152, "Qwen38StateReplayDesc must keep its 152 bytes");
_Static_assert(sizeof(void*) != 8 || offsetof(Qwen38StateReplayDesc, layout) == 148, "layout must take the former tail padding");
#endif

// Limits of the compact state-update replay kernels (k_replay_gdn, k_replay_generic). The plane sizes its shared tiles and its
// grid from them and the host checks every replay descriptor against them while it builds it (ValidateQwen38ReplayDescriptor),
// so a geometry the kernels cannot hold fails the model load instead of silently corrupting the recurrent state.
#define QWEN38_PLANE_MAX_REPLAY_TRANSITIONS 8  // transitions one replay keeps: the capacity of the state_update outputs
#define QWEN38_PLANE_MAX_REPLAY_KEY_WIDTH 256  // GatedDeltaNet key width: a row of the key tile (a multiple of 4: float4 loads)
#define QWEN38_PLANE_REPLAY_ROWS_PER_BLOCK 32  // value rows of a GatedDeltaNet state one thread block of k_replay_gdn replays
// Thread blocks per value head of a state of `width` >= 1 value rows: ceil(width / QWEN38_PLANE_REPLAY_ROWS_PER_BLOCK).
// The kernel counts them in int and the host in uint64_t; the state can be INT32_MAX rows tall, where the usual
// (width + rows - 1) / rows overflows an int.
#define QWEN38_PLANE_REPLAY_BLOCKS_PER_HEAD(width) (1 + ((width) - 1) / QWEN38_PLANE_REPLAY_ROWS_PER_BLOCK)

// Optional exports "qwen38_plane_dflash_select" and "qwen38_plane_replay_state_updates": a library that predates them lacks the
// symbols, the loader leaves the pointers null, and the host then selects the draft chain itself and refuses compact state
// updates. Both return 0 or a cudaError_t, including the error of their own launch; cudaErrorInvalidValue comes before anything
// is launched, for the arguments named below.
//
// dflash_select walks the lattice of the drafter on the device: step s takes the best-scoring of the top_k candidates of the
// branch that step s - 1 chose (branch 0 for s = 0) and writes the token id to control word {2, 8, 9, 10, 11, 12, 13}[s]. The
// candidates are [drafts, top_k] int32 and the scores [drafts, top_k, top_k] float (the row of step s is that of the branch
// chosen before it); a tie takes the first of the best. drafts has to be in [1, 7], top_k at least 1, no pointer null.
typedef int (*Qwen38PlaneDflashSelectFn)(void* stream, const void* candidates, const void* scores, void* control, int drafts,
                                         int top_k);

// replay_state_updates keeps the first kept_count (1 .. QWEN38_PLANE_MAX_REPLAY_TRANSITIONS) transitions of the state updates
// of a verify step. descs holds fast_count GatedDeltaNet descriptors (kind 2) followed by generic_count descriptors of the
// other kinds (1 = CausalConvWithState, 4 = PackedSparseAttentionIndexer; the Snapshot kind 3 of the host is not replayed by
// the plane). Every GatedDeltaNet descriptor gets fast_blocks_per_descriptor thread blocks, which have to cover its
// heads * QWEN38_PLANE_REPLAY_BLOCKS_PER_HEAD(state_width) blocks. kept_count replaces the one of every descriptor.
// Negative counts, a missing block budget for GatedDeltaNet descriptors, a null descs and a kept_count out of range are
// refused: a call that would skip its work and report success would leave the destination state stale. What the descriptors
// hold lives on the device and cannot be checked by the plane; the host validates it while it builds them
// (ValidateQwen38ReplayDescriptor).
typedef int (*Qwen38PlaneReplayStateUpdatesFn)(void* stream, const void* descs, int fast_count, int generic_count,
                                               int fast_blocks_per_descriptor, int kept_count);

// Limits of the folded multi-row prep (prep with a negative pos_add, prep_batched).
#define QWEN38_PLANE_MAX_PREP_ROWS 8        // rows of one folded verify step
#define QWEN38_PLANE_MAX_PREP_BATCH 8       // request slots prep_batched prepares in one call
#define QWEN38_PLANE_CTL_WORDS_PER_SLOT 16  // int64 control words of one request slot (the `state` of prep)
// int64 words that prep_batched reserves for the n-gram contexts of one request slot in `ctx_out`: two for each of the
// QWEN38_PLANE_MAX_PREP_ROWS rows, whatever the row count of the call, so slot s starts at word s * 16 and the layout of the
// buffer does not change with the number of rows. The consumers of the buffer (the engine's per-slot views and the
// SpeculativePrefixCommit node, which reads slot s at s * (words / slots)) depend on that stride.
#define QWEN38_PLANE_CTX_WORDS_PER_SLOT (2 * QWEN38_PLANE_MAX_PREP_ROWS)

// X(return type, name, parameter list). The exported symbol is "qwen38_plane_" + name. `stream` is a cudaStream_t.
// CUDA runtime helpers return the cudaError_t as int (0 = success).
//
// prep and ple_rows return 0 on success. A PLE table that is not GPU-addressable (Qwen38PleParams::pad_ == 1) is "staged":
// the token and context words are read back to the host and the FP8 rows are gathered there (a gather of 32 or more
// rows by worker threads that the module starts when it first needs them). Every step of that staging (pinned
// allocation, device alias, creation of the completion event, copy, synchronisation, starting the worker threads, which
// fails with cudaErrorMemoryAllocation) is checked; when one fails, or when the table was never registered with
// set_ple_host_params, the function returns the cudaError_t (cudaErrorInvalidValue for the unregistered table) before
// it gathers or launches anything, so the outputs are untouched. Both also return the error of their own kernel
// launch. The one step that can fail after the launch is recording the completion event, which fences the staged
// buffers against a call on another stream: the function then waits for its stream before it returns the error, so
// the kernel has finished and the outputs are complete. Callers must check the result. ple_rows reads ctx_in while its
// last block writes ctx_out, so the two must not alias.
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
// replay_state_updates: a CausalConvWithState state (descriptor kind 1) in the history-first layout, [history, channels]
// (Qwen38StateReplayDesc::layout, Qwen38PlaneConvStateLayout). A library without it reads every conv state as channel-first,
// so a history-first state would be corrupted by a partial acceptance: the host requires the bit only for a package that has
// such a state, and channel-first packages keep loading with a library that lacks it.
#define QWEN38_PLANE_CAP_CONV_HISTORY_FIRST_REPLAY 0x10ull

// X(bit, name), in bit order.
#define QWEN38_PLANE_CAPABILITIES(X)                                   \
  X(QWEN38_PLANE_CAP_INT8_EMBEDDING, "int8_embedding")                 \
  X(QWEN38_PLANE_CAP_INT8_PLE, "int8_ple")                             \
  X(QWEN38_PLANE_CAP_NEGATIVE_POS_ADD_BATCH, "negative_pos_add_batch") \
  X(QWEN38_PLANE_CAP_BATCHED_PREP, "batched_prep")                     \
  X(QWEN38_PLANE_CAP_CONV_HISTORY_FIRST_REPLAY, "conv_history_first_replay")

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
// launches anything for table_rows <= 0, width <= 0, an unknown format, a BF16 table whose width is not a multiple of 8
// (a row is copied in 16-byte vectors; an INT8 row has no such limit), or an INT8 table whose scales would not be 4-byte
// aligned (table_rows * width not a multiple of 4).
typedef int (*Qwen38PlaneEmbedGatherFn)(void* stream, const void* table, long long table_rows, int format, const void* ids,
                                        void* out, int n, int width);

// Optional export "qwen38_plane_embed_rows" (void; stream, table, ids, out, rows, width): the gather of libraries that predate
// embed_gather. It reads a BF16 table of an unknown row count, so it applies no upper bound to the ids, it does nothing
// for a width <= 0 (it has no INT8 mode), and its width must be a multiple of 8 (the loader refuses a BF16 table of another
// width). The loader uses it only for a BF16 table when the library has no embed_gather.

// Optional export "qwen38_plane_prep_batched" (QWEN38_PLANE_CAP_BATCHED_PREP): prep of the `rows` rows of one folded verify
// step (see prep; rows == 1 is the plain next-token row) for each of `batch` request slots, in one call. `rows` is in
// [1, QWEN38_PLANE_MAX_PREP_ROWS] and `batch` in [1, QWEN38_PLANE_MAX_PREP_BATCH]; the pointers are those of prep except
// that every one is the base of an array over the slots, which are numbered 0 .. batch - 1:
//   state         int64, QWEN38_PLANE_CTL_WORDS_PER_SLOT words per slot, the control words of the slot (prep's `state`)
//   ctx_in        int64[batch][2]                       the n-gram context in front of row 0
//   ctx_out       int64[batch][QWEN38_PLANE_CTX_WORDS_PER_SLOT]: slot s starts at word s * QWEN38_PLANE_CTX_WORDS_PER_SLOT
//                 whatever `rows` is, and row r of it is at words 2r .. 2r + 1 as for prep; the words of a slot beyond
//                 2 * rows are not written
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
// ("qwen38_plane_test_inject_staged_fault", "qwen38_plane_test_stats", "qwen38_plane_test_peek_last_error",
// "qwen38_plane_test_stream_capture") to fail one CUDA step of the staged PLE path, to observe what the module did, to see the
// last CUDA error it left on the calling thread and to call it while a stream is being captured.

// The step of the staged PLE path that an injected fault replaces with an error.
typedef enum Qwen38PlaneFaultSite {
  QWEN38_PLANE_FAULT_NONE = 0,
  QWEN38_PLANE_FAULT_HOST_ALLOC = 1,       // pinned cudaHostAlloc of a staging buffer
  QWEN38_PLANE_FAULT_HOST_MALLOC = 2,      // malloc of the host n-gram row cache
  QWEN38_PLANE_FAULT_DEVICE_ALIAS = 3,     // cudaHostGetDevicePointer of a mapped staging buffer
  QWEN38_PLANE_FAULT_COPY = 4,             // device-to-host cudaMemcpyAsync
  QWEN38_PLANE_FAULT_SYNC = 5,             // cudaStreamSynchronize / cudaDeviceSynchronize
  QWEN38_PLANE_FAULT_EVENT_CREATE = 6,     // cudaEventCreateWithFlags of a completion event (only when it is created)
  QWEN38_PLANE_FAULT_EVENT_RECORD = 7,     // cudaEventRecord of a completion event, after the kernel was launched
  QWEN38_PLANE_FAULT_THREAD_START = 8,     // creation of a gather worker thread (std::thread throws std::system_error)
  QWEN38_PLANE_FAULT_LAUNCH = 9,           // a kernel launch that the CUDA runtime rejects (see Qwen38PlaneTestInjectFaultFn)
  QWEN38_PLANE_FAULT_GATHER_SCRATCH = 10,  // the scratch lists of a gather thread (std::bad_alloc)
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
  long long gather_pool_threads;      // live worker threads of the staged PLE gather
} Qwen38PlaneTestStats;

// Arms one fault: the (skip + 1)-th execution of `site` from now on fails with `error` instead of calling CUDA, once. A site
// of QWEN38_PLANE_FAULT_NONE disarms. Returns 0, or cudaErrorInvalidValue for an unknown site or an `error` of 0.
// QWEN38_PLANE_FAULT_LAUNCH does not replace a CUDA call: the (skip + 1)-th kernel launch that the module reports the status of
// (every export that launches a kernel) is preceded by a launch of no blocks, which the runtime rejects like any bad launch
// configuration, so what the export reports for its launch is the error that the runtime recorded (its code depends on the CUDA
// version). The kernel itself still runs, so every argument has to be valid. `error` is not used, but like for every other site
// it must not be 0.
typedef int (*Qwen38PlaneTestInjectFaultFn)(int site, int skip, int error);
// Fills `stats`. Returns 0, or cudaErrorInvalidValue for a null `stats`.
typedef int (*Qwen38PlaneTestStatsFn)(Qwen38PlaneTestStats* stats);
// Returns the last CUDA error of the calling thread without clearing it (cudaPeekAtLastError).
typedef int (*Qwen38PlaneTestPeekLastErrorFn)(void);
// Starts (`begin` != 0) or ends the capture of `stream` into a CUDA graph that is thrown away, in the thread-local mode so
// that only the calling thread is affected. Returns the CUDA error of the call, 0 on success.
typedef int (*Qwen38PlaneTestStreamCaptureFn)(void* stream, int begin);
