// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Stands in for a qwen38 plane library built before the INT8 embedding table and the INT8 PLE shards: it exports the mandatory
// symbols of the plane ABI (generated from QWEN38_PLANE_FUNCTIONS, so it follows the ABI) and the BF16-only
// qwen38_plane_embed_rows, but neither qwen38_plane_capabilities nor qwen38_plane_embed_gather. It does no GPU work: the
// pointers it is given are host pointers. qwen38_plane_tests.cpp loads it to check what the loader does with a library that
// predates a capability.
#include <stdint.h>
#include <string.h>

#include "qwen38_plane/qwen38_plane_abi.h"

#if defined(_WIN32)
#define QWEN38_LEGACY_PLANE_EXPORT __declspec(dllexport)
#else
#define QWEN38_LEGACY_PLANE_EXPORT __attribute__((visibility("default")))
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

namespace {

enum class Function {
#define QWEN38_LEGACY_PLANE_ID(ret, name, args) name,
  QWEN38_PLANE_FUNCTIONS(QWEN38_LEGACY_PLANE_ID)
#undef QWEN38_LEGACY_PLANE_ID
};

// What a stub returns: success (0) and nothing else, except for the answers the loader asks for when it loads a library.
template <typename Result>
Result Stub(Function function);

template <>
int Stub<int>(Function function) {
  if (function == Function::abi_version) return QWEN38_PLANE_ABI_VERSION;
  if (function == Function::sizeof_ple_params) return static_cast<int>(sizeof(Qwen38PleParams));
  return 0;
}

template <>
const char* Stub<const char*>(Function) {
  return "legacy plane stub";
}

template <>
void Stub<void>(Function) {}

}  // namespace

extern "C" {

#define QWEN38_LEGACY_PLANE_STUB(ret, name, args) \
  QWEN38_LEGACY_PLANE_EXPORT ret qwen38_plane_##name args { return Stub<ret>(Function::name); }
QWEN38_PLANE_FUNCTIONS(QWEN38_LEGACY_PLANE_STUB)
#undef QWEN38_LEGACY_PLANE_STUB

// The gather of the plane before embed_gather, on host memory: copies the bfloat16 row of each int64 token id, applies no bound
// to the ids, and has no INT8 mode: a width <= 0 (how the INT8 table was once requested) returns without writing anything.
QWEN38_LEGACY_PLANE_EXPORT void qwen38_plane_embed_rows(void* stream, const void* table, const void* ids, void* out,
                                                        int rows, int width) {
  if (rows <= 0 || width <= 0) return;
  const size_t row_bytes = static_cast<size_t>(width) * sizeof(uint16_t);
  for (int row = 0; row < rows; ++row) {
    const int64_t token = static_cast<const int64_t*>(ids)[row];
    memcpy(static_cast<uint8_t*>(out) + static_cast<size_t>(row) * row_bytes,
           static_cast<const uint8_t*>(table) + static_cast<size_t>(token) * row_bytes, row_bytes);
  }
}

}  // extern "C"
