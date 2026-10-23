// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

namespace Generators {

inline constexpr uint32_t kQwen38PlaneVersion = 1;

// No CUDA types cross the core/add-on boundary. Streams and device buffers are opaque.
struct Qwen38PlaneFunctions {
  int (*sizeof_params)(){};
  int (*set_device)(int){};
  const char* (*error_string)(int){};
  int (*malloc_)(void**, size_t){};
  int (*free_)(void*){};
  int (*host_alloc)(void**, size_t){};
  int (*host_free)(void*){};
  int (*host_register)(void*, size_t, unsigned){};
  int (*host_unregister)(void*){};
  int (*host_get_device_pointer)(void**, void*, unsigned){};
  int (*memset_)(void*, int, size_t, void*){};
  int (*memcpy_h2d)(void*, const void*, size_t, void*){};
  int (*memcpy_d2h)(void*, const void*, size_t, void*){};
  int (*memcpy_d2d)(void*, const void*, size_t, void*){};
  int (*stream_sync)(void*){};
  void (*prep)(void*, const void*, const void*, const void*, int, const void*, void*, void*, void*, void*, void*, void*,
               void*){};
  void (*argmax_part)(void*, const void*, int, void*, void*, int){};
  void (*post_tok)(void*, const void*, void*, void*, const void*, const void*, const void*, int, void*, int){};
  void (*post1)(void*, const void*, const void*, int, void*, void*, const void*, const void*, int, void*, int){};
  void (*head_prep)(void*, const void*, int, int, const void*, void*, void*, void*, void*){};
  void (*head_post)(void*, const void*, void*){};
  void (*or_status)(void*, const void*, int, void*){};
  int (*memcpy2d_d2d)(void*, size_t, const void*, size_t, size_t, size_t, void*){};
  int (*stream_create)(void**){};
  int (*stream_destroy)(void*){};
  void (*ple_rows)(void*, const void*, const void*, int, const void*, void*, void*){};
  void (*pool_keys)(void*, const void*, int, const void*, const void*, const void*, int, float, int, void*){};
  void (*transpose_f32)(void*, const void*, void*, int, int){};
  void (*embed_rows)(void*, const void*, const void*, void*, int, int, int){};
  int (*dflash_select)(void*, const void*, const void*, void*, int, int){};
  int (*argmax_finish)(void*, const void*, const void*, int, void*){};
};

using GetQwen38PlaneFunctionsFn = const Qwen38PlaneFunctions* (*)(uint32_t, size_t);

}  // namespace Generators
