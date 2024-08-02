/*
 * The implementation of this file is based on code provided by https://github.com/NVIDIA/FasterTransformer
 *
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modifications Copyright (c) Microsoft.
// Licensed under the MIT License.

// Modifications:
// (1) Minor routine name changes for integration into the ORT code-base

#pragma once

// #include "core/providers/cuda/cuda_common.h"
// #include "core/providers/cuda/cu_inc/common.cuh"

// using namespace Generators::cuda;

namespace Generators {
namespace cuda {

struct __align__(8) Half4 {
  half2 x;
  half2 y;
};

__device__ __forceinline__ Half4 operator+(const Half4& a, const Half4& b) {
  Half4 r;
  r.x = a.x + b.x;
  r.y = a.y + b.y;
  return r;
}

__device__ __forceinline__ float2 operator+(const float2& a, const float2& b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float4 operator+(const float4& a, const float4& b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

// struct Float8_ {
//   float2 x;
//   float2 y;
//   float2 z;
//   float2 w;
// };

// struct Float4_ {
//   float2 x;
//   float2 y;
// };

// template <typename T>
// struct num_elems;
// template <>
// struct num_elems<float> {
//   static constexpr int value = 1;
// };
// template <>
// struct num_elems<float2> {
//   static constexpr int value = 2;
// };
// template <>
// struct num_elems<float4> {
//   static constexpr int value = 4;
// };
// template <>
// struct num_elems<Float4_> {
//   static constexpr int value = 4;
// };
// template <>
// struct num_elems<Float8_> {
//   static constexpr int value = 8;
// };

// template <>
// struct num_elems<uint32_t> {
//   static constexpr int value = 2;
// };
// template <>
// struct num_elems<uint2> {
//   static constexpr int value = 4;
// };
// template <>
// struct num_elems<uint4> {
//   static constexpr int value = 8;
// };

// template <typename T>
// struct Vec_t {
//   static constexpr int size = 0;
// };

// template <>
// struct Vec_t<float> {
//   using Type = float2;
//   static constexpr int size = 2;
// };

// template <>
// struct Vec_t<float2> {
//   using Type = float4;
//   static constexpr int size = 4;
// };

// template <>
// struct Vec_t<float4> {
//   using Type = Float8_;
//   static constexpr int size = 8;
// };

// template <>
// struct Vec_t<half> {
//   using Type = uint32_t;
//   static constexpr int size = 2;
// };

// template <>
// struct Vec_t<half2> {
//   using Type = uint2;
//   static constexpr int size = 4;
// };

// template <>
// struct Vec_t<Half4> {
//   using Type = uint4;
//   static constexpr int size = 8;
// };

// //------------------------------------------------------------
// // Qk_vec
// //------------------------------------------------------------
// template <typename T, int head_size>
// struct Qk_vec_m_ {
// };

// template <>
// struct Qk_vec_m_<float, 32> {
//   using Type = float;
// };

// template <>
// struct Qk_vec_m_<float, 64> {
//   using Type = float2;
// };

// template <>
// struct Qk_vec_m_<float, 128> {
//   using Type = float4;
// };

// template <>
// struct Qk_vec_m_<float, 256> {
//   using Type = float4;
// };

// template <>
// struct Qk_vec_m_<uint16_t, 32> {
//   using Type = uint32_t;
// };

// template <>
// struct Qk_vec_m_<uint16_t, 64> {
//   using Type = uint32_t;
// };

// template <>
// struct Qk_vec_m_<uint16_t, 128> {
//   using Type = uint2;
// };

// template <>
// struct Qk_vec_m_<uint16_t, 256> {
//   using Type = uint4;
// };

// template <typename T, int head_size>
// struct Qk_vec_k_ {
//   using Type = typename Qk_vec_m_<T, head_size>::Type;
// };

// //------------------------------------------------------------
// // K_vec
// //------------------------------------------------------------
// template <typename T, int THREADS_PER_KEY>
// struct K_vec_m_ {
// };

// template <>
// struct K_vec_m_<float, 4> {
//   using Type = float;
// };

// template <>
// struct K_vec_m_<float, 2> {
//   using Type = float2;
// };

// template <>
// struct K_vec_m_<float, 1> {
//   using Type = float4;
// };

// template <>
// struct K_vec_m_<uint16_t, 4> {
//   using Type = uint32_t;
// };

// template <>
// struct K_vec_m_<uint16_t, 2> {
//   using Type = uint2;
// };

// template <>
// struct K_vec_m_<uint16_t, 1> {
//   using Type = uint4;
// };

// template <typename T, int THREADS_PER_KEY>
// struct K_vec_k_ {
//   using Type = typename K_vec_m_<T, THREADS_PER_KEY>::Type;
// };

//------------------------------------------------------------
// V_vec
//------------------------------------------------------------
template <typename T, int V_VEC_SIZE>
struct V_vec_m_ {
};

template <>
struct V_vec_m_<float, 1> {
  using Type = float;
};

template <>
struct V_vec_m_<float, 2> {
  using Type = float2;
};

template <>
struct V_vec_m_<float, 4> {
  using Type = float4;
};

template <>
struct V_vec_m_<uint16_t, 2> {
  using Type = uint32_t;
};

template <>
struct V_vec_m_<uint16_t, 4> {
  using Type = uint2;
};

template <>
struct V_vec_m_<uint16_t, 8> {
  using Type = uint4;
};

template <>
struct V_vec_m_<half, 2> {
  using Type = half2;
};

template <>
struct V_vec_m_<half, 4> {
  using Type = Half4;
};

template <typename T, size_t size>
struct TypeMapper : public V_vec_m_<T, size> {};

template <>
struct TypeMapper<int32_t, 2> {
  using Type = uint2;
};

template <>
struct TypeMapper<int32_t, 4> {
  using Type = uint4;
};

constexpr int GPU_WARP_SIZE_HOST = 32;

}  // namespace cuda
}  // namespace Generators
