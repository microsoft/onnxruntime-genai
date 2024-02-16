// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Generators {

namespace cuda {

void BeamSearchTopK(
    const float* input,
    int32_t batch_size,
    int32_t num_beams,
    int32_t vocab_size,
    int32_t k,
    float* tmp_values_1st_stage,
    int32_t* tmp_indices_1st_stage,
    float* tmp_values_2st_stage,
    int32_t* tmp_indices_2st_stage,
    float* output_values,
    int32_t* output_tokens,
    int32_t* output_indices,
    cudaStream_t stream);

template <typename T, int max_k>
struct TopK {
  int32_t key[max_k];
  T value[max_k];

  __device__ __forceinline__ void Insert(T elem, int elem_id) {
    T v = value[max_k - 1];
    if (v < elem ||
        (key[max_k - 1] == -1) ||
        ((elem == value[max_k - 1]) && (elem_id < key[max_k - 1]))) {
      value[max_k - 1] = elem;
      key[max_k - 1] = elem_id;
    }

    for (int k = max_k - 2; k >= 0; --k) {
      if (value[k + 1] > value[k] ||
          key[k] == -1 ||
          ((value[k + 1] == value[k]) && (key[k + 1] < key[k]))) {
        T u2 = value[k];
        int p2 = key[k];
        value[k] = value[k + 1];
        key[k] = key[k + 1];
        value[k + 1] = u2;
        key[k + 1] = p2;
      }
    }
  }

  __device__ __forceinline__ void Init() {
    for (int i = 0; i < max_k; i++) {
      key[i] = -1;
      value[i] = -std::numeric_limits<T>::infinity();
    }
  }
};

template <typename T, int max_k>
__device__ __forceinline__ TopK<T, max_k> reduce_topk_op(const TopK<T, max_k>& a, const TopK<T, max_k>& b) {
  TopK<T, max_k> res = a;
  for (int i = 0; i < max_k; ++i)
    res.Insert(b.value[i], b.key[i]);
  return res;
}

}  // namespace cuda
}  // namespace Generators
