// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

namespace Generators {

using ScoreType = float;  // TODO: Move to header includable by cuda

namespace cuda {

void BeamSearchTopK(
    const ScoreType* input,
    int32_t batch_size,
    int32_t num_beams,
    int32_t vocab_size,
    int32_t k,
    ScoreType* tmp_values_1st_stage,
    int32_t* tmp_indices_1st_stage,
    ScoreType* tmp_values_2st_stage,
    int32_t* tmp_indices_2st_stage,
    ScoreType* output_values,
    int32_t* output_tokens,
    int32_t* output_indices,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace Generators
