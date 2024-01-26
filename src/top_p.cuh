
namespace Generators {
namespace cuda {

// TODO: o lord i flubbed the naming schemes
void launch_populate_indices(int* indices, int size, int batch_size, cudaStream_t stream);
void SampleTopPKernel(int32_t* d_next_token, float* d_scores, int size, int batch_size, float threshold, float temperature, cudaStream_t stream);

}  // namespace cuda
}  // namespace Generators