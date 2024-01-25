
namespace Generators {
namespace cuda {

void SampleTopPKernel(int32_t* d_next_token, float* d_scores, int size, int batch_size, float threshold, float temperature, cudaStream_t stream);

}  // namespace cuda
}  // namespace Generators