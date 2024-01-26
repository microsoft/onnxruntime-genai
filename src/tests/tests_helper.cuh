
void LaunchGeometricDecayKernel(float* logits, int vocab_size, int batch_size, cudaStream_t stream);
void LaunchFisherYatesKernel(float* logits, int* indices, int vocab_size, int batch_size, cudaStream_t stream);