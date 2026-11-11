// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

void LaunchGeometricDecayKernel(float* logits, int vocab_size, int batch_size, int num_large, float large_val, void* stream);
void LaunchFisherYatesKernel(float* logits, int* indices, int vocab_size, int batch_size, void* stream);