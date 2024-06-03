
//------------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//------------------------------------------------------------------------------

#define ROOT_SIG_DEF "DescriptorTable(UAV(u0, numDescriptors=2, flags=DATA_VOLATILE | DESCRIPTORS_VOLATILE)), RootConstants(num32BitConstants=1, b0)"
#define NUM_THREADS 256

RWStructuredBuffer<T> input_mask : register(u0);
RWStructuredBuffer<T> output_mask : register(u1);

cbuffer Constants
{
    uint max_seq_len;
    uint seq_len;
    uint element_count;
    uint start_index;
};

[RootSignature(ROOT_SIG_DEF)]
[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 dispatch_thread_id : SV_DispatchThreadID)
{
    uint global_index = dispatch_thread_id.x + start_index;
    if (global_index < element_count)
    {
        uint sequence_index = global_index % max_seq_len;

        if (seq_len > 1)
        {
            const T value = sequence_index < seq_len ? 1 : 0;
            output_mask[global_index] = value;
        }
        else
        {
            output_mask[global_index] = (sequence_index == 0 || input_mask[sequence_index] == 1 || input_mask[sequence_index - 1] == 1) ? 1 : 0;
        }
    }
}
