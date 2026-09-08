
//------------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//------------------------------------------------------------------------------

#define ROOT_SIG_DEF "DescriptorTable(UAV(u0, numDescriptors=1, flags=DATA_VOLATILE | DESCRIPTORS_VOLATILE)), RootConstants(num32BitConstants=1, b0)"
#define NUM_THREADS 256

RWStructuredBuffer<T> values : register(u0);

cbuffer Constants
{
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
        ++values[global_index];
    }
}
