
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
    uint elementCount;
    uint startIndex;
};

[RootSignature(ROOT_SIG_DEF)]
[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint globalIndex = dispatchThreadID.x + startIndex;
    if (globalIndex < elementCount)
    {
        ++values[globalIndex];
    }
}
