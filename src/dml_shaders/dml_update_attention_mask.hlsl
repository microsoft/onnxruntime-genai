
//------------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//------------------------------------------------------------------------------

#define ROOT_SIG_DEF "DescriptorTable(UAV(u0, numDescriptors=2, flags=DATA_VOLATILE | DESCRIPTORS_VOLATILE)), RootConstants(num32BitConstants=1, b0)"
#define NUM_THREADS 256

RWStructuredBuffer<T> inputMask : register(u0);
RWStructuredBuffer<T> outputMask : register(u1);

cbuffer Constants
{
    uint maxSeqLen;
    uint elementCount;
    uint startIndex;
    uint seqLen;
};

[RootSignature(ROOT_SIG_DEF)]
[numthreads(NUM_THREADS, 1, 1)]
void CSMain(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    uint globalIndex = dispatchThreadID.x + startIndex;
    if (globalIndex < elementCount)
    {
        uint sequenceIndex = globalIndex % maxSeqLen;

        if (seqLen > 1 && sequenceIndex < seqLen)
        {
            outputMask[globalIndex] = 1;
        }
        else if (sequenceIndex == 0 || inputMask[sequenceIndex] == 1 || inputMask[sequenceIndex - 1] == 1)
        {
            outputMask[globalIndex] = 1;
        }
        else
        {
            outputMask[globalIndex] = 0;
        }
    }
}
