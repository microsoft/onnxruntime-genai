// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    [StructLayout(LayoutKind.Sequential)]
    public readonly struct TokenTiming
    {
        public int TokenId { get; }
        public long StartFrame { get; }
        public long StopFrame { get; }
    }
}