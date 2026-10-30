// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.CompilerServices;

namespace Microsoft.ML.OnnxRuntimeGenAI.Tests
{
    internal static class TestAssemblyInitializer
    {
        [ModuleInitializer]
        internal static void Initialize()
        {
            // Root one handle for all test classes; collecting a per-test handle would shut down GenAI.
            OgaHandle ogaHandle = new OgaHandle();
            AppDomain.CurrentDomain.ProcessExit += (sender, e) => ogaHandle.Dispose();
        }
    }
}
