// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>
    /// An exception which contains the error message and code produced by the native layer.
    /// </summary>
    public class OnnxRuntimeGenAIException: Exception
    {
        internal OnnxRuntimeGenAIException(string message)
            :base(message)
        {
        }
    }
}
