// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    ///<summary>
    /// OnnxRuntime GenAI exception.
    ///</summary>
    public class OnnxRuntimeGenAIException: Exception
    {
        internal OnnxRuntimeGenAIException(string message)
            :base(message)
        {
        }
    }
}
