// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class OnnxRuntimeGenAIException: Exception
    {
        internal OnnxRuntimeGenAIException(string message)
            :base(message)
        {
        }
    }


}
