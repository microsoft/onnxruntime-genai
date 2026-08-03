// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using Xunit;

namespace Microsoft.ML.OnnxRuntimeGenAI.Tests
{
    public class NativeMethodsMarshallingTests
    {
        [Fact]
        public void NativeBoolParametersUseOneByteMarshalling()
        {
            Type nativeMethods = typeof(Utils).Assembly.GetType(
                "Microsoft.ML.OnnxRuntimeGenAI.NativeMethods",
                throwOnError: true)!;

            Type boolType = typeof(bool);
            Type boolByRefType = boolType.MakeByRefType();
            ParameterInfo[] boolParameters = nativeMethods
                .GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                .SelectMany(method => method.GetParameters())
                .Where(parameter =>
                    parameter.ParameterType == boolType ||
                    parameter.ParameterType == boolByRefType)
                .ToArray();

            Assert.NotEmpty(boolParameters);
            foreach (ParameterInfo parameter in boolParameters)
            {
                MarshalAsAttribute marshalAs = parameter.GetCustomAttribute<MarshalAsAttribute>();
                Assert.NotNull(marshalAs);
                Assert.Equal(UnmanagedType.I1, marshalAs.Value);
            }
        }
    }
}
