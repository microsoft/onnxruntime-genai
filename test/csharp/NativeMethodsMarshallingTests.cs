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
        public void TokenPostProcessorOutputsUseOpaquePointers()
        {
            var nativeMethods = typeof(TokenPostProcessor).Assembly.GetType("Microsoft.ML.OnnxRuntimeGenAI.NativeMethods", true)!;
            var method = nativeMethods.GetMethod("OgaTokenPostProcessorGetResultData", BindingFlags.Public | BindingFlags.Static)!;
            Assert.Equal(typeof(IntPtr), method.ReturnType);
            var parameters = method.GetParameters();
            Assert.Equal(4, parameters.Length);
            Assert.Equal(typeof(IntPtr).MakeByRefType(), parameters[2].ParameterType);
            Assert.Equal(typeof(IntPtr).MakeByRefType(), parameters[3].ParameterType);
            Assert.All(parameters.Skip(2), parameter => Assert.True(parameter.IsOut));
        }

        [Fact]
        public void TimestampIndexedReadUsesTypedOutputs()
        {
            var nativeMethods = typeof(TokenPostProcessor).Assembly.GetType("Microsoft.ML.OnnxRuntimeGenAI.NativeMethods", true)!;
            var method = nativeMethods.GetMethod("OgaTokenPostProcessorGetTimestampRecordData", BindingFlags.Public | BindingFlags.Static)!;
            Assert.Equal(typeof(IntPtr), method.ReturnType);
            var parameters = method.GetParameters();
            Assert.Equal(8, parameters.Length);
            Assert.Equal(typeof(IntPtr).MakeByRefType(), parameters[3].ParameterType);
            Assert.Equal(typeof(long).MakeByRefType(), parameters[4].ParameterType);
            Assert.Equal(typeof(long).MakeByRefType(), parameters[5].ParameterType);
            Assert.Equal(typeof(double).MakeByRefType(), parameters[6].ParameterType);
            Assert.Equal(typeof(double).MakeByRefType(), parameters[7].ParameterType);
            Assert.All(parameters.Skip(3), parameter => Assert.True(parameter.IsOut));
        }

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
