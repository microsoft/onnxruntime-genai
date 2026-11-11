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
        public void ASRPostProcessorOutputsUseOpaquePointers()
        {
            var nativeMethods = typeof(ASRPostProcessor).Assembly.GetType("Microsoft.ML.OnnxRuntimeGenAI.NativeMethods", true)!;
            foreach (var name in new[] { "OgaASRPostProcessorAsPostProcessor", "OgaPostProcessorProcessNext",
                "OgaPostProcessorFinalize", "OgaPostProcessorReset",
                "OgaASRPostProcessorGetResult", "OgaASRPostProcessorResultGetTimestampResult",
                "OgaASRPostProcessorTimestampResultGetTimestampRecord" })
            {
                var method = nativeMethods.GetMethod(name, BindingFlags.Public | BindingFlags.Static)!;
                Assert.Equal(typeof(IntPtr), method.ReturnType);
            }
            var destroy = nativeMethods.GetMethod("OgaDestroyPostProcessor", BindingFlags.Public | BindingFlags.Static)!;
            Assert.Equal(typeof(void), destroy.ReturnType);
            Assert.Equal(typeof(IntPtr), Assert.Single(destroy.GetParameters()).ParameterType);
        }

        [Fact]
        public void TimestampRecordLookupUsesKindSelector()
        {
            var nativeMethods = typeof(ASRPostProcessor).Assembly.GetType("Microsoft.ML.OnnxRuntimeGenAI.NativeMethods", true)!;
            var method = nativeMethods.GetMethod("OgaASRPostProcessorTimestampResultGetTimestampRecord", BindingFlags.Public | BindingFlags.Static)!;
            var parameters = method.GetParameters();
            Assert.Equal(3, parameters.Length);
            Assert.Equal(typeof(IntPtr), parameters[0].ParameterType);
            Assert.True(parameters[1].ParameterType.IsEnum);
            Assert.Equal(typeof(int), Enum.GetUnderlyingType(parameters[1].ParameterType));
            Assert.Equal(typeof(UIntPtr), parameters[2].ParameterType);
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
