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
        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void TokenMetadataCopiesNullableInterval(bool hasTiming)
        {
            var assembly = typeof(TokenMetadataInput).Assembly;
            var nativeType = assembly.GetType("Microsoft.ML.OnnxRuntimeGenAI.NativeTokenMetadataInput", true)!;
            var intervalType = assembly.GetType("Microsoft.ML.OnnxRuntimeGenAI.NativeTokenMetadataAcousticFrameInterval", true)!;
            const BindingFlags fields = BindingFlags.Instance | BindingFlags.NonPublic;
            Assert.Equal(24, Marshal.SizeOf(nativeType));
            Assert.Equal((IntPtr)8, Marshal.OffsetOf(nativeType, "TokenAcousticFrameInterval"));
            var interval = Activator.CreateInstance(intervalType)!;
            intervalType.GetField("Start", fields)!.SetValue(interval, 2L);
            intervalType.GetField("Stop", fields)!.SetValue(interval, 7L);
            var native = Activator.CreateInstance(nativeType)!;
            nativeType.GetField("TokenId", fields)!.SetValue(native, 42);
            nativeType.GetField("HasTokenAcousticFrameInterval", fields)!.SetValue(native, hasTiming ? 1 : 0);
            var intervalField = nativeType.GetField("TokenAcousticFrameInterval", fields)!;
            Assert.Equal(intervalType, intervalField.FieldType);
            var buffer = Marshal.AllocHGlobal(Marshal.SizeOf(nativeType));
            TokenMetadataInput token;
            try
            {
                intervalField.SetValue(native, interval);
                Marshal.StructureToPtr(native, buffer, false);
                var copy = Marshal.PtrToStructure(buffer, nativeType)!;
                token = (TokenMetadataInput)Activator.CreateInstance(
                    typeof(TokenMetadataInput), fields, null, new[] { copy }, null)!;
                Marshal.WriteInt64(buffer, 8, 99L);
                Marshal.WriteInt64(buffer, 16, 100L);
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
            Assert.Equal(42, token.TokenId);
            Assert.Equal(hasTiming, token.TokenAcousticFrameInterval.HasValue);
            if (hasTiming)
                Assert.Equal((2L, 7L), token.TokenAcousticFrameInterval.Value);
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
