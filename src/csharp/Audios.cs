// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using System;
using System.Runtime.InteropServices;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public class Audios : IDisposable
    {
        private IntPtr _audiosHandle;
        private bool _disposed = false;

        private Audios(IntPtr audiosHandle)
        {
            _audiosHandle = audiosHandle;
        }

        internal IntPtr Handle { get { return _audiosHandle; } }

        public static Audios Load(string[] audioPaths)
        {
            Result.VerifySuccess(NativeMethods.OgaCreateStringArray(out IntPtr stringArray));
            foreach (string audioPath in audioPaths)
            {
                Result.VerifySuccess(NativeMethods.OgaStringArrayAddString(stringArray, StringUtils.ToUtf8(audioPath)));
            }
            Result.VerifySuccess(NativeMethods.OgaLoadAudios(stringArray, out IntPtr audiosHandle));
            NativeMethods.OgaDestroyStringArray(stringArray);
            return new Audios(audiosHandle);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }
            NativeMethods.OgaDestroyAudios(_audiosHandle);
            _audiosHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}