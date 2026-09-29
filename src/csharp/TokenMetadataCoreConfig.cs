using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    /// <summary>Mutable metadata options copied into a tokenizer stream when its state is created.</summary>
    public sealed class TokenMetadataCoreConfig : IDisposable
    {
        private IntPtr _configHandle;
        private bool _disposed = false;

        public TokenMetadataCoreConfig()
        {
            Result.VerifySuccess(NativeMethods.OgaCreateTokenMetadataCoreConfig(out _configHandle));
        }

        internal IntPtr Handle { get { return _configHandle; } }

        public void Overlay(string json)
        {
            Result.VerifySuccess(NativeMethods.OgaTokenMetadataCoreConfigOverlay(_configHandle, StringUtils.ToUtf8(json)));
        }

        ~TokenMetadataCoreConfig()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (_disposed)
            {
                return;
            }
            NativeMethods.OgaDestroyTokenMetadataCoreConfig(_configHandle);
            _configHandle = IntPtr.Zero;
            _disposed = true;
        }
    }
}