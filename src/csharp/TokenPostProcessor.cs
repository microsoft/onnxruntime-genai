using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public sealed class TokenPostProcessor : IDisposable
    {
        private IntPtr _handle;
        private readonly Generator _generator;
        private object _resultLifetime = new object();

        public TokenPostProcessor(Generator generator)
        {
            _generator = generator ?? throw new ArgumentNullException(nameof(generator));
            Result.VerifySuccess(NativeMethods.OgaCreateTokenPostProcessor(generator.Handle, out _handle));
        }

        private IntPtr Handle
        {
            get
            {
                if (_handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(TokenPostProcessor));
                return _handle;
            }
        }

        public void ProcessNextTokens()
        {
            PrepareMutation();
            Result.VerifySuccess(NativeMethods.OgaTokenPostProcessorProcessNextTokens(Handle));
            GC.KeepAlive(_generator);
        }

        public void FinalizeProcessing()
        {
            PrepareMutation();
            Result.VerifySuccess(NativeMethods.OgaTokenPostProcessorFinalize(Handle));
            GC.KeepAlive(_generator);
        }

        public void Reset()
        {
            PrepareMutation();
            Result.VerifySuccess(NativeMethods.OgaTokenPostProcessorReset(Handle));
            GC.KeepAlive(_generator);
        }

        public TokenPostProcessorResult[] GetNextTokenResults()
        {
            int count = checked((int)NativeMethods.OgaTokenPostProcessorGetResultCount(Handle).ToUInt64());
            var results = new TokenPostProcessorResult[count];
            for (int index = 0; index < count; ++index)
            {
                var result = NativeMethods.OgaTokenPostProcessorGetResult(Handle, (UIntPtr)index);
                results[index] = new TokenPostProcessorResult(this, _resultLifetime, result);
            }
            return results;
        }

        private void PrepareMutation()
        {
            _ = Handle;
            _ = _generator.Handle;
            _resultLifetime = new object();
        }

        internal IntPtr GetResultHandle(IntPtr handle, object lifetime)
        {
            _ = Handle;
            if (!ReferenceEquals(lifetime, _resultLifetime))
                throw new InvalidOperationException("The postprocessor result expired after a processor mutation.");
            return handle;
        }

        ~TokenPostProcessor() { Dispose(); }

        public void Dispose()
        {
            if (_handle == IntPtr.Zero) return;
            NativeMethods.OgaDestroyTokenPostProcessor(_handle);
            _handle = IntPtr.Zero;
            GC.SuppressFinalize(this);
        }
    }
}