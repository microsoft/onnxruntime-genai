using System;

namespace Microsoft.ML.OnnxRuntimeGenAI
{
    public sealed class ASRPostProcessor : IDisposable
    {
        private IntPtr _handle;
        private readonly Generator _generator;
        private object _resultLifetime = new object();

        public ASRPostProcessor(Generator generator)
        {
            _generator = generator ?? throw new ArgumentNullException(nameof(generator));
            Result.VerifySuccess(NativeMethods.OgaCreateASRPostProcessor(generator.Handle, out _handle));
        }

        private IntPtr Handle
        {
            get
            {
                if (_handle == IntPtr.Zero) throw new ObjectDisposedException(nameof(ASRPostProcessor));
                return _handle;
            }
        }

        public void ProcessNext()
        {
            PrepareMutation();
            Result.VerifySuccess(NativeMethods.OgaPostProcessorProcessNext(
                NativeMethods.OgaASRPostProcessorAsPostProcessor(Handle)));
            GC.KeepAlive(_generator);
        }

        public void FinalizeProcessing()
        {
            PrepareMutation();
            Result.VerifySuccess(NativeMethods.OgaASRPostProcessorFinalize(Handle));
            GC.KeepAlive(_generator);
        }

        public void Reset()
        {
            PrepareMutation();
            Result.VerifySuccess(NativeMethods.OgaASRPostProcessorReset(Handle));
            GC.KeepAlive(_generator);
        }

        public ASRPostProcessorResult[] GetNextResults()
        {
            int count = checked((int)NativeMethods.OgaASRPostProcessorGetResultCount(Handle).ToUInt64());
            var results = new ASRPostProcessorResult[count];
            for (int index = 0; index < count; ++index)
            {
                var result = NativeMethods.OgaASRPostProcessorGetResult(Handle, (UIntPtr)index);
                results[index] = new ASRPostProcessorResult(this, _resultLifetime, result);
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

        ~ASRPostProcessor() { Dispose(); }

        public void Dispose()
        {
            if (_handle == IntPtr.Zero) return;
            NativeMethods.OgaDestroyASRPostProcessor(_handle);
            _handle = IntPtr.Zero;
            GC.SuppressFinalize(this);
        }
    }
}