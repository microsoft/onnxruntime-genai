# Deterministic Compute

To request deterministic implementations from ONNX Runtime, set the boolean
`model.decoder.session_options.use_deterministic_compute` before creating the model:

```python
import onnxruntime_genai as og

config = og.Config(model_path)
config.overlay('''{
  "model": {
    "decoder": {
      "session_options": {"use_deterministic_compute": true}
    }
  }
}''')
model = og.Model(config)
```

The same setting can be placed in `genai_config.json`. It is a session option,
not a dynamically changeable `set_runtime_option` key. It is disabled by default.
An MTP decoder inherits the target decoder's value unless explicitly overridden.

Support depends on the operator and ONNX Runtime version. CUDA MatMulNBits needs
a runtime that honors this flag for fpA-intB tactic selection; older runtimes may
still choose tactics using timing measurements. Deterministic implementations
may be slower than profiled implementations.

This option does not disable random sampling. Use greedy generation when testing
repeatability. It also does not promise identical results across different
hardware, runtime versions, batch shapes, or target-only versus speculative
execution.