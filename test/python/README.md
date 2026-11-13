To run a test:

python -m pytest -sv test_onnxruntime_genai_api.py -k "<your_test_name>" --test_models ..\models

For example:

python -m pytest -sv test_onnxruntime_genai_api.py -k "test_greedy_search" --test_models ..\models

For real Qwen38 Flash CUDA graph ownership tests, set `ORT_QWEN38_TEST_MODEL`
to an exported package with CUDA graph capture enabled, then run
`python -m pytest -sv engine/test_qwen38_graph_lifecycle.py`. For a plugin CUDA
provider not registered by the test environment, also set
`ORT_QWEN38_TEST_CUDA_PLUGIN` to its shared library path. These tests serialize
Engine ownership across repeated main/worker/main handoffs and a loader thread
that exits, checking exact multi-step greedy output equality. They also inject
an initialization failure after warmup captures, then verify both a clean model
reload and a failed handoff retry on the same model.
An exited-loader-to-new-worker case also overwrites path-backed graph buffers
after model creation, checking that session recreation uses package files rather
than the no-longer-valid borrowed data.
