To run a test:

python -m pytest -sv test_onnxruntime_genai_api.py -k "<your_test_name>" --test_models ..\models

For example:

python -m pytest -sv test_onnxruntime_genai_api.py -k "test_greedy_search" --test_models ..\models

CI runs four telemetry smoke tests. Run the complete regression suite with
`python -m pytest -q test_telemetry.py`, or pass `--telemetry-tests full` to
`test_onnxruntime_genai.py`. Manual Windows/Linux CPU x64 workflows and the
macOS CPU pipeline expose the same `telemetry_tests` selection.
