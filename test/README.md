# ONNX Runtime GenAI Test Folder Structure

This directory contains assets, scripts, and definitions used by the test suite.

## Layout

- `test/audios/*` contains all audio files used for testing.
- `test/images/*` contains all image files used for testing.
- `test/models/model_name/*` contains uploaded assets for `model_name` such as `model.onnx`, `genai_config.json`, `tokenizer.json`, and related files. These are already-created models stored in the repo.
- `test/python/*` contains all Python scripts used for testing.
    - `test/python/builder/*` contains Python scripts used to validate the model builder.
    - `test/python/create/*` contains Python scripts used to create ONNX models. Scripts follow the `create_*.py` naming format.
    - `test/python/models/*` contains Python scripts used to test ONNX models. Scripts follow the `test_*.py` naming format.
- `test/tool-definitions/*` contains all tool definitions used for testing.
