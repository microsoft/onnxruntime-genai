# Running OrtGenAI on NPU

## Download the NPU model

1. Download and install [foundry-local](https://github.com/microsoft/Foundry-Local/releases/download/v0.4.91/FoundryLocal-arm64-0.4.91.9885.msix)
2. List available models: `foundry model list`
3. Download an NPU model (models with `Device` specified as `NPU`): `foundry model download deepseek-r1-distill-qwen-7b-qnn-npu`
4. Find out where the model is saved on disk: `foundry cache location`
5. Identify the path to the model on disk. For example: `C:\Users\bmeswani\.foundry\Microsoft\deepseek-r1-distill-qwen-7b-qnn-npu\qnn-deepseek-r1-distill-qwen-7b`

## Build the C++ Example

1. Clone the repo branch: `git clone --branch baijumeswani/run-on-npu https://github.com/microsoft/onnxruntime-genai.git`
2. `cd onnxruntime-genai`
3. Download [ONNX Runtime libraries](https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.QNN/1.22.0).
   - Rename extension from `*.nupkg` to `*.zip`.
   - Extract the zip and copy over all the files from `microsoft.ml.onnxruntime.qnn.1.22.0\runtimes\win-arm64\native` to `examples\c\lib`
4. Download [ONNX Runtime GenAI libraries](https://github.com/microsoft/onnxruntime-genai/releases/download/v0.8.3/onnxruntime-genai-0.8.3-win-arm64.zip)
   - Extract the zip and copy over all the files from `onnxruntime-genai-0.8.3-win-arm64\onnxruntime-genai-0.8.3-win-arm64\lib` to `examples\c\lib`
   - Copy over all the header files from `onnxruntime-genai-0.8.3-win-arm64\onnxruntime-genai-0.8.3-win-arm64\include` to `examples\c\include`
5. Build using cmake

   ```sh
   cd examples\c
   cmake -G "Visual Studio 17 2022" -S . -B build -DMODEL_QA=ON -A ARM64
   cmake --build build --parallel --config Debug
   ```

## Run the Sample

1. Open `Task Manager` -> `Performance` -> `NPU 0`
2. Run the sample
  
   ```sh
   cd build\Debug
   .\model_qa.exe <path\to\model\from\above>
   ```
3. Enter a prompt: `What is the square root of 16?`
4. See NPU usage in the `Task Manager`
