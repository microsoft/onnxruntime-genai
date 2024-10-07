# ONNX Runtime GenAI Model Validation  Example

## Setup

Clone this repository and navigate to the `tools/python/model_validation folder`.

```bash
git clone https://github.com/microsoft/onnxruntime-genai.git
cd tools/python/model_validation
```

In the model_validation folder, you should find the validation_tool.py script, validation_config.json file, and this README.md.

### Current Support
* Gemma
* Llama 
* Mistral
* Phi
* Qwen

### Usage - Build the Model
This step creates optimized and quantized ONNX models that run with ONNX Runtime GenAI.

1. In the validation_config.json file, enter the supported Hugging Face model name. Models can be found here.
2. Include the path to the output folder, precision, and execution provider.

Once the model is built, you can find it in path_to_output_folder/{model_name}. This should include the ONNX model data and tokenizer.

### Run the Model Validation Script 
```bash
python validation_tool.py -j validation_config.json
```

