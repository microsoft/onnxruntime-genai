# Gen-AI C# Phi-2 Example

## Install the onnxruntime-genai library


## Get the model

You can generate the model using the model builder this library, or bring your own model.

If you bring your own model, you need to provide the configuration. See the [config reference](https://onnxruntime.ai/docs/genai/reference/config).

To generate the model with model builder:

1. Install the python package

   ```bash
   pip install onnxruntime-genai
   ```

2. Install the model builder script dependencies

   ```bash
   pip install numpy
   pip install transformers
   pip install torch
   pip install onnx
   pip install onnxruntime
   ```
3. Run the model builder script to export, optimize, and quantize the model. More details can be found [here](../../src/python/py/models/README.md)

   ```bash
   cd examples/python
   python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -e cpu -p int4 -o ./example-models/phi2-int4-cpu
   ```

## Run the phi-2 model

Install the OnnxRuntime.GenAI nuget according to the [installation instructions](https://onnxruntime.ai/docs/genai/install).

Open [HelloPhi2.sln](HelloPhi2.sln) and run the console application.
