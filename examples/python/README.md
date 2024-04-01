# Gen-AI Python Examples

## Install the onnxruntime-genai library

Install the python package according to the [installation instructions](https://onnxruntime.ai/docs/genai/howto/install).

  ```bash
  cd build/wheel
  pip install onnxruntime_genai-*.whl
  ```

## Get the model

You can generate the model using the model builder this library, or bring your own model.

If you bring your own model, you need to provide the configuration. See the [config reference](https://onnxruntime.ai/docs/genai/reference/config).

To generate the model with model builder:

1. Install the model builder script dependencies

   ```bash
   pip install numpy
   pip install transformers
   pip install torch
   pip install onnx
   pip install onnxruntime
   ```

2. Choose a model. Examples of supported ones are:
   - Phi-2
   - Mistral
   - Gemma 2B IT
   - LLama 7B

3. Run the model builder script to export, optimize, and quantize the model. More details can be found [here](../../src/python/py/models/README.md)

   ```bash
   cd examples/python
   python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -e cpu -p int4 -o ./example-models/phi2-int4-cpu
   ```

## Run the example model script

See accompanying chat-e2e-example.sh and generate-e2e-example.sh scripts for end-to-end examples of workflow.

To run the python examples...
```bash
python model-generate.py -m {path to model folder} -ep {cpu or cuda} -i {string prompt}
python model-chat.py -m {path to model folder} -ep {cpu or cuda}
```
