# ONNX Runtime GenAI Python Examples

## Install ONNX Runtime GenAI

Install the python package according to the [installation instructions](https://onnxruntime.ai/docs/genai/howto/install).

## Get the model

You can generate the model using the model builder with this library, or bring your own model.

If you bring your own model, you need to provide the configuration. See the [config reference](https://onnxruntime.ai/docs/genai/reference/config).

To generate the model with model builder:

1. Install the model builder's dependencies

   ```bash
   pip install numpy transformers torch onnx onnxruntime
   ```

2. Choose a model. Examples of supported ones are listed on the repo's main README.

3. Run the model builder to export, optimize, and quantize the model. More details can be found [here](../../src/python/py/models/README.md)

   ```bash
   cd examples/python
   python -m onnxruntime_genai.models.builder -m microsoft/phi-2 -e cpu -p int4 -o ./example-models/phi2-int4-cpu
   ```

## Run the example model script

See accompanying qa-e2e-example.sh and generate-e2e-example.sh scripts for end-to-end examples of workflow.

The `model-generate` script generates the output sequence all on one function call.

The `model-qa` script streams the output text token by token.

To run the python examples...
```bash
python model-generate.py -m {path to model folder} -e {execution provider} -pr {input prompt}
python model-qa.py -m {path to model folder} -e {execution provider}
```
