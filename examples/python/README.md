# ONNX Runtime GenAI Python Examples

## Install ONNX Runtime GenAI

Install the python package according to the [installation instructions](https://onnxruntime.ai/docs/genai/howto/install) or [build from source](https://onnxruntime.ai/docs/genai/howto/build-from-source.html).

## Get the model

You can generate the model using the model builder with this library, download the model from huggingface ([example](https://github.com/microsoft/onnxruntime-genai?tab=readme-ov-file#sample-code-for-phi-3-in-python)), or bring your own model.

If you bring your own model, you need to provide the configuration. See the [config reference](https://onnxruntime.ai/docs/genai/reference/config).

To generate the model with model builder:

1. Install the model builder's dependencies

   ```bash
   pip install numpy transformers torch onnx onnxruntime
   ```

2. Choose a model. Examples of supported ones are listed on the repo's main [README](../../README.md).

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

## Use Constrained Decoding for the model output

Constrained Decoding is useful when using function/tool calling as it helps in ensuring the output is in the correct format.

We have integrated [LLGuidance](https://github.com/guidance-ai/llguidance) for constrained decoding. There are three types of constrained decoding enabled right now:
1. Lark Grammar (Recommended): This option allows you to have an option for a regular output as well as function/tool output in JSON format.
2. JSON Schema: Output will be JSON schema and it will be one of the function/tools provided.
3. Regex: If a particular regular expression is desired.

To ensure that the function/tool call works correctly with constrained decoding, you need to modify your tokenizer.json file. For each model that has its own tool calling token, the tool calling token's `special` attribute needs to be set to true. For example, Phi-4 mini uses the <|tool_call|> token so you should set the `special` attribute for <|tool_call|> as `true` inside `tokenizer.json`.

To run the Python examples with function/tool calling:
```
# Using Lark Grammar with 1 function/tool call
python model-qa.py -m {path to model folder} -e {execution provider} --guidance_type "lark_grammar"  --guidance_info '[{"name": "get_weather", "description": "Get weather of a city.", "parameters": {"city": {"description": "The city for which weather information is requested", "type": "string", "default": "Dallas"}}}]'

# With 2 function/tool calls in chat mode
python model-chat.py -m {path to model folder} -e {execution provider} --guidance_type "lark_grammar"  --guidance_info '[{"name": "get_weather", "description": "Get weather of a city.", "parameters": {"city": {"description": "The city for which weather information is requested", "type": "string", "default": "Dallas"}}},{"name": "get_population", "description": "Get population of a city.", "parameters": {"city": {"description": "The city for which population information is requested", "type": "string", "default": "Dallas"}}}]'
```
