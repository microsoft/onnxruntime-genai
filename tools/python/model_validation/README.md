# ONNX Runtime GenAI Model Validation Tutorial

## Background
Gen-AI serves as an API framework designed to operate generative models via ONNX Runtime. With the expansion in the variety of models, there's a growing need for a tool chain that can effectively assess the compatibility between Gen-AI and different model variants.

## Setup and Requirements
Clone this repository and navigate to the `tools/python/model_validation folder`.

```bash
git clone https://github.com/microsoft/onnxruntime-genai.git
cd tools/python/model_validation
pip install -r requirements.txt
```

Ensure you log into HuggingFace. 

More about the HuggingFace CLI [here](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) 
```bash
huggingface-cli login
```

Within the model_validation directory, you'll locate the script named `validation_tool.py`, alongside the `validation_config.json` configuration file and a `README.md` document.

### Current Supported Model Architectures
* Gemma
* Llama 
* Mistral
* Phi (language + vision)
* Qwen

### Current Supported Execution Providers
* CPU
* DirectML
* CUDA

## Usage 
### Steps to Configure
1. Input the name of the Hugging Face model you're using into the validation_config.json file. You can find a list of supported models via this link: (https://huggingface.co)

    * Add the chat_template associated with your model. This is located in the tokenizer_config.json file on the Hugging Face website. Make sure to replace ``` message['content'] ``` with ``` {input} ```.
        * Discover more about chat templates [here](https://huggingface.co/docs/transformers/main/chat_templating)

    * Please include the evaluation metrics you want to use with the validation tool.
        * Supported metrics for evaluation include Perplexity.
        * Learn more about evaluation metrics [here](https://huggingface.co/spaces/evaluate-metric/perplexity)


2. Specify the path for the output folder you prefer, along with the precision and execution provider details.

After the model has been created, it will be located in the `path_to_output_folder/{model_name} directory`. This directory will contain both the ONNX model data and the tokenizer.

### Run the Model Validation Script 
```bash
python validation_tool.py -j validation_config.json
```

### Output
Once the tool has been executed successfully, it generates a file named model_validation.csv. This file contains the Model Name, the validation tool's completion status, metrics calculations, and details of any exceptions or failures encountered by the model during the validation process.

