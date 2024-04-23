# Run the Phi-3 Mini models with the ONNX Runtime generate() API

## Steps
1. [Download Phi-3 Mini](#download-the-model)
2. [Install the generate() API](#install-the-generate()-api-package)
3. [Run Phi-3 Mini](#run-the-model)

## Download the model 

Download either or both of the [short](https://aka.ms/phi3-mini-4k-instruct-onnx) and [long](https://aka.ms/phi3-mini-128k-instruct-onnx) context Phi-3 mini models from Hugging Face.

There are ONNX models for CPU (used for mobile too), as well as DirectML and CUDA.


## Install the generate() API package

### DirectML

```
pip install numpy onnxruntime-genai-directml --pre 
```

### CPU

```
pip install numpy onnxruntime-genai --pre
```

### CUDA

```
pip install numpy onnxruntime-genai-cuda --pre --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

## Run the model

Run the model with [this script](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py).

The script accepts a model folder and takes the generation parameters from the config in that model folder. You can also override the parameters on the command line.

This example is using the long context model running with DirectML on Windows.

```bash
python model-qa.py -m models/phi3-mini-128k-instruct-directml-int4-awq-block-128
```

Once the script has loaded the model, it will ask you for input in a loop, streaming the output as it is produced the model. For example:

```bash
Input: <|user|>Tell me a joke about creative writing<|end|><|assistant|>
 
Output:  Why don't writers ever get lost? Because they always follow the plot! 
```
