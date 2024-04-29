# Run the Phi-3 Mini models with the ONNX Runtime generate() API

## Steps
1. [Download Phi-3 Mini](#download-the-model)
2. [Install the generate() API](#install-the-generate()-api-package)
3. [Run Phi-3 Mini](#run-the-model)

## Download the model 

Download either or both of the [short](https://aka.ms/phi3-mini-4k-instruct-onnx) and [long](https://aka.ms/phi3-mini-128k-instruct-onnx) context Phi-3 mini models from Hugging Face.


For the short context model.

```bash
git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
```

For the long context model

```bash
git clone https://huggingface.co/microsoft/Phi-3-mini-128k-instruct-onnx
```

These model repositories have models that run with DirectML, CPU and CUDA.

## Install the generate() API package

You only need to install one of these packages, depending on your hardware configuration.

### Option 1: On Windows

Install the DirectML package when you are running a Windows machine with a GPU. If you don't know whether you have a GPU in your Windows machine, you can check with this [guide](https://www.microsoft.com/en-us/windows/learning-center/how-to-check-gpu).

```
pip install numpy
pip install --pre onnxruntime-genai-directml
```

### Option 2: Other operating systems with CUDA

Install the CUDA package when you have CUDA installed. You can verify whether you have a CUDA capable GPU using this [guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#verify-you-have-a-cuda-capable-gpu)

```
pip install numpy
pip install --pre onnxruntime-genai-cuda --index-url=https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-genai/pypi/simple/
```

### Option 3: Other configurations

If you are not running on a GPU, use the CPU package.

```
pip install numpy
pip install --pre onnxruntime-genai
```

## Run the model

Run the model with [model-qa.py](https://github.com/microsoft/onnxruntime-genai/blob/main/examples/python/model-qa.py).

The script accepts a model folder and takes the generation parameters from the config in that model folder. You can also override the parameters on the command line.

This example is using the long context model running with DirectML on Windows.

The `-m` argument is the path to the model you downloaded from HuggingFace above.
The `-l` argument is the length of output you would like to generate with the model.

```bash
curl https://raw.githubusercontent.com/microsoft/onnxruntime-genai/main/examples/python/model-qa.py -o model-qa.py
python model-qa.py -m Phi-3-mini-128k-instruct-onnx/directml/directml-int4-awq-block-128 -l 2048
```

Once the script has loaded the model, it will ask you for input in a loop, streaming the output as it is produced the model. For example:

```bash
Input: <|user|>Tell me a joke about creative writing<|end|><|assistant|>
 
Output:  Why don't writers ever get lost? Because they always follow the plot! 
```