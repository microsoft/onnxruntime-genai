# LLM Chat UI <!-- omit in toc -->
This is a chat demo using the various versions of the LLMs

> The app supports all backend that onnxruntime-genai supports. DirectML is used as an example to showcase how to use it

**Contents**:
- [set up](#set-up)
- [Launch the app](#launch-the-app)

# set up

1. Install **onnxruntime-genai-directml** 
    > If you want to use CUDA model, you can download `onnxruntime-genai-cuda` package.
   
   ```
   pip install numpy
   pip install --pre onnxruntime-genai-directml
   ```

   Or following [Build onnxruntime-genai from source](https://onnxruntime.ai/docs/genai/howto/build-from-source.html#build-onnxruntime-genai-from-source)

2. Install the requirements

    ```
    pip install -r requirements.txt
    ```

# Launch the app

1. Create folder named models at the root directory of chat_app.

2. Download models to the created folder, take phi-3-mini directml as example.

    ```bash
    huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include directml/* --local-dir .
    ```

     File structure should look as the below:
    ```
    --chat_app
    --models
        --directml
            --phi-3-mini-directml-int4-awq-block-128
            --meta-llama_Llama-2-7b-chat-hf
            --mistralai_Mistral-7B-Instruct-v0.1
            ...
        --cuda
            --phi3-mini-v
    ```




3. Launch the app

    ```
    python chat_app/app.py
    ```

    You should see output from console
    ```
    Running on local URL:  http://127.0.0.1:7860

        To create a public link, set `share=True` in `launch()`.
    ```

   Then open the local URL in broswer
   ![alt text](image.png)