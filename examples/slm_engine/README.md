# SLM Engine for running GenAI on the Edge Compute

SLM Engine is a C++ based application that uses ONNX Runtime (ORT) genrate() API library and runs GenAI models on the edge computing resources. The SLM engine is developed to deploy generative AI models (sometimes fine tuned for API function calling specific to certain Edge use cases). Any small language based generative AI model quantized to ONNX can run on SLM engine.

The following diagram illustrates a hugh level architecture of the SLM Engine and it's relationship with the ONNX Runtime libraries.

<div align="center">
    <img src="doc/SLM-Engine-Arch.png" height="240">
    <p>SLM Engine Architecture</p>
</div>

SLM Engine is designed to be built from sources for different types of target hardware to maximize the portability. Building from source allows maximum customization to ensure most efficient execution of workloads for specific target hardware.

The current version is tested on Windows 11, MacOS, Linux and Android running on the CPU of various platforms. The SLM Engine also runs on various accelerators (such as GPU and NPU) via the execution provider mechanism of the ONNX runtime.

## Getting Started

To use the SLM Engine in your AI application, use one of the following methods:

- Use the C++ library from your C++ application by including the `slm_engine.h` and creating an instance of the `SLMEngine` class.

OR

- Run the command line program `slm-server` which starts a web server and exposes an API endpoint that accepts OpenAI like text completion API. Then make REST API calls pointing at this endpoint from your application.

### SLM Server

The command line options of the slm-server are the following:

```shell
$ ./slm-server  --help
SLM Runner Version: v0.3.0
Unknown argument: --help
Usage: slm_server --model_family VAR --model_path VAR [--port_number VAR] [--verbose]

Optional arguments:
  -mf, --model_family   Type of model: llama3.2 or phi3
  -m,  --model_path     Path to the model file [required]
  -p, --port_number     HTTP Port Number to use (default 8080)
  -v, --verbose         If provided, more debugging information printed on standard output

```

### Example Launch Command

```shell
$ ./slm-server -mf phi3  -m <path to the ONNX model> -v

```

Once the server is running, you can use a HTTP client to send user queries to the server and generate responses. Following is an example cURL command that talks to the SLM Engine vis the REST API:

### Example cURL

```bash
curl -X POST http://localhost:8000/completion -H "Content-Type: application/json" --data '{"messages": [{"role": "system", "content": "You are a helpful AI Assistant. Please answer the questions very accurately. Use emojis and markdown as appropriate"},{"role": "user", "content": "What are the top 5 places to visit in San Diego? Be brief."}], "max_tokens": 1200, "temperature": 0.7}'
```

The SLM server supports the following REST APIs (click to expand):

<details>
 <summary><code>GET</code> <code><b>/</b></code> <code>Returns SLM server status</code></summary>

##### Parameters

> None

##### Responses

> | http code | content-type       | response      |
> | --------- | ------------------ | ------------- |
> | `200`     | `application/json` | `JSON Object` |

##### Example cURL

```javascript
>  curl -X GET http://localhost:8000
```

##### JSON Schema for the Response for GET /

```json
{
    "response":
    {
        "status": "success",
        "engine_state":
        {
            "engine_version": <Version String>,
            "model": <Model name>
        }
    }
}
```

</details>

<details>
 <summary><code>POST</code> <code><b>/complete</b></code> <code>Given the prompt, generates response from SLM</code></summary>

##### Parameters

> | name | type     | data type                                   | description |
> | ---- | -------- | ------------------------------------------- | ----------- |
> | None | required | object (JSON formatted using OpenAI schema) | N/A         |

##### Responses

> | http code | content-type       | response      |
> | --------- | ------------------ | ------------- |
> | `200`     | `application/json` | `JSON Object` |

##### JSON Schema for the Response for /completion (success)

```json
{
    "response":
    {
        "status": "success",
        "answer": "{<SLM Generated text>}",
        "kpi": {
            "generated_toks": <Value>,
            "prompt_toks": <Value>,
            "tok_rate": <value>,
            "total_time": <Value>,
            "ttft": <Value>
        },
        "question": <User's Question>,
        "llm_input": "<Actual String input to LLM"
    }
}
```

##### JSON Schema for the Response for /completion (error)

```json
{
  "response": {
    "status": "error",
    "message": "Error Message"
  }
}
```

</details>

## Installation

Since this is targeted for various devices running on the Edge we provide a simple to use build setup that the developers can use to build for any system of their choosing.

### Prerequisites

The SLM Engine first builds the `onnxruntime` and `onnxruntime-genai` libraries from source. Therefore any prerequisites that apply to build from source of these two libraries also applicable to SLM Engine. There are no additional requirements for building SLM engine.

#### Windows Long File Path

For Windows, often the maximum path length is 260 which results in breaking the onnxruntime dependency build. Therefore, the long file path needs to be enabled in the group policy editor using the following steps:

- Open the 'Run' command (Win+R) and type gpedit.msc, then press Enter.
- Navigate to: Computer Configuration > Administrative Templates > System > Filesystem.
- Double-click Enable Win32 long paths and set it to Enabled, then click Apply.

Following are the platforms we tested the builds.

| Platform         | Binary Directory Name | Comments                                                           |
| ---------------- | --------------------- | ------------------------------------------------------------------ |
| Android-aarch64  | Android-aarch64       | Can be cross compiled on Windows, MacOS or Linux                   |
| MacOS-aarch64    | Darwin-aarch64        | Great for host side development                                    |
| Ubuntu 24.04/x86 | Linux-x86_64          | Necessary for Cross Compiling for Android and Qualcomm QNN support |
| Ubuntu 24.04/ARM | Linux-aarch64         |
| Windows 11       | Windows-AMD64         | Can be used to run on AI PCs                                       |

### Build from Source

In order to install the software from source, you will need C++ toolchain such as clang/llvm and cmake. Since the build scripts use Python3 any Python 3 would work. However, to enable Qualcomm QNN support, you need to use a Linux host and Python 3.8

Building is as easy as following these steps:

#### Build the Dependencies

This program is based on ONNRuntime-GenAI library which in turn depends on ONNX Runtime core libraries. First step is to build the dependencies. Open a terminal window and run the following steps:

```bash
$ cd build_scripts
$ python build_deps.py
...
```

This will first build the onnxruntime by cloning a local copy in the deps/src directory. Once the build is complete, the script will next build the onnxruntime-gen - which is this repo itself. Upon completion, the script will clone and build few other `header only` dependencies. At the end of the build, all the built artifacts such as the header files and libraries are stored inside the deps/artifacts directory under a subdirectory that's named after the target platform.

For example, if you are building on MacOS then the built artifacts will be stored in `deps/artifacts/Darwin-aarch64/`. Similarly, if you are cross compiling for Android, then the artifacts will be stored in `deps/artifacts/Android-aarch64/`.

Following are the command line options applicable for the dependency build:

```bash
usage: build_deps.py [-h] [--android] [--android_sdk_path ANDROID_SDK_PATH] [--android_ndk_path ANDROID_NDK_PATH]
                     [--api_level API_LEVEL] [--qnn_sdk_path QNN_SDK_PATH] [--build_type BUILD_TYPE] [--skip_ort_build]

Build script for dependency libraries

options:
  -h, --help            show this help message and exit
  --android             Build for Android
  --android_sdk_path ANDROID_SDK_PATH
                        Path to ANDROID SDK
  --android_ndk_path ANDROID_NDK_PATH
                        Path to ANDROID NDK
  --api_level API_LEVEL
                        Android API Level
  --qnn_sdk_path QNN_SDK_PATH
                        Path to Qualcomm QNN SDK (AI Engine Direct)
  --build_type BUILD_TYPE
                        {Release|RelWithDebInfo|Debug}
  --skip_ort_build      If set, skip building ONNX Runtime

```

#### Build SLM Engine

Next step is to build the program itself. For that use the script `build.py` with appropriate command line options as needed for Android build.

```bash
$ python build.py
```

For Android build, the following commandline options are important:

```bash
usage: build.py [-h] [--android] [--android_ndk_path ANDROID_NDK_PATH] [--build_type BUILD_TYPE]

Build script for this repo

options:
  -h, --help            show this help message and exit
  --android             Build for Android
  --android_ndk_path ANDROID_NDK_PATH
                        Path to ANDROID NDK
  --build_type BUILD_TYPE
                        {Release|RelWithDebInfo|Debug}

```

## Testing the build

After the build is complete, the binaries are available in the build_scripts/builds/<TARGET_NAME>/install/bin directory. To test the build, download the ONNX model first and then run following command to execute SLM engine with a sample input file.

### Download ONNX Model

1. Navigate to the Hugging Face and download an SLM such as [Microsoft Phi-3-mini-4k-instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx)
1. Next run the `slm-runner` CLI program pointing at the CPU version of the quantized model.

Following example show how to test this on a Windows 11:

```Powershell
>  cd .\build_scripts\builds\Windows-AMD64\install\bin\
>  .\slm-runner.exe -mf phi3 -t ..\..\..\..\..\test\batch-input.jsonl -m ..\..\..\..\..\..\..\..\..\models\Phi-3-mini-4k-instruct-onnx\cpu_and_mobile\cpu-int4-rtn-block-32-acc-level-4\ -o output.jsonl -v

```

The SLM Runner used above is a CLI program for testing the SLM engine in a batch mode. The following section provides more details.

#### SLM Runner

The `slm-runner` CLI application works in a batch mode and thus useful for benchmarking and testing. In addition to the ONNX model, you will also need to prepare a JSONL file that contains the system and user prompts formatted like OpenAI API. Following is an example of a line of JSON fragment that contains the `system` and `user` messages:

```shell
$ ./slm-runner --help
SLM Runner Version: v0.3.0
Unknown argument: --help
Usage: slm_runner --model_family VAR --model_path VAR --test_data_file VAR --output_file VAR [--verbose]

Optional arguments:
  -mf, --model_family   Type of model: llama3.2 or phi3
  -m, --model_path      Path to the model file [required]
  -t, --test_data_file  Path to the test data file (JSONL) [required]
  -o, --output_file     Path to the output file (JSONL) [required]
  -v, --verbose         If provided, more debugging information printed on standard output
```

```JSON
{"messages":
    [
        {
            "role": "system",
            "content": "You are an in car virtual assistant that maps user's inputs to the corresponding function call in the vehicle. You must respond with only a JSON object matching the following schema: {\"function_name\": <name of the function>, \"arguments\": <arguments of the function>}"
        },
        {
            "role": "user", "content": "Can you please set the radio to 90.3"
        }
    ],
    "max_tokens": 300,
    "temperature": 0.0,
    "stop": ["\n"]
}

```
