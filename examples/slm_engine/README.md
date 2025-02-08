# SLM Engine for running GenAI on the Edge Compute

SLM Engine is a C++ based application that uses ONNX Runtime (ORT) genrate() API library and runs GenAI models on the edge computing resource. The SLM engine was developed to deploy generative AI models fine tuned for API function calling specific to in-car applications but it's not limited to that specific use case. Any small language based generative AI model quantized to ONNX can run on SLM engine.

The following diagram illustrates a hugh level architecture of the SLM Engine and it's relationship with the ONNX Runtime libraries.

<div align="center">
    <img src="doc/SLM-Engine-Arch.png" height="240">
    <p>SLM Engine Architecture</p>
</div>

At present, SLM Engine is tested using the CPU version of the ORT GenAI library. It is tested on Linux running on ARM64 virtual machines on Azure, Android running on Snapdragon based smartphone, and MacOS on M3 Pro using quantized version of Phi3-Mini 4K model. In the following sections we will provide the details of this application.

## Getting Started

The SLM Engine is shipped with two command line executable programs at the moment. They are as follows:

1. `slm-server`: This is a http server that launches the engine and accepts text completion requests using JSON and modeled after OpenAI API
2. `slm-runner`: An executable that takes a JSONL file containing the input data formatted as OpenAI API messages, meant for running benchmarks and testing.

After installation, you can run the SLM engine by pointing to the ONNX version of the SLM model.

Following are the usage of these two tools:

### SLM Server

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

#### Example Launch Command

```shell
$ ./slm-server -mf phi3  -m <path to the ONNX model> -p 8000 -v

```

Once the server is running, you can use a HTTP client to send user queries to the server and generate responses.

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

##### Example cURL

```javascript
>  curl -X POST http://localhost:8000/completion -H "Content-Type: application/json"
>   --data '{"messages": [{"role": "system", "content": "You are an in car virtual assistant that maps user'\''s inputs to the corresponding function call in the vehicle. You must respond with only a JSON object matching the following schema: {\"function_name\": <name of the function>, \"arguments\": <arguments of the function>}"},{"role": "user", "content": "Can you please set it to 74 degrees?"}], "stop": ["\n"]}'
```

##### JSON Schema for the Response for /completion (success)

```json
{
    "response":
    {
        "status": "success",
        "answer": "{<JSON STRING with FUNCTION INFO>}",
        "kpi": {
            "generated_toks": <Value>,
            "prompt_toks": <Value>,
            "tok_rate": <value>,
            "total_time": <Value>,
            "ttfs": <Value>
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

### SLM Runner

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

The `slm-runner` CLI application works in a batch mode and thus useful for benchmarking and testing. In addition to the ONNX model, you will also need to prepare a JSONL file that contains the system and user prompts formatted like OpenAI API. Following is an example of a line of JSON fragment that contains the `system` and `user` messages:

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

## Installation

Since this is targeted for various devices running on the Edge, it's not possible to provide binaries for specific platforms (while exclude the others). Therefore we provide a simple to use build setup that the developers can use to build for any system of their choosing.

Following are the platforms we tested the builds.

| Platform         | Binary Directory Name | Comments                                                           |
| ---------------- | --------------------- | ------------------------------------------------------------------ |
| Android-aarch64  | Android-aarch64       | Only available via cross compiling                                 |
| MacOS-aarch64    | Darwin-aarch64        | Great for host side development                                    |
| Ubuntu 24.04/x86 | Linux-x86_64          | Necessary for Cross Compiling for Android and Qualcomm QNN support |
| Ubuntu 24.04/ARM | Linux-aarch64         |
| Windows          | TBD                   | Planned                                                            |

### Install From Source

In order to install the software from source, you will need C++ toolchain such as clang/llvm and cmake. Since the build scripts use Python3 any Python 3 would work. However, to enable Qualcomm QNN support, you need to use a Linux host and Python 3.8

Building is as easy as following these steps:

### Build the Dependencies

This program is based on ONNRuntime-GenAI library which in turn depends on ONNX Runtime core libraries. First step is to build the depndencies. Open a terminal window and run the following steps:

```bash
$ cd build_scripts
$ python build_deps.py
...
```

This will first build the onnxruntime by cloning a local copy in the deps/src directory. Once the build is complete, the script will next build the onnxruntime-gen - which is this repo itself. Upon completion, the script will clone and build few other `header only` dependencies. At the end of the build, all the built artifacts such as the header files and libraries are stored inside the deps/artifacts directory under a subdirectory that's named after the target platform.

For example, if you are building on MacOS then the built artifacts will be stored in `deps/artifacts/Darwin-aarch64/`. Similarly, if you are cross compiling for Android, then the artifacts will be stored in `deps/artifacts/Android-aarch64/`.

Following are the command line options applicable for the dependecy build:

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

### Build this program

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

TBD
