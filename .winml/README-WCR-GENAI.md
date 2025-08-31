# README - GenAI.WinML

https://microsoft.visualstudio.com/ProjectOxford/_wiki/wikis/ProjectOxford.wiki/296073/WindowsML

## NUGET

1. Find the latest version you want to use.
https://microsoft.visualstudio.com/ProjectOxford/_artifacts/feed/windows-aifabric/NuGet/Microsoft.WindowsAppSDK.ML/overview/1.8.1078-preview

2. Set the version or update the pipeline. 

  [nuget - ort_winml_version](/.pipelines/nuget-publishing.yml)
  [python - ort_winml_version](/.pipelines/pypl-publishing.yml)

## WCR Official Build 

## Build Pipelines

- [win-genai-nuget-publishing](https://aiinfra.visualstudio.com/ONNX%20Runtime/_build?definitionId=1938)
- [win-genai-python-publishing](https://aiinfra.visualstudio.com/ONNX%20Runtime/_build?definitionId=1964)

## ATTIC

## CMake

```bash

cmake --preset windows_arm64_winml_relwithdebinfo -DWINML_SDK_VERSION=1.8.1078-preview-genai

cmake --build --preset windows_arm64_winml_relwithdebinfo --target onnxruntime-genai

```
