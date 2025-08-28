# README - GenAI.WinML

https://microsoft.visualstudio.com/ProjectOxford/_wiki/wikis/ProjectOxford.wiki/296073/WindowsML

## NUGET

1. Find the latest version you want to use.
https://microsoft.visualstudio.com/ProjectOxford/_artifacts/feed/windows-aifabric/NuGet/Microsoft.WindowsAppSDK.ML/overview/1.8.1058-experimental

2. Update CMakeList.txt with the version. 

  install_nuget_package(
    Microsoft.WindowsAppSDK.ML
    1.8.1065-experimental
    WINML_ROOT)

3. TODO: Make this a variable and pass it in from the build.

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
