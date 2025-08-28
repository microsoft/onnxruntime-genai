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

### NUGET_PAT_WCR

The WSSI artifact feed is where all the WinML stuff lives. Sadly this is in another org and so we need a way to 'cross' the streams. Today we need to do this with a user PAT, which can only live for 7 days. 

Here are the steps to generate your own PAT and update the pipeline variables to unblock a build.

1. Navigate tohttps://dev.azure.com/WSSI/, click on the user menu (top right), and select Personal access key.
  ![Personal access tokens](.readme-assets/images/PAT-step-1.png)

2. Create a token with Packaging - Read & Write permissions. `NUGET_PAT_WCR` is the recomended name, but is complety arbitrary. 
  ![Create personal access token with Packaging - Read & Write permissions](.readme-assets/images/PAT-step-2.png)

3. Navigate to one of the two pipeines, e.g. https://aiinfra.visualstudio.com/ONNX%20Runtime/_build?definitionId=1964, and select *edit*.
  ![Navigate to pipeline](.readme-assets/images/NUGET_PAT_WCR-step-1.png)

4. Select *variables*.
  ![Select variables button](.readme-assets/images/NUGET_PAT_WCR-step-2.png)

5. Select the `NUGET_PAT_WCR` variable
  ![Select variable NUGET_PAT_WCR](.readme-assets/images/NUGET_PAT_WCR-step-3.png)

5. Update the value with your PAT, OK and save. 
  ![Select variables button](.readme-assets/images/NUGET_PAT_WCR-step-4.png)

Note: Some times it can take 30-45s to update the PAT, so a bit of patience is advised.

## ATTIC

## CMake

```bash

cmake --build --preset windows_x64_cpu_relwithdebinfo --target onnxruntime-genai
cmake --preset windows_x64_cpu_relwithdebinfo

```
