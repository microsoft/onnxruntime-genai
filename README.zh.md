# ONNX Runtime GenAI

## 状态 (Status)

[![Latest version](https://img.shields.io/nuget/vpre/Microsoft.ML.OnnxRuntimeGenAI.Managed?label=latest)](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntimeGenAI.Managed/absoluteLatest)

[![Nightly Build](https://github.com/microsoft/onnxruntime-genai/actions/workflows/linux-cpu-x64-nightly-build.yml/badge.svg)](https://github.com/microsoft/onnxruntime-genai/actions/workflows/linux-cpu-x64-nightly-build.yml)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

## 项目简介 (Description)

借助 ONNX Runtime 运行生成式 AI 模型。该 API 为端侧设备运行大语言模型（LLM）提供了便捷、灵活且高效的方式。它为 ONNX 模型完整实现了生成式 AI 循环逻辑，涵盖输入预处理与后处理、基于 ONNX Runtime 的模型推理、Logits 计算与处理、搜索与采样（Search and Sampling）、KV 缓存管理（KV cache management）以及面向工具调用的语法规范约束（Grammar Specification）。

ONNX Runtime GenAI 为 Foundry Local、Windows ML 以及 Visual Studio Code AI Toolkit 提供底层支持。

详情请参阅 [ONNX Runtime 官方网站](https://onnxruntime.ai/docs/genai) 文档。

| 支持矩阵 (Support matrix) | 当前已支持 (Supported now) | 开发中 (Under development) | 规划中 (On the roadmap)|
| -------------- | ------------- | ----------------- | -------------- |
| 模型架构 (Model architectures) | AMD OLMo <br/> ChatGLM <br/> DeepSeek <br/> ERNIE 4.5 <br/> Fara <br/> Gemma <br/> gpt-oss <br/> Granite <br/> Granite MoE Hybrid <br/> HunYuan Dense V1 <br/> InternLM2 <br/> LFM2（文本 + 图像 + 语音输入） <br/> Llama <br/> Mistral <br/> Nemotron <br/> Phi（文本 + 视觉） <br/> Qwen（文本 + 视觉） <br/> SmolLM3 <br/> Whisper | Stable diffusion | 多模态模型 |
| 开发语言 API | Python <br/>C# <br/>C/C++ <br/> Java ^ | Objective-C ||
| 操作系统 (O/S) | Linux <br/> Windows <br/>Mac <br/>Android || iOS |||
| 硬件架构 (Architecture) | x86 <br/> x64 <br/> arm64 ||||
| 硬件加速 (Hardware Acceleration) | CPU <br/> CUDA <br/> DirectML <br/> NvTensorRtRtx (TRT-RTX) <br/> OpenVINO <br/> QNN <br/> WebGPU | | AMD GPU |
| 核心特性 (Features) | 多 LoRA (Multi-LoRA) <br/> 连续解码 (Continuous decoding) <br/> 约束解码 (Constrained decoding) | | 投机解码 (Speculative decoding) |

^ 需要从源码构建

## 安装指南 (Installation)

请参阅[安装指南](https://onnxruntime.ai/docs/genai/howto/install)或[从源码构建说明](https://onnxruntime.ai/docs/genai/howto/build-from-source.html)。

## Python 运行 Phi-3 示例代码 (Sample code for Phi-3 in Python)

1. 下载模型权重：

   ```shell
   huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .
   ```

2. 安装 API 包：

   ```shell
   pip install numpy
   pip install --pre onnxruntime-genai
   ```

3. 运行模型：

   ```python
   import onnxruntime_genai as og

   model = og.Model('cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4')
   tokenizer = og.Tokenizer(model)
   stream = tokenizer.create_stream()

   # 默认设置一个合理的 max_length，否则将默认为整个上下文长度
   search_options = {}
   search_options['max_length'] = 2048
   search_options['batch_size'] = 1

   chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

   text = input("Input: ")
   if not text:
      print("Error, input cannot be empty")
      exit()

   prompt = f'{chat_template.format(input=text)}'

   input_tokens = tokenizer.encode(prompt)

   params = og.GeneratorParams(model)
   params.set_search_options(**search_options)
   generator = og.Generator(model, params)

   print("Output: ", end='', flush=True)

   try:
      generator.append_tokens(input_tokens)
      while not generator.is_done():
         generator.generate_next_token()
         new_token = generator.get_next_tokens()[0]
         print(stream.decode(new_token), end='', flush=True)
   except KeyboardInterrupt:
         print("  --control+c pressed, aborting generation--")

   print()
   del generator
   ```

### 选择匹配的示例代码版本 (Choose the correct version of the examples)

由于本项目持续演进并不断加入新特性，`main` 分支中的示例代码可能无法与最新的稳定发布版完全对齐。本节介绍如何确保示例代码与您使用的版本保持兼容。

### 稳定版本 (Stable version)

按照[安装说明](https://onnxruntime.ai/docs/genai/howto/install)安装对应软件包。例如安装 Python 包：

```bash
pip install onnxruntime-genai
```

查看已安装软件包的版本：

Linux/Mac：
```bash
pip list | grep onnxruntime-genai
```

Windows：
```bash
pip list | findstr "onnxruntime-genai"
```

然后检出与该版本相对应的示例代码分支：

```bash
# 克隆仓库
git clone https://github.com/microsoft/onnxruntime-genai.git && cd onnxruntime-genai
# 检出与您所用版本相对应的分支
git checkout v0.11.5
cd examples
```

### Nightly 开发版（main 分支）

检出代码仓库的 main 分支：

```bash
git clone https://github.com/microsoft/onnxruntime-genai.git && cd onnxruntime-genai
```

参考[从源码构建文档](https://onnxruntime.ai/docs/genai/howto/build-from-source.html)进行编译构建。例如构建 Python wheel 安装包：

```bash
python build.py
```

进入 main 分支下的 examples 目录：

```bash
cd examples
```

如需直接安装 Nightly Python 构建包：

```bash
# 将 onnxruntime-genai 替换为您需要安装的包名
pip install --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-genai
```

## 路线图 (Roadmap)

请参阅 [Discussions](https://github.com/microsoft/onnxruntime-genai/discussions) 提交新功能需求或对已有功能提案进行投票。

## 数据与遥测 (Data/Telemetry)

本项目可能会收集使用情况数据并将其发送给微软，以帮助改进我们的产品与服务。详情请参阅[隐私声明](docs/Privacy.md)。

## 参与贡献 (Contributing)

本项目诚挚欢迎社区贡献与建议！大多数贡献要求您签署贡献者许可协议（CLA），声明您有权并确实授予我们使用您贡献成果的权利。详情请访问 https://cla.opensource.microsoft.com。

有关如何从源码构建、测试与格式化本项目的详细说明，请参阅 [DEVELOPMENT.md](docs/DEVELOPMENT.md)。

当您提交 Pull Request 时，CLA 机器人会自动检测您是否需要签署 CLA 并标注 PR（例如添加状态检查、评论提醒）。只需按照机器人提示操作即可。在使用微软 CLA 的所有开源代码库中，您仅需签署一次。

本项目已采纳 [Microsoft 开源行为准则](https://opensource.microsoft.com/codeofconduct/)。欲了解更多信息，请参阅[行为准则常见问题解答](https://opensource.microsoft.com/codeofconduct/faq/)，若有任何疑问亦可发送邮件至 [opencode@microsoft.com](mailto:opencode@microsoft.com)。

### 代码格式化与检查 (Linting)

本项目使用 [lintrunner](https://github.com/suo/lintrunner) 进行代码检查与格式化。您可以通过以下命令安装依赖并完成初始化：

```sh
pip install -r requirements-lintrunner.txt
lintrunner init
```

这将在您的系统上安装 lintrunner 并下载本地运行 linter 所需的所有依赖项。

格式化本地变动的代码：

```bash
lintrunner -a
```

格式化仓库中的全部文件：

```bash
lintrunner -a --all-files
```

## 商标声明 (Trademarks)

本项目可能包含项目、产品或服务的商标或标志。授权使用微软商标或标志需遵守并遵循 [Microsoft 商标与品牌指南](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general)。在修改后的项目版本中使用微软商标或标志不得造成混淆或暗示微软的官方赞助。任何第三方商标或标志的使用均受相应第三方的政策约束。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月18日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
