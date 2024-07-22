# GenAI Flatbuffer Schemas
This directory contains [the GenAI Lora Parameter format schema](genai_lora.fbs) and [the generated C++ header file](genai_lora.fbs.h) for the
Lora Parameters file format. This file format is defined as a portable intermediate between Numpy `npz` format that is produced
by Olive when exporting Lora Parameters as external files.

The format supports multiple Lora Parameters per file.

[The GenAi Lora Parameter file format schema](genai_lora.fbs) uses the [FlatBuffers](https://github.com/google/flatbuffers) serialization library.

Please do not directly modify the generated C++ header file for [The GenAi Lora Parameter file format]((genai_lora.fbs.h)).

The binary from Onnxruntime build can be re-used for the purpose.

e.g.
  - Windows Debug build
    - \build\Windows\Debug\_deps\flatbuffers-build\Debug\flatc.exe
  - Linux Debug build
    - /build/Linux/Debug/_deps/flatbuffers-build/flatc

It is possible to use another flatc as well, e.g., from a separate installation.

To update the flatbuffers schemas and generated files:
1. Modify [GenAI Lora Parameter file format schema](genai_lora.fbs).
2. Run [compile_schema.py](./compile_schema.py) to generate the C++ and Python bindings.

    ```
    python onnxruntime/core/flatbuffers/schema/compile_schema.py --flatc <path to flatc>
    ```
3. Update the version history and record the changes. Changes made to [GenAI Lora Parameter file format schema](genai_lora.fbs)
warrants not only updating the ort format version, but also the checkpoint version since the checkpoint schema
depends on the ort format schema.

# ORT FB format version history
In [genai_lora_format_version.h](../genai_lora_format_version.h), see `IsLoraParameterslVersionSupported()` for the supported versions and
`kLoraParametersVersion` for the current version.

## Version 1
History begins.

Initial support for FlatBuffers that Lora Parameters support. This includes a definition of Tensor entity
so it can be saved in a tensor per file format.
