---
description: "Use when implementing, refactoring, exporting, or reviewing the Python Model Builder, graph expansions, loaders, and quantization."
applyTo: "src/python/py/models/**"
---

# Python Model Builder - Copilot Instructions

When generating or reviewing code in the Model Builder (`src/python/py/models/`), evaluate all changes against the criteria defined in the following files:

- [README.md](../../src/python/py/models/README.md)
- [DESIGN.md](../../src/python/py/models/DESIGN.md)
- [LOADERS.md](../../src/python/py/models/loaders/LOADERS.md)
- [QUANTIZATION.md](../../src/python/py/models/quantization/QUANTIZATION.md)

Read all documents to understand the intended usage, supported models, design principles, and architectural constraints before suggesting or reviewing any code changes in this area.

## Code Style Guidelines

- Lots of AI-generated code will be proposed in pull requests. Scrutinize every single line of code related to the model builder in every PR very heavily. Most of the generated code is not actually necessary if designed correctly. Most PRs do not get this right and most proposers do not understand this.
- When a node is inserted into the model, prefer using `self.make_op_name` as the wrapper method for `self.make_node` + `self.make_value` calls.
- Do NOT use global variables in any logic. Any static variables should be declared inside a standard Python method (e.g. `init` methods, a regular Python function, etc).
- Ignore any CodeQL warnings about how an __init__ method calls an overridden method. These warnings are false positives and can be safely ignored. The warning message is: "this call to ABC in an initialization method is overwritten by XYZ".
- Before adding an abstraction, helper, flag, or instance attribute, look for an existing builder API or convention that expresses the change more simply. Reuse shared base-class functionality instead of duplicating it.
- For any new `extra_options` that are added, make sure that they are documented in both `README.md` and `builder.py`. In `README.md`, there should be a description of the option and its purpose. There should be a usage example thereafter showing how to use the option when calling the model builder from wheel or from source. In `builder.py`, there should be a description of the option and its purpose, its default value, and any possible values. Any constraints or limitations should also be documented. Make sure the documentation is consistent across both files.
- Always raise an issue if `staticmethod` is used in any method added EXCEPT if the method it is attached to is a `from_pretrained` method. The use of `staticmethod` should otherwise be avoided. If you find a case where it is used, raise an issue to find a better way to implement the functionality without using `staticmethod`.
- Always raise an issue if a variable or method is named with a leading underscore. The use of leading underscores should be avoided. If you find a case where it is used, raise an issue to find a better way to implement the functionality without using leading underscores.
- Always raise an issue if there is a lot of complicated logic in a method. If you find a case where there is a lot of complicated logic, raise an issue to find a better way to implement the functionality without using complicated logic.
- Avoid putting customized logic inside the base class constructor (i.e. avoid putting inside `Model.__init__`). Reuse the appropriate existing initialization method rather than adding one-off initialization helpers.
- When many new variables are created inside the model builder (e.g. many variables to control how an MoE subgraph looks, how quantized KV caches work, etc.), you should move those variables into settings stored within `_attrs` dictionaries. Each setting should be documented, explain its purpose, and mention when it's used.
- Ensure that existing dispatcher methods only continue to dispatch across branches. For example, `make_matmul_op`, `make_attention_op`, and `make_mlp_op` should only dispatch across branches. If you find a case where a dispatcher method is doing more than dispatching, raise an issue to find a better way to implement the functionality without using dispatcher methods for more than dispatching.
- As new data types and new quantization schemas are introduced, make sure that the core feature of the model builder still remains. The logic in `base.py` + each model class should be generic enough. It is the job of the parser class from the new data type or new quantization schema to convert the model into a form that the model builder can understand. The core model builder logic should not be modified directly to support new data types or new quantization schemas. If you find a case where the model builder is being modified to support new data types or new quantization schemas, raise an issue to find a better way to implement the functionality without modifying the model builder.
- Make sure that any bit unpacking and repacking logic is done when the quantized model is parsed through the parser class. The model builder should not need to be modified to support new quantization schemas. If you find a case where the model builder is being modified to support new quantization schemas, raise an issue to find a better way to implement the functionality without modifying the model builder.
- Put generic builder methods in `Model`, architecture-specific methods in the corresponding builder class, and portable standard-ONNX decompositions and their helpers in `expansions/standard.py`. Keep provider-specific behavior, such as WebGPU graph capture, in the provider's expansion module. Register expansion overrides under the same method names they replace, without an `_expansion` suffix.

## Configuration and Optimization

- Keep the configuration surface small. Apply safe optimizations automatically using the existing execution-provider, dtype, weight-layout, and quantization settings; do not introduce independent fusion switches or flags for intrinsic graph properties such as main/MTP index sharing.
- When an option is removed, remove its reads, aliases, structured configuration fields, help text, examples, and special validation or migration handling unless compatibility handling is explicitly requested. Use the remaining supported options for policies such as the speculative draft budget.
- Avoid redundant state, configuration snapshots, and attributes that can be derived from the graph or existing configuration. If extra state is necessary, explain the concrete behavior that requires it and test that behavior.
- Build from the separated operations supplied by the existing model-loading interface, then combine compatible emitted operations where appropriate. Preserve adapters, packed weights, exclusions, independent precision policies, and unsupported-provider paths rather than changing their semantics to enable an optimization.

## Graph Construction and Export

- Reject exporters that construct an independent model from scratch through ONNX helper APIs or direct ONNX IR model/graph construction. Every exported model must inherit from the base `Model` class and use the existing graph initialization, node/value emission, initializer, quantization, and serialization infrastructure. Put any necessary shared infrastructure in `Model` rather than duplicating it in a new exporter.
- Reject post-export graph modification or splitting as a way to implement a model feature. Emit the required components, bindings, index-sharing behavior, and shared weights during model construction using the existing builder infrastructure. Do not add standalone graph-surgery or other side scripts under `src/python/py/models/`; keep export logic in the owning builder or loader.
- Use ONNX IR APIs inside the shared `Model` infrastructure for graph and tensor emission, attributes, and serialization instead of `onnx.helper` or `onnx.numpy_helper`. This is not permission to construct independent models or add post-export graph rewrites. Keep protobuf interoperability narrowly scoped to APIs that require it, such as the ORT quantizer, structural checker, or raw external-data validation in the source-format loader.
- Reuse existing normalization, constant, initializer, and node-building APIs before adding specialized helpers. Do not introduce a separate constant mechanism merely to support a new model; demonstrate any shape-inference requirement with an exported-model load test.
- Follow the existing node/output naming convention in wrapper methods. For example, `make_reshape` creates `{name}/output_0`; if a Reshape must directly produce a separately named graph output, use `make_node` with that explicit output instead of adding optional output-name behavior to the wrapper.
- Select weight packing for the deployment target through the explicit MatMulNBits and QMoE `weights_prepacked` settings, never through build-host GPU detection. Preserve documentation of the distinct integer, NVFP4, and FP8 layout contracts.
- Save each generated ONNX model's external weights in one separate, model-owned data file. Companion models must not depend on another model's external-data file. Preserve metadata, tensor contents, and any required runtime sharing records.

## Review and Validation

- Preserve existing function locations and ordering unless moving them is necessary for the requested architectural change. Do not relocate unchanged methods to the end of a file, reformat unrelated code, add unnecessary blank lines, or manually wrap Markdown prose.
- Inspect the relevant implementation and lower layers of a PR stack before declaring a change necessary or blocked. Resolve stack conflicts from the bottom upward, using merges rather than cherry-picking target updates.
- Verify the requested behavior rather than preserving the first implementation. Use numerical comparisons, emitted graph structure, provider compatibility, external-data save/reload tests, and edge cases as appropriate; distinguish new failures from a verified pre-existing baseline.
- If an existing mechanism appears insufficient, reproduce the limitation and explain why before introducing additional complexity. Prefer simplifying the implementation over defending abstractions or configuration added during an earlier iteration.
