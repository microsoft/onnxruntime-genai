---
applyTo: "src/python/py/models/**"
---

# Python Model Builder - Copilot Instructions

When generating or reviewing code in the Python Model Builder (`src/python/py/models/`), evaluate all changes against the criteria defined in the following files:

- [README.md](../../src/python/py/models/README.md)
- [DESIGN.md](../../src/python/py/models/DESIGN.md)
- [LOADERS.md](../../src/python/py/models/loaders/LOADERS.md)
- [QUANTIZATION.md](../../src/python/py/models/quantization/QUANTIZATION.md)

Read both documents to understand the intended usage, supported models, design principles, and architectural constraints before suggesting or reviewing any code changes in this area.

## Code Style Guidelines

- Lots of AI-generated code will be proposed in pull requests. Scrutinize every single line of code related to the model builder in every PR very heavily. Most of the generated code is not actually necessary if designed correctly. Most PRs do not get this right and most proposers do not understand this.
- When a node is inserted into the model, prefer using `self.make_op_name` as the wrapper method for `self.make_node` + `self.make_value` calls.
- Do NOT use global variables in any logic. Any static variables should be declared inside a standard Python method (e.g. `init` methods, a regular Python function, etc).
- Ignore any CodeQL warnings about how an __init__ method calls an overridden method. These warnings are false positives and can be safely ignored. The warning message is: "this call to ABC in an initialization method is overwritten by XYZ".
- Find ways to reduce code duplication by reusing existing functionality and implementing common patterns.
- Discover ways to leverage the use of shared code in the base classes to avoid code duplication and improve maintainability.
- For any new `extra_options` that are added, make sure that they are documented in both `README.md` and `builder.py`. In `README.md`, there should be a description of the option and its purpose. There should be a usage example thereafter showing how to use the option when calling the model builder from wheel or from source. In `builder.py`, there should be a description of the option and its purpose, its default value, and any possible values. Any constraints or limitations should also be documented. Make sure the documentation is consistent across both files.
- Always raise an issue if `staticmethod` is used in any method added EXCEPT if the method it is attached to is a `from_pretrained` method. The use of `staticmethod` should otherwise be avoided. If you find a case where it is used, raise an issue to find a better way to implement the functionality without using `staticmethod`.
- Always raise an issue if a variable or method is named with a leading underscore. The use of leading underscores should be avoided. If you find a case where it is used, raise an issue to find a better way to implement the functionality without using leading underscores.
- Always raise an issue if there is a lot of complicated logic in a method. If you find a case where there is a lot of complicated logic, raise an issue to find a better way to implement the functionality without using complicated logic.
- Avoid putting customized logic inside the base class constructor (i.e. avoid putting inside `Model.__init__`). Prefer using `make_description_init` methods that wrap that customized logic in a standalone `init` method.
- When many new variables are created inside the model builder (e.g. many variables to control how an MoE subgraph looks, how quantized KV caches work, etc.), you should move those variables into settings stored within `_attrs` dictionaries. Each setting should be documented, explain its purpose, and mention when it's used.
- Ensure that existing dispatcher methods only continue to dispatch across branches. For example, `make_matmul_op`, `make_attention_op`, and `make_mlp_op` should only dispatch across branches. If you find a case where a dispatcher method is doing more than dispatching, raise an issue to find a better way to implement the functionality without using dispatcher methods for more than dispatching.
- As new data types and new quantization schemas are introduced, make sure that the core feature of the model builder still remains. The logic in `base.py` + each model class should be generic enough. It is the job of the parser class from the new data type or new quantization schema to convert the model into a form that the model builder can understand. The core model builder logic should not be modified directly to support new data types or new quantization schemas. If you find a case where the model builder is being modified to support new data types or new quantization schemas, raise an issue to find a better way to implement the functionality without modifying the model builder.
- Make sure that any bit unpacking and repacking logic is done when the quantized model is parsed through the parser class. The model builder should not need to be modified to support new quantization schemas. If you find a case where the model builder is being modified to support new quantization schemas, raise an issue to find a better way to implement the functionality without modifying the model builder.
- Make sure all new methods are added inside the `Model` class. If you find a case where a new method is being added outside of the `Model` class, raise an issue to find a better way to implement the functionality without adding new methods outside of the `Model` class.
