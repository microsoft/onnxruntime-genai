# Design of ONNX Runtime GenAI Model Builder

This document explains how the model builder is designed and how new contributions can be made. By following these guidelines, the model builder will remain lightweight, flexible, and useful after future changes.

## Contents

- [Design Principles](#design-principles)
  - [Simplicity](#simplicity)
  - [Efficiency](#efficiency)
  - [Modularity](#modularity)
  - [Compatibility](#compatibility)
- [Implementation Details](#implementation-details)
  - [`Model`](#model)
  - [Architecture Classes](#architecture-classes)
  - [`Attrs` Dictionaries](#attrs-dictionaries)
  - [`Make` Functions](#make-functions)
  - [Namespaces](#namespaces)
  - [Constants](#constants)
- [Contributing](#contributing)

## Design Principles

The model builder is designed under four main guiding principles: simplicity, efficiency, modularity, and compatibility.

### Simplicity

There should only be a few command-line arguments that are passed to the model builder. Additional arguments that are scenario-specific should not be created as separate command-line arguments. Instead, they should be passed to the `--extra_options` argument and the model builder can parse these custom arguments as key-value pairs.

There should only be a few dependencies on other packages. Additional package dependencies should not be added unless there is a strong justification.

These requirements are in place to make the model builder both lightweight and portable. The model builder has no dependencies on the ONNX Runtime GenAI package and it can be used as a standalone tool.

### Efficiency

Excluding the time it may take to download the original model, the model builder should produce optimized and quantized ONNX models within a few minutes. The traditional pipeline of converting models to ONNX, optimizing ONNX models, and quantizing ONNX models should be greatly accelerated.

### Modularity

Model architectures are defined in classes. By defining model architectures in this way, future models can leverage class inheritance to greatly reduce the amount of time and effort needed to onboard new model architectures. Optimizations and quantizations are also inherited from the base classes so there is no additional work needed to add those opportunities per model.

### Compatibility

The models produced by the model builder should directly work in ONNX Runtime GenAI and other solutions that use ONNX Runtime such as Hugging Face's Optimum. There should be no additional model modifications needed.

## Implementation Details

### `Model`

The `Model` base class holds all of the information for making models. It auto-determines optimizations and quantizations that can be applied (e.g. replace MultiHeadAttention with GroupQueryAttention). It also holds all important attributes and the many functions that make the final ONNX model.

After the final ONNX model is created, additional files are saved in the output folder to run with ONNX Runtime GenAI. These include the GenAI config and the pre-processing/post-processing files (e.g. tokenizer).

### Architecture Classes

Classes are the main abstraction within the model builder. Information that is specific to a particular model architecture is stored within the model architecture's class. Class inheritance is used to re-use existing code and reduce the time it takes to support a new model architecture. It also allows for any functions in the `Model` class to be overwritten as needed. Any changes in the `Model` class that add or modify support for optimization or quantization will benefit all models without any additional effort to support per model.

When adding support for a new model architecture, it is preferred to make changes in the `Model` class when possible. This allows the functions in `Model` to become more generic to multiple model architectures. An example of this is the changes made to support the `GemmaModel` class. If the changes are large or the model architecture is different than the currently supported options, it is easier to override or create new `Make` functions in the new model architecture's class. Examples of this include the changes made to support the `MistralModel` and `PhiModel` classes.

### `Attrs` Dictionaries

Attributes specific to operators and their scenario-specific variables are created and defined in the `Model` class under the operator's dictionary of variables. They should not be created and defined within functions. With this approach, information is globally accessible at all times and across all layers. This is particularly useful when creating an ONNX model where components are added or removed within it to support the user's requested scenario.

For example, a `LayerNorm` node can have two outputs during inference. One output is used as the input for the next nodes and one output is used as the input for the next `LayerNorm`. The next nodes may change depending on the layer and the next `LayerNorm` may happen after several nodes are added into the ONNX model. To handle these scenarios, the `LayerNorm` outputs need to be saved and updated globally.

New atributes and new scenario-specific variables (e.g. `add_offset` in `self.layernorm_attrs`, `normalize` in `self.embed_attrs`, etc.) should be created and defined in the `Model` class under the operator's dictionary of variables. For every new entry added, please add a comment that describes its purpose.

### `Make` Functions

`Make` functions are another main abstraction in the model builder. The base `Model` class holds many of the `Make` functions and their implementations. Each model architecture class inherits the `Make` functions and can override them as needed (e.g. `make_layer` in `PhiModel`).

Each `Make` function should take only a few basic parameters that are required for all situations when creating an op. If there are parameters that are scenario-specific, you can either initialize them in the `Attrs` dictionaries, create a new `Attrs` dictionary, or pass them as key-value parameters through `**kwargs` if already used to create an op.

A strong example of this is `make_attention`. There are no specified parameters besides the layer number, the attention module holding all input weights, and the root input to the attention layer because the remaining parameters will differ across attention ops. Similarly, when the attention op is created in `make_attention_op`, only the op's name is a specified parameter.

By using `**kwargs`, the `Make` function's signature is simplified and easy to read while still retaining the flexibility to pass any parameters needed for attention to it. With this approach, subclasses that inherit the `Model` class can also add or override any parameters as needed (e.g. `make_rotary_embedding` in `PhiModel`).

If a `Make` function is creating more than one op, please draw a diagram of the subgraph before its implementation. Diagrams help clarify the purpose of the function and visually show what the subgraph looks like afterwards.

Finally, creating many `Make` functions to abstract the entire model building process allows for fully customizing the process of building new models without having to modify other function signatures and worry about the fallout effects from those changes.

### Namespaces

Names are created using the following namespace format.

```
"/model/{unique description to reach location in the graph through multiple category names that are split up by '/'}/{op type}"
```

For example,

```
"/model/layers.0/attn/q_proj/MatMul"
```

can be read as

1. Look inside the model
2. Look at the 0th layer in the model
3. Look at the attention subgraph in the 0th layer in the model
4. Look at the Q projection path in the attention subgraph in the 0th layer in the model
5. Look at the MatMul node in the Q projection path in the attention subgraph in the 0th layer in the model

Thus, each node in the model has a unique name that describes its location in the model.

The output names for each node are created using the following namespace format.

```
"{node name}/output_{output index}"
```

Because each node's name is unique, the output names for all nodes are unique. The output names are based on the node names and the output index to avoid needing to create and keep track of unique output names as well.

For example,

```
"/model/layers.0/attn/o_proj/MatMul/output_0"
```

is the 0th output of the MatMul node in the output projection path in the attention subgraph in the 0th layer in the model. Therefore, whether the next node after this MatMul is an `Add` node or a `RotaryEmbedding` node, the next node just has to be aware of the parent node's name. The output index to use can be inferred based on the scenario in the graph.

This namespace format also allows components to be moved around or swapped in and out as needed. By knowing what the output node is in a component or layer, only its name needs to be provided to other components or layers. For example, a graph structure of `MLP --> layernorm` vs `MLP --> residual add` only has to differ by calling the `Make` function that creates the layernorm vs. the `Make` function that creates the residual add.

### Constants

Within the `Model` class, constants in ONNX are created through an automated process.

The `self.constants` holds the names of all constants that are used in the ONNX model. By keeping track of which constants are already created, the size of the final ONNX model can be reduced by re-using constants in this set.

In the traditional export process, a unique constant is created each time a node in the ONNX model needs one. The new constants may have the same values and dtypes but their stored names are different. Because of this, the ONNX model will store them both. Many of these stored constants are duplicates that do not need to be in the ONNX model.

To automate the creation process, the names of the constants are stored in the following namespace format.

```
"/model/constants/{onnx_dtype}/{num}"
```

- The `onnx_dtype` is the string representation of the TensorProto enum name used in ONNX to represent the constant's dtype (e.g. `INT64`). It is created using `self.to_str_dtype`.
- The `num` is the numerical constant. It can be a scalar or an array (e.g. `0` or `[1,2,3]`).

When a node is added to the ONNX model, its inputs are parsed to identify names in this format. If recognized, the input name (which is the name of a constant) is looked up in `self.constants`. If found, then the constant does not need to be added to the ONNX model again. If not found, the input name is parsed to obtain the necessary info to create the ONNX constant.

## Contributing

To contribute to the model builder, please ensure the following requirements are met.

1. Please verify your changes maintain the above information.

2. For new model architectures added, please verify that ONNX models produced by the model builder are valid. This can be done by loading the ONNX model into an ONNX Runtime inference session.

    Python example:

    ```py
    import onnxruntime as ort

    model_path = "path_to_onnx_model"
    ep = "name_of_desired_execution_provider"

    sess = ort.InferenceSession(model_path, providers=[ep])
    ```

    ONNX Runtime GenAI has additional CI tests to verify valid models when pull requests are opened.

3. Please ensure no attributes and no scenario-specific variables are created within a `Make` function or its function signature in `Model`. They should be defined in the `Attrs` dictionaries (preferably) or by overriding the function signature in the model architecture's class.

4. For adding new classes of model architectures (e.g. encoder-decoder, diffusion pipelines, multi-modal), please create a separate file that defines the new architecture class and have the base class in the new file inherit the `Model` class. While decoder-only architectures are currently the only ones supported, some of the logic in `Model` may be re-factored and put into a `DecoderModel` class in the future to help with this.

Please feel free to open an issue if you have any questions!