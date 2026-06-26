---
applyTo: "src/python/py/models/**"
---

# Python Model Builder - Copilot Instructions

When generating or reviewing code in the Python Model Builder (`src/python/py/models/`), evaluate all changes against the criteria defined in the following files:

- [README.md](../../src/python/py/models/README.md)
- [DESIGN.md](../../src/python/py/models/DESIGN.md)

Read both documents to understand the intended usage, supported models, design principles, and architectural constraints before suggesting or reviewing any code changes in this area.

## Code Style Guidelines

1. When a node is inserted into the model, prefer using `self.make_op_name` as the wrapper method for `self.make_node` + `self.make_value` calls.
2. Ignore any CodeQL warnings about how an __init__ method calls an overridden method. These warnings are false positives and can be safely ignored. The warning message is: "this call to ABC in an initialization method is overwritten by XYZ".
3. Find ways to reduce code duplication by reusing existing functionality and implementing common patterns.
4. Discover ways to leverage the use of shared code in the base classes to avoid code duplication and improve maintainability.
