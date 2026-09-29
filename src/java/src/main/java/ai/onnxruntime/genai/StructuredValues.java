/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.util.Map;

final class StructuredValues {
  static void validate(Object value) {
    if (value == null
        || value instanceof String
        || value instanceof Boolean
        || value instanceof Byte
        || value instanceof Short
        || value instanceof Integer
        || value instanceof Long
        || value instanceof Float
        || value instanceof Double) {
      return;
    }
    if (value instanceof Map) {
      for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
        if (!(entry.getKey() instanceof String)) {
          throw new IllegalArgumentException("Structured object keys must be strings");
        }
        validate(entry.getValue());
      }
      return;
    }
    if (value instanceof Iterable) {
      for (Object item : (Iterable<?>) value) {
        validate(item);
      }
      return;
    }
    throw new IllegalArgumentException(
        "Unsupported structured value type: " + value.getClass().getName());
  }

  private StructuredValues() {}
}
