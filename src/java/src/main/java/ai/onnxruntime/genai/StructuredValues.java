/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.util.IdentityHashMap;
import java.util.Map;

final class StructuredValues {
  private static final int MAX_DEPTH = 128;

  static void validate(Object value) {
    validate(value, new IdentityHashMap<>(), 0);
  }

  private static void validate(Object value, IdentityHashMap<Object, Boolean> active, int depth) {
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
      enter(value, active, depth);
      try {
        for (Map.Entry<?, ?> entry : ((Map<?, ?>) value).entrySet()) {
          if (!(entry.getKey() instanceof String)) {
            throw new IllegalArgumentException("Structured object keys must be strings");
          }
          validate(entry.getValue(), active, depth + 1);
        }
      } finally {
        active.remove(value);
      }
      return;
    }
    if (value instanceof Iterable) {
      enter(value, active, depth);
      try {
        for (Object item : (Iterable<?>) value) {
          validate(item, active, depth + 1);
        }
      } finally {
        active.remove(value);
      }
      return;
    }
    throw new IllegalArgumentException(
        "Unsupported structured value type: " + value.getClass().getName());
  }

  private static void enter(Object value, IdentityHashMap<Object, Boolean> active, int depth) {
    if (depth >= MAX_DEPTH) {
      throw new IllegalArgumentException(
          "Structured value exceeds the maximum nesting depth of " + MAX_DEPTH);
    }
    if (active.put(value, Boolean.TRUE) != null) {
      throw new IllegalArgumentException("Structured value contains a reference cycle");
    }
  }

  private StructuredValues() {}
}
