/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.util.LinkedHashMap;
import java.util.Map;

/** One typed question in a structured non-generative request. */
public final class StructuredQuestion {
  private final String type;
  private final Object instructions;
  private final Object criteria;

  public StructuredQuestion(String type, Object instructions) {
    this(type, instructions, null);
  }

  public StructuredQuestion(String type, Object instructions, Object criteria) {
    if (type == null || type.isEmpty()) {
      throw new IllegalArgumentException("Question type is required");
    }
    StructuredValues.validate(instructions);
    StructuredValues.validate(criteria);
    this.type = type;
    this.instructions = instructions;
    this.criteria = criteria;
  }

  Map<String, Object> toMap() {
    Map<String, Object> value = new LinkedHashMap<>();
    value.put("type", type);
    value.put("instructions", instructions);
    value.put("criteria", criteria);
    return value;
  }
}
