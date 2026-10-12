/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.util.LinkedHashMap;
import java.util.Map;

/** Structured state and typed questions accepted by ranking and decision sessions. */
public final class StructuredRequest {
  private final Object state;
  private final Map<String, StructuredQuestion> questions;
  private final Float temperature;

  public StructuredRequest(Object state, Map<String, StructuredQuestion> questions) {
    this(state, questions, null);
  }

  public StructuredRequest(
      Object state, Map<String, StructuredQuestion> questions, Float temperature) {
    if (questions == null) {
      throw new NullPointerException("questions");
    }
    StructuredValues.validate(state);
    this.state = state;
    this.questions = new LinkedHashMap<>(questions);
    this.temperature = temperature;
  }

  Object state() {
    return state;
  }

  Float temperature() {
    return temperature;
  }

  Map<String, Object> nativeQuestions() {
    Map<String, Object> result = new LinkedHashMap<>();
    for (Map.Entry<String, StructuredQuestion> entry : questions.entrySet()) {
      if (entry.getKey() == null || entry.getKey().isEmpty() || entry.getValue() == null) {
        throw new IllegalArgumentException("Question ids and values are required");
      }
      result.put(entry.getKey(), entry.getValue().toMap());
    }
    return result;
  }
}
