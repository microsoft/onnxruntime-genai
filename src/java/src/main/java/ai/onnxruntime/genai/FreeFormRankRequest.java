/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.util.LinkedHashMap;
import java.util.Map;

/** Free-form ranking request. */
public final class FreeFormRankRequest {
  final Object state;
  final Object instructions;
  final Map<String, Object> candidates;
  final Float temperature;

  public FreeFormRankRequest(
      Object state, Object instructions, Map<String, Object> candidates, Float temperature) {
    if (candidates == null) {
      throw new NullPointerException("candidates");
    }
    StructuredValues.validate(state);
    StructuredValues.validate(instructions);
    StructuredValues.validate(candidates);
    this.state = state;
    this.instructions = instructions;
    this.candidates = new LinkedHashMap<>(candidates);
    this.temperature = temperature;
  }

  public FreeFormRankRequest(Object state, Object instructions, Map<String, Object> candidates) {
    this(state, instructions, candidates, null);
  }
}
