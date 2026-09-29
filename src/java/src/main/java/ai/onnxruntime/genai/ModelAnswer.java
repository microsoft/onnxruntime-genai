/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** Typed answer returned by CLM or KEV. */
public final class ModelAnswer {
  public final String id;
  public final String type;
  public final Double noul;
  public final String choice;
  public final Double score;
  public final Double confidence;
  public final Map<String, Double> probabilities;
  public final Map<String, String> legend;

  @SuppressWarnings("unchecked")
  ModelAnswer(Map<String, Object> value) {
    id = (String) value.get("id");
    type = (String) value.get("type");
    noul = (Double) value.get("noul");
    choice = (String) value.get("choice");
    score = (Double) value.get("score");
    confidence = (Double) value.get("confidence");
    probabilities =
        Collections.unmodifiableMap(
            new LinkedHashMap<>((Map<String, Double>) value.get("probabilities")));
    legend =
        Collections.unmodifiableMap(
            new LinkedHashMap<>((Map<String, String>) value.get("legend")));
  }
}
