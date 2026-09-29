/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** Structured result returned by a ranking or decision session. */
public final class ModelResult {
  public final String model;
  public final List<ModelAnswer> answers;

  @SuppressWarnings("unchecked")
  ModelResult(Map<String, Object> value) {
    model = (String) value.get("model");
    List<ModelAnswer> converted = new ArrayList<>();
    for (Map<String, Object> answer : (List<Map<String, Object>>) value.get("answers")) {
      converted.add(new ModelAnswer(answer));
    }
    answers = Collections.unmodifiableList(converted);
  }
}
