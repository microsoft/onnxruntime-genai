/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** Result of free-form ranking. */
public final class RankingResult {
  public final String model;
  public final List<RankedItem> items;

  @SuppressWarnings("unchecked")
  RankingResult(Map<String, Object> value) {
    model = (String) value.get("model");
    List<RankedItem> converted = new ArrayList<>();
    for (Map<String, Object> item : (List<Map<String, Object>>) value.get("items")) {
      converted.add(new RankedItem(item));
    }
    items = Collections.unmodifiableList(converted);
  }
}
