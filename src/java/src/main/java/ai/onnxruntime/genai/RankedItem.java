/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.util.Map;

/** One free-form candidate and its stable rank. */
public final class RankedItem {
  public final long rank;
  public final String key;
  public final Object value;
  public final double probability;

  RankedItem(Map<String, Object> item) {
    rank = ((Long) item.get("rank")).longValue();
    key = (String) item.get("key");
    value = item.get("value");
    probability = ((Double) item.get("probability")).doubleValue();
  }
}
