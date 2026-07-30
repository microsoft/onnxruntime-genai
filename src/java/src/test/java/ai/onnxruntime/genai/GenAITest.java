/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import org.junit.jupiter.api.Test;

public class GenAITest {
  @Test
  public void testTelemetryControl() {
    GenAI.setTelemetry(false);
    GenAI.setTelemetry(true);
  }
}
