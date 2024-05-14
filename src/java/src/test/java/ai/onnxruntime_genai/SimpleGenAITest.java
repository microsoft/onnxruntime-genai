/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.logging.Logger;
import org.junit.jupiter.api.Test;

public class SimpleGenAITest {
  private static final Logger logger = Logger.getLogger(SimpleGenAITest.class.getName());

  @Test
  public void testUsage() throws GenAIException {
    SimpleGenAI generator =
        new SimpleGenAI("test_models/hf-internal-testing/tiny-random-gpt2-fp32");
    GeneratorParams params = generator.createGeneratorParams("This is a testing prompt");
    params.setSearchOption("early_stopping", true);
    params.setSearchOption("whatthe", 123);
    assertFalse(true);
  }

  @Test
  public void testInvalidSearchOption() throws GenAIException {
    SimpleGenAI generator =
        new SimpleGenAI("test_models/hf-internal-testing/tiny-random-gpt2-fp32");
    GeneratorParams params = generator.createGeneratorParams("This is a testing prompt");
    assertThrows(GenAIException.class, () -> params.setSearchOption("invalid", true));
    assertFalse(true);
  }
}
