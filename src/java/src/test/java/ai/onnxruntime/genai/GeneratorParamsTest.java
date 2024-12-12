/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

// NOTE: Typical usage is covered in GenerationTest.java so we are just filling test gaps here.
public class GeneratorParamsTest {
  @Test
  public void testValidSearchOption() throws GenAIException {
    // test setting an invalid search option throws a GenAIException
    try (SimpleGenAI generator = new SimpleGenAI(TestUtils.tinyGpt2ModelPath());
        GeneratorParams params = generator.createGeneratorParams(); ) {
      params.setSearchOption("early_stopping", true); // boolean
      params.setSearchOption("max_length", 20); // number
    }
  }

  @Test
  public void testInvalidSearchOption() throws GenAIException {
    // test setting an invalid search option throws a GenAIException
    try (SimpleGenAI generator = new SimpleGenAI(TestUtils.tinyGpt2ModelPath());
        GeneratorParams params = generator.createGeneratorParams(); ) {
      assertThrows(GenAIException.class, () -> params.setSearchOption("invalid", true));
    }
  }
}
