/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import static org.junit.jupiter.api.Assertions.assertNotNull;

import java.util.logging.Logger;
import org.junit.jupiter.api.Test;

// NOTE: Typical usage is covered in GenerationTest.java so we are just filling test gaps here.
public class StableDiffusionTest {
  private static final Logger logger = Logger.getLogger(StableDiffusionTest.class.getName());

  @Test
  public void testGenerateImage() throws GenAIException {
    try (Model model =
            new Model(
                "C:\\Users\\yangselena\\onnxruntime-genai\\onnxruntime-genai\\test\\test_models\\sd");
        ImageGeneratorParams imageGeneratorParams = new ImageGeneratorParams(model)) {
      imageGeneratorParams.setPrompts("A couple walking in the park", null);

      try (Tensor result = Generator.generateImage(model, imageGeneratorParams)) {
        assertNotNull(result);
      }
    }
  }
}
