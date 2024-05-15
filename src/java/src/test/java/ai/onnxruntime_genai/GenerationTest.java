/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.util.logging.Logger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

// Test the overall generation.
// Uses SimpleGenAI with phi-2 (if available) for text -> text generation.
// Uses the HF test model with pre-defined input tokens for token -> token generation
//
// This indirectly tests the majority of the bindings. Any gaps are covered in the class specific
// tests.
public class GenerationTest implements SimpleGenAI.TokenUpdateListener {
  private static final Logger logger = Logger.getLogger(GenerationTest.class.getName());

  // use to debug locally if all the native libs (genai and onnxruntime) are in one directory
  // See /src/java/Debugging.md for more details.
  // Ensure the property is set before any tests run by calling during static initialization.
  // private static final boolean customPathRegistered = TestUtils.setLocalNativeLibraryPath();

  // phi-2 can be used in full end-to-end testing but needs to be manually downloaded.
  // it's also used this way in the C# unit tests.
  private static final String phi2ModelPath() {
    String repoRoot = TestUtils.getRepoRoot();
    File f = new File(repoRoot + "examples/python/example-models/phi2-int4-cpu");

    if (!f.exists()) {
      logger.warning("phi2 model not found at: " + f.getPath());
      logger.warning(
          "Please install as per https://github.com/microsoft/onnxruntime-genai/tree/rel-0.2.0/examples/csharp/HelloPhi2");
      return null;
    }

    return f.getPath();
  }

  @SuppressWarnings("unused") // Used in EnabledIf
  private static boolean havePhi2() {
    return phi2ModelPath() != null;
  }

  @Override
  public void onTokenGenerate(String token) {
    logger.info("onTokenGenerate: " + token);
  }

  @Test
  @EnabledIf("havePhi2")
  public void testUsageNoListener() throws GenAIException {
    SimpleGenAI generator = new SimpleGenAI(phi2ModelPath());
    GeneratorParams params = generator.createGeneratorParams("What's 6 times 7?");

    String result = generator.generate(params, null);
    logger.info("Result: " + result);
    assertTrue(result.indexOf("Answer: 42") != -1);
  }

  @Test
  @EnabledIf("havePhi2")
  public void testUsageWithListener() throws GenAIException {
    SimpleGenAI generator = new SimpleGenAI(phi2ModelPath());
    GeneratorParams params = generator.createGeneratorParams("What's 6 times 7?");

    String result = generator.generate(params, this);
    logger.info("Result: " + result);
    assertTrue(result.indexOf("Answer: 42") != -1);
  }

  @Test
  public void testWithInputIds() throws GenAIException {
    // test using the HF model. input id values must be < 1000 so we use manually created input.
    // Input/expected output copied from the C# unit tests
    Model model = new Model(TestUtils.testModelPath());
    GeneratorParams params = new GeneratorParams(model);
    int batchSize = 2;
    int sequenceLength = 4;
    int maxLength = 10;
    int[] inputIDs =
        new int[] {
          0, 0, 0, 52,
          0, 0, 195, 731
        };

    params.setInput(inputIDs, sequenceLength, batchSize);
    params.setSearchOption("max_length", maxLength);

    int[] expectedOutput =
        new int[] {
          0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
          0, 0, 195, 731, 731, 114, 114, 114, 114, 114
        };

    Sequences output = model.generate(params);
    assertEquals(output.numSequences(), batchSize);

    for (int i = 0; i < batchSize; i++) {
      int[] outputIds = output.getSequence(i);
      for (int j = 0; j < maxLength; j++) {
        assertEquals(outputIds[j], expectedOutput[i * maxLength + j]);
      }
    }
  }
}
