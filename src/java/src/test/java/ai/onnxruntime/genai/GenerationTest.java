/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.util.function.Consumer;
import java.util.logging.Logger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

// Test the overall generation.
// Uses SimpleGenAI with phi-2 (if available) for text -> text generation.
// Uses the HF test model with pre-defined input tokens for token -> token generation
//
// This indirectly tests the majority of the bindings. Any gaps are covered in the class specific
// tests.
public class GenerationTest {
  private static final Logger logger = Logger.getLogger(GenerationTest.class.getName());

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

  @Test
  @EnabledIf("havePhi2")
  public void testUsageNoListener() throws GenAIException {
    SimpleGenAI generator = new SimpleGenAI(phi2ModelPath());
    GeneratorParams params = generator.createGeneratorParams();

    String result = generator.generate(params, "What's 6 times 7?", null);
    logger.info("Result: " + result);
    assertTrue(result.indexOf("Answer: 42") != -1);
  }

  @Test
  @EnabledIf("havePhi2")
  public void testUsageWithListener() throws GenAIException {
    SimpleGenAI generator = new SimpleGenAI(phi2ModelPath());
    GeneratorParams params = generator.createGeneratorParams();
    Consumer<String> listener = token -> logger.info("onTokenGenerate: " + token);
    String result = generator.generate(params, "What's 6 times 7?", listener);

    logger.info("Result: " + result);
    assertTrue(result.indexOf("Answer: 42") != -1);
  }

  @Test
  public void testWithInputIds() throws GenAIException {
    // test using the HF model. input id values must be < 1000 so we use manually created input.
    // Input/expected output copied from the C# unit tests
    Config config = new Config(TestUtils.testModelPath());
    Model model = new Model(config);
    GeneratorParams params = new GeneratorParams(model);
    int batchSize = 2;
    int sequenceLength = 4;
    int maxLength = 10;
    int[] inputIDs =
        new int[] {
          0, 0, 0, 52,
          0, 0, 195, 731
        };

    params.setSearchOption("max_length", maxLength);
    params.setSearchOption("batch_size", batchSize);

    int[] expectedOutput =
        new int[] {
          0, 0, 0, 52, 204, 204, 204, 204, 204, 204,
          0, 0, 195, 731, 731, 114, 114, 114, 114, 114
        };

    Generator generator = new Generator(model, params);
    generator.appendTokens(inputIDs);
    while (!generator.isDone()) {
      generator.generateNextToken();
    }
    
    for (int i = 0; i < batchSize; i++) {
      int[] outputIds = generator.getSequence(i);
      for (int j = 0; j < maxLength; j++) {
        assertEquals(outputIds[j], expectedOutput[i * maxLength + j]);
      }
    }
  }
}
