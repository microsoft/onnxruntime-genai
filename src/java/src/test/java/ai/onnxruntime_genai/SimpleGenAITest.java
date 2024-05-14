/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.logging.Logger;
import org.junit.jupiter.api.Test;

public class SimpleGenAITest implements SimpleGenAI.TokenUpdateListener {
  private static final Logger logger = Logger.getLogger(SimpleGenAITest.class.getName());

  private static final String testModelPath() {
    // TODO: Figure out the settings for this model. At the very least, max_length needs to be 20.
    // URL url = SimpleGenAI.class.getResource("/hf-internal-testing/tiny-random-gpt2-fp32");
    // File f = new File(url.getFile());
    // return f.getPath();
    System.setProperty(
        "onnxruntime_genai.native.path",
        "D:\\src\\github\\ort.genai\\build\\Windows\\Debug\\src\\java\\native-jni\\ai\\onnxruntime_genai\\native\\win-x64");
    // return
    // "D:/src/github/ort.genai/src/java/build/resources/test/hf-internal-testing/tiny-random-gpt2-fp32";
    return "D:\\src\\github\\ort.genai\\examples\\python\\example-models\\phi2-int4-cpu";
  }

  @Override
  public void onTokenGenerate(String token) {
    logger.info("Token: " + token);
  }

  @Test
  public void testUsageNoListener() throws GenAIException {
    SimpleGenAI generator = new SimpleGenAI(testModelPath());
    GeneratorParams params = generator.createGeneratorParams("What's 6 times 7?");
    // test we can set a valid search option from here.
    // createGeneratorParams sets a number option for the max_length so test a boolean here for
    // completeness
    params.setSearchOption("early_stopping", false);

    String results = generator.generate(params, null);
    System.out.println("Results: " + results);
    logger.info("Results: " + results);
  }

  @Test
  public void testUsageWithListener() throws GenAIException {
    SimpleGenAI generator = new SimpleGenAI(testModelPath());
    GeneratorParams params = generator.createGeneratorParams("What's 6 times 7?");
    String results = generator.generate(params, this);
    System.out.println("Results: " + results);
    logger.info("Results: " + results);
  }

  @Test
  public void testInvalidSearchOption() throws GenAIException {
    SimpleGenAI generator = new SimpleGenAI(testModelPath());
    GeneratorParams params = generator.createGeneratorParams("This is a testing prompt");
    assertThrows(GenAIException.class, () -> params.setSearchOption("invalid", true));
  }
}
