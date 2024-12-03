/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

// NOTE: Typical usage is covered in GenerationTest.java so we are just filling test gaps here.
public class TokenizerTest {
  @Test
  public void testBatchEncodeDecode() throws GenAIException {
    try (Model model = new Model(TestUtils.tinyGpt2ModelPath());
        Tokenizer tokenizer = new Tokenizer(model)) {
      String[] inputs = new String[] {"This is a test", "This is another test"};
      try (Sequences encoded = tokenizer.encodeBatch(inputs)) {
        String[] decoded = tokenizer.decodeBatch(encoded);

        assertEquals(inputs.length, decoded.length);
        for (int i = 0; i < inputs.length; i++) {
          assert inputs[i].equals(decoded[i]);
        }
      }
    }
  }
}
