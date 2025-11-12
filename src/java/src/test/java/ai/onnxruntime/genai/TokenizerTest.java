/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

// NOTE: Typical usage is covered in GenerationTest.java so we are just filling test gaps here.
public class TokenizerTest {
  @SuppressWarnings("unused") // Used in EnabledIf
  private static boolean havePhi2() {
    return TestUtils.phi2ModelPath() != null;
  }

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

  @Test
  @EnabledIf("havePhi2")
  public void testGetBosTokenId() throws GenAIException {
    try (Model model = new Model(TestUtils.phi2ModelPath());
        Tokenizer tokenizer = new Tokenizer(model)) {
      int bosTokenId = tokenizer.getBosTokenId();
      assertTrue(bosTokenId == 50256, "BOS token ID should be 50256");
    }
  }

  @Test
  @EnabledIf("havePhi2")
  public void testGetEosTokenIds() throws GenAIException {
    try (Model model = new Model(TestUtils.phi2ModelPath());
        Tokenizer tokenizer = new Tokenizer(model)) {
      int[] eosTokenIds = tokenizer.getEosTokenIds();
      assertNotNull(eosTokenIds, "EOS token IDs should not be null");
      assertTrue(eosTokenIds.length == 1, "Should have exactly one EOS token sequence");

      if (eosTokenIds.length > 0) {
        assertTrue(eosTokenIds[0] == 50256, "First EOS token ID should be 50256");
      }
    }
  }

  @Test
  @EnabledIf("havePhi2")
  public void testGetPadTokenId() throws GenAIException {
    try (Model model = new Model(TestUtils.phi2ModelPath());
        Tokenizer tokenizer = new Tokenizer(model)) {
      int padTokenId = tokenizer.getPadTokenId();
      assertTrue(padTokenId == 50256, "Pad token ID should be 50256");
    }
  }

  @Test
  @EnabledIf("havePhi2")
  public void testApplyChatTemplate() throws GenAIException {
    // We load the phi-2 model just to get a tokenizer (phi-2 does not have a chat template)
    try (Model model = new Model(TestUtils.phi2ModelPath());
        Tokenizer tokenizer = new Tokenizer(model)) {
      // Testing phi-4-mini chat template
      String messagesJson =
          "[\n"
              + "  {\n"
              + "    \"role\": \"system\",\n"
              + "    \"content\": \"You are a helpful assistant.\",\n"
              + "    \"tools\": \"[{\\\"name\\\": \\\"calculate_sum\\\", \\\"description\\\": \\\"Calculate the sum of two numbers.\\\", \\\"parameters\\\": {\\\"a\\\": {\\\"type\\\": \\\"int\\\"}, \\\"b\\\": {\\\"type\\\": \\\"int\\\"}}}]\"\n"
              + "  },\n"
              + "  {\n"
              + "    \"role\": \"user\",\n"
              + "    \"content\": \"How do I add two numbers?\"\n"
              + "  },\n"
              + "  {\n"
              + "    \"role\": \"assistant\",\n"
              + "    \"content\": \"You can add numbers by using the '+' operator.\"\n"
              + "  }\n"
              + "]";

      String chatTemplate =
          "{% for message in messages %}{% if message['role']"
              + " == 'system' and 'tools' in message and message['tools']"
              + " is not none %}{{ '<|' + message['role'] + '|>' + message['content']"
              + " + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}"
              + "{% else %}{{ '<|' + message['role'] + '|>' + message['content']"
              + " + '<|end|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}"
              + "{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %}";

      // From HuggingFace Python output for 'microsoft/Phi-4-mini-instruct'
      String expectedOutput =
          "<|system|>You are a helpful assistant.<|tool|>[{\"name\": \"calculate_sum\", \"description\": \"Calculate the sum of two numbers.\", \"parameters\": {\"a\": {\"type\": \"int\"}, \"b\": {\"type\": \"int\"}}}]<|/tool|><|end|><|user|>"
              + "How do I add two numbers?<|end|><|assistant|>You can add numbers by using the \"+\" operator.<|end|><|assistant|>";

      String result = tokenizer.applyChatTemplate(chatTemplate, messagesJson, null, true);
      assertEquals(expectedOutput, result, "Chat template output should match expected result");
    }
  }

  @Test
  @EnabledIf("havePhi2")
  public void testUpdateOptionsWithMap() throws GenAIException {
    try (Model model = new Model(TestUtils.phi2ModelPath());
        Tokenizer tokenizer = new Tokenizer(model)) {
      // Test with valid options map
      Map<String, String> options = new HashMap<>();
      options.put("add_special_tokens", "true");
      options.put("skip_special_tokens", "true");

      // This should not throw an exception
      tokenizer.updateOptions(options);
    }
  }
}
