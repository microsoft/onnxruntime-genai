/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

// NOTE: Typical usage is covered in GenerationTest.java so we are just filling test gaps here.
public class MultiModalProcessorTest {
    @Test
    public void testBatchEncodeDecode() throws GenAIException {
        try (Model model = new Model(TestUtils.testModelPath());
        MultiModalProcessor multiModalProcessor = new MultiModalProcessor(model)) {
            String[] inputs = new String[] {"This is a test", "This is another test"};
            Images image = new Images("\src\java\src\test\java\ai\onnxruntime\genai\landscape.jpg");
            Sequences processImages = MultiModalProcessor.processImages(inputs, image);
            String[] decoded = MultiModalProcessor.decode(processImages);

            assertEquals(inputs.length, decoded.length);
            for (int i = 0; i < inputs.length; i++) {
                assert inputs[i].equals(decoded[i]);
            }
        }
    }
}
