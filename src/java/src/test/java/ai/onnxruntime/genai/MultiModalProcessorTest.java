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
            TokenizerStream stream = multiModalProcessor.createStream();
            GeneratorParams generatorParams = model.createGeneratorParams();
            String inputs = new String("This is a test");
            Images image = new Images("/src/java/src/test/java/ai/onnxruntime/genai/landscape.jpg");
            NamedTensors processed = multiModalProcessor.processImages(inputs, image);
            generatorParams.setInputs(processed);

            Generator generator = new Generator(model, generatorParams);

            String fullAnswer = new String();
            while (!generator.isDone()) {
                generator.generateNextToken();
                 
                int token = generator.getLastTokenInSequence(0);
                 
                fullAnswer += stream.decode(token);
            }
        }
    }
}
