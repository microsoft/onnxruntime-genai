/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import org.junit.jupiter.api.Test;

public class TensorTest {

  @Test
  public void testAddTensorInput() throws GenAIException {
    // test setting an invalid search option throws a GenAIException
    try (SimpleGenAI generator = new SimpleGenAI(TestUtils.tinyGpt2ModelPath());
        GeneratorParams params = generator.createGeneratorParams(); ) {
      long[] shape = {2, 2};
      Tensor.ElementType elementType = Tensor.ElementType.float32;
      ByteBuffer data = ByteBuffer.allocateDirect(4 * Float.BYTES).order(ByteOrder.nativeOrder());

      FloatBuffer floatBuffer = data.asFloatBuffer();
      floatBuffer.put(new float[] {1.0f, 2.0f, 3.0f, 4.0f});
      try (Tensor tensor = new Tensor(data, shape, elementType)) {
        // no error on setting.
        // assuming there's an error on execution if an invalid input has been provided so the user
        // is aware of the issue
        params.setInput("unknown_value", tensor);
      }
    }
  }

  @Test
  public void testInvalidParams() throws GenAIException {
    // ByteBuffer that is not directly allocated
    long[] shape = {2, 2};
    Tensor.ElementType elementType = Tensor.ElementType.float32;
    ByteBuffer indirect_data = ByteBuffer.allocate(4 * Float.BYTES);
    assertThrows(GenAIException.class, () -> new Tensor(indirect_data, shape, elementType));

    // missing data
    assertThrows(GenAIException.class, () -> new Tensor(null, shape, elementType));

    ByteBuffer data = ByteBuffer.allocateDirect(4 * Float.BYTES).order(ByteOrder.nativeOrder());

    // missing shape
    assertThrows(GenAIException.class, () -> new Tensor(data, null, elementType));

    // undefined data type
    assertThrows(GenAIException.class, () -> new Tensor(data, shape, Tensor.ElementType.undefined));
  }
}
