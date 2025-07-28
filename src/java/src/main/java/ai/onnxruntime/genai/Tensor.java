/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Currently wraps an ORT Tensor. */
public final class Tensor implements AutoCloseable {
  private long nativeHandle = 0;
  private final ElementType elementType;
  private final long[] shape;

  // Buffer that owns the Tensor data.
  private ByteBuffer dataBuffer = null;

  // The values in this enum must match ONNX values
  // https://github.com/onnx/onnx/blob/159fa47b7c4d40e6d9740fcf14c36fff1d11ccd8/onnx/onnx.proto#L499-L544
  /** Element types that correspond to OnnxRuntime supported element types. */
  public enum ElementType {
    /** Undefined element type. */
    undefined,
    /** 32-bit IEEE 754 floating point. */
    float32,
    /** 8-bit unsigned integer. */
    uint8,
    /** 8-bit signed integer. */
    int8,
    /** 16-bit unsigned integer. */
    uint16,
    /** 16-bit signed integer. */
    int16,
    /** 32-bit signed integer. */
    int32,
    /** 64-bit signed integer. */
    int64,
    /** UTF-8 encoded string. */
    string,
    /** Boolean value (true or false). */
    bool,
    /** 16-bit IEEE 754 floating point (half precision). */
    float16,
    /** 64-bit IEEE 754 floating point (double precision). */
    float64,
    /** 32-bit unsigned integer. */
    uint32,
    /** 64-bit unsigned integer. */
    uint64,
  }

  /**
   * Constructs a Tensor with the given data, shape and element type.
   *
   * @param data The data for the Tensor. Must be a direct ByteBuffer with native byte order.
   * @param shape The shape of the Tensor.
   * @param elementType The type of elements in the Tensor.
   * @throws GenAIException If the call to the GenAI native API fails.
   */
  public Tensor(ByteBuffer data, long[] shape, ElementType elementType) throws GenAIException {
    if (data == null || shape == null || elementType == ElementType.undefined) {
      throw new GenAIException(
          "Invalid input. data and shape must be provided, and elementType must not be undefined.");
    }

    // for now require the buffer to be direct.
    // we could support non-direct but need to do an allocate and copy here.
    if (!data.isDirect()) {
      throw new GenAIException(
          "Tensor data must be direct. Allocate with ByteBuffer.allocateDirect");
    }

    // for now, require native byte order as the bytes will be used directly.
    if (data.order() != ByteOrder.nativeOrder()) {
      throw new GenAIException("Tensor data must have native byte order.");
    }

    this.elementType = elementType;
    this.shape = shape;
    this.dataBuffer = data; // save a reference so the owning buffer will stay around.

    nativeHandle = createTensor(data, shape, elementType.ordinal());
  }

  /**
   * Construct a Tensor from native handle.
   *
   * @param handle The native tensor handle.
   */
  Tensor(long handle) {
    nativeHandle = handle;
    elementType = ElementType.values()[getTensorType(handle)];
    shape = getTensorShape(handle);
  }

  /**
   * Get the element type.
   *
   * @return The element type.
   */
  public ElementType getType() {
    return this.elementType;
  }

  /**
   * Get the tensor shape.
   *
   * @return The tensor type.
   */
  public long[] getShape() {
    return this.shape;
  }

  @Override
  public void close() {
    if (nativeHandle != 0) {
      destroyTensor(nativeHandle);
      nativeHandle = 0;
    }
  }

  long nativeHandle() {
    return nativeHandle;
  }

  static {
    try {
      GenAI.init();
    } catch (Exception e) {
      throw new RuntimeException("Failed to load onnxruntime-genai native libraries", e);
    }
  }

  private native long createTensor(ByteBuffer data, long[] shape, int elementType)
      throws GenAIException;

  private native void destroyTensor(long tensorHandle);

  private native int getTensorType(long tensorHandle);

  private native long[] getTensorShape(long tensorHandle);
}
