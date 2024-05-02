/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime-genai;

/** An exception which contains the error message and code produced by the native onnxruntime. */
public class OrtGenAIException extends Exception {
  /**
   * Creates an OrtGenAIException with the specified message.
   *
   * @param message The message to use.
   */
  public OrtGenAIException(String message) {
    super(message);
  }
}
