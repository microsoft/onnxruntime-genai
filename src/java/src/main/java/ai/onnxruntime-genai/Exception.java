/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime-genai;

/** An exception which contains the error message and code produced by the native layer. */
public class GenAIException extends Exception {
  /**
   * Creates an GenAIException with the specified message.
   *
   * @param message The message to use.
   */
  public GenAIException(String message) {
    super(message);
  }
}
