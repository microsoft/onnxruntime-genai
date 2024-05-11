/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime_genai;

/** An exception which contains the error message and code produced by the native layer. */
public class GenAIException extends Exception {
  public GenAIException(String message) {
    super(message);
  }
}
