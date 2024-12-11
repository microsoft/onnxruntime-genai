/*
 * Copyright (c) Microsoft Corporation. All rights reserved. Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

/** An exception which contains the error message and code produced by the native layer. */
public final class GenAIException extends Exception {
  GenAIException(String message) {
    super(message);
  }

  GenAIException(String message, Exception innerException) {
    super(message, innerException);
  }
}
