/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

/**
 * A Java interface to the ONNX Runtime GenAI library.
 *
 * <p>There are two shared libraries required: <code>onnxruntime-genai</code> and <code>
 * onnxruntime-genai-jni
 * </code>. The loader is in ai.onnxruntime.genai.GenAI and the logic is in this order:
 *
 * <ol>
 *   <li>The user may signal to skip loading of a shared library using a property in the form <code>
 *       onnxruntime-genai.native.LIB_NAME.skip</code> with a value of <code>true</code>. This means
 *       the user has decided to load the library by some other means.
 *   <li>The user may specify an explicit location of all native library files using a property in
 *       the form <code>onnxruntime-genai.native.path</code>. This uses {java.lang.System#load}.
 *   <li>The user may specify an explicit location of the shared library file using a property in
 *       the form <code>onnxruntime-genai.native.LIB_NAME.path</code>. This uses
 *       {java.lang.System#load}.
 *   <li>The shared library is autodiscovered:
 *       <ol>
 *         <li>If the shared library is present in the classpath resources, load using {
 *             java.lang.System#load} via a temporary file. Ideally, this should be the default use
 *             case when adding JAR's/dependencies containing the shared libraries to your
 *             classpath.
 *         <li>If the shared library is not present in the classpath resources, then load using
 *             {java.lang.System#loadLibrary}, which usually looks elsewhere on the filesystem for
 *             the library. The semantics and behavior of that method are system/JVM dependent.
 *             Typically, the <code>java.library.path</code> property is used to specify the
 *             location of native libraries.
 *       </ol>
 * </ol>
 *
 * For troubleshooting, all shared library loading events are reported to Java logging at the level
 * FINE.
 */
package ai.onnxruntime.genai;
