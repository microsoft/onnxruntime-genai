/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;
import java.util.Optional;
import java.util.logging.Level;
import java.util.logging.Logger;

final class GenAI {
  private static final Logger logger = Logger.getLogger(GenAI.class.getName());

  /**
   * The name of the system property which when set gives the path on disk where the ONNX Runtime
   * native libraries are stored.
   */
  static final String GENAI_NATIVE_PATH = "onnxruntime-genai.native.path";

  /** The short name of the ONNX Runtime GenAI shared library */
  static final String GENAI_LIBRARY_NAME = "onnxruntime-genai";

  /** The short name of the ONNX Runtime GenAI JNI shared library */
  static final String GENAI_JNI_LIBRARY_NAME = "onnxruntime-genai-jni";

  /** The short name of the ONNX runtime shared library */
  static final String ONNXRUNTIME_LIBRARY_NAME = "onnxruntime";

  static final String ONNXRUNTIME_GENAI_RESOURCE_PATH = "/ai/onnxruntime/genai/native/";
  static final String ONNXRUNTIME_RESOURCE_PATH = "/ai/onnxruntime/native/";

  /** The value of the GENAI_NATIVE_PATH system property */
  private static String libraryDirPathProperty;

  /** The OS & CPU architecture string */
  private static final String OS_ARCH_STR = initOsArch();

  /** Have the native libraries been loaded */
  private static boolean loaded = false;

  /** The temp directory where native libraries are extracted */
  private static Path tempDirectory;

  static synchronized void init() throws IOException {
    if (loaded) {
      return;
    }

    tempDirectory = isAndroid() ? null : Files.createTempDirectory("onnxruntime-genai-java");

    try {
      libraryDirPathProperty = System.getProperty(GENAI_NATIVE_PATH);

      load(ONNXRUNTIME_LIBRARY_NAME, ONNXRUNTIME_RESOURCE_PATH); // ORT native
      load(GENAI_LIBRARY_NAME, ONNXRUNTIME_GENAI_RESOURCE_PATH); // ORT GenAI native
      load(GENAI_JNI_LIBRARY_NAME, ONNXRUNTIME_GENAI_RESOURCE_PATH); // GenAI JNI layer
      loaded = true;
    } finally {
      if (tempDirectory != null) {
        cleanUp(tempDirectory.toFile());
      }
    }
  }

  static native void shutdown();

  /* Computes and initializes OS_ARCH_STR (such as linux-x64) */
  private static String initOsArch() {
    String detectedOS = null;
    String os = System.getProperty("os.name", "generic").toLowerCase(Locale.ENGLISH);
    if (os.contains("mac") || os.contains("darwin")) {
      detectedOS = "osx";
    } else if (os.contains("win")) {
      detectedOS = "win";
    } else if (os.contains("nux")) {
      detectedOS = "linux";
    } else if (isAndroid()) {
      detectedOS = "android";
    } else {
      throw new IllegalStateException("Unsupported os:" + os);
    }

    String detectedArch = null;
    String arch = System.getProperty("os.arch", "generic").toLowerCase(Locale.ENGLISH);
    if (arch.startsWith("amd64") || arch.startsWith("x86_64")) {
      detectedArch = "x64";
    } else if (arch.startsWith("x86")) {
      // 32-bit x86 is not supported by the Java API
      detectedArch = "x86";
    } else if (arch.startsWith("aarch64")) {
      detectedArch = "aarch64";
    } else if (arch.startsWith("ppc64")) {
      detectedArch = "ppc64";
    } else if (isAndroid()) {
      detectedArch = arch;
    } else {
      throw new IllegalStateException("Unsupported arch:" + arch);
    }

    return detectedOS + '-' + detectedArch;
  }

  /**
   * Check if we're running on Android.
   *
   * @return True if the property java.vendor equals The Android Project, false otherwise.
   */
  static boolean isAndroid() {
    return System.getProperty("java.vendor", "generic").equals("The Android Project");
  }

  /**
   * Marks the file for delete on exit.
   *
   * @param file The file to remove.
   */
  private static void cleanUp(File file) {
    if (!file.exists()) {
      return;
    }

    logger.log(Level.FINE, "Deleting " + file + " on exit");
    file.deleteOnExit();
  }

  /**
   * Load a shared library by name.
   *
   * <p>If the library path is not specified via a system property then it attempts to extract the
   * library from the classpath before loading it.
   *
   * @param library The bare name of the library.
   * @throws IOException If the file failed to read or write.
   */
  private static void load(String library, String resourcePath) throws IOException {
    if (isAndroid()) {
      // On Android, we simply use System.loadLibrary.
      // We only need to load the JNI library as it will load the GenAI native library and ORT
      // native library
      // via the library's dependencies.
      if (library == GENAI_JNI_LIBRARY_NAME) {
        logger.log(Level.INFO, "Loading native library '" + library + "'");
        System.loadLibrary(library);
      }

      return;
    }

    // 1) The user may skip loading of this library:
    String skip = System.getProperty("onnxruntime-genai.native." + library + ".skip");
    if (Boolean.TRUE.toString().equalsIgnoreCase(skip)) {
      logger.log(Level.FINE, "Skipping load of native library '" + library + "'");
      return;
    }

    // Resolve the platform dependent library name.
    String libraryFileName = mapLibraryName(library);

    // 2) The user may explicitly specify the path to a directory containing all shared libraries:
    if (libraryDirPathProperty != null) {
      logger.log(
          Level.FINE,
          "Attempting to load native library '"
              + library
              + "' from specified path: "
              + libraryDirPathProperty);

      // TODO: Switch this to Path.of when the minimum Java version is 11.
      File libraryFile = Paths.get(libraryDirPathProperty, libraryFileName).toFile();
      String libraryFilePath = libraryFile.getAbsolutePath();
      if (!libraryFile.exists()) {
        throw new IOException("Native library '" + library + "' not found at " + libraryFilePath);
      }

      System.load(libraryFilePath);
      logger.log(Level.FINE, "Loaded native library '" + library + "' from specified path");
      return;
    }

    // 3) The user may explicitly specify the path to their shared library:
    String libraryPathProperty =
        System.getProperty("onnxruntime-genai.native." + library + ".path");
    if (libraryPathProperty != null) {
      logger.log(
          Level.FINE,
          "Attempting to load native library '"
              + library
              + "' from specified path: "
              + libraryPathProperty);
      File libraryFile = new File(libraryPathProperty);
      String libraryFilePath = libraryFile.getAbsolutePath();
      if (!libraryFile.exists()) {
        throw new IOException("Native library '" + library + "' not found at " + libraryFilePath);
      }

      System.load(libraryFilePath);
      logger.log(Level.FINE, "Loaded native library '" + library + "' from specified path");
      return;
    }

    // 4) try loading from resources or library path:
    Optional<File> extractedPath = extractFromResources(library, resourcePath);
    if (extractedPath.isPresent()) {
      // extracted library from resources
      System.load(extractedPath.get().getAbsolutePath());
      logger.log(Level.FINE, "Loaded native library '" + library + "' from resource path");
    } else {
      // failed to load library from resources, try to load it from the library path
      logger.log(
          Level.FINE, "Attempting to load native library '" + library + "' from library path");
      System.loadLibrary(library);
      logger.log(Level.FINE, "Loaded native library '" + library + "' from library path");
    }
  }

  /**
   * Extracts the library from the classpath resources. returns optional.empty if it failed to
   * extract or couldn't be found.
   *
   * @param library The library name
   * @param baseResourcePath The base resource path
   * @return An optional containing the file if it is successfully extracted, or an empty optional
   *     if it failed to extract or couldn't be found.
   */
  private static Optional<File> extractFromResources(String library, String baseResourcePath) {
    String libraryFileName = mapLibraryName(library);
    String resourcePath = baseResourcePath + OS_ARCH_STR + '/' + libraryFileName;
    File tempFile = tempDirectory.resolve(libraryFileName).toFile();

    try (InputStream is = GenAI.class.getResourceAsStream(resourcePath)) {
      if (is == null) {
        // Not found in classpath resources
        return Optional.empty();
      } else {
        // Found in classpath resources, load via temporary file
        logger.log(
            Level.FINE,
            "Attempting to load native library '"
                + library
                + "' from resource path "
                + resourcePath
                + " copying to "
                + tempFile);

        byte[] buffer = new byte[4096];
        int readBytes;
        try (FileOutputStream os = new FileOutputStream(tempFile)) {
          while ((readBytes = is.read(buffer)) != -1) {
            os.write(buffer, 0, readBytes);
          }
        }

        logger.log(Level.FINE, "Extracted native library '" + library + "' from resource path");
        return Optional.of(tempFile);
      }
    } catch (IOException e) {
      logger.log(
          Level.WARNING, "Failed to extract library '" + library + "' from the resources", e);
      return Optional.empty();
    } finally {
      cleanUp(tempFile);
    }
  }

  /**
   * Maps the library name into a platform dependent library filename. Converts macOS's "jnilib" to
   * "dylib" but otherwise is the same as System#mapLibraryName(String).
   *
   * @param library The library name
   * @return The library filename.
   */
  private static String mapLibraryName(String library) {
    return System.mapLibraryName(library).replace("jnilib", "dylib");
  }
}
