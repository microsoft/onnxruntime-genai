/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.genai;

import android.content.ContentProvider;
import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.util.Log;

import com.microsoft.applications.events.HttpClient;

/** Initializes the Android transport and platform context used by native 1DS telemetry. */
public final class TelemetryInitializer extends ContentProvider {
  private static final String TAG = "ORTGenAITelemetry";
  private static HttpClient client;

  static synchronized boolean isInitialized() {
    return client != null;
  }

  @Override
  public boolean onCreate() {
    Context context = getContext();
    if (context == null) {
      return false;
    }

    try {
      System.loadLibrary("mat");
      synchronized (TelemetryInitializer.class) {
        if (client == null) {
          client = new HttpClient(context.getApplicationContext());
        }
      }
      return true;
    } catch (LinkageError | RuntimeException e) {
      Log.w(TAG, "Unable to initialize telemetry", e);
      return false;
    }
  }

  @Override
  public Cursor query(
      Uri uri, String[] projection, String selection, String[] selectionArgs, String sortOrder) {
    return null;
  }

  @Override
  public String getType(Uri uri) {
    return null;
  }

  @Override
  public Uri insert(Uri uri, ContentValues values) {
    return null;
  }

  @Override
  public int delete(Uri uri, String selection, String[] selectionArgs) {
    return 0;
  }

  @Override
  public int update(Uri uri, ContentValues values, String selection, String[] selectionArgs) {
    return 0;
  }
}
