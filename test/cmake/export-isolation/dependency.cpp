// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

extern "C" int OgaArchiveImplementation() {
  return 20;
}

extern "C" __attribute__((weak)) int OgaWeakArchiveImplementation() {
  return 22;
}
