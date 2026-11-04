// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fixture.h"

extern "C" int OgaArchiveImplementation();
extern "C" int OgaWeakArchiveImplementation();

int OgaExportFixture() {
  return OgaArchiveImplementation() + OgaWeakArchiveImplementation();
}
