// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#define OGA_EXPORT __attribute__((visibility("default")))
#define OGA_API_CALL

extern "C" OGA_EXPORT int OGA_API_CALL OgaExportFixture();
