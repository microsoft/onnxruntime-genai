// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"

NS_ASSUME_NONNULL_BEGIN

NSString* const kOgaErrorDomain = @"onnxruntime-genai";
const int kOgaErrorCode = 0x0A;

void OGASaveCodeAndDescriptionToError(int code, const char* descriptionCstr, NSError** error) {
  if (!error) return;

  NSString* description = [NSString stringWithCString:descriptionCstr
                                             encoding:NSUTF8StringEncoding];

  *error = [NSError errorWithDomain:kOgaErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey : description}];
}

void OGASaveCodeAndDescriptionToError(int code, NSString* description, NSError** error) {
  if (!error) return;

  *error = [NSError errorWithDomain:kOgaErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey : description}];
}

void OGASaveExceptionToError(const std::exception& e, NSError** error) {
  OGASaveCodeAndDescriptionToError(kOgaErrorCode, e.what(), error);
}

NS_ASSUME_NONNULL_END
