// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#include <exception>

#import "cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

extern NSString* const kOgaErrorDomain;

void OGASaveCodeAndDescriptionToError(int code, const char* description, NSError** error);
void OGASaveCodeAndDescriptionToError(int code, NSString* description, NSError** error);
void OGASaveExceptionToError(const std::exception& e, NSError** error);

// helper macros to catch and handle C++ exceptions
#define OGA_OBJC_API_IMPL_CATCH(error, failure_return_value) \
  catch (const std::exception& e) {                          \
    OGASaveExceptionToError(e, (error));                     \
    return (failure_return_value);                           \
  }

#define OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error) OGA_OBJC_API_IMPL_CATCH(error, NO)

#define OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error) OGA_OBJC_API_IMPL_CATCH(error, nil)

NS_ASSUME_NONNULL_END
