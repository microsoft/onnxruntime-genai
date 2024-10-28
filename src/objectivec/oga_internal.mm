// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "oga_internal.h"

@implementation OGAInt32Span {
  const int32_t* _ptr;
  size_t _size;
}

- (nullable instancetype)initWithDataPointer:(const int32_t*)pointer size:(size_t)size {
  if ((self = [super init]) == nil) {
    return nil;
  }

  _ptr = pointer;
  _size = size;
  return self;
}

- (const int32_t*)pointer {
  return _ptr;
}

- (size_t)size {
  return _size;
}

- (int32_t)lastElementWithError:(NSError**)error {
  if (_size == 0) {
    if (error != nil) {
      NSDictionary *errorDictionary = @{NSLocalizedDescriptionKey : @"The size of this span is invalid"};
      *error = [[NSError alloc] initWithDomain:kOgaErrorDomain code:-1 userInfo:errorDictionary];
    }
    return -1;
  }
  return *(_ptr + (_size - 1));
}

@end
