// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "oga_internal.h"

@implementation OGAInt32Span {
  const int32_t* _ptr;
  size_t _size;
}

- (nullable instancetype)initWithRawPointer:(const int32_t*)pointer size:(size_t)size {
  _ptr = pointer;
  _size = size;
  return [self init];
}

- (const int32_t*)pointer {
  return _ptr;
}

- (size_t)size {
  return _size;
}

- (int32_t)last {
  if (_size - 1 <= 0) {
    NSException* exception = [NSException exceptionWithName:@"onnxruntime-genai"
                                                     reason:@"The size of this span is invalid"
                                                   userInfo:nil];
    @throw exception;
  }
  return *(_ptr + (_size - 1));
}

@end

@implementation OGAInt64Span {
  const int64_t* _ptr;
  size_t _size;
}

- (nullable instancetype)initWithRawPointer:(const int64_t*)pointer size:(size_t)size {
  _ptr = pointer;
  _size = size;
  return [self init];
}

- (const int64_t*)pointer {
  return _ptr;
}

- (size_t)size {
  return _size;
}

- (int64_t)last {
  if (_size - 1 <= 0) {
    NSException* exception = [NSException exceptionWithName:@"onnxruntime-genai"
                                                     reason:@"The size of this span is invalid"
                                                   userInfo:nil];
    @throw exception;
  }
  return *(_ptr + (_size - 1));
}

@end
