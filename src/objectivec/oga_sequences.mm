// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"
#import "oga_internal.h"
#import "ort_genai_objc.h"

@implementation OGASequences {
  std::unique_ptr<OgaSequences> _sequences;
}

- (instancetype)initWithCXXPointer:(std::unique_ptr<OgaSequences>)ptr {
  _sequences = std::move(ptr);
  return self;
}

- (nullable instancetype)initWithError:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    _sequences = OgaSequences::Create();
    return self;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (size_t)getCountWithError:(NSError**)error {
  try {
    return _sequences->Count();
  }
  OGA_OBJC_API_IMPL_CATCH(error, size_t(-1))
}

- (nullable OGAInt32Span*)sequenceAtIndex:(size_t)index
                                    error:(NSError**)error {
  try {
    size_t sequenceLength = _sequences->SequenceCount(index);
    const int32_t* data = _sequences->SequenceData(index);
    return [[OGAInt32Span alloc] initWithDataPointer:data size:sequenceLength];
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (OgaSequences&)CXXAPIOgaSequences {
  return *(_sequences.get());
}

@end
