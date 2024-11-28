// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"
#import "oga_internal.h"
#import "ort_genai_objc.h"

@implementation OGASequences {
  std::unique_ptr<OgaSequences> _sequences;
}

- (instancetype)initWithCXXPointer:(std::unique_ptr<OgaSequences>)ptr {
  if ((self = [super init]) == nil) {
    return nil;
  }

  _sequences = std::move(ptr);
  return self;
}

- (nullable instancetype)initWithError:(NSError**)error {
  try {
    self = [self initWithCXXPointer:OgaSequences::Create()];
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

- (nullable const int32_t*)sequenceDataAtIndex:(size_t)index error:(NSError**)error {
  try {
    return _sequences->SequenceData(index);
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (size_t)sequenceCountAtIndex:(size_t)index error:(NSError**)error {
  try {
    return _sequences->SequenceCount(index);
  }
  OGA_OBJC_API_IMPL_CATCH(error, size_t(-1))
}

- (OgaSequences&)CXXAPIOgaSequences {
  return *(_sequences.get());
}

@end
