// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"
#import "oga_internal.h"
#import "ort_genai_objc.h"

@implementation OGAGenerator {
  std::unique_ptr<OgaGenerator> _generator;
}

- (nullable instancetype)initWithModel:(OGAModel*)model
                   params:(OGAGeneratorParams*)params
                    error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    _generator = OgaGenerator::Create([model CXXAPIOgaModel], [params CXXAPIOgaGeneratorParams]);
    return self;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (BOOL)isDone {
  return _generator->IsDone();
}

- (void)computeLogits {
  _generator->ComputeLogits();
}

- (void)generateNextToken {
  _generator->GenerateNextToken();
}

- (nullable OGAInt32Span*)sequenceAtIndex:(size_t)index {
  try {
    size_t sequenceLength = _generator->GetSequenceCount(index);
    const int32_t* data = _generator->GetSequenceData(index);
    return [[OGAInt32Span alloc] initWithRawPointer:data size:sequenceLength];
  } catch (std::exception) {
    return nil;
  }
}

@end
