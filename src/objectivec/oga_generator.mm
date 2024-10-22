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

- (NSNumber*)isDoneWithError:(NSError**)error {
  try {
    return [NSNumber numberWithBool:_generator->IsDone()];
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (BOOL)computeLogitsWithError:(NSError**)error {
  try {
    _generator->ComputeLogits();
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)generateNextTokenWithError:(NSError**)error {
  try {
    _generator->GenerateNextToken();
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (OGATensor*)getOutput:(NSString*)name {
  std::unique_ptr<OgaTensor> output = _generator->GetOutput([name UTF8String]);
  return [[OGATensor alloc] initWithCXXPointer:std::move(output)];
}

- (nullable OGAInt32Span*)sequenceAtIndex:(size_t)index
                                    error:(NSError**)error {
  try {
    size_t sequenceLength = _generator->GetSequenceCount(index);
    const int32_t* data = _generator->GetSequenceData(index);
    return [[OGAInt32Span alloc] initWithDataPointer:data size:sequenceLength];
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

@end
