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

- (BOOL)isDoneWithError:(NSError**)error {
  try {
    return _generator->IsDone();
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)appendTokenSequences:(OGASequences*)sequences error:(NSError**)error {
  try {
    _generator->AppendTokenSequences([sequences CXXAPIOgaSequences]);
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)appendTokens:(NSArray<NSNumber*>*)tokens error:(NSError**)error {
  std::vector<int32_t> cxxTokens;
  for (NSNumber* object in tokens) {
    cxxTokens.push_back([object intValue]);
  }

  try {
    _generator->AppendTokens(cxxTokens.data(), cxxTokens.size());
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)rewindTo:(NSUInteger)length error:(NSError**)error {
  try {
    _generator->RewindTo(length);
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

- (nullable OGATensor*)getOutput:(NSString*)name
                           error:(NSError**)error {
  try {
    std::unique_ptr<OgaTensor> output = _generator->GetOutput([name UTF8String]);
    return [[OGATensor alloc] initWithCXXPointer:std::move(output)];
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
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

+ (void)shutdown {
  OgaShutdown();
}
@end
