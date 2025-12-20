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

- (BOOL)setInputs:(OGANamedTensors*)namedTensors error:(NSError**)error {
  try {
    _generator->SetInputs([namedTensors CXXAPIOgaNamedTensors]);
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)setModelInput:(NSString*)name tensor:(OGATensor*)tensor error:(NSError**)error {
  try {
    _generator->SetModelInput([name UTF8String], [tensor CXXAPIOgaTensor]);
    return YES;
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
  try {
    std::vector<int32_t> cxxTokens;
    for (NSNumber* object in tokens) {
      cxxTokens.push_back([object intValue]);
    }

    _generator->AppendTokens(cxxTokens.data(), cxxTokens.size());
    return YES;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error)
}

- (BOOL)rewindTo:(size_t)newLength error:(NSError**)error {
  try {
    _generator->RewindTo(newLength);
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

- (nullable OGATensor*)getInput:(NSString*)name error:(NSError**)error {
  try {
    std::unique_ptr<OgaTensor> input = _generator->GetInput([name UTF8String]);
    return [[OGATensor alloc] initWithCXXPointer:std::move(input)];
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable OGATensor*)getOutput:(NSString*)name error:(NSError**)error {
  try {
    std::unique_ptr<OgaTensor> output = _generator->GetOutput([name UTF8String]);
    return [[OGATensor alloc] initWithCXXPointer:std::move(output)];
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable const int32_t*)sequenceDataAtIndex:(size_t)index error:(NSError**)error {
  try {
    return _generator->GetSequenceData(index);
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (size_t)sequenceCountAtIndex:(size_t)index error:(NSError**)error {
  try {
    return _generator->GetSequenceCount(index);
  }
  OGA_OBJC_API_IMPL_CATCH(error, size_t(-1))
}

+ (void)shutdown {
  OgaShutdown();
}
@end
