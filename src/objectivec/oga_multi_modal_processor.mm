// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"
#import "oga_internal.h"
#import "ort_genai_objc.h"

@implementation OGAMultiModalProcessor {
  std::unique_ptr<OgaMultiModalProcessor> _processor;
}

- (nullable instancetype)initWithModel:(OGAModel*)model error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    _processor = OgaMultiModalProcessor::Create([model CXXAPIOgaModel]);
    return self;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable OGANamedTensors*)processImages:(NSString*)prompt
                                    images:(OGAImages*)images
                                     error:(NSError**)error {
  try {
    OGANamedTensors* result = [[OGANamedTensors alloc]
        initWithCXXPointer:_processor->ProcessImages([prompt UTF8String],
                                                     &[images CXXAPIOgaImages])];
    return result;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable NSString*)decode:(const int32_t*)tokensData
                      length:(size_t)tokensLength
                       error:(NSError**)error {
  try {
    OgaString result = _processor->Decode(tokensData, tokensLength);
    return [NSString stringWithUTF8String:result];
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (const OgaMultiModalProcessor&)CXXAPIOgaMultiModalProcessor {
  return *(_processor.get());
}

@end
