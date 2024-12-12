// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"
#import "oga_internal.h"
#import "ort_genai_objc.h"

@implementation OGATokenizer {
  std::unique_ptr<OgaTokenizer> _tokenizer;
}

- (nullable instancetype)initWithModel:(OGAModel*)model error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    _tokenizer = OgaTokenizer::Create([model CXXAPIOgaModel]);
    return self;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable OGASequences*)encode:(NSString*)str error:(NSError**)error {
  OGASequences* sequences = [[OGASequences alloc] initWithError:error];
  if (!sequences) {
    return nil;
  }
  try {
    _tokenizer->Encode([str UTF8String], [sequences CXXAPIOgaSequences]);
    return sequences;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable NSString*)decode:(const int32_t*)tokensData
                      length:(size_t)tokensLength
                       error:(NSError**)error {
  try {
    OgaString result = _tokenizer->Decode(tokensData, tokensLength);
    return [NSString stringWithUTF8String:result];
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (const OgaTokenizer&)CXXAPIOgaTokenizer {
  return *(_tokenizer.get());
}

@end
