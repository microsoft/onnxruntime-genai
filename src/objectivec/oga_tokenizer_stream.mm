// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"
#import "oga_internal.h"
#import "ort_genai_objc.h"

@implementation OGATokenizerStream {
  std::unique_ptr<OgaTokenizerStream> _stream;
}

- (nullable instancetype)initWithTokenizer:(OGATokenizer*)tokenizer error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    _stream = OgaTokenizerStream::Create([tokenizer CXXAPIOgaTokenizer]);
    return self;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable instancetype)initWithMultiModalProcessor:(OGAMultiModalProcessor*)processor
                                               error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    _stream = OgaTokenizerStream::Create([processor CXXAPIOgaMultiModalProcessor]);
    return self;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable NSString*)decode:(int32_t)token error:(NSError**)error {
  try {
    return [NSString stringWithUTF8String:_stream->Decode(token)];
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

@end