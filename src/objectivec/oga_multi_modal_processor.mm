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

- (nullable OGANamedTensors*)processImages:(NSArray<NSString*>*)prompts
                                    images:(OGAImages*)images
                                     error:(NSError**)error {
  try {
    std::vector<std::string> prompts_strings;
    prompts_strings.reserve([prompts count]);
    std::vector<const char*> prompts_;
    prompts_.reserve([prompts count]);

    for (NSString* prompt in prompts) {
      std::string prompt_str = [prompt UTF8String];
      prompts_strings.push_back(std::move(prompt_str));
      prompts_.push_back(prompt_strings.back().c_str());
    }

    OGANamedTensors* result = [[OGANamedTensors alloc]
        initWithCXXPointer:_processor->ProcessImages(prompts_,
                                                     &[images CXXAPIOgaImages])];
    return result;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable OGANamedTensors*)processAudios:(NSString*)prompt
                                    audios:(OGAAudios*)audios
                                     error:(NSError**)error {
  try {
    OGANamedTensors* result = [[OGANamedTensors alloc]
        initWithCXXPointer:_processor->ProcessAudios([prompt UTF8String],
                                                     &[audios CXXAPIOgaAudios])];
    return result;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable OGANamedTensors*)processAudios:(NSArray<NSString*>*)prompts
                                    audios:(OGAAudios*)audios
                                     error:(NSError**)error {
  try {
    std::vector<std::string> prompts_strings;
    prompts_strings.reserve([prompts count]);
    std::vector<const char*> prompts_;
    prompts_.reserve([prompts count]);

    for (NSString* prompt in prompts) {
      std::string prompt_str = [prompt UTF8String];
      prompts_strings.push_back(std::move(prompt_str));
      prompts_.push_back(prompt_strings.back().c_str());
    }

    OGANamedTensors* result = [[OGANamedTensors alloc]
        initWithCXXPointer:_processor->ProcessAudios(prompts_,
                                                     &[audios CXXAPIOgaAudios])];
    return result;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable OGANamedTensors*)processImagesAndAudios:(NSString*)prompt
                                             images:(OGAImages*)images
                                             audios:(OGAAudios*)audios
                                              error:(NSError**)error {
  try {
    OGANamedTensors* result = [[OGANamedTensors alloc]
        initWithCXXPointer:_processor->ProcessImagesAndAudios([prompt UTF8String],
                                                              &[images CXXAPIOgaImages],
                                                              &[audios CXXAPIOgaAudios])];
    return result;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable OGANamedTensors*)processImagesAndAudios:(NSArray<NSString*>*)prompts
                                             images:(OGAImages*)images
                                             audios:(OGAAudios*)audios
                                              error:(NSError**)error {
  try {
    std::vector<std::string> prompts_strings;
    prompts_strings.reserve([prompts count]);
    std::vector<const char*> prompts_;
    prompts_.reserve([prompts count]);

    for (NSString* prompt in prompts) {
      std::string prompt_str = [prompt UTF8String];
      prompts_strings.push_back(std::move(prompt_str));
      prompts_.push_back(prompt_strings.back().c_str());
    }

    OGANamedTensors* result = [[OGANamedTensors alloc]
        initWithCXXPointer:_processor->ProcessImagesAndAudios(prompts_,
                                                              &[images CXXAPIOgaImages],
                                                              &[audios CXXAPIOgaAudios])];
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
