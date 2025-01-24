// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"
#import "oga_internal.h"
#import "ort_genai_objc.h"

@implementation OGAImages {
  std::unique_ptr<OgaImages> _images;
}

- (nullable instancetype)initWithPath:(NSArray<NSString*>*)paths error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    std::vector<const char*> cpp_paths;
    cpp_paths.reserve([paths count]);

    for (NSString* path in paths) {
      cpp_paths.push_back([path UTF8String]);
    }

    _images = OgaImages::Load(cpp_paths);
    return self;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (OgaImages&)CXXAPIOgaImages {
  return *(_images.get());
}

@end