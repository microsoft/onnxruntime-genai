// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"
#import "oga_internal.h"
#import "ort_genai_objc.h"

@implementation OGAModel {
  std::unique_ptr<OgaModel> _model;
}

- (nullable instancetype)initWithPath:(NSString*)path error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    _model = OgaModel::Create([path UTF8String]);
    return self;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (const OgaModel&)CXXAPIOgaModel {
  return *(_model.get());
}

@end
