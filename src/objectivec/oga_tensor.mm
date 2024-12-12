// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"
#import "oga_internal.h"
#import "ort_genai_objc.h"

@implementation OGATensor {
  std::unique_ptr<OgaTensor> _tensor;
}

- (instancetype)initWithCXXPointer:(std::unique_ptr<OgaTensor>)ptr {
  if ((self = [super init]) == nil) {
    return nil;
  }

  _tensor = std::move(ptr);
  return self;
}

- (nullable instancetype)initWithDataPointer:(void*)data
                                       shape:(NSArray<NSNumber*>*)shape
                                        type:(OGAElementType)elementType
                                       error:(NSError**)error {
  try {
    std::vector<int64_t> cxxShape;
    for (NSNumber* object in shape) {
      cxxShape.push_back([object longLongValue]);
    }
    self = [self initWithCXXPointer:OgaTensor::Create(data, cxxShape.data(), cxxShape.size(),
                                                      static_cast<OgaElementType>(elementType))];
    return self;
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (OGAElementType)getTypeWithError:(NSError**)error {
  try {
    return OGAElementType(_tensor->Type());
  }
  OGA_OBJC_API_IMPL_CATCH(error, OGAElementTypeUndefined)
}

- (nullable void*)getDataPointerWithError:(NSError**)error {
  try {
    return _tensor->Data();
  }
  OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (OgaTensor&)CXXAPIOgaTensor {
  return *(_tensor.get());
}

@end
