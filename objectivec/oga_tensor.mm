// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_genai_objc.h"
#import "error_utils.h"
#import "oga_internal.h"

@implementation OGATensor {
    std::unique_ptr<OgaTensor> _tensor;
}


- (nullable)initWithDataPointer:(void *)data
                          shape:(OGASpan)shape
                           type:(OGAElementType)elementType
                          error:(NSError **)error {
    if ((self = [super init]) == nil) {
        return nil;
    }

    try {
        _model = OgaTensor::Create(data, shape.pointer, shape.size, elementType);
        return self;
    }
    OGA_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (OGAElementType)type {
    return _tensor->Type();
}

- (void *)data {
    return _tensor->Data();
}

- (OgaModel&)CXXAPIOgaTensor {
    return *(_tensor.get());
}

@end
