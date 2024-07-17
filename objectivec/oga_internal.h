// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#import "cxx_api.h"


NS_ASSUME_NONNULL_BEGIN

@interface OGAModel ()

- (const OgaModel&)CXXAPIOgaModel;

@end

@interface OGATokenizer ()

- (const OgaTokenizer&)CXXAPIOgaTokenizer;

@end

@interface OGASpan ()

- (nullable)initWithRawPointer:(const int32_t *) pointer
                          size:(size_t)size;

- (const int32_t *)pointer;
- (size_t)size;

@end

@implementation OGASpan {
    const int32_t * _ptr;
    size_t _size;
}

- (nullable)initWithRawPointer:(const int32_t * )pointer
                          size:(size_t)size {
    _ptr = pointer;
    _size = size;
    return [self init];
}

- (const int32_t * )pointer {
    return _ptr;
}

- (size_t)size {
    return _size;
}

- (int32_t)last {
    return *(_ptr + (_size - 1));
}

@end


@interface OGASequences ()

- (nullable)initWithError:(NSError **)error;
- (instancetype)initWithNativePointer:(std::unique_ptr<OgaSequences>)ptr;

- (OgaSequences&)CXXAPIOgaSequences;

@end

@interface OGAGeneratorParams ()

- (OgaGeneratorParams&)CXXAPIOgaGeneratorParams;

@end

@interface OGAImages ()

- (OgaImages *)CXXAPIOgaImages;

@end

@interface OGATensor ()

- (OGATensor&)CXXAPIOgaTensor;

@end

@interface OGANamedTensors ()

- (OgaNamedTensor&)CXXAPIOgaNamedTensors;

@end

@interface OGAMultiModalProcessor ()

- (const OgaMultiModalProcessor&)CXXAPIOgaMultiModalProcessor;

@end

NS_ASSUME_NONNULL_END
