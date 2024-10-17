// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "cxx_api.h"
#import "error_utils.h"
#import "ort_genai_objc.h"

NS_ASSUME_NONNULL_BEGIN

@interface OGAModel ()

- (const OgaModel&)CXXAPIOgaModel;

@end

@interface OGATokenizer ()

- (const OgaTokenizer&)CXXAPIOgaTokenizer;

@end

@interface OGAInt32Span ()

- (const int32_t*)pointer;
- (size_t)size;

@end

@interface OGAInt64Span ()

- (const int64_t*)pointer;
- (size_t)size;

@end

@interface OGASequences ()

- (nullable instancetype)initWithError:(NSError**)error;
- (instancetype)initWithNativePointer:(std::unique_ptr<OgaSequences>)ptr;

- (OgaSequences&)CXXAPIOgaSequences;

@end

@interface OGAGeneratorParams ()

- (OgaGeneratorParams&)CXXAPIOgaGeneratorParams;

@end

@interface OGAImages ()

- (OgaImages*)CXXAPIOgaImages;

@end

@interface OGATensor ()

- (OgaTensor&)CXXAPIOgaTensor;

@end

@interface OGANamedTensors ()

- (instancetype)initWithNativePointer:(std::unique_ptr<OgaNamedTensors>)ptr
    NS_DESIGNATED_INITIALIZER;
- (OgaNamedTensors&)CXXAPIOgaNamedTensors;

@end

@interface OGAMultiModalProcessor ()

- (const OgaMultiModalProcessor&)CXXAPIOgaMultiModalProcessor;

@end

NS_ASSUME_NONNULL_END
