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

@interface OGASequences ()

- (nullable instancetype)initWithError:(NSError**)error;
- (instancetype)initWithCXXPointer:(std::unique_ptr<OgaSequences>)ptr NS_DESIGNATED_INITIALIZER;

- (OgaSequences&)CXXAPIOgaSequences;

@end

@interface OGAGeneratorParams ()

- (OgaGeneratorParams&)CXXAPIOgaGeneratorParams;

@end

@interface OGAImages ()

- (OgaImages&)CXXAPIOgaImages;

@end

@interface OGAAudios ()

- (OgaAudios&)CXXAPIOgaAudios;

@end

@interface OGATensor ()

- (instancetype)initWithCXXPointer:(std::unique_ptr<OgaTensor>)ptr NS_DESIGNATED_INITIALIZER;

- (OgaTensor&)CXXAPIOgaTensor;

@end

@interface OGANamedTensors ()

- (instancetype)initWithCXXPointer:(std::unique_ptr<OgaNamedTensors>)ptr NS_DESIGNATED_INITIALIZER;
- (OgaNamedTensors&)CXXAPIOgaNamedTensors;

@end

@interface OGAMultiModalProcessor ()

- (const OgaMultiModalProcessor&)CXXAPIOgaMultiModalProcessor;

@end

NS_ASSUME_NONNULL_END
